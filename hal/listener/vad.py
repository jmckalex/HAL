"""Silero VAD (v5/v6 ONNX model) through onnxruntime, one 32 ms frame at a time.

The model takes 512 new samples at 16 kHz preceded by the last 64 samples of
the previous frame, plus a recurrent state, and returns the probability that
the frame is speech. This mirrors silero_vad.utils_vad.OnnxWrapper without
needing torch.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512          # samples per frame (32 ms)
_CONTEXT = 64


class SileroVAD:
    def __init__(self, model_path: str):
        import onnxruntime

        try:  # ONNX Runtime documents its telemetry as Windows-only; make sure regardless.
            onnxruntime.disable_telemetry_events()
        except Exception:
            pass
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(str(model_path), sess_options=opts,
                                                     providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability of one frame of FRAME float32 samples in [-1, 1]."""
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FRAME:
            raise ValueError(f"expected {FRAME} samples, got {x.shape[1]}")
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out[0][0])
