"""Audio sources: the microphone, or a file/array fed through the same pipeline.

Each source yields (t_end, frame): a frame of 512 float32 samples at 16 kHz
and the time (time.monotonic, or the injected clock) at the end of that frame.
Audio stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import queue
import subprocess
import time

import numpy as np

RATE = 16000
FRAME = 512
FRAME_SECONDS = FRAME / RATE


def decode_file(path: str) -> np.ndarray:
    """Any audio file -> 16 kHz mono float32, via ffmpeg (for tests and `run --file`)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "s16le",
                          "-ac", "1", "-ar", str(RATE), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def find_device(spec):
    """None, an index, or a substring of an input device's name -> sounddevice device."""
    if spec is None or isinstance(spec, int):
        return spec
    import sounddevice as sd

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and str(spec).lower() in dev["name"].lower():
            return i
    raise ValueError(f"no input device matching {spec!r}")


class MicSource:
    """The default (or configured) input device, opened at 16 kHz mono.

    CoreAudio converts from the device's native rate. The callback only copies
    frames into a bounded queue; if the consumer falls behind by more than
    max_seconds the oldest audio is dropped (and counted)."""

    def __init__(self, device=None, max_seconds: float = 30.0, clock=time.monotonic):
        self.device = device
        self.clock = clock
        self.q: queue.Queue = queue.Queue(maxsize=int(max_seconds / FRAME_SECONDS))
        self.stream = None
        self.dropped = 0
        self.status_flags = 0
        self.finished = False
        self.name = "?"

    @property
    def running(self) -> bool:
        return self.stream is not None

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            self.status_flags += 1
        item = (self.clock(), indata[:, 0].copy())
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put_nowait(item)

    def start(self) -> None:
        import sounddevice as sd

        device = find_device(self.device)
        info = sd.query_devices(device, kind="input")
        self.name = info["name"]
        self.stream = sd.InputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32",
                                     device=device, latency="low", callback=self._callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # a device that vanished can't be stopped cleanly
                pass
            finally:
                self.stream = None
        while True:  # forget whatever was captured
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float = 0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class ArraySource:
    """Samples already in memory, delivered frame by frame.

    realtime=True paces delivery like a microphone (the end-to-end test uses
    this, so the voice gate and timers see real time). realtime=False delivers
    as fast as it is read; with a FakeClock it advances that clock instead."""

    keeps_time = True   # while the listener is paused, frames go on arriving (and are dropped)

    def __init__(self, audio: np.ndarray, *, realtime: bool = True, clock=time.monotonic, lead_in: float = 0.0):
        audio = np.asarray(audio, dtype=np.float32)
        pad = (-len(audio)) % FRAME
        self.lead = int(lead_in * RATE)
        self.audio = np.concatenate([np.zeros(self.lead, np.float32), audio, np.zeros(pad, np.float32)])
        self.realtime = realtime
        self.clock = clock
        self.i = 0
        self.t0 = None
        self.finished = False
        self.name = "file"

    @property
    def running(self) -> bool:
        return self.t0 is not None

    def start(self) -> None:
        if self.t0 is None:
            self.t0 = self.clock()

    def stop(self) -> None:
        pass

    def time_of(self, sample: int) -> float:
        """Clock time at which sample index (of the original audio) was "captured"."""
        return self.t0 + (self.lead + sample) / RATE

    def read(self, timeout: float = 0.1):
        if self.i + FRAME > len(self.audio):
            self.finished = True
            if self.realtime:
                time.sleep(min(timeout, 0.05))
            elif hasattr(self.clock, "advance_to"):  # simulated time goes on after the end
                self.clock.advance_to(self.clock() + timeout)
            return None
        t_end = self.t0 + (self.i + FRAME) / RATE
        if self.realtime:
            wait = t_end - self.clock()
            if wait > timeout:
                time.sleep(timeout)
                return None
            if wait > 0:
                time.sleep(wait)
        elif hasattr(self.clock, "advance_to"):
            self.clock.advance_to(t_end)
        frame = self.audio[self.i:self.i + FRAME]
        self.i += FRAME
        return t_end, frame
