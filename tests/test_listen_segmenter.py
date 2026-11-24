"""VAD segmentation: the state machine on scripted probabilities, then Silero on synthesized speech.

The speech is made with macOS `say` and converted with ffmpeg into memory;
tests that need them are skipped when they (or the Silero model) are missing.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hal.listener import settings  # noqa: E402
from hal.listener.segmenter import Discarded, Segmenter, Utterance  # noqa: E402

CFG = dict(settings.DEFAULTS)
DT = 512 / 16000
VAD_MODEL = settings.project_path(CFG["vad_model"])


def run_probs(probs, cfg=CFG):
    """Feed frames whose sample values encode their index; return the segmenter's outputs."""
    seg = Segmenter(cfg)
    out = []
    for i, p in enumerate(probs):
        frame = np.full(512, i, dtype=np.float32)
        r = seg.push(frame, p, (i + 1) * DT)
        if r is not None:
            out.append(r)
    return seg, out


def frames_of(utt: Utterance) -> list:
    return sorted(set(int(v) for v in utt.audio[::512]))


class StateMachineTest(unittest.TestCase):
    def test_silence_only(self):
        _, out = run_probs([0.01] * 200)
        self.assertEqual(out, [])

    def test_one_utterance_with_pre_and_post_roll(self):
        # 50 frames silence, 40 speech, 60 silence.
        _, out = run_probs([0.0] * 50 + [0.95] * 40 + [0.0] * 60)
        self.assertEqual(len(out), 1)
        u = out[0]
        pre = round(CFG["pre_roll"] / DT)
        post = round(CFG["post_roll"] / DT)
        self.assertEqual(frames_of(u)[0], 50 - pre + 1)        # pre-roll includes frames before onset
        self.assertEqual(frames_of(u)[-1], 89 + post)           # speech ends at frame 89, then post-roll
        self.assertAlmostEqual(u.speech_start, 50 * DT)
        self.assertAlmostEqual(u.speech_end, 90 * DT)
        self.assertAlmostEqual(u.speech_seconds, 40 * DT)
        self.assertFalse(u.cut or u.continued)

    def test_ends_after_end_silence(self):
        end_frames = round(CFG["end_silence"] / DT)
        seg, out = run_probs([0.9] * 20 + [0.0] * (end_frames - 1))
        self.assertEqual(out, [])
        self.assertTrue(seg.in_speech)
        r = seg.push(np.zeros(512, np.float32), 0.0, 999)
        self.assertIsInstance(r, Utterance)

    def test_short_pause_does_not_split(self):
        _, out = run_probs([0.9] * 20 + [0.0] * 15 + [0.9] * 20 + [0.0] * 60)   # 0.48 s pause
        self.assertEqual(len(out), 1)

    def test_long_pause_splits(self):
        _, out = run_probs([0.9] * 20 + [0.0] * 40 + [0.9] * 20 + [0.0] * 60)   # 1.28 s pause
        self.assertEqual(len(out), 2)

    def test_hysteresis(self):
        # Frames between neg_threshold and threshold don't start the end-silence count...
        _, out = run_probs([0.9] * 20 + [0.4] * 60 + [0.9] * 5 + [0.0] * 60)
        self.assertEqual(len(out), 1)
        # ...but once it has started they don't stop it.
        _, out = run_probs([0.9] * 20 + [0.1] + [0.4] * 30 + [0.9] * 20 + [0.0] * 60)
        self.assertEqual(len(out), 2)

    def test_short_burst_is_noise(self):
        _, out = run_probs([0.0] * 10 + [0.9] * 3 + [0.0] * 60)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], Discarded)
        self.assertEqual(out[0].reason, "short")

    def test_long_utterance_is_cut_into_continued_parts(self):
        cfg = dict(CFG, max_utterance=5.0)
        _, out = run_probs([0.9] * 400 + [0.0] * 60, cfg)   # 12.8 s of speech
        self.assertEqual([(u.cut, u.continued) for u in out],
                         [(True, False), (True, True), (False, True)])
        total = sum(len(u.audio) for u in out)
        self.assertGreaterEqual(total, 400 * 512)            # nothing lost at the cuts
        self.assertLessEqual(max(u.duration for u in out), 5.0 + 1e-6)

    def test_reset_discards_partial(self):
        seg, _ = run_probs([0.9] * 20)
        dropped = seg.reset()
        self.assertEqual(dropped.reason, "gated")
        self.assertFalse(seg.in_speech)
        self.assertIsNone(seg.reset())

    def test_flush(self):
        seg, _ = run_probs([0.9] * 20)
        self.assertIsInstance(seg.flush(1.0), Utterance)


def _have_tools():
    return shutil.which("say") and shutil.which("ffmpeg") and VAD_MODEL.exists()


def synth(text: str, voice: str = "Samantha") -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["say", "-v", voice, "-o", f"{d}/a.aiff", text], check=True, timeout=60)
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", f"{d}/a.aiff", "-f", "s16le",
                              "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), np.float32)


@unittest.skipUnless(_have_tools(), "needs macOS say, ffmpeg and models/silero_vad.onnx")
class SileroSegmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hal.listener.vad import SileroVAD

        cls.vad = SileroVAD(str(VAD_MODEL))
        cls.a = synth("Good morning, Claude.")
        cls.b = synth("Claude, what's the weather tomorrow?")
        cls.long = synth("Claude, " + "this is a long dictation that goes on and on without a real pause, " * 4
                         + "and now it ends.")

    def segment(self, audio, cfg=CFG):
        self.vad.reset()
        seg = Segmenter(cfg)
        out = []
        for i in range(0, len(audio) - 511, 512):
            r = seg.push(audio[i:i + 512], self.vad(audio[i:i + 512]), (i + 512) / 16000)
            if isinstance(r, Utterance):
                out.append(r)
        return out

    def onset(self, x):
        return np.where(np.abs(x) > 0.005)[0][0] / 16000

    def test_speech_then_silence(self):
        audio = np.concatenate([silence(1.0), self.a, silence(1.5)])
        out = self.segment(audio)
        self.assertEqual(len(out), 1)
        onset = 1.0 + self.onset(self.a)
        self.assertLessEqual(out[0].start, onset - 0.2)            # pre-roll: the onset isn't clipped
        self.assertAlmostEqual(out[0].speech_end, 1.0 + len(self.a) / 16000, delta=0.25)

    def test_silence_and_low_noise(self):
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16000 * 5) * 0.003).astype(np.float32)   # about -50 dBFS
        self.assertEqual(self.segment(audio), [])

    def test_two_utterances_close_together(self):
        merged = self.segment(np.concatenate([silence(1), self.a, silence(0.4), self.b, silence(1.5)]))
        self.assertEqual(len(merged), 1)
        split = self.segment(np.concatenate([silence(1), self.a, silence(1.3), self.b, silence(1.5)]))
        self.assertEqual(len(split), 2)
        self.assertLess(split[0].end, split[1].start + CFG["pre_roll"] + 0.01)

    def test_long_utterance(self):
        cfg = dict(CFG, max_utterance=6.0)
        out = self.segment(np.concatenate([silence(1), self.long, silence(1.5)]), cfg)
        self.assertGreater(len(self.long) / 16000, 12)
        self.assertGreaterEqual(len(out), 2)
        self.assertTrue(out[0].cut and not out[0].continued)
        self.assertTrue(all(u.continued for u in out[1:]))
        self.assertFalse(out[-1].cut)


if __name__ == "__main__":
    unittest.main()
