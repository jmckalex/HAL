"""The voice check (hal/listener/voiceprint.py): features, the voiceprint file,
the check, the log report, and (when the model and macOS voices are present)
the real model telling two synthetic voices apart."""

import os
import shutil
import stat
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal.listener import settings, voiceprint  # noqa: E402
from hal.listener.voiceprint import Verdict, VoiceCheck, Voiceprint, fbank  # noqa: E402
from listen_harness import make_tmp, rmtree  # noqa: E402

CFG = settings.DEFAULTS
RATE = 16000


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class FeaturesTest(unittest.TestCase):
    def test_frames_and_mean_normalisation(self):
        rng = np.random.default_rng(1)
        feats = fbank(rng.normal(0, 0.1, RATE).astype(np.float32))
        self.assertEqual(feats.shape, (1 + (RATE - 400) // 160, 80))    # 25 ms frames every 10 ms, inside the signal
        self.assertTrue(np.allclose(feats.mean(axis=0), 0, atol=1e-4))
        self.assertEqual(fbank(np.zeros(399, np.float32)).shape, (0, 80))

    def test_mel_filters(self):
        banks = voiceprint._mel_banks()
        self.assertEqual(banks.shape, (80, 257))
        self.assertTrue((banks[:, -1] == 0).all())                       # Kaldi gives the Nyquist bin no weight
        peaks = banks.argmax(axis=1)
        self.assertTrue((np.diff(peaks) >= 0).all())
        self.assertTrue((banks.max(axis=1) > 0).all())                   # no empty filter (the lowest are narrow)

    def test_a_tone_lands_in_its_band(self):
        t = np.arange(RATE) / RATE
        audio = np.concatenate([0.3 * np.sin(2 * np.pi * 1000 * t), 1e-4 * np.ones(RATE)]).astype(np.float32)
        feats = fbank(audio)
        rise = feats[: 90].mean(axis=0) - feats[-90:].mean(axis=0)       # tone frames over quiet frames
        mel = voiceprint._mel
        delta = (mel(8000) - mel(20)) / 81
        expected = int(np.argmin(np.abs(mel(20) + (np.arange(80) + 1) * delta - mel(1000))))
        self.assertIn(int(np.argmax(rise)), (expected - 1, expected, expected + 1))


class VoiceprintFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp("hal-vp-")
        self.addCleanup(rmtree, self.tmp)
        self.path = self.tmp / "voiceprint.json"

    def test_round_trip_private_file(self):
        vp = voiceprint.new_voiceprint("m.onnx", [unit(1, 0, 0), unit(1, 0.1, 0), unit(1, 0, 0.1)])
        voiceprint.save(vp, self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        back = voiceprint.load(self.path)
        self.assertEqual(back.model, "m.onnx")
        self.assertTrue(np.allclose(back.embeddings, vp.embeddings, atol=1e-5))
        self.assertAlmostEqual(float(np.linalg.norm(back.vector)), 1.0, places=5)
        self.assertEqual(len(back.self_scores()), 3)
        self.assertNotIn("audio", self.path.read_text())

    def test_adding_and_replacing(self):
        old = voiceprint.new_voiceprint("m.onnx", [unit(1, 0, 0)])
        self.assertEqual(len(voiceprint.new_voiceprint("m.onnx", [unit(0, 1, 0)], old).embeddings), 2)
        self.assertEqual(len(voiceprint.new_voiceprint("other.onnx", [unit(0, 1, 0)], old).embeddings), 1)

    def test_unreadable_is_none(self):
        self.assertIsNone(voiceprint.load(self.path))
        self.path.write_text("{not json")
        self.assertIsNone(voiceprint.load(self.path))
        self.path.write_text('{"embeddings": []}')
        self.assertIsNone(voiceprint.load(self.path))

    def test_forget(self):
        voiceprint.save(voiceprint.new_voiceprint("m", [unit(1, 0)]), self.path)
        self.assertTrue(voiceprint.forget(self.path))
        self.assertFalse(voiceprint.forget(self.path))


class VoiceCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp("hal-vc-")
        self.addCleanup(rmtree, self.tmp)
        self.path = self.tmp / "voiceprint.json"
        self.next = unit(1, 0, 0)
        self.check = VoiceCheck(dict(CFG, voice_threshold=0.5, voice_min_seconds=1.0),
                                embedder=lambda audio: self.next, path=self.path)

    def test_nothing_without_a_voiceprint(self):
        self.assertIsNone(self.check(np.zeros(RATE, np.float32), 2.0))

    def test_verdicts(self):
        voiceprint.save(voiceprint.new_voiceprint("m", [unit(1, 0, 0)]), self.path)
        v = self.check(np.zeros(RATE, np.float32), 2.0)
        self.assertEqual((round(v.score, 3), v.match, v.reliable, v.describe()), (1.0, True, True, "you"))
        self.next = unit(0, 1, 0)
        v = self.check(np.zeros(RATE, np.float32), 2.0)
        self.assertEqual((v.match, v.reliable, v.describe()), (False, True, "another voice"))
        v = self.check(np.zeros(RATE, np.float32), 0.6)
        self.assertEqual((v.match, v.reliable), (False, False))
        self.next = None                                       # too short for an embedding
        self.assertIsNone(self.check(np.zeros(100, np.float32), 0.0))

    def test_a_new_voiceprint_is_picked_up(self):
        voiceprint.save(voiceprint.new_voiceprint("m", [unit(0, 1, 0)]), self.path)
        self.assertFalse(self.check(np.zeros(RATE, np.float32), 2.0).match)
        voiceprint.save(voiceprint.new_voiceprint("m", [unit(1, 0, 0)]), self.path)
        os.utime(self.path, (1, 1))                            # a different mtime whatever the clock's grain
        self.assertTrue(self.check(np.zeros(RATE, np.float32), 2.0).match)

    def test_a_missing_model_means_no_check(self):
        logged = []
        check = VoiceCheck(dict(CFG, voice_model="models/speaker/none.onnx"), path=self.path, log=logged.append)
        voiceprint.save(voiceprint.new_voiceprint("m", [unit(1, 0, 0)]), self.path)
        self.assertIsNone(check(np.zeros(RATE, np.float32), 2.0))
        self.assertIsNone(check(np.zeros(RATE, np.float32), 2.0))
        self.assertEqual(len(logged), 1)                       # said once, not at every utterance

    def test_speech_only(self):
        audio = np.arange(5 * RATE, dtype=np.float32)
        cut = voiceprint.speech_only(audio, 100.0, 101.0, 103.0)
        self.assertEqual((len(cut), cut[0]), (2 * RATE, RATE))
        self.assertIs(voiceprint.speech_only(audio, 100.0, 101.0, 101.2), audio)   # too little: keep it all


class ReportTest(unittest.TestCase):
    def test_scores_from_the_log(self):
        lines = [
            "2026-11-26 11:00:01 [1 listen] heard (2.1s of speech, fragment 1 of a turn, named, voice 0.74 (you)): 'x'",
            "2026-11-26 11:00:05 [1 listen] voice 0.21 (another voice; not addressed; 1.4s of speech)",
            "2026-11-26 11:00:09 [1 listen] voice 0.08 (another voice: ignored unheard; 3.0s of speech)",
            "2026-11-26 11:00:12 [1 listen] ignored an utterance not addressed to Claude (1.4s of speech, 2.0s long)",
        ]
        self.assertEqual(voiceprint.logged_scores(lines),
                         {"addressed": [0.74], "unaddressed": [0.21], "refused": [0.08]})


class EnrolTest(unittest.TestCase):
    def test_live_enrolment_takes_one_embedding_per_sentence(self):
        import argparse
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        from hal.listener import audio, cli, vad
        from hal.listener.audio import ArraySource
        from test_listen_privacy import FakeClock, FakeVAD, utterances

        clock = FakeClock(1000.0)
        src = ArraySource(utterances(2.0, 0.5, 2.5, gap=1.0), realtime=False, clock=clock)
        heard = []
        with mock.patch.object(audio, "MicSource", lambda device=None: src), \
                mock.patch.object(vad, "SileroVAD", lambda path: FakeVAD()), \
                mock.patch.object(cli, "_running_pid", lambda: 0), redirect_stdout(io.StringIO()) as out:
            got = cli._enrol_live(argparse.Namespace(count=2, device=None), CFG,
                                  lambda a: heard.append(len(a)) or unit(1, 0))
        self.assertEqual(len(got), 2)
        self.assertIn("too short", out.getvalue())             # the 0.5 s one was asked for again
        self.assertEqual([round(n / RATE, 1) for n in heard], [2.0, 2.5])   # speech only, without the margins


MODEL = voiceprint.model_path(CFG)


@unittest.skipUnless(MODEL.exists() and shutil.which("say") and shutil.which("ffmpeg"),
                     "needs models/speaker (scripts/fetch-voice-model.sh), macOS `say` and ffmpeg")
class RealModelTest(unittest.TestCase):
    """Two synthetic voices: the same voice scores far above a different one."""

    SENTENCES = ["The birch canoe slid on the smooth planks.", "Glue the sheet to the dark blue background.",
                 "It's easy to tell the depth of a well.", "Claude, please run the tests and tell me what failed."]

    def speak(self, voice, text, name):
        path = self.tmp / f"{name}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(path), text], check=True, capture_output=True)
        from hal.listener.audio import decode_file

        return decode_file(str(path))

    def setUp(self):
        self.tmp = make_tmp("hal-vr-")
        self.addCleanup(rmtree, self.tmp)

    def test_same_voice_far_above_another(self):
        try:
            own = [self.speak("Samantha", s, f"s{i}") for i, s in enumerate(self.SENTENCES)]
            other = self.speak("Daniel", self.SENTENCES[3], "d3")
        except subprocess.CalledProcessError:
            self.skipTest("the Samantha and Daniel voices aren't installed")
        embed = voiceprint.Embedder(MODEL)
        vp = voiceprint.new_voiceprint(embed.name, [embed(a) for a in own[:3]])
        same, diff = vp.score(embed(own[3])), vp.score(embed(other))
        self.assertGreater(same, 0.7)
        self.assertLess(diff, 0.5)
        self.assertGreater(same - diff, 0.3)


if __name__ == "__main__":
    unittest.main()
