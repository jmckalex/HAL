"""Privacy: words not addressed to Claude never reach the log or any file; audio never touches disk.

Runs the real listening loop (segmenter, decider, logging) on in-memory audio
with stand-ins for the VAD, the speech-to-text model, the gate and the proxy.
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal import spoken  # noqa: E402
from hal.listener import logs, settings  # noqa: E402
from hal.listener.audio import ArraySource  # noqa: E402
from hal.listener.decider import Decider  # noqa: E402
from hal.listener.service import Listener  # noqa: E402
from listen_harness import make_tmp, rmtree  # noqa: E402


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance_to(self, t):
        self.t = max(self.t, t)


class FakeVAD:
    """Loud frames are speech."""

    def __call__(self, frame):
        return 0.95 if float(np.max(np.abs(frame))) > 0.1 else 0.01

    def reset(self):
        pass


class FakeASR:
    name = "fake"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


class FakeGate:
    def __init__(self, blocked):
        self._blocked = blocked
        self.reply_end = -1e9

    def blocked(self, t):
        return self._blocked(t)


class FakeInjector:
    """Records what would be typed: .raw as typed (with the label and marks, hal/spoken.py),
    .typed the words alone."""
    holding = False

    def __init__(self):
        self.typed = []
        self.raw = []
        self.last_typed = None

    def submit(self, text, speech_end):
        self.raw.append(text)
        self.typed.append(spoken.plain(text))
        self.last_typed = time.time()

    def tick(self):
        pass


class FakeVoice:
    def __init__(self):
        self.said = []

    def __call__(self, text):
        self.said.append(text)

    def busy(self):
        return False


class Rig:
    """A Listener over in-memory audio (1000.0 on its clock is the start of the
    audio), with stand-in parts and its state in a temporary directory."""

    def __init__(self, audio, texts, blocked=lambda t: False, show_ignored=False, *, make_judge=None, asr=None,
                 session_state=None, cue_sound=None, linger=0.0, **cfg_over):
        self.tmp = make_tmp()
        self.console = []
        self._env = mock.patch.dict(os.environ, {"HAL_STATE_DIR": str(self.tmp / "state"),
                                                 "HAL_CONFIG": str(self.tmp / "c.json")})
        self._env.start()
        cfg = dict(settings.DEFAULTS, **cfg_over)
        self.clock = clock = FakeClock(1000.0)
        self.lst = Listener(cfg, source=ArraySource(audio, realtime=False, clock=clock), vad=FakeVAD(),
                            asr=asr or FakeASR(texts), gate=FakeGate(blocked), injector=FakeInjector(),
                            decider=Decider(cfg), voice=FakeVoice(), console=self.console.append, clock=clock,
                            show_ignored=show_ignored, judge=make_judge(clock) if make_judge else None,
                            session_state=session_state, cue_sound=cue_sound, linger=linger)

    def close(self):
        self._env.stop()
        rmtree(self.tmp)


def utterances(*seconds, gap=1.5, level=0.5):
    parts = [np.zeros(int(gap * 16000), np.float32)]
    for s in seconds:
        parts += [np.full(int(s * 16000), level, np.float32), np.zeros(int(gap * 16000), np.float32)]
    return np.concatenate(parts)


SECRET_WORDS = ["4321", "swordfish", "grandmother", "Sunday", "PIN"]
TEXTS = ["Claude, open my thesis.",
         "My bank PIN is 4321 and the password is swordfish.",
         "We should visit grandmother on Sunday.",
         "Thanks, Claude."]


class PrivacyTest(unittest.TestCase):
    def run_listener(self, show_ignored=False):
        rig = Rig(utterances(1.2, 2.0, 1.6, 0.8), TEXTS, show_ignored=show_ignored)
        self.addCleanup(rig.close)
        rig.lst.run()
        return rig.lst, rig.tmp, rig.console

    def files_text(self, tmp: Path) -> str:
        return "\n".join(p.read_text(errors="replace") for p in tmp.rglob("*") if p.is_file())

    def test_unaddressed_words_never_written(self):
        lst, tmp, _ = self.run_listener()
        self.assertEqual(lst.injector.typed, ["Claude, open my thesis.", "Thanks, Claude."])
        log = (tmp / "state" / "listen.log").read_text()
        self.assertIn("Claude, open my thesis.", log)          # addressed utterances may be logged
        self.assertIn("Thanks, Claude.", log)
        self.assertEqual(log.count("ignored an utterance not addressed to Claude"), 2)
        everything = self.files_text(tmp)
        for word in SECRET_WORDS:
            self.assertNotIn(word, everything)
        # ... nor kept in memory for reports.
        self.assertNotIn("swordfish", json.dumps(lst.events))

    def test_ignored_log_lines_give_durations(self):
        _, tmp, _ = self.run_listener()
        lines = [l for l in (tmp / "state" / "listen.log").read_text().splitlines() if "ignored" in l]
        self.assertIn("2.0s of speech", lines[0])
        self.assertIn("1.6s of speech", lines[1])

    def test_console_shows_ignored_words_only_when_asked(self):
        _, tmp, console = self.run_listener(show_ignored=False)
        self.assertFalse(any("swordfish" in line for line in console))
        _, tmp, console = self.run_listener(show_ignored=True)
        self.assertTrue(any("swordfish" in line for line in console))
        self.assertNotIn("swordfish", self.files_text(tmp))   # still never in a file

    def test_no_audio_written(self):
        _, tmp, _ = self.run_listener()
        names = {p.name for p in tmp.rglob("*") if p.is_file()}
        self.assertLessEqual(names, {"listen.log", "listener.json"})

    def test_only_addressed_turns_reach_the_judge(self):
        """The front desk's judge (turn_taking "semantic") sees fragments of addressed turns, nothing else."""

        class Done:
            done, kind, why, seconds = True, "complete", "stand-in", 0.1

        class RecordingJudge:
            def __init__(self):
                self.asked = []

            def __call__(self, text):
                self.asked.append(text)
                return Done()

        judge = RecordingJudge()
        texts = ["My bank PIN is 4321 and the password is swordfish.",       # before any turn
                 "Claude, I think the conclusion of my paper is weak.",      # addressed: may be judged
                 "We should visit grandmother on Sunday."]                   # after the turn was sent
        # Claude is busy with a turn typed at the keyboard; it ends once the spoken request has been answered.
        session = {"turn": "busy", "last_stop": 0.0}
        rig = Rig(utterances(2.5, 2.0, 2.0, gap=4.0), texts, make_judge=lambda clock: judge,
                  session_state=lambda: session)
        self.addCleanup(rig.close)
        typed = rig.lst.injector.submit

        def submit(text, speech_end):
            typed(text, speech_end)
            session.update(turn="idle", last_stop=time.time() + 1)
        rig.lst.injector.submit = submit
        rig.lst.run()
        self.assertEqual(judge.asked, ["Claude, I think the conclusion of my paper is weak."])
        self.assertEqual(rig.lst.injector.typed, ["Claude, I think the conclusion of my paper is weak."])
        everything = self.files_text(rig.tmp) + json.dumps(rig.lst.events)
        for word in SECRET_WORDS:
            self.assertNotIn(word, everything + " ".join(judge.asked))

    def test_logs_api_cannot_take_text(self):
        import inspect

        self.assertEqual(list(inspect.signature(logs.ignored).parameters), ["speech_seconds", "total_seconds"])
        self.assertEqual(list(inspect.signature(logs.discarded).parameters), ["reason", "seconds"])


if __name__ == "__main__":
    unittest.main()
