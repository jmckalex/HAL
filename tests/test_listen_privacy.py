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
from hal.listener import voiceprint  # noqa: E402
from hal.listener.voiceprint import Verdict  # noqa: E402
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
                 session_state=None, cue_sound=None, linger=0.0, screen_locked=None, voices=None,
                 **cfg_over):
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
                            session_state=session_state, cue_sound=cue_sound, linger=linger,
                            screen_locked=screen_locked, voice_check=voices)

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

    def test_nothing_heard_while_the_screen_is_locked(self):
        # Speech at 1001.5-1003.0 and 1004.5-1006.0 on the rig's clock; the screen is locked 1000.2-1003.5.
        holder = {}
        texts = ["Claude, what's the weather tomorrow?"]  # the stand-in transcriber answers in order
        rig = Rig(utterances(1.5, 1.5), texts, screen_locked=lambda: 1000.2 <= holder["rig"].clock() < 1003.5)
        holder["rig"] = rig
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, ["Claude, what's the weather tomorrow?"])
        self.assertEqual(rig.lst.asr.calls, 1)          # the first utterance was never even transcribed
        log = (rig.tmp / "state" / "listen.log").read_text()
        self.assertIn("paused (screen locked)", log)
        self.assertIn("resumed (screen unlocked)", log)

    def voices(self, *verdicts):
        """A stand-in voice check answering with these verdicts, in order."""
        queue = list(verdicts)
        return lambda audio, speech_seconds: queue.pop(0) if queue else None

    def test_another_voice_is_ignored_unheard(self):
        texts = ["Claude, what's the weather tomorrow?"]
        rig = Rig(utterances(1.5, 1.5), texts, voices=self.voices(Verdict(0.12, False, True),
                                                                       Verdict(0.81, True, True)),
                  voice_check="enforce", greet_strangers=False)
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.asr.calls, 1)              # the other voice was never transcribed
        self.assertEqual(rig.lst.injector.typed, texts)
        self.assertEqual((rig.lst.counts.voice_checked, rig.lst.counts.other_voice), (2, 1))
        log = (rig.tmp / "state" / "listen.log").read_text()
        self.assertIn("voice 0.12 (another voice: ignored unheard; 1.5s of speech)", log)
        self.assertIn("voice 0.81 (you)", log)

    def test_log_mode_changes_nothing(self):
        texts = ["Claude, open my thesis.", "Claude, what's the weather tomorrow?"]
        rig = Rig(utterances(1.5, 1.5), texts, voices=self.voices(Verdict(0.1, False, True),
                                                                       Verdict(0.2, False, True)),
                  voice_check="log", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, texts)
        self.assertIn("voice 0.10 (another voice)", (rig.tmp / "state" / "listen.log").read_text())

    def test_short_and_unlike_you_needs_the_name(self):
        texts = ["Claude, could you look at the draft of my paper and", "the second chapter."]
        rig = Rig(utterances(2.0, 1.0), texts, voices=self.voices(Verdict(0.8, True, True),
                                                                       Verdict(0.2, False, False)),
                  voice_check="enforce")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, [texts[0]])   # the stranger's words didn't join the turn
        log = (rig.tmp / "state" / "listen.log").read_text()
        self.assertIn("voice 0.20 (unclear (too short to judge); not addressed", log)
        self.assertNotIn("second chapter", log)

    def test_no_verdict_no_change(self):
        texts = ["Claude, open my thesis."]
        rig = Rig(utterances(1.5), texts, voices=lambda audio, s: None, voice_check="enforce")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, texts)

    def test_only_your_addressed_requests_are_learned(self):
        class Voices:
            def __init__(self, *verdicts):
                self.queue, self.learned = list(verdicts), []

            def __call__(self, audio, speech_seconds):
                return self.queue.pop(0)

            def learn(self, verdict, speech_seconds):
                self.learned.append(round(verdict.score, 2))
                return True

        texts = ["Claude, open my thesis.", "Nice weather today.", "Claude, what's the weather tomorrow?"]
        voices = Voices(Verdict(0.8, True, True, embedding=np.ones(2)), Verdict(0.75, True, True, embedding=np.ones(2)),
                        Verdict(0.3, False, False, embedding=np.ones(2)))
        rig = Rig(utterances(1.5, 1.5, 0.8), texts, voices=voices, voice_check="enforce", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, [texts[0], texts[2]])   # the third: short, but it has the name
        self.assertEqual(voices.learned, [0.8])                          # not the unaddressed one, nor the unsure one
        self.assertIn("voice: learned from this request (0.80)", (rig.tmp / "state" / "listen.log").read_text())

    def test_a_guest_is_named_in_the_label(self):
        texts = ["Claude, what's the weather tomorrow?", "Claude, open my thesis."]
        rig = Rig(utterances(1.5, 1.5), texts, voices=self.voices(Verdict(0.8, True, True, who="Robin"),
                                                                   Verdict(0.8, True, True)),
                  voice_check="log", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, texts)
        self.assertRegex(rig.lst.injector.raw[0], r"^\[spoken \d\d:\d\d; by Robin\] Claude, what's")
        self.assertRegex(rig.lst.injector.raw[1], r"^\[spoken \d\d:\d\d\] Claude, open")

    def test_a_turn_says_everyone_who_spoke_in_it(self):
        texts = ["Claude, could you look at the draft of my paper and", "the second chapter."]
        rig = Rig(utterances(2.0, 1.5), texts, voices=self.voices(Verdict(0.8, True, True),
                                                                   Verdict(0.7, True, True, who="Robin")),
                  voice_check="enforce")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(len(rig.lst.injector.raw), 1)
        self.assertIn("; by the user and Robin] Claude, could you look", rig.lst.injector.raw[0])

    def test_guest_requests_can_be_ignored(self):
        texts = ["Claude, open my thesis."]
        rig = Rig(utterances(1.5, 1.5), texts, voices=self.voices(Verdict(0.8, True, True, who="Robin"),
                                                                   Verdict(0.8, True, True)),
                  voice_check="enforce", guest_requests="ignore", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, texts)
        self.assertEqual(rig.lst.asr.calls, 1)
        self.assertIn("Robin: ignored unheard (guest_requests)", (rig.tmp / "state" / "listen.log").read_text())

    def test_a_stranger_who_says_the_name_is_greeted_and_their_answer_typed(self):
        emb = np.ones(4, np.float32) / 2
        texts = ["Claude, could you help me with something?", "I'm Robin, a friend of the family.",
                 "Claude, are you there?", "Nice weather today."]
        stranger = lambda: Verdict(0.2, False, True, embedding=emb)  # noqa: E731
        rig = Rig(utterances(2.0, 2.0, 2.0, 2.0), texts,
                  voices=self.voices(stranger(), stranger(), stranger(), stranger()),
                  voice_check="enforce", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual(rig.lst.voice.said, [settings.DEFAULTS["stranger_greeting"]])   # once, not again
        self.assertEqual(len(rig.lst.injector.raw), 1)
        self.assertRegex(rig.lst.injector.raw[0],
                         r"^\[spoken \d\d:\d\d; by an unrecognised voice, answering the greeting\] I'm Robin")
        kept = voiceprint.load_stranger(900, base=rig.tmp / "state" / "listen")
        self.assertEqual(len(kept), 2)                                 # the request and the answer
        log = (rig.tmp / "state" / "listen.log").read_text()
        self.assertIn("an unrecognised voice said the name: greeted", log)
        self.assertNotIn("help me with something", log)                # the stranger's request isn't kept
        self.assertNotIn("Nice weather", log)

    def test_a_stranger_who_does_not_say_the_name_is_left_alone(self):
        texts = ["Shall I do the kitchen next?"]
        rig = Rig(utterances(2.0), texts, voices=self.voices(Verdict(0.2, False, True, embedding=np.ones(4))),
                  voice_check="enforce", turn_taking="silence")
        self.addCleanup(rig.close)
        rig.lst.run()
        self.assertEqual((rig.lst.voice.said, rig.lst.injector.raw), ([], []))
        self.assertIsNone(voiceprint.load_stranger(900, base=rig.tmp / "state" / "listen"))

    def test_logs_api_cannot_take_text(self):
        import inspect

        self.assertEqual(list(inspect.signature(logs.ignored).parameters), ["speech_seconds", "total_seconds"])
        self.assertEqual(list(inspect.signature(logs.discarded).parameters), ["reason", "seconds"])


if __name__ == "__main__":
    unittest.main()
