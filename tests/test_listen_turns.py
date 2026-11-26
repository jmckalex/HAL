"""Semantic turn-taking (hal/listener/turns.py): rules, the judge, timers, spoken controls, and the
listening loop holding fragments until a request is finished. Also during_turn.

Everything runs on a simulated clock with stand-ins for the VAD, the speech-to-text model, the
gate, the proxy and the front desk's judge, except DeskJudgeTest, which talks to a stand-in desk
on a real Unix socket.
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal.listener import settings, turns  # noqa: E402
from hal.listener.judge import DeskJudge  # noqa: E402
from listen_harness import make_tmp, rmtree  # noqa: E402
from test_listen_privacy import FakeASR, FakeClock, Rig  # noqa: E402

CFG = dict(settings.DEFAULTS)


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------

class FakeHandle:
    """A judgement that becomes available at a given time on the simulated clock."""

    def __init__(self, clock, kind, ready_at):
        self.clock, self._kind, self.ready_at = clock, kind, ready_at
        self.why = "stand-in"
        self.seconds = 0.5

    @property
    def done(self):
        return self.clock() >= self.ready_at

    @property
    def kind(self):
        return self._kind if self.done else None


class FakeJudge:
    """answer(text) -> (kind or None, seconds until it answers)."""

    def __init__(self, clock, answer=lambda text: ("complete", 0.5)):
        self.clock, self.answer = clock, answer
        self.asked = []

    def __call__(self, text):
        self.asked.append(text)
        kind, delay = self.answer(text)
        return FakeHandle(self.clock, kind, self.clock() + delay)


def track(*pieces, lead=1.5):
    """pieces: (seconds of speech, seconds of silence after it). Speech is loud, silence is zeros."""
    parts = [np.zeros(int(lead * 16000), np.float32)]
    for speech, gap in pieces:
        parts += [np.full(int(speech * 16000), 0.5, np.float32), np.zeros(int(gap * 16000), np.float32)]
    return np.concatenate(parts)


def starts(*pieces, lead=1.5):
    """Start time of each piece of speech on the rig's clock (1000.0 is the start of the audio)."""
    t, out = 1000.0 + lead, []
    for speech, gap in pieces:
        out.append(t)
        t += speech + gap
    return out


class RigTest(unittest.TestCase):
    def rig(self, pieces, texts, **kw):
        rig = Rig(track(*pieces), texts, **kw)
        self.addCleanup(rig.close)
        return rig

    def run_rig(self, pieces, texts, **kw):
        rig = self.rig(pieces, texts, **kw)
        rig.lst.run()
        return rig

    def sent(self, rig):
        return [e for e in rig.lst.events if e["action"] == "submit"]


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

class RulesTest(unittest.TestCase):
    def judge(self, text):
        return turns.judge_rules(text, CFG)[0]

    def test_his_own_fragments_are_unfinished(self):
        for text in ("Another topic in mind. In fact there are several, but for the moment...",
                     "and lengthen the pause for",
                     "Another topic in mind. In fact there are several, but for the moment... and lengthen the "
                     "pause for"):
            self.assertEqual(self.judge(text), "incomplete", text)

    def test_unfinished(self):
        for text in ("Claude, I'd like you to look at the draft of my paper and", "Claude, open the",
                     "Claude, could you look at?", "Claude, the thing is", "Claude, what I'd like to know is",
                     "Claude, compare the two drafts, such as", "So.", "Um.", "Claude, I was wondering",
                     "Claude, check the paper,", "Claude, send it to", "Claude, there are two options:",
                     "Claude, choose a.", "Claude, the bug is in the parser because",
                     "Claude, open the second chapter of my thesis and then.", "and lengthen the pause for",
                     "Another topic in mind, in fact there are several, but for the moment."):
            self.assertEqual(self.judge(text), "incomplete", text)

    def test_finished(self):
        for text in ("Claude, what's the weather tomorrow?", "Good morning, Claude.", "Thanks, Claude.",
                     "Claude, open my thesis.", "By the way, Claude, open my thesis.", "Open my thesis.",
                     "Please read me the last paragraph.", "Can you switch it on?", "What is it for?",
                     "Claude, can you open the second chapter.", "It's very helpful.", "Yes.", "Okay.",
                     "Claude, do you only recognize my speech, or will you recognize anyone else in the room?",
                     "Hello, Claude, can you hear me?"):
            self.assertEqual(self.judge(text), "complete", text)

    def test_unclear_goes_to_the_judge(self):
        for text in ("Claude, I think the second section of the paper is too long.",
                     "Claude, can I ask you something?", "Claude, can you tell me?",
                     "Claude, I have a question about the paper.", "Find the book I was looking for.",
                     "claude what time is it", "Leave it for the moment.", "Why?",
                     "Claude, I'd like you to look at the draft and check whether the argument in the second "
                     "section."):
            self.assertEqual(self.judge(text), "unclear", text)

    def test_controls(self):
        c = lambda t: turns.control(t, CFG)  # noqa: E731
        self.assertEqual(c("That's all."), ("send", ""))
        self.assertEqual(c("Okay, go ahead, Claude."), ("send", ""))
        self.assertEqual(c("That's all, thank you."), ("send", ""))
        self.assertEqual(c("and the weather in London. That's all."), ("send", "and the weather in London."))
        self.assertEqual(c("No, never mind."), ("cancel", ""))
        self.assertEqual(c("Cancel that."), ("cancel", ""))
        self.assertEqual(c("Scratch that. Open my thesis instead."), ("restart", "Open my thesis instead."))
        for text in ("Cancel the meeting with the dean.", "Never mind the weather, tell me about Paris.",
                     "Yes, go ahead and do that.", "Forget about the draft."):
            self.assertEqual(c(text)[0], None, text)


# ---------------------------------------------------------------------------
# The turn-taker alone, on a simulated clock
# ---------------------------------------------------------------------------

class TurnTakerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(100.0)
        self.judge = None

    def taker(self, judge=None, **over):
        self.judge = judge
        return turns.TurnTaker(dict(CFG, **over), judge=judge, clock=self.clock)

    def add(self, tt, text, end, how="named"):
        self.clock.t = end + CFG["end_silence"]
        return tt.add(text, how=how, speech_start=end - 1.0, speech_end=end)

    def run_until(self, tt, until, in_speech=lambda t: False, step=0.1):
        while self.clock.t < until:
            self.clock.t = round(self.clock.t + step, 3)
            out = tt.tick(in_speech(self.clock.t))
            if out is not None and out.action != "cue":
                return out
        return None

    def test_complete_is_sent_after_turn_end_silence(self):
        tt = self.taker()
        self.assertEqual(self.add(tt, "Claude, open my thesis.", 100.0).action, "wait")
        out = self.run_until(tt, 110)
        self.assertEqual((out.action, out.text), ("send", "Claude, open my thesis."))
        self.assertAlmostEqual(self.clock.t - 100.0, CFG["turn_end_silence"], delta=0.11)

    def test_incomplete_waits_through_a_long_pause_and_joins(self):
        tt = self.taker()
        self.add(tt, "Claude, I'd like you to look at the draft of my paper and", 100.0)
        self.assertIsNone(self.run_until(tt, 130.0), "a 30 s pause must not send an unfinished turn")
        self.add(tt, "check it against the objection from the second reviewer.", 131.0)
        out = self.run_until(tt, 140)
        self.assertEqual(out.text, "Claude, I'd like you to look at the draft of my paper and check it against "
                                   "the objection from the second reviewer.")

    def test_maximum_wait(self):
        tt = self.taker(turn_max_wait=20.0)
        self.add(tt, "Claude, I'd like you to look at", 100.0)
        out = self.run_until(tt, 200)
        self.assertEqual((out.action, out.text), ("send", "Claude, I'd like you to look at"))
        self.assertAlmostEqual(self.clock.t - 100.0, 20.0, delta=0.11)
        self.assertIn("20s of silence", out.why)

    def test_judge_says_complete(self):
        judge = FakeJudge(self.clock, lambda text: ("complete", 1.2))
        tt = self.taker(judge)
        self.add(tt, "Claude, I think the second section is too long.", 100.0)
        self.assertEqual(tt.turn.verdict, "judging")
        out = self.run_until(tt, 110)
        self.assertEqual(out.action, "send")
        self.assertAlmostEqual(self.clock.t, 100.0 + CFG["end_silence"] + 1.2, delta=0.11)
        self.assertEqual(judge.asked, ["Claude, I think the second section is too long."])

    def test_judge_says_incomplete(self):
        judge = FakeJudge(self.clock, lambda text: ("incomplete", 1.0) if "question" in text else ("complete", 1.0))
        tt = self.taker(judge)
        self.add(tt, "Claude, I have a question about the paper.", 100.0)
        self.assertIsNone(self.run_until(tt, 125.0))
        self.assertEqual(tt.turn.verdict, "incomplete")
        self.add(tt, "Is the second section too long?", 126.0)   # a question: finished by rule, no judge
        out = self.run_until(tt, 140)
        self.assertEqual(out.text, "Claude, I have a question about the paper. Is the second section too long?")
        self.assertEqual(len(judge.asked), 1)

    def test_rules_decide_without_the_judge(self):
        judge = FakeJudge(self.clock)
        tt = self.taker(judge)
        self.add(tt, "Claude, what's the weather tomorrow?", 100.0)
        self.run_until(tt, 105)
        self.add(tt, "Claude, open the", 110.0)
        self.assertEqual(judge.asked, [], "rules settle these; the judge is not asked")

    def test_judge_unavailable_falls_back_to_the_timer(self):
        tt = self.taker(FakeJudge(self.clock, lambda text: (None, 0.01)))   # not running: answers at once, no kind
        self.add(tt, "Claude, I think the second section is too long.", 100.0)
        out = self.run_until(tt, 110)
        self.assertEqual(out.action, "send")
        self.assertAlmostEqual(self.clock.t - 100.0, CFG["turn_unclear_wait"], delta=0.11)

    def test_judge_too_slow_falls_back_to_the_timer(self):
        tt = self.taker(FakeJudge(self.clock, lambda text: ("incomplete", 60.0)))
        self.add(tt, "Claude, I think the second section is too long.", 100.0)
        out = self.run_until(tt, 110)
        self.assertEqual(out.action, "send")
        # The judge's time is up at end_silence + timeout + 0.5; by then the unclear wait has passed.
        self.assertAlmostEqual(self.clock.t - 100.0, CFG["end_silence"] + CFG["turn_judge_timeout"] + 0.5,
                               delta=0.15)
        self.assertIn("too slow", tt_reason(out))

    def test_no_judge_at_all(self):
        tt = self.taker(None)
        self.add(tt, "Claude, I think the second section is too long.", 100.0)
        out = self.run_until(tt, 110)
        self.assertAlmostEqual(self.clock.t - 100.0, CFG["turn_unclear_wait"], delta=0.11)
        self.assertEqual(out.action, "send")

    def test_resumed_speech_holds_even_a_complete_turn(self):
        tt = self.taker()
        self.add(tt, "Claude, open my thesis.", 100.0)
        self.assertIsNone(self.run_until(tt, 103.0, in_speech=lambda t: t >= 100.9))
        self.add(tt, "And go to chapter three.", 103.5)
        out = self.run_until(tt, 110)
        self.assertEqual(out.text, "Claude, open my thesis. And go to chapter three.")

    def test_send_phrase(self):
        tt = self.taker()
        self.add(tt, "Claude, I'd like you to compare the two drafts and", 100.0)
        out = self.add(tt, "That's all.", 105.0)
        self.assertEqual((out.action, out.text), ("send", "Claude, I'd like you to compare the two drafts and"))
        self.assertIsNone(tt.turn)
        tt.add("Claude, look at the draft and", how="named", speech_start=110, speech_end=111)
        out = tt.add("the second chapter. Go ahead.", how="continued", speech_start=115, speech_end=116)
        self.assertEqual(out.text, "Claude, look at the draft and the second chapter.")

    def test_send_phrase_in_the_first_fragment_is_ordinary_words(self):
        tt = self.taker()
        self.add(tt, "Yes, please go ahead.", 100.0)
        self.assertEqual(tt.turn.verdict, "complete")
        self.assertEqual(self.run_until(tt, 105).text, "Yes, please go ahead.")

    def test_cancel_phrase(self):
        tt = self.taker()
        self.add(tt, "Claude, I'd like you to email the dean about", 100.0)
        out = self.add(tt, "No, never mind.", 104.0)
        self.assertEqual(out.action, "cancel")
        self.assertIsNone(tt.turn)
        self.assertIsNone(self.run_until(tt, 200))

    def test_cancel_then_start_afresh(self):
        tt = self.taker()
        self.add(tt, "Claude, I'd like you to email the dean about", 100.0)
        self.add(tt, "Scratch that. Open my thesis instead.", 104.0)
        self.assertEqual(self.run_until(tt, 110).text, "Open my thesis instead.")

    def test_the_name_then_a_pause_opens_an_empty_turn(self):
        tt = self.taker(turn_max_wait=10.0)
        tt.open("named", 100.0)
        self.assertTrue(tt.pending)
        out = self.run_until(tt, 200)
        self.assertEqual(out.action, "drop", "nothing followed: nothing is sent")
        tt.open("named", 300.0)
        self.clock.t = 305.0
        self.add(tt, "what's the weather tomorrow?", 306.0, how="continued")
        self.assertEqual(self.run_until(tt, 310).text, "what's the weather tomorrow?")

    def test_one_cue_while_waiting(self):
        tt = self.taker(turn_cue="sound", turn_cue_after=5.0)
        self.add(tt, "Claude, I'd like you to look at", 100.0)
        cues = []
        while self.clock.t < 130:
            self.clock.t = round(self.clock.t + 0.1, 3)
            out = tt.tick(False)
            if out is not None:
                cues.append((round(self.clock.t - 100.0, 1), out.action))
        self.assertEqual(cues, [(5.0, "cue")])

    def test_no_cue_by_default(self):
        tt = self.taker()
        self.add(tt, "Claude, I'd like you to look at", 100.0)
        out = self.run_until(tt, 200)
        self.assertEqual(out.action, "send")

    def test_rejoined_audio_is_preferred(self):
        tt = turns.TurnTaker(CFG, clock=self.clock, rejoin=lambda parts: f"Cloud, look at the draft and the "
                                                                         f"chapter ({len(parts)} parts).")
        tt.add("Claude, look at the draft and", how="named", speech_start=99, speech_end=100, audio=[np.ones(10)])
        out = tt.add("the chapter.", how="continued", speech_start=104, speech_end=105, audio=[np.ones(10)])
        self.assertEqual(out.text, "Claude, look at the draft and the chapter (2 parts).",
                         "transcribed afresh, the name variant rewritten")


def tt_reason(out):
    return out.turn.reason if out.turn else out.why


# ---------------------------------------------------------------------------
# The listening loop
# ---------------------------------------------------------------------------

class ListenerTurnsTest(RigTest):
    def test_one_request_across_pauses_of_3_10_and_30_seconds(self):
        texts = ["Claude, could you look at the draft of my paper and",
                 "check whether the argument in the second section holds up against",
                 "the objection raised in the report from",
                 "the second reviewer?"]
        pieces = [(2.0, 3.0), (2.5, 10.0), (1.5, 30.0), (1.0, 5.0)]
        rig = self.run_rig(pieces, texts)
        self.assertEqual(rig.lst.injector.typed, [" ".join(texts)])
        sent = self.sent(rig)[0]
        self.assertEqual(sent["fragments"], 4)
        last_end = starts(*pieces)[-1] + 1.0
        self.assertAlmostEqual(sent["at"] - last_end, CFG["turn_end_silence"], delta=0.1)

    def test_short_requests_are_sent_promptly(self):
        texts = ["Claude, what's the weather tomorrow?", "Claude, open my thesis."]
        rig = self.run_rig([(1.5, 4.0), (1.2, 4.0)], texts)
        self.assertEqual(rig.lst.injector.typed, texts)
        for e in self.sent(rig):
            self.assertAlmostEqual(e["at"] - e["speech_end"], CFG["turn_end_silence"], delta=0.1)

    def test_the_judge_decides_unclear_turns(self):
        texts = ["Claude, I have a question about the paper.", "Is the second section too long?",
                 "Claude, I think the conclusion is weak."]
        make = lambda clock: FakeJudge(clock, lambda t: ("incomplete", 1.2) if t.endswith("paper.")  # noqa: E731
                                       else ("complete", 1.2))
        rig = self.run_rig([(2.0, 8.0), (1.5, 5.0), (2.0, 5.0)], texts, make_judge=make)
        self.assertEqual(rig.lst.injector.typed, [texts[0] + " " + texts[1], texts[2]])
        self.assertEqual(rig.lst.turns.judge.asked, [texts[0], texts[2]])

    def test_silence_mode_keeps_the_old_behaviour(self):
        texts = ["Claude, could you look at the draft of my paper and", "the second chapter."]
        rig = self.run_rig([(2.0, 3.0), (1.5, 3.0)], texts, turn_taking="silence")
        self.assertEqual(rig.lst.injector.typed, [texts[0]], "without the name the second part is ignored")
        self.assertIsNone(rig.lst.turns)

    def test_closing_and_cancelling_phrases(self):
        texts = ["Claude, I'd like you to compare the two drafts and", "That's all.",
                 "Claude, please email the dean about", "Never mind.",
                 "The weather is lovely today."]   # after the cancel: not addressed, ignored
        rig = self.run_rig([(2.0, 4.0), (0.8, 4.0), (1.5, 4.0), (0.8, 4.0), (1.5, 3.0)], texts)
        self.assertEqual(rig.lst.injector.typed, ["Claude, I'd like you to compare the two drafts and"])
        self.assertEqual([e["action"] for e in rig.lst.events if e["action"] in ("submit", "cancel", "ignore")],
                         ["submit", "cancel", "ignore"])
        # The send phrase is answered at once, not after a silence.
        sent = self.sent(rig)[0]
        self.assertLess(sent["at"] - sent["speech_end"], CFG["end_silence"] + 0.1)

    def test_maximum_wait_then_later_speech_needs_the_name(self):
        texts = ["Claude, I'd like you to look at", "Open my thesis."]
        rig = self.run_rig([(1.5, 25.0), (1.0, 3.0)], texts, turn_max_wait=20.0)
        self.assertEqual(rig.lst.injector.typed, ["Claude, I'd like you to look at"])
        self.assertEqual(rig.lst.events[-1]["action"], "ignore")

    def test_name_only_still_answers_and_arms(self):
        texts = ["Claude?", "Open my thesis."]
        rig = self.run_rig([(0.6, 3.0), (1.0, 3.0)], texts)
        self.assertEqual(rig.lst.voice.said, [CFG["name_only_reply"]])
        self.assertEqual(rig.lst.injector.typed, ["Open my thesis."])
        self.assertEqual(self.sent(rig)[0]["how"], "armed")

    def test_the_name_then_a_comma_waits_without_answering(self):
        texts = ["Claude,", "would it be possible to make the pause longer?"]
        rig = self.run_rig([(0.6, 6.0), (2.0, 3.0)], texts)
        self.assertEqual(rig.lst.voice.said, [])
        self.assertEqual(rig.lst.injector.typed, ["would it be possible to make the pause longer?"])

    def test_the_name_then_um_waits_too(self):
        self.assertTrue(turns.opens_with_name("Claude, um."))
        self.assertTrue(turns.opens_with_name("Claude..."))
        for text in ("Claude.", "Claude?", "Hey Claude.", "Um."):
            self.assertFalse(turns.opens_with_name(text), text)
        rig = self.run_rig([(0.8, 5.0), (2.0, 3.0)], ["Claude, um.", "what's on my calendar tomorrow?"])
        self.assertEqual(rig.lst.voice.said, [])
        self.assertEqual(rig.lst.injector.typed, ["what's on my calendar tomorrow?"])

    def test_stop_listening_discards_a_pending_turn(self):
        texts = ["Claude, I'd like you to look at", "Claude, stop listening."]
        rig = self.run_rig([(1.5, 4.0), (1.2, 4.0)], texts)
        self.assertEqual(rig.lst.injector.typed, [])
        self.assertTrue((rig.tmp / "state" / "listen" / "paused").exists())
        self.assertIn("discarded (stop listening)", (rig.tmp / "state" / "listen.log").read_text())

    def test_gate_still_drops_speech_while_the_voice_speaks(self):
        texts = ["Claude, I'd like you to look at the draft and", "the second chapter."]
        pieces = [(2.0, 2.0), (1.5, 2.0), (1.5, 4.0)]
        s = starts(*pieces)
        # The voice starts speaking during the second piece: it is dropped unheard; the turn waits, and the
        # third joins it.
        rig = self.run_rig(pieces, texts, blocked=lambda t: s[1] + 0.5 <= t <= s[1] + 2.5)
        self.assertEqual(rig.lst.asr.calls, 2)
        self.assertEqual(rig.lst.injector.typed, [" ".join(texts)])
        self.assertEqual([e["action"] for e in rig.lst.events if e["action"] == "discarded"], ["discarded"])

    def test_follow_up_window_still_works(self):
        texts = ["And the day after?", "And the week after that?"]
        rig = self.rig([(1.0, 4.0), (1.0, 4.0)], texts, follow_up_seconds=3.0)
        rig.lst.gate.reply_end = 1000.5          # a reply finished just before the first piece
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, ["And the day after?"])
        self.assertEqual(self.sent(rig)[0]["how"], "follow-up")

    def test_long_utterance_parts_are_one_fragment(self):
        texts = ["Claude, here is a long dictation", "that keeps going", "and ends here."]
        rig = self.run_rig([(65.0, 4.0)], texts)
        self.assertEqual(rig.lst.injector.typed, ["Claude, here is a long dictation that keeps going and ends here."])
        self.assertEqual(self.sent(rig)[0]["fragments"], 1)

    def test_the_whole_audio_is_transcribed_afresh(self):
        class RejoiningASR(FakeASR):
            rejoins = True

            def transcribe(self, audio):
                if len(audio) > 3.2 * 16000:     # longer than any one piece: the whole turn
                    return "Claude, could you look at the draft, and the second chapter?"
                return super().transcribe(audio)

        texts = ["Claude, could you look at the draft and", "the second chapter?"]
        rig = self.run_rig([(2.0, 5.0), (1.0, 4.0)], texts, asr=RejoiningASR(texts))
        self.assertEqual(rig.lst.injector.typed, ["Claude, could you look at the draft, and the second chapter?"])

    def test_cue_sound_once(self):
        played = []
        rig = self.run_rig([(1.5, 20.0), (1.0, 3.0)], ["Claude, I'd like you to look at", "the second chapter."],
                           turn_cue="sound", turn_cue_after=8.0, cue_sound=played.append)
        self.assertEqual(played, [CFG["turn_cue_sound"]])
        self.assertEqual(rig.lst.voice.said, [])

    def test_status_shows_the_turn_without_words(self):
        rig = self.rig([(1.5, 30.0)], ["Claude, I'd like you to look at the secret plan for"])
        rig.lst.linger = 1.0
        seen = []
        orig = rig.lst._timers

        def spy():
            orig()
            if rig.lst.turns.pending and not seen:
                rig.lst._status()
                seen.append(json.loads((rig.tmp / "state" / "listen" / "listener.json").read_text()))
        rig.lst._timers = spy
        rig.lst.run()
        self.assertEqual(seen[0]["turn"]["fragments"], 1)
        self.assertNotIn("secret", json.dumps(seen[0]))


class DuringTurnTest(RigTest):
    """during_turn: while Claude works on a spoken request, speech needs no name."""

    def session(self, state):
        return lambda: state

    def test_speech_while_the_turn_is_busy_is_typed(self):
        st = {"turn": "busy", "last_stop": 0.0}
        texts = ["Claude, what's the weather tomorrow?", "And please include the wind speed."]
        rig = self.run_rig([(1.5, 3.0), (2.0, 4.0)], texts, session_state=self.session(st))
        self.assertEqual(rig.lst.injector.typed, texts)
        self.assertEqual(self.sent(rig)[1]["how"], "during-turn")

    def test_it_ends_with_the_turn(self):
        st = {"turn": "busy", "last_stop": 0.0}
        texts = ["Claude, what's the weather tomorrow?", "We should visit grandmother on Sunday."]
        rig = self.rig([(1.5, 3.0), (2.0, 4.0)], texts, session_state=self.session(st))
        orig = rig.lst.injector.submit

        def submit(text, speech_end):
            orig(text, speech_end)
            st.update(turn="idle", last_stop=time.time() + 1)   # Claude answered: the turn is over
        rig.lst.injector.submit = submit
        rig.lst.run()
        self.assertEqual(rig.lst.injector.typed, texts[:1])
        self.assertNotIn("grandmother", (rig.tmp / "state" / "listen.log").read_text())

    def test_not_after_a_turn_started_at_the_keyboard(self):
        texts = ["Claude, what's the weather tomorrow?", "We should visit grandmother on Sunday."]
        st = {"turn": "busy", "last_stop": time.time() + 3600}   # stopped after our typing, then busy again
        rig = self.run_rig([(1.5, 3.0), (2.0, 4.0)], texts, session_state=self.session(st))
        self.assertEqual(rig.lst.injector.typed, texts[:1])

    def test_not_without_a_spoken_request(self):
        st = {"turn": "busy", "last_stop": 0.0}
        rig = self.run_rig([(2.0, 4.0)], ["We should visit grandmother on Sunday."], session_state=self.session(st))
        self.assertEqual(rig.lst.injector.typed, [])

    def test_setting_off(self):
        st = {"turn": "busy", "last_stop": 0.0}
        texts = ["Claude, what's the weather tomorrow?", "And please include the wind speed."]
        rig = self.run_rig([(1.5, 3.0), (2.0, 4.0)], texts, session_state=self.session(st), during_turn=False)
        self.assertEqual(rig.lst.injector.typed, texts[:1])

    def test_reads_the_session_file_the_hooks_keep(self):
        """The default session_state: the state file kept by `hal-listen hook`, through the real reducer."""
        from hal.listener import session
        from hal.listener.service import Listener

        rig = self.rig([(1.0, 1.0)], [])
        pid = os.getpid()   # stands for the proxy: alive
        lst = Listener(rig.lst.cfg, source=rig.lst.source, vad=rig.lst.vad, asr=rig.lst.asr, gate=rig.lst.gate,
                       injector=rig.lst.injector, decider=rig.lst.decider, voice=rig.lst.voice, clock=rig.clock,
                       proxy_pid=pid)
        session.update(pid, {"hook_event_name": "SessionStart", "source": "startup"})
        self.assertFalse(lst._during_turn())
        lst.injector.last_typed = time.time()
        session.update(pid, {"hook_event_name": "UserPromptSubmit", "prompt": "x"})
        self.assertTrue(lst._during_turn())
        session.update(pid, {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {}})
        self.assertTrue(lst._during_turn(), "still working")
        session.update(pid, {"hook_event_name": "Stop"})
        self.assertFalse(lst._during_turn(), "the turn has ended")
        session.update(pid, {"hook_event_name": "UserPromptSubmit", "prompt": "typed at the keyboard"})
        self.assertFalse(lst._during_turn(), "a later turn the listener didn't start")
        lst.injector.last_typed = time.time() - 3600 * 2
        self.assertFalse(lst._during_turn())

    def test_fragments_during_the_turn_are_joined_too(self):
        st = {"turn": "busy", "last_stop": 0.0}
        texts = ["Claude, what's the weather tomorrow?", "And please include the wind speed for",
                 "London and Edinburgh."]
        rig = self.run_rig([(1.5, 3.0), (2.0, 12.0), (1.5, 4.0)], texts, session_state=self.session(st))
        self.assertEqual(rig.lst.injector.typed, [texts[0], texts[1] + " " + texts[2]])


# ---------------------------------------------------------------------------
# The client for the front desk's "turn" request
# ---------------------------------------------------------------------------

class StandInDesk:
    """A Unix socket that answers one JSON line per connection, as the desk does."""

    def __init__(self, path, answer):
        self.path, self.answer, self.requests = path, answer, []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(4)
        self.sock.settimeout(0.2)
        self.stop = False
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while not self.stop:
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                req = json.loads(conn.makefile().readline())
                self.requests.append(req)
                delay, reply = self.answer(req)
                time.sleep(delay)
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    pass

    def close(self):
        self.stop = True
        self.thread.join(2)
        self.sock.close()


class DeskJudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp("hal-dj-")
        self.path = str(self.tmp / "desk.sock")
        self.addCleanup(rmtree, self.tmp)

    def desk(self, answer):
        d = StandInDesk(self.path, answer)
        self.addCleanup(d.close)
        return d

    def judged(self, timeout=1.0):
        handle = DeskJudge(timeout, path_fn=lambda: self.path)("Claude, I think the conclusion is weak.")
        self.assertTrue(handle.wait(timeout + 2))
        return handle

    def test_answer(self):
        d = self.desk(lambda req: (0.1, {"kind": "incomplete", "seconds": 0.1}))
        h = self.judged()
        self.assertEqual(h.kind, "incomplete")
        self.assertEqual(d.requests[0]["op"], "turn")
        self.assertEqual(d.requests[0]["text"], "Claude, I think the conclusion is weak.")

    def test_not_running(self):
        started = time.monotonic()
        h = self.judged()
        self.assertIsNone(h.kind)
        self.assertIn("not running", h.why)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_absent_desk_is_woken_for_next_time(self):
        woken = []
        handle = DeskJudge(1.0, path_fn=lambda: self.path, on_absent=lambda: woken.append(1))("Claude, hello.")
        self.assertTrue(handle.wait(3))
        self.assertEqual((handle.kind, woken), (None, [1]))
        self.desk(lambda req: (0.0, {"kind": "complete", "seconds": 0.0}))
        handle = DeskJudge(1.0, path_fn=lambda: self.path, on_absent=lambda: woken.append(2))("Claude, hello.")
        self.assertTrue(handle.wait(3))
        self.assertEqual((handle.kind, woken), ("complete", [1]))   # a running desk isn't woken

    def test_too_slow(self):
        self.desk(lambda req: (2.0, {"kind": "complete"}))
        h = self.judged(timeout=0.5)
        self.assertIsNone(h.kind)
        self.assertIn("in time", h.why)
        self.assertLess(h.seconds, 1.5)

    def test_desk_from_before_turns(self):
        self.desk(lambda req: (0.0, {"error": "unknown op 'turn'"}))
        h = self.judged()
        self.assertIsNone(h.kind)
        self.assertIn("restart", h.why)

    def test_unavailable(self):
        self.desk(lambda req: (0.0, {"kind": "unavailable", "why": "no turn model running"}))
        self.assertEqual(self.judged().why, "no turn model running")


class SettingsTest(unittest.TestCase):
    def test_new_keys(self):
        tmp = make_tmp()
        self.addCleanup(rmtree, tmp)
        path = tmp / "c.json"
        path.write_text(json.dumps({"listener": {"turn_taking": "Silence", "turn_cue": "loud", "turn_max_wait": 0,
                                                 "turn_judge_timeout": 60, "turn_send_phrases": ["Over and out"],
                                                 "during_turn": "yes", "turn_judge": False}}))
        cfg = settings.load(path)
        self.assertEqual(cfg["turn_taking"], "silence")
        self.assertEqual(cfg["turn_cue"], "none")
        self.assertEqual(cfg["turn_max_wait"], 1.0)
        self.assertEqual(cfg["turn_judge_timeout"], 5.0)
        self.assertEqual(cfg["turn_send_phrases"], ["over and out"])
        self.assertIs(cfg["during_turn"], True)
        self.assertIs(cfg["turn_judge"], False)
        self.assertEqual(settings.load(tmp / "missing.json")["turn_taking"], "semantic")


if __name__ == "__main__":
    unittest.main()
