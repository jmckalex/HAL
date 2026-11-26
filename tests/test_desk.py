"""Tests of the front desk (hal/desk.py), end to end through the real hook command.

Like tests/test_hook.py, every hook call pipes JSON into the command string
from .claude/settings.json, state and config are temporary (HAL_STATE_DIR,
HAL_CONFIG), speech is really synthesised and "played" by the silent
tests/fake_player.py. The desk runs as its own process, with
tests/fake_warm_model.py standing in for Haiku (HAL_WARM_MODEL); the stand-in
answers according to words in the prompt ("slow", "pause", "garbage",
"chatter", plain greetings).

Run: .venv/bin/python -m unittest tests.test_desk -v      (takes ~1.5 min)
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hook import LONG_REPLY, ROOT, HookTestBase  # noqa: E402

HAL = str(ROOT / "bin" / "hal")
LONG_PROMPT = ("pause please look through the build logs from last night and tell me which of the tests "
               "failed and why they failed")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); ps shows it as Z.
    state = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


class DeskTest(HookTestBase):
    def setUp(self):
        super().setUp()
        self.env["HAL_WARM_MODEL"] = (f"{shlex.quote(sys.executable)} "
                                      f"{shlex.quote(str(ROOT / 'tests' / 'fake_warm_model.py'))}")
        self.pids = set()

    def tearDown(self):
        info = self.desk_status()
        if info:
            self.pids.add(info["pid"])
            for role in ("desk_model", "summary_model", "turn_model"):
                if info.get(role):
                    self.pids.add(info[role]["pid"])
        self.pids.update(int(line.split()[1]) for line in self.model_calls(role=None))
        subprocess.run([HAL, "desk", "stop"], env=self.env, capture_output=True, timeout=20)
        super().tearDown()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and any(alive(p) for p in self.pids):
            time.sleep(0.1)
        leftover = [p for p in self.pids if alive(p)]
        for pid in leftover:
            os.kill(pid, signal.SIGKILL)
        self.assertEqual(leftover, [], "the desk and its models must all exit")

    # -- helpers ------------------------------------------------------------

    def set_config(self, **overrides):
        settings = {"voice_sessions": "master", "spoken_reply": "full", "acknowledge": True,
                    "ack_phrases": ["One moment, please."], "max_chars": 600, "front_desk": True,
                    "desk_wait_seconds": 1.2, "user_name": "Dr. Alexander", "desk_idle_minutes": 5}
        settings.update(overrides)
        self.config.write_text(json.dumps(settings))

    def start_desk(self):
        proc = subprocess.run([HAL, "desk", "start"], env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("running", proc.stdout)

    def desk_status(self):
        proc = subprocess.run([HAL, "desk", "status", "--json"], env=self.env, capture_output=True, text=True,
                              timeout=10)
        return json.loads(proc.stdout or "null")

    def prompt(self, text, stdout=b""):
        return self.run_hook(self.prompt_payload(text), event="UserPromptSubmit", stdout=stdout)

    def model_calls(self, role="desk"):
        try:
            lines = (self.out / "warm_model.log").read_text().splitlines()
        except FileNotFoundError:
            return []
        return [line for line in lines if role is None or line.split(" ", 1)[0] == role]

    def timed_events(self):
        """(time, event) pairs from the silent player."""
        try:
            lines = self.events.read_text().splitlines()
        except FileNotFoundError:
            return []
        return [(float(line.split(" ", 2)[0]), line.split(" ", 2)[2]) for line in lines]

    def starts(self):
        return [e for e in self.event_lines() if e.startswith("start")]

    def wait_until(self, predicate, timeout=10.0, what="condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}; log:\n{self.log_text()}")

    # -- social remarks -----------------------------------------------------

    def test_social_reply_blocks_the_prompt_and_is_spoken(self):
        self.start_desk()
        called = time.time()
        elapsed = self.prompt("Good morning, Claude.", stdout=None)
        decision = json.loads(self.last_stdout)
        self.assertEqual(decision["decision"], "block")
        self.assertIn("Good morning, Dr. Alexander.", decision["reason"])
        self.assertLess(elapsed, 2.0)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        first_start = self.timed_events()[0][0]
        self.assertLess(first_start - called, 2.5, "speech should start soon after the hook is called")
        log = self.log_text()
        self.assertIn("prompt blocked", log)
        self.assertIn("Doctor Alexander", log, "the spoken text goes through pronounce()")
        self.assertNotIn("acknowledged (", log, "no canned acknowledgement for a remark the desk answered")
        self.assertIn("Good morning, Claude.", self.model_calls()[0])
        prompt_given = (self.out / "warm_model_prompt.desk.txt").read_text()
        self.assertIn("Dr. Alexander", prompt_given)
        self.assertIn("You are Claude", prompt_given)

    def test_blocked_exchanges_reach_claude_once_with_the_spoken_summary(self):
        self.set_config(spoken_reply="summary", summary_min_chars=350)
        self.start_desk()
        # A long reply is summarised by the desk's warm model, not the one-shot call.
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertEqual(self.summarizer_calls(), [], "the one-shot summariser should not be needed")
        self.assertEqual(len(self.model_calls("summary")), 1)
        self.assertIn("summary from the front desk", self.log_text())
        # Two social remarks are answered and blocked; Claude hears nothing yet.
        for remark in ("Good morning, Claude.", "Thank you."):
            self.prompt(remark, stdout=None)
            self.assertEqual(json.loads(self.last_stdout)["decision"], "block")
        # The next prompt that goes through carries both notes.
        self.prompt("What is the weather tomorrow?", stdout=None)
        note = self.last_stdout
        self.assertNotIn('"decision"', note)
        self.assertIn("spoken summary", note)
        self.assertIn("All is well, and Hal has one question for you.", note)
        self.assertIn('User: "Good morning, Claude." Front desk: "Good morning, Dr. Alexander. I trust you slept well."',
                      note)
        self.assertLess(note.index("Good morning, Claude."), note.index("Thank you."))
        # Only once.
        self.prompt("And the day after?")

    def test_blocking_off_lets_the_remark_through_and_mutes_the_reply(self):
        self.set_config(desk_block=False)
        self.start_desk()
        self.prompt("Good morning, Claude.", stdout=None)
        self.assertNotIn('"decision"', self.last_stdout)
        self.assertIn("has already answered this remark aloud", self.last_stdout)
        self.assertIn("Good morning, Dr. Alexander.", self.last_stdout)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.run_hook(self.stop_payload("Good morning."))  # Claude's brief reply
        time.sleep(1.5)
        self.assertEqual(len(self.starts()), 1, "Claude's reply must not be spoken")
        self.assertNotIn("killed", self.event_lines(), "nor cut off the desk's reply")
        self.assertIn("Stop: silent", self.log_text())
        self.prompt("What is the weather tomorrow?")  # nothing was deferred

    # -- acknowledgements ---------------------------------------------------

    def test_tailored_acknowledgement_is_spoken_and_the_prompt_goes_through(self):
        self.start_desk()
        called = time.time()
        elapsed = self.prompt("What is the weather tomorrow?")
        self.assertLess(elapsed, 1.5)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertLess(self.timed_events()[0][0] - called, 2.5)
        log = self.log_text()
        self.assertIn("front desk decided", log)
        self.assertIn("said \"Certainly. I'll look into that.\"", log)
        self.assertNotIn("acknowledged (", log)
        # A fast reply waits for the acknowledgement to finish.
        self.run_hook(self.stop_payload("It will rain."))
        events = self.wait_for(lambda e: sum(x.startswith("start") for x in e) >= 2)
        self.assertEqual([x.split()[0] for x in events[:3]], ["start", "end", "start"])

    def test_long_prompt_does_not_wait(self):
        self.set_config(desk_wait_seconds=2.5)
        self.start_desk()
        called = time.time()
        elapsed = self.prompt(LONG_PROMPT)
        self.assertLess(elapsed, 0.6, "a long prompt must not wait for the model")
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertGreater(self.timed_events()[0][0] - called, 1.5, "spoken once the model had answered")
        log = self.log_text()
        self.assertIn("long prompt: front desk will acknowledge it", log)
        self.assertIn("said \"Certainly. I'll look into that.\"", log)

    def test_brief_answers_go_straight_to_claude(self):
        self.start_desk()
        elapsed = self.prompt("OK.")
        self.assertLess(elapsed, 0.6, "a brief answer never waits for the desk")
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertIn("brief answer: straight to Claude; acknowledged ('Very well.')", self.log_text())
        # Praise that may also be an answer goes straight through, without an acknowledgement.
        self.prompt("Perfect, thank you.")
        self.assertIn("brief answer: straight to Claude; thanks or a greeting, so no acknowledgement", self.log_text())
        time.sleep(1.0)
        self.assertEqual(len(self.starts()), 1)
        self.assertEqual(self.model_calls(), [], "the desk never sees a brief answer")

    def test_commands_skip_the_desk(self):
        self.start_desk()
        self.prompt("/help")
        time.sleep(1.0)
        self.assertEqual(self.event_lines(), [])
        self.assertEqual(self.model_calls(), [])

    # -- fallbacks ----------------------------------------------------------

    def test_desk_not_running_falls_back_and_is_started(self):
        elapsed = self.prompt("What is the weather tomorrow?")
        self.assertLess(elapsed, 1.0)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        log = self.log_text()
        self.assertIn("front desk not running; starting it", log)
        self.assertIn("acknowledged ('One moment, please.')", log)
        self.wait_until(lambda: self.desk_status() is not None, what="the desk to start")
        self.wait_until(lambda: (self.desk_status().get("desk_model") or {}).get("alive"), what="its model")
        self.prompt("Good morning, Claude.", stdout=None)
        self.assertEqual(json.loads(self.last_stdout)["decision"], "block")

    def test_unreachable_desk_keeps_the_rule_for_thanks(self):
        (self.tmp / "state").mkdir(parents=True, exist_ok=True)
        (self.tmp / "state" / "desk.spawned").touch()  # an automatic start was just attempted
        self.prompt("You did perfectly. Thank you very much.")
        time.sleep(1.5)
        self.assertEqual(self.event_lines(), [])
        log = self.log_text()
        self.assertIn("front desk not running;", log)
        self.assertNotIn("starting it", log)
        self.assertIn("no acknowledgement", log)

    def test_slow_or_senseless_answers_fall_back(self):
        self.start_desk()
        elapsed = self.prompt("slow question")
        self.assertGreater(elapsed, 1.0)
        self.assertLess(elapsed, 2.5)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertIn("acknowledged ('One moment, please.')", self.log_text())
        self.assertIn("no decision within", self.log_text())
        # The answer that comes later (6 s) is not spoken; an answer in neither form falls back.
        time.sleep(5.5)
        self.assertEqual(len(self.starts()), 1)
        self.prompt("garbage")
        self.wait_until(lambda: self.log_text().count("acknowledged (") == 2, timeout=12, what="second fallback")
        self.assertIn("answer in neither form", self.log_text())
        time.sleep(0.5)
        # The desk is back in step: the next answer is its own.
        self.prompt("What is the weather tomorrow?")
        self.wait_until(lambda: "said \"Certainly. I'll look into that.\"" in self.log_text(), what="tailored ack")
        self.assertEqual(self.log_text().count("acknowledged ("), 2)

    def test_front_desk_off_ignores_a_running_desk(self):
        self.start_desk()
        self.set_config(front_desk=False)
        self.prompt("What is the weather tomorrow?")
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertIn("acknowledged ('One moment, please.')", self.log_text())
        self.assertEqual(self.model_calls(), [])

    def test_killed_model_is_restarted(self):
        self.start_desk()
        old = self.desk_status()["desk_model"]["pid"]
        os.kill(old, signal.SIGKILL)
        elapsed = self.prompt("What is the weather tomorrow?", stdout=None)  # still answered somehow, quickly
        self.assertLess(elapsed, 2.5)
        self.assertNotIn('"decision"', self.last_stdout)
        self.wait_until(lambda: (self.desk_status()["desk_model"] or {}).get("pid") not in (None, old)
                        and self.desk_status()["desk_model"]["alive"], what="a new model")
        self.assertIn("exited after", self.log_text())
        self.prompt("Good morning, Claude.", stdout=None)
        self.assertEqual(json.loads(self.last_stdout)["decision"], "block")

    def test_model_is_replaced_after_max_turns(self):
        self.set_config(desk_max_turns=2)
        self.start_desk()
        for text in ("What is the weather tomorrow?", "Run the tests.", "Is the deploy finished?"):
            self.prompt(text)
            time.sleep(1.2)  # the reply, then the supervisor's replacement
        pids = [line.split()[1] for line in self.model_calls()]
        self.assertEqual(len(pids), 3)
        self.assertEqual(pids[0], pids[1])
        self.assertNotEqual(pids[1], pids[2], "a fresh model after two turns")
        self.assertIn("said \"Certainly. I'll look into that.\"", self.log_text().rsplit("desk model started", 1)[1])

    # -- the voice's own controls -------------------------------------------

    def test_stop_request_still_silences(self):
        self.start_desk()
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.prompt("Please stop reading.")
        events = self.wait_for(lambda e: "killed" in e)
        time.sleep(1.5)
        self.assertEqual(self.event_lines(), events)
        self.assertEqual(self.model_calls(), [], "a stop request never reaches the desk")

    def test_turns_not_started_by_the_user_are_left_alone(self):
        self.start_desk()
        for prompt in ("<task-notification>done</task-notification>", "[SYSTEM NOTIFICATION - NOT USER INPUT]",
                       "Another Claude session sent a message:\n<cross-session-message>hi</cross-session-message>"):
            self.prompt(prompt)
        time.sleep(1.0)
        self.assertEqual(self.event_lines(), [])
        self.assertEqual(self.model_calls(), [])

    def test_hal_stop_silences_front_desk_speech(self):
        self.set_config(desk_wait_seconds=2.5)
        self.start_desk()
        self.prompt("chatter with me", stdout=None)
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        subprocess.run([HAL, "stop"], env=self.env, check=True, capture_output=True)
        events = self.wait_for(lambda e: "killed" in e)
        # Speech the desk is still preparing is dropped too.
        self.prompt(LONG_PROMPT, stdout=None)  # carries the note about the remark above
        self.assertIn("chatter with me", self.last_stdout)
        subprocess.run([HAL, "stop"], env=self.env, check=True, capture_output=True)
        self.wait_until(lambda: "superseded" in self.log_text(), what="the desk to drop its speech")
        time.sleep(0.5)
        self.assertEqual(self.event_lines(), events)

    def test_muted_desk_says_nothing(self):
        self.set_config(desk_wait_seconds=2.5)
        self.start_desk()
        self.prompt(LONG_PROMPT)
        subprocess.run([HAL, "off"], env=self.env, check=True, capture_output=True)
        try:
            time.sleep(3.0)
            self.assertEqual(self.event_lines(), [])
        finally:
            subprocess.run([HAL, "on"], env=self.env, check=True, capture_output=True)

    # -- turn judgements for the listener -----------------------------------

    def judge(self, text, timeout=2.5):
        sys.path.insert(0, str(ROOT))
        from hal import desk
        return desk.judge_turn(text, timeout, state_dir=self.tmp / "state")

    def test_turn_judgements(self):
        self.start_desk()
        self.wait_until(lambda: (self.desk_status().get("turn_model") or {}).get("alive"), what="the turn model")
        done = self.judge("Claude, I think the conclusion of my paper is weak.")
        self.assertEqual(done["kind"], "complete", done)
        self.assertLess(done["seconds"], 1.5)
        self.assertEqual(self.judge("Another topic in mind. In fact there are several, but for the moment...")["kind"],
                         "incomplete")
        slow = self.judge("slow: is this finished?", timeout=0.8)
        self.assertEqual((slow["kind"], slow["why"]), ("unavailable", "the model was too slow"))
        time.sleep(5.5)   # let the slow answer arrive, so the model is back in step
        nonsense = self.judge("garbage in")
        self.assertEqual(nonsense["kind"], "unavailable")
        self.assertIn("neither form", nonsense["why"])
        self.assertEqual(self.judge("Claude, open my thesis.")["kind"], "complete")
        # Its own model and instructions; nothing spoken; the log has counts, never the words.
        prompt = (self.out / "warm_model_prompt.turn.txt").read_text()
        self.assertIn("INCOMPLETE", prompt)
        self.assertIn("Never answer or carry out", prompt)
        self.assertEqual(len(self.model_calls("turn")), 5)
        self.assertIn("<speech>Claude, open my thesis.</speech>", self.model_calls("turn")[-1])
        self.assertEqual(self.model_calls("desk"), [], "the front desk's own model never sees these")
        self.assertEqual(self.event_lines(), [])
        log = self.log_text()
        self.assertIn("desk: turn of 10 words judged complete", log)
        for words in ("conclusion", "Another topic", "thesis"):
            self.assertNotIn(words, log)
        out = subprocess.run([HAL, "desk", "status"], env=self.env, capture_output=True, text=True).stdout
        self.assertIn("turn model: pid", out)
        self.assertIn("turns judged", out)

    def test_turn_judgements_off(self):
        self.set_config(desk_turns=False)
        self.start_desk()
        self.wait_until(lambda: (self.desk_status().get("desk_model") or {}).get("alive"), what="the desk model")
        self.assertIsNone(self.desk_status()["turn_model"])
        answer = self.judge("Claude, I think the conclusion of my paper is weak.")
        self.assertEqual(answer["kind"], "unavailable")
        self.assertIn("desk_turns", answer["why"])
        out = subprocess.run([HAL, "desk", "turn", "Claude, open my thesis."], env=self.env, capture_output=True,
                             text=True).stdout
        self.assertIn("unavailable", out)

    # -- lifecycle ----------------------------------------------------------

    def test_only_one_desk_runs(self):
        self.start_desk()
        pid = self.desk_status()["pid"]
        proc = subprocess.run([HAL, "desk", "run"], env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("already running", proc.stderr)
        self.start_desk()  # says it is already running
        self.assertEqual(self.desk_status()["pid"], pid)

    def test_idle_desk_exits_with_its_models(self):
        self.set_config(desk_idle_minutes=0.05)  # 3 s
        self.start_desk()
        info = self.desk_status()
        pids = [info["pid"], info["desk_model"]["pid"], info["summary_model"]["pid"], info["turn_model"]["pid"]]
        self.wait_until(lambda: self.desk_status() is None, timeout=10, what="the desk to exit")
        self.wait_until(lambda: not any(alive(p) for p in pids), timeout=10, what="its models to exit")
        self.assertIn("idle for", self.log_text())
        self.assertFalse((self.tmp / "state" / "desk.sock").exists())


class DeskAnswerTest(unittest.TestCase):
    """The parsing of the model's answers (no processes)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT))
        from hal import desk
        cls.desk = desk

    def test_a_guest_is_not_greeted_as_the_user(self):
        w = self.desk._with_speaker
        self.assertEqual(w("Hello Claude.", None), "Hello Claude.")
        self.assertEqual(w("Hello Claude.", "Robin"),
                         "(Said by Robin, not by the user: if you use a name, use Robin.) Hello Claude.")
        for doubtful in ("Robin?", "the user or Robin?", "Robin and the user", "an unrecognised voice"):
            self.assertTrue(w("Hi.", doubtful).startswith("(The speaker may not be the user: address no one"))

    def test_parse(self):
        parse = self.desk.parse
        self.assertEqual(parse("REPLY: Good morning."), ("social", "Good morning."))
        self.assertEqual(parse("ACK: Certainly. I'll check the forecast."), ("ack", "Certainly. I'll check the forecast."))
        self.assertEqual(parse("  **ACK:** Very well."), ("ack", "Very well."))
        self.assertEqual(parse('reply - "Thank you."'), ("social", "Thank you."))
        self.assertEqual(parse("ACK: Very well.\nREPLY: something else"), ("ack", "Very well."))
        for nonsense in ("", "Good morning.", "ACKNOWLEDGED", "REPLY:", "HANDOFF", None, "ACK: " + "x" * 300):
            self.assertIsNone(parse(nonsense), nonsense)

    def test_brief_answers(self):
        from hal.hook import is_brief_answer
        for yes in ("OK.", "Okay, thanks.", "Sounds good.", "Hmm.", "Yes, go ahead.", "No.", "No, thank you.",
                    "The second one, please.", "Read that again.", "Why?", "Perfect.", "Please do."):
            self.assertTrue(is_brief_answer(yes), yes)
        for no in ("", "Thank you.", "Thanks, Claude.", "Good morning, Claude.", "What is the weather tomorrow?",
                   "Run the tests.", "Yes, and send the draft to Sarah as well.", "Tell me more."):
            self.assertFalse(is_brief_answer(no), no)

    def test_early_kind(self):
        early = self.desk.early_kind
        self.assertIsNone(early(""))
        self.assertIsNone(early("AC"))
        self.assertIsNone(early("**RE"))
        self.assertEqual(early("ACK"), "ack")
        self.assertEqual(early("REPLY: Hi"), "social")
        self.assertEqual(early("Good"), "nonsense")

    def test_turn_answers(self):
        self.assertEqual(self.desk.parse_turn("COMPLETE"), "complete")
        self.assertEqual(self.desk.parse_turn("**INCOMPLETE**\nThe speaker..."), "incomplete")
        self.assertIsNone(self.desk.parse_turn("The speaker has finished."))
        self.assertIsNone(self.desk.early_turn("CO"))
        self.assertIsNone(self.desk.early_turn("I"))
        self.assertEqual(self.desk.early_turn("COM"), "complete")
        self.assertEqual(self.desk.early_turn(" **INC"), "incomplete")
        self.assertEqual(self.desk.early_turn("The"), "nonsense")
        long = "word " * 1000
        self.assertTrue(self.desk.turn_message(long).startswith("<speech>… "))
        self.assertLess(len(self.desk.turn_message(long)), 2100)

    def test_socket_path_for_another_state_dir(self):
        self.assertEqual(self.desk.socket_path("/tmp/x"), "/tmp/x/desk.sock")
        long = "/tmp/" + "x" * 120
        self.assertTrue(self.desk.socket_path(long).startswith("/tmp/hal-desk-"))

    def test_prompt_uses_the_name_only_if_given(self):
        named = self.desk.desk_prompt("Dr. Alexander")
        self.assertIn("The user is Dr. Alexander. Use their name when greeting them or saying goodbye", named)
        self.assertIn("Do not address the user by name", self.desk.desk_prompt(""))
        self.assertNotIn("HAL", named, "the desk is Claude; the prompt never names HAL")


if __name__ == "__main__":
    unittest.main()
