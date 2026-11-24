"""End-to-end tests of the hook, run exactly as .claude/settings.json runs it.

Each test pipes hook JSON into the command string from .claude/settings.json
via `sh -c`, with CLAUDE_PROJECT_DIR set, the way Claude Code does. Speech is
really synthesised with the HAL model, but "played" by tests/fake_player.py,
which is silent: it records what it was given and sleeps for the duration.
State goes to a temporary directory (HAL_STATE_DIR), not ./state.

Run: .venv/bin/python -m unittest tests.test_hook -v      (takes ~30 s)
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / ".claude" / "settings.json").read_text())
sys.path.insert(0, str(ROOT / "tests"))

from test_textclean import REALISTIC_REPLY as REPLY  # noqa: E402


def hook_command(event: str) -> str:
    # An event can carry several hook groups (the listener adds its own state
    # hook, bin/hal-listen); the voice hook is the one that runs `bin/hal hook`.
    (handler,) = [h for group in SETTINGS["hooks"][event] for h in group["hooks"]
                  if h["command"].rstrip().endswith("bin/hal hook")]
    return handler["command"]


LONG_REPLY = "## Report\n\n" + "\n".join(
    f"- Step {i}: I checked the module, ran the suite and confirmed the result is stable." for i in range(1, 30)
)


class HookTestBase(unittest.TestCase):
    """Temporary state and config, the silent player and the stand-in summariser."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hal-test-"))
        self.out = self.tmp / "out"
        self.events = self.out / "events.log"
        self.config = self.tmp / "config.json"
        self.set_config()  # the original behaviour: read replies in full, no acknowledgement
        self.env = dict(
            os.environ,
            CLAUDE_PROJECT_DIR=str(ROOT),
            CLAUDE_CODE_ENTRYPOINT="cli",
            HAL_VOICE="1",
            HAL_STATE_DIR=str(self.tmp / "state"),
            HAL_CONFIG=str(self.config),
            HAL_SUMMARIZER=f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tests' / 'fake_summarizer.py'))}",
            HAL_FAKE_OUT=str(self.out),
            HAL_PLAYER=f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tests' / 'fake_player.py'))}",
        )

    def tearDown(self):
        subprocess.run([str(ROOT / "bin" / "hal"), "stop"], env=self.env, capture_output=True)
        time.sleep(0.2)

    # -- helpers ------------------------------------------------------------

    def set_config(self, **overrides):
        settings = {"voice_sessions": "master", "spoken_reply": "full", "acknowledge": False, "max_chars": 600}
        settings.update(overrides)
        self.config.write_text(json.dumps(settings))

    def run_hook(self, payload, event="Stop", env=None, stdout=b""):
        stdin = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        if isinstance(stdin, str):
            stdin = stdin.encode()
        started = time.monotonic()
        proc = subprocess.run(
            ["/bin/sh", "-c", hook_command(event)],
            input=stdin, capture_output=True, env=env or self.env, timeout=10,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(proc.returncode, 0)
        if stdout is not None:
            self.assertEqual(proc.stdout, stdout, "unexpected hook stdout")
        self.assertEqual(proc.stderr, b"", "hook must not print to stderr")
        self.last_stdout = proc.stdout.decode()
        return elapsed

    def stop_payload(self, message, **extra):
        return {
            "session_id": "test-session",
            "transcript_path": "/tmp/none.jsonl",
            "cwd": str(ROOT),
            "permission_mode": "default",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": message,
            "background_tasks": [],
            "session_crons": [],
            **extra,
        }

    def prompt_payload(self, prompt="next question"):
        return {"session_id": "test-session", "cwd": str(ROOT), "hook_event_name": "UserPromptSubmit",
                "prompt": prompt}

    def summarizer_calls(self):
        try:
            return (self.out / "summarizer.log").read_text().splitlines()
        except FileNotFoundError:
            return []

    def event_lines(self):
        try:
            return [line.split(" ", 2)[2] for line in self.events.read_text().splitlines()]
        except FileNotFoundError:
            return []

    def wait_for(self, predicate, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.event_lines()):
                return self.event_lines()
            time.sleep(0.05)
        self.fail(f"timed out; events so far: {self.event_lines()}")

    def log_text(self):
        try:
            return (self.tmp / "state" / "hal.log").read_text()
        except FileNotFoundError:
            return ""


class HookTest(HookTestBase):
    def test_stop_hook_is_fast_and_produces_audio(self):
        elapsed = self.run_hook(self.stop_payload(REPLY))
        self.assertLess(elapsed, 0.5, "hook must return quickly")
        lines = self.wait_for(lambda ev: any(e.startswith("start") for e in ev))
        _, wav_copy, seconds = lines[0].split(" ")
        with wave.open(wav_copy) as w:
            self.assertEqual(w.getframerate(), 22050)
            self.assertEqual(w.getnchannels(), 1)
            self.assertGreater(w.getnframes() / w.getframerate(), 5.0)
        self.assertIn("Stop: speaking", self.log_text())

    def test_second_reply_cuts_off_the_first(self):
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.wait_for(lambda ev: ev.count("killed") == 0 and any(e.startswith("start") for e in ev))
        self.run_hook(self.stop_payload("Second reply. Short and sweet."))
        lines = self.wait_for(lambda ev: sum(e.startswith("start") for e in ev) == 2)
        starts = [i for i, e in enumerate(lines) if e.startswith("start")]
        self.assertEqual(starts[0], 0)
        self.assertIn("killed", lines[starts[0]:starts[1]],
                      f"first utterance should be killed before the second starts: {lines}")
        self.assertNotIn("end", lines[:starts[1]])

    def test_user_prompt_submit_silences(self):
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.wait_for(lambda ev: any(e.startswith("start") for e in ev))
        elapsed = self.run_hook(self.prompt_payload(), event="UserPromptSubmit")
        self.assertLess(elapsed, 0.5)
        self.wait_for(lambda ev: "killed" in ev)
        time.sleep(1.0)
        self.assertNotIn("end", self.event_lines())

    def test_prompt_during_synthesis_prevents_playback(self):
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit")  # before synthesis finishes
        time.sleep(3.0)
        self.assertEqual(self.event_lines(), [], "nothing may play after the user started a new turn")

    def test_muted_does_nothing(self):
        subprocess.run([str(ROOT / "bin" / "hal"), "off"], env=self.env, check=True, capture_output=True)
        try:
            self.run_hook(self.stop_payload(REPLY))
            time.sleep(2.5)
            self.assertEqual(self.event_lines(), [])
            self.assertIn("Stop: muted", self.log_text())
        finally:
            subprocess.run([str(ROOT / "bin" / "hal"), "on"], env=self.env, check=True, capture_output=True)

    def test_non_voice_session_neither_speaks_nor_silences(self):
        other = {k: v for k, v in self.env.items() if k != "HAL_VOICE"}
        self.run_hook(self.stop_payload(REPLY), env=other)
        time.sleep(2.5)
        self.assertEqual(self.event_lines(), [])
        self.assertIn("Stop: not the voice session", self.log_text())
        # The master session speaks; a prompt in another session leaves it alone.
        self.run_hook(self.stop_payload(LONG_REPLY))
        self.wait_for(lambda events: len(events) >= 1)
        before = self.event_lines()
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit", env=other)
        time.sleep(1.0)
        self.assertEqual(self.event_lines(), before)
        self.assertIn("UserPromptSubmit: not the voice session", self.log_text())

    # -- spoken summaries ---------------------------------------------------

    def test_long_reply_is_summarised_and_claude_is_told(self):
        self.set_config(spoken_reply="summary", summary_min_chars=350, acknowledge=False)
        elapsed = self.run_hook(self.stop_payload(LONG_REPLY))
        self.assertLess(elapsed, 1.0)
        events = self.wait_for(lambda e: any(x.startswith("start") for x in e))
        seconds = float(events[0].split()[-1])
        self.assertLess(seconds, 12, "the summary, not the long reply, should be spoken")
        (call,) = self.summarizer_calls()
        self.assertTrue(call.endswith("HAL_VOICE="), "the summariser must not inherit the voice")
        self.assertIn("summary of", self.log_text())
        # The next prompt tells Claude what was said, once.
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit", stdout=None)
        self.assertIn("spoken summary", self.last_stdout)
        self.assertIn("All is well, and Hal has one question for you.", self.last_stdout)
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit")  # nothing printed the second time

    def test_lead_mode_speaks_the_opening_paragraph_without_the_model(self):
        self.set_config(spoken_reply="lead", summary_min_chars=350, lead_max_chars=450, acknowledge=False)
        reply = "The tests pass and the build is ready.\n\n" + LONG_REPLY
        self.run_hook(self.stop_payload(reply))
        events = self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertLess(float(events[0].split()[-1]), 6, "only the opening paragraph should be spoken")
        self.assertEqual(self.summarizer_calls(), [])
        self.assertIn("speaking the opening paragraph", self.log_text())
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit")  # Claude wrote it: nothing to report

    def test_lead_mode_falls_back_to_a_summary_when_the_opening_is_a_heading(self):
        self.set_config(spoken_reply="lead", summary_min_chars=350, acknowledge=False)
        self.run_hook(self.stop_payload(LONG_REPLY))  # starts with "## Report"
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertEqual(len(self.summarizer_calls()), 1)

    def test_short_reply_is_read_in_full_without_the_model(self):
        self.set_config(spoken_reply="summary", summary_min_chars=350, acknowledge=False)
        self.run_hook(self.stop_payload("The tests pass. Shall I commit the change?"))
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertEqual(self.summarizer_calls(), [])
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit")  # read in full: nothing to tell Claude

    def test_failed_summary_falls_back_to_the_opening(self):
        self.set_config(spoken_reply="summary", summary_min_chars=200, acknowledge=False)
        self.run_hook(self.stop_payload(LONG_REPLY), env=dict(self.env, HAL_FAKE_SUMMARY_FAIL="1"))
        events = self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.assertLess(float(events[0].split()[-1]), 25)
        self.assertIn("summary failed", self.log_text())
        self.run_hook(self.prompt_payload(), event="UserPromptSubmit", stdout=None)
        self.assertIn("Only the opening", self.last_stdout)

    # -- acknowledgements and stop requests ---------------------------------

    def test_prompt_is_acknowledged_and_a_fast_reply_waits_for_it(self):
        self.set_config(spoken_reply="summary", acknowledge=True, ack_phrases=["One moment, please."])
        elapsed = self.run_hook(self.prompt_payload("What is the weather tomorrow?"), event="UserPromptSubmit")
        self.assertLess(elapsed, 1.0)
        self.wait_for(lambda e: len(e) >= 1 and e[0].startswith("start"))
        self.run_hook(self.stop_payload("It will rain."))  # arrives while the acknowledgement plays
        events = self.wait_for(lambda e: sum(x.startswith("start") for x in e) >= 2)
        self.assertEqual([x.split()[0] for x in events[:3]], ["start", "end", "start"],
                         "the acknowledgement must finish, not be cut off, before the reply starts")
        self.assertIn("acknowledged", self.log_text())

    def test_stop_request_silences_without_acknowledgement_or_spoken_reply(self):
        self.set_config(spoken_reply="summary", acknowledge=True)
        self.run_hook(self.stop_payload(LONG_REPLY), env=dict(self.env, HAL_FAKE_SUMMARY_FAIL="1"))
        self.wait_for(lambda e: any(x.startswith("start") for x in e))
        self.run_hook(self.prompt_payload("You can stop reading, please."), event="UserPromptSubmit")
        events = self.wait_for(lambda e: "killed" in e)
        self.run_hook(self.stop_payload("Stopped."))
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), events, "neither an acknowledgement nor the reply may be spoken")
        self.assertIn("stop request", self.log_text())
        self.assertIn("Stop: silent", self.log_text())
        # The reply after the next ordinary prompt is spoken again.
        self.set_config(spoken_reply="summary", acknowledge=False)
        self.run_hook(self.prompt_payload("What time is it?"), event="UserPromptSubmit", stdout=None)
        self.run_hook(self.stop_payload("It is noon."))
        self.wait_for(lambda e: len(e) > len(events))

    def test_turns_not_started_by_the_user_are_left_alone(self):
        self.set_config(spoken_reply="summary", acknowledge=True)
        self.run_hook(self.stop_payload(LONG_REPLY), env=dict(self.env, HAL_FAKE_SUMMARY_FAIL="1"))
        events = self.wait_for(lambda e: any(x.startswith("start") for x in e))
        for prompt in ("<task-notification>done</task-notification>",
                       "Another Claude session sent a message:\n<cross-session-message>hi</cross-session-message>",
                       "[SYSTEM NOTIFICATION - NOT USER INPUT]", ""):
            self.run_hook(self.prompt_payload(prompt), event="UserPromptSubmit")
        time.sleep(1.0)
        self.assertEqual(self.event_lines(), events, "HAL must keep speaking and not acknowledge")

    def test_stop_phrase_detection(self):
        sys.path.insert(0, str(ROOT))
        from hal.hook import is_stop_request
        for yes in ("Please stop reading.", "You can stop reading, please.", "Stop", "That's enough, thank you, HAL.",
                    "Quiet please", "OK HAL, stop talking now", "Thank you, Claude, that's enough."):
            self.assertTrue(is_stop_request(yes), yes)
        for no in ("Stop the server", "Please stop reading the file and summarise it instead",
                   "Why did the build stop?", "Thank you", "", "Is that enough memory?"):
            self.assertFalse(is_stop_request(no), no)

    def test_thanks_and_greetings_get_no_canned_acknowledgement(self):
        sys.path.insert(0, str(ROOT))
        from hal.hook import is_social_remark
        for yes in ("You did perfectly. Thank you very much.", "No, this is perfect. Thank you very much.",
                    "Thank you very much. That is very useful. You are an excellent assistant and a good friend. Thank you.",
                    "Good morning, Claude.", "Well done.", "You have all done very good work. Thank you. This is excellent."):
            self.assertTrue(is_social_remark(yes), yes)
        for no in ("Thanks. Now please commit that.", "Hello, please explain to me where we are.",
                   "Thank you, Claude. Could you please also refer to me as Dr. Alexander?",
                   "Great, what is the weather tomorrow?", "Open the Axelrod article.", "Yes, go ahead."):
            self.assertFalse(is_social_remark(no), no)
        self.set_config(spoken_reply="summary", acknowledge=True)
        self.run_hook(self.prompt_payload("You did perfectly. Thank you very much."), event="UserPromptSubmit")
        time.sleep(1.5)
        self.assertEqual(self.event_lines(), [])
        self.assertIn("no acknowledgement", self.log_text())

    def test_bad_input_exits_zero_silently(self):
        for stdin in (b"", b"   \n", b"not json", b"{\"truncated\": ", b"[1, 2]", b"null", b"\xff\xfe\x00",
                      json.dumps({"hook_event_name": "Stop"}).encode(),
                      json.dumps({"hook_event_name": "Stop", "last_assistant_message": None}).encode(),
                      json.dumps({"hook_event_name": "Stop", "last_assistant_message": 42}).encode(),
                      json.dumps({"hook_event_name": "Stop", "last_assistant_message": "```\ncode only\n```"}).encode(),
                      json.dumps({"hook_event_name": "SessionStart"}).encode()):
            with self.subTest(stdin=stdin[:30]):
                self.assertLess(self.run_hook(stdin), 0.5)
        time.sleep(1.5)
        self.assertEqual(self.event_lines(), [])
        self.assertNotIn("Traceback", self.log_text())

    def test_subagent_and_print_mode_are_silent(self):
        self.run_hook(self.stop_payload("From a teammate.", agent_id="abc", agent_type="Explore"))
        self.run_hook(self.stop_payload("From claude -p."), env=dict(self.env, CLAUDE_CODE_ENTRYPOINT="sdk-cli"))
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])

    def test_missing_venv_exits_zero_quietly(self):
        # A copy of bin/hal in a tree with no .venv must still exit 0 and print nothing.
        fake_root = self.tmp / "noproject"
        (fake_root / "bin").mkdir(parents=True)
        (fake_root / "bin" / "hal").write_bytes((ROOT / "bin" / "hal").read_bytes())
        (fake_root / "bin" / "hal").chmod(0o755)
        env = dict(self.env, CLAUDE_PROJECT_DIR=str(fake_root))
        self.run_hook(self.stop_payload("Hello."), env=env)
        self.assertIn("no virtualenv", self.log_text())


if __name__ == "__main__":
    unittest.main()
