"""Tests of speaking while Claude works (hal/progress.py): the PreToolUse hook.

End-to-end like tests/test_hook.py: hook JSON is piped through the same
command string as .claude/settings.json uses for `bin/hal hook`, speech is
really synthesised and "played" by the silent tests/fake_player.py, and state
goes to a temporary directory. A small transcript file stands in for Claude
Code's.

Run: .venv/bin/python -m unittest tests.test_progress -v      (takes ~40 s)
"""

import json
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_hook import SETTINGS, HookTestBase  # noqa: E402

from hal import progress  # noqa: E402


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ProgressTestBase(HookTestBase):
    def setUp(self):
        super().setUp()
        self.transcript = self.tmp / "transcript.jsonl"
        self.transcript.write_text("")
        self.n = 0

    def configure(self, **overrides):
        settings = {"progress_cue_seconds": 0}
        settings.update(overrides)
        self.set_config(**settings)

    def write_text(self, text, when=None, **extra):
        self.n += 1
        entry = {"type": "assistant", "uuid": f"u{self.n}", "timestamp": _iso(when or time.time()),
                 "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}, **extra}
        with open(self.transcript, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def tool(self, **extra):
        payload = {"session_id": "test-session", "transcript_path": str(self.transcript), "cwd": str(ROOT),
                   "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "true"},
                   **extra}
        # The same command as every other `bin/hal hook` registration.
        return self.run_hook(payload, event="Stop")

    def prompt(self, text="what is the weather tomorrow"):
        self.run_hook(self.prompt_payload(text), event="UserPromptSubmit")

    def starts(self):
        return [e for e in self.event_lines() if e.startswith("start")]

    def seconds(self, start_line):
        return float(start_line.split(" ")[2])


class ProgressTest(ProgressTestBase):
    def test_registered_async_for_every_tool(self):
        handlers = [(group.get("matcher"), h) for group in SETTINGS["hooks"].get("PreToolUse", [])
                    for h in group["hooks"] if h["command"].rstrip().endswith("bin/hal hook")]
        self.assertEqual(len(handlers), 1, "bin/hal hook must be registered once for PreToolUse")
        matcher, handler = handlers[0]
        self.assertIn(matcher, (None, "", "*"))
        self.assertIs(handler.get("async"), True, "the PreToolUse hook must never delay a tool call")

    def test_line_before_a_tool_is_spoken_once(self):
        self.configure()
        self.prompt()
        self.write_text("Let me fetch the forecast for **London**.")
        elapsed = self.tool()
        self.assertLess(elapsed, 0.6, "the hook must return quickly")
        self.wait_for(lambda ev: "end" in ev)
        self.tool()  # a second tool call in the same message: nothing new to say
        time.sleep(1.5)
        self.assertEqual(len(self.starts()), 1, self.event_lines())
        self.assertGreater(self.seconds(self.starts()[0]), 1.0)
        self.assertIn("PreToolUse: speaking Claude's line", self.log_text())

    def test_nothing_without_a_spoken_turn(self):
        self.configure(progress_cue_seconds=0.1)
        self.write_text("Let me check.")
        self.tool()
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])
        self.assertNotIn("PreToolUse", self.log_text(), "nothing to note, so nothing logged")

    def test_text_from_before_the_prompt_is_not_spoken(self):
        self.configure()
        self.write_text("An old line from the previous turn.", when=time.time() - 60)
        self.prompt()
        self.tool()
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])

    def test_subagent_tool_calls_are_ignored(self):
        self.configure()
        self.prompt()
        self.write_text("Let me check.")
        self.tool(agent_id="agent-1", agent_type="Explore")
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])

    def test_sidechain_and_unsuitable_text_are_not_spoken(self):
        self.configure()
        self.prompt()
        self.write_text("A subagent's words.", isSidechain=True)
        self.write_text("- a list item\n- another")
        self.tool()
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])
        self.assertIn("not short prose", self.log_text())

    def test_off_switch(self):
        self.configure(speak_progress=False)
        self.prompt()
        self.write_text("Let me check.")
        self.tool()
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])

    def test_stock_phrase_after_silence_at_most_twice(self):
        self.configure(progress_cue_seconds=0.5)
        self.prompt()
        for _ in range(3):
            time.sleep(0.7)
            self.tool()
            self.wait_for(lambda ev: ev.count("end") == len([e for e in ev if e.startswith("start")]))
            time.sleep(0.7)  # the hook waits briefly for text before deciding
        self.assertEqual(len(self.starts()), 2, self.event_lines())
        self.assertEqual(self.log_text().count("speaking a progress cue"), 2)

    def test_reply_waits_for_a_progress_line(self):
        self.configure()
        self.prompt()
        self.write_text("Let me run the whole test suite, which takes a little while.")
        self.tool()
        self.wait_for(lambda ev: any(e.startswith("start") for e in ev))
        self.run_hook(self.stop_payload("All tests pass."))
        lines = self.wait_for(lambda ev: sum(e.startswith("start") for e in ev) == 2)
        second = [i for i, e in enumerate(lines) if e.startswith("start")][1]
        self.assertEqual(lines[second - 1], "end", f"the reply must start after the line ends: {lines}")
        self.assertNotIn("killed", lines)
        self.assertFalse((self.tmp / "state" / "turn.json").exists(), "Stop ends the turn")

    def test_hal_stop_silences_and_ends_progress_speech(self):
        self.configure()
        self.prompt()
        self.write_text("Let me run the whole test suite, which takes a little while.")
        self.tool()
        self.wait_for(lambda ev: any(e.startswith("start") for e in ev))
        subprocess.run([str(ROOT / "bin" / "hal"), "stop"], env=self.env, check=True, capture_output=True)
        self.wait_for(lambda ev: "killed" in ev)
        self.write_text("Now checking the second module.")
        self.tool()
        time.sleep(2.5)
        self.assertEqual(len(self.starts()), 1, self.event_lines())

    def test_newer_line_replaces_one_still_waiting(self):
        self.configure()
        self.prompt()
        self.write_text("Let me run the whole test suite, which takes a little while to finish.")
        self.tool()
        self.wait_for(lambda ev: any(e.startswith("start") for e in ev))
        self.write_text("Now the second module.")
        self.tool()
        time.sleep(0.3)
        self.write_text("Now the third module.")
        self.tool()
        lines = self.wait_for(lambda ev: sum(e.startswith("start") for e in ev) == 2 and ev[-1] == "end",
                              timeout=20)
        time.sleep(1.5)
        self.assertEqual(len(self.starts()), 2, self.event_lines())
        self.assertNotIn("killed", lines)
        self.assertIn("progress line superseded", self.log_text())

    def test_muted_and_other_sessions_say_nothing(self):
        self.configure(progress_cue_seconds=0.1)
        self.prompt()
        self.write_text("Let me check.")
        other = {k: v for k, v in self.env.items() if k != "HAL_VOICE"}
        payload = {"session_id": "s", "transcript_path": str(self.transcript), "hook_event_name": "PreToolUse",
                   "tool_name": "Bash", "tool_input": {}}
        self.run_hook(payload, event="Stop", env=other)
        subprocess.run([str(ROOT / "bin" / "hal"), "off"], env=self.env, check=True, capture_output=True)
        try:
            self.tool()
        finally:
            subprocess.run([str(ROOT / "bin" / "hal"), "on"], env=self.env, check=True, capture_output=True)
        time.sleep(2.0)
        self.assertEqual(self.event_lines(), [])

    def test_stop_requests_and_commands_start_no_turn(self):
        self.configure()
        self.prompt("please stop talking")
        self.assertFalse((self.tmp / "state" / "turn.json").exists())
        self.prompt("/compact")
        self.assertFalse((self.tmp / "state" / "turn.json").exists())
        self.prompt("check the weather")
        self.assertTrue((self.tmp / "state" / "turn.json").exists())


class SpeechForTest(unittest.TestCase):
    def test_short_prose_is_kept_in_whole_sentences(self):
        self.assertEqual(progress.speech_for("Let me fetch the forecast for **London**.", 160),
                         "Let me fetch the forecast for London.")
        text = ("Baseline is clean: all 453 tests pass. Now I am checking whether the line I write before a "
                "tool call reaches the transcript in time, which the whole design depends on.")
        self.assertEqual(progress.speech_for(text, 160), "Baseline is clean: all 453 tests pass.")

    def test_unsuitable_text_gives_nothing(self):
        self.assertEqual(progress.speech_for("- a list item", 160), "")
        self.assertEqual(progress.speech_for("## Heading\n\nText.", 160), "")
        self.assertEqual(progress.speech_for("word " * 50 + "end.", 160), "", "one long sentence is not cut")

    def test_paths_and_code_are_made_speakable(self):
        self.assertEqual(progress.speech_for("Checking `/Users/x/Source/HAL/hal/speaker.py` now.", 160),
                         "Checking speaker.py now.")
        self.assertEqual(progress.speech_for("I have what I need.\n\n```py\nx = 1\n```", 160),
                         "I have what I need.")


class TextsSinceTest(unittest.TestCase):
    def test_reads_only_recent_main_conversation_text(self):
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="hal-test-")) / "t.jsonl"
        now = time.time()
        rows = [
            {"type": "assistant", "uuid": "old", "timestamp": _iso(now - 100),
             "message": {"content": [{"type": "text", "text": "old"}]}},
            {"type": "user", "uuid": "u", "timestamp": _iso(now), "message": {"content": "a text prompt"}},
            {"type": "assistant", "uuid": "think", "timestamp": _iso(now),
             "message": {"content": [{"type": "thinking", "thinking": "private", "text": "x"}]}},
            {"type": "assistant", "uuid": "side", "timestamp": _iso(now), "isSidechain": True,
             "message": {"content": [{"type": "text", "text": "side"}]}},
            {"type": "assistant", "uuid": "new", "timestamp": _iso(now),
             "message": {"content": [{"type": "text", "text": "new"}, {"type": "tool_use", "name": "Bash"}]}},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{not json\n")
        self.assertEqual(progress.texts_since(path, now - 1), [("new:0", "new")])
        self.assertEqual(progress.texts_since(path.with_name("missing.jsonl"), 0), [])


if __name__ == "__main__":
    unittest.main()
