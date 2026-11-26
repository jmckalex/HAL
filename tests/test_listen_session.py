"""Session state kept by the hooks, the keyboard model, and the verdict on typing."""

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal.listener import session, settings  # noqa: E402
from hal.listener.session import Keyboard, apply_event, verdict  # noqa: E402
from listen_harness import HAL_LISTEN, isolated_env, make_tmp, rmtree, write_config  # noqa: E402

CFG = dict(settings.DEFAULTS)


def ev(name, **fields):
    return dict(hook_event_name=name, session_id="s1", **fields)


def run(events, t0=1000.0):
    st = session.empty(1)
    for i, e in enumerate(events):
        st = apply_event(st, e, t0 + i)
    return st


BASH = {"command": "rm -rf build", "description": "Remove build"}
STARTED = [ev("SessionStart", source="startup")]


class ReducerTest(unittest.TestCase):
    def test_not_started(self):
        ok, reason, _ = verdict(None, None, 0, CFG)
        self.assertEqual((ok, reason), (False, "no-session"))
        ok, reason, _ = verdict(session.empty(), None, 0, CFG)
        self.assertEqual(reason, "no-session")

    def test_idle_and_busy_allow_typing(self):
        st = run(STARTED)
        self.assertEqual(st["turn"], "idle")
        self.assertTrue(verdict(st, None, 2000, CFG)[0])
        st = run(STARTED + [ev("UserPromptSubmit", prompt="hi")])
        self.assertEqual(st["turn"], "busy")
        self.assertTrue(verdict(st, None, 2000, CFG)[0])   # Claude Code queues what is typed
        st = run(STARTED + [ev("UserPromptSubmit", prompt="hi"), ev("Stop")])
        self.assertEqual(st["turn"], "idle")

    def test_permission_dialog_blocks_until_tool_finishes(self):
        st = run(STARTED + [ev("UserPromptSubmit", prompt="x"),
                            ev("PermissionRequest", tool_name="Bash", tool_input=BASH)])
        ok, reason, detail = verdict(st, None, 2000, CFG)
        self.assertEqual((ok, reason, detail), (False, "dialog", "permission: Bash"))
        st = apply_event(st, ev("PostToolUse", tool_name="Bash", tool_input=BASH, tool_response={}), 1010)
        self.assertTrue(verdict(st, None, 2000, CFG)[0])

    def test_unrelated_tool_does_not_close_dialog(self):
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("PostToolUse", tool_name="Read", tool_input={"file_path": "/x"})])
        self.assertEqual(verdict(st, None, 2000, CFG)[1], "dialog")
        # A parallel Bash call with different input closes only one dialog, the oldest.
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("PermissionRequest", tool_name="Bash", tool_input={"command": "ls"}),
                            ev("PostToolUse", tool_name="Bash", tool_input={"command": "ls"})])
        self.assertEqual(len(st["dialogs"]), 1)
        self.assertEqual(st["dialogs"][0]["key"], session.tool_key("Bash", BASH))

    def test_failure_denial_and_batch_close_dialogs(self):
        for closing in (ev("PostToolUseFailure", tool_name="Bash", tool_input=BASH, error="x"),
                        ev("PermissionDenied", tool_name="Bash", tool_input=BASH, reason="auto"),
                        ev("PostToolBatch", tool_calls=[{"tool_name": "Bash", "tool_input": BASH}])):
            st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH), closing])
            self.assertEqual(st["dialogs"], [], closing["hook_event_name"])

    def test_turn_boundaries_close_everything(self):
        opened = STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("PreToolUse", tool_name="AskUserQuestion", tool_input={"questions": []})]
        for closing in (ev("Stop"), ev("StopFailure"), ev("UserPromptSubmit", prompt="x"),
                        ev("Notification", notification_type="idle_prompt"),
                        ev("SessionStart", source="clear")):
            self.assertEqual(run(opened + [closing])["dialogs"], [], closing)

    def test_subagent_stop_does_not_close_main_dialog(self):
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("Stop", agent_id="a1")])
        self.assertEqual(len(st["dialogs"]), 1)

    def test_interactive_tools(self):
        st = run(STARTED + [ev("PreToolUse", tool_name="AskUserQuestion", tool_input={"q": 1})])
        self.assertEqual(verdict(st, None, 2000, CFG)[2], "question: AskUserQuestion")
        st = apply_event(st, ev("PostToolUse", tool_name="AskUserQuestion", tool_input={"q": 1}), 1005)
        self.assertTrue(verdict(st, None, 2000, CFG)[0])
        st = run(STARTED + [ev("PreToolUse", tool_name="ExitPlanMode", tool_input={"plan": "p"})])
        self.assertEqual(st["dialogs"][0]["kind"], "plan")
        # Ordinary PreToolUse events are not dialogs.
        self.assertEqual(run(STARTED + [ev("PreToolUse", tool_name="Read", tool_input={})])["dialogs"], [])

    def test_notifications(self):
        st = run(STARTED + [ev("Notification", notification_type="permission_prompt", message="m")])
        self.assertEqual(verdict(st, None, 2000, CFG)[1], "dialog")
        st = run(STARTED + [ev("Notification", notification_type="auth_success")])
        self.assertTrue(verdict(st, None, 2000, CFG)[0])

    def test_elicitation(self):
        st = run(STARTED + [ev("Elicitation", mcp_server_name="srv", elicitation_id="e1", message="m")])
        self.assertEqual(verdict(st, None, 2000, CFG)[1], "dialog")
        st = apply_event(st, ev("ElicitationResult", mcp_server_name="srv", elicitation_id="e1", action="accept"), 1009)
        self.assertTrue(verdict(st, None, 2000, CFG)[0])

    def test_session_end(self):
        st = run(STARTED + [ev("SessionEnd", reason="prompt_input_exit")])
        self.assertEqual(verdict(st, None, 2000, CFG)[1], "ended")
        st = run(STARTED + [ev("SessionEnd", reason="clear"), ev("SessionStart", source="clear")])
        self.assertTrue(verdict(st, None, 2000, CFG)[0])

    def test_stale_dialog_ignored(self):
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH)])
        self.assertTrue(verdict(st, None, 1001 + CFG["dialog_max_age"] + 1, CFG)[0])

    def test_malformed_input_is_harmless(self):
        st = apply_event({"dialogs": "nonsense", "turn": 3}, {"hook_event_name": None}, 1.0)
        self.assertEqual(st["dialogs"], [])
        st = apply_event(st, "not a dict", 2.0)
        self.assertIsInstance(st, dict)


class KeyboardTest(unittest.TestCase):
    def dialog_state(self, since=1000.0, **kw):
        return run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH, **kw)], t0=since - 1)

    def test_enter_answers_permission_dialog(self):
        st = self.dialog_state()
        kb = Keyboard()
        kb.feed(b"\r", 1000.1)            # too soon after it appeared: not an answer
        self.assertEqual(verdict(st, kb, 1010, CFG)[1], "dialog")
        kb.feed(b"\r", 1003.0)
        self.assertTrue(verdict(st, kb, 1010, CFG)[0])

    def test_one_enter_per_dialog(self):
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("PermissionRequest", tool_name="Write", tool_input={"file_path": "/a"})], t0=999)
        kb = Keyboard()
        kb.feed(b"\r", 1005.0)
        self.assertEqual(verdict(st, kb, 1010, CFG)[2], "permission: Write")
        kb.feed(b"\r", 1006.0)
        self.assertTrue(verdict(st, kb, 1010, CFG)[0])

    def test_escape_closes_all_dialogs(self):
        st = run(STARTED + [ev("PermissionRequest", tool_name="Bash", tool_input=BASH),
                            ev("PreToolUse", tool_name="AskUserQuestion", tool_input={})], t0=999)
        kb = Keyboard()
        kb.feed(b"\x1b", 1005.0)
        self.assertTrue(verdict(st, kb, 1010, CFG)[0])
        kb = Keyboard()
        kb.feed(b"\x03", 1005.0)          # Ctrl-C too
        self.assertTrue(verdict(st, kb, 1010, CFG)[0])

    def test_enter_does_not_close_question(self):
        st = run(STARTED + [ev("PreToolUse", tool_name="AskUserQuestion", tool_input={})], t0=999)
        kb = Keyboard()
        kb.feed(b"\r", 1005.0)
        self.assertEqual(verdict(st, kb, 1010, CFG)[1], "dialog")

    def test_arrow_keys_are_not_escape(self):
        st = self.dialog_state()
        kb = Keyboard()
        kb.feed(b"\x1b[B", 1005.0)        # down arrow
        self.assertEqual(verdict(st, kb, 1010, CFG)[1], "dialog")

    def test_draft(self):
        st = run(STARTED)
        kb = Keyboard()
        kb.feed(b"hello", 1000.0)
        self.assertEqual(verdict(st, kb, 1005, CFG)[1], "draft")
        kb.feed(b"\x7f" * 5, 1001.0)      # deleted again
        self.assertTrue(verdict(st, kb, 1005, CFG)[0])
        kb.feed(b"more", 1002.0)
        kb.feed(b"\r", 1003.0)            # sent
        self.assertTrue(verdict(st, kb, 1005, CFG)[0])
        kb.feed("café".encode(), 1004.0)
        self.assertEqual(kb.draft, 4)     # one per character, not per byte
        self.assertTrue(verdict(st, kb, 1004.0 + CFG["draft_hold_seconds"] + 1, CFG)[0])  # abandoned

    def test_stray_keys_do_not_hold(self):
        st = run(STARTED)
        for stray in (b"   ", b"j", b"jk", b"asdf", b"Cna yuo"):
            kb = Keyboard()
            kb.feed(stray, 1000.0)
            ok, reason, _ = verdict(st, kb, 1005, CFG)
            self.assertTrue(ok, (stray, reason))
            self.assertTrue(kb.negligible(CFG))
        for kept in (b"fix the docs", b"ok then", b"line one\x1b\rasdf"):
            kb = Keyboard()
            kb.feed(kept, 1000.0)
            self.assertEqual(verdict(st, kb, 1005, CFG)[1], "draft", kept)
        kb = Keyboard()
        kb.feed(b"\x1b[200~xq\x1b[201~", 1000.0)  # pasted: always kept
        self.assertEqual(verdict(st, kb, 1005, CFG)[1], "draft")

    def test_stray_rules_can_be_turned_off(self):
        st = run(STARTED)
        cfg = dict(CFG, draft_ignore_chars=0, draft_ignore_wordless=False)
        kb = Keyboard()
        kb.feed(b"  ", 1000.0)
        self.assertTrue(verdict(st, kb, 1005, cfg)[0])        # blank still never counts
        kb.feed(b"jk", 1000.0)
        self.assertEqual(verdict(st, kb, 1005, cfg)[1], "draft")

    def test_draft_detail_says_when_it_was_typed(self):
        kb = Keyboard()
        kb.feed(b"fix the docs", 1000.0)
        self.assertEqual(verdict(run(STARTED), kb, 1012, CFG)[2],
                         "about 12 unsent characters in the prompt box, last typed 12s ago")

    def test_word_delete(self):
        for key in (b"\x17", b"\x1b\x7f", b"\x1b[119;5u", b"\x1b[127;3u"):
            kb = Keyboard()
            kb.feed(b"fix the  ", 1.0)
            kb.feed(key, 2.0)
            self.assertEqual("".join(kb.text), "fix ", key)

    def test_paste_mark_ends_with_the_draft(self):
        kb = Keyboard()
        kb.feed(b"\x1b[200~ab\x1b[201~", 1.0)
        self.assertFalse(kb.negligible(CFG))
        kb.feed(b"\x7f\x7fj", 2.0)
        self.assertTrue(kb.negligible(CFG))

    def test_ctrl_c_and_ctrl_u_clear_draft(self):
        for key in (b"\x03", b"\x15"):
            kb = Keyboard()
            kb.feed(b"abc", 1.0)
            kb.feed(key, 2.0)
            self.assertEqual(kb.draft, 0)

    def test_bracketed_paste_counts_as_draft_not_enter(self):
        kb = Keyboard()
        kb.feed(b"\x1b[200~line one\rline two\x1b[201~", 1.0)
        self.assertGreater(kb.draft, 0)
        self.assertEqual(list(kb.enters), [])

    def test_kitty_keyboard_protocol(self):
        kb = Keyboard()
        kb.feed(b"x\x1b[13u", 1.0)        # Enter
        self.assertEqual((kb.draft, list(kb.enters)), (0, [1.0]))
        kb.feed(b"\x1b[27u", 2.0)         # Esc
        self.assertEqual(list(kb.escapes), [2.0])
        kb.feed(b"ab\x1b[13;2u", 3.0)     # Shift+Enter: a newline in the box, not a send
        self.assertEqual(len(kb.enters), 1)
        self.assertEqual(kb.draft, 3)

    def test_keyboard_quiet(self):
        st = run(STARTED)
        kb = Keyboard()
        kb.feed(b"\x1b[A", 1000.0)        # up arrow: no draft, but someone is at the keys
        self.assertEqual(verdict(st, kb, 1000.5, CFG)[1], "keyboard")
        self.assertTrue(verdict(st, kb, 1000.0 + CFG["keyboard_quiet"] + 0.01, CFG)[0])


class HookCommandTest(unittest.TestCase):
    """`bin/hal-listen hook` as Claude Code runs it."""

    def setUp(self):
        self.tmp = make_tmp()
        write_config(self.tmp)
        self.env = isolated_env(self.tmp)

    def tearDown(self):
        rmtree(self.tmp)

    def fire(self, payload, proxy="4242", data=None):
        env = dict(self.env)
        if proxy:
            env["HAL_LISTEN_PROXY"] = proxy
        started = time.monotonic()
        p = subprocess.run([str(HAL_LISTEN), "hook"], input=data if data is not None else json.dumps(payload).encode(),
                           env=env, capture_output=True, timeout=10)
        return p, time.monotonic() - started

    def state(self, proxy=4242):
        path = self.tmp / "state" / "listen" / f"session-{proxy}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def test_updates_state_silently_and_fast(self):
        p, _ = self.fire(ev("SessionStart", source="startup"))
        self.assertEqual((p.returncode, p.stdout, p.stderr), (0, b"", b""))
        times = []
        for e in (ev("UserPromptSubmit", prompt="secret prompt text"),
                  ev("PermissionRequest", tool_name="Bash", tool_input=BASH)):
            p, seconds = self.fire(e)
            times.append(seconds)
            self.assertEqual((p.returncode, p.stdout), (0, b""))
        st = self.state()
        self.assertEqual((st["started"], st["turn"], len(st["dialogs"])), (True, "busy", 1))
        self.assertLess(max(times), 1.0, times)
        log = (self.tmp / "state" / "listen.log").read_text()
        self.assertIn("PermissionRequest", log)
        self.assertNotIn("secret prompt text", log)
        self.assertNotIn("rm -rf", log)

    def test_does_nothing_outside_a_proxied_session(self):
        p, _ = self.fire(ev("SessionStart"), proxy=None)
        self.assertEqual((p.returncode, p.stdout), (0, b""))
        self.assertFalse((self.tmp / "state" / "listen").exists())

    def test_bad_input_never_fails(self):
        for data in (b"", b"not json", b"[1,2]", b"\xff\xfe"):
            p, _ = self.fire(None, data=data)
            self.assertEqual((p.returncode, p.stdout), (0, b""))


if __name__ == "__main__":
    unittest.main()
