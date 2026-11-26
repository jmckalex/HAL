"""The pty proxy, driven from a scripted terminal: transparency, job control, exit status, injection."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from listen_harness import (Terminal, hook, isolated_env, make_tmp, proxy_argv, proxy_info, rmtree,  # noqa: E402
                            standin_argv, write_config)

from hal.listener import control  # noqa: E402

FAST = {"autostart": False, "keyboard_quiet": 0.2, "submit_delay": 0.15}


class ProxyCase(unittest.TestCase):
    listener_cfg = FAST

    def setUp(self):
        self.tmp = make_tmp()
        write_config(self.tmp, self.listener_cfg)
        self.env = isolated_env(self.tmp, STANDIN_HEX="1", STANDIN_OUT=self.tmp / "lines.jsonl",
                                BASH_SILENCE_DEPRECATION_WARNING="1", PS1="$ ")
        self.terms = []

    def tearDown(self):
        for t in self.terms:
            t.close()
        rmtree(self.tmp)

    def start(self, argv, **kw):
        t = Terminal(argv, env=self.env, **kw)
        self.terms.append(t)
        return t

    def client(self):
        info = proxy_info(self.tmp / "state")
        return control.ProxyClient(info["socket"]), info

    def session_started(self, info):
        hook(self.env, {"hook_event_name": "SessionStart", "source": "startup", "session_id": "t"}, info["pid"])

    def lines(self):
        path = self.tmp / "lines.jsonl"
        return [json.loads(l)["line"] for l in path.read_text().splitlines()] if path.exists() else []


class StandinTest(ProxyCase):
    def test_keystrokes_and_output_pass_through_unchanged(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        keys = "héllo \x1b[A\x1b[1;5C\t\x03\x7f".encode()
        t.send(keys)
        t.wait_for("KEYS " + keys.hex())
        t.send("\rexit 0\r")
        t.wait_exit()

    def test_window_size_initial_and_resized(self):
        t = self.start(proxy_argv(*standin_argv()), rows=33, cols=101)
        t.wait_for(r"READY rows=33 cols=101")
        t.resize(50, 132)
        t.wait_for(r"WINCH rows=50 cols=132")
        t.resize(20, 70)
        t.wait_for(r"WINCH rows=20 cols=70")
        t.send("exit 0\r")
        t.wait_exit()

    def test_exit_status(self):
        for code in (0, 3, 255):
            t = self.start(proxy_argv(*standin_argv()))
            t.wait_for(r"READY")
            t.send(f"exit {code}\r")
            self.assertEqual(os.waitstatus_to_exitcode(t.wait_exit()), code)

    def test_killed_by_signal(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        t.send(f"die {int(signal.SIGTERM)}\r")
        status = t.wait_exit()
        self.assertTrue(os.WIFSIGNALED(status))
        self.assertEqual(os.WTERMSIG(status), signal.SIGTERM)

    def test_signals_to_the_proxy_reach_the_program(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        os.kill(t.pid, signal.SIGTERM)
        status = t.wait_exit()
        self.assertEqual(os.WTERMSIG(status), signal.SIGTERM)

    def test_terminal_hangup_ends_the_program(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        os.kill(t.pid, signal.SIGHUP)
        self.assertTrue(os.WIFSIGNALED(t.wait_exit()))

    def test_ctrl_z_suspends_and_fg_resumes(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        for how in ("tstp", "stop"):     # SIGTSTP as Node sends it on Ctrl-Z, and SIGSTOP
            before = t.count("CONT")
            t.send(how + "\r")
            self.assertTrue(t.wait_stopped(), f"proxy did not stop after {how}")
            os.kill(t.pid, signal.SIGCONT)
            deadline = time.monotonic() + 5
            while t.count("CONT") == before and time.monotonic() < deadline:
                t.pump(0.02)
            self.assertGreater(t.count("CONT"), before, f"program not continued after {how}")
        t.send("exit 4\r")
        self.assertEqual(os.waitstatus_to_exitcode(t.wait_exit()), 4)

    def test_cannot_exec(self):
        t = self.start(proxy_argv("/nonexistent/program"))
        t.wait_for(r"cannot run /nonexistent/program")
        self.assertEqual(os.waitstatus_to_exitcode(t.wait_exit()), 127)

    def test_files_cleaned_up(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        _, info = self.client()
        self.session_started(info)
        listen = self.tmp / "state" / "listen"
        self.assertTrue((listen / f"session-{info['pid']}.json").exists())
        t.send("exit 0\r")
        t.wait_exit()
        self.assertEqual(sorted(p.name for p in listen.iterdir()), ["proxy.lock"])
        self.assertFalse(Path(info["socket"]).exists())


class InjectionTest(ProxyCase):
    def test_typed_once_with_enter_after_the_text(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        reply = c.type("Claude, what's the weather tomorrow?", "id-1")
        self.assertEqual(reply, {"ok": True, "typed": 36, "submitted": True})
        self.assertTrue(c.type("Claude, what's the weather tomorrow?", "id-1")["duplicate"])
        t.wait_for(r'LINE "Claude, what\'s the weather tomorrow\?"')
        time.sleep(0.3)
        self.assertEqual(self.lines(), ["Claude, what's the weather tomorrow?"])
        # The text and Enter arrive as separate writes (Claude Code takes a burst with Enter as a paste).
        self.assertIn("KEYS 0d", t.text)
        t.send("exit 0\r")
        t.wait_exit()

    def test_control_characters_are_never_typed(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        self.assertTrue(c.type("one\rtwo\x1b[2J\x03three\nfour", "id-2")["ok"])
        t.wait_for(r'LINE "one two three four"')
        self.assertEqual(self.lines(), ["one two three four"])
        t.send("exit 0\r")
        t.wait_exit()

    def test_refused_before_session_start_and_during_a_dialog(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.assertEqual(c.type("Claude, hello.", "a")["reason"], "no-session")
        self.session_started(info)
        hook(self.env, {"hook_event_name": "PermissionRequest", "tool_name": "Bash",
                        "tool_input": {"command": "ls"}}, info["pid"])
        reply = c.type("Claude, hello.", "b")
        self.assertEqual((reply["ok"], reply["reason"], reply["detail"]), (False, "dialog", "permission: Bash"))
        self.assertEqual(c.check()["reason"], "dialog")
        hook(self.env, {"hook_event_name": "PostToolUse", "tool_name": "Bash",
                        "tool_input": {"command": "ls"}}, info["pid"])
        self.assertTrue(c.type("Claude, hello.", "b")["ok"])
        t.wait_for(r'LINE "Claude, hello."')
        self.assertEqual(self.lines(), ["Claude, hello."])
        t.send("exit 0\r")
        t.wait_exit()

    def test_enter_at_the_keyboard_answers_the_dialog(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        hook(self.env, {"hook_event_name": "PermissionRequest", "tool_name": "Bash",
                        "tool_input": {"command": "sleep 600"}}, info["pid"])
        time.sleep(0.4)
        t.send("1\r")                                   # the user picks "Yes"
        time.sleep(0.4)
        self.assertTrue(c.type("Claude, hello.", "c")["ok"])
        t.send("exit 0\r")
        t.wait_exit()

    def test_draft_and_recent_keys_refused(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        t.send("half typed")
        time.sleep(0.3)
        self.assertEqual(c.type("Claude, hello.", "d")["reason"], "draft")
        t.send("\x7f" * 10)                             # erased again
        self.assertEqual(c.type("Claude, hello.", "e")["reason"], "keyboard")
        time.sleep(0.4)
        self.assertTrue(c.type("Claude, hello.", "e")["ok"])
        t.send("exit 0\r")
        t.wait_exit()

    def test_stray_keys_are_cleared_before_typing(self):
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        t.send("jk ")                                   # stray keys, no word: not a draft worth keeping
        time.sleep(0.4)
        self.assertTrue(c.status()["draft_negligible"])
        self.assertTrue(c.type("Claude, hello.", "g")["ok"])
        t.wait_for(r'LINE "Claude, hello."')
        self.assertIn("KEYS 15", t.text)                # Ctrl-U, written on its own before the text
        self.assertEqual(self.lines(), ["Claude, hello."])
        self.assertEqual(c.status()["draft"], 0)
        t.send("exit 0\r")
        t.wait_exit()

    def test_keys_pressed_while_typing_wait_until_after_enter(self):
        write_config(self.tmp, dict(FAST, submit_delay=1.0))
        t = self.start(proxy_argv(*standin_argv()))
        t.wait_for(r"READY")
        c, info = self.client()
        self.session_started(info)
        result = {}
        thread = threading.Thread(target=lambda: result.update(c.type("injected line", "f")))
        thread.start()
        t.wait_for(r"KEYS " + b"injected line".hex())
        t.send("xyz")                                   # typed during the injection
        thread.join(5)
        self.assertTrue(result.get("ok"))
        t.wait_for(r'LINE "injected line"')
        t.wait_for(r"KEYS 78797a")
        out = t.text
        self.assertLess(out.index('LINE "injected line"'), out.index("KEYS 78797a"))
        t.send("\r")
        t.wait_for(r'LINE "xyz"')
        t.send("exit 0\r")
        t.wait_exit()

    def test_only_one_proxy_takes_requests(self):
        first = self.start(proxy_argv(*standin_argv()))
        first.wait_for(r"READY")
        _, info = self.client()
        second = self.start(proxy_argv(*standin_argv()))
        second.wait_for(r"READY")
        self.assertEqual(proxy_info(self.tmp / "state")["pid"], info["pid"])
        second.send("exit 2\r")                          # the second one still relays fully
        self.assertEqual(os.waitstatus_to_exitcode(second.wait_exit()), 2)
        first.send("exit 0\r")
        first.wait_exit()


class ShellTest(ProxyCase):
    """A real interactive shell under the proxy."""

    def test_bash(self):
        t = self.start(proxy_argv("/bin/bash", "--norc", "--noprofile", "-i"), rows=30, cols=90)
        t.wait_for(r"\$ ")
        t.send("echo $((6*7))\r")
        t.wait_for(r"\r\n42\r\n")
        t.send("stty size\r")
        t.wait_for(r"30 90")
        t.resize(40, 120)
        time.sleep(0.2)
        t.send("stty size\r")
        t.wait_for(r"40 120")
        t.send("tty -s && echo is-a-tty\r")
        t.wait_for(r"\r\nis-a-tty")
        c, info = self.client()
        self.session_started(info)
        time.sleep(0.3)
        self.assertTrue(c.type("echo $((1000+234))", "g")["ok"])
        t.wait_for(r"\r\n1234\r\n")
        time.sleep(0.5)
        self.assertEqual(t.count(r"\r\n1234\r\n"), 1)
        t.send("sleep 30 &\r")
        t.wait_for(r"\[1\]")
        t.send("kill %1; exit 3\r")
        self.assertEqual(os.waitstatus_to_exitcode(t.wait_exit()), 3)


class NoTerminalTest(unittest.TestCase):
    def test_without_a_terminal_the_program_just_runs(self):
        tmp = make_tmp()
        try:
            write_config(tmp, FAST)
            p = subprocess.run(proxy_argv("/bin/sh", "-c", "echo plain; exit 5"), env=isolated_env(tmp),
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
            self.assertEqual((p.returncode, p.stdout), (5, b"plain\n"))
            self.assertFalse((tmp / "state" / "listen" / "proxy.json").exists())
        finally:
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
