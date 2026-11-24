"""Tests for the autostart (AUTOSTART.md): bin/hal-session (the supervising
launcher), scripts/open-voice-window.sh, the install and uninstall scripts, the
LaunchAgent plists, and the menu-bar item's state detection and menu.

Nothing here starts Claude Code, opens a window, loads a LaunchAgent or touches
the real state/. The launcher runs a stand-in program (HAL_SESSION_COMMAND), the
opener a stand-in for `open` (HAL_OPEN), the install scripts use a temporary
folder for ~/Library/LaunchAgents and a stand-in launchctl, and everything uses
a temporary HAL_STATE_DIR. "Running" sessions, listeners and launchers are idle
Python processes whose arguments carry the marks the checks look for.

The menu-bar tests need menubar/HALMenu (menubar/build.sh) and are skipped
without it; the install tests are skipped if it is out of date, since
installing would build it.
"""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "bin" / "hal-session"
OPENER = ROOT / "scripts" / "open-voice-window.sh"
INSTALL = ROOT / "scripts" / "install-autostart.sh"
UNINSTALL = ROOT / "scripts" / "uninstall-autostart.sh"
COMMAND_FILE = ROOT / "launchd" / "hal-voice.command"
MENU = ROOT / "menubar" / "HALMenu"
LABELS = ("local.hal.voice-session", "local.hal.menubar")
UID = os.getuid()

# Stand-in for `bin/hal claude`: records its arguments, then does what the next
# word of the plan file says: N (exit N), kill:SIG (die by that signal), stop
# (stop, then exit 0 once continued), or sleep:S,STEP (sleep S seconds, then STEP).
STANDIN = textwrap.dedent('''
    import json, os, signal, sys, time
    with open(os.environ["STANDIN_LOG"], "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "pid": os.getpid()}) + "\\n")
    plan = os.environ["STANDIN_PLAN"]
    with open(plan) as f:
        steps = f.read().split()
    step = steps.pop(0) if steps else "0"
    with open(plan, "w") as f:
        f.write(" ".join(steps))
    if step.startswith("sleep:"):
        seconds, _, step = step[6:].partition(",")
        time.sleep(float(seconds))
    if step == "stop":                    # stopped, as the pty proxy stops itself on Ctrl-Z
        os.kill(os.getpid(), signal.SIGSTOP)
        step = "0"
    if step.startswith("kill:"):
        sig = getattr(signal, "SIG" + step[5:])
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        time.sleep(5)
    sys.exit(int(step or 0))
''')

# Stand-in for /bin/launchctl: records each call; "loaded" is a marker file.
FAKE_LAUNCHCTL = textwrap.dedent('''\
    #!/bin/bash
    echo "$*" >> "$STUB_DIR/calls"
    case "$1" in
      print) [ -e "$STUB_DIR/loaded-${2##*/}" ] ;;
      bootstrap) touch "$STUB_DIR/loaded-$(/usr/bin/plutil -extract Label raw -o - "$3")" ;;
      bootout) rm -f "$STUB_DIR/loaded-${2##*/}" ;;
      enable) exit 0 ;;
      *) exit 1 ;;
    esac
''')

# Stand-in for /usr/bin/open: records each call; fails for iTerm2 if asked to.
FAKE_OPEN = textwrap.dedent('''\
    #!/bin/bash
    echo "$*" >> "$STUB_DIR/open-calls"
    if [ -n "${STUB_FAIL_ITERM:-}" ] && [ "$2" = com.googlecode.iterm2 ]; then exit 1; fi
    exit 0
''')


class _Temp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hal-autostart-"))
        self.state = self.tmp / "state"
        (self.state / "listen").mkdir(parents=True)
        self.stub_dir = self.tmp / "stub"
        self.stub_dir.mkdir()
        self.procs: list[subprocess.Popen] = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def script(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    def idle_process(self, *marks: str) -> subprocess.Popen:
        """An idle process whose arguments include marks (what the pid checks look for)."""
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *marks],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        return p

    def dead_pid(self) -> int:
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        return p.pid

    def fake_proxy(self, pid: int | None = None) -> int:
        pid = pid or self.idle_process("hal.listener.ptyproxy").pid
        (self.state / "listen" / "proxy.json").write_text(json.dumps({"pid": pid, "tty": "/dev/ttys009"}))
        return pid

    def fake_listener(self, state: str = "listening", age: float = 0.0) -> int:
        pid = self.idle_process("hal.listener.cli", "_serve").pid
        (self.state / "listen" / "listener.json").write_text(json.dumps(
            {"pid": pid, "state": state, "updated": time.time() - age}))
        return pid


# ------------------------------------------------------------ bin/hal-session

class LauncherTest(_Temp):
    def setUp(self):
        super().setUp()
        self.runs_file = self.tmp / "runs.jsonl"
        self.plan = self.tmp / "plan"
        standin = self.script("standin.py", STANDIN)
        fake_bin = self.tmp / "bin"
        fake_bin.mkdir()
        # Belt and braces: if the stand-in were ever bypassed, `claude` is not the real one.
        (fake_bin / "claude").write_text("#!/bin/sh\nexit 97\n")
        (fake_bin / "claude").chmod(0o755)
        self.env = dict(os.environ, HAL_STATE_DIR=str(self.state),
                        HAL_SESSION_COMMAND=f"{shlex.quote(sys.executable)} {shlex.quote(str(standin))}",
                        HAL_SESSION_DELAY="0.05", HAL_SESSION_NOW="2026-11-16T10:00:00",
                        STANDIN_LOG=str(self.runs_file), STANDIN_PLAN=str(self.plan), HAL_LISTEN="0",
                        PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
        self.env.pop("HAL_SESSION_STABLE", None)

    def session_env(self, plan=(), now=None, **extra) -> dict:
        self.plan.write_text(" ".join(plan))
        env = dict(self.env, **extra)
        if now:
            env["HAL_SESSION_NOW"] = now
        return env

    def session(self, *args, plan=(), now=None, timeout=30, **extra) -> subprocess.CompletedProcess:
        return subprocess.run([str(SESSION), *args], env=self.session_env(plan, now, **extra),
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)

    def start_session(self, *args, plan=(), **extra) -> subprocess.Popen:
        p = subprocess.Popen([str(SESSION), *args], env=self.session_env(plan, **extra), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.procs.append(p)
        return p

    def runs(self) -> list[list[str]]:
        if not self.runs_file.exists():
            return []
        return [json.loads(line)["argv"] for line in self.runs_file.read_text().splitlines()]

    def status(self) -> dict:
        return json.loads((self.state / "autostart" / "session.json").read_text())

    def last_start(self) -> dict:
        return json.loads((self.state / "autostart" / "last-start.json").read_text())

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        self.fail("timed out waiting")

    # -------------------------------------------------------- exits

    def test_normal_exit_stops_with_one_line(self):
        r = self.session(plan=["0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.runs()), 1)
        self.assertIn("voice session ended (exit code 0). To start it again, run bin/hal-session", r.stdout)
        self.assertEqual(self.status()["state"], "stopped")

    def test_signal_and_conventional_codes_are_deliberate(self):
        for step in ("kill:HUP", "kill:TERM", "kill:INT", "130", "143"):
            with self.subTest(step=step):
                self.runs_file.unlink(missing_ok=True)
                r = self.session(plan=[step, "0"])
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(len(self.runs()), 1, f"{step} must not be restarted")
                self.assertIn("To start it again", r.stdout)

    def test_crash_restarts_until_the_limit(self):
        r = self.session(plan=["1"] * 10)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.runs()), 6)  # the first start and 5 restarts
        self.assertEqual(r.stdout.count("Restarting in"), 5)
        self.assertIn("exited unexpectedly 6 times in a row (last: exit code 1); not restarting", r.stdout)
        self.assertEqual(self.status()["state"], "gave-up")

    def test_kill_and_segfault_count_as_crashes(self):
        r = self.session(plan=["kill:KILL", "kill:SEGV", "0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.runs()), 3)
        self.assertIn("killed by SIGKILL", r.stdout)
        self.assertIn("killed by SIGSEGV", r.stdout)

    def test_a_stable_run_resets_the_count(self):
        r = self.session(plan=["sleep:0.3,1"] * 7 + ["0"], HAL_SESSION_STABLE="0.2")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(len(self.runs()), 8)
        self.assertIn("restart 1 of 5", r.stdout)
        self.assertNotIn("restart 2 of 5", r.stdout)

    def test_a_suspended_session_is_continued(self):
        # Ctrl-Z with no job-control shell around (stdin is not a terminal here):
        # the session is continued rather than left stopped.
        r = self.session(plan=["stop"], timeout=15)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(len(self.runs()), 1)
        self.assertIn("voice session ended (exit code 0)", r.stdout)
        self.assertIn("suspended (Ctrl-Z)", (self.state / "autostart" / "autostart.log").read_text())

    def test_ctrl_c_while_waiting_stops(self):
        p = self.start_session(plan=["1", "0"], HAL_SESSION_DELAY="10")
        self.wait_for(lambda: self.status()["state"] == "waiting")
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=5)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(len(self.runs()), 1)
        self.assertIn("not restarting", out)
        self.assertEqual(self.status()["state"], "stopped")

    # -------------------------------------------------------- one conversation a day

    def test_first_start_ever_is_fresh_and_recorded(self):
        self.session(plan=["0"], now="2026-11-16T09:00:00")
        self.assertEqual(self.runs(), [[]])
        self.assertEqual(self.last_start()["day"], "2026-11-16")
        self.assertFalse(self.last_start()["continued"])

    def test_same_day_continues_and_a_new_day_is_fresh(self):
        self.session(plan=["0"], now="2026-11-16T09:00:00")
        self.session(plan=["0"], now="2026-11-16T18:30:00")
        self.session(plan=["0"], now="2026-11-17T09:00:00")
        self.session(plan=["0"], now="2026-11-17T09:30:00")
        self.assertEqual(self.runs(), [[], ["--continue"], [], ["--continue"]])
        self.assertEqual(self.last_start()["day"], "2026-11-17")

    def test_the_day_turns_at_four_in_the_morning(self):
        self.session(plan=["0"], now="2026-11-16T23:30:00")
        self.session(plan=["0"], now="2026-11-17T01:30:00")   # still the 16th's evening
        self.session(plan=["0"], now="2026-11-17T04:30:00")   # a new day
        self.assertEqual(self.runs(), [[], ["--continue"], []])

    def test_restart_after_a_crash_continues_the_same_day(self):
        r = self.session(plan=["sleep:0.2,kill:KILL", "0"], HAL_SESSION_STABLE="0.1")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.runs(), [[], ["--continue"]])

    def test_a_quick_failure_to_continue_is_retried_fresh(self):
        self.session(plan=["0"], now="2026-11-16T09:00:00")
        r = self.session(plan=["1", "0"], now="2026-11-16T10:00:00")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.runs(), [[], ["--continue"], []])

    def test_fresh_and_continue_options_and_extra_arguments(self):
        self.session(plan=["0"], now="2026-11-16T09:00:00")
        self.session("--fresh", plan=["0"], now="2026-11-16T10:00:00")
        self.session("--continue", "--", "--model", "opus", plan=["0"], now="2026-11-20T10:00:00")
        self.assertEqual(self.runs(), [[], [], ["--continue", "--model", "opus"]])

    def test_dry_run_starts_and_records_nothing(self):
        self.session(plan=["0"], now="2026-11-16T09:00:00")
        before = self.last_start()
        r = self.session("--dry-run", now="2026-11-16T12:00:00")
        self.assertEqual(r.returncode, 0)
        self.assertIn("would run", r.stdout)
        self.assertIn("--continue", r.stdout)
        self.assertEqual(len(self.runs()), 1)
        self.assertEqual(self.last_start(), before)

    # -------------------------------------------------------- one session at a time

    def test_a_second_launch_does_nothing(self):
        first = self.start_session(plan=["sleep:30,0"])
        self.wait_for(lambda: self.status()["state"] == "running" and len(self.runs()) == 1)
        r = self.session(plan=["0"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("not starting another voice session: a voice session is running under bin/hal-session", r.stdout)
        self.assertEqual(len(self.runs()), 1)
        check = self.session("--check")
        self.assertEqual(check.returncode, 0)
        # Closing the window (SIGHUP) ends the first without a restart.
        first.send_signal(signal.SIGHUP)
        out, _ = first.communicate(timeout=10)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(len(self.runs()), 1)
        self.assertIn("voice session ended (SIGHUP)", out)
        self.assertEqual(self.session("--check").returncode, 1)

    def test_a_session_under_the_proxy_blocks_a_start(self):
        pid = self.fake_proxy()
        r = self.session(plan=["0"])
        self.assertEqual(r.returncode, 0)
        self.assertIn(f"under the pty proxy (pid {pid}, /dev/ttys009)", r.stdout)
        self.assertEqual(self.runs(), [])
        self.assertFalse((self.state / "autostart" / "last-start.json").exists())

    def test_stale_or_reused_pids_are_not_a_session(self):
        self.fake_proxy(self.dead_pid())
        self.assertEqual(self.session("--check").returncode, 1)
        self.fake_proxy(self.idle_process("something-else").pid)   # alive, but not a proxy
        self.assertEqual(self.session("--check").returncode, 1)
        r = self.session(plan=["0"])
        self.assertEqual(len(self.runs()), 1, r.stdout)


# ------------------------------------------------------------ scripts/open-voice-window.sh

class OpenerTest(_Temp):
    def opener(self, *args, **extra) -> subprocess.CompletedProcess:
        fake_open = self.script("open", FAKE_OPEN)
        env = dict(os.environ, HAL_STATE_DIR=str(self.state), HAL_OPEN=str(fake_open), STUB_DIR=str(self.stub_dir),
                   **extra)
        return subprocess.run([str(OPENER), *args], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=20)

    def open_calls(self) -> list[str]:
        calls = self.stub_dir / "open-calls"
        return calls.read_text().splitlines() if calls.exists() else []

    def test_dry_run_prints_the_open_command(self):
        r = self.opener("--dry-run", HAL_TERMINAL_APP="iterm")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"would run: {self.tmp / 'open'} -b com.googlecode.iterm2 {COMMAND_FILE}", r.stdout)
        self.assertEqual(self.open_calls(), [])
        self.assertFalse((self.state / "autostart").exists())

    def test_opens_iterm2(self):
        r = self.opener(HAL_TERMINAL_APP="iterm")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.open_calls(), [f"-b com.googlecode.iterm2 {COMMAND_FILE}"])
        self.assertIn("opened an iTerm2 window", (self.state / "autostart" / "autostart.log").read_text())

    def test_falls_back_to_terminal(self):
        r = self.opener(HAL_TERMINAL_APP="iterm", STUB_FAIL_ITERM="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.open_calls(), [f"-b com.googlecode.iterm2 {COMMAND_FILE}",
                                             f"-b com.apple.Terminal {COMMAND_FILE}"])

    def test_terminal_without_iterm2(self):
        self.opener(HAL_TERMINAL_APP="terminal")
        self.assertEqual(self.open_calls(), [f"-b com.apple.Terminal {COMMAND_FILE}"])

    def test_a_running_session_opens_nothing(self):
        self.fake_proxy()
        r = self.opener(HAL_TERMINAL_APP="iterm")
        self.assertEqual(r.returncode, 0)
        self.assertIn("not opening another window", r.stdout)
        self.assertEqual(self.open_calls(), [])

    def test_login_dry_run_waits_for_nothing(self):
        started = time.monotonic()
        r = self.opener("--login", "--dry-run", HAL_TERMINAL_APP="iterm")
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn("would wait for the Dock", r.stdout)

    def test_the_command_file_runs_the_launcher(self):
        self.assertTrue(os.access(COMMAND_FILE, os.X_OK))
        text = COMMAND_FILE.read_text()
        self.assertIn("bin/hal-session", text)
        self.assertIn('cd "$(dirname "$0")/.."', text)


# ------------------------------------------------------------ the LaunchAgents

def rendered(label: str) -> bytes:
    """launchd/<label>.plist as install-autostart.sh installs it: @HAL_ROOT@ filled in."""
    return (ROOT / "launchd" / f"{label}.plist").read_bytes().replace(b"@HAL_ROOT@", str(ROOT).encode())


class PlistTest(unittest.TestCase):
    def load(self, label: str) -> dict:
        path = ROOT / "launchd" / f"{label}.plist"
        subprocess.run(["/usr/bin/plutil", "-lint", "-s", str(path)], check=True)
        return plistlib.loads(rendered(label))

    def test_templates_name_no_folder(self):
        for label in LABELS:
            with self.subTest(label=label):
                text = (ROOT / "launchd" / f"{label}.plist").read_text()
                self.assertIn("@HAL_ROOT@/", text)
                self.assertNotIn("/Users/", text)

    def test_common(self):
        for label in LABELS:
            with self.subTest(label=label):
                data = self.load(label)
                self.assertEqual(data["Label"], label)
                self.assertIs(data["RunAtLoad"], True)
                self.assertEqual(data["LimitLoadToSessionType"], "Aqua")
                self.assertTrue(data["ProgramArguments"][0].startswith(f"{ROOT}/"))
                for key in ("StandardOutPath", "StandardErrorPath"):
                    self.assertEqual(Path(data[key]).parent, ROOT / "state")

    def test_voice_session_runs_the_opener_once(self):
        data = self.load("local.hal.voice-session")
        self.assertEqual(data["ProgramArguments"], [str(OPENER), "--login"])
        self.assertTrue(os.access(OPENER, os.X_OK))
        self.assertNotIn("KeepAlive", data)

    def test_menubar_is_restarted_only_after_a_crash(self):
        data = self.load("local.hal.menubar")
        self.assertEqual(data["ProgramArguments"], [str(MENU)])
        self.assertEqual(data["KeepAlive"], {"SuccessfulExit": False})


# ------------------------------------------------------------ install / uninstall

@unittest.skipUnless(MENU.exists() and MENU.stat().st_mtime >= (ROOT / "menubar" / "HALMenu.swift").stat().st_mtime,
                     "menubar/HALMenu is missing or out of date (run menubar/build.sh)")
class InstallTest(_Temp):
    def setUp(self):
        super().setUp()
        self.agents = self.tmp / "LaunchAgents"
        self.launchctl = self.script("launchctl", FAKE_LAUNCHCTL)

    def run_script(self, script: Path, *args, launchctl=True) -> subprocess.CompletedProcess:
        env = dict(os.environ, HAL_LAUNCH_AGENTS_DIR=str(self.agents), STUB_DIR=str(self.stub_dir))
        env.pop("HAL_LAUNCHCTL", None)
        if launchctl:
            env["HAL_LAUNCHCTL"] = str(self.launchctl)
        r = subprocess.run([str(script), *args], env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def calls(self) -> list[str]:
        calls = self.stub_dir / "calls"
        return calls.read_text().splitlines() if calls.exists() else []

    def changing_calls(self) -> list[str]:
        return [c for c in self.calls() if not c.startswith("print ")]

    def test_dry_run_changes_nothing(self):
        r = self.run_script(INSTALL, "--dry-run")
        self.assertFalse(self.agents.exists())
        self.assertEqual(self.changing_calls(), [])
        self.assertIn(f"would run: {self.launchctl} bootstrap gui/{UID} {self.agents}/local.hal.menubar.plist",
                      r.stdout)
        self.run_script(UNINSTALL, "--dry-run")
        self.assertFalse(self.agents.exists())

    def test_install_is_idempotent_and_uninstall_undoes_it(self):
        self.run_script(INSTALL)
        for label in LABELS:
            self.assertEqual((self.agents / f"{label}.plist").read_bytes(), rendered(label))
        self.assertEqual(self.changing_calls(), [
            f"enable gui/{UID}/local.hal.voice-session",
            f"bootstrap gui/{UID} {self.agents}/local.hal.voice-session.plist",
            f"enable gui/{UID}/local.hal.menubar",
            f"bootstrap gui/{UID} {self.agents}/local.hal.menubar.plist",
        ])

        # Again: nothing to do.
        (self.stub_dir / "calls").unlink()
        r = self.run_script(INSTALL)
        self.assertEqual(self.changing_calls(), [])
        self.assertEqual(r.stdout.count("already loaded"), 2)

        # A changed file is replaced and reloaded.
        installed = self.agents / "local.hal.voice-session.plist"
        installed.write_bytes(installed.read_bytes().replace(b"--login", b"--old"))
        (self.stub_dir / "calls").unlink()
        self.run_script(INSTALL)
        self.assertEqual(self.changing_calls(), [
            f"bootout gui/{UID}/local.hal.voice-session",
            f"enable gui/{UID}/local.hal.voice-session",
            f"bootstrap gui/{UID} {installed}",
        ])
        self.assertEqual(installed.read_bytes(), rendered("local.hal.voice-session"))

        # Uninstall, dry run first.
        (self.stub_dir / "calls").unlink()
        self.run_script(UNINSTALL, "--dry-run")
        self.assertEqual(self.changing_calls(), [])
        self.assertTrue(installed.exists())
        self.run_script(UNINSTALL)
        self.assertEqual(self.changing_calls(), [f"bootout gui/{UID}/local.hal.menubar",
                                                 f"bootout gui/{UID}/local.hal.voice-session"])
        self.assertEqual(list(self.agents.iterdir()), [])
        (self.stub_dir / "calls").unlink()
        r = self.run_script(UNINSTALL)
        self.assertEqual(self.changing_calls(), [])
        self.assertEqual(r.stdout.count("not loaded"), 2)

    def test_a_stand_in_folder_never_runs_launchctl(self):
        r = self.run_script(INSTALL, launchctl=False)
        self.assertIn("skipped (stand-in folder, so no launchctl): launchctl bootstrap", r.stdout)
        self.assertEqual(self.calls(), [])
        self.assertTrue((self.agents / "local.hal.menubar.plist").exists())

    def test_uninstall_leaves_a_file_it_did_not_install(self):
        self.agents.mkdir()
        other = self.agents / "local.hal.menubar.plist"
        other.write_bytes(plistlib.dumps({"Label": "someone.else"}))
        r = self.run_script(UNINSTALL)
        self.assertTrue(other.exists())
        self.assertIn("left alone", r.stdout)


# ------------------------------------------------------------ the menu-bar item

@unittest.skipUnless(MENU.exists(), "menubar/HALMenu is not built (run menubar/build.sh)")
class MenuBarTest(_Temp):
    def free_port(self) -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def menu(self, *args, port: int | None = None) -> str:
        env = dict(os.environ, HAL_STATE_DIR=str(self.state), HAL_DECISIONS_PORT=str(port or self.free_port()))
        r = subprocess.run([str(MENU), *args], env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def status(self) -> dict:
        out = {}
        for line in self.menu("--status").splitlines():
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
        return out

    def test_nothing_running(self):
        s = self.status()
        self.assertEqual(s["session"], "not running")
        self.assertEqual(s["listening"], "not running")
        self.assertEqual(s["icon"], "moon.zzz (symbols found)")
        menu = self.menu("--print-menu")
        self.assertIn(f"Start Voice Session  ->  {OPENER}", menu)
        self.assertIn(f"Pause Listening  ->  {ROOT}/bin/hal-listen pause", menu)
        self.assertIn(f"Mute Voice  ->  {ROOT}/bin/hal off", menu)
        self.assertIn(f"Stop Speaking Now  ->  {ROOT}/bin/hal stop", menu)
        self.assertIn("Quit HAL Menu", menu)

    def test_running_and_listening(self):
        proxy = self.fake_proxy()
        self.fake_listener()
        s = self.status()
        self.assertEqual(s["session"], f"running (pty proxy pid {proxy}, /dev/ttys009)")
        self.assertEqual(s["listening"], "listening")
        self.assertEqual(s["icon"], "mic (symbols found)")
        menu = self.menu("--print-menu")
        self.assertIn(f"[info] Voice session: running (pid {proxy}, ttys009)", menu)
        self.assertNotIn("Start Voice Session  ->", menu.split("in other states:")[0])

    def test_paused_and_muted(self):
        self.fake_proxy()
        self.fake_listener("paused")
        (self.state / "listen" / "paused").touch()
        (self.state / "muted").touch()
        s = self.status()
        self.assertEqual(s["listening"], "paused")
        self.assertEqual(s["icon"], "mic.slash + speaker.slash (symbols found)")
        menu = self.menu("--print-menu")
        self.assertIn(f"Resume Listening  ->  {ROOT}/bin/hal-listen resume", menu)
        self.assertIn(f"Unmute Voice  ->  {ROOT}/bin/hal on", menu)
        self.assertIn("[info] Voice: muted", menu)

    def test_loading_and_unresponsive_listener(self):
        self.fake_proxy()
        self.fake_listener("loading", age=60)   # loading the model doesn't refresh the file
        self.assertEqual(self.status()["icon"], "hourglass (symbols found)")
        self.fake_listener("loading", age=180)
        self.assertEqual(self.status()["listening"], "not responding")
        self.fake_listener("listening", age=12)
        self.assertEqual(self.status()["listening"], "listening")
        self.fake_listener("listening", age=60)
        s = self.status()
        self.assertEqual(s["listening"], "not responding")
        self.assertEqual(s["icon"], "mic.badge.xmark (symbols found)")

    def test_restarting(self):
        launcher = self.idle_process("scripts/hal_session.py").pid
        (self.state / "autostart").mkdir()
        (self.state / "autostart" / "session.json").write_text(json.dumps(
            {"pid": launcher, "state": "waiting", "restart_at": time.time() + 30, "restarts": 2}))
        s = self.status()
        self.assertEqual(s["session"], "restarting (restart 2)")
        self.assertEqual(s["icon"], "arrow.triangle.2.circlepath (symbols found)")
        menu = self.menu("--print-menu").split("in other states:")[0]
        self.assertRegex(menu, r"Voice session: restarting in \d+ s \(restart 2\)")
        self.assertNotIn("Start Voice Session", menu)

    def test_stale_and_reused_pids_are_ignored(self):
        self.fake_proxy(self.dead_pid())
        self.assertEqual(self.status()["session"], "not running")
        self.fake_proxy(self.idle_process("something-else").pid)
        self.assertEqual(self.status()["session"], "not running")

    def test_speaking(self):
        worker = self.idle_process("hal.cli", "_worker").pid
        (self.state / "speaker.pid").write_text(json.dumps({"pid": worker}))
        self.assertEqual(self.status()["speaking"], "yes")

    def test_decisions_page_only_when_the_port_answers(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            menu = self.menu("--print-menu", port=port)
            self.assertIn(f"Open Decisions Page  ->  /usr/bin/open http://127.0.0.1:{port}/", menu.split("in other states:")[0])
        menu = self.menu("--print-menu", port=port)
        self.assertIn("does not answer", menu)
        self.assertNotIn("Open Decisions Page", menu.split("in other states:")[0])


if __name__ == "__main__":
    unittest.main()
