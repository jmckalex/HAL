"""Run the HAL voice session and keep it running: `bin/hal-session`.

    bin/hal-session [--fresh | --continue] [--dry-run] [-- CLAUDE_ARGS...]
    bin/hal-session --check

Run it in a terminal window; launchd/hal-voice.command does that at login (see
AUTOSTART.md). It runs the voice session, `bin/hal claude`, in the foreground:

- A deliberate exit is not fought. If Claude Code exits normally (/exit, Ctrl-C
  twice, Ctrl-D: exit code 0), the window is closed (SIGHUP), or the session is
  terminated (SIGTERM, SIGINT), it prints one line on how to start again and stops.
- A crash is restarted. Any other exit (another exit code, or a kill by another
  signal such as SIGKILL or SIGSEGV) starts Claude Code again after a delay of
  5, 10, 20, 40, then 60 s, at most RESTART_LIMIT times in a row. A run that
  lasted STABLE_AFTER seconds resets the count. Ctrl-C during the delay stops.
- One conversation per day. It continues the previous conversation
  (`claude --continue`) when the last voice session it started began on the
  same day, and starts a fresh one on the first start of a new day. A day
  begins at NEW_DAY_HOUR (04:00) local time, so a restart after midnight still
  continues the evening's conversation. If a `--continue` start fails within
  QUICK_FAIL seconds (perhaps there was nothing to continue), the retry is fresh.
- One voice session at a time. It does nothing if another launcher is
  supervising one, or if a voice session is already running under the pty
  proxy (state/listen/proxy.json), however it was started.

    --check     say whether a voice session is running (exit 0) or not (exit 1)
    --dry-run   say what would be run, and start nothing
    --fresh     start a fresh conversation this time
    --continue  continue the last conversation this time

State, in state/autostart/:

    session.lock      held (flock) by the launcher while it supervises
    session.json      the launcher's pid and state: running, waiting (to restart),
                      stopped or gave-up; the menu-bar item reads it
    last-start.json   the day and time of the last start, and how it started
    autostart.log     a line per start, exit and decision

Standard library only. Testing aids (environment):

    HAL_STATE_DIR        state directory (default <project>/state)
    HAL_SESSION_COMMAND  run this instead of `bin/hal claude` (a stand-in)
    HAL_SESSION_NOW      the current local time, ISO format, for the day decision
    HAL_SESSION_DELAY    first restart delay in seconds (default 5)
    HAL_SESSION_STABLE   seconds after which a run counts as stable (default 600)
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESTART_LIMIT = 5        # restarts in a row before giving up
DELAY_CAP = 60.0         # longest wait before a restart, in seconds
NEW_DAY_HOUR = 4         # a new day (and a fresh conversation) starts at 04:00
QUICK_FAIL = 15.0        # a --continue start that fails this fast is retried fresh
# Exits that are the user's (or the system's) decision, not a crash: a normal
# exit, the conventional codes for SIGHUP, SIGINT and SIGTERM, and those signals.
DELIBERATE_CODES = {0, 129, 130, 143}
STOP_SIGNALS = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
STATES_RUNNING = ("starting", "running", "waiting")
LOG_LIMIT = 256 * 1024
AGAIN = "To start it again, run bin/hal-session in this folder, or choose Start in the HAL menu."


def state_dir() -> Path:
    return Path(os.environ.get("HAL_STATE_DIR") or ROOT / "state")


def auto_dir() -> Path:
    return state_dir() / "autostart"


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ[name]))
    except (KeyError, ValueError):
        return default


def now() -> dt.datetime:
    fake = os.environ.get("HAL_SESSION_NOW")
    if fake:
        try:
            return dt.datetime.fromisoformat(fake)
        except ValueError:
            pass
    return dt.datetime.now()


def day_of(moment: dt.datetime) -> str:
    """The day a moment belongs to: the calendar date, NEW_DAY_HOUR hours late."""
    return (moment - dt.timedelta(hours=NEW_DAY_HOUR)).date().isoformat()


# ------------------------------------------------------------ files

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def log(message: str) -> None:
    try:
        path = auto_dir() / "autostart.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size > LOG_LIMIT:
                path.replace(path.with_name(path.name + ".1"))
        except FileNotFoundError:
            pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] session: {message}\n")
    except OSError:
        pass


def say(message: str) -> None:
    """One line for the person at the window (which may have been closed)."""
    try:
        sys.stdout.write(f"hal-session: {message}\r\n" if _raw_tty() else f"hal-session: {message}\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _raw_tty() -> bool:
    """True if the terminal is not translating newlines (a crashed program left it raw)."""
    try:
        import termios

        return os.isatty(1) and not termios.tcgetattr(1)[1] & termios.OPOST
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------ is a voice session running?

def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _command_of(pid: int) -> str:
    try:
        return subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def running_session(exclude_pid: int | None = None) -> str | None:
    """A description of the voice session that is running, or None.

    Reads files only (no lock is probed, so a launcher or proxy that is just
    starting can't be made to fail), and checks each recorded pid is alive and
    is the program it should be, in case the pid has been reused."""
    info = _read_json(auto_dir() / "session.json")
    pid = info.get("pid")
    if pid != exclude_pid and info.get("state") in STATES_RUNNING and _alive(pid) \
            and "hal_session" in _command_of(pid):
        what = "waiting to restart Claude Code" if info.get("state") == "waiting" else "running"
        return f"a voice session is running under bin/hal-session (pid {pid}, {what})"
    proxy = _read_json(state_dir() / "listen" / "proxy.json")
    pid = proxy.get("pid")
    if _alive(pid) and "ptyproxy" in _command_of(pid):
        tty = proxy.get("tty")
        return f"a voice session is running under the pty proxy (pid {pid}{', ' + tty if tty else ''})"
    return None


# ------------------------------------------------------------ the launcher

class Launcher:
    def __init__(self, args: list[str]):
        self.mode = None                  # "fresh", "continue" or None (decide by day)
        self.dry_run = False
        self.check = False
        self.extra: list[str] = []
        if "--" in args:
            i = args.index("--")
            args, self.extra = args[:i], args[i + 1:]
        for a in args:
            if a == "--fresh":
                self.mode = "fresh"
            elif a == "--continue":
                self.mode = "continue"
            elif a == "--dry-run":
                self.dry_run = True
            elif a == "--check":
                self.check = True
            elif a in ("-h", "--help"):
                print(__doc__)
                raise SystemExit(0)
            else:
                print(f"hal-session: unknown option {a} (see --help)", file=sys.stderr)
                raise SystemExit(2)
        self.child: subprocess.Popen | None = None
        self.stop_signal: int | None = None
        self.started_at = time.time()
        self.tty_attrs = None

    # -------------------------------------------------------- decisions

    def decide(self, force_fresh: bool = False) -> tuple[bool, str]:
        """(continue?, why) for a start now."""
        today = day_of(now())
        if self.mode == "continue":
            return True, "--continue given"
        if self.mode == "fresh" or force_fresh:
            return False, "--fresh given" if self.mode == "fresh" else "the last start failed at once"
        last = _read_json(auto_dir() / "last-start.json").get("day")
        if last == today:
            return True, f"the last voice session started today ({today})"
        if last:
            return False, f"the first start of {today} (the last was on {last})"
        return False, f"no earlier start recorded (today is {today})"

    def command(self, cont: bool) -> list[str]:
        base = shlex.split(os.environ.get("HAL_SESSION_COMMAND") or "") or [str(ROOT / "bin" / "hal"), "claude"]
        return base + (["--continue"] if cont else []) + self.extra

    def env(self) -> dict:
        env = dict(os.environ)
        if not shutil.which("claude", path=env.get("PATH")):
            # Not started from a login shell: add where Claude Code is usually installed.
            extra = [str(Path.home() / ".local" / "bin"), "/opt/homebrew/bin", "/usr/local/bin"]
            env["PATH"] = os.pathsep.join([env.get("PATH") or "/usr/bin:/bin:/usr/sbin:/sbin"]
                                          + [p for p in extra if os.path.isdir(p)])
        return env

    # -------------------------------------------------------- state

    def status(self, state: str, **fields) -> None:
        data = {"pid": os.getpid(), "state": state, "started": self.started_at, "updated": time.time(),
                "child_pid": self.child.pid if self.child and state == "running" else None}
        data.update(fields)
        _write_json(auto_dir() / "session.json", data)

    def _lock(self) -> int | None:
        d = auto_dir()
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d / "session.lock", os.O_RDWR | os.O_CREAT, 0o600)  # not inherited by children
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(0.05)

    # -------------------------------------------------------- signals and the terminal

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self.stop_signal = signum
        child = self.child
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)  # pass it on; the session decides how to end
            except OSError:
                pass

    def _save_tty(self) -> None:
        try:
            import termios

            if os.isatty(0):
                self.tty_attrs = termios.tcgetattr(0)
        except Exception:  # noqa: BLE001
            self.tty_attrs = None

    def _restore_tty(self, abnormal: bool) -> None:
        """Put the terminal back as it was, in case a crashed program left it raw."""
        if self.tty_attrs is None:
            return
        try:
            import termios

            termios.tcsetattr(0, termios.TCSADRAIN, self.tty_attrs)
        except Exception:  # noqa: BLE001
            return
        if abnormal and os.isatty(1):
            # Show the cursor; turn off mouse reporting, bracketed paste, focus
            # events and the kitty keyboard protocol, which a crash can leave on.
            try:
                sys.stdout.write("\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l"
                                 "\x1b[?2004l\x1b[?1004l\x1b[<u\n")
                sys.stdout.flush()
            except OSError:
                pass

    def _wait(self) -> int:
        """Wait for the session to end; its exit code, or minus the signal that killed it."""
        child = self.child
        while True:
            try:
                _, status = os.waitpid(child.pid, os.WUNTRACED)
            except ChildProcessError:
                return child.returncode if child.returncode is not None else 0
            if os.WIFSTOPPED(status):
                self._suspended()
                continue
            child.returncode = os.waitstatus_to_exitcode(status)
            return child.returncode

    def _suspended(self) -> None:
        """Claude Code was suspended (Ctrl-Z), and the pty proxy has stopped
        itself so that a shell can take the terminal back. A shell only notices
        when the whole job stops, and this launcher (and the window's script) are
        in that job too: stop them as well, so the shell shows its prompt and
        `fg` resumes the session. Then make sure the session continues. Where no
        shell manages the job, the kernel discards the stop, so it continues at
        once instead of hanging."""
        log("Claude Code was suspended (Ctrl-Z)")
        try:
            if os.isatty(0) and os.tcgetpgrp(0) == os.getpgrp():
                os.killpg(os.getpgrp(), signal.SIGTSTP)   # returns when continued (fg)
        except OSError:
            pass
        try:
            os.kill(self.child.pid, signal.SIGCONT)
        except OSError:
            pass
        log("Claude Code continued")

    def _sleep(self, seconds: float) -> bool:
        """Wait, unless a stop signal arrives (False)."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_signal is not None:
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return self.stop_signal is None

    # -------------------------------------------------------- main

    def run(self) -> int:
        if self.check:
            found = running_session()
            print(found or "no voice session is running")
            return 0 if found else 1
        if self.dry_run:
            return self._dry_run()

        lock = self._lock()
        if lock is None:
            other = running_session() or "another bin/hal-session holds state/autostart/session.lock"
            say(f"not starting another voice session: {other}.")
            log(f"not started: {other}")
            return 0
        other = running_session(exclude_pid=os.getpid())
        if other:
            say(f"not starting another voice session: {other}.")
            log(f"not started: {other}")
            os.close(lock)
            return 0
        self.status("starting")   # visible to --check and the menu from now on

        for sig in STOP_SIGNALS:
            signal.signal(sig, self._on_signal)
        self._save_tty()
        delay0 = _float_env("HAL_SESSION_DELAY", 5.0)
        stable = _float_env("HAL_SESSION_STABLE", 600.0)
        crashes = 0
        force_fresh = False
        try:
            while True:
                cont, why = self.decide(force_fresh)
                force_fresh = False
                argv = self.command(cont)
                moment = now()
                _write_json(auto_dir() / "last-start.json",
                            {"day": day_of(moment), "time": moment.isoformat(timespec="seconds"),
                             "continued": cont, "argv": argv, "launcher_pid": os.getpid()})
                say(("continuing the conversation: " if cont else "starting a fresh conversation: ") + why)
                log(f"starting ({why}): {shlex.join(argv)}")
                began = time.monotonic()
                try:
                    self.child = subprocess.Popen(argv, cwd=str(ROOT), env=self.env())
                except OSError as exc:
                    say(f"cannot start {argv[0]}: {exc}")
                    log(f"cannot start {argv[0]}: {exc}")
                    self.status("stopped", reason=f"cannot start: {exc}")
                    return 1
                self.status("running", continued=cont, restarts=crashes)
                code = self._wait()
                ran = time.monotonic() - began
                how = f"exit code {code}" if code >= 0 else f"killed by {signal.Signals(-code).name}"
                self._restore_tty(abnormal=code != 0)
                log(f"Claude Code ended after {ran:.0f} s: {how}")

                if self.stop_signal is not None:
                    name = signal.Signals(self.stop_signal).name
                    log(f"stopping: the launcher received {name}")
                    self.status("stopped", reason=f"{name} received; {how}")
                    say(f"voice session ended ({name}). {AGAIN}")
                    return 0
                if code in DELIBERATE_CODES or (code < 0 and -code in STOP_SIGNALS):
                    self.status("stopped", reason=how)
                    say(f"voice session ended ({how}). {AGAIN}")
                    return 0

                # An unexpected exit: restart, within limits.
                if ran >= stable:
                    crashes = 0
                crashes += 1
                if crashes > RESTART_LIMIT:
                    msg = (f"Claude Code exited unexpectedly {crashes} times in a row (last: {how}); "
                           f"not restarting. {AGAIN}")
                    log(f"gave up: {msg}")
                    self.status("gave-up", reason=how)
                    say(msg)
                    return 1
                if cont and ran < QUICK_FAIL:
                    force_fresh = True
                delay = min(DELAY_CAP, delay0 * 2 ** (crashes - 1))
                say(f"Claude Code exited unexpectedly ({how}). Restarting in {delay:.0f} s "
                    f"(restart {crashes} of {RESTART_LIMIT}); press Ctrl-C to stop.")
                log(f"restart {crashes} of {RESTART_LIMIT} in {delay:.1f} s"
                    + ("; the next start is fresh" if force_fresh else ""))
                self.status("waiting", restart_at=time.time() + delay, restarts=crashes, reason=how)
                if not self._sleep(delay):
                    name = signal.Signals(self.stop_signal).name
                    log(f"not restarting: {name} received while waiting")
                    self.status("stopped", reason=f"{name} received while waiting")
                    say(f"not restarting. {AGAIN}")
                    return 0
                other = running_session(exclude_pid=os.getpid())
                if other:
                    log(f"not restarting: {other}")
                    self.status("stopped", reason=f"another session: {other}")
                    say(f"not restarting: {other}.")
                    return 0
        finally:
            os.close(lock)

    def _dry_run(self) -> int:
        other = running_session()
        if other:
            print(f"hal-session: {other}; would not start another")
            return 0
        cont, why = self.decide()
        print(f"hal-session: would run {shlex.join(self.command(cont))}")
        print(f"hal-session: {'continue' if cont else 'fresh'}, because {why}; in {ROOT}")
        return 0


def main(argv: list[str] | None = None) -> int:
    code = Launcher(sys.argv[1:] if argv is None else argv).run()
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        # The window has closed, so output can't be flushed; without this the
        # interpreter would turn the exit status into 120.
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
