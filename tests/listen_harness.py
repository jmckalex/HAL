"""Test helpers for the listener: a scripted terminal and an isolated environment.

Terminal runs a command inside a pseudo-terminal of its own and plays the
part of the user's terminal emulator: it reads everything the command
prints, types keys, resizes the window and collects the exit status.
"""

import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAL_LISTEN = ROOT / "bin" / "hal-listen"
STANDIN = Path(__file__).resolve().parent / "listen_standin.py"
LISTEN_PY = ROOT / ".venv-listen" / "bin" / "python"


def isolated_env(tmp: Path, **extra) -> dict:
    """Environment whose state, config and audio player are all in tmp."""
    env = dict(os.environ)
    state = tmp / "state"
    state.mkdir(parents=True, exist_ok=True)
    env.update({
        "HAL_STATE_DIR": str(state),
        "HAL_CONFIG": str(tmp / "config.json"),
        "HAL_PLAYER": f"{sys.executable} {ROOT / 'tests' / 'fake_player.py'}",
        "HAL_FAKE_OUT": str(tmp / "played"),
        "HAL_SUMMARIZER": f"{sys.executable} {ROOT / 'tests' / 'fake_summarizer.py'}",
    })
    for name in ("HAL_LISTEN_PROXY", "HAL_VOICE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(name, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def write_config(tmp: Path, listener: dict | None = None, **top) -> None:
    data = dict(top)
    data["listener"] = listener or {}
    (tmp / "config.json").write_text(json.dumps(data))


class Terminal:
    def __init__(self, argv, env=None, rows=24, cols=80, cwd=None):
        self.argv = [str(a) for a in argv]
        pid, fd = pty.fork()
        if pid == 0:
            try:
                fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                if cwd:
                    os.chdir(cwd)
                os.execvpe(self.argv[0], self.argv, env or os.environ)
            finally:
                os._exit(127)
        self.pid, self.fd = pid, fd
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.buf = b""
        self.status = None
        self.closed = False

    def pump(self, timeout=0.0):
        """Read whatever output is available (no threads: pty.fork() and threads don't mix)."""
        deadline = time.monotonic() + timeout
        while not self.closed:
            try:
                r, _, _ = select.select([self.fd], [], [], max(0.0, deadline - time.monotonic()))
            except (OSError, ValueError):
                self.closed = True
                break
            if not r:
                break
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                data = b""
            if not data:
                self.closed = True
                break
            self.buf += data

    @property
    def text(self) -> str:
        self.pump()
        return self.buf.decode("utf-8", "replace")

    def send(self, data):
        os.write(self.fd, data.encode() if isinstance(data, str) else data)

    def resize(self, rows, cols):
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def wait_for(self, pattern, timeout=10.0, start=0):
        """Wait until pattern (a regex) appears in the output after offset start."""
        deadline = time.monotonic() + timeout
        rx = re.compile(pattern)
        while time.monotonic() < deadline:
            m = rx.search(self.text, start)
            if m:
                return m
            self.pump(0.02)
        raise AssertionError(f"timed out waiting for {pattern!r}; output so far:\n{self.text[-3000:]}")

    def count(self, pattern) -> int:
        return len(re.findall(pattern, self.text))

    def wait_exit(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.status = status
                self.pump(0.1)
                return status
            self.pump(0.02)
        raise AssertionError(f"process did not exit; output:\n{self.text[-3000:]}")

    def wait_stopped(self, timeout=5.0) -> bool:
        """Wait until the process stops (job control), reading its output meanwhile:
        a terminal that isn't read can't drain, and the proxy drains before stopping."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if pid and os.WIFSTOPPED(status):
                return True
            self.pump(0.02)
        return False

    def close(self):
        if self.status is None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Keep reading while it dies: exiting closes the terminal, which waits for output to drain.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    if os.waitpid(self.pid, os.WNOHANG)[0]:
                        break
                except ChildProcessError:
                    break
                self.pump(0.02)
        try:
            os.close(self.fd)
        except OSError:
            pass


def proxy_argv(*cmd):
    return [str(HAL_LISTEN), "pty", "--", *[str(c) for c in cmd]]


def standin_argv():
    return [sys.executable, str(STANDIN)]


def hook(env: dict, payload: dict, proxy_pid: int) -> None:
    """Fire `hal-listen hook` the way Claude Code would, for the given proxy's session."""
    env = dict(env, HAL_LISTEN_PROXY=str(proxy_pid))
    subprocess.run([str(HAL_LISTEN), "hook"], input=json.dumps(payload).encode(), env=env, check=True, timeout=10)


def proxy_info(state_dir: Path, timeout=5.0) -> dict:
    path = state_dir / "listen" / "proxy.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.02)
    raise AssertionError(f"no {path}")


def make_tmp(prefix="hal-lt-"):
    # The proxy moves its socket to a shorter path itself if this one is too long.
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()


def rmtree(path: Path):
    shutil.rmtree(path, ignore_errors=True)
