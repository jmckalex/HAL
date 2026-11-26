"""Run a program (claude) under a pseudo-terminal and relay the real terminal to it.

    hal-listen pty -- claude [args...]

To the person at the keyboard nothing changes: the real terminal is put in
raw mode and every byte is passed through in both directions; window-size
changes, job control (Ctrl-Z in the child), signals sent to the proxy and the
child's exit status are relayed as well. The proxy also listens on a Unix
socket (state/listen/pty.sock) for requests to type a line into the program
and press Enter, which is how the listener submits what was said.

Typing is refused while it would be unsafe (session.verdict): a dialog is
open, unsent text sits in the prompt box, or a key was pressed a moment ago.
Checked at the instant of typing, inside the single-threaded loop, so no
keystroke from the real keyboard can interleave with an injected line.

Control protocol: one JSON object per line, one reply per request.
    {"op": "ping"}                        -> {"ok": true, "pid": .., "child_pid": ..}
    {"op": "check"}                       -> {"ok": bool, "reason": .., "detail": ..}
    {"op": "status"}                      -> {"ok": true, ...}
    {"op": "type", "id": "..", "text": ".."}
        -> {"ok": true, "typed": n}  once the text and Enter have been written
        -> {"ok": false, "reason": "dialog" | "draft" | "keyboard" | "no-session" | ...}
    A repeated id is never typed twice: the first reply is returned again.

The child gets HAL_LISTEN_PROXY=<proxy pid> in its environment, so the hooks
of that Claude Code session (and only that one) keep its session-state file.
Only one proxy at a time offers the socket (proxy.lock); a second one still
relays the terminal but accepts no requests. Standard library only.
"""

from __future__ import annotations

import collections
import contextlib
import fcntl
import json
import os
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
import tty
import unicodedata
from pathlib import Path

from . import logs, session, settings

_FORWARD = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2)
_HANDLED = (signal.SIGWINCH, signal.SIGCHLD, signal.SIGCONT) + _FORWARD
_MAX_TEXT = 4000
_MAX_REQUEST = 64 * 1024
_RELEASE_GAP = 0.05   # after Enter, before keys pressed meanwhile are passed on
_CLEAR_LINE = b"\x15"  # Ctrl-U: clears a stray draft from Claude Code's prompt box
_CLEAR_GAP = 0.05     # after Ctrl-U, before the text


def proxy_file() -> Path:
    return settings.listen_dir() / "proxy.json"


def default_socket_path() -> Path:
    path = settings.listen_dir() / "pty.sock"
    if len(str(path)) < 100:  # AF_UNIX paths are limited to 104 bytes on macOS
        return path
    import hashlib

    tag = hashlib.sha1(str(settings.state_dir()).encode()).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / f"hal-listen-{os.getuid()}-{tag}.sock"


_ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]?|\x9b[0-?]*[ -/]*[@-~]")


def clean_text(text: str) -> str:
    """One line of printable text. Escape sequences go entirely; other control
    characters (a newline would submit early, Ctrl-C would interrupt) become spaces."""
    text = _ESCAPES.sub(" ", str(text))
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cc":
            out.append(" ")
        elif cat in ("Cf", "Cs", "Co", "Cn") and ch != "‍":
            continue
        else:
            out.append(ch)
    return " ".join("".join(out).split())[:_MAX_TEXT]


def _winsize(fd: int) -> bytes | None:
    try:
        return fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
    except OSError:
        return None


class _Conn:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.inbuf = b""
        self.outbuf = b""
        self.closing = False


class Proxy:
    def __init__(self, argv: list[str], *, control: bool = True, autostart: bool | None = None,
                 stdin_fd: int = 0, stdout_fd: int = 1):
        if not argv:
            raise ValueError("no command")
        self.argv = argv
        self.want_control = control
        self.cfg = settings.load()
        self.autostart = self.cfg["autostart"] if autostart is None else autostart
        self.in_fd, self.out_fd = stdin_fd, stdout_fd
        self.is_tty = os.isatty(stdin_fd)
        self.saved_attrs = None
        self.keyboard = session.Keyboard()
        self.child_pid = 0
        self.child_status: int | None = None
        self.master = -1
        self.to_child = bytearray()
        self.held_keys = bytearray()
        self.injecting: dict | None = None
        self.done: collections.OrderedDict = collections.OrderedDict()
        self.signals: collections.deque = collections.deque()
        self.listen_sock: socket.socket | None = None
        self.sock_path: Path | None = None
        self.lock_fd = -1
        self.conns: list[_Conn] = []
        self.starter: subprocess.Popen | None = None
        self.stdin_open = True
        self.stdout_ok = True
        self.drain_until = 0.0
        self.kill_at = 0.0
        self.cfg_loaded = time.monotonic()

    # ------------------------------------------------------------ setup

    def _on_signal(self, signum, frame):  # noqa: ARG002
        self.signals.append(signum)

    def _open_control(self) -> bool:
        d = settings.listen_dir()
        d.mkdir(parents=True, exist_ok=True)
        self.lock_fd = os.open(d / "proxy.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = -1
            logs.log("another proxy already serves the listener; relaying only", source="pty")
            return False
        path = default_socket_path()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            sock.bind(str(path))
        finally:
            os.umask(old)
        sock.listen(8)
        sock.setblocking(False)
        self.listen_sock, self.sock_path = sock, path
        return True

    def _write_proxy_file(self) -> None:
        info = {"pid": os.getpid(), "child_pid": self.child_pid, "socket": str(self.sock_path),
                "started": time.time(), "argv0": os.path.basename(self.argv[0]), "cwd": os.getcwd(),
                "tty": os.ttyname(self.in_fd) if self.is_tty else None}
        path = proxy_file()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, path)

    def _spawn(self) -> None:
        master, slave = os.openpty()
        if self.is_tty:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(slave, termios.TCSANOW, termios.tcgetattr(self.in_fd))
            size = _winsize(self.in_fd)
            if size:
                fcntl.ioctl(slave, termios.TIOCSWINSZ, size)
        env = dict(os.environ)
        if self.listen_sock is not None:
            env["HAL_LISTEN_PROXY"] = str(os.getpid())
        else:
            env.pop("HAL_LISTEN_PROXY", None)
        signal.pthread_sigmask(signal.SIG_BLOCK, _HANDLED)
        pid = os.fork()
        if pid == 0:
            try:
                signal.set_wakeup_fd(-1)
                os.close(master)
                _session_leader(slave, self.argv, env)
            except BaseException as exc:  # noqa: BLE001
                with contextlib.suppress(OSError):
                    os.write(2, f"hal-listen pty: cannot run {self.argv[0]}: {exc}\r\n".encode())
            os._exit(127)
        os.close(slave)
        self.child_pid, self.master = pid, master
        os.set_blocking(master, False)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _HANDLED)

    def _raw(self) -> None:
        if self.is_tty:
            with contextlib.suppress(termios.error):
                tty.setraw(self.in_fd, termios.TCSANOW)

    def _restore(self) -> None:
        if self.is_tty and self.saved_attrs is not None:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(self.in_fd, termios.TCSADRAIN, self.saved_attrs)

    def _sync_size(self) -> None:
        if self.is_tty and self.master >= 0:
            size = _winsize(self.in_fd)
            if size:
                with contextlib.suppress(OSError):
                    fcntl.ioctl(self.master, termios.TIOCSWINSZ, size)

    def _start_listener(self) -> None:
        launcher = settings.ROOT / "bin" / "hal-listen"
        try:
            self.starter = subprocess.Popen([str(launcher), "start", "--quiet"], stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                            start_new_session=True)
        except OSError as exc:
            logs.log(f"could not start the listener: {exc}", source="pty")

    # ------------------------------------------------------------ run

    def run(self) -> int:
        """Relay until the child exits; returns its wait status."""
        wake_r, wake_w = os.pipe()
        os.set_blocking(wake_r, False)
        os.set_blocking(wake_w, False)
        old_wakeup = signal.set_wakeup_fd(wake_w, warn_on_full_buffer=False)
        old_handlers = {sig: signal.signal(sig, self._on_signal) for sig in _HANDLED}
        try:
            if self.want_control:
                try:
                    self._open_control()
                except OSError as exc:
                    logs.log(f"control socket unavailable ({exc}); relaying only", source="pty")
                    self.listen_sock = None
            if self.is_tty:
                self.saved_attrs = termios.tcgetattr(self.in_fd)
            self._spawn()
            self._raw()
            if self.listen_sock is not None:
                self._write_proxy_file()
                logs.log(f"serving {os.path.basename(self.argv[0])} (pid {self.child_pid}) on {self.sock_path}",
                         source="pty")
                if self.autostart:
                    self._start_listener()
            self._loop(wake_r)
        finally:
            self._restore()
            self._cleanup()
            signal.set_wakeup_fd(old_wakeup)
            for sig, handler in old_handlers.items():
                with contextlib.suppress(Exception):
                    signal.signal(sig, handler)
            os.close(wake_r)
            os.close(wake_w)
        return self.child_status if self.child_status is not None else 0

    def _loop(self, wake_r: int) -> None:
        while True:
            now = time.monotonic()
            if self.child_status is not None and (self.master < 0 or now >= self.drain_until):
                return
            rl = [wake_r]
            if self.master >= 0:
                rl.append(self.master)
            if self.stdin_open:
                rl.append(self.in_fd)
            if self.listen_sock is not None:
                rl.append(self.listen_sock)
                rl += [c.sock for c in self.conns]
            wl = [self.master] if self.to_child and self.master >= 0 else []
            wl += [c.sock for c in self.conns if c.outbuf]
            timeout = self._timeout(now)
            try:
                readable, writable, _ = select.select(rl, wl, [], timeout)
            except InterruptedError:
                continue
            except (OSError, ValueError):  # a control connection went bad: drop the closed ones
                for c in [c for c in self.conns if c.sock.fileno() < 0]:
                    self._drop(c)
                readable, writable = [], []
            if wake_r in readable:
                with contextlib.suppress(OSError):
                    while os.read(wake_r, 512):
                        pass
            self._handle_signals()
            if self.master >= 0 and self.master in readable:
                self._from_child()
            if self.stdin_open and self.in_fd in readable:
                self._from_keyboard()
            if self.master >= 0 and self.master in writable:
                self._to_child()
            if self.listen_sock is not None:
                if self.listen_sock in readable:
                    self._accept()
                for c in list(self.conns):
                    try:
                        if c.sock in readable:
                            self._conn_read(c)
                        if c.outbuf and c.sock in writable:
                            self._conn_write(c)
                    except Exception as exc:  # the relay must survive any control-socket fault
                        logs.log(f"control connection error: {exc!r}", source="pty")
                        self._drop(c)
            self._timers()

    def _timeout(self, now: float) -> float | None:
        deadlines = []
        if self.injecting:
            deadlines.append(self.injecting["due"])
        if self.child_status is not None:
            deadlines.append(self.drain_until)
        if self.kill_at:
            deadlines.append(self.kill_at)
        if self.starter is not None:
            deadlines.append(now + 1.0)
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - now)

    # ------------------------------------------------------------ data

    def _write_out(self, data: bytes) -> None:
        if not self.stdout_ok:
            return
        view = memoryview(data)
        while view:
            try:
                n = os.write(self.out_fd, view)
            except InterruptedError:
                continue
            except BlockingIOError:
                select.select([], [self.out_fd], [], 1.0)
                continue
            except OSError:  # the terminal went away
                self.stdout_ok = False
                self._forward(signal.SIGHUP)
                return
            view = view[n:]

    def _from_child(self) -> None:
        try:
            data = os.read(self.master, 1 << 16)
        except BlockingIOError:
            return
        except OSError:  # EIO: every holder of the slave side has closed it
            data = b""
        if not data:
            os.close(self.master)
            self.master = -1
            if self.child_status is None:
                self._reap(block=True)
            return
        self._write_out(data)

    def _from_keyboard(self) -> None:
        try:
            data = os.read(self.in_fd, 1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self.stdin_open = False
            self._forward(signal.SIGHUP)
            return
        self.keyboard.feed(data, time.time())
        if self.injecting:
            self.held_keys += data
        else:
            self.to_child += data

    def _to_child(self) -> None:
        try:
            n = os.write(self.master, self.to_child[:4096])
        except BlockingIOError:
            return
        except OSError:
            self.to_child.clear()
            return
        del self.to_child[:n]

    # ------------------------------------------------------------ signals and the child

    def _forward(self, sig: int) -> None:
        if self.child_status is None and self.child_pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.child_pid, sig)
            if sig in (signal.SIGHUP, signal.SIGTERM) and not self.kill_at:
                self.kill_at = time.monotonic() + 10.0

    def _reap(self, block: bool = False) -> None:
        if not self.child_pid or self.child_status is not None:
            return
        deadline = time.monotonic() + (2.0 if block else 0.0)
        while True:
            try:
                pid, status = os.waitpid(self.child_pid, os.WNOHANG | os.WUNTRACED)
            except ChildProcessError:
                self.child_status = 0
                return
            if pid == 0:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.02)
                continue
            if os.WIFSTOPPED(status):
                self._suspend()
                return
            self.child_status = status
            self.drain_until = time.monotonic() + 0.3
            return

    def _suspend(self) -> None:
        """The child stopped itself (Ctrl-Z): stop too, so the shell regains the terminal."""
        self._restore()
        os.kill(os.getpid(), signal.SIGSTOP)
        # ... continued (fg): take the terminal back and wake the child.
        self._raw()
        self._sync_size()
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.child_pid, signal.SIGCONT)

    def _handle_signals(self) -> None:
        while self.signals:
            sig = self.signals.popleft()
            if sig == signal.SIGWINCH:
                self._sync_size()
            elif sig == signal.SIGCHLD:
                self._reap()
                if self.starter is not None and self.starter.poll() is not None:
                    self.starter = None
            elif sig == signal.SIGCONT:
                self._raw()
                self._sync_size()
                if self.child_pid and self.child_status is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(self.child_pid, signal.SIGCONT)
            elif sig in _FORWARD:
                self._forward(sig)

    def _timers(self) -> None:
        now = time.monotonic()
        inj = self.injecting
        if inj and now >= inj["due"]:
            if self.to_child:  # the previous piece is still being written; keep the pieces apart
                inj["due"] = now + 0.02
            elif inj["phase"] == "clear":
                self.to_child += inj.pop("text")
                self._to_child()
                inj["phase"] = "text"
                inj["due"] = now + inj["delay"]
            elif inj["phase"] == "text":
                # Enter on its own: Claude Code takes a burst that contains Enter as a paste.
                if inj["submit"]:
                    self.to_child += b"\r"
                    self._to_child()
                inj["phase"] = "enter"
                inj["due"] = now + _RELEASE_GAP
            else:
                self._finish_injection()
        if self.kill_at and now >= self.kill_at and self.child_status is None:
            logs.log("child ignored SIGHUP/SIGTERM for 10 s; killing it", source="pty")
            if self.master >= 0:  # the program's own process group first (its session leader can't pass SIGKILL on)
                with contextlib.suppress(OSError):
                    pgrp = os.tcgetpgrp(self.master)
                    if pgrp > 1 and pgrp != os.getpgrp():
                        os.killpg(pgrp, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.child_pid, signal.SIGKILL)
            self.kill_at = 0.0
        if self.starter is not None and self.starter.poll() is not None:
            self.starter = None
        if self.child_status is None and self.child_pid:
            self._reap()

    # ------------------------------------------------------------ the control socket

    def _accept(self) -> None:
        try:
            sock, _ = self.listen_sock.accept()
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            logs.log(f"accept failed: {exc}", source="pty")
            return
        sock.setblocking(False)
        self.conns.append(_Conn(sock))

    def _drop(self, c: _Conn) -> None:
        with contextlib.suppress(OSError):
            c.sock.close()
        if c in self.conns:
            self.conns.remove(c)
        if self.injecting and self.injecting.get("conn") is c:
            self.injecting["conn"] = None

    def _reply(self, c: _Conn | None, obj: dict) -> None:
        if c is None or c not in self.conns:
            return
        c.outbuf += (json.dumps(obj) + "\n").encode()
        self._conn_write(c)

    def _conn_read(self, c: _Conn) -> None:
        try:
            data = c.sock.recv(1 << 16)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            if not c.outbuf and not (self.injecting and self.injecting.get("conn") is c):
                self._drop(c)
            else:
                c.closing = True
            return
        c.inbuf += data
        if len(c.inbuf) > _MAX_REQUEST:
            self._reply(c, {"ok": False, "reason": "too-large"})
            c.closing = True
            c.inbuf = b""
            return
        while b"\n" in c.inbuf:
            line, c.inbuf = c.inbuf.split(b"\n", 1)
            try:
                req = json.loads(line.decode("utf-8"))
            except ValueError:
                self._reply(c, {"ok": False, "reason": "bad-request"})
                continue
            self._request(c, req if isinstance(req, dict) else {})

    def _conn_write(self, c: _Conn) -> None:
        try:
            n = c.sock.send(c.outbuf)
            c.outbuf = c.outbuf[n:]
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._drop(c)
            return
        if not c.outbuf and c.closing:
            self._drop(c)

    def _settings(self) -> dict:
        if time.monotonic() - self.cfg_loaded > 2.0:
            self.cfg = settings.load()
            self.cfg_loaded = time.monotonic()
        return self.cfg

    def _verdict(self) -> tuple[bool, str, str]:
        if self.child_status is not None or self.master < 0:
            return False, "exited", "the program has exited"
        return session.verdict(session.read(os.getpid()), self.keyboard, time.time(), self._settings())

    def _request(self, c: _Conn, req: dict) -> None:
        op = req.get("op")
        if op == "ping":
            self._reply(c, {"ok": True, "pid": os.getpid(), "child_pid": self.child_pid})
        elif op == "check":
            ok, reason, detail = self._verdict()
            self._reply(c, {"ok": ok, "reason": reason, "detail": detail})
        elif op == "status":
            ok, reason, detail = self._verdict()
            st = session.read(os.getpid()) or {}
            now = time.time()
            self._reply(c, {
                "ok": True, "pid": os.getpid(), "child_pid": self.child_pid,
                "child_alive": self.child_status is None, "can_type": ok, "reason": reason, "detail": detail,
                "keyboard_idle": round(now - self.keyboard.last_key, 1) if self.keyboard.last_key else None,
                "draft": self.keyboard.draft, "draft_negligible": self.keyboard.negligible(self._settings()),
                "turn": st.get("turn"), "started": st.get("started"),
                "dialogs": [session.describe(d) for d in session.open_dialogs(st, self.keyboard, now, self._settings())],
            })
        elif op == "type":
            self._type(c, req)
        else:
            self._reply(c, {"ok": False, "reason": "unknown-op"})

    def _type(self, c: _Conn, req: dict) -> None:
        ident = str(req.get("id") or "")
        if ident and ident in self.done:
            self._reply(c, dict(self.done[ident], duplicate=True))
            return
        if self.injecting:
            if ident and self.injecting["id"] == ident:
                self.injecting.setdefault("waiters", []).append(c)
            else:
                self._reply(c, {"ok": False, "reason": "typing", "detail": "another line is being typed"})
            return
        text = clean_text(req.get("text", ""))
        if not text:
            self._reply(c, {"ok": False, "reason": "empty"})
            return
        ok, reason, detail = self._verdict()
        if not ok:
            self._reply(c, {"ok": False, "reason": reason, "detail": detail})
            return
        cfg = self._settings()
        submit = req.get("submit", True) is not False
        delay = cfg["submit_delay"] if submit else 0.0
        self.injecting = {"id": ident, "conn": c, "chars": len(text), "submit": submit}
        if self.keyboard.draft and self.keyboard.negligible(cfg):
            # Stray keys in the prompt box (see session.negligible): clear the line
            # first, apart from the text, so the request isn't typed onto them.
            logs.log(f"cleared {self.keyboard.draft} stray characters from the prompt box before typing a request",
                     source="pty")
            self.keyboard.clear()
            self.to_child += _CLEAR_LINE
            self._to_child()
            self.injecting.update(phase="clear", text=text.encode("utf-8"), delay=delay,
                                  due=time.monotonic() + _CLEAR_GAP)
            return
        self.to_child += text.encode("utf-8")
        self._to_child()
        self.injecting.update(phase="text", due=time.monotonic() + delay)

    def _finish_injection(self) -> None:
        """Release the keys the user pressed meanwhile, and answer."""
        inj, self.injecting = self.injecting, None
        self.to_child += self.held_keys
        self.held_keys.clear()
        self._to_child()
        reply = {"ok": True, "typed": inj["chars"], "submitted": inj["submit"]}
        if inj["id"]:
            self.done[inj["id"]] = reply
            while len(self.done) > 256:
                self.done.popitem(last=False)
        for conn in [inj.get("conn")] + inj.get("waiters", []):
            self._reply(conn, reply)

    # ------------------------------------------------------------ cleanup

    def _cleanup(self) -> None:
        for c in list(self.conns):
            self._drop(c)
        if self.listen_sock is not None:
            with contextlib.suppress(OSError):
                self.listen_sock.close()
            with contextlib.suppress(OSError):
                if self.sock_path:
                    self.sock_path.unlink()
            try:
                info = json.loads(proxy_file().read_text(encoding="utf-8"))
                if info.get("pid") == os.getpid():
                    proxy_file().unlink()
            except (OSError, ValueError):
                pass
            session.remove(os.getpid())
            self._stop_listener()
        if self.lock_fd >= 0:
            os.close(self.lock_fd)
            self.lock_fd = -1
        if self.master >= 0:
            os.close(self.master)
            self.master = -1

    def _stop_listener(self) -> None:
        """Stop a listener serving this session (it would notice within a second anyway)."""
        try:
            info = json.loads((settings.listen_dir() / "listener.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if info.get("proxy_pid") == os.getpid() and isinstance(info.get("pid"), int) and info["pid"] > 1:
            with contextlib.suppress(OSError):
                os.kill(info["pid"], signal.SIGTERM)


def _session_leader(slave: int, argv: list[str], env: dict) -> None:
    """Body of the proxy's child: own the pty's session and run argv as its
    foreground job, the way a shell would. Never returns.

    The program could lead the session itself, but a session leader's process
    group is orphaned, and the kernel discards SIGTSTP sent to an orphaned
    group, so a program that implements Ctrl-Z by stopping itself with SIGTSTP
    (as Node programs do) would never stop. Here it runs in a process group of
    its own whose parent (this process) is in the same session, exactly as under
    an interactive shell. When it stops, this process stops too (SIGSTOP), which
    the proxy sees; when the proxy continues it, it hands the terminal back and
    continues the program. Signals are forwarded to the program's group, and the
    program's exit status (or killing signal) is reproduced as this process's own.
    """
    os.setsid()
    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    for fd in (0, 1, 2):
        os.dup2(slave, fd)
    if slave > 2:
        os.close(slave)
    os.closerange(3, 1 << 16)  # the proxy's sockets, lock and pipes stay with the proxy
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    child = os.fork()
    if child == 0:
        try:
            os.setpgid(0, 0)
            os.tcsetpgrp(0, os.getpid())
            for sig in _HANDLED + (signal.SIGPIPE, signal.SIGXFSZ, signal.SIGTTOU):
                signal.signal(sig, signal.SIG_DFL)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _HANDLED)
            os.execvpe(argv[0], argv, env)
        except BaseException as exc:  # noqa: BLE001
            with contextlib.suppress(OSError):
                os.write(2, f"hal-listen pty: cannot run {argv[0]}: {exc}\r\n".encode())
        os._exit(127)
    with contextlib.suppress(OSError):
        os.setpgid(child, child)
    with contextlib.suppress(OSError):
        os.tcsetpgrp(0, child)

    def forward(signum, frame):  # noqa: ARG001
        with contextlib.suppress(OSError):
            os.killpg(child, signum)

    for sig in _FORWARD:
        signal.signal(sig, forward)
    for sig in (signal.SIGWINCH, signal.SIGCHLD, signal.SIGCONT):
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _HANDLED)
    while True:
        try:
            _, status = os.waitpid(child, os.WUNTRACED)
        except ChildProcessError:
            os._exit(0)
        if os.WIFSTOPPED(status):
            os.kill(os.getpid(), signal.SIGSTOP)
            # Continued by the proxy (fg): give the terminal back and wake the program.
            with contextlib.suppress(OSError):
                os.tcsetpgrp(0, child)
            with contextlib.suppress(OSError):
                os.killpg(child, signal.SIGCONT)
            continue
        code = os.waitstatus_to_exitcode(status)
        if code < 0:
            signal.signal(-code, signal.SIG_DFL)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, [-code])
            os.kill(os.getpid(), -code)
            time.sleep(1)
            os._exit(128 - code)
        os._exit(code)


def exit_with(status: int) -> int:
    """Turn a wait status into this process's exit: same code, or the same signal."""
    code = os.waitstatus_to_exitcode(status)
    if code >= 0:
        return code
    sig = -code
    if sig not in (signal.SIGSTOP, signal.SIGTSTP, signal.SIGTTIN, signal.SIGTTOU):
        sys.stdout.flush()
        signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [sig])
        os.kill(os.getpid(), sig)
        time.sleep(0.2)
    return 128 + sig


def main(argv: list[str]) -> int:
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("usage: hal-listen pty -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    if not os.isatty(0):  # nothing to relay: just become the program
        os.execvp(argv[0], argv)
    try:
        proxy = Proxy(argv)
    except Exception as exc:  # never stand between the user and claude
        logs.log(f"proxy setup failed ({exc!r}); running {argv[0]} directly", source="pty")
        os.execvp(argv[0], argv)
    return exit_with(proxy.run())
