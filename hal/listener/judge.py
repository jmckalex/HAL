"""Ask the front desk whether a turn is finished. Standard library only.

The front desk (hal/desk.py) keeps a warm Claude Haiku for this, apart from
its other two. One request per connection on its socket:

    {"op": "turn", "text": TEXT, "timeout": SECONDS}
        -> {"kind": "complete" | "incomplete" | "unavailable", "seconds", "why"}

Each request runs in a thread of its own, so the listening loop never waits:
the turn-taker polls the handle. A desk that isn't running, is too old to
know "turn", or doesn't answer in time leaves the handle done with kind None,
and the turn falls back to the rules and a timer.

Only the text of a turn that is addressed to Claude (and so would be typed
into the session anyway) is ever passed here; turns.TurnTaker is the only
caller.
"""

from __future__ import annotations

import json
import socket
import threading
import time

from . import settings


def socket_path() -> str:
    """The desk's socket for the current state directory (the same rule as hal.desk)."""
    try:
        from hal import desk

        return desk.socket_path(settings.state_dir())
    except Exception:
        return str(settings.state_dir() / "desk.sock")


class Handle:
    def __init__(self):
        self.done = False
        self.kind: str | None = None     # "complete" | "incomplete" | None
        self.why = ""
        self.seconds = 0.0
        self._event = threading.Event()

    def _finish(self, kind: str | None, why: str, started: float) -> None:
        self.kind, self.why = kind, why
        self.seconds = time.monotonic() - started
        self.done = True
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


def ask(text: str, timeout: float, path: str | None = None) -> tuple[str | None, str]:
    """One judgement, blocking: (kind or None, why)."""
    path = path or socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            s.connect(path)
            s.settimeout(1.0)
            s.sendall((json.dumps({"op": "turn", "text": text, "timeout": timeout}) + "\n").encode())
            deadline = time.monotonic() + timeout + 0.5
            buf = b""
            while b"\n" not in buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, "no answer in time"
                s.settimeout(remaining)
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except (FileNotFoundError, ConnectionRefusedError):
        return None, "the front desk is not running"
    except socket.timeout:
        return None, "no answer in time"
    except OSError as exc:
        return None, f"connection: {exc}"
    try:
        reply = json.loads(buf.split(b"\n", 1)[0] or b"null")
    except ValueError:
        return None, "unreadable answer"
    if not isinstance(reply, dict):
        return None, "no answer"
    kind = reply.get("kind")
    if kind in ("complete", "incomplete"):
        return kind, f"{reply.get('seconds', '?')}s"
    if "error" in reply:   # a desk from before the "turn" request
        return None, f"the front desk can't judge turns ({reply['error']}); restart it"
    return None, str(reply.get("why") or f"answer {kind!r}")


class DeskJudge:
    """callable(text) -> Handle, answered in the background."""

    def __init__(self, timeout: float, path_fn=socket_path):
        self.timeout = timeout
        self.path_fn = path_fn

    def __call__(self, text: str) -> Handle:
        handle = Handle()
        started = time.monotonic()

        def run():
            try:
                kind, why = ask(text, self.timeout, self.path_fn())
            except Exception as exc:  # never let a thread die silently with the handle open
                kind, why = None, repr(exc)
            handle._finish(kind, why, started)

        threading.Thread(target=run, name="turn-judge", daemon=True).start()
        return handle
