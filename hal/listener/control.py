"""Client for the pty proxy's control socket. Standard library only."""

from __future__ import annotations

import json
import os
import socket
import uuid

from . import ptyproxy


class Unreachable(Exception):
    """No proxy is serving, or it did not answer."""


def find_proxy() -> dict | None:
    """The registered proxy, if its process is alive: {"pid", "socket", "child_pid", ...}."""
    try:
        info = json.loads(ptyproxy.proxy_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = info.get("pid") if isinstance(info, dict) else None
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return info


def new_id() -> str:
    return uuid.uuid4().hex


class ProxyClient:
    def __init__(self, socket_path: str, timeout: float = 5.0):
        self.path = socket_path
        self.timeout = timeout

    @classmethod
    def discover(cls, timeout: float = 5.0) -> "ProxyClient":
        info = find_proxy()
        if not info or not info.get("socket"):
            raise Unreachable("no Claude Code session is running under the pty proxy")
        return cls(info["socket"], timeout)

    def request(self, obj: dict) -> dict:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect(self.path)
                s.sendall((json.dumps(obj) + "\n").encode())
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as exc:
            raise Unreachable(str(exc)) from exc
        try:
            reply = json.loads(buf.split(b"\n", 1)[0].decode())
        except ValueError as exc:
            raise Unreachable(f"bad reply {buf[:80]!r}") from exc
        return reply if isinstance(reply, dict) else {"ok": False, "reason": "bad-reply"}

    def ping(self) -> dict:
        return self.request({"op": "ping"})

    def check(self) -> dict:
        return self.request({"op": "check"})

    def status(self) -> dict:
        return self.request({"op": "status"})

    def type(self, text: str, ident: str | None = None, submit: bool = True) -> dict:
        """Type text and press Enter. Safe to retry with the same ident: the
        proxy types each id at most once."""
        req = {"op": "type", "id": ident or new_id(), "text": text, "submit": submit}
        try:
            return self.request(req)
        except Unreachable:
            # One retry with the same id: if the first attempt did type, the
            # proxy answers from its record instead of typing again.
            return self.request(req)
