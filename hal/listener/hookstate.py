"""`hal-listen hook`: Claude Code hook entry point that keeps the session-state file.

Register it for SessionStart, UserPromptSubmit, Stop, StopFailure,
SessionEnd, PermissionRequest, PreToolUse (AskUserQuestion|ExitPlanMode),
PostToolUse, PostToolUseFailure, PostToolBatch, PermissionDenied,
Notification, Elicitation and ElicitationResult (LISTENER.md has the exact
settings.json block).

It acts only inside a session started under the pty proxy, which puts
HAL_LISTEN_PROXY=<proxy pid> in claude's environment (and so in every hook's);
anywhere else it exits at once. It never prints (Claude Code would show the
output to Claude), never blocks for long and always exits 0. Standard
library only; problems go to state/listen.log. Nothing about prompts or tool
inputs is logged: only the event name and the resulting state.
"""

from __future__ import annotations

import json
import os
import select
import sys
import time

_STDIN_TIMEOUT = 2.0
_STDIN_LIMIT = 16 * 1024 * 1024


def _read_stdin() -> bytes:
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return b""
    if os.isatty(fd):
        return b""
    chunks, total = [], 0
    deadline = time.monotonic() + _STDIN_TIMEOUT
    while total < _STDIN_LIMIT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        data = os.read(fd, 1 << 16)
        if not data:
            break
        chunks.append(data)
        total += len(data)
    return b"".join(chunks)


def main() -> int:
    proxy = os.environ.get("HAL_LISTEN_PROXY", "")
    if not proxy.isdigit():
        return 0
    try:
        from . import logs, session

        raw = _read_stdin()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace")) if raw.strip() else None
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            logs.log(f"hook: unreadable event ({len(raw)} bytes)", source="hook")
            return 0
        st = session.update(int(proxy), payload)
        event = payload.get("hook_event_name")
        if event not in ("PostToolUse", "PostToolUseFailure", "PostToolBatch") or st["dialogs"]:
            extra = f" {payload.get('notification_type')}" if event == "Notification" else ""
            extra += f" {payload.get('tool_name')}" if payload.get("tool_name") else ""
            logs.log(f"hook: {event}{extra} -> turn={st['turn']} dialogs="
                     f"{[session.describe(d) for d in st['dialogs']]}", source="hook")
    except Exception as exc:  # a hook must never fail
        try:
            from . import logs

            logs.log(f"hook: error {exc!r}", source="hook")
        except Exception:
            pass
    return 0
