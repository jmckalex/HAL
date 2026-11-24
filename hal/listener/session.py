"""Session state: may text be typed into Claude Code right now? Standard library only.

Two sources of evidence decide it:

1. The session-state file, state/listen/session-<proxy pid>.json, kept by
   Claude Code hooks (`hal-listen hook`, see hookstate.py) through
   apply_event(). It records whether the session has started, whether a turn
   is running, and which dialogs are open:

    {"version": 1, "proxy_pid": 4242, "session_id": "...",
     "started": true,                 # SessionStart (or any turn event) seen
     "turn": "idle" | "busy" | "ended",
     "dialogs": [{"kind": "permission", "tool": "Bash", "key": "3f2a...",
                  "agent": null, "since": 1731700000.12, "enter": true}],
     "last_stop": 1731699990.0, "last_event": "PermissionRequest", "updated": ...}

   Dialog kinds and what opens / closes them:
     permission    PermissionRequest  (fires the moment the dialog is shown)
     question      PreToolUse AskUserQuestion
     plan          PreToolUse ExitPlanMode
     elicitation   Elicitation        closed by ElicitationResult
     notification  Notification permission_prompt / elicitation_dialog /
                   elicitation_url_dialog / agent_needs_input /
                   quota_auto_resume_stale (about 6 s late; the only signal
                   for a sandbox network-request prompt)
   permission/question/plan close on PostToolUse, PostToolUseFailure,
   PermissionDenied or PostToolBatch for the same tool call (matched on tool
   name and input, because PermissionRequest carries no tool_use_id). Every
   dialog closes at a turn boundary: UserPromptSubmit, Stop, StopFailure,
   SessionStart, SessionEnd, or Notification idle_prompt.

2. The real keyboard, as seen by the pty proxy (Keyboard below): after a
   dialog appeared, Esc or Ctrl-C closes it (and every older one), and Enter
   answers the oldest permission or notification dialog. A manual "No" fires
   no hook at all, and an approved tool fires PostToolUse only when it
   finishes, so without this a long-running approved command would hold
   spoken requests until it ended. Questions, plans and elicitations can take
   several Enters, so only Esc or their hook event closes those.

verdict() combines them. Typing is refused while a dialog is open, while
unsent text seems to sit in the prompt box (draft), and for a moment after
any keystroke. A busy turn is fine: Claude Code queues what is typed.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import settings

VERSION = 1
INTERACTIVE_TOOLS = {"AskUserQuestion": "question", "ExitPlanMode": "plan"}
_RESOLVING = ("permission", "question", "plan")
_NOTIFY_DIALOG = {"permission_prompt", "elicitation_dialog", "elicitation_url_dialog", "agent_needs_input",
                  "quota_auto_resume_stale"}
_NOTIFY_ELICITATION_DONE = {"elicitation_complete", "elicitation_response"}
KEY_GRACE = 0.25   # a key pressed within this long of a dialog appearing doesn't answer it


# ---------------------------------------------------------------- the file

def state_path(proxy_pid: int) -> Path:
    return settings.listen_dir() / f"session-{int(proxy_pid)}.json"


def empty(proxy_pid: int = 0) -> dict:
    return {"version": VERSION, "proxy_pid": proxy_pid, "session_id": None, "started": False,
            "turn": "idle", "dialogs": [], "last_stop": 0.0, "last_event": None, "updated": 0.0}


def read(proxy_pid: int) -> dict | None:
    try:
        data = json.loads(state_path(proxy_pid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextlib.contextmanager
def _locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def update(proxy_pid: int, payload: dict, now: float | None = None) -> dict:
    """Apply one hook event to the state file of proxy_pid, atomically."""
    path = state_path(proxy_pid)
    with _locked(path):
        state = apply_event(read(proxy_pid) or empty(proxy_pid), payload, time.time() if now is None else now)
        state["proxy_pid"] = int(proxy_pid)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
    return state


def remove(proxy_pid: int) -> None:
    for p in (state_path(proxy_pid), state_path(proxy_pid).with_suffix(".lock")):
        with contextlib.suppress(OSError):
            p.unlink()


# ---------------------------------------------------------------- the reducer

def tool_key(tool_name, tool_input) -> str:
    try:
        blob = json.dumps(tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(tool_input)
    return hashlib.sha1(f"{tool_name}\0{blob}".encode("utf-8", "replace")).hexdigest()[:16]


def _normalize(state: dict) -> dict:
    st = empty()
    if isinstance(state, dict):
        st.update({k: v for k, v in state.items() if k in st})
    if not isinstance(st["dialogs"], list):
        st["dialogs"] = []
    st["dialogs"] = [d for d in st["dialogs"] if isinstance(d, dict) and isinstance(d.get("since"), (int, float))]
    return st


def _open(st: dict, kind: str, now: float, *, tool=None, key=None, agent=None, enter=True) -> None:
    st["dialogs"].append({"kind": kind, "tool": tool, "key": key, "agent": agent, "since": now, "enter": enter})


def _resolve(st: dict, tool, tool_input, agent) -> None:
    """A tool call finished or was denied: close its dialog(s)."""
    key = tool_key(tool, tool_input)
    dialogs = st["dialogs"]
    matched = [d for d in dialogs if d["kind"] in _RESOLVING and d.get("key") == key and d.get("agent") == agent]
    if not matched:  # input differs slightly between events: fall back to the oldest of that tool
        same_tool = [d for d in dialogs if d["kind"] in _RESOLVING and d.get("tool") == tool and d.get("agent") == agent]
        matched = sorted(same_tool, key=lambda d: d["since"])[:1]
    st["dialogs"] = [d for d in dialogs if not any(d is m for m in matched)]


def apply_event(state: dict, payload: dict, now: float) -> dict:
    """The state after one hook event. Pure: state is not modified."""
    st = _normalize(json.loads(json.dumps(state)) if state else {})
    if not isinstance(payload, dict):
        return st
    event = payload.get("hook_event_name")
    agent = payload.get("agent_id") or None
    tool = payload.get("tool_name")
    st["updated"] = now
    st["last_event"] = event
    if isinstance(payload.get("session_id"), str):
        st["session_id"] = payload["session_id"]

    if event == "SessionStart":
        st["started"] = True
        if payload.get("source") != "compact":   # compaction can happen mid-turn
            st["turn"] = "idle"
            st["dialogs"] = []
    elif event == "UserPromptSubmit":
        st["started"] = True
        if not agent:
            st["turn"] = "busy"
            st["dialogs"] = []
    elif event in ("Stop", "StopFailure"):
        if not agent:
            st["started"] = True
            st["turn"] = "idle"
            st["dialogs"] = []
            st["last_stop"] = now
    elif event == "SessionEnd":
        st["dialogs"] = []
        if payload.get("reason") not in ("clear", "resume"):  # those are followed by SessionStart
            st["turn"] = "ended"
    elif event == "PermissionRequest":
        _open(st, "permission", now, tool=tool, key=tool_key(tool, payload.get("tool_input")), agent=agent)
    elif event == "PreToolUse" and tool in INTERACTIVE_TOOLS:
        _open(st, INTERACTIVE_TOOLS[tool], now, tool=tool, key=tool_key(tool, payload.get("tool_input")),
              agent=agent, enter=False)
    elif event == "Elicitation":
        ident = payload.get("elicitation_id") or payload.get("mcp_server_name")
        _open(st, "elicitation", now, tool=payload.get("mcp_server_name"), key=f"elicitation:{ident}",
              agent=agent, enter=False)
    elif event == "ElicitationResult":
        ident = payload.get("elicitation_id") or payload.get("mcp_server_name")
        same = [d for d in st["dialogs"] if d["kind"] == "elicitation" and d.get("key") == f"elicitation:{ident}"]
        drop = same or [d for d in st["dialogs"] if d["kind"] == "elicitation"][:1]
        st["dialogs"] = [d for d in st["dialogs"] if not any(d is x for x in drop)]
    elif event == "Notification":
        kind = payload.get("notification_type")
        if kind in _NOTIFY_DIALOG:
            _open(st, "notification", now, tool=kind, agent=agent)
        elif kind == "idle_prompt":
            st["turn"] = "idle"
            st["dialogs"] = []
        elif kind in _NOTIFY_ELICITATION_DONE:
            st["dialogs"] = [d for d in st["dialogs"] if d["kind"] != "elicitation"
                             and not (d["kind"] == "notification" and str(d.get("tool", "")).startswith("elicitation"))]
    elif event in ("PostToolUse", "PostToolUseFailure", "PermissionDenied"):
        _resolve(st, tool, payload.get("tool_input"), agent)
    elif event == "PostToolBatch":
        for call in payload.get("tool_calls") or []:
            if isinstance(call, dict):
                _resolve(st, call.get("tool_name"), call.get("tool_input"), agent)
    return st


# ---------------------------------------------------------------- the keyboard

_CSI = re.compile(rb"\x1b\[([0-9;:?<>=]*)([\x20-\x2f]*)([\x40-\x7e])")
_SS3 = re.compile(rb"\x1bO[\x20-\x7e]")
_PASTE_START, _PASTE_END = b"\x1b[200~", b"\x1b[201~"


class Keyboard:
    """What the person at the real keyboard has done, as far as it matters here.

    draft is a rough count of characters typed since the last Enter (or Ctrl-C,
    Ctrl-U): positive means text may be sitting unsent in the prompt box.
    Arrow keys, history recall and Ctrl-W aren't modelled; see LISTENER.md."""

    def __init__(self):
        self.last_key = 0.0
        self.last_draft_key = 0.0
        self.draft = 0
        self.enters: collections.deque = collections.deque(maxlen=64)    # times of Enter
        self.escapes: collections.deque = collections.deque(maxlen=64)   # times of Esc or Ctrl-C
        self._paste = False

    def _typed(self, n: int, now: float) -> None:
        if n > 0:
            self.draft += n
            self.last_draft_key = now

    def _key(self, name: str, now: float) -> None:
        if name == "enter":
            self.enters.append(now)
            self.draft = 0
        elif name == "newline":
            self._typed(1, now)
        elif name in ("esc", "ctrl-c"):
            self.escapes.append(now)
            if name == "ctrl-c":
                self.draft = 0
        elif name == "ctrl-u":
            self.draft = 0
        elif name == "backspace":
            self.draft = max(0, self.draft - 1)

    def feed(self, data: bytes, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if not data:
            return
        self.last_key = now
        if data == b"\x1b":
            self._key("esc", now)
            return
        i, n = 0, len(data)
        while i < n:
            if self._paste:
                end = data.find(_PASTE_END, i)
                chunk = data[i:] if end < 0 else data[i:end]
                self._typed(sum(1 for b in chunk if b >= 0x20 and not 0x80 <= b <= 0xBF), now)
                if end < 0:
                    return
                self._paste = False
                i = end + len(_PASTE_END)
                continue
            b = data[i]
            if b == 0x1B:
                if data.startswith(_PASTE_START, i):
                    self._paste = True
                    i += len(_PASTE_START)
                    continue
                m = _CSI.match(data, i)
                if m:
                    self._csi(m.group(1), m.group(3), now)
                    i = m.end()
                    continue
                m = _SS3.match(data, i)
                if m:
                    i = m.end()
                    continue
                if i + 1 < n and data[i + 1] in (0x0D, 0x0A):   # Meta/Shift+Enter: a newline in the box
                    self._key("newline", now)
                    i += 2
                    continue
                if i + 1 >= n:
                    self._key("esc", now)
                i += 2   # Alt+key
                continue
            if b in (0x0D, 0x0A):
                self._key("enter", now)
            elif b == 0x03:
                self._key("ctrl-c", now)
            elif b == 0x15:
                self._key("ctrl-u", now)
            elif b in (0x7F, 0x08):
                self._key("backspace", now)
            elif b >= 0x20 and not 0x80 <= b <= 0xBF:
                self._typed(1, now)
            i += 1

    def _csi(self, params: bytes, final: bytes, now: float) -> None:
        """Keys reported with the kitty keyboard protocol (CSI code;mods u) or
        xterm modifyOtherKeys (CSI 27;mods;code ~)."""
        parts = [p.split(b":")[0] for p in params.split(b";")] if params else []
        try:
            nums = [int(p) if p else 1 for p in parts]
        except ValueError:
            return
        if final == b"u" and nums:
            code, mods = nums[0], (nums[1] if len(nums) > 1 else 1)
        elif final == b"~" and len(nums) == 3 and nums[0] == 27:
            mods, code = nums[1], nums[2]
        else:
            return
        shift, ctrl = bool((mods - 1) & 1), bool((mods - 1) & 4)
        if code == 13:
            self._key("newline" if (shift or mods > 1) else "enter", now)
        elif code == 27:
            self._key("esc", now)
        elif code in (127, 8):
            self._key("backspace", now)
        elif ctrl and code == 99:
            self._key("ctrl-c", now)
        elif ctrl and code == 117:
            self._key("ctrl-u", now)
        elif 32 <= code and not ctrl:
            self._typed(1, now)


# ---------------------------------------------------------------- the verdict

def open_dialogs(state: dict | None, kb: Keyboard | None, now: float, cfg: dict) -> list[dict]:
    """Dialogs still open, after taking the real keyboard into account."""
    if not state:
        return []
    dialogs = sorted((d for d in _normalize(state)["dialogs"] if now - d["since"] <= cfg["dialog_max_age"]),
                     key=lambda d: d["since"])
    if kb is None or not dialogs:
        return dialogs
    for esc in kb.escapes:
        dialogs = [d for d in dialogs if not d["since"] + KEY_GRACE <= esc]
    enters = sorted(kb.enters)
    remaining = []
    for d in dialogs:
        if d.get("enter", True):
            used = next((e for e in enters if e >= d["since"] + KEY_GRACE), None)
            if used is not None:
                if d["kind"] != "notification":
                    enters.remove(used)
                continue
        remaining.append(d)
    return remaining


def describe(dialog: dict) -> str:
    what = dialog.get("tool") or ""
    return f"{dialog['kind']}{': ' + what if what else ''}"


def verdict(state: dict | None, kb: Keyboard | None, now: float, cfg: dict) -> tuple[bool, str, str]:
    """(ok, reason, detail). reason is "ok", "no-session", "ended", "dialog", "draft" or "keyboard"."""
    if state is None or not state.get("started"):
        if cfg["require_session_state"]:
            return False, "no-session", "the session's hooks have not reported yet (SessionStart)"
        state = state or empty()
    if state.get("turn") == "ended":
        return False, "ended", "the Claude Code session ended"
    dialogs = open_dialogs(state, kb, now, cfg)
    if dialogs:
        return False, "dialog", describe(dialogs[0])
    if kb is not None:
        if kb.draft > 0 and now - kb.last_draft_key < cfg["draft_hold_seconds"]:
            return False, "draft", f"about {kb.draft} unsent characters in the prompt box"
        if now - kb.last_key < cfg["keyboard_quiet"]:
            return False, "keyboard", f"a key was pressed {now - kb.last_key:.1f}s ago"
    return True, "ok", ""
