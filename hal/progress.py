"""Speaking while Claude works: the PreToolUse side of `hal hook`.

A turn that uses tools used to be silent between the acknowledgement and the
reply. Now, at each tool call:

- The line Claude wrote just before it ("Let me fetch the forecast.") is
  spoken, if it is short prose (at most progress_max_chars, whole sentences).
  It is read from the session transcript, which holds the text before the
  tool call is made.
- If there is no new line and nothing has been said for progress_cue_seconds,
  a short stock phrase is said instead ("Still working on it."), at most twice
  a turn.

Only a turn whose reply will be spoken counts: the prompt hook calls
start_turn() with the token stop_for_prompt() left in state/current, and
anything that replaces that token (the reply starting, `hal stop`, a newer
prompt, a spoken "stop") ends progress speech for the turn.

Progress lines play in the acknowledgement slot (speaker.say_progress), one
at a time and never over anything else; a newer line drops an older one that
has not started, and the reply waits for a line that is playing.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from . import config

TURN_FILE = config.STATE_DIR / "turn.json"
_TAIL_BYTES = 512 * 1024      # how much of the transcript's end is read
_RETRY_SECONDS = 0.4          # wait once for the transcript to catch up
_MAX_CUES = 2                 # stock phrases per turn
_MAX_TURN_AGE = 3 * 3600


def start_turn(token: str) -> None:
    """The prompt hook: a turn whose reply will be spoken has begun."""
    from . import speaker

    now = time.time()
    speaker._write(TURN_FILE, json.dumps(
        {"token": token, "started": now, "said_at": now, "spoken": [], "cues": 0, "seq": 0}))


def end_turn() -> None:
    TURN_FILE.unlink(missing_ok=True)


def _load_turn() -> dict | None:
    try:
        turn = json.loads(TURN_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(turn, dict) or not isinstance(turn.get("token"), str):
        return None
    try:
        for key in ("started", "said_at"):
            turn[key] = float(turn[key])
        turn["cues"] = int(turn.get("cues") or 0)
        turn["seq"] = int(turn.get("seq") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(turn.get("spoken"), list):
        turn["spoken"] = []
    return turn


def _timestamp(entry: dict) -> float:
    try:
        return datetime.fromisoformat(str(entry["timestamp"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def texts_since(transcript: str | Path, since: float) -> list[tuple[str, str]]:
    """(key, text) for each text block Claude wrote in the main conversation
    since the given time, oldest first, from the end of the transcript."""
    try:
        with open(transcript, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read()
    except (OSError, TypeError, ValueError):
        return []
    lines = data.split(b"\n")
    if size > _TAIL_BYTES:
        lines = lines[1:]  # the first line may be cut
    found = []
    for raw in lines:
        if b'"assistant"' not in raw or b'"text"' not in raw:
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant" or entry.get("isSidechain"):
            continue
        if _timestamp(entry) < since:
            continue
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    found.append((f"{entry.get('uuid') or _timestamp(entry)}:{i}", text))
    return found


def speech_for(text: str, max_chars: int) -> str:
    """What to say for a line written before a tool call: its opening
    paragraph, if that is prose, cut to whole sentences within max_chars.
    Empty if there is nothing suitable (a list, code, one long sentence)."""
    from .speaker import _lead_paragraph
    from .textclean import pronounce, split_sentences, to_speech

    lead = _lead_paragraph(text)
    if not lead:
        return ""
    kept = ""
    for sentence in split_sentences(to_speech(lead)):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    return pronounce(kept) if kept else ""


def on_tool(payload: dict, cfg: dict) -> str | None:
    """PreToolUse: speak Claude's latest line, or a stock phrase after a long
    silence. Returns a note for the log, or None when there is nothing to note."""
    from . import speaker

    if payload.get("agent_id"):
        return None  # a subagent's tool call
    if not cfg["speak_progress"] and not cfg["progress_cue_seconds"]:
        return None
    turn = _load_turn()
    if turn is None or time.time() - turn["started"] > _MAX_TURN_AGE:
        return None
    if speaker.current() != turn["token"]:
        return None  # the reply has started, or speech was stopped, or a newer prompt came
    transcript = payload.get("transcript_path")

    def unspoken(t: dict) -> list[tuple[str, str]]:
        if not cfg["speak_progress"] or not isinstance(transcript, str):
            return []
        done = set(t["spoken"])
        return [(k, s) for k, s in texts_since(transcript, t["started"] - 1.0) if k not in done]

    fresh = unspoken(turn)
    if not fresh and cfg["speak_progress"]:
        time.sleep(_RETRY_SECONDS)  # the hook is async; let the transcript catch up
        fresh = unspoken(_load_turn() or turn)

    with speaker._locked():
        turn = _load_turn()
        if turn is None or speaker._read(speaker.CURRENT_FILE) != turn["token"]:
            return None
        now = time.time()
        done = set(turn["spoken"])
        fresh = [(k, s) for k, s in fresh if k not in done]
        text, how = "", ""
        if fresh:
            turn["spoken"] = (turn["spoken"] + [k for k, _ in fresh])[-200:]
            text = speech_for(fresh[-1][1], cfg["progress_max_chars"])
            how = "line"
        cue_due = (cfg["progress_cue_seconds"] and turn["cues"] < min(_MAX_CUES, len(cfg["progress_phrases"]))
                   and now - turn["said_at"] >= cfg["progress_cue_seconds"] and not speaker.is_speaking())
        if not text and cue_due:
            text = cfg["progress_phrases"][turn["cues"]]
            turn["cues"] += 1
            how = "cue"
        if not text:
            speaker._write(TURN_FILE, json.dumps(turn))
            return "PreToolUse: Claude's line is not short prose; not spoken" if fresh else None
        turn["said_at"] = now
        turn["seq"] += 1
        seq = turn["seq"]
        speaker._write(TURN_FILE, json.dumps(turn))
    proc = speaker.say_progress(text, turn["token"], seq)
    if proc is None:
        return None
    what = "Claude's line" if how == "line" else "a progress cue"
    return f"PreToolUse: speaking {what} ({len(text)} chars, worker {proc.pid})"


def is_latest(token: str, seq: int) -> bool:
    """True if seq is still the newest progress line of the turn with this token."""
    turn = _load_turn()
    return turn is not None and turn["token"] == token and turn["seq"] == seq
