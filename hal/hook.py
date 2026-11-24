"""Claude Code hook handler: `hal hook`, with the event JSON on stdin.

    Stop              speak the reply: a short spoken summary if it is long,
                      the reply itself if it is short (or always, in "full" mode)
    UserPromptSubmit  stop any speech, speak a brief acknowledgement, and tell
                      Claude what the user heard if it was not the whole reply
    PreToolUse        (async) speak the line Claude wrote before the tool call,
                      or a stock progress phrase after a long silence
                      (hal/progress.py)
    anything else     ignored

Contract with Claude Code: never fail, never hang. This always returns 0 and
logs problems to state/hal.log. On UserPromptSubmit the only stdout is the
note for Claude (Claude Code adds that hook's stdout to Claude's context),
which is why that hook is not async, or, when the front desk has answered a
social remark itself, the JSON that blocks the prompt. It returns as soon as
the speech worker is spawned; synthesis and playback happen in the detached
worker, so the audio outlives this process.

With front_desk on (hal/desk.py), a prompt is handed to the front desk
instead of being met with a canned acknowledgement. A short prompt (at most
desk_wait_words words) waits up to desk_wait_seconds for its decision: a
social reply, spoken by the desk, blocks the prompt (desk_block) so Claude
takes no turn; anything else goes through while the desk speaks an
acknowledgement in line with it. Longer prompts don't wait. If the desk is
not running, it is started, and this prompt gets the canned acknowledgement;
so does any prompt the desk can't handle in time.

A request typed by the listener (spoken, LISTENER.md) starts with a label,
"[spoken 21:14] ...", and may carry marks such as "[?]" (hal/spoken.py). Every
test here judges the words alone (spoken.plain()), so a spoken request is
treated exactly as the same words typed, and the label never makes a prompt
look like one Claude Code started itself.
"""

from __future__ import annotations

import json
import os
import re
import select
import sys
import time
import traceback

from . import config, spoken

_STDIN_TIMEOUT = 2.0
_STDIN_LIMIT = 8 * 1024 * 1024


def _read_stdin() -> bytes:
    """Read stdin to EOF, but never hang: give up after a short timeout."""
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
        data = os.read(fd, 65536)
        if not data:
            break
        chunks.append(data)
        total += len(data)
    return b"".join(chunks)


def _non_interactive() -> bool:
    """`claude -p` and Agent SDK sessions run with CLAUDE_CODE_ENTRYPOINT=sdk-*;
    the interactive CLI uses "cli"."""
    return os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").startswith("sdk")


SKIP_NEXT_FILE = config.STATE_DIR / "skip_next_reply"
LAST_PROMPT_FILE = config.STATE_DIR / "last_prompt.json"
DESK_EXCHANGES_FILE = config.STATE_DIR / "desk_exchanges.json"  # answered by the desk, not yet told to Claude
_SKIP_VALID_SECONDS = 300
_CONTEXT_MAX_AGE = 6 * 3600
_MAX_DEFERRED = 20

# A prompt made only of these words, with at least one trigger, is a request
# to stop talking: "please stop reading", "that's enough, thank you, HAL".
_STOP_TRIGGERS = {"stop", "quiet", "silence", "enough", "hush", "shush"}
_STOP_FILLER = {
    "please", "hal", "claude", "reading", "talking", "speaking", "be", "that's", "thats", "that", "is",
    "thank", "thanks", "you", "ok", "okay", "now", "can", "could", "would", "will", "just",
    "it", "there", "hello", "hey", "hi", "shut", "up", "aloud", "out", "loud", "for", "a", "moment",
    "i", "said", "the", "voice", "right", "all", "fine", "good", "very",
}
# Turns Claude Code starts on its own (task notifications, messages from other
# sessions, scheduled prompts) also fire UserPromptSubmit; they are not the user.
_NOT_THE_USER = ("<", "[", "another claude session", "system notification")


# With the front desk on, a short remark made only of these words may be the
# answer to a question Claude just asked ("OK", "sounds good", "yes, go ahead",
# "the second one", "read that again"), so it never goes to the desk, which
# can't see the question: it is acknowledged with BRIEF_ANSWER_ACK and goes
# straight through to Claude.
BRIEF_ANSWER_ACK = "Very well."
_ANSWER_WORDS = {
    "yes", "yeah", "yep", "yup", "aye", "no", "nope", "nah", "ok", "okay", "sure", "fine", "right", "alright",
    "all", "good", "great", "perfect", "excellent", "agreed", "correct", "exactly", "indeed", "absolutely",
    "definitely", "certainly", "of", "course", "hmm", "hm", "hmmm", "mm", "mmm", "mhm", "sounds", "go", "ahead",
    "do", "it", "that", "this", "proceed", "continue", "carry", "on", "true", "cool", "nice", "lovely",
    "brilliant", "wonderful", "makes", "sense", "i", "see", "got", "understood", "noted", "first", "second",
    "third", "fourth", "fifth", "last", "one", "two", "three", "four", "five", "both", "neither", "either",
    "option", "number", "it's", "that's", "thats", "its", "let's", "lets", "why", "what", "how", "when", "where",
    "who", "which", "not", "really", "seriously", "again", "read", "repeat", "more", "back", "undo", "same",
    "too", "also", "instead", "wait", "hold", "maybe", "perhaps", "later", "is", "was",
}
_ANSWER_FILLER = {"please", "claude", "hal", "thanks", "thank", "you", "the", "a", "then", "so", "now", "just",
                  "well", "very", "oh", "ah", "uh", "um", "er"}


def is_brief_answer(prompt: str) -> bool:
    words = re.findall(r"[a-z']+", spoken.plain(prompt).lower())
    return (0 < len(words) <= 6 and all(w in _ANSWER_WORDS or w in _ANSWER_FILLER for w in words)
            and any(w in _ANSWER_WORDS for w in words))


def is_stop_request(prompt: str) -> bool:
    words = re.findall(r"[a-z']+", spoken.plain(prompt).lower())
    if not words or len(words) > 10:
        return False
    return any(w in _STOP_TRIGGERS for w in words) and all(w in _STOP_TRIGGERS or w in _STOP_FILLER for w in words)


# A short remark that is only thanks, praise or a greeting needs no "I'll look
# into that": the reply itself follows in a moment.
_SOCIAL = re.compile(
    r"\b(thank(s| you)|well done|good (work|job|morning|afternoon|evening|night)|excellent|perfect(ly)?|"
    r"wonderful|brilliant|marvellous|splendid|lovely|great|nicely done|much appreciated|appreciate|"
    r"hello|goodbye|good ?bye|you're welcome|no problem|never mind)\b", re.I)
_ASKS_FOR_SOMETHING = re.compile(
    r"\?|\b(please|can you|could you|would you|will you|what|why|how|when|where|which|who|tell me|show me|"
    r"open|run|check|find|make|build|add|fix|change|explain|let's|let us)\b", re.I)


def is_social_remark(prompt: str) -> bool:
    prompt = spoken.plain(prompt)
    words = re.findall(r"[A-Za-z']+", prompt)
    return 0 < len(words) <= 30 and bool(_SOCIAL.search(prompt)) and not _ASKS_FOR_SOMETHING.search(prompt)


def is_user_prompt(prompt: str) -> bool:
    if spoken.is_spoken(prompt):   # the listener's label starts with "[", but the words are the user's
        return bool(spoken.plain(prompt).strip())
    head = prompt.strip().lower()
    return bool(head) and not head.startswith(_NOT_THE_USER)


def _spoken_context() -> str:
    """Tell Claude what the user actually heard, when it wasn't the whole reply."""
    from . import speaker

    last = speaker.last_spoken()
    if not last or last.get("reported") or not last.get("text"):
        return ""
    if time.time() - float(last.get("time") or 0) > _CONTEXT_MAX_AGE:
        return ""
    speaker.mark_last_spoken_reported()
    if last.get("how") == "summary":
        lead = "Your previous reply was not read aloud in full. The user heard this spoken summary of it instead"
    else:
        lead = "Only the opening of your previous reply was read aloud. The user heard"
    return (f"[HAL voice] {lead}: \"{last['text']}\" The full reply is on their screen. "
            "If they refer to something that was said, or ask you to elaborate on a point, it is from this spoken version.")


def _defer_exchange(prompt: str, reply: str) -> None:
    """Keep an exchange the front desk handled alone, for Claude's next turn."""
    from . import speaker

    try:
        items = json.loads(DESK_EXCHANGES_FILE.read_text(encoding="utf-8"))
        items = items if isinstance(items, list) else []
    except (OSError, ValueError):
        items = []
    items.append({"time": time.time(), "user": " ".join(prompt.split()), "reply": reply})
    speaker._write(DESK_EXCHANGES_FILE, json.dumps(items[-_MAX_DEFERRED:]))


def _deferred_context() -> str:
    """Tell Claude, once, about exchanges the front desk handled alone since its last turn."""
    try:
        items = json.loads(DESK_EXCHANGES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    DESK_EXCHANGES_FILE.unlink(missing_ok=True)
    said = []
    for item in items if isinstance(items, list) else []:
        try:
            if time.time() - float(item["time"]) <= _CONTEXT_MAX_AGE:
                said.append(f"User: \"{item['user']}\" Front desk: \"{item['reply']}\"")
        except (KeyError, TypeError, ValueError):
            continue
    if not said:
        return ""
    return ("[HAL voice] Before this prompt the user said things that never reached you, because the voice "
            "front desk (a small model that speaks aloud as you) answered them by itself. In order: "
            + " ".join(said) + " Treat these exchanges as part of your conversation with the user.")


def _consult_desk(prompt: str, token: str, cfg: dict) -> dict:
    """Hand the prompt to the front desk; see desk.consult(). Adds "note" for the log."""
    from . import desk

    wait = len(prompt.split()) <= cfg["desk_wait_words"]
    started = time.monotonic()
    result = desk.consult(prompt, token, wait, cfg["desk_wait_seconds"], cfg["acknowledge"])
    ms = (time.monotonic() - started) * 1000
    kind = result["kind"]
    if kind == "down":
        note = "front desk not running" + ("; starting it" if desk.spawn() else "")
    elif kind == "unavailable":
        note = f"front desk unavailable after {ms:.0f} ms ({result.get('why')})"
    elif kind == "queued":
        note = "long prompt: front desk will acknowledge it"
    else:
        note = f"front desk decided in {ms:.0f} ms"
    result["note"] = "; " + note
    return result


def _on_prompt(payload: dict, cfg: dict) -> str:
    from . import progress, speaker

    prompt = payload.get("prompt")
    prompt = prompt if isinstance(prompt, str) else ""
    if not is_user_prompt(prompt):
        return f"UserPromptSubmit: not typed or spoken by the user ({prompt.strip()[:24]!r}); left alone"
    progress.end_turn()
    stopped, token = speaker.stop_for_prompt()
    state = "stopped speech" if stopped else "nothing playing"
    # A spoken request is judged by its words alone; Claude still sees the label and marks.
    said = spoken.plain(prompt)
    if is_stop_request(said):
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        SKIP_NEXT_FILE.write_text(str(time.time()))
        return f"UserPromptSubmit: stop request; {state}; the reply to it will not be spoken"
    SKIP_NEXT_FILE.unlink(missing_ok=True)
    command = prompt.lstrip().startswith(("/", "!", "#"))
    if not command:
        # Remembered so the spoken summary of the reply can lead with what answers it.
        try:
            LAST_PROMPT_FILE.write_text(json.dumps({"text": said.strip()[:2000], "time": time.time()}))
        except OSError:
            pass
    desk = {"kind": "off", "note": ""}
    if cfg["front_desk"] and not command:
        if is_brief_answer(said):
            desk = {"kind": "brief", "note": "; brief answer: straight to Claude"}
        else:
            desk = _consult_desk(said, token, cfg)
    if desk["kind"] == "social" and cfg["desk_block"]:
        # The desk is saying its reply; Claude takes no turn, and hears of it next time.
        _defer_exchange(said, desk["text"])
        sys.stdout.write(json.dumps({"decision": "block",
                                     "reason": f"Answered aloud by the front desk: \"{desk['text']}\""}) + "\n")
        sys.stdout.flush()
        return f"UserPromptSubmit: {state}{desk['note']}; it answered {desk['text']!r}; prompt blocked"
    context, deferred = _spoken_context(), _deferred_context()
    notes = [note for note in (context, deferred) if note]
    if desk["kind"] == "social":
        notes.append(f"[HAL voice] The voice front desk (a small model that speaks aloud as you) has already "
                     f"answered this remark aloud: \"{desk['text']}\" Your reply will not be read aloud. "
                     "Reply with at most one brief line.")
        SKIP_NEXT_FILE.write_text(str(time.time()))
    if notes:
        sys.stdout.write("\n".join(notes) + "\n")
        sys.stdout.flush()
    acked = ""
    if desk["kind"] == "social":
        acked = f"; it answered {desk['text']!r}; Claude told, its reply will not be spoken"
    elif desk["kind"] in ("ack", "queued"):
        pass
    elif is_social_remark(said):
        acked = "; thanks or a greeting, so no acknowledgement"
    elif cfg["acknowledge"] and not command:
        phrase = BRIEF_ANSWER_ACK if desk["kind"] == "brief" else speaker.choose_ack_phrase(cfg)
        speaker.ack(phrase)
        acked = f"; acknowledged ({phrase!r})"
    if desk["kind"] != "social" and not command:
        progress.start_turn(token)  # its reply will be spoken: so may lines written while it works
    told = "; told Claude what was spoken" if context else ""
    told += "; told Claude of earlier front-desk exchanges" if deferred else ""
    heard = "; spoken" if spoken.is_spoken(prompt) else ""
    return f"UserPromptSubmit: {state}{desk['note']}{acked}{told}{heard}"


def handle(payload: dict) -> str | None:
    """Act on one hook event; returns a short description for the log (None: nothing to log)."""
    from . import speaker  # deferred: keeps a malformed-input exit as cheap as possible

    event = payload.get("hook_event_name")
    if event not in ("Stop", "UserPromptSubmit", "PreToolUse"):
        return f"ignored event {event!r}"
    if config.is_muted():
        return None if event == "PreToolUse" else f"{event}: muted"
    cfg = config.load_config()
    if not config.is_voice_session(cfg):
        # Not the master session: neither speak nor silence the one that is.
        return None if event == "PreToolUse" else f"{event}: not the voice session"
    if event == "UserPromptSubmit":
        return _on_prompt(payload, cfg)
    if event == "PreToolUse":
        if _non_interactive() and not cfg["speak_in_print_mode"]:
            return None
        from . import progress

        return progress.on_tool(payload, cfg)

    # Stop. Subagents report through SubagentStop, which isn't registered;
    # this also skips agent-team teammates, whose events carry an agent_id.
    if payload.get("agent_id"):
        return "Stop: skipped (agent_id present)"
    from . import progress

    progress.end_turn()
    if _non_interactive() and not cfg["speak_in_print_mode"]:
        return "Stop: skipped (non-interactive session)"
    try:
        asked_to_stop = time.time() - float(SKIP_NEXT_FILE.read_text()) < _SKIP_VALID_SECONDS
    except (OSError, ValueError):
        asked_to_stop = False
    if asked_to_stop:
        SKIP_NEXT_FILE.unlink(missing_ok=True)
        return "Stop: silent (reply to a stop request)"
    message = payload.get("last_assistant_message")
    if not isinstance(message, str) or not message.strip():
        return "Stop: no last_assistant_message"

    from .textclean import speakable, to_speech

    # `touch state/full_next` (Claude does this when the user asks to be talked
    # through something) makes this one reply be read in full, not summarised.
    full_next = config.STATE_DIR / "full_next"
    read_in_full = full_next.exists()
    full_next.unlink(missing_ok=True)
    if cfg["spoken_reply"] in ("lead", "summary") and not read_in_full:
        if not to_speech(message):
            return f"Stop: nothing speakable in {len(message)} chars"
        proc = speaker.speak(message, kind="reply")
        return f"Stop: speaking a reply of {len(message)} chars, summarised if long (worker {proc.pid if proc else '-'})"
    spoken = speakable(message, cfg["max_chars"], cfg["overflow_message"])
    if not spoken:
        return f"Stop: nothing speakable in {len(message)} chars"
    proc = speaker.speak(spoken)
    return f"Stop: speaking {len(spoken)} of {len(message)} chars (worker {proc.pid if proc else '-'})"


def main() -> int:
    started = time.monotonic()
    try:
        raw = _read_stdin()
        if not raw.strip():
            config.log("hook: empty stdin")
            return 0
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            config.log(f"hook: stdin is not valid JSON ({exc}); {len(raw)} bytes")
            return 0
        if not isinstance(payload, dict):
            config.log(f"hook: stdin JSON is {type(payload).__name__}, not an object")
            return 0
        result = handle(payload)
        if result:  # PreToolUse fires on every tool call; it notes only what it did
            config.log(f"hook: {result} [{(time.monotonic() - started) * 1000:.0f} ms]")
    except Exception as exc:  # the hook must never fail
        config.log(f"hook: error {exc!r}\n{traceback.format_exc()}")
    return 0
