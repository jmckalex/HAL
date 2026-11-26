"""The label on spoken requests, and the marks inside them. Standard library only.

The listener (hal/listener/) types what it hears into the Claude Code session.
Everything it types starts with a label, so the model, the hooks and the
front desk can tell a spoken request from a typed one (LISTENER.md, "What the
model is told"):

    [spoken 21:14] Claude, ask Clew-boss [heard as "Clue Boss"] to add a lamprey [?] to the manual.

    [spoken HH:MM]                  spoken, starting at this local time
    [spoken HH:MM; by Robin]        spoken by someone else enrolled in the voice
                                    check ("by Robin?": probably; "by an
                                    unrecognised voice"; "by Robin and the user")
    [spoken HH:MM; interrupted the voice during "…"]
                                    and it cut off the voice, which was saying
                                    that sentence
    word [?]                        the transcriber was unsure of the word
    word [? Name]                   unsure; perhaps the name or term Name
    Name [heard as "words"]         the listener changed "words" to Name
    [pause 12s]                     a pause of that length, where it occurred

The listener removes square brackets from transcripts (sanitize()), so in a
spoken request every bracketed part is one of these marks, added by the
listener: none can be spoken into existence. A prompt that does not begin
with the label is left alone by everything here.

plain() gives the words alone: the request as the listener heard and
corrected it, without the label and marks. The hooks and the front desk judge
that, so they behave exactly as they did before the label existed.
for_speech() removes these marks wherever they appear, for text that is about
to be read aloud (the model may quote one back).
"""

from __future__ import annotations

import re
import time

# The label that starts a spoken request. The part after ";" never contains a
# bracket (the listener removes them from what it quotes).
_LABEL = re.compile(r"\[spoken (\d{1,2}):(\d{2})(?:;[^\[\]\n]*)?\]")
_LABEL_AT_START = re.compile(r"^[ \t]*" + _LABEL.pattern + r"[ \t]*")
# Any bracketed mark inside a spoken request (there are no other brackets in one).
_MARK = re.compile(r"[ \t]*\[[^\[\]\n]*\]")
# The marks themselves, recognised anywhere (for speech): only these exact forms.
_ANY_MARK = re.compile(
    r"[ \t]*\[(?:spoken \d{1,2}:\d{2}(?:;[^\[\]\n]*)?"
    r"|\?(?: [^\[\]\n]{1,80})?"
    r"|heard as \"[^\"\[\]\n]{0,120}\""
    r"|pause \d{1,4}(?:\.\d)? ?s)\]")
_QUOTE_MAX = 90


def label(when: float | None = None, interrupted: str = "", speaker: str = "") -> str:
    """The label for a request spoken at `when` (time.time(); now if None). With
    `interrupted`, the sentence the voice was saying when this speech cut it off;
    with `speaker`, who said it when it wasn't the user (the voice check)."""
    stamp = time.strftime("%H:%M", time.localtime(time.time() if when is None else when))
    parts = [f"spoken {stamp}"]
    speaker = " ".join(str(speaker or "").replace("[", "").replace("]", "").replace(";", ",").split())
    if speaker:
        parts.append(f"by {speaker}")
    if interrupted:
        parts.append(f"interrupted the voice during {quote(interrupted)}")
    return f"[{'; '.join(parts)}]"


_SPEAKER = re.compile(r";\s*by ([^;\[\]\n]+?)\s*(?:;|\])")


def speaker(prompt: str | None) -> str | None:
    """Who said a spoken request, from its label, when it wasn't the user (or None)."""
    m = _LABEL_AT_START.match(prompt or "")
    if not m:
        return None
    s = _SPEAKER.search(m.group(0))
    return s.group(1) if s else None


def quote(text: str, limit: int = _QUOTE_MAX) -> str:
    """Text for a mark: in double quotes, without brackets or inner double quotes, at most
    `limit` characters (cut at a word, with an ellipsis)."""
    text = " ".join(str(text).replace("[", "(").replace("]", ")").replace('"', "'").split())
    if len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
        text = cut.rstrip(",;:.-–— ") + "…"
    return f'"{text}"'


def sanitize(transcript: str) -> str:
    """A transcript with nothing in it that could pass for a label or a mark: square
    brackets become round ones. The listener applies this to everything it hears."""
    return str(transcript or "").replace("[", "(").replace("]", ")")


def is_spoken(prompt: str | None) -> bool:
    """Does the prompt (or, for several queued prompts, its first line) begin with the label?"""
    return bool(prompt) and bool(_LABEL_AT_START.match(prompt))


def spoken_at(prompt: str | None) -> str | None:
    """"HH:MM" from the label, or None."""
    m = _LABEL_AT_START.match(prompt or "")
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def _plain_line(line: str) -> str:
    m = _LABEL_AT_START.match(line)
    if not m:
        return line
    return " ".join(_MARK.sub("", line[m.end():]).split())


def plain(prompt: str | None) -> str:
    """The words of a spoken request, without the label and marks. Any other prompt is
    returned unchanged. Each line is treated alone (Claude Code may join queued prompts)."""
    if not prompt:
        return prompt or ""
    if "[spoken " not in prompt:
        return prompt
    return "\n".join(_plain_line(line) for line in prompt.split("\n"))


def for_speech(text: str | None) -> str:
    """Text with every label and mark removed, wherever they are: for reading aloud."""
    if not text or "[" not in text:
        return text or ""
    return _ANY_MARK.sub("", text)
