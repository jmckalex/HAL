"""The listener's log, state/listen.log. Standard library only.

Privacy rule: the text of an utterance that was not addressed to Claude is
never written here. ignored() and discarded() take no text argument at all,
so the rule holds by construction; addressed() is the only function that
records what was said, and the listener calls it only for utterances it
submits (or acts on as a command).
"""

from __future__ import annotations

import os
import time

from . import settings

_LIMIT = 512 * 1024


def log(message: str, source: str = "listen") -> None:
    """Append one line. Never raises."""
    try:
        path = settings.log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size > _LIMIT:
                path.replace(path.with_name(path.name + ".1"))
        except FileNotFoundError:
            pass
        line = message.replace("\n", " ").replace("\r", " ")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()} {source}] {line}\n")
    except Exception:
        pass


def addressed(text: str, speech_seconds: float, what: str) -> None:
    """An utterance addressed to Claude (or accepted without the name)."""
    log(f"heard ({speech_seconds:.1f}s of speech, {what}): {text!r}")


def ignored(speech_seconds: float, total_seconds: float) -> None:
    """An utterance not addressed to Claude: duration only, never the words."""
    log(f"ignored an utterance not addressed to Claude ({speech_seconds:.1f}s of speech, {total_seconds:.1f}s long)")


def discarded(reason: str, seconds: float) -> None:
    log(f"discarded {seconds:.1f}s of audio ({reason})")
