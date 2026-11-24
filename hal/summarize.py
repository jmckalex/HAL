"""Turn a written reply into a short spoken summary, using a small fast model.

The summary is written by Claude Haiku through a headless `claude -p` call:
no tools, no settings, no MCP servers, no saved session, so it starts quickly
(about 3 s in all) and cannot trigger HAL's own hooks. It runs in the detached
speech worker, never in the hook itself.

HAL_SUMMARIZER overrides the command (the tests use a stand-in): it is split
like a shell command, gets the reply's Markdown on stdin and must print the
words to speak.

When the front desk is on and running (hal/desk.py, desk_summaries), the
summary comes from a second Haiku process that it keeps warm, which saves the
start-up time. If that fails, the one-shot call above is made in the time
that is left.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PROMPT = (
    "You turn a written reply from an AI assistant into what the assistant says aloud to its user, "
    "who can also read the full reply on screen. The reply is given between <reply> tags. If the user's "
    "question or remark that the reply answers is known, it comes first, between <question> tags, only so "
    "that you know what was asked: never answer it yourself and never repeat it.\n\n"
    "Write a spoken summary of the reply: two or three short sentences, no more than forty-five words in "
    "total (the voice is slow, so brevity matters more than completeness), calm and plain. Lead with the "
    "part of the reply that answers the question. Keep the names, titles, numbers, days and times that "
    "are the answer; drop detail before you drop those.\n\n"
    "Keep the reply's own point of view exactly. In the reply, \"I\" is the assistant and \"you\" is the "
    "user; in your summary \"I\" must still be the assistant and \"you\" must still be the user. Never "
    "swap them, never speak as the user, never refer to the user in the third person, and never describe "
    "the reply from outside (\"the assistant says\"). If the reply asks the user a question or needs a "
    "decision from them, end with that question in the reply's own sense; do not invent a question or an "
    "offer that the reply does not contain.\n\n"
    "No markdown, no lists, no file paths, no code, no URLs. Add nothing that is not in the reply, and do "
    "not mention the screen. Output only the words to be spoken."
)


def command(cfg: dict) -> list[str]:
    override = os.environ.get("HAL_SUMMARIZER")
    if override:
        return shlex.split(override)
    claude = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
    return [
        claude, "-p",
        "--model", str(cfg["summary_model"]),
        "--effort", "low",
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt", PROMPT,
    ]


def _from_desk(markdown: str, cfg: dict) -> str:
    from . import config, desk

    started = time.monotonic()
    answer = desk.request({"op": "summarize", "text": markdown, "timeout": cfg["summary_timeout"]},
                          cfg["summary_timeout"])
    text = answer.get("text") if answer else None
    if isinstance(text, str) and text.strip():
        config.log(f"worker: summary from the front desk in {time.monotonic() - started:.1f}s")
        return text.strip()
    why = answer.get("error") if answer else "not running or no answer"
    config.log(f"worker: no summary from the front desk ({why}) after {time.monotonic() - started:.1f}s")
    return ""


def summarize(markdown: str, cfg: dict) -> str:
    """Return the spoken summary, or "" if the model call fails or times out."""
    timeout = cfg["summary_timeout"]
    if cfg.get("front_desk") and cfg.get("desk_summaries"):
        started = time.monotonic()
        text = _from_desk(markdown, cfg)
        if text:
            return text
        timeout -= time.monotonic() - started
        if timeout < 3:
            return ""
    env = {k: v for k, v in os.environ.items() if k != "HAL_VOICE"}  # the helper must not speak
    try:
        proc = subprocess.run(
            command(cfg), input=markdown, capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir(), env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()
