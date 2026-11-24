"""Paths, settings, mute state and logging shared by the HAL voice tools.

Everything lives inside the project: runtime state (mute flag, lock, the
PID of whatever is speaking, temporary WAVs, the log) goes in ./state/.
Environment overrides exist mainly for tests:

    HAL_STATE_DIR  state directory           (default: <project>/state)
    HAL_MODEL      Piper .onnx voice model   (default: <project>/models/hal.onnx)
    HAL_CONFIG     settings file             (default: <project>/config.json)
    HAL_SUMMARIZER command that writes the spoken summary (default: claude -p, Haiku)
    HAL_WARM_MODEL command for the front desk's long-running models (default:
                   claude -p in stream-json mode); it is given
                   `--role desk|summary --system-prompt TEXT`
    HAL_PLAYER     audio player command      (default: /usr/bin/afplay, or the
                   listener's audio helper while it runs, see player_command());
                   the WAV path is appended as the last argument
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("HAL_STATE_DIR") or ROOT / "state")
MODEL = Path(os.environ.get("HAL_MODEL") or ROOT / "models" / "hal.onnx")
CONFIG_FILE = Path(os.environ.get("HAL_CONFIG") or ROOT / "config.json")
LOG_FILE = STATE_DIR / "hal.log"
MUTE_FILE = STATE_DIR / "muted"

DEFAULTS: dict = {
    # Spoken replies are cut at a sentence boundary after this many characters
    # (0 = no cap), then the overflow message is spoken. The HAL voice is slow,
    # about 14-17 characters a second, so 600 is roughly 35-45 s of speech.
    "max_chars": 600,
    "overflow_message": "The rest is on your screen.",
    # Piper phoneme length: >1 is slower, <1 faster. null = the model's own.
    "length_scale": None,
    # Speak replies from `claude -p` and Agent SDK runs too (normally silent).
    "speak_in_print_mode": False,
    # Which Claude Code sessions have the voice. "master": only a session
    # started with HAL_VOICE=1 in its environment (`hal claude` does that);
    # hooks from every other session do nothing. "all": every session.
    "voice_sessions": "master",
    # What is spoken for a reply. "summary": a short spoken summary written by
    # a small model (replies whose speakable text is at most summary_min_chars
    # are simply read out). "full": the cleaned reply itself, up to max_chars.
    # "lead": speak the reply's opening paragraph, which Claude writes to be
    # spoken (no model call, so speech starts at once); falls back to "summary"
    # when the opening is missing or longer than lead_max_chars.
    "spoken_reply": "lead",
    "lead_max_chars": 450,
    "summary_min_chars": 350,
    "summary_model": "haiku",
    "summary_timeout": 20,
    # Speak a short acknowledgement as soon as a prompt is submitted.
    "acknowledge": True,
    "ack_phrases": [
        "Just a moment.",
        "One moment, please.",
        "I'll look into that. Please give me a moment.",
        "Let me look into that.",
    ],
    # While a turn uses tools (hal/progress.py): speak the short line Claude
    # writes before a tool call (whole sentences up to progress_max_chars),
    # and, at a tool call after progress_cue_seconds with nothing said, one of
    # progress_phrases (at most two a turn; 0 = never).
    "speak_progress": True,
    "progress_max_chars": 160,
    "progress_cue_seconds": 12,
    "progress_phrases": [
        "Still working on it.",
        "Bear with me, this is taking a little longer.",
    ],
    # The front desk (hal/desk.py): a long-lived process that keeps a small
    # model and the voice loaded and hears each prompt first. It answers purely
    # social remarks itself and gives everything else a short acknowledgement
    # in line with the request. Off: everything works as without it.
    "front_desk": False,
    # A social remark the desk answered is blocked, so Claude takes no turn.
    # false: it goes through, Claude is told it has been answered, and Claude's
    # reply to it is not spoken.
    "desk_block": True,
    "desk_model": "haiku",
    # Prompts of at most this many words wait for the desk's decision (they may
    # be social); longer ones never are, and go straight through.
    "desk_wait_words": 12,
    # How long such a prompt waits before falling back to the canned
    # acknowledgement. The hook's own timeout is 5 s.
    "desk_wait_seconds": 3.0,
    # Spoken summaries come from a second warm model in the desk when it runs.
    "desk_summaries": True,
    # A third warm model judges, for the always-on listener (LISTENER.md,
    # "Turns"), whether a spoken request is finished or the speaker paused
    # mid-thought. Off: the listener uses its rules and a timer instead.
    "desk_turns": True,
    # The desk's model is replaced by a fresh one after this many prompts,
    # which bounds its conversation. 1 means no history at all, but a model
    # started only seconds earlier answers about 0.5 s slower, which a quick
    # exchange would then pay on every prompt.
    "desk_max_turns": 6,
    # The desk exits after this long without a prompt.
    "desk_idle_minutes": 120,
    # How the front desk addresses the user, e.g. "Dr. Alexander". "" = no name.
    "user_name": "",
}

_LOG_LIMIT = 512 * 1024


def player_command() -> list[str]:
    """The command that plays a WAV (its path is appended).

    HAL_PLAYER if set; otherwise, while the listener's audio helper runs (its
    socket exists: listener setting duplex "aec", LISTENER.md "Interrupting the
    voice"), `hal-duplex play`, so the voice plays through the helper and its
    echo canceller knows what to remove; it runs afplay itself if the helper
    doesn't answer. Otherwise afplay."""
    override = os.environ.get("HAL_PLAYER")
    if override:
        return shlex.split(override)
    helper = ROOT / "duplex" / "hal-duplex"
    sock = STATE_DIR / "listen" / "duplex.sock"
    try:
        if sock.is_socket() and os.access(helper, os.X_OK):
            return [str(helper), "play", "--socket", str(sock)]
    except OSError:
        pass
    return ["/usr/bin/afplay"]


def load_config() -> dict:
    """DEFAULTS overlaid with config.json; a broken file is logged and ignored."""
    config = dict(DEFAULTS)
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            config.update({k: v for k, v in data.items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        log(f"config: ignoring {CONFIG_FILE}: {exc}")
    try:
        config["max_chars"] = int(config["max_chars"])
    except (TypeError, ValueError):
        config["max_chars"] = DEFAULTS["max_chars"]
    if not isinstance(config["overflow_message"], str):
        config["overflow_message"] = DEFAULTS["overflow_message"]
    if not isinstance(config["length_scale"], (int, float)) or config["length_scale"] <= 0:
        config["length_scale"] = None
    if config["spoken_reply"] not in ("lead", "summary", "full"):
        config["spoken_reply"] = DEFAULTS["spoken_reply"]
    for key in ("summary_min_chars", "summary_timeout", "lead_max_chars"):
        try:
            config[key] = max(0, int(config[key]))
        except (TypeError, ValueError):
            config[key] = DEFAULTS[key]
    phrases = config["ack_phrases"]
    if not isinstance(phrases, list) or not all(isinstance(p, str) and p.strip() for p in phrases) or not phrases:
        config["ack_phrases"] = DEFAULTS["ack_phrases"]
    if not isinstance(config["speak_progress"], bool):
        config["speak_progress"] = DEFAULTS["speak_progress"]
    try:
        config["progress_max_chars"] = min(600, max(20, int(config["progress_max_chars"])))
    except (TypeError, ValueError):
        config["progress_max_chars"] = DEFAULTS["progress_max_chars"]
    try:
        config["progress_cue_seconds"] = max(0.0, float(config["progress_cue_seconds"]))
    except (TypeError, ValueError):
        config["progress_cue_seconds"] = DEFAULTS["progress_cue_seconds"]
    phrases = config["progress_phrases"]
    if not isinstance(phrases, list) or not all(isinstance(p, str) and p.strip() for p in phrases):
        config["progress_phrases"] = DEFAULTS["progress_phrases"]
    if config["voice_sessions"] not in ("master", "all"):
        config["voice_sessions"] = DEFAULTS["voice_sessions"]
    for key in ("front_desk", "desk_block", "desk_summaries", "desk_turns"):
        if not isinstance(config[key], bool):
            config[key] = DEFAULTS[key]
    for key, low, high in (("desk_wait_words", 0, 100), ("desk_max_turns", 1, 1000)):
        try:
            config[key] = min(high, max(low, int(config[key])))
        except (TypeError, ValueError):
            config[key] = DEFAULTS[key]
    for key, low, high in (("desk_wait_seconds", 0.5, 3.5), ("desk_idle_minutes", 0.05, 7 * 24 * 60)):
        try:
            config[key] = min(high, max(low, float(config[key])))
        except (TypeError, ValueError):
            config[key] = DEFAULTS[key]
    if not isinstance(config["desk_model"], str) or not config["desk_model"].strip():
        config["desk_model"] = DEFAULTS["desk_model"]
    if not isinstance(config["user_name"], str):
        config["user_name"] = ""
    config["user_name"] = " ".join(config["user_name"].split())[:80]
    return config


def is_voice_session(cfg: dict | None = None) -> bool:
    """True if hooks from this Claude Code session may speak or stop speech."""
    cfg = cfg or load_config()
    return cfg["voice_sessions"] == "all" or os.environ.get("HAL_VOICE") == "1"


def is_muted() -> bool:
    return MUTE_FILE.exists()


def set_muted(muted: bool) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if muted:
        MUTE_FILE.touch()
    else:
        MUTE_FILE.unlink(missing_ok=True)


def log(message: str) -> None:
    """Append a line to state/hal.log. Never raises."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if LOG_FILE.stat().st_size > _LOG_LIMIT:
                LOG_FILE.replace(LOG_FILE.parent / (LOG_FILE.name + ".1"))
        except FileNotFoundError:
            pass
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {message}\n")
    except Exception:
        pass
