"""Paths and settings for the listener. Standard library only.

Settings live in the "listener" object of config.json (HAL_CONFIG overrides
the path, as for the rest of HAL). Every key is optional; anything missing or
of the wrong type falls back to DEFAULTS, so a partial or broken block never
stops the listener.

Runtime files live in <state>/listen/ (HAL_STATE_DIR overrides <state>):

    proxy.json          the pty proxy serving the voice session (pid, socket)
    proxy.lock          held by that proxy for its lifetime
    pty.sock            its control socket
    session-<pid>.json  session state written by the hooks of that proxy's claude
    listener.json       status of the running listener (no utterance text)
    listener.lock       held by the listener for its lifetime: one at a time
    paused              present while listening is paused
    duplex.sock         the audio helper's socket (duplex "aec"): voice workers play through it
    interrupted.json    where the voice was last cut off by speech over it (its own text only)
    listen.log          the log (in <state>/, beside hal.log)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def state_dir() -> Path:
    return Path(os.environ.get("HAL_STATE_DIR") or ROOT / "state")


def listen_dir() -> Path:
    return state_dir() / "listen"


def config_file() -> Path:
    return Path(os.environ.get("HAL_CONFIG") or ROOT / "config.json")


def log_file() -> Path:
    return state_dir() / "listen.log"


def speaker_pid_files() -> list[Path]:
    """PID files of the HAL speech workers (reply and acknowledgement).

    Taken from hal.speaker when it imports, so a rename there is followed;
    the fallback names are the ones it uses today."""
    try:
        from hal import speaker  # stdlib-only at import time

        files = [Path(speaker.PID_FILE), Path(getattr(speaker, "ACK_PID_FILE", speaker.PID_FILE))]
        # hal.config fixes STATE_DIR at import; honour a later HAL_STATE_DIR change.
        base = state_dir()
        return [base / f.name for f in dict.fromkeys(files)]
    except Exception:
        return [state_dir() / "speaker.pid", state_dir() / "ack.pid"]


DEFAULTS: dict = {
    # Start the listener automatically when `hal claude` starts the session
    # through the pty proxy. It exits when that session exits.
    "autostart": True,
    # Whole words that mark an utterance as addressed to Claude wherever they
    # occur (case-insensitive; a possessive 's is allowed).
    "names": ["claude", "klaud", "klaude", "clawd", "klawd", "clode", "klode", "cloat", "cload", "gload"],
    # Ordinary words (and "claud", which it also writes for "clod") that speech-to-text produces for "Claude". They count only
    # in the position of a name being called: "Cloud, what's ...", "Thanks,
    # cloud.", "By the way, cloud, ...", "Hey cloud." - not "in the cloud".
    "vocative_names": ["claud", "cloud", "clod", "clawed", "claw", "clause", "clothes", "cloth", "cloak", "clog"],
    # Words allowed before a vocative name at the start of a sentence when no
    # comma separates them: "Hey cloud.", "Good morning cloud."
    "lead_words": ["hey", "hi", "hello", "ok", "okay", "oh", "so", "well", "and", "but", "now", "yes", "right",
                   "um", "uh", "er", "erm", "hmm", "good", "morning", "afternoon", "evening", "night",
                   "thanks", "thank", "you", "excuse", "me", "sorry", "please", "alright", "yo"],
    # An utterance made only of a name and these words is the name alone:
    # "Claude." or "Hey Claude?" Claude answers and the next utterance needs no name.
    "name_only_words": ["hey", "ok", "okay", "oh", "um", "uh", "er", "erm", "hm", "hmm", "so", "well", "yo"],
    "name_only_reply": "Yes?",
    # Seconds after that reply during which an utterance needs no name.
    "armed_seconds": 8.0,
    # Wait this long after the name alone before answering; if speech resumes
    # in the meantime, don't answer and take that speech as the request.
    "name_only_grace": 0.25,
    # Seconds after the voice finishes speaking a reply during which an
    # utterance needs no name. 0 = off.
    "follow_up_seconds": 0.0,
    # While Claude is still working on a request the listener typed (the
    # session's turn is busy and has not stopped since), speech needs no name:
    # it is part of the conversation, and Claude Code queues it for the model.
    "during_turn": True,
    # "Claude, stop listening" pauses; resume with `hal-listen resume`.
    "stop_phrases": ["stop listening", "pause listening", "stop listening for now", "stop listening please"],
    "paused_reply": "Very well. I will stop listening until you resume me from the keyboard.",
    # Said when a request can't be typed because Claude Code shows a dialog.
    "dialog_notice": "I need you at the keyboard. I will pass on your request afterwards.",
    # Said when unsent text sits in the prompt box.
    "draft_notice": "There is unsent text at the keyboard. I will pass on your request once it is cleared.",
    "unreachable_notice": "I can't reach the Claude Code session.",
    # How long a held request waits for the dialog to close before it is dropped.
    "hold_seconds": 120.0,
    # Typing is held while the keyboard was used within this many seconds.
    "keyboard_quiet": 1.0,
    # A draft (keys typed since the last Enter) stops typing for this long
    # after the last keystroke; after that it is assumed abandoned.
    "draft_hold_seconds": 60.0,
    # Pause between typing the text and pressing Enter. Claude Code treats a
    # burst that already contains Enter as a paste, so Enter is sent apart.
    "submit_delay": 0.15,
    # Ignore dialogs older than this; a missed closing event can't wedge input.
    "dialog_max_age": 1800.0,
    # Don't type until the session's hooks have reported in (SessionStart).
    "require_session_state": True,
    # Voice-activity detection (Silero VAD, 32 ms frames at 16 kHz).
    "vad_threshold": 0.5,
    "vad_neg_threshold": 0.35,
    "end_silence": 0.8,        # silence that ends an utterance (with turn_taking "semantic": a fragment)
    "pre_roll": 0.4,           # audio kept from before speech starts
    "post_roll": 0.2,          # silence kept after speech ends
    "min_speech": 0.25,        # shorter bursts are noise
    "max_utterance": 30.0,     # longer speech is cut and the parts joined
    # Discard audio while the voice speaks and for this long afterwards.
    "echo_tail": 0.6,
    # Input device: null = system default, or a name substring or index.
    "device": None,
    # Speech-to-text model: a directory under the project, or a HuggingFace id
    # (downloaded under models/hf/).
    "asr_model": "models/parakeet-tdt-0.6b-v2",
    "vad_model": "models/silero_vad.onnx",
    # When has the speaker finished (turns.py, LISTENER.md "Turns")?
    # "semantic": a pause yields a fragment; addressed fragments are held as a
    # turn and judged by what was said; speech that continues a turn needs no
    # name. "silence": every utterance is a request (the behaviour before turns).
    "turn_taking": "semantic",
    # A turn judged complete is sent once this much silence has followed it.
    "turn_end_silence": 1.2,
    # A turn judged unfinished is sent as it stands after this much silence.
    "turn_max_wait": 60.0,
    # A turn the rules can't judge, with no judge's answer (front desk off,
    # not running or too slow), is sent after this much silence.
    "turn_unclear_wait": 3.0,
    # Ask the front desk's warm model (hal/desk.py, op "turn") when the rules
    # can't tell. Only fragments of an addressed turn are ever sent to it.
    "turn_judge": True,
    # How long to wait for its answer before falling back to the timer.
    "turn_judge_timeout": 2.5,
    # Transcribe the turn's audio again as a whole once it has several
    # fragments, for better words and punctuation across the pauses.
    "turn_retranscribe": True,
    # Said as (or at the end of) a later fragment: send the turn now, without
    # these words. In the first fragment they are ordinary words that only
    # mark it complete ("Yes, go ahead." is an answer).
    "turn_send_phrases": ["that's all", "that is all", "that's it", "that's everything", "that will be all",
                          "that's all for now", "that's my question", "go ahead", "send it", "over to you",
                          "end of request"],
    # Said as (or at the end of) a later fragment: discard the turn. At the
    # start of one ("Scratch that, ..."): discard what came before it.
    "turn_cancel_phrases": ["never mind", "nevermind", "cancel that", "cancel", "forget it", "forget that",
                            "scratch that", "disregard that", "ignore that", "start again", "start over"],
    # Said when a turn is cancelled. "" says nothing (anything spoken closes
    # the microphone while it plays, so a fresh start could be lost).
    "turn_cancel_reply": "",
    # A sign that a turn is being held: "none", "sound" (turn_cue_sound,
    # played softly) or "words" (turn_cue_words, spoken). At most once per
    # turn, after turn_cue_after seconds of silence.
    "turn_cue": "none",
    "turn_cue_after": 10.0,
    "turn_cue_sound": "/System/Library/Sounds/Tink.aiff",
    "turn_cue_words": "Go on.",
    # Listening while the voice speaks (LISTENER.md, "Interrupting the voice").
    # "half": the voice gate drops what is heard while HAL speaks (as before).
    # "aec": capture through duplex/hal-duplex with Apple's echo cancellation and
    # keep listening, so speaking over the voice interrupts it. Falls back to
    # "half" by itself (and logs why) if the helper or echo cancellation is
    # unavailable.
    "duplex": "half",
    # With "aec": how much other apps' audio is ducked while voice processing
    # runs ("min", "default", "mid", "max"); whether Apple's automatic gain
    # control is on; whether voice processing stays on with headphones (there
    # is no echo to cancel there); and whether it steps aside while another app
    # captures audio (a call), the listener being half-duplex meanwhile.
    "aec_ducking": "min",
    "aec_agc": False,
    "aec_on_headphones": False,
    "aec_yield_to_other_input": True,
    # Speaking over the voice pauses it at once, when, within barge_in_within
    # seconds, at least barge_in_speech seconds of frames are speech (VAD
    # probability >= barge_in_vad) and loud enough: at least barge_in_min_db
    # dBFS after echo cancellation, and no more than barge_in_echo_margin dB
    # below what the voice is playing at that moment. What was said then
    # decides: a stop phrase silences the voice; "go on" (or its own echo, or
    # no words) resumes it; anything else silences it and is taken as a request
    # without the name.
    "barge_in": True,
    "barge_in_vad": 0.6,
    "barge_in_speech": 0.2,
    "barge_in_within": 0.3,
    "barge_in_min_db": -50.0,
    "barge_in_echo_margin": 20.0,
    "barge_in_echo_ratio": 0.6,
    "barge_in_stop_phrases": ["stop", "stop it", "stop talking", "stop there", "that's enough", "that is enough",
                              "enough", "thank you", "thanks", "thank you very much", "thanks very much",
                              "that will do", "that'll do", "quiet", "be quiet", "shush", "hush", "silence",
                              "never mind", "got it", "i've got it", "i get it", "hold on", "wait", "wait a moment",
                              "one moment", "just a moment", "pause"],
    "barge_in_resume_phrases": ["go on", "continue", "carry on", "keep going", "resume", "go ahead",
                                "you were saying", "what were you saying", "as you were saying", "sorry go on",
                                "please continue", "do go on"],
    # After the voice is cut off, "go on" resumes it (from the start of the
    # sentence it was cut in) for this many seconds, unless it has spoken since.
    "resume_seconds": 300.0,
    # What the listener types (LISTENER.md, "What the model is told";
    # hal/spoken.py, annotate.py). Every request starts "[spoken 21:14]", and
    # may carry marks: "[?]", "[? Name]", "[heard as ...]", "[pause 12s]".
    # false: typed exactly as before, with no label, marks or corrections
    # (the marks and corrections need the label to be recognised).
    "spoken_label": True,
    # Correct names and terms the transcriber gets wrong (lexicon.py), in
    # addressed speech only, and mark each correction: the entries and aliases
    # in lexicon_file, and the names of the folders directly inside lexicon_dirs.
    "lexicon": True,
    "lexicon_file": "lexicon.txt",
    "lexicon_dirs": [],
    # Mark words the transcriber was unsure of: Parakeet's confidence (0 to 1)
    # below doubt_threshold; at most two in a sentence.
    "doubt_marks": True,
    "doubt_threshold": 0.82,
    # Mark a pause of at least pause_mark_seconds between the fragments of a turn.
    "pause_marks": True,
    "pause_mark_seconds": 3.0,
    # When the request cut off the voice (duplex "aec"), say what it was saying.
    "interruption_note": True,
}

_FLOATS = ("armed_seconds", "name_only_grace", "follow_up_seconds", "hold_seconds", "keyboard_quiet",
           "draft_hold_seconds", "submit_delay", "dialog_max_age", "vad_threshold", "vad_neg_threshold",
           "end_silence", "pre_roll", "post_roll", "min_speech", "max_utterance", "echo_tail",
           "turn_end_silence", "turn_max_wait", "turn_unclear_wait", "turn_judge_timeout", "turn_cue_after",
           "barge_in_vad", "barge_in_speech", "barge_in_within", "barge_in_echo_margin", "barge_in_echo_ratio",
           "resume_seconds", "doubt_threshold", "pause_mark_seconds")
_SIGNED_FLOATS = ("barge_in_min_db",)
_WORD_LISTS = ("names", "vocative_names", "lead_words", "name_only_words", "stop_phrases",
               "turn_send_phrases", "turn_cancel_phrases", "barge_in_stop_phrases", "barge_in_resume_phrases")
_PATH_LISTS = ("lexicon_dirs",)
_STRINGS = ("name_only_reply", "paused_reply", "dialog_notice", "draft_notice", "unreachable_notice",
            "asr_model", "vad_model", "turn_cancel_reply", "turn_cue_sound", "turn_cue_words", "lexicon_file")
_BOOLS = ("autostart", "require_session_state", "turn_judge", "turn_retranscribe", "during_turn",
          "aec_agc", "aec_on_headphones", "aec_yield_to_other_input", "barge_in", "spoken_label", "lexicon",
          "doubt_marks", "pause_marks", "interruption_note")
_CHOICES = {"turn_taking": ("semantic", "silence"), "turn_cue": ("none", "sound", "words"),
            "duplex": ("half", "aec"), "aec_ducking": ("min", "default", "mid", "max")}


def _raw_block(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    block = data.get("listener") if isinstance(data, dict) else None
    return block if isinstance(block, dict) else {}


def load(path: Path | None = None) -> dict:
    """DEFAULTS overlaid with the valid entries of config.json's "listener" block."""
    cfg = dict(DEFAULTS)
    for key, value in _raw_block(path or config_file()).items():
        if key not in DEFAULTS:
            continue
        if key in _FLOATS:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                cfg[key] = float(value)
        elif key in _SIGNED_FLOATS:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cfg[key] = float(value)
        elif key in _WORD_LISTS:
            if isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value):
                cfg[key] = [v.strip().lower() for v in value]
        elif key in _PATH_LISTS:
            if isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value):
                cfg[key] = [v.strip() for v in value]
        elif key in _STRINGS:
            if isinstance(value, str):
                cfg[key] = value
        elif key in _BOOLS:
            if isinstance(value, bool):
                cfg[key] = value
        elif key in _CHOICES:
            if isinstance(value, str) and value.strip().lower() in _CHOICES[key]:
                cfg[key] = value.strip().lower()
        elif key == "device":
            if value is None or isinstance(value, (str, int)) and not isinstance(value, bool):
                cfg[key] = value
    if cfg["vad_neg_threshold"] > cfg["vad_threshold"]:
        cfg["vad_neg_threshold"] = max(0.0, cfg["vad_threshold"] - 0.15)
    cfg["max_utterance"] = max(cfg["max_utterance"], 2.0)
    cfg["turn_max_wait"] = max(cfg["turn_max_wait"], 1.0)
    cfg["turn_judge_timeout"] = min(max(cfg["turn_judge_timeout"], 0.3), 5.0)
    cfg["barge_in_within"] = min(max(cfg["barge_in_within"], 0.064), 2.0)
    cfg["barge_in_speech"] = min(max(cfg["barge_in_speech"], 0.032), cfg["barge_in_within"])
    cfg["barge_in_vad"] = min(cfg["barge_in_vad"], 1.0)
    cfg["barge_in_echo_ratio"] = min(max(cfg["barge_in_echo_ratio"], 0.1), 1.0)
    cfg["doubt_threshold"] = min(cfg["doubt_threshold"], 1.0)
    return cfg


def project_path(value: str) -> Path:
    """A path from the settings: relative ones are relative to the project."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p
