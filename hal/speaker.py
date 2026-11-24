"""Synthesise speech with the HAL Piper voice and play it, one voice at a time.

    speak(text)        stop whatever is speaking, then speak text from a detached
                       worker process (returns at once; the audio outlives the caller)
    stop()             stop whatever is speaking, from any session
    synthesize(text, wav_path)   render to a WAV file without playing it
    play_prepared(wav, slot, token)   play a WAV rendered elsewhere (the front
                       desk keeps the voice loaded) through the same slots

Only one utterance plays at a time, machine-wide for this installation. The
coordination lives in the state directory, under an flock:

    current      ticket of the utterance that may play ("stopped <ns>" after stop())
    speaker.pid  PID of the worker for that ticket; it leads its own process
                 group, so killing the group also kills its afplay child

A worker synthesises first, then checks under the lock that its ticket is
still current before starting the player, so a stop() or newer speak() that
lands mid-synthesis is never followed by stale audio. Speech prepared for a
prompt by the front desk carries the value stop() wrote for that prompt (its
token) and plays only if nothing has replaced it since: no newer prompt, no
`hal stop`, no reply.

Piper (and numpy/onnxruntime) are imported only inside synthesize(), so
speak() and stop() stay fast enough for a hook.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from . import config

LOCK_FILE = config.STATE_DIR / "lock"
CURRENT_FILE = config.STATE_DIR / "current"
PID_FILE = config.STATE_DIR / "speaker.pid"
ACK_PID_FILE = config.STATE_DIR / "ack.pid"          # worker speaking an acknowledgement
ACK_CACHE = config.STATE_DIR / "ack-cache"           # pre-rendered acknowledgement WAVs
LAST_SPOKEN_FILE = config.STATE_DIR / "last_spoken.json"  # what was said instead of the full reply

# How the worker is started: the venv's python, isolated from PYTHON* env vars
# and from the caller's working directory, with the project root on sys.path.
_BOOTSTRAP = "import sys; sys.path.insert(0, sys.argv[1]); from hal.cli import main; sys.exit(main(sys.argv[2:]))"
_WORKER_MARK = "_worker"
_STALE_WAV_SECONDS = 600


@contextlib.contextmanager
def _locked():
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    import fcntl

    fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _write(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _is_worker(pid: int) -> bool:
    """True if pid is alive, leads its own process group and is a HAL worker
    (guards against killing an unrelated process that reused the PID)."""
    if pid <= 1:
        return False
    try:
        if os.getpgid(pid) != pid:
            return False
        out = subprocess.run(
            ["/bin/ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return _WORKER_MARK in out and "hal.cli" in out


def _pid_from(path: Path) -> int:
    try:
        return int(json.loads(_read(path) or "{}").get("pid", 0))
    except (ValueError, TypeError, AttributeError):
        return 0


def _speaker_pid() -> int:
    return _pid_from(PID_FILE)


def _kill_speaker(pid_file: Path = PID_FILE) -> bool:
    """Kill the worker recorded in pid_file (its whole process group). Call with the lock held."""
    pid = _pid_from(pid_file)
    pid_file.unlink(missing_ok=True)
    if not _is_worker(pid):
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _sweep_stale_wavs() -> None:
    cutoff = time.time() - _STALE_WAV_SECONDS
    for wav in [*config.STATE_DIR.glob("speech-*.wav"), *config.STATE_DIR.glob("speech-*.wav.json")]:
        try:
            if wav.stat().st_mtime < cutoff:
                wav.unlink()
        except OSError:
            pass


def sidecar(wav: str | Path) -> Path:
    """<wav>.json: the text of a rendered WAV and its sentences' times (see synthesize())."""
    return Path(f"{wav}.json")


def _unlink_wav(wav: str | Path) -> None:
    Path(wav).unlink(missing_ok=True)
    sidecar(wav).unlink(missing_ok=True)


_SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))\s+")


def _write_sidecar(wav: str | Path, text: str, chunks: list[int], rate: int) -> None:
    """Record what a WAV says, sentence by sentence, for the listener (LISTENER.md,
    "Interrupting the voice"): it compares what it hears with this to ignore the
    voice's own echo, and resumes a cut reply from the sentence it was cut in.

    Piper renders one chunk per sentence; when its split matches ours, each
    sentence gets its chunk's exact times, otherwise times in proportion to
    length. Never raises."""
    try:
        total = sum(chunks) / rate if rate else 0.0
        parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()] or [text.strip()]
        if len(parts) == len(chunks):
            lengths = [n / rate for n in chunks]
        else:
            chars = sum(len(p) for p in parts) or 1
            lengths = [total * len(p) / chars for p in parts]
        sentences, t = [], 0.0
        for p, d in zip(parts, lengths):
            sentences.append({"text": p, "start": round(t, 3), "end": round(t + d, 3)})
            t += d
        _write(sidecar(wav), json.dumps({"text": text, "seconds": round(total, 3), "sentences": sentences}))
    except Exception:
        pass


def current() -> str:
    """What the current file holds: the ticket that may play, or stop()'s token."""
    return _read(CURRENT_FILE)


def is_speaking() -> bool:
    return _is_worker(_speaker_pid()) or _is_worker(_pid_from(ACK_PID_FILE))


def _stop_locked() -> tuple[bool, str]:
    token = f"stopped {time.time_ns()}"
    _write(CURRENT_FILE, token)
    killed_ack = _kill_speaker(ACK_PID_FILE)
    return _kill_speaker() or killed_ack, token


def stop() -> bool:
    """Silence any speech, pending or playing. Returns True if something was killed."""
    with _locked():
        return _stop_locked()[0]


def stop_for_prompt() -> tuple[bool, str]:
    """stop(), also returning its token: the value it left in the current file.
    Speech prepared for this prompt may play only while that is still current."""
    with _locked():
        return _stop_locked()


def _spawn_worker(ticket: str, kind: str) -> subprocess.Popen:
    with open(config.LOG_FILE, "ab") as log_file:
        return subprocess.Popen(
            [sys.executable, "-E", "-P", "-c", _BOOTSTRAP, str(config.ROOT), _WORKER_MARK, ticket, kind],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            cwd=str(config.ROOT),
            start_new_session=True,  # own session and process group: survives the hook
        )


def _feed(proc: subprocess.Popen, text: str) -> None:
    try:
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
    except OSError:
        pass


def speak(text: str, kind: str = "plain") -> subprocess.Popen | None:
    """Stop the reply being spoken and speak text from a detached worker process.

    kind "plain": text is already cleaned for speech and is spoken as is.
    kind "reply": text is a reply's Markdown; the worker decides what to say
                  (the reply itself if short, otherwise a spoken summary).
    An acknowledgement that is still playing is left to finish first.
    Returns the worker's Popen (callers may wait() on it), or None if there
    was nothing to say.
    """
    text = text.strip()
    if not text:
        return None
    with _locked():
        _kill_speaker()
        _sweep_stale_wavs()
        ticket = str(time.time_ns())
        _write(CURRENT_FILE, ticket)
        proc = _spawn_worker(ticket, kind)
        _write(PID_FILE, json.dumps({"pid": proc.pid, "ticket": ticket}))
    _feed(proc, text)
    return proc


def ack(phrase: str) -> subprocess.Popen | None:
    """Speak a short acknowledgement from a detached worker. It does not
    replace the reply slot, and a reply that follows waits for it to finish."""
    phrase = phrase.strip()
    if not phrase:
        return None
    with _locked():
        _kill_speaker(ACK_PID_FILE)
        proc = _spawn_worker("ack", "ack")
        _write(ACK_PID_FILE, json.dumps({"pid": proc.pid}))
    _feed(proc, phrase)
    return proc


def say_progress(text: str, token: str, seq: int) -> subprocess.Popen | None:
    """Speak a progress line while Claude works (hal/progress.py), in the
    acknowledgement slot: it waits for whatever is playing there, plays only
    while token is current and it is still the turn's newest line, and a reply
    that follows waits for it. Returns the worker's Popen, or None."""
    text = text.strip()
    if not text:
        return None
    with _locked():
        if _read(CURRENT_FILE) != token or config.is_muted():
            return None
        proc = _spawn_worker(token, f"progress:{int(seq)}")
    _feed(proc, text)
    return proc


def choose_ack_phrase(cfg: dict) -> str:
    """One of the configured acknowledgements, at random, not the one used last time."""
    import random

    phrases = cfg["ack_phrases"]
    last_file = config.STATE_DIR / "ack.last"
    try:
        previous = int(last_file.read_text())
    except (OSError, ValueError):
        previous = -1
    choices = [i for i in range(len(phrases)) if i != previous] or [0]
    index = random.choice(choices)
    try:
        last_file.write_text(str(index))
    except OSError:
        pass
    return phrases[index]


def play_prepared(wav: Path, slot: str, token: str, keep: bool = False) -> subprocess.Popen | None:
    """Play a WAV rendered elsewhere, if token is still current and HAL isn't muted.

    slot "ack":   the acknowledgement slot; a reply that follows waits for it.
    slot "reply": the reply slot, like speak(): it replaces whatever reply is
                  playing and takes a new ticket.
    token is what stop_for_prompt() returned for the prompt this answers; if
    anything has replaced it since (a newer prompt, `hal stop`, a reply), the
    WAV is dropped and None is returned. The worker deletes the WAV when done
    unless keep is set (cached acknowledgements).
    """
    with _locked():
        if _read(CURRENT_FILE) != token or config.is_muted():
            if not keep:
                _unlink_wav(wav)
            return None
        kind = "wavkeep" if keep else "wav"
        if slot == "reply":
            _kill_speaker()
            _sweep_stale_wavs()
            ticket = str(time.time_ns())
            _write(CURRENT_FILE, ticket)
            proc = _spawn_worker(ticket, kind)
            _write(PID_FILE, json.dumps({"pid": proc.pid, "ticket": ticket}))
        else:
            _kill_speaker(ACK_PID_FILE)
            proc = _spawn_worker("ack", kind)
            _write(ACK_PID_FILE, json.dumps({"pid": proc.pid}))
    _feed(proc, str(wav))
    return proc


def ack_wav(phrase: str, cfg: dict | None = None, voice=None) -> Path:
    """Path of the cached WAV for an acknowledgement phrase, rendering it on first use."""
    import hashlib

    cfg = cfg or config.load_config()
    try:
        model_size = config.MODEL.stat().st_size
    except OSError:
        model_size = 0
    key = hashlib.sha1(f"{phrase}|{cfg['length_scale']}|{model_size}".encode()).hexdigest()[:16]
    wav = ACK_CACHE / f"{key}.wav"
    if not wav.exists():
        from .textclean import pronounce

        ACK_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = ACK_CACHE / f".{key}.{os.getpid()}.{time.time_ns()}.wav"
        synthesize(pronounce(phrase), tmp, length_scale=cfg["length_scale"], voice=voice, sidecar=True)
        try:
            os.replace(sidecar(tmp), sidecar(wav))
        except OSError:
            pass
        os.replace(tmp, wav)
    elif not sidecar(wav).exists():
        _ack_sidecar(wav, phrase)    # rendered before sidecars existed
    return wav


def _ack_sidecar(wav: Path, phrase: str) -> None:
    try:
        from .textclean import pronounce

        with wave.open(str(wav)) as w:
            rate, frames = w.getframerate(), w.getnframes()
        _write_sidecar(wav, pronounce(phrase), [frames], rate)
    except Exception:
        pass


def _wait_for_ack(limit: float = 6.0) -> None:
    """Let an acknowledgement or progress line that is playing finish before a
    reply starts. After limit seconds the reply goes ahead, and the caller
    silences the slot (_silence_ack_slot) so the two never overlap."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline and _is_worker(_pid_from(ACK_PID_FILE)):
        time.sleep(0.1)


def _silence_ack_slot() -> None:
    """Stop the acknowledgement slot if it is still playing. Call with the lock held."""
    if ACK_PID_FILE.exists():
        _kill_speaker(ACK_PID_FILE)


def _summary_input(reply: str) -> str:
    """The reply in <reply> tags, preceded by the user's prompt in <question>
    tags when the hook recorded one in the last half hour."""
    question = ""
    try:
        data = json.loads(_read(config.STATE_DIR / "last_prompt.json") or "{}")
        if isinstance(data, dict) and time.time() - float(data.get("time") or 0) < 1800:
            question = str(data.get("text") or "").strip()
    except (ValueError, TypeError):
        pass
    head = f"<question>\n{question}\n</question>\n\n" if question else ""
    return f"{head}<reply>\n{reply}\n</reply>"


def _lead_paragraph(markdown: str) -> str:
    """The reply's opening paragraph, if it is prose: not a heading, a list,
    a table, a quotation or code."""
    for block in markdown.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith(("#", "-", "*", "|", ">", "```", "~~~")) or block[:2].rstrip(".)").isdigit():
            return ""
        return block
    return ""


def prepare_reply(markdown: str, cfg: dict) -> tuple[str, str]:
    """What to say for a reply: (text, how), how being "full", "summary" or "fallback"."""
    from .textclean import pronounce, speakable, to_speech, truncate

    full = speakable(markdown, 0, "")
    if cfg["spoken_reply"] not in ("lead", "summary") or len(full) <= cfg["summary_min_chars"]:
        return full, "full"
    if cfg["spoken_reply"] == "lead":
        lead = _lead_paragraph(markdown)
        spoken = speakable(lead, 0, "") if lead else ""
        if spoken and len(spoken) <= cfg["lead_max_chars"]:
            config.log(f"worker: speaking the opening paragraph ({len(spoken)} of {len(full)} chars)")
            return spoken, "lead"
        config.log("worker: no usable opening paragraph; summarising instead")
    from .summarize import summarize
    from .textclean import _drop_sources

    started = time.monotonic()
    summary = pronounce(to_speech(summarize(_summary_input(_drop_sources(markdown)), cfg)))
    if summary:
        config.log(f"worker: summary of {len(full)} chars -> {len(summary)} chars in {time.monotonic() - started:.1f}s")
        return summary, "summary"
    config.log(f"worker: summary failed after {time.monotonic() - started:.1f}s; reading the opening instead")
    return pronounce(truncate(to_speech(markdown), cfg["summary_min_chars"], cfg["overflow_message"])), "fallback"


def last_spoken() -> dict:
    try:
        data = json.loads(_read(LAST_SPOKEN_FILE) or "{}")
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def mark_last_spoken_reported() -> None:
    data = last_spoken()
    if data:
        data["reported"] = True
        _write(LAST_SPOKEN_FILE, json.dumps(data))


def _espeak_data_dir() -> Path:
    """Piper's bundled espeak-ng data directory, via a short symlink if needed.

    espeak-ng keeps its data path in a 160-byte buffer; given a longer path
    it silently falls back to a path on Piper's build machine and fails
    ("Error processing file '/Users/runner/work/...phontab'"). That happens
    when the project lives deep in the filesystem, so link to it from the
    temp directory instead.
    """
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    data = Path(ESPEAK_DATA_DIR)
    if len(str(data)) < 120:
        return data
    link = Path(tempfile.gettempdir()) / f"hal-espeak-ng-data-{os.getuid()}"
    try:
        if not (link.is_symlink() and os.readlink(link) == str(data)):
            tmp = link.with_name(f"{link.name}.{os.getpid()}")
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(data)
            os.replace(tmp, link)
        return link
    except OSError:
        return data


def load_voice(model: str | Path | None = None):
    """Load the Piper voice (about 0.2-0.4 s); a long-lived process keeps it."""
    from piper import PiperVoice  # heavy; keep out of the hook's import path

    return PiperVoice.load(str(model or config.MODEL), espeak_data_dir=_espeak_data_dir())


def synthesize(text: str, wav_path: str | Path, model: str | Path | None = None,
               length_scale: float | None = None, voice=None, sidecar: bool = False) -> float:
    """Render text to a 16-bit mono WAV with the HAL voice; returns its duration in seconds.

    text is passed to Piper as is; callers clean it first (textclean.speakable).
    voice is a voice from load_voice(); without one the model is loaded here.
    sidecar: also write <wav>.json, the text and its sentences' times (_write_sidecar).
    """
    from piper import SynthesisConfig

    voice = voice or load_voice(model)
    frames = 0
    chunks: list[int] = []
    rate = voice.config.sample_rate
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for chunk in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale)):
            data = chunk.audio_int16_bytes
            wav.writeframes(data)
            frames += len(data) // 2
            chunks.append(len(data) // 2)
    if sidecar:
        _write_sidecar(wav_path, text, chunks, rate)
    return frames / rate


def _exit_on_signal(signum, frame):  # noqa: ARG001
    raise SystemExit(0)


def _run_ack_worker(wav: Path | None = None) -> int:
    """Play an acknowledgement: the cached WAV of the phrase on stdin, or wav."""
    player = None
    try:
        wav = wav or ack_wav(sys.stdin.read().strip())
        player = subprocess.Popen(
            [*config.player_command(), str(wav)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        player.wait()
        return 0
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
        with _locked():
            if _pid_from(ACK_PID_FILE) == os.getpid():
                ACK_PID_FILE.unlink(missing_ok=True)


def _run_wav_worker(ticket: str, keep: bool) -> int:
    """Play a WAV rendered elsewhere (path on stdin); see play_prepared()."""
    wav = Path(sys.stdin.read().strip())
    player = None
    try:
        if ticket == "ack":
            return _run_ack_worker(wav)
        _wait_for_ack()
        with _locked():
            if _read(CURRENT_FILE) != ticket:
                config.log("worker: superseded before playback")
                return 0
            _silence_ack_slot()
            player = subprocess.Popen(
                [*config.player_command(), str(wav)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        player.wait()
        return 0
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
        if not keep:
            _unlink_wav(wav)
        if ticket != "ack":
            with _locked():
                if _speaker_pid() == os.getpid():
                    PID_FILE.unlink(missing_ok=True)


def run_worker(ticket: str, kind: str = "plain") -> int:
    """Body of the detached worker: read text on stdin, work out what to say
    (for a reply), synthesise, then play if this ticket is still the current one."""
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)
    if kind == "ack":
        return _run_ack_worker()
    if kind in ("wav", "wavkeep"):
        return _run_wav_worker(ticket, keep=kind == "wavkeep")
    if kind.startswith("progress:"):
        try:
            seq = int(kind.split(":", 1)[1])
        except ValueError:
            return 0
        return _run_progress_worker(ticket, seq)
    text = sys.stdin.read()
    wav = config.STATE_DIR / f"speech-{ticket}.wav"
    player = None
    try:
        if _read(CURRENT_FILE) != ticket:
            return 0
        cfg = config.load_config()
        if kind == "reply":
            text, how = prepare_reply(text, cfg)
            if not text or _read(CURRENT_FILE) != ticket:
                return 0
            if how in ("full", "lead"):  # Claude wrote these words itself; nothing to tell it
                LAST_SPOKEN_FILE.unlink(missing_ok=True)
            else:
                _write(LAST_SPOKEN_FILE, json.dumps(
                    {"how": how, "text": text, "time": time.time(), "reported": False}))
        started = time.monotonic()
        seconds = synthesize(text, wav, length_scale=cfg["length_scale"], sidecar=True)
        config.log(f"worker: {len(text)} chars -> {seconds:.1f}s of audio in {time.monotonic() - started:.2f}s")
        if seconds <= 0:
            return 0
        _wait_for_ack()
        with _locked():
            if _read(CURRENT_FILE) != ticket:
                config.log("worker: superseded before playback")
                return 0
            _silence_ack_slot()
            player = subprocess.Popen(
                [*config.player_command(), str(wav)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        player.wait()
        return 0
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
        _unlink_wav(wav)
        with _locked():
            if _speaker_pid() == os.getpid():
                PID_FILE.unlink(missing_ok=True)


_PROGRESS_WAIT = 20.0  # longest a progress line waits for the slot before giving up


def _run_progress_worker(token: str, seq: int) -> int:
    """Synthesise a progress line, wait until nothing else is playing, then
    play it in the acknowledgement slot, unless it has been superseded."""
    from . import progress

    text = sys.stdin.read().strip()
    wav = config.STATE_DIR / f"progress-{os.getpid()}-{time.time_ns()}.wav"
    player, registered = None, False
    try:
        if not text or _read(CURRENT_FILE) != token:
            return 0
        cfg = config.load_config()
        seconds = synthesize(text, wav, length_scale=cfg["length_scale"], sidecar=True)
        if seconds <= 0:
            return 0
        deadline = time.monotonic() + _PROGRESS_WAIT
        while player is None:
            with _locked():
                if _read(CURRENT_FILE) != token or config.is_muted() or not progress.is_latest(token, seq):
                    config.log("worker: progress line superseded before playback")
                    return 0
                if not _is_worker(_pid_from(ACK_PID_FILE)) and not _is_worker(_speaker_pid()):
                    _write(ACK_PID_FILE, json.dumps({"pid": os.getpid()}))
                    registered = True
                    player = subprocess.Popen(
                        [*config.player_command(), str(wav)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    break
            if time.monotonic() > deadline:
                config.log("worker: progress line dropped (the voice stayed busy)")
                return 0
            time.sleep(0.15)
        player.wait()
        return 0
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
        _unlink_wav(wav)
        if registered:
            with _locked():
                if _pid_from(ACK_PID_FILE) == os.getpid():
                    ACK_PID_FILE.unlink(missing_ok=True)
