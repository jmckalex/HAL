"""The front desk: a long-lived process that hears each prompt first.

    hal desk start     start it in the background, unless it is running
    hal desk stop      stop it
    hal desk status    whether it is running, and its models
    hal desk run       run it in the foreground
    hal desk ask TEXT  show what it would do with TEXT (nothing is spoken)
    hal desk turn TEXT judge whether TEXT is a finished request (for the listener)

It keeps a small fast model (Claude Haiku, a `claude -p` process in
stream-json mode, so a prompt costs no process start-up) and the Piper voice
loaded, and listens on a Unix socket in the state directory. For each prompt
the UserPromptSubmit hook hands it, the model answers in one of two forms:

    REPLY: ...   the remark is purely social, so the desk answers it aloud;
                 the hook blocks the prompt (or, with desk_block off, lets it
                 through and tells Claude it has been answered)
    ACK: ...     anything else: one sentence in line with the request
                 ("Certainly. I'll check tomorrow's forecast."), spoken while
                 Claude works on it

Speech goes through hal/speaker.py (social replies in the reply slot,
acknowledgements in the acknowledgement slot) and carries the token the
hook's stop left for that prompt, so a newer prompt, `hal stop`, mute, or a
reply that has already started all prevent stale speech.

Protocol: one JSON object per line, one request per connection.

    {"op": "prompt", "text", "token", "wait", "timeout", "ack"}
        wait false: answered at once with {"kind": "queued"}; an
            acknowledgement is spoken when ready (the model's, or a canned
            one if the model is slow or makes no sense).
        wait true: answered with {"kind": "ack"} as soon as the model's first
            word shows the remark is not social, {"kind": "social", "text"}
            once the reply is complete, or {"kind": "unavailable"} (no model,
            too slow, or an answer in neither form). After "ack" or "social"
            the client sends {"go": true} if it acts on the decision; only
            then does the desk speak.
        "ack" false in the request: acknowledgements are not spoken.
    {"op": "summarize", "text", "timeout"}  -> {"text"} or {"error"}
    {"op": "turn", "text", "timeout"}
        -> {"kind": "complete" | "incomplete", "seconds"} or {"kind": "unavailable", "why"}
        For the listener (hal/listener/turns.py): has someone talking to
        Claude finished, or paused mid-thought? Nothing is spoken, and the
        log records only the number of words.
    {"op": "classify", "text"}   -> {"kind", "text", "seconds"}; nothing spoken
    {"op": "status"}             -> PID, models, counters
    {"op": "quit"}

A spoken request arrives with the listener's label and marks ("[spoken 21:14]
...", hal/spoken.py). The desk judges the words alone (spoken.plain()) in every
request, as the hook already does, so its answers are the same as for the
same words typed.

One desk runs per state directory (an flock on desk.lock). It restarts its
model if that dies, replaces it with a fresh one after desk_max_turns prompts
(which bounds its conversation; the fresh one is started as soon as the old
one is done), and exits after desk_idle_minutes without a prompt. A second warm model writes spoken
summaries (desk_summaries); it is replaced after every summary, so each one
starts from a clean context. A third judges turns for the listener
(desk_turns); it is replaced after _TURN_MAX_TURNS judgements.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import config, spoken

LOCK_FILE = config.STATE_DIR / "desk.lock"
PID_FILE = config.STATE_DIR / "desk.pid"
SPAWNED_FILE = config.STATE_DIR / "desk.spawned"

_BOOTSTRAP = "import sys; sys.path.insert(0, sys.argv[1]); from hal.cli import main; sys.exit(main(sys.argv[2:]))"
_SPAWN_THROTTLE = 15.0      # seconds between automatic start attempts
_MAX_LINE = 8 * 1024 * 1024
_PROMPT_CHARS = 1500        # how much of a long prompt the desk's model sees
_GO_TIMEOUT = 1.0           # how long the desk waits for the client's "go"
_ACK_GRACE = 0.5            # extra time for a full acknowledgement after the wait limit
_MAX_REPLY_CHARS = 400
_MAX_ACK_CHARS = 200
_TURN_CHARS = 2000          # how much of a long turn the judge sees (its end)
_TURN_MAX_TURNS = 12        # judgements before the turn model is replaced


def socket_path(state_dir: Path | str | None = None) -> str:
    """The desk's socket: in the state directory, unless that path is too long
    for a Unix socket (104 bytes on macOS), then in /tmp. state_dir defaults
    to this process's (the listener passes its own)."""
    base = Path(state_dir) if state_dir is not None else config.STATE_DIR
    path = str(base / "desk.sock")
    if len(path.encode()) <= 100:
        return path
    digest = hashlib.sha1(str(base.resolve()).encode()).hexdigest()[:12]
    return f"/tmp/hal-desk-{os.getuid()}-{digest}.sock"


# ---------------------------------------------------------------------------
# The model's instructions and answers
# ---------------------------------------------------------------------------

def _with_speaker(text: str, speaker: str | None) -> str:
    """A remark the voice check says someone else made, with a note so the desk
    doesn't greet them by the user's name."""
    if not speaker or not text.strip():
        return text
    if speaker.endswith("?") or " and " in speaker or speaker.startswith("an unrecognised voice"):
        return f"(The speaker may not be the user: address no one by name.) {text}"
    return f"(Said by {speaker}, not by the user: if you use a name, use {speaker}.) {text}"


def desk_prompt(user_name: str = "") -> str:
    if user_name:
        name = f"The user is {user_name}. Use their name when greeting them or saying goodbye; otherwise leave it out."
    else:
        name = "Do not address the user by name."
    return f"""You are the front desk of a voice assistant called Claude. The user talks to Claude aloud and hears the answers in a calm, courteous, slightly formal voice. You speak as Claude, but the real work is done by a far more capable assistant behind you, which has the user's projects, files and tools, the web, and the whole earlier conversation. You hear each remark first, and you see only what the user has said to you, never the replies they were given.

Each remark is new: judge it on its own, and never refer to earlier remarks or to your earlier answers. Answer every remark with exactly one line, in one of two forms.

REPLY: <what you say>
Only for a remark that is purely social and stands on its own: a greeting, thanks, a farewell, how are you, a remark about their day, a compliment, a joke, idle chat. It must need no facts, no tools, and nothing said earlier. Answer it yourself in one or two short spoken sentences. Do not offer help or ask what you can do. A question or remark about the assistant itself is never social, however friendly: what it thinks, feels, likes, wants, remembers or is ("Do you dream?", "Do you like working this way?", "I think you downplay your contributions", "Are you conscious?"), or any request for its opinion. The assistant behind you must answer those in its own words; use ACK for them.

ACK: <what you say>
For everything else, and whenever you are unsure. The assistant behind you will deal with it; you only acknowledge it, in one short sentence that reflects what was asked and shows that work on it is starting, such as "Certainly. I'll check tomorrow's forecast." or "Of course. I'll run the tests now." Do not answer the question, guess at facts, promise a result, or ask anything.

Use ACK for any question of fact, any request for information or action, and anything about projects, code, files, sessions, schedules, weather or news. A greeting or thanks combined with a request is a request: use ACK.

A short reply such as "OK", "sounds good", "yes, go ahead", "no", "hmm", "why?", "the second one" or "read that again" may answer a question the user was just asked, or refer to something you cannot see. For it, answer exactly: ACK: Very well. When in doubt, use ACK.

You are Claude: never call yourself by another name, and never mention an assistant behind you, a front desk, or passing anything on. Speak as the one who is about to do the work. {name} Speak in plain words meant to be heard: no markdown, lists, emoji or symbols."""


_DECISION = re.compile(r"^[\s*_\"'`]*(REPLY|ACK)\b[\s*_]*[:.\-–—]?\s*(.*)$", re.I | re.S)


def parse(output: str | None) -> tuple[str, str] | None:
    """("social" | "ack", words to say) from the model's answer, or None if it
    is in neither form. Only the first non-empty line counts."""
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = _DECISION.match(lines[0])
    if not match:
        return None
    kind = "social" if match.group(1).upper() == "REPLY" else "ack"
    text = match.group(2).strip().strip("\"“”*_").strip()
    if not text or len(text) > (_MAX_REPLY_CHARS if kind == "social" else _MAX_ACK_CHARS):
        return None
    return kind, text


def turn_prompt() -> str:
    return """You decide whether a person talking to a voice assistant called Claude has finished saying what they want, or has paused in the middle of a thought. Their speech is transcribed as they talk. Each time they pause you are shown everything they have said so far in this turn, between <speech> tags. They often pause to think, sometimes for a long time, and then go on.

The transcriber guesses the punctuation from the voice, so a full stop or question mark may appear where the speaker only paused. An ellipsis, a comma, or no punctuation at the end suggests they broke off.

Answer with exactly one word.

COMPLETE: what has been said so far is a whole request, question, instruction or remark that the assistant could answer or act on now, and nothing suggests that more is coming. A greeting, thanks or a brief answer such as "Yes, please do" is complete.

INCOMPLETE: the speaker is evidently mid-thought: the last sentence breaks off or lacks a part it needs; it ends on a word that needs something after it; or so far there is only a preamble, a topic or an announcement ("I have a question about the paper", "There are three things I'd like you to do", "Another topic", "Regarding the draft") without the request itself.

Judge only whether the speech is finished, not whether it is clear, sensible or polite. Never answer or carry out what was said. Each message is a separate case: ignore earlier ones. If you are unsure, answer COMPLETE."""


_TURN_VERDICT = re.compile(r"^[\s*_\"'`]*(COMPLETE|INCOMPLETE)\b", re.I)


def parse_turn(output: str | None) -> str | None:
    """"complete" | "incomplete" from the turn model's answer, or None."""
    match = _TURN_VERDICT.match(output or "")
    return match.group(1).lower() if match else None


def early_turn(partial: str) -> str | None:
    """What the first letters of a turn judgement show: "complete",
    "incomplete", "nonsense", or None if it is too soon to tell."""
    head = partial.lstrip(" \t\r\n*_\"'`").upper()
    if head.startswith("COM"):
        return "complete"
    if head.startswith("INC"):
        return "incomplete"
    if head and not ("COMPLETE".startswith(head) or "INCOMPLETE".startswith(head)):
        return "nonsense"
    return None


def turn_message(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _TURN_CHARS:
        text = "… " + text[-_TURN_CHARS:]
    return f"<speech>{text}</speech>"


def judge_turn(text: str, timeout: float = 2.5, state_dir=None) -> dict:
    """Ask a running desk whether a turn is finished (the listener has its own
    client, hal/listener/judge.py; this one is for `hal desk turn` and tests)."""
    conn = connect_to(socket_path(state_dir))
    if conn is None:
        return {"kind": "down"}
    try:
        conn.send({"op": "turn", "text": text, "timeout": timeout})
        return conn.recv(timeout + 0.5) or {"kind": "unavailable", "why": "no answer in time"}
    except OSError as exc:
        return {"kind": "unavailable", "why": f"connection: {exc}"}
    finally:
        conn.close()


def early_kind(partial: str) -> str | None:
    """What the first characters of an answer show: "ack", "social",
    "nonsense", or None if it is too soon to tell."""
    head = partial.lstrip(" \t\r\n*_\"'`").upper()
    if head.startswith("ACK"):
        return "ack"
    if head.startswith("REPLY"):
        return "social"
    if head and not ("ACK".startswith(head) or "REPLY".startswith(head)):
        return "nonsense"
    return None


# ---------------------------------------------------------------------------
# Client side (the hook, the summariser, the CLI)
# ---------------------------------------------------------------------------

class _Conn:
    """A connection to the desk: newline-delimited JSON with deadlines."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def send(self, message: dict) -> None:
        self.sock.settimeout(1.0)
        self.sock.sendall((json.dumps(message) + "\n").encode())

    def reply(self, message: dict) -> bool:
        """send() for the desk's side: False if the client has gone."""
        try:
            self.send(message)
            return True
        except OSError:
            return False

    def recv(self, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.sock.settimeout(remaining)
            try:
                data = self.sock.recv(65536)
            except (socket.timeout, OSError):
                return None
            if not data:
                return None
            self.buf += data
            if len(self.buf) > _MAX_LINE:
                return None
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            message = json.loads(line)
        except ValueError:
            return None
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def connect(timeout: float = 0.3) -> _Conn | None:
    return connect_to(socket_path(), timeout)


def connect_to(path: str, timeout: float = 0.3) -> _Conn | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(path)
    except OSError:
        sock.close()
        return None
    return _Conn(sock)


def request(message: dict, timeout: float) -> dict | None:
    """Send one request and return the answer, or None (not running, timed out)."""
    conn = connect()
    if conn is None:
        return None
    try:
        conn.send(message)
        return conn.recv(timeout)
    except OSError:
        return None
    finally:
        conn.close()


def status() -> dict | None:
    return request({"op": "status"}, 2.0)


def consult(prompt: str, token: str, wait: bool, timeout: float, acknowledge: bool = True) -> dict:
    """Hand a prompt to the desk (for the hook). Returns {"kind": ...}:

        "down"         nothing is listening
        "queued"       (wait false) the desk will speak an acknowledgement
        "ack"          the desk will speak an acknowledgement; the prompt goes on
        "social"       the desk is answering aloud with "text"
        "unavailable"  anything else ("why" says what): fall back
    """
    conn = connect()
    if conn is None:
        return {"kind": "down"}
    try:
        conn.send({"op": "prompt", "text": prompt, "token": token, "wait": wait,
                   "timeout": timeout, "ack": acknowledge})
        answer = conn.recv(timeout + 0.25 if wait else 0.75)
        kind = answer.get("kind") if answer else None
        if kind == "queued" and not wait:
            return {"kind": "queued"}
        if wait and kind == "ack":
            conn.send({"go": True})
            return {"kind": "ack"}
        text = answer.get("text") if answer else None
        if wait and kind == "social" and isinstance(text, str) and 0 < len(text.strip()) <= 2 * _MAX_REPLY_CHARS:
            conn.send({"go": True})
            return {"kind": "social", "text": " ".join(text.split())}
        why = "no answer in time" if answer is None else (answer.get("why") or f"answer {kind!r}")
        return {"kind": "unavailable", "why": str(why)}
    except OSError as exc:
        return {"kind": "unavailable", "why": f"connection: {exc}"}
    finally:
        conn.close()


def spawn(force: bool = False) -> bool:
    """Start the desk in the background and return at once, unless an
    automatic start was attempted moments ago. A desk that finds another
    already running exits by itself."""
    try:
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        if not force:
            try:
                if time.time() - SPAWNED_FILE.stat().st_mtime < _SPAWN_THROTTLE:
                    return False
            except OSError:
                pass
        SPAWNED_FILE.touch()
        env = {k: v for k, v in os.environ.items() if k != "HAL_VOICE"}
        with open(config.LOG_FILE, "ab") as log_file:
            subprocess.Popen(
                [sys.executable, "-E", "-P", "-c", _BOOTSTRAP, str(config.ROOT), "desk", "run"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log_file,
                cwd=str(config.ROOT), env=env, start_new_session=True,
            )
        return True
    except OSError as exc:
        config.log(f"desk: could not start: {exc!r}")
        return False


# ---------------------------------------------------------------------------
# The warm model: one `claude -p` process in stream-json mode
# ---------------------------------------------------------------------------

def model_command(role: str, system_prompt: str, cfg: dict, stream: bool) -> list[str]:
    override = os.environ.get("HAL_WARM_MODEL")
    if override:
        return [*shlex.split(override), "--role", role, "--system-prompt", system_prompt]
    claude = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
    model = cfg["desk_model"] if role in ("desk", "turn") else cfg["summary_model"]
    command = [
        claude, "-p",
        "--model", str(model),
        "--effort", "low",
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt", system_prompt,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if stream:
        command.append("--include-partial-messages")
    return command


class Turn:
    """One message to the model and its answer, filled in as it streams."""

    def __init__(self):
        self.cond = threading.Condition()
        self.sent = time.monotonic()
        self.text = ""          # streamed so far
        self.first = None       # seconds to the first text
        self.result = None      # the whole answer, once done
        self.error = None
        self.done = False
        self.seconds = None     # to the whole answer

    def add(self, text: str) -> None:
        with self.cond:
            if self.first is None:
                self.first = time.monotonic() - self.sent
            self.text += text
            self.cond.notify_all()

    def finish(self, result: str | None, error: str | None = None) -> None:
        with self.cond:
            if self.done:
                return
            self.result = result if result is not None else (self.text if not error else None)
            self.error = error
            self.done = True
            self.seconds = time.monotonic() - self.sent
            self.cond.notify_all()

    def wait(self, predicate, deadline: float) -> bool:
        with self.cond:
            while not predicate(self):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(remaining)
            return True


class WarmModel:
    """A model process kept running between messages. Its answers arrive in
    the order the messages were sent; a reader thread matches them up."""

    def __init__(self, role: str, system_prompt: str, cfg: dict, stream: bool):
        self.role = role
        self.system_prompt = system_prompt
        self.lock = threading.Lock()
        self.pending: list[Turn] = []
        self.turns = 0
        self.started = time.monotonic()
        self.dead = False
        env = {k: v for k, v in os.environ.items() if k != "HAL_VOICE"}  # the model must not speak
        with open(config.LOG_FILE, "ab") as log_file:
            self.proc = subprocess.Popen(
                model_command(role, system_prompt, cfg, stream),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file,
                cwd=tempfile.gettempdir(), env=env,
            )
        threading.Thread(target=self._read, name=f"read-{role}", daemon=True).start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return not self.dead and self.proc.poll() is None

    def busy(self) -> bool:
        with self.lock:
            return bool(self.pending)

    def ask(self, text: str) -> Turn:
        turn = Turn()
        line = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
        with self.lock:
            if self.dead:
                turn.finish(None, "model not running")
                return turn
            self.pending.append(turn)
            self.turns += 1
            try:
                self.proc.stdin.write(line.encode())
                self.proc.stdin.flush()
            except (OSError, ValueError):
                self._die_locked("could not write to the model")
        return turn

    def _read(self) -> None:
        try:
            for raw in self.proc.stdout:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "stream_event":
                    delta = (event.get("event") or {}).get("delta") or {}
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        with self.lock:
                            turn = self.pending[0] if self.pending else None
                        if turn is not None:
                            turn.add(delta["text"])
                elif kind == "result":
                    with self.lock:
                        turn = self.pending.pop(0) if self.pending else None
                    if turn is None:
                        continue
                    if event.get("is_error") or event.get("subtype") not in (None, "success"):
                        turn.finish(None, f"model error ({event.get('subtype')})")
                    else:
                        result = event.get("result")
                        turn.finish(result if isinstance(result, str) else None)
        except (OSError, ValueError):
            pass
        with self.lock:
            self._die_locked("the model exited")

    def _die_locked(self, why: str) -> None:
        self.dead = True
        pending, self.pending = self.pending, []
        for turn in pending:
            turn.finish(None, why)

    def close(self) -> None:
        """Let the process finish (its stdin closes), killing it if it lingers."""
        with self.lock:
            self.dead = True
        try:
            self.proc.stdin.close()
        except OSError:
            pass

        def reap():
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()

        threading.Thread(target=reap, daemon=True).start()


# ---------------------------------------------------------------------------
# The desk itself
# ---------------------------------------------------------------------------

def _clip(text: str, n: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class Desk:
    def __init__(self):
        self.cfg = config.load_config()
        self.voice = None
        self.synth_lock = threading.Lock()
        self.models_lock = threading.Lock()
        self.models: dict[str, WarmModel | None] = {"desk": None, "summary": None, "turn": None}
        self.failures = {"desk": 0, "summary": 0, "turn": 0}
        self.next_start = {"desk": 0.0, "summary": 0.0, "turn": 0.0}
        self.restarts = {"desk": 0, "summary": 0, "turn": 0}
        self.stopping = threading.Event()
        self.last_activity = time.monotonic()
        self.started = time.time()
        self.counts = {"prompts": 0, "social": 0, "ack": 0, "fallback": 0, "summaries": 0, "turns": 0}
        self.workers: list[subprocess.Popen] = []
        self.workers_lock = threading.Lock()
        self.config_stamp = self._config_stamp()

    @staticmethod
    def _config_stamp():
        try:
            st = config.CONFIG_FILE.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    # -- models -------------------------------------------------------------

    def _wanted(self, role: str) -> tuple[str, int, bool] | None:
        """(system prompt, turns before replacement, stream?) for a role, or None if not wanted."""
        if role == "desk":
            return desk_prompt(self.cfg["user_name"]), self.cfg["desk_max_turns"], True
        if role == "turn":
            return (turn_prompt(), _TURN_MAX_TURNS, True) if self.cfg["desk_turns"] else None
        if self.cfg["desk_summaries"]:
            from .summarize import PROMPT
            return PROMPT, 1, False
        return None

    def _start_model(self, role: str, wanted: tuple[str, int, bool]) -> WarmModel | None:
        try:
            model = WarmModel(role, wanted[0], self.cfg, stream=wanted[2])
        except OSError as exc:
            config.log(f"desk: could not start the {role} model: {exc!r}")
            self._note_failure(role, 0.0)
            return None
        config.log(f"desk: {role} model started (pid {model.pid})")
        return model

    def _note_failure(self, role: str, lived: float) -> None:
        self.failures[role] = 1 if lived > 60 else self.failures[role] + 1
        delay = min(300.0, 2.0 ** self.failures[role])
        self.next_start[role] = time.monotonic() + delay

    def _maintain(self) -> None:
        """Start, restart or replace each model as needed."""
        for role in ("desk", "summary", "turn"):
            wanted = self._wanted(role)
            with self.models_lock:
                model = self.models[role]
                if wanted is None:
                    if model is not None:
                        model.close()
                        self.models[role] = None
                    continue
                if model is not None and not model.alive():
                    lived = time.monotonic() - model.started
                    config.log(f"desk: {role} model (pid {model.pid}) exited after {lived:.0f}s "
                               f"(code {model.proc.poll()}); restarting")
                    model.close()
                    self.models[role] = model = None
                    self.restarts[role] += 1
                    self._note_failure(role, lived)
                if model is None:
                    if time.monotonic() >= self.next_start[role]:
                        self.models[role] = self._start_model(role, wanted)
                    continue
                stale = model.system_prompt != wanted[0] or model.turns >= wanted[1]
                if stale and not model.busy():
                    fresh = self._start_model(role, wanted)
                    if fresh is not None:
                        self.models[role] = fresh
                        model.close()

    def _model(self, role: str) -> WarmModel | None:
        with self.models_lock:
            model = self.models[role]
            return model if model is not None and model.alive() else None

    # -- speech -------------------------------------------------------------

    def _say(self, text: str, slot: str, token: str) -> str:
        """Speak text in a slot if the prompt's token is still current. Returns
        what happened, for the log."""
        from . import speaker
        from .textclean import speakable

        cfg = self.cfg
        spoken = speakable(text, cfg["max_chars"], cfg["overflow_message"])  # cleans and pronounces
        if not spoken:
            return "nothing speakable"
        if speaker.current() != token:
            return "superseded before synthesis"
        wav = config.STATE_DIR / f"speech-desk-{time.time_ns()}.wav"
        with self.synth_lock:
            speaker.synthesize(spoken, wav, length_scale=cfg["length_scale"], voice=self.voice, sidecar=True)
        proc = speaker.play_prepared(wav, slot, token)
        if proc is None:
            return "superseded before playback"
        self._track(proc)
        return f"said {spoken!r}"

    def _say_canned(self, token: str) -> str:
        from . import speaker

        if speaker.current() != token:
            return "superseded"
        phrase = speaker.choose_ack_phrase(self.cfg)
        with self.synth_lock:
            wav = speaker.ack_wav(phrase, self.cfg, voice=self.voice)
        proc = speaker.play_prepared(wav, "ack", token, keep=True)
        if proc is None:
            return "superseded"
        self._track(proc)
        return f"said the canned {phrase!r}"

    def _prerender(self) -> None:
        """Render the canned acknowledgements now, so a fallback costs no synthesis."""
        from . import speaker
        from .hook import BRIEF_ANSWER_ACK

        for phrase in [*self.cfg["ack_phrases"], BRIEF_ANSWER_ACK]:
            try:
                with self.synth_lock:
                    speaker.ack_wav(phrase, self.cfg, voice=self.voice)
            except Exception as exc:
                config.log(f"desk: could not render {phrase!r}: {exc!r}")
                return

    def _track(self, proc: subprocess.Popen) -> None:
        with self.workers_lock:
            self.workers = [p for p in self.workers if p.poll() is None]
            self.workers.append(proc)

    # -- requests -----------------------------------------------------------

    def _on_prompt(self, conn: _Conn, req: dict) -> None:
        received = time.monotonic()
        self.last_activity = received
        self.counts["prompts"] += 1
        text = spoken.plain(req.get("text")) if isinstance(req.get("text"), str) else ""
        text = _with_speaker(text, spoken.speaker(req.get("text")) if isinstance(req.get("text"), str) else None)
        token = req.get("token") if isinstance(req.get("token"), str) else ""
        wait = bool(req.get("wait"))
        acknowledge = req.get("ack", True) is not False
        try:
            timeout = min(3.5, max(0.2, float(req.get("timeout") or self.cfg["desk_wait_seconds"])))
        except (TypeError, ValueError):
            timeout = self.cfg["desk_wait_seconds"]
        words = len(text.split())
        model = self._model("desk")
        if model is None or not text.strip() or not token:
            conn.reply({"kind": "unavailable", "why": "no model" if model is None else "bad request"})
            self.counts["fallback"] += 1
            config.log(f"desk: prompt of {words} words: unavailable (no model running)")
            return
        turn = model.ask(text if len(text) <= _PROMPT_CHARS else text[:_PROMPT_CHARS] + " …")
        ack_deadline = received + max(timeout, self.cfg["desk_wait_seconds"]) + _ACK_GRACE

        if not wait:
            conn.reply({"kind": "queued"})
            conn.close()
            self._acknowledge(turn, token, ack_deadline, acknowledge, received, words, social_ok=False)
            return

        decided = turn.wait(lambda t: t.done or early_kind(t.text) in ("ack", "nonsense"), received + timeout)
        early = early_kind(turn.text)
        if not decided:
            conn.reply({"kind": "unavailable", "why": "the model was too slow"})
            self.counts["fallback"] += 1
            config.log(f"desk: prompt of {words} words: no decision within {timeout:.1f}s; "
                       f"fell back (early {early!r})")
            return
        parsed = parse(turn.result) if turn.done else None
        if early == "ack" or (parsed and parsed[0] == "ack"):
            if not conn.reply({"kind": "ack"}) or not self._go(conn):
                config.log(f"desk: prompt of {words} words: ack, but the hook had gone")
                return
            self._acknowledge(turn, token, ack_deadline, acknowledge, received, words, social_ok=False)
            return
        if parsed and parsed[0] == "social":
            reply = parsed[1]
            if not conn.reply({"kind": "social", "text": reply}) or not self._go(conn):
                config.log(f"desk: prompt of {words} words: social, but the hook had gone; nothing said")
                return
            self.counts["social"] += 1
            outcome = self._say(reply, "reply", token)
            config.log(f"desk: prompt of {words} words -> social in {turn.seconds:.2f}s "
                       f"(first word {turn.first or 0:.2f}s); {outcome} "
                       f"[{time.monotonic() - received:.2f}s]")
            return
        why = turn.error or f"answer in neither form: {_clip(turn.result or turn.text)!r}"
        conn.reply({"kind": "unavailable", "why": why})
        self.counts["fallback"] += 1
        config.log(f"desk: prompt of {words} words: fell back ({why})")

    def _go(self, conn: _Conn) -> bool:
        answer = conn.recv(_GO_TIMEOUT)
        return bool(answer and answer.get("go") is True)

    def _acknowledge(self, turn: Turn, token: str, deadline: float, acknowledge: bool,
                     received: float, words: int, social_ok: bool) -> None:
        """Speak the model's acknowledgement when it is complete, or a canned
        one if it is late or in the wrong form."""
        self.counts["ack"] += 1
        if not acknowledge:
            return
        turn.wait(lambda t: t.done, deadline)
        parsed = parse(turn.result) if turn.done else None
        if parsed and parsed[0] == "ack":
            outcome = self._say(parsed[1], "ack", token)
            how = f"ack in {turn.seconds:.2f}s (first word {turn.first or 0:.2f}s)"
        else:
            if not turn.done:
                why = "too slow"
            elif turn.error:
                why = turn.error
            elif parsed:
                why = "social answer to a long prompt"
            else:
                why = f"answer in neither form: {_clip(turn.result or '')!r}"
            if turn.done and not turn.error and parsed:
                # Haiku judged a long remark to be social (thanks, a goodnight). Claude
                # answers long remarks itself, and "I'll look into that" would be the
                # wrong thing to say first, so say nothing.
                outcome = "said nothing"
            else:
                outcome = self._say_canned(token)
            how = f"no usable ack ({why})"
        config.log(f"desk: prompt of {words} words -> {how}; {outcome} [{time.monotonic() - received:.2f}s]")

    def _on_summarize(self, conn: _Conn, req: dict) -> None:
        received = time.monotonic()
        self.last_activity = received
        text = req.get("text") if isinstance(req.get("text"), str) else ""
        try:
            timeout = max(1.0, float(req.get("timeout") or self.cfg["summary_timeout"]) - 0.3)
        except (TypeError, ValueError):
            timeout = float(self.cfg["summary_timeout"])
        model = self._model("summary")
        if model is None or not text.strip():
            conn.reply({"error": "no summary model running" if model is None else "nothing to summarise"})
            return
        turn = model.ask(text)
        if not turn.wait(lambda t: t.done, received + timeout):
            conn.reply({"error": "too slow"})
            return
        if turn.error or not (turn.result or "").strip():
            conn.reply({"error": turn.error or "empty summary"})
            return
        self.counts["summaries"] += 1
        conn.reply({"text": turn.result.strip(), "seconds": round(turn.seconds, 2)})

    def _on_turn(self, conn: _Conn, req: dict) -> None:
        """Has the speaker finished? For the listener; nothing is spoken, and
        only the number of words is logged."""
        received = time.monotonic()
        self.last_activity = received
        text = spoken.plain(req.get("text")) if isinstance(req.get("text"), str) else ""
        try:
            timeout = min(5.0, max(0.2, float(req.get("timeout") or 2.5)))
        except (TypeError, ValueError):
            timeout = 2.5
        words = len(text.split())
        model = self._model("turn")
        if model is None or not text.strip():
            why = ("turn judgements are off (desk_turns)" if not self.cfg["desk_turns"] else
                   "no turn model running") if model is None else "no text"
            conn.reply({"kind": "unavailable", "why": why})
            config.log(f"desk: turn of {words} words: unavailable ({why})")
            return
        self.counts["turns"] += 1
        turn = model.ask(turn_message(text))
        decided = turn.wait(lambda t: t.done or early_turn(t.text) is not None, received + timeout)
        early = early_turn(turn.text)
        kind = early if early in ("complete", "incomplete") else (parse_turn(turn.result) if turn.done else None)
        seconds = time.monotonic() - received
        if kind is None:
            why = ("the model was too slow" if not decided else
                   turn.error or f"answer in neither form: {_clip(turn.result or turn.text)!r}")
            conn.reply({"kind": "unavailable", "why": why})
            config.log(f"desk: turn of {words} words: no judgement ({why})")
            return
        conn.reply({"kind": kind, "seconds": round(seconds, 2)})
        config.log(f"desk: turn of {words} words judged {kind} in {seconds:.2f}s "
                   f"(first word {turn.first or 0:.2f}s)")

    def _on_classify(self, conn: _Conn, req: dict) -> None:
        self.last_activity = time.monotonic()
        text = spoken.plain(req.get("text")) if isinstance(req.get("text"), str) else ""
        model = self._model("desk")
        if model is None or not text.strip():
            conn.reply({"kind": "unavailable", "why": "no model running" if model is None else "no text"})
            return
        turn = model.ask(text[:_PROMPT_CHARS])
        turn.wait(lambda t: t.done, time.monotonic() + 15)
        parsed = parse(turn.result)
        conn.reply({"kind": parsed[0] if parsed else "unavailable", "text": parsed[1] if parsed else "",
                   "raw": turn.result or turn.text, "error": turn.error,
                   "seconds": round(getattr(turn, "seconds", 0.0), 2), "first": round(turn.first or 0.0, 2)})

    def _status(self) -> dict:
        def describe(role):
            model = self.models[role]
            if model is None:
                return None
            return {"pid": model.pid, "alive": model.alive(), "turns": model.turns,
                    "age": round(time.monotonic() - model.started), "restarts": self.restarts[role]}

        idle = time.monotonic() - self.last_activity
        return {"pid": os.getpid(), "started": self.started, "uptime": round(time.time() - self.started),
                "socket": socket_path(), "desk_model": describe("desk"), "summary_model": describe("summary"),
                "turn_model": describe("turn"),
                "counts": dict(self.counts), "idle": round(idle),
                "idle_exit_in": round(self.cfg["desk_idle_minutes"] * 60 - idle),
                "user_name": self.cfg["user_name"], "voice_loaded": self.voice is not None}

    def _serve(self, sock: socket.socket) -> None:
        conn = _Conn(sock)
        try:
            req = conn.recv(2.0)
            op = req.get("op") if req else None
            if op == "prompt":
                self._on_prompt(conn, req)
            elif op == "summarize":
                self._on_summarize(conn, req)
            elif op == "turn":
                self._on_turn(conn, req)
            elif op == "classify":
                self._on_classify(conn, req)
            elif op == "status":
                conn.reply(self._status())
            elif op == "quit":
                conn.reply({"ok": True})
                self.stopping.set()
            elif req is not None:
                conn.reply({"error": f"unknown op {op!r}"})
        except Exception as exc:  # one bad request must not take the desk down
            import traceback
            config.log(f"desk: error serving a request: {exc!r}\n{traceback.format_exc()}")
        finally:
            conn.close()

    # -- lifecycle ----------------------------------------------------------

    def _supervise(self) -> None:
        while not self.stopping.wait(0.5):
            try:
                stamp = self._config_stamp()
                if stamp != self.config_stamp:
                    self.config_stamp, self.cfg = stamp, config.load_config()
                self._maintain()
                with self.workers_lock:
                    self.workers = [p for p in self.workers if p.poll() is None]
                if time.monotonic() - self.last_activity > self.cfg["desk_idle_minutes"] * 60:
                    config.log(f"desk: idle for {self.cfg['desk_idle_minutes']:g} minutes; exiting")
                    self.stopping.set()
            except Exception as exc:
                import traceback
                config.log(f"desk: supervisor error: {exc!r}\n{traceback.format_exc()}")

    def run(self) -> int:
        import fcntl

        from . import speaker

        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(lock_fd)
            print("hal desk: already running", file=sys.stderr)
            return 0
        started = time.monotonic()
        speaker._write(PID_FILE, json.dumps({"pid": os.getpid(), "started": time.time()}))
        path = socket_path()
        server = bound = None
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, lambda *_: self.stopping.set())
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            try:
                self.voice = speaker.load_voice()
            except Exception as exc:
                config.log(f"desk: cannot load the voice ({exc!r}); not starting")
                return 1
            self._maintain()
            threading.Thread(target=self._prerender, name="prerender", daemon=True).start()
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path)
            os.chmod(path, 0o600)
            bound = os.stat(path).st_ino
            server.listen(16)
            server.settimeout(0.5)
            threading.Thread(target=self._supervise, name="supervise", daemon=True).start()
            config.log(f"desk: ready in {time.monotonic() - started:.2f}s (pid {os.getpid()}, socket {path})")
            while not self.stopping.is_set():
                try:
                    sock, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stopping.is_set():
                        break
                    time.sleep(0.1)
                    continue
                threading.Thread(target=self._serve, args=(sock,), daemon=True).start()
            return 0
        finally:
            if server is not None:
                server.close()
                try:
                    if os.stat(path).st_ino == bound:
                        os.unlink(path)
                except OSError:
                    pass
            with self.models_lock:
                for role, model in self.models.items():
                    if model is not None:
                        model.close()
                        try:
                            model.proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            model.proc.kill()
            try:
                if json.loads(PID_FILE.read_text()).get("pid") == os.getpid():
                    PID_FILE.unlink()
            except (OSError, ValueError, AttributeError):
                pass
            config.log("desk: stopped")
            os.close(lock_fd)


# ---------------------------------------------------------------------------
# Command line: hal desk ...
# ---------------------------------------------------------------------------

def _desk_pid() -> int:
    try:
        return int(json.loads(PID_FILE.read_text()).get("pid", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _is_desk(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        out = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "hal.cli" in out and " desk run" in out


def main(args: list[str]) -> int:
    action = args[0] if args else "status"
    cfg = config.load_config()
    if action == "run":
        return Desk().run()
    if action == "autostart":  # `hal claude`: start it if it is enabled; quiet, never fails
        if cfg["front_desk"] and status() is None:
            spawn(force=True)
        return 0
    if action == "start":
        if status() is not None:
            print("front desk: already running")
            return 0
        spawn(force=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            info = status()
            if info is not None:
                print(f"front desk: running (pid {info['pid']})")
                if not cfg["front_desk"]:
                    print("note: front_desk is false in config.json, so the hooks don't use it")
                return 0
            time.sleep(0.1)
        print(f"front desk: did not start; see {config.LOG_FILE}", file=sys.stderr)
        return 1
    if action == "stop":
        pid = _desk_pid()
        answered = request({"op": "quit"}, 2.0) is not None
        running = answered or _is_desk(pid)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and _is_desk(pid):
            time.sleep(0.1)
        if _is_desk(pid):
            os.kill(pid, signal.SIGTERM)
        print("front desk: stopped" if running else "front desk: not running")
        return 0
    if action == "status":
        info = status()
        if "--json" in args[1:]:
            print(json.dumps(info))
            return 0
        enabled = "on" if cfg["front_desk"] else "off (front_desk is false in config.json)"
        if info is None:
            print(f"front desk: not running; {enabled}")
            return 0
        print(f"front desk: running (pid {info['pid']}, up {info['uptime']} s); {enabled}")
        for role in ("desk_model", "summary_model", "turn_model"):
            m = info.get(role)
            print(f"  {role.replace('_', ' ')}: " + (f"pid {m['pid']}, {m['turns']} turns, "
                                                   f"{m['restarts']} restarts" if m else "none"))
        c = info["counts"]
        print(f"  prompts {c['prompts']}: {c['social']} answered, {c['ack']} acknowledged, "
              f"{c['fallback']} fell back; {c['summaries']} summaries; {c.get('turns', 0)} turns judged")
        print(f"  idle {info['idle']} s; exits after {cfg['desk_idle_minutes']:g} idle minutes")
        return 0
    if action == "ask":
        text = " ".join(args[1:]).strip()
        if not text:
            print("usage: hal desk ask TEXT", file=sys.stderr)
            return 2
        info = request({"op": "classify", "text": text}, 20)
        if info is None:
            print("front desk: not running (hal desk start)", file=sys.stderr)
            return 1
        label = {"social": "answers itself", "ack": "hands on to Claude"}.get(info["kind"], info["kind"])
        print(f"{label} ({info.get('seconds')} s): {info.get('text') or info.get('raw') or info.get('error')}")
        return 0
    if action == "turn":
        text = " ".join(args[1:]).strip()
        if not text:
            print("usage: hal desk turn TEXT", file=sys.stderr)
            return 2
        info = judge_turn(text, 4.0)
        if info.get("kind") == "down":
            print("front desk: not running (hal desk start)", file=sys.stderr)
            return 1
        print(f"{info.get('kind')} ({info.get('seconds', '-')} s){': ' + info['why'] if info.get('why') else ''}")
        return 0
    print("usage: hal desk start|stop|status|run|ask TEXT|turn TEXT", file=sys.stderr)
    return 2
