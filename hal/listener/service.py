"""The listening loop: audio -> gate -> VAD -> segmenter -> speech-to-text -> decision -> proxy.

Single-threaded apart from the audio callback (and the front desk's
judgement of a turn, which is polled), so every decision happens in one place
and in order. Captured audio lives only in memory (frames in a queue, an
utterance's samples until it is transcribed; with turn_taking "semantic", the
samples of an addressed turn until the turn is sent, so that it can be
transcribed afresh as a whole) and is never written to disk. Only addressed
utterances are logged with their words.

With duplex "aec" the source is a DuplexSource (duplex.py): while it cancels
the voice's echo (or the output is headphones) the voice gate is not applied,
and speech over the voice interrupts it (bargein.py, LISTENER.md "Interrupting
the voice").

What is typed carries a label and marks (annotate.py, hal/spoken.py; LISTENER.md
"What the model is told"): "[spoken 21:14]", corrections from the lexicon,
doubtful words, long pauses, and what the voice was saying if the request cut
it off. Transcripts are sanitised first, so that nothing heard can pass for a
label or mark.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass

from hal import spoken as spoken_label

from . import annotate, logs, settings
from .bargein import BargeIn, Interrupted, Spoken, is_resume_phrase, level_db, resolve
from .segmenter import Discarded, Segmenter, Utterance
from .turns import Outcome, TurnTaker, opens_with_name

_DURING_TURN_MAX = 1800.0   # a turn that never reported its end stops counting as "in progress" after this


class VoiceOut:
    """Speak through the existing HAL voice: `bin/hal say --raw --wait TEXT`.

    --wait keeps the process alive while the voice speaks, so busy() tells the
    gate which speech was the listener's own (it doesn't open the follow-up
    window). Respects `hal off` (the mute flag)."""

    def __init__(self, command: list[str] | None = None):
        self.command = command or [str(settings.ROOT / "bin" / "hal"), "say", "--raw", "--wait"]
        self.proc: subprocess.Popen | None = None
        self.said: list[str] = []

    def __call__(self, text: str) -> None:
        if (settings.state_dir() / "muted").exists():
            logs.log(f"voice muted; not saying {text!r}")
            return
        self.said.append(text)
        try:
            self.proc = subprocess.Popen(self.command + [text], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            logs.log(f"could not speak: {exc}")

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class SilentVoice:
    """Stand-in that only prints what would be said (`run --silent`)."""

    def __init__(self, console=print):
        self.console = console
        self.said: list[str] = []

    def __call__(self, text: str) -> None:
        self.said.append(text)
        self.console(f'(would say: "{text}")')

    def busy(self) -> bool:
        return False


def stop_hal_voice() -> None:
    """Silence HAL's voice and what is queued behind it (`bin/hal stop`), without waiting."""
    try:
        subprocess.Popen([str(settings.ROOT / "bin" / "hal"), "stop"], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        logs.log(f"could not stop the voice: {exc}")


def play_cue(path: str) -> None:
    """The soft sound for turn_cue "sound": afplay at low volume (HAL_PLAYER in tests). Respects `hal off`."""
    if (settings.state_dir() / "muted").exists():
        return
    override = os.environ.get("HAL_PLAYER")
    command = shlex.split(override) if override else ["/usr/bin/afplay", "-v", "0.25"]
    try:
        subprocess.Popen(command + [path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        logs.log(f"could not play the cue: {exc}")


@dataclass
class Counts:
    utterances: int = 0
    submitted: int = 0
    ignored: int = 0
    discarded: int = 0
    noise: int = 0
    gated_seconds: float = 0.0   # audio dropped unheard while the voice was speaking
    fragments: int = 0           # addressed fragments held in turns (turn_taking "semantic")
    cancelled: int = 0           # turns discarded by a cancel phrase
    barge_ins: int = 0           # times speech over the voice paused it (duplex "aec")
    echo: int = 0                # utterances discarded as the voice's own echo
    voice_cut: int = 0           # times the voice was silenced by speech over it
    voice_resumed: int = 0       # ... and resumed ("go on", or it was only echo or noise)
    corrected: int = 0           # words changed by the lexicon in what was typed (each marked)
    flagged: int = 0             # words flagged with a suggestion from the lexicon
    doubtful: int = 0            # words marked [?]
    voice_checked: int = 0       # utterances scored against the voiceprint (voice_check)
    other_voice: int = 0         # ... ignored as another voice (voice_check "enforce")
    voice_learned: int = 0       # ... of yours added to the voiceprint (voice_learn)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def socket_alive(path: str | None) -> bool:
    """Does the proxy's control socket accept connections? (A dead proxy's
    socket refuses them even while its process lingers as a zombie.)"""
    if not path:
        return True
    import socket

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(path)
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except OSError:
        return True   # busy or slow is not gone


class Listener:
    def __init__(self, cfg: dict, *, source, vad, asr, gate, injector, decider, voice, console=lambda m: None,
                 clock=time.monotonic, proxy_pid: int | None = None, proxy_socket: str | None = None,
                 dry_run: bool = False,
                 show_ignored: bool = False, linger: float = 0.0, write_status: bool = True,
                 judge=None, session_state=None, cue_sound=play_cue, stop_voice=stop_hal_voice, lexicon=None,
                 screen_locked=None, voice_check=None):
        self.cfg = cfg
        self.source, self.vad, self.asr, self.gate = source, vad, asr, gate
        self.injector, self.decider, self.voice = injector, decider, voice
        self.judge = judge
        self.screen_locked = screen_locked   # callable -> True while the screen is locked (screen.locked)
        self.voice_check = voice_check       # callable(audio, speech_seconds) -> voiceprint.Verdict | None
        self.locked = False
        self.console = console
        self.clock = clock
        self.proxy_pid = proxy_pid
        self.proxy_socket = proxy_socket
        self.dry_run = dry_run
        self.show_ignored = show_ignored
        self.linger = linger
        self.write_status = write_status
        self.segmenter = Segmenter(cfg)
        self.counts = Counts()
        # Turns (turns.py): None with turn_taking "silence", where every utterance is a request.
        self.turns: TurnTaker | None = None
        if cfg.get("turn_taking", "semantic") == "semantic":
            rejoin = self._rejoin if cfg.get("turn_retranscribe") and getattr(asr, "rejoins", False) else None
            self.turns = TurnTaker(cfg, judge=judge if cfg.get("turn_judge", True) else None, rejoin=rejoin,
                                   clock=clock, log=logs.log, console=console)
        self.cue_sound = cue_sound
        if session_state is None and proxy_pid:
            from . import session

            session_state = lambda: session.read(proxy_pid)  # noqa: E731
        self.session_state = session_state
        self._chain_audio: list = []     # samples of the parts of a cut utterance (semantic turns only)
        self._chain_start: float | None = None
        self.paused = False
        self._stop = False
        self._vad_dirty = False
        self._reply_at: float | None = None
        self._answering = False
        self._next_house = 0.0
        self._next_status = 0.0
        self._finished_at: float | None = None
        self._last_frame = 0.0
        self.started = time.time()
        self.last_event = ""
        self.events: list[dict] = []   # decisions, for tests and the e2e report (no ignored text)
        # Listening while the voice speaks (duplex "aec"): source is then a DuplexSource.
        self.duplex = source if getattr(source, "duplex", False) else None
        self.barge = BargeIn(cfg) if self.duplex is not None and cfg.get("barge_in", True) else None
        self.stop_voice = stop_voice
        self._plays: list[dict] = []        # recent playbacks through the helper (its events)
        self._voice_paused_at: float | None = None
        self.interrupted: Interrupted | None = None
        self._paused_since: float | None = None
        self._fresh_cut: Interrupted | None = None   # the voice was cut off by the utterance being handled
        self._chain_words: list = []                 # words of the parts of a cut utterance
        # The names and terms for corrections (lexicon.py); None: no corrections.
        self.lexicon = lexicon if cfg.get("spoken_label", True) and cfg.get("lexicon", True) else None
        self._next_lexicon = 0.0
        self._lexicon_names()

    # ------------------------------------------------------------ control

    def request_stop(self, *_args) -> None:
        self._stop = True

    def _lexicon_names(self) -> None:
        """The lexicon's aliases for the name go to addressing.py through the settings."""
        anywhere, calling = self.lexicon.name_aliases() if self.lexicon is not None else ([], [])
        self.cfg["alias_names"], self.cfg["calling_names"] = anywhere, calling

    def _paused_flag(self):
        return settings.listen_dir() / "paused"

    def _pause(self) -> None:
        settings.listen_dir().mkdir(parents=True, exist_ok=True)
        self._paused_flag().touch()
        logs.log("paused by voice command")
        self.console("paused (resume with `hal-listen resume`)")
        self.voice(self.cfg["paused_reply"])

    def _housekeeping(self, now: float) -> None:
        self._next_house = now + 0.5
        flagged = self._paused_flag().exists()
        was_locked, self.locked = self.locked, bool(self.screen_locked is not None and self.screen_locked() is True)
        paused = flagged or self.locked
        if paused != self.paused:
            self.paused = paused
            why = ""
            if paused and self.locked and not flagged:
                why = " (screen locked)"
            elif not paused and was_locked:
                why = " (screen unlocked)"
            logs.log(("paused" if paused else "resumed") + why)
            self.console(("paused" if paused else "resumed") + why)
            if not paused and hasattr(self.judge, "wake"):
                self.judge.wake()   # the front desk may have retired while listening was paused
            if paused and self.turns is not None and self.turns.pending:
                self.turns.discard("listening paused")
                self.console("a request in progress was discarded: listening paused")
        if self.lexicon is not None and now >= self._next_lexicon:
            self._next_lexicon = now + 5.0
            try:
                if self.lexicon.refresh():
                    self._lexicon_names()
                    logs.log(f"lexicon: {self.lexicon.describe()}")
            except Exception as exc:  # a bad lexicon file must not stop the listener
                logs.log(f"lexicon: could not refresh ({exc!r})")
        if self.proxy_pid and not (pid_alive(self.proxy_pid) and socket_alive(self.proxy_socket)):
            logs.log("the Claude Code session it served has exited; stopping")
            self.console("the Claude Code session has exited; stopping")
            self._stop = True
        if self.write_status and now >= self._next_status:
            self._next_status = now + 2.0
            self._status()

    def _status(self, state: str | None = None) -> None:
        if state is None:
            state = "paused" if self.paused else ("holding" if self.injector.holding else "listening")
        info = {"pid": os.getpid(), "proxy_pid": self.proxy_pid, "state": state, "screen_locked": self.locked,
                "started": self.started,
                "updated": time.time(), "source": getattr(self.source, "name", "?"),
                "model": getattr(self.asr, "name", "?"), "dry_run": self.dry_run,
                "counts": self.counts.__dict__, "last_event": self.last_event,
                "turn_taking": "semantic" if self.turns is not None else "silence",
                "turn": self.turns.describe() if self.turns is not None else None,   # no words
                "duplex": self.duplex.describe() if self.duplex is not None else None,
                "lexicon": len(self.lexicon) if self.lexicon is not None else None}
        path = settings.listen_dir() / "listener.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    # ------------------------------------------------------------ the loop

    def run(self) -> int:
        try:
            while not self._stop:
                now = self.clock()
                if now >= self._next_house:
                    self._housekeeping(now)
                    if self._stop:
                        break
                if self.paused:
                    if getattr(self.source, "keeps_time", False):
                        # A file keeps playing while paused; its frames are dropped unheard.
                        if self.source.read(timeout=0.1) is None and self.source.finished:
                            break
                        self.segmenter.reset()
                        self._timers()
                        continue
                    if self.source.running:
                        if self._paused_since is None:
                            self._paused_since = now
                        if getattr(self.source, "playing", False) and now - self._paused_since < 30:
                            # The voice plays through the helper ("I will stop listening ..."):
                            # let it finish before closing the helper; what is heard is dropped.
                            self.source.read(timeout=0.1)
                            continue
                        self.source.stop()
                        self.segmenter.reset()
                        self.console("microphone closed while paused")
                    time.sleep(0.2)
                    continue
                self._paused_since = None
                if not self.source.running:
                    if not self._open_source():
                        continue
                item = self.source.read(timeout=0.1)
                if item is not None:
                    self._last_frame = self.clock()
                    t, frame = item
                    if not isinstance(t, (int, float)):
                        # A malformed item from the audio source (seen on 17 Nov 2026 after an
                        # input-device change, when it killed the listener): drop it and carry on.
                        logs.log(f"dropped a malformed audio frame (time was {type(t).__name__})")
                        continue
                    self._frame(t, frame)
                elif not getattr(self.source, "keeps_time", False) and self.clock() - self._last_frame > 3.0:
                    # The device went away (unplugged headset, say): reopen the current default.
                    logs.log("no audio for 3 s; reopening the input device")
                    self.console("no audio for 3 s; reopening the input device")
                    self.source.stop()
                    self.segmenter.reset()
                    continue
                self._timers()
                if item is None and getattr(self.source, "finished", False):
                    if self._finish_input():
                        break
        finally:
            self.source.stop()
            if self.write_status:
                self._status("stopped")
        return 0

    def _open_source(self) -> bool:
        try:
            self.source.start()
        except Exception as exc:  # no device, or it vanished: try again shortly
            logs.log(f"cannot open the audio input ({exc}); retrying")
            self.console(f"cannot open the audio input ({exc}); retrying")
            time.sleep(2.0)
            return False
        self._vad_dirty = True
        self._last_frame = self.clock()
        self.console(f"listening on {getattr(self.source, 'name', '?')}")
        return True

    def _finish_input(self) -> bool:
        """The file source is exhausted: flush, then linger for pending work."""
        now = self.clock()
        if self._finished_at is None:
            self._finished_at = now
            res = self.segmenter.flush(now)
            if isinstance(res, Utterance):
                self._utterance(res)
        pending = self.turns is not None and self.turns.pending
        busy = (self.injector.holding or self.voice.busy() or self._reply_at is not None or self._answering
                or pending)
        done = not busy or now - self._finished_at > self.linger
        if done and pending:
            self._turn_outcome(self.turns.flush("the input ended"))
        return done

    def _over_voice_ok(self) -> bool:
        """Is the voice's echo taken care of (cancelled, or headphones), so listen while it plays?"""
        return self.duplex is not None and self.duplex.active

    def _frame(self, t: float, frame) -> None:
        if self.gate.blocked(t) and not self._over_voice_ok():
            self.counts.gated_seconds += self.segmenter.dt
            dropped = self.segmenter.reset()
            if dropped is not None:
                self.counts.discarded += 1
                logs.discarded("the voice was speaking", dropped.seconds)
                self.console(f"discarded {dropped.seconds:.1f}s of speech: the voice was speaking")
                self._event("discarded", seconds=dropped.seconds)
            self._vad_dirty = True
            return
        if self._vad_dirty:
            self.vad.reset()
            self._vad_dirty = False
        prob = self.vad(frame)
        if self.barge is not None:
            playing = self._over_voice_ok() and bool(getattr(self.source, "playing", False))
            if self.barge.frame(t, prob, _frame_db(frame), playing, getattr(self.source, "ref_db", None)) == "pause":
                self._pause_voice(t)
        res = self.segmenter.push(frame, prob, t)
        if res is None:
            return
        if isinstance(res, Discarded):
            self.counts.noise += 1
            if self._voice_paused_at is not None:
                self._resume_voice("only a noise")
            return
        self._utterance(res)

    def _during_turn(self) -> bool:
        """Is Claude still working on a request this listener typed? (during_turn)

        True while the session's turn is busy and has not stopped since the
        listener last typed into it, for at most _DURING_TURN_MAX after that."""
        if not self.cfg.get("during_turn") or self.dry_run or self.session_state is None:
            return False
        typed = getattr(self.injector, "last_typed", None)
        if not typed or time.time() - typed > _DURING_TURN_MAX:
            return False
        try:
            st = self.session_state()
        except Exception:
            return False
        if not isinstance(st, dict) or st.get("turn") != "busy":
            return False
        try:
            return float(st.get("last_stop") or 0.0) < typed
        except (TypeError, ValueError):
            return False

    def _transcribe(self, audio) -> tuple[str, list | None]:
        """(text, words with confidences or None), the text sanitised (hal/spoken.py)."""
        with_words = getattr(self.asr, "transcribe_words", None)
        if with_words is not None:
            text, words = with_words(audio)
        else:
            text, words = self.asr.transcribe(audio), None
        return spoken_label.sanitize(text), words

    def _check_voice(self, u: Utterance):
        """Whose voice is this (voiceprint.py)? A Verdict, or None when not checking."""
        if self.voice_check is None or self.cfg.get("voice_check", "off") == "off" or u.audio is None:
            return None
        try:
            from .voiceprint import speech_only

            verdict = self.voice_check(speech_only(u.audio, u.start, u.speech_start, u.speech_end), u.speech_seconds)
        except Exception as exc:  # never lose a request to the check
            logs.log(f"voice check failed ({exc!r}); this utterance is not checked")
            return None
        if verdict is not None:
            self.counts.voice_checked += 1
        return verdict

    def _learn_voice(self, verdict, speech_seconds: float) -> None:
        """A request clearly in your voice refines the voiceprint (voice_learn)."""
        learn = getattr(self.voice_check, "learn", None)
        if learn is None:
            return
        try:
            if learn(verdict, speech_seconds):
                self.counts.voice_learned += 1
                logs.log(f"voice: learned from this request ({verdict.score:.2f})")
        except Exception as exc:  # never lose a request to learning
            logs.log(f"voice: could not learn from this request ({exc!r})")

    def _utterance(self, u: Utterance) -> None:
        self.counts.utterances += 1
        voice = self._check_voice(u)
        enforce = voice is not None and self.cfg.get("voice_check") == "enforce"
        if enforce and not voice.match and voice.reliable:
            # Another voice: ignored before it is even transcribed, and it ends no window.
            self.counts.other_voice += 1
            logs.voice(voice.score, u.speech_seconds, "another voice: ignored unheard")
            self.console(f"ignored {u.speech_seconds:.1f}s of speech in another voice (voice {voice.score:.2f})")
            self._event("other_voice", score=round(voice.score, 3), speech=u.speech_seconds, speech_end=u.speech_end)
            self._chain_audio, self._chain_start, self._chain_words, u.audio = [], None, [], None
            self.decider.forget()
            if self._voice_paused_at is not None:
                self._resume_voice("another voice")
            return
        # Too short to judge, and not like you: only the name counts, as for speech heard over the voice.
        unsure = enforce and not voice.match
        started = time.perf_counter()
        audio = u.audio
        text, asr_words = self._transcribe(audio)
        u.audio = None
        asr_seconds = time.perf_counter() - started
        self._fresh_cut = None
        if not u.continued:
            self._chain_audio, self._chain_start = [], u.speech_start
            self._chain_words = []
        pending = self.turns is not None and self.turns.pending
        over, overlapped = "pass", False
        if self.barge is not None and not u.cut and (self._voice_paused_at is not None
                                                    or self.barge.overlaps(u.speech_start)):
            overlapped = True
            over = self._said_over_voice(text, u)
        elif not pending and self.interrupted is not None and not u.cut and self._resume_cut(text):
            over = "drop"
        if over == "drop":
            self._chain_audio, self._chain_start, u.audio = [], None, None
            self._chain_words = []
            self.decider.forget()        # and the earlier parts of a long, cut utterance
            return
        # Heard over the voice without interrupting it: only the name counts (it may be residual echo).
        quiet_overlap = overlapped and over == "pass"
        named_only = quiet_overlap or unsure
        d = self.decider.decide(text, u, self.clock(), self.gate.reply_end, pending=pending and not named_only,
                                during_turn=not pending and not named_only and self._during_turn(),
                                barge_in=over == "barge", named_only=named_only)
        if (over == "pass" and d.action in ("submit", "name_only", "stop_listening")
                and self.barge is not None and self._voice_audible()):
            self._cut_voice("a request said over it")
        self.last_event = d.action
        if d.action == "partial":
            if self.turns is not None:
                self._chain_audio.append(audio)
            self._chain_words.extend(asr_words or [])
            self.console(f"long utterance: {u.duration:.1f}s so far")
            return
        parts = [*self._chain_audio, audio] if self.turns is not None else []
        speech_start = self._chain_start if u.continued and self._chain_start is not None else u.speech_start
        heard = [*self._chain_words, *asr_words] if asr_words is not None else None
        self._chain_audio, self._chain_start, audio, self._chain_words = [], None, None, []
        cut, self._fresh_cut = self._fresh_cut, None
        if d.action in ("ignore", "empty"):
            parts = heard = None   # the samples (and words) go with the decision
            self.counts.ignored += 1
            logs.ignored(d.speech_seconds, d.duration)
            if voice is not None:
                logs.voice(voice.score, d.speech_seconds, f"{voice.describe()}; not addressed")
            words = f': "{text}"' if self.show_ignored and text else ""
            self.console(f"ignored {d.speech_seconds:.1f}s of speech, not addressed to Claude{words}")
            self._event(d.action, speech=d.speech_seconds, asr=asr_seconds, speech_end=u.speech_end)
            return
        semantic = self.turns is not None
        what = f"{d.action}, {d.how}"
        if semantic and d.action == "submit":
            what = f"fragment {len(self.turns.turn.fragments) + 1 if self.turns.turn else 1} of a turn, {d.how}"
        if voice is not None:
            what += f", voice {voice.score:.2f} ({voice.describe()}{'; only the name counted' if unsure else ''})"
        logs.addressed(d.text, d.speech_seconds, what)
        if voice is not None and d.action == "submit" and not overlapped and not unsure:
            self._learn_voice(voice, u.speech_seconds)
        self.console(f'heard ({d.how}): "{d.text}" -> {"fragment" if semantic and d.action == "submit" else d.action}'
                     f"  [speech-to-text {asr_seconds * 1000:.0f} ms]")
        if d.action == "submit" and semantic:
            self.counts.fragments += 1
            out = self.turns.add(d.text, how=d.how, speech_start=speech_start, speech_end=u.speech_end,
                                 speech_seconds=d.speech_seconds, duration=d.duration, audio=parts, words=heard,
                                 interrupted=cut)
            self._event("fragment", text=d.text, how=d.how, speech=d.speech_seconds, asr=asr_seconds,
                        speech_end=u.speech_end, verdict=out.turn.verdict if out.turn else "", why=out.why)
            self._turn_outcome(out)
            return
        self._event(d.action, text=d.text, how=d.how, speech=d.speech_seconds, asr=asr_seconds,
                    speech_end=u.speech_end)
        if d.action == "submit":
            typed = self._typed(d.text, heard=heard or [], fragments=[(speech_start, u.speech_end)],
                                speech_start=speech_start, interrupted=cut)
            self._submit(typed.text, u.speech_end)
        elif d.action == "name_only":
            if semantic and not pending and opens_with_name(text):
                # "Claude," or "Claude...": more is coming, so wait for it without answering.
                self.turns.open(d.how, u.speech_end)
                self.console("the name, then a pause: waiting for the request")
            else:
                self._reply_at = self.clock() + self.cfg["name_only_grace"]
        elif d.action == "stop_listening":
            if semantic and self.turns.pending:
                self.turns.discard("stop listening")
            self._pause()

    def _typed(self, text: str, *, heard: list, fragments: list, speech_start: float,
               interrupted: Interrupted | None) -> annotate.Annotated:
        """The request as it is typed: labelled and marked (annotate.py)."""
        when = time.time() - max(0.0, self.clock() - speech_start)
        sentence = ""
        if interrupted is not None:
            now = self.interrupted   # the exact position may have been noted since
            sentence = (now if now is not None and now.play_id == interrupted.play_id else interrupted).reached
        try:
            typed = annotate.build(text, self.cfg, heard=heard, fragments=fragments, when=when,
                                   interrupted=sentence, lex=self.lexicon)
        except Exception as exc:  # never lose a request to its marks
            logs.log(f"could not mark the request ({exc!r}); typing it with the label alone")
            clean = " ".join(spoken_label.sanitize(text).split())
            typed = annotate.Annotated(f"{spoken_label.label(when)} {clean}"
                                       if self.cfg.get("spoken_label", True) else clean, clean)
        self.counts.corrected += len(typed.corrections)
        self.counts.flagged += len(typed.flags)
        self.counts.doubtful += len(typed.doubts)
        if typed.corrections or typed.flags or typed.doubts or typed.pauses or typed.interrupted:
            logs.log(f"marked: {typed.summary()}")
        return typed

    def _submit(self, text: str, speech_end: float) -> None:
        self.counts.submitted += 1
        if self.dry_run:
            self.console("(dry run: not typed)")
        else:
            self.injector.submit(text, speech_end)

    def _turn_outcome(self, out: Outcome | None) -> None:
        if out is None:
            return
        t = out.turn
        if out.action == "wait":
            verdict = t.verdict if t else "?"
            note = {"complete": "complete; sending after a moment's silence",
                    "judging": "asking the front desk whether it is finished",
                    "unclear": f"unclear; sending after {self.cfg['turn_unclear_wait']:g}s of silence",
                    "incomplete": f"unfinished; waiting up to {self.cfg['turn_max_wait']:g}s for more"}.get(verdict, verdict)
            self.console(f"turn: {note} ({out.why})")
        elif out.action == "send":
            span = t.speech_end - t.fragments[0].speech_start if t and t.fragments else 0.0
            typed = self._typed(out.text, heard=self.turns.heard_words(t),
                                fragments=[(f.speech_start, f.speech_end) for f in t.fragments],
                                speech_start=t.fragments[0].speech_start, interrupted=t.interrupted)
            logs.log(f"turn of {len(t.fragments)} fragment(s) over {span:.1f}s, {t.how}, sent ({out.why}): "
                     f"{typed.text!r}")
            self.console(f'turn sent ({out.why}): "{typed.text}"')
            self._event("submit", text=typed.text, plain=out.text, how=t.how, speech=t.speech_seconds,
                        fragments=len(t.fragments), speech_end=t.speech_end, span=span, why=out.why,
                        marks=typed.summary())
            self._submit(typed.text, t.speech_end)
        elif out.action == "cancel":
            self.counts.cancelled += 1
            self.console(f"turn cancelled ({out.why})")
            self._event("cancel", why=out.why)
            if self.cfg.get("turn_cancel_reply"):
                self.voice(self.cfg["turn_cancel_reply"])
        elif out.action == "drop":
            self.console(f"turn dropped ({out.why})")
            self._event("drop", why=out.why)
        elif out.action == "cue":
            self._cue()

    def _cue(self) -> None:
        kind = self.cfg.get("turn_cue", "none")
        if kind == "words" and self.cfg.get("turn_cue_words"):
            self.voice(self.cfg["turn_cue_words"])
        elif kind == "sound" and self.cue_sound is not None:
            self.cue_sound(self.cfg["turn_cue_sound"])
        else:
            return
        logs.log(f"turn: cue ({kind}) while waiting for more")
        self._event("cue", kind=kind)

    def _rejoin(self, parts: list):
        """Transcribe a turn's fragments afresh as one piece (0.3 s of silence between them):
        (text, words), each word with the piece of audio it fell in (.part), or the text alone."""
        import numpy as np

        gap = np.zeros(int(0.3 * 16000), np.float32)
        audio = np.concatenate([x for p in parts for x in (np.asarray(p, np.float32), gap)])
        try:
            if getattr(self.asr, "transcribe_words", None) is None:
                return spoken_label.sanitize(self.asr.transcribe(audio))
            text, words = self.asr.transcribe_words(audio)
            ends, t = [], 0.0
            for p in parts:
                t += len(p) / 16000 + 0.3
                ends.append(t)
            for w in words:
                mid = (w.start + w.end) / 2
                w.part = next((k for k, e in enumerate(ends) if mid < e), len(parts) - 1)
            return spoken_label.sanitize(text), words
        finally:
            del audio

    # ------------------------------------------------------------ speech over the voice (duplex "aec")

    def _voice_audible(self) -> bool:
        return bool(getattr(self.source, "playing", False)) or self._voice_paused_at is not None

    def _pause_voice(self, t: float) -> None:
        if self.duplex is not None and self.duplex.command("pause"):
            self._voice_paused_at = self.clock()
            self.counts.barge_ins += 1
            logs.log("barge-in: speech over the voice; paused it")
            self.console("speech over the voice: paused it")
            self._event("barge_in", heard_at=t)
        elif self.barge is not None:
            self.barge.settle()

    def _resume_voice(self, why: str) -> None:
        if self._voice_paused_at is None:
            return
        self._voice_paused_at = None
        if self.barge is not None:
            self.barge.settle()
        if self.duplex is not None:
            self.duplex.command("resume")
        self.counts.voice_resumed += 1
        logs.log(f"barge-in: resumed the voice ({why})")
        self.console(f"resumed the voice ({why})")
        self._event("voice_resumed", why=why)

    def _play_now(self) -> dict | None:
        return next((p for p in reversed(self._plays) if p.get("end") is None), None)

    @staticmethod
    def _position(play: dict, now: float) -> float:
        return play["pos"] + (0.0 if play["paused"] else max(0.0, now - play["pos_at"]))

    def _cut_voice(self, why: str) -> None:
        """Silence the voice for good (helper first, at once; then `hal stop` for what is queued) and
        note where it was, so that "go on" can resume it."""
        now = self.clock()
        play = self._play_now()
        if self.duplex is not None:
            self.duplex.command("stop")
        try:
            self.stop_voice()
        except Exception as exc:
            logs.log(f"could not stop the voice: {exc!r}")
        self._voice_paused_at = None
        if self.barge is not None:
            self.barge.settle()
        self.counts.voice_cut += 1
        where = ""
        if play is not None and play.get("spoken") is not None:
            self._note_interrupted(play, self._position(play, now), why)
            i = self._fresh_cut = self.interrupted
            where = f" at {i.position:.1f}s, in sentence {i.sentence + 1} of {i.sentences}"
        logs.log(f"barge-in: silenced the voice{where} ({why})")
        self.console(f"silenced the voice{where} ({why})")
        self._event("voice_cut", why=why)

    def _note_interrupted(self, play: dict, position: float, why: str) -> None:
        spoken: Spoken = play["spoken"]
        index, rest = spoken.rest_from(position)
        self.interrupted = Interrupted(at=self.clock(), wall=time.time(), position=position, sentence=index,
                                       sentences=len(spoken.sentences), rest=rest, why=why, play_id=play.get("id"),
                                       reached=spoken.sentence(index))
        path = settings.listen_dir() / "interrupted.json"     # the voice's own text, never the user's
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.interrupted.to_json()), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    def _spoken_around(self, u: Utterance) -> str:
        """What the voice was saying while u was spoken (for the echo check)."""
        texts = []
        for p in self._plays:
            end = p["end"] if p["end"] is not None else self.clock()
            if p.get("spoken") is None or end + 1.0 < u.start or p["start"] > u.end:
                continue
            a, b = u.start - p["start"], u.end - p["start"]
            texts.append(p["spoken"].around(a, b, before=4.0, after=1.5))
        return " ".join(texts)

    def _said_over_voice(self, text: str, u: Utterance) -> str:
        """An utterance said while the voice played (or paused it): "drop", "barge" or "pass"."""
        paused = self._voice_paused_at is not None
        kind = resolve(text, self._spoken_around(u), self.cfg)
        if kind == "echo":
            self.counts.echo += 1
            logs.discarded("the voice's own echo", u.duration)     # no words
            self.console(f"discarded {u.speech_seconds:.1f}s: the voice's own echo")
            self._event("echo", speech=u.speech_seconds)
            self._resume_voice("it was the voice's own echo")
            return "drop"
        if kind == "noise":
            self._resume_voice("no words")
            return "pass"           # counted as empty by the decider
        if kind == "resume":
            if paused or self._voice_audible():
                self._resume_voice("asked to go on")
                return "drop"
            return "pass"
        if kind == "stop":
            self._cut_voice("a stop phrase")
            self._event("stop_phrase", speech=u.speech_seconds)
            return "drop"
        if paused:
            self._cut_voice("speech over it")
            return "barge"
        return "pass"

    def _resume_cut(self, text: str) -> bool:
        """ "Go on" after the voice was cut off: say the rest, from the sentence it was cut in."""
        cut = self.interrupted
        if cut is None or not is_resume_phrase(text, self.cfg):
            return False
        self.interrupted = None
        if self.clock() - cut.at > float(self.cfg.get("resume_seconds", 300.0)) or not cut.rest:
            return False
        self.counts.voice_resumed += 1
        logs.log(f"resuming the voice from sentence {cut.sentence + 1} of {cut.sentences}")
        self.console(f"resuming the voice from sentence {cut.sentence + 1} of {cut.sentences}")
        self._event("voice_resumed", why="go on", sentence=cut.sentence)
        self.voice(cut.rest)
        return True

    def _duplex_events(self) -> None:
        now = self.clock()
        for ev in self.duplex.events():
            kind = ev.get("event")
            if kind == "play":
                self._plays = [p for p in self._plays if p["end"] is None or now - p["end"] < 60][-5:]
                self._plays.append({"id": ev.get("id"), "start": now, "end": None, "pos": 0.0, "pos_at": now,
                                    "paused": False, "spoken": Spoken.load(ev.get("path"))})
                self.interrupted = None          # it has spoken since: "go on" no longer resumes the old one
            elif kind in ("paused", "resumed", "played"):
                play = next((p for p in self._plays if p["id"] == ev.get("id")), None)
                if play is None:
                    continue
                play["pos"], play["pos_at"] = float(ev.get("position") or 0.0), now
                play["paused"] = kind == "paused"
                if kind == "played":
                    play["end"] = now
                    cut = self.interrupted
                    if cut is not None and cut.play_id == play["id"] and play.get("spoken"):
                        self._note_interrupted(play, play["pos"], cut.why)   # the exact position
                    if self._voice_paused_at is not None and self._play_now() is None:
                        self._voice_paused_at = None      # it ended (stopped elsewhere) while paused
                        if self.barge is not None:
                            self.barge.settle()
            elif kind == "devices":
                state = ("echo cancellation on" if ev.get("vp") else
                         "headphones, no echo to cancel" if ev.get("headphones") else "half-duplex")
                msg = (f"duplex: devices now input {ev.get('input')}, output {ev.get('output')} "
                       f"({ev.get('output_kind')}): {state}")
                logs.log(msg)
                self.console(msg)
            elif kind == "error":
                logs.log(f"duplex: the audio helper reports: {ev.get('why')}")
        if self._voice_paused_at is not None and not self.segmenter.in_speech and now - self._voice_paused_at > 3.0:
            self._resume_voice("nothing followed")

    def _timers(self) -> None:
        now = self.clock()
        if self.duplex is not None:
            self._duplex_events()
        if self.turns is not None:
            self._turn_outcome(self.turns.tick(self.segmenter.in_speech))
        if not self.dry_run:
            self.injector.tick()
        if self._reply_at is not None and now >= self._reply_at:
            self._reply_at = None
            if self.segmenter.in_speech:
                self.console("speech resumed after the name; taking it as the request")
            else:
                self.voice(self.cfg["name_only_reply"])
                self._answering = True
        if self._answering:
            # Armed until armed_seconds after the "Yes?" (name_only_reply) has finished.
            self.decider.arm(now + self.cfg["armed_seconds"])
            if not self.voice.busy():
                self._answering = False

    def _event(self, action: str, **fields) -> None:
        self.events.append(dict(action=action, at=self.clock(), **fields))


def _frame_db(frame) -> float:
    import numpy as np

    x = np.asarray(frame, dtype=np.float32)
    return level_db(float(np.sqrt(np.mean(x * x))) if len(x) else 0.0)
