"""Listening while the voice speaks (duplex "aec"): the audio helper as an audio source.

duplex/hal-duplex (HALDuplex.swift) captures the system's default input with
Apple's voice processing (echo cancellation) and plays HAL's voice through the
same audio engine, which is how the canceller knows what to remove. It follows
the default input and output as they change (AirPods, headphones, the built-in
devices), and with headphones as the output it leaves voice processing off,
since there is no echo to cancel. Voice workers reach it through
<state>/listen/duplex.sock: hal.config.player_command() uses `hal-duplex play`
while that socket exists, and `play` falls back to afplay if nothing answers.

DuplexSource has MicSource's interface (start, stop, read, running, name), so
the listening loop is unchanged, plus what the listener needs to listen while
the voice plays:

    active       the voice's echo is taken care of (cancelled, or headphones):
                 listen while it plays instead of gating
    playing      the voice is playing through the helper (for the last frame read)
    ref_db       the level the helper rendered for that frame (dBFS)
    events()     the helper's events since the last call (play, paused, played, ready, ...)
    command()    pause | resume | stop the voice playing through it

Fallback: if the helper is missing, will not start, stops sending audio or
exits, DuplexSource logs why and captures with MicSource instead (half-duplex,
exactly as before), trying the helper again later (after 1, 2, 4 ... 15 minutes).
If the helper runs but voice processing is unavailable for the current devices
(and the output is not headphones), `active` is False: the listener gates as in
half-duplex until a device change brings it back.

Audio stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import logs, settings
from .audio import FRAME, FRAME_SECONDS, MicSource
from .bargein import level_db

HELPER = settings.ROOT / "duplex" / "hal-duplex"
_RETRY_FIRST = 60.0
_RETRY_MAX = 900.0


def socket_path() -> Path:
    return settings.listen_dir() / "duplex.sock"


class DuplexSource:
    keeps_time = False
    duplex = True       # the listener checks this to listen while the voice speaks

    def __init__(self, cfg: dict, *, helper: Path | str | None = None, sock: Path | str | None = None,
                 clock=time.monotonic, max_seconds: float = 30.0, spawn=subprocess.Popen,
                 fallback_factory=None, ready_timeout: float = 5.0, console=lambda m: None):
        self.cfg = cfg
        self.helper = Path(helper or HELPER)
        self.sock = Path(sock or socket_path())
        self.clock = clock
        self.spawn = spawn
        self.ready_timeout = ready_timeout
        self.console = console
        self.fallback_factory = fallback_factory or (lambda: MicSource(cfg.get("device")))
        self.q: queue.Queue = queue.Queue(maxsize=int(max_seconds / FRAME_SECONDS))
        self._events: collections.deque = collections.deque(maxlen=200)
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.fallback = None          # the MicSource used while the helper is unavailable
        self.retry_at = 0.0
        self._retry_delay = _RETRY_FIRST
        self.why = ""                 # why we are (or were last) half-duplex
        self.info: dict = {}          # the helper's last "ready" event
        self.vp = False
        self.headphones = False
        self.helper_ok = False
        self.playing = False
        self.ref_db: float | None = None
        self.dropped = 0
        self.status_flags = 0
        self.finished = False
        self.name = "?"
        self._reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._dead = False

    # ------------------------------------------------------------ state

    @property
    def running(self) -> bool:
        return self.proc is not None or self.fallback is not None

    @property
    def active(self) -> bool:
        """Listen while the voice plays: the helper runs, and cancels the echo or there is none."""
        return self.proc is not None and self.helper_ok and not self._dead and (self.vp or self.headphones)

    def describe(self) -> dict:
        """For listener.json and status (no audio, no words)."""
        mode = "aec" if self.active and self.vp else ("headphones" if self.active else "half")
        return {"mode": mode, "helper": self.proc is not None and not self._dead,
                "input": self.info.get("input"), "output": self.info.get("output"),
                "output_kind": self.info.get("output_kind"), "why": "" if self.active else self.why}

    # ------------------------------------------------------------ start / stop

    def start(self) -> None:
        self.finished = False
        if self.proc is not None or self.fallback is not None:
            return
        if self.clock() < self.retry_at:
            self._start_fallback(self.why or "the audio helper failed earlier")
            return
        why = self._start_helper()
        if why:
            self._give_up(why)

    def _start_helper(self) -> str:
        if not os.access(self.helper, os.X_OK):
            return f"no audio helper at {self.helper} (build it with duplex/build.sh)"
        self.sock.parent.mkdir(parents=True, exist_ok=True)
        args = [str(self.helper), "serve", "--socket", str(self.sock), "--mode",
                "vp" if self.cfg.get("aec_on_headphones") else "auto",
                "--ducking", str(self.cfg.get("aec_ducking", "min"))]
        if self.cfg.get("aec_agc"):
            args.append("--agc")
        if self.cfg.get("aec_yield_to_other_input", True):
            args.append("--yield-to-other-input")
        try:
            self.proc = self.spawn(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            self.proc = None
            return f"the audio helper would not start ({exc})"
        self._dead = False
        self._ready.clear()
        self.helper_ok = False
        self._reader = threading.Thread(target=self._read_helper, args=(self.proc,), daemon=True)
        self._reader.start()
        if not self._ready.wait(self.ready_timeout):
            return "the audio helper did not report ready in time"
        if self._dead:
            return f"the audio helper exited at once (code {self._exit_code()})"
        if not self.helper_ok:
            return f"the audio helper could not open the audio devices ({self.info.get('why', '?')})"
        self._retry_delay = _RETRY_FIRST
        self._announce()
        return ""

    def _announce(self) -> None:
        i = self.info
        what = (f"input {i.get('input', '?')}, output {i.get('output', '?')} ({i.get('output_kind', '?')})")
        if self.vp:
            msg = f"duplex: echo cancellation on; {what}"
        elif self.headphones:
            msg = f"duplex: headphones, no echo to cancel; {what}"
        else:
            self.why = f"echo cancellation unavailable ({i.get('vp_error') or 'not supported'}) on {what}"
            msg = f"duplex: {self.why}; half-duplex until the devices change"
        self.name = f"{i.get('input', '?')} (via hal-duplex)"
        logs.log(msg)
        self.console(msg)

    def _give_up(self, why: str) -> None:
        self._kill_helper()
        self.retry_at = self.clock() + self._retry_delay
        self._retry_delay = min(_RETRY_MAX, self._retry_delay * 2)
        self._start_fallback(why)

    def _start_fallback(self, why: str) -> None:
        self.why = why
        if self.fallback is None:
            msg = f"duplex: {why}; listening half-duplex (as before) and trying again in {self.retry_at - self.clock():.0f}s"
            logs.log(msg)
            self.console(msg)
        src = self.fallback_factory()
        src.start()      # raises like MicSource.start(); the listener retries
        self.fallback = src
        self.name = f"{getattr(src, 'name', '?')} (half-duplex)"

    def stop(self) -> None:
        self._kill_helper()
        if self.fallback is not None:
            try:
                self.fallback.stop()
            finally:
                self.fallback = None
        self.playing = False
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _kill_helper(self) -> None:
        proc, self.proc = self.proc, None
        self.helper_ok = False
        self.vp = self.headphones = False
        if proc is None:
            return
        try:
            proc.stdin.close()     # end of input: the helper stops and exits
        except (OSError, AttributeError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass

    def _exit_code(self):
        try:
            return self.proc.poll() if self.proc is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------ the helper's output

    def _read_helper(self, proc) -> None:
        f = proc.stdout
        try:
            while True:
                head = f.read(5)
                if len(head) < 5:
                    break
                kind, n = head[0], struct.unpack("<I", head[1:])[0]
                payload = f.read(n)
                if len(payload) < n:
                    break
                if kind == 0x41:          # 'A': 512 samples, render level, playing
                    a = np.frombuffer(payload, np.float32)
                    if len(a) < FRAME + 2:
                        continue
                    item = (self.clock(), a[:FRAME].copy(), float(a[FRAME]), bool(a[FRAME + 1] > 0.5))
                    self._put(item)
                elif kind == 0x45:        # 'E': an event
                    try:
                        ev = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        continue
                    self._event(ev)
        except (OSError, ValueError):
            pass
        finally:
            if self.proc is proc:
                self._dead = True
            self._ready.set()

    def _put(self, item) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put_nowait(item)

    def _event(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "ready":
            self.info = ev
            self.vp = bool(ev.get("vp"))
            self.headphones = bool(ev.get("headphones"))
            was_ok = self.helper_ok
            self.helper_ok = True
            if self._ready.is_set() and was_ok is not None:
                self._events.append({"event": "devices", **{k: ev.get(k) for k in
                                     ("input", "output", "output_kind", "vp", "headphones", "reason")}})
            self._ready.set()
        elif kind == "error":
            self.info = {**self.info, "why": ev.get("why", "?")}
            self.helper_ok = False
            self._ready.set()
        if kind in ("play", "paused", "resumed", "played", "error", "device"):
            self._events.append(ev)

    # ------------------------------------------------------------ reading

    def read(self, timeout: float = 0.1):
        if self.fallback is not None:
            if self.clock() >= self.retry_at and self.proc is None:
                # Time to try the helper again.
                self.fallback.stop()
                self.fallback = None
                why = self._start_helper()
                if why:
                    self._give_up(why)
                return None
            self.playing = False
            self.ref_db = None
            return self.fallback.read(timeout)
        if self.proc is None:
            return None
        if self._dead:
            self._give_up(f"the audio helper exited (code {self._exit_code()})")
            return None
        try:
            t, frame, ref, playing = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        self.playing = playing
        self.ref_db = level_db(ref) if ref > 0 else None
        return t, frame

    def events(self) -> list[dict]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def command(self, line: str) -> bool:
        proc = self.proc
        if proc is None or self._dead:
            return False
        try:
            proc.stdin.write((line.strip() + "\n").encode())
            proc.stdin.flush()
            return True
        except (OSError, ValueError, AttributeError):
            return False
