"""Half-duplex gate: don't listen while the HAL voice is speaking. Standard library only.

hal/speaker.py records the worker that is speaking in state/speaker.pid (a
reply, or anything spoken through `hal say`) and state/ack.pid (an
acknowledgement). A worker's PID file appears when it is spawned, before
synthesis, and goes when it exits, after playback. The gate polls those
files; audio captured while either worker is alive, or within echo_tail
seconds after, is discarded. Speaking is detected before any sound plays
(synthesis takes longer than one poll), so only the tail needs a margin.

The gate also remembers when the voice last finished speaking a reply that
the listener itself didn't ask for, which starts the follow-up window.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def _pid_from(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        return int(data.get("pid", 0)) if isinstance(data, dict) else 0
    except (OSError, ValueError, TypeError):
        return 0


def _alive_group_leader(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return os.getpgid(pid) == pid
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _is_hal_worker(pid: int) -> bool:
    """Same test as hal.speaker._is_worker: a guard against a reused PID."""
    try:
        out = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return True  # can't tell: assume it is speaking (the safe side)
    return "_worker" in out and "hal.cli" in out


class SpeakingGate:
    def __init__(self, pid_files: list[Path], tail: float, *, clock=time.monotonic, poll_interval: float = 0.05,
                 own_speaking=lambda: False, is_worker=_is_hal_worker, alive=_alive_group_leader):
        self.files = [Path(p) for p in pid_files]
        self.reply_file = self.files[0].name if self.files else None
        self.tail = tail
        self.clock = clock
        self.poll_interval = poll_interval
        self.own_speaking = own_speaking
        self._is_worker = is_worker
        self._alive = alive
        self._verified: dict[int, bool] = {}
        self._last_poll = -1e9
        self.speaking = False
        self._interval_start = 0.0
        self._interval_files: set[str] = set()
        self._interval_own = False
        self.intervals: list[tuple[float, float]] = []   # recent (start, end) of speech
        self.last_end = -1e9      # when the voice last stopped speaking
        self.reply_end = -1e9     # ... a reply the listener did not ask for

    def _worker_alive(self, pid: int) -> bool:
        if not self._alive(pid):
            self._verified.pop(pid, None)
            return False
        if pid not in self._verified:
            self._verified[pid] = self._is_worker(pid)
        return self._verified[pid]

    def poll(self, force: bool = False) -> bool:
        now = self.clock()
        if not force and now - self._last_poll < self.poll_interval:
            return self.speaking
        self._last_poll = now
        active = {f.name for f in self.files if self._worker_alive(_pid_from(f))}
        if active and not self.speaking:
            self.speaking = True
            self._interval_start = now - self.poll_interval  # it may have started just after the last poll
            self._interval_files = set()
            self._interval_own = False
        if active:
            self._interval_files |= active
            self._interval_own = self._interval_own or bool(self.own_speaking())
        elif self.speaking:
            self.speaking = False
            self.last_end = now
            self.intervals = [iv for iv in self.intervals if iv[1] > now - 120] + [(self._interval_start, now)]
            if self.reply_file in self._interval_files and not self._interval_own:
                self.reply_end = now
        return self.speaking

    def blocked(self, t: float) -> bool:
        """Should audio captured at time t (same clock) be discarded?"""
        if self.poll() and t >= self._interval_start:
            return True
        return any(start <= t <= end + self.tail for start, end in self.intervals)
