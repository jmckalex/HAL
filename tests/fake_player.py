"""Silent stand-in for afplay, used by the tests via HAL_PLAYER.

Usage: fake_player.py WAV
Copies WAV into $HAL_FAKE_OUT, then "plays" it by sleeping for its duration.
Appends `start <copy> <seconds>` and then `end` or `killed` to $HAL_FAKE_OUT/events.log.
"""

import os
import shutil
import signal
import sys
import time
import wave
from pathlib import Path

out = Path(os.environ["HAL_FAKE_OUT"])
out.mkdir(parents=True, exist_ok=True)
events = out / "events.log"


def note(line: str) -> None:
    with open(events, "a") as f:
        f.write(f"{time.time():.3f} {os.getpid()} {line}\n")


def on_term(signum, frame):
    note("killed")
    sys.exit(0)


signal.signal(signal.SIGTERM, on_term)
src = Path(sys.argv[-1])
copy = out / f"played-{time.time_ns()}.wav"
shutil.copy(src, copy)
with wave.open(str(src)) as w:
    seconds = w.getnframes() / w.getframerate()
note(f"start {copy} {seconds:.2f}")
time.sleep(seconds)
note("end")
