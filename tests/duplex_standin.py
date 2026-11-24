"""Stand-in for duplex/hal-duplex serve: speaks its protocol, opens no audio device.

    duplex_standin.py serve --socket S --mode M --ducking D [--agc]

Behaviour from $HAL_DUPLEX_STANDIN (comma-separated):
    vp            ready with voice processing on (the default)
    novp          ready, voice processing unavailable, speakers
    headphones    ready, no voice processing, headphones
    error         an error event instead of ready, then waits
    exit          exits at once (code 3)
    die=N         exits after N frames
    devices=N     after N frames, a second ready (the devices changed: AirPods)
    playing       frames say the voice is playing
Frames: 512 samples of silence every 32 ms, then the render level and playing flag.
Commands on stdin: pause, resume, stop -> paused / resumed / played events.
Writes its arguments and the commands it got to $HAL_DUPLEX_STANDIN_LOG, if set.
"""

import json
import os
import struct
import sys
import threading
import time

opts = {}
for part in (os.environ.get("HAL_DUPLEX_STANDIN") or "vp").split(","):
    k, _, v = part.partition("=")
    opts[k.strip()] = v.strip()
log_path = os.environ.get("HAL_DUPLEX_STANDIN_LOG")
out = sys.stdout.buffer
lock = threading.Lock()


def note(line):
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def record(kind, payload):
    with lock:
        out.write(bytes([kind]) + struct.pack("<I", len(payload)) + payload)
        out.flush()


def event(name, **fields):
    record(0x45, json.dumps({"event": name, **fields}).encode())


def ready(output="MacBook Pro Speakers", kind="built-in speakers", reason="start"):
    vp = "novp" not in opts and "headphones" not in opts
    event("ready", input="MacBook Pro Microphone", output=output, output_kind=kind, headphones="headphones" in opts,
          vp=vp, vp_wanted="headphones" not in opts, vp_error="" if vp or "headphones" in opts else "not supported",
          reason=reason, rate=48000, channels=3)


note("args " + " ".join(sys.argv[1:]))
if "exit" in opts:
    sys.exit(3)
if "error" in opts:
    event("error", why="the input has no audio format", failures=1)
else:
    ready()

position = 0.0


def commands():
    global position
    for line in sys.stdin:
        cmd = line.strip()
        note("cmd " + cmd)
        if cmd == "pause":
            event("paused", id=1, position=position, seconds=5.0, path="")
        elif cmd == "resume":
            event("resumed", id=1, position=position)
        elif cmd == "stop":
            event("played", id=1, reason="stopped", position=position, seconds=5.0, path="")
    note("eof")
    os._exit(0)


threading.Thread(target=commands, daemon=True).start()
die = int(opts["die"]) if opts.get("die") else None
devices = int(opts["devices"]) if opts.get("devices") else None
n = 0
silence = b"\0" * (512 * 4)
while True:
    if "error" in opts:
        time.sleep(0.05)
        continue
    playing = 1.0 if "playing" in opts else 0.0
    record(0x41, silence + struct.pack("<ff", 0.1 if playing else 0.0, playing))
    n += 1
    position += 0.032
    if devices is not None and n == devices:
        event("ready", input="AirPods Pro", output="AirPods Pro", output_kind="Bluetooth headphones",
              headphones=True, vp=False, vp_wanted=False, vp_error="", reason="the default output changed",
              rate=24000, channels=1)
    if die is not None and n >= die:
        os._exit(4)
    time.sleep(0.032)
