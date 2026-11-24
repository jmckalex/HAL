"""Plays Claude Code's part for the end-to-end test: fires the hooks for one submitted line.

Run by the stand-in program (STANDIN_ON_LINE) with the submitted line as its
argument, in the stand-in's environment (so HAL_LISTEN_PROXY, HAL_VOICE and the
temporary HAL_STATE_DIR are set). Like Claude Code, it fires UserPromptSubmit
for both the HAL voice hook (which acknowledges aloud) and the listener's state
hook, waits as if Claude were thinking, then fires Stop with a short reply,
which the HAL voice speaks. "Played" by tests/fake_player.py, so silent.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THINK = float(os.environ.get("FAKE_THINK_SECONDS", "0.8"))
REPLY = os.environ.get("FAKE_REPLY", "Certainly, Doctor Alexander.")


def fire(command, payload):
    subprocess.run(command, input=json.dumps(payload).encode(), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=30)


def main():
    line = sys.argv[1] if len(sys.argv) > 1 else ""
    common = {"session_id": "e2e", "cwd": str(ROOT), "transcript_path": "/dev/null"}
    for cmd in ([str(ROOT / "bin" / "hal-listen"), "hook"], [str(ROOT / "bin" / "hal"), "hook"]):
        fire(cmd, dict(common, hook_event_name="UserPromptSubmit", prompt=line, permission_mode="default"))
    time.sleep(THINK)
    for cmd in ([str(ROOT / "bin" / "hal"), "hook"], [str(ROOT / "bin" / "hal-listen"), "hook"]):
        fire(cmd, dict(common, hook_event_name="Stop", stop_hook_active=False, last_assistant_message=REPLY))


if __name__ == "__main__":
    main()
