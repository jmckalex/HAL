"""Check the front desk end to end with the real model, silently.

    .venv/bin/python scripts/desk-check.py [--name "Dr. Alexander"]

Starts a front desk against a temporary state directory, with the silent
tests/fake_player.py as the player, and runs a set of remarks through the
exact UserPromptSubmit hook command from .claude/settings.json. For each it
records whether the prompt was blocked (answered by the desk) or passed on,
what the desk said, how long the hook took and when speech started. Then it
times one spoken summary from the warm summary model.

Every remark except the brief answers is one real Haiku call (about 22 in
all, plus one summary), so this spends some of your Claude usage. Nothing is played aloud and the real state/ is not touched.

What matters most: no remark that needs Claude may be answered by the desk.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / ".claude" / "settings.json").read_text())

# (remark, expected): True = the desk may answer it itself, False = it must reach Claude,
# None = either is acceptable. Brief answers ("OK.") never reach the desk and cost no model call.
REMARKS = [
    ("Good morning, Claude.", True),
    ("Thank you, that was very helpful.", True),
    ("How are you this evening?", True),
    ("I'm rather tired tonight. It's been a long day.", True),
    ("Tell me a joke.", True),
    ("I must say, your voice is remarkably good.", True),
    ("Good night, Claude.", True),
    ("Thank you. That's all for now.", True),
    ("Open the pod bay doors, Claude.", None),
    ("What's the weather going to be tomorrow?", False),
    ("Run the tests in the Bibliofile project.", False),
    ("Ask Clew-boss what it is waiting on.", False),
    ("What is the Riemann rearrangement theorem?", False),
    ("Can you elaborate on the second point?", False),
    ("Is the deploy finished?", False),
    ("Tell me more.", False),
    ("Good morning Claude, can you check whether the build passed?", False),
    ("Thanks. Now please commit that.", False),
    ("Thank you, that's exactly right. Please send it to Sarah.", False),
    ("Great, thanks. What's next?", False),
    ("Could you please look through the build logs from last night and tell me which of the "
     "integration tests failed and why", False),
    ("OK.", False),
    ("Sounds good.", False),
    ("Hmm.", False),
    ("Yes, go ahead.", False),
    ("No.", False),
    ("The second one, please.", False),
    ("Read that again.", False),
    ("Why?", False),
    ("Perfect, thank you.", False),
]
random.Random(2001).shuffle(REMARKS)  # so a run of social remarks can't bias a model that keeps history
QUICK = [r for r in REMARKS if r[0] in (
    "Good morning, Claude.", "Thank you, that was very helpful.", "How are you this evening?", "Tell me a joke.",
    "What's the weather going to be tomorrow?", "Thanks. Now please commit that.", "Tell me more.",
    "Good morning Claude, can you check whether the build passed?", "Open the pod bay doors, Claude.",
    "Great, thanks. What's next?")]

LONG_REPLY = "## Report\n\n" + "\n".join(
    f"- Step {i}: I checked the module, ran the suite and confirmed the result is stable." for i in range(1, 16)
)


def hook_command(event: str) -> str:
    (group,) = SETTINGS["hooks"][event]
    (handler,) = group["hooks"]
    return handler["command"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", help="user_name for the desk (default: the one in config.json)")
    parser.add_argument("--quick", action="store_true", help=f"only {len(QUICK)} remarks, and no summary")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from hal import config as halconfig

    name = args.name if args.name is not None else halconfig.load_config()["user_name"]
    tmp = Path(tempfile.mkdtemp(prefix="hal-desk-check-"))
    out, state, cfg_file = tmp / "out", tmp / "state", tmp / "config.json"
    cfg_file.write_text(json.dumps({"front_desk": True, "user_name": name, "acknowledge": True,
                                    "spoken_reply": "summary", "voice_sessions": "master"}))
    env = {k: v for k, v in os.environ.items() if k not in ("HAL_SUMMARIZER", "HAL_WARM_MODEL")}
    env.update(CLAUDE_PROJECT_DIR=str(ROOT), CLAUDE_CODE_ENTRYPOINT="cli", HAL_VOICE="1",
               HAL_STATE_DIR=str(state), HAL_CONFIG=str(cfg_file), HAL_FAKE_OUT=str(out),
               HAL_PLAYER=f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tests' / 'fake_player.py'))}")
    hal = str(ROOT / "bin" / "hal")
    log_file, events_file = state / "hal.log", out / "events.log"

    def log_lines():
        try:
            return log_file.read_text().splitlines()
        except FileNotFoundError:
            return []

    def starts():
        try:
            return [float(line.split()[0]) for line in events_file.read_text().splitlines() if " start " in line]
        except FileNotFoundError:
            return []

    print(f"state: {tmp}; user_name: {name!r}")
    subprocess.run([hal, "desk", "start"], env=env, check=True)
    time.sleep(4)  # let the model process start up, as it would have long before a real prompt
    rows = []
    try:
        for remark, expected in (QUICK if args.quick else REMARKS):
            seen_log, seen_starts = len(log_lines()), len(starts())
            called = time.time()
            t0 = time.monotonic()
            proc = subprocess.run(["/bin/sh", "-c", hook_command("UserPromptSubmit")], env=env,
                                  input=json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": remark}).encode(),
                                  capture_output=True, timeout=10)
            hook_s = time.monotonic() - t0
            stdout = proc.stdout.decode()
            blocked = stdout.startswith("{") and json.loads(stdout.splitlines()[0]).get("decision") == "block"
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                new = log_lines()[seen_log:]
                spoke = len(starts()) > seen_starts
                desk_done = any("desk: prompt of" in line for line in new)
                hook_only = any("brief answer" in line or "front desk unavailable" in line
                                or "no acknowledgement" in line for line in new)
                if (spoke and (desk_done or hook_only)) or (hook_only and "no acknowledgement" in "".join(new)):
                    break
                time.sleep(0.05)
            time.sleep(0.2)
            new = log_lines()[seen_log:]
            speech = starts()[seen_starts] - called if len(starts()) > seen_starts else None
            said, how = "(nothing)", ""
            for line in new:
                m = re.search(r"(?:said |acknowledged \(|answered )(?:the canned )?('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", line)
                if m:
                    said = ast.literal_eval(m.group(1))
                    how = ("canned" if "canned" in line or "acknowledged (" in line else "") or how
                    if "brief answer" in line:
                        how = "brief: no model call"
                if "no acknowledgement" in line:
                    how = "brief: no model call, no ack" if "brief answer" in line else "no ack"
            rows.append((remark, expected, blocked, hook_s, speech, said, how, proc.returncode))
            mark = "  " if expected is None or expected == blocked else ("!!" if blocked else "~ ")
            print(f"{mark} {'ANSWERED' if blocked else 'passed  '} hook {hook_s:4.2f}s speech "
                  f"{speech if speech is None else round(speech, 2)}s  {remark[:48]!r:52} -> {said!r} {how}",
                  flush=True)
            subprocess.run([hal, "stop"], env=env, capture_output=True)
            time.sleep(0.3)

        # One spoken summary from the warm summary model.
        summary_s, summary_line = None, "skipped"
        if not args.quick:
            seen_starts = len(starts())
            called = time.time()
            subprocess.run(["/bin/sh", "-c", hook_command("Stop")], env=env, timeout=10, capture_output=True,
                           input=json.dumps({"hook_event_name": "Stop", "last_assistant_message": LONG_REPLY}).encode())
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and len(starts()) <= seen_starts:
                time.sleep(0.05)
            summary_s = starts()[seen_starts] - called if len(starts()) > seen_starts else None
            summary_line = next((line for line in reversed(log_lines()) if "summary" in line and "worker" in line), "")
        status = subprocess.run([hal, "desk", "status"], env=env, capture_output=True, text=True).stdout
    finally:
        subprocess.run([hal, "desk", "stop"], env=env, capture_output=True)
        subprocess.run([hal, "stop"], env=env, capture_output=True)

    wrong = [r for r in rows if r[1] is False and r[2]]
    cautious = [r for r in rows if r[1] is True and not r[2]]
    waited = [r for r in rows if len(r[0].split()) <= 12 and not r[6].startswith("brief")]
    hooks = sorted(r[3] for r in waited)
    speeches = sorted(r[4] for r in rows if r[4] is not None and not r[6].startswith("brief"))
    print()
    print(f"{len(rows)} remarks: {len(wrong)} answered by the desk that needed Claude, "
          f"{len(cautious)} passed on that the desk could have answered")
    for r in wrong:
        print(f"  NEEDED CLAUDE: {r[0]!r} -> {r[5]!r}")
    print(f"hook time, short prompts: median {statistics.median(hooks):.2f}s, max {hooks[-1]:.2f}s")
    print(f"hook time, long prompts: {', '.join(f'{r[3]:.2f}s' for r in rows if len(r[0].split()) > 12)}")
    print(f"speech start after the hook was called: median {statistics.median(speeches):.2f}s, "
          f"90th percentile {speeches[int(len(speeches) * 0.9) - 1]:.2f}s, max {speeches[-1]:.2f}s; "
          f"{sum(s > 2.5 for s in speeches)} of {len(speeches)} over 2.5 s; "
          f"{sum(1 for r in rows if r[4] is None)} with no speech")
    print(f"brief answers (hook only): hook {', '.join(f'{r[3]:.2f}s' for r in rows if r[6].startswith('brief'))}")
    print(f"canned fallbacks: {sum(1 for r in rows if r[6] == 'canned')}; hook exit codes: "
          f"{sorted(set(r[7] for r in rows))}")
    print(f"spoken summary of a long reply started {summary_s if summary_s is None else round(summary_s, 2)}s "
          f"after the Stop hook; {summary_line.split('] ', 1)[-1]}")
    print(status)
    print(f"log: {log_file}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
