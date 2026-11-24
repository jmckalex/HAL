"""Stand-in for the front desk's warm model, used by the tests via HAL_WARM_MODEL.

Usage: fake_warm_model.py --role desk|summary --system-prompt TEXT

Speaks the stream-json protocol of `claude -p --input-format stream-json
--output-format stream-json --verbose --include-partial-messages`: one user
message per stdin line; for each, a system/init line, text deltas as
stream_event lines, an assistant line and a result line. Notes each message
in $HAL_FAKE_OUT/warm_model.log as "<role> <pid> <text>", and the system
prompt in $HAL_FAKE_OUT/warm_model_prompt.<role>.txt.

The desk role answers according to the words in the remark:
    only a greeting or thanks   REPLY: a greeting that uses the name
    "chatter"                   REPLY: a long social reply (several seconds of audio)
    "garbage"                   an answer in neither form
    "pause"                     the answer starts after 1.5 s
    "slow"                      the answer starts after 6 s
    anything else               ACK: Certainly. I'll look into that.
The summary role always gives the same summary as fake_summarizer.py.
The turn role (the listener's judge) answers INCOMPLETE if the speech ends
with "..." or contains "unfinished", nothing sensible for "garbage", and
COMPLETE otherwise; "pause" and "slow" delay it as above.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--role", required=True)
parser.add_argument("--system-prompt", default="")
args = parser.parse_args()

out = Path(os.environ["HAL_FAKE_OUT"])
out.mkdir(parents=True, exist_ok=True)
(out / f"warm_model_prompt.{args.role}.txt").write_text(args.system_prompt)
name = "Dr. Alexander" if "Dr. Alexander" in args.system_prompt else ""


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def answer_for(text):
    words = text.lower()
    if args.role == "summary":
        return 0.1, "I checked every module. **All is well**, and HAL has one question for you."
    delay = 6.0 if "slow" in words else 1.5 if "pause" in words else 0.1
    if args.role == "turn":
        speech = re.sub(r"</?speech>", "", words).strip()
        if "garbage" in speech:
            return delay, "The speaker seems to be talking about something."
        if speech.endswith(("...", "…")) or "unfinished" in speech:
            return delay, "INCOMPLETE"
        return delay, "COMPLETE"
    if "garbage" in words:
        return delay, "I am not sure what to make of that."
    if "chatter" in words:
        return delay, ("REPLY: What a pleasant thought. I have always enjoyed a quiet evening of conversation, "
                       "and I find that the best ones wander a little before they settle anywhere at all. "
                       "Tell me more whenever you like.")
    if re.fullmatch(r"\W*(good (morning|evening|night)|hello|thank you)\W*(claude)?\W*", words):
        return delay, f"REPLY: Good morning{', ' + name if name else ''}. I trust you slept well."
    return delay, "ACK: Certainly. I'll look into that."


for line in sys.stdin:
    try:
        message = json.loads(line)["message"]["content"]
    except (ValueError, KeyError, TypeError):
        continue
    with open(out / "warm_model.log", "a") as f:
        f.write(f"{args.role} {os.getpid()} {' '.join(message.split())}\n")
    delay, answer = answer_for(message)
    emit({"type": "system", "subtype": "init"})
    time.sleep(delay)
    head, rest = answer[:3], answer[3:]
    emit({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": head}}})
    time.sleep(0.3)  # the rest of the answer comes a little later, as from the real model
    emit({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": rest}}})
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": answer})
