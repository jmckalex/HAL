"""Stand-in for the summarising model, used by the tests via HAL_SUMMARIZER.

Reads the reply on stdin, notes the call in $HAL_FAKE_OUT/summarizer.log, and
prints a fixed summary. Exits 1 if $HAL_FAKE_SUMMARY_FAIL is set.
"""

import os
import sys
from pathlib import Path

text = sys.stdin.read()
out = Path(os.environ["HAL_FAKE_OUT"])
out.mkdir(parents=True, exist_ok=True)
with open(out / "summarizer.log", "a") as f:
    f.write(f"{len(text)} HAL_VOICE={os.environ.get('HAL_VOICE', '')}\n")
if os.environ.get("HAL_FAKE_SUMMARY_FAIL"):
    sys.exit(1)
print("I checked every module. **All is well**, and HAL has one question for you.")
