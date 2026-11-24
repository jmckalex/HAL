"""Measure the addressing rule on synthesized speech: false accepts and false rejects.

    .venv-listen/bin/python tests/listen_accuracy.py [--voices N] [--json FILE]

Each phrase is spoken by each macOS `say` voice, padded with silence, and run
through the same VAD segmentation, Parakeet transcription and addressing rule
as live audio. Prints rates per voice group and every error with its transcript.
Not part of the unit tests (it takes a few minutes).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hal.listener import addressing, settings  # noqa: E402
from hal.listener.segmenter import Segmenter, Utterance  # noqa: E402

# Concatenative/neural voices, closest to a person; the rest are formant synthesizers.
NATURAL = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa"]
ROBOTIC = ["Fred", "Ralph", "Kathy", "Albert", "Junior", "Eddy (English (US))", "Flo (English (UK))",
           "Reed (English (US))", "Sandy (English (UK))", "Shelley (English (US))", "Rocko (English (UK))",
           "Grandpa (English (US))"]

ADDRESSED = [
    "Good morning, Claude.",
    "Claude, what's the weather tomorrow?",
    "By the way, Claude, open my thesis.",
    "Claude, read me the last paragraph.",
    "Thanks, Claude.",
    "Claude, can you check my calendar for Friday?",
    "What do you think about this, Claude?",
    "Hey Claude, summarize that email.",
    "Claude, stop listening.",
    "Okay Claude, let's start.",
    "Claude, please run the tests again.",
    "Good evening, Claude. How was your day?",
    "Claude, open the file called notes.",
    "Could you look that up for me, Claude?",
    "Claude.",
    "Hey Claude.",
]
NOT_ADDRESSED = [
    "The files are stored in the cloud.",
    "It looks like rain, there's a big cloud over there.",
    "The cat clawed at the door all night.",
    "That clause in the contract worries me.",
    "Please close the door behind you.",
    "I need to buy some new clothes.",
    "It is cloudy and cold today.",
    "Claudia called while you were out.",
    "We should visit grandmother on Sunday.",
    "Can you pass me the salt?",
    "What's the weather like tomorrow?",
    "Open the window, it's hot in here.",
    "I think the meeting is at three o'clock.",
    "Did you back up the photos to the cloud?",
    "He is such a clod sometimes.",
    "The clock on the wall is slow.",
    "Good morning, everyone.",
    "Thanks, that was delicious.",
]


def synth(text, voice):
    """`say` occasionally hangs (seen once in 90 minutes): time out and retry."""
    for attempt in range(3):
        with tempfile.TemporaryDirectory() as d:
            try:
                subprocess.run(["say", "-v", voice, "-o", f"{d}/a.aiff", text], check=True, timeout=30)
            except subprocess.TimeoutExpired:
                print(f"  (say timed out for {voice!r}; retrying)", flush=True)
                continue
            raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", f"{d}/a.aiff", "-f", "s16le",
                                  "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    raise RuntimeError(f"say kept hanging for {voice!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=int, default=0, help="use only the first N voices of each group")
    ap.add_argument("--json")
    args = ap.parse_args()
    cfg = settings.load(Path("/nonexistent"))  # defaults, whatever config.json says
    from hal.listener.asr import ParakeetASR
    from hal.listener.vad import SileroVAD

    vad = SileroVAD(str(settings.project_path(cfg["vad_model"])))
    asr = ParakeetASR(cfg["asr_model"])
    asr.warmup()
    groups = {"natural": NATURAL, "robotic": ROBOTIC}
    if args.voices:
        groups = {k: v[:args.voices] for k, v in groups.items()}
    pad = np.zeros(16000, np.float32)
    results = []
    started = time.monotonic()
    for group, voices in groups.items():
        for voice in voices:
            for expected, phrases in ((True, ADDRESSED), (False, NOT_ADDRESSED)):
                for phrase in phrases:
                    audio = np.concatenate([pad, synth(phrase, voice), pad, pad])
                    vad.reset()
                    seg = Segmenter(cfg)
                    texts = []
                    for i in range(0, len(audio) - 511, 512):
                        r = seg.push(audio[i:i + 512], vad(audio[i:i + 512]), (i + 512) / 16000)
                        if isinstance(r, Utterance):
                            texts.append(asr.transcribe(r.audio))
                    text = " ".join(texts)
                    got = addressing.match(text, cfg).addressed
                    results.append({"group": group, "voice": voice, "phrase": phrase, "expected": expected,
                                    "got": got, "transcript": text, "segments": len(texts)})
    tally = defaultdict(lambda: [0, 0, 0, 0])  # addressed total, missed, other total, accepted
    for r in results:
        t = tally[r["group"]]
        if r["expected"]:
            t[0] += 1
            t[1] += not r["got"]
        else:
            t[2] += 1
            t[3] += r["got"]
    print(f"{len(results)} utterances in {time.monotonic() - started:.0f}s")
    for group, (n_a, miss, n_o, acc) in tally.items():
        print(f"{group:8s} false rejects {miss}/{n_a} ({100 * miss / n_a:.1f}%)   "
              f"false accepts {acc}/{n_o} ({100 * acc / n_o:.1f}%)")
    print("\nerrors:")
    for r in results:
        if r["got"] != r["expected"]:
            kind = "MISSED " if r["expected"] else "ACCEPTED"
            print(f"  {kind} [{r['voice']}] {r['phrase']!r} -> {r['transcript']!r}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
