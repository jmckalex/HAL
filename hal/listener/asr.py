"""Speech-to-text with Parakeet (parakeet-mlx), kept loaded, entirely in memory.

parakeet-mlx's own transcribe() reads a file through ffmpeg; this feeds the
samples straight to the model instead, so captured audio never touches disk.

Confidence. parakeet-mlx 0.7.0 gives each token (a SentencePiece piece) of its
greedy decoding a `confidence`: 1 minus the entropy of the model's distribution
over the vocabulary and blank at that step, divided by its maximum,
log(vocabulary + 1) (parakeet.py, decode_greedy). It offers no alternative
words: beam search (Beam) keeps only the best hypothesis, and its confidences
are exp(log-probabilities) instead. transcribe_words() groups the pieces into
words (a piece starting with a space starts a word) and gives each word the
lowest confidence of its letter-bearing pieces; punctuation pieces don't count.
On synthetic speech, clearly heard words score 0.9 to 1.0 and misheard names
0.56 to 0.87 (annotate.py marks words below doubt_threshold, 0.82).
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import settings
from .annotate import HeardWord

_PAD = np.zeros(1600, dtype=np.float32)   # 0.1 s of silence on each side


class ParakeetASR:
    # A turn's fragments may be transcribed afresh as one piece (turns.py).
    # About 6 ms per second of audio on an M4 Max: 60 s in 0.4 s.
    rejoins = True

    def __init__(self, model: str):
        # Any download goes under models/hf/, never ~/.cache.
        os.environ.setdefault("HF_HOME", str(settings.ROOT / "models" / "hf"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        path = settings.project_path(model)
        if path.is_dir():
            os.environ["HF_HUB_OFFLINE"] = "1"
            ref = str(path)
        else:
            ref = model  # a HuggingFace id
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx, self._get_logmel = mx, get_logmel
        started = time.monotonic()
        self.model = from_pretrained(ref)
        self.load_seconds = time.monotonic() - started
        self.name = os.path.basename(ref.rstrip("/"))

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_words(audio)[0]

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[HeardWord]]:
        """(text, its words with their confidences and times in seconds from the start of `audio`)."""
        if audio is None or len(audio) < 400:
            return "", []
        samples = np.concatenate([_PAD, np.asarray(audio, dtype=np.float32), _PAD])
        mel = self._get_logmel(self._mx.array(samples), self.model.preprocessor_config)
        result = self.model.generate(mel)[0]
        return " ".join(result.text.split()), words_of(result.tokens, offset=len(_PAD) / 16000)

    def warmup(self) -> float:
        started = time.monotonic()
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        self.transcribe(0.01 * np.sin(2 * np.pi * 220 * t).astype(np.float32))
        return time.monotonic() - started


def words_of(tokens, offset: float = 0.0) -> list[HeardWord]:
    """Group parakeet-mlx's AlignedTokens into words (see the module docstring)."""
    words: list[HeardWord] = []
    confs: list[list[float]] = []
    for t in tokens:
        text = str(getattr(t, "text", ""))
        if not words or text.startswith(" "):
            words.append(HeardWord(text.strip(), None, float(t.start) - offset, float(t.end) - offset))
            confs.append([])
        else:
            words[-1].text += text
            words[-1].end = float(t.end) - offset
        if any(ch.isalnum() for ch in text):
            confs[-1].append(float(getattr(t, "confidence", 1.0)))
    for w, c in zip(words, confs):
        w.conf = min(c) if c else None
    return [w for w in words if w.text]
