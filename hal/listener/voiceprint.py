"""Telling your voice from others: a voiceprint and a check against it.

A speaker-embedding model (WeSpeaker ResNet34, trained on VoxCeleb, run with
onnxruntime) turns a stretch of speech into 256 numbers that are close
together for one voice and further apart for different voices. Enrolment
(`hal-listen enrol`) records you reading sentences and keeps one embedding
per sentence in state/listen/voiceprint.json; their normalised average is the
voiceprint. The listener then scores each utterance by its cosine similarity
to the voiceprint (voice_check "log" records the scores; "enforce" also acts
on them, see service.py).

The voiceprint is biometric data: it stays in the state directory (never in
git) and is written readable by you alone. No audio is ever stored, during
enrolment or after. numpy and onnxruntime only.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import settings

RATE = 16000
MIN_SECONDS = 0.5          # shorter speech gives no embedding at all

# ---------------------------------------------------------------- features

_FRAME, _SHIFT, _NFFT, _MELS = 400, 160, 512, 80
_EPS = np.finfo(np.float32).eps


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_banks(low: float = 20.0, high: float = RATE / 2) -> np.ndarray:
    """Kaldi's triangular mel filters, (80, 257): the last FFT bin gets no weight."""
    mel_low, mel_high = _mel(low), _mel(high)
    delta = (mel_high - mel_low) / (_MELS + 1)
    bins = _mel(np.arange(_NFFT // 2) * RATE / _NFFT)
    banks = np.zeros((_MELS, _NFFT // 2 + 1))
    for b in range(_MELS):
        left, center, right = mel_low + b * delta, mel_low + (b + 1) * delta, mel_low + (b + 2) * delta
        up = (bins - left) / (center - left)
        down = (right - bins) / (right - center)
        banks[b, : _NFFT // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks()
_WINDOW = np.hamming(_FRAME)


def fbank(samples: np.ndarray) -> np.ndarray:
    """80 log mel energies per 10 ms frame, as Kaldi computes them (torchaudio's
    kaldi.fbank with a Hamming window, no dither, frames inside the signal),
    with the mean over time subtracted. samples: float32 in [-1, 1] at 16 kHz."""
    x = np.asarray(samples, dtype=np.float64) * 32768.0    # WeSpeaker scales to the 16-bit range
    if len(x) < _FRAME:
        return np.zeros((0, _MELS), np.float32)
    n = 1 + (len(x) - _FRAME) // _SHIFT
    idx = np.arange(_FRAME)[None, :] + _SHIFT * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)                       # DC offset
    frames = frames - 0.97 * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)  # pre-emphasis
    frames = frames * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=_NFFT, axis=1)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, _EPS))
    feats -= feats.mean(axis=0, keepdims=True)                                 # mean normalisation
    return feats.astype(np.float32)


# ---------------------------------------------------------------- the model

class Embedder:
    """samples -> unit-length embedding (None for too little audio)."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.name = Path(model_path).name

    def __call__(self, samples: np.ndarray) -> np.ndarray | None:
        if len(samples) < MIN_SECONDS * RATE:
            return None
        feats = fbank(samples)
        emb = self.session.run(None, {self.input: feats[None]})[0][0].astype(np.float64)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else None


def model_path(cfg: dict) -> Path:
    return settings.project_path(cfg["voice_model"])


# ---------------------------------------------------------------- the voiceprint

def voiceprint_file() -> Path:
    return settings.listen_dir() / "voiceprint.json"


@dataclass
class Voiceprint:
    model: str
    embeddings: np.ndarray      # (n, 256), one per enrolment sentence, unit length
    created: str
    updated: str

    @property
    def vector(self) -> np.ndarray:
        v = self.embeddings.mean(axis=0)
        return v / np.linalg.norm(v)

    def score(self, emb: np.ndarray) -> float:
        return float(np.dot(self.vector, emb))

    def self_scores(self) -> list[float]:
        """Each enrolment sentence against the average of the others (leave one out)."""
        n = len(self.embeddings)
        if n < 2:
            return []
        total = self.embeddings.sum(axis=0)
        out = []
        for e in self.embeddings:
            rest = total - e
            out.append(float(np.dot(e, rest / np.linalg.norm(rest))))
        return out


def load(path: Path | None = None) -> Voiceprint | None:
    path = path or voiceprint_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or len(emb) == 0:
            return None
        return Voiceprint(str(data.get("model", "")), emb, str(data.get("created", "")), str(data.get("updated", "")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(vp: Voiceprint, path: Path | None = None) -> Path:
    path = path or voiceprint_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": vp.model, "created": vp.created, "updated": vp.updated,
            "embeddings": [[round(float(x), 6) for x in e] for e in vp.embeddings]}
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return path


def forget(path: Path | None = None) -> bool:
    path = path or voiceprint_file()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------- learning from use

# Embeddings of your own requests, added as you use the voice (voice_learn):
# only requests scored well above the threshold, with enough speech, a few a
# day at most, the newest voice_learn_max kept. They are blended with the
# enrolment, which always keeps at least half the weight, so the voiceprint
# can follow rooms, distances and days without drifting away from you.
LEARNED_WEIGHT = 0.5


def learned_file() -> Path:
    return settings.listen_dir() / "voiceprint-learned.json"


@dataclass
class Learned:
    embeddings: np.ndarray                         # (k, 256), unit length
    days: list[str] = field(default_factory=list)  # the day each was added, YYYY-MM-DD
    scores: list[float] = field(default_factory=list)


def load_learned(path: Path | None = None) -> Learned | None:
    path = path or learned_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or len(emb) == 0:
            return None
        days = [str(d) for d in data.get("days", [])][: len(emb)]
        scores = [float(x) for x in data.get("scores", [])][: len(emb)]
        return Learned(emb, days + [""] * (len(emb) - len(days)), scores + [0.0] * (len(emb) - len(scores)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_learned(learned: Learned, path: Path | None = None) -> Path:
    path = path or learned_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"days": learned.days, "scores": [round(x, 3) for x in learned.scores],
            "embeddings": [[round(float(x), 6) for x in e] for e in learned.embeddings]}
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return path


def forget_learned(path: Path | None = None) -> bool:
    return forget(path or learned_file())


def combined(vp: Voiceprint, learned: Learned | None) -> np.ndarray:
    """The voiceprint used for scoring: the enrolment, blended with what was
    learned. The learned part's weight grows with its size, up to LEARNED_WEIGHT
    once there are as many learned embeddings as enrolled ones."""
    enrolled = vp.vector
    if learned is None or len(learned.embeddings) == 0 or learned.embeddings.shape[1:] != vp.embeddings.shape[1:]:
        return enrolled
    lv = learned.embeddings.mean(axis=0)
    lv = lv / np.linalg.norm(lv)
    w = LEARNED_WEIGHT * min(1.0, len(learned.embeddings) / len(vp.embeddings))
    v = (1 - w) * enrolled + w * lv
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------- the check

@dataclass
class Verdict:
    score: float
    match: bool         # at or above voice_threshold
    reliable: bool      # enough speech (voice_min_seconds) to trust a mismatch
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    def describe(self) -> str:
        if self.match:
            return "you"
        return "another voice" if self.reliable else "unclear (too short to judge)"


class VoiceCheck:
    """callable(audio, speech_seconds) -> Verdict, or None when there is no
    voiceprint for the configured model or no model. The voiceprint (and what
    was learned) is reread when its file changes, so enrolling takes effect
    without a restart. learn() adds a request of yours (voice_learn)."""

    def __init__(self, cfg: dict, embedder=None, path: Path | None = None, log=lambda msg: None,
                 learned_path: Path | None = None, today=lambda: time.strftime("%Y-%m-%d")):
        self.cfg = cfg
        self.path = path or voiceprint_file()
        self.learned_path = learned_path or (self.path.with_name("voiceprint-learned.json") if path else learned_file())
        self.log = log
        self.today = today
        self._embedder = embedder
        self._embedder_failed = False
        self._mtimes: tuple | None = None
        self.voiceprint: Voiceprint | None = None
        self.learned: Learned | None = None
        self.vector: np.ndarray | None = None

    def _refresh(self) -> None:
        mtimes = []
        for p in (self.path, self.learned_path):
            try:
                mtimes.append(p.stat().st_mtime)
            except OSError:
                mtimes.append(None)
        mtimes = tuple(mtimes)
        if mtimes != self._mtimes:
            self._mtimes = mtimes
            self.voiceprint = load(self.path) if mtimes[0] is not None else None
            self.learned = load_learned(self.learned_path) if mtimes[1] is not None else None
            self.vector = combined(self.voiceprint, self.learned) if self.voiceprint is not None else None

    def embedder(self):
        if self._embedder is None and not self._embedder_failed:
            try:
                self._embedder = Embedder(model_path(self.cfg))
            except Exception as exc:  # no model: no check, never no listener
                self._embedder_failed = True
                self.log(f"voice check: cannot load the model ({exc!r}); not checking voices")
        return self._embedder

    def ready(self) -> bool:
        self._refresh()
        return self.voiceprint is not None

    def __call__(self, audio: np.ndarray, speech_seconds: float) -> Verdict | None:
        if not self.ready():
            return None
        emb_fn = self.embedder()
        if emb_fn is None:
            return None
        emb = emb_fn(audio)
        if emb is None or emb.shape != self.voiceprint.embeddings.shape[1:]:
            return None
        score = float(np.dot(self.vector, emb))
        return Verdict(score, score >= self.cfg["voice_threshold"], speech_seconds >= self.cfg["voice_min_seconds"],
                       embedding=emb)

    def learn(self, verdict: Verdict | None, speech_seconds: float) -> bool:
        """Add a request's embedding to what was learned, if it is clearly yours
        (voice_learn_threshold, voice_learn_min_seconds) and today's quota
        (voice_learn_per_day) isn't used up. True if it was added."""
        cfg = self.cfg
        if (not cfg.get("voice_learn") or verdict is None or verdict.embedding is None or self.voiceprint is None
                or verdict.score < cfg["voice_learn_threshold"] or speech_seconds < cfg["voice_learn_min_seconds"]):
            return False
        learned = load_learned(self.learned_path)
        if learned is not None and learned.embeddings.shape[1:] != verdict.embedding.shape:
            learned = None   # made with another model: start again
        today = self.today()
        if learned is not None and sum(d == today for d in learned.days) >= cfg["voice_learn_per_day"]:
            return False
        emb = verdict.embedding[None].astype(np.float32)
        if learned is None:
            learned = Learned(emb, [today], [verdict.score])
        else:
            learned = Learned(np.concatenate([learned.embeddings, emb]), learned.days + [today],
                              learned.scores + [verdict.score])
        keep = max(1, int(cfg["voice_learn_max"]))
        learned = Learned(learned.embeddings[-keep:], learned.days[-keep:], learned.scores[-keep:])
        save_learned(learned, self.learned_path)
        return True


def speech_only(audio: np.ndarray, start: float, speech_start: float, speech_end: float) -> np.ndarray:
    """The utterance's audio without its pre- and post-roll."""
    a = max(0, int((speech_start - start) * RATE))
    b = max(a, int((speech_end - start) * RATE))
    return audio[a:b] if b - a >= MIN_SECONDS * RATE else audio


# ---------------------------------------------------------------- tuning

_ADDRESSED = re.compile(r"\] heard \(.*, voice (-?\d+\.\d+) \(")
_SCORED = re.compile(r"\] voice (-?\d+\.\d+) \(([^;]*);")


def logged_scores(lines) -> dict[str, list[float]]:
    """Scores from the listener's log: "addressed" (requests that were heard,
    yours or not), "unaddressed" (speech not addressed to Claude) and "refused"
    (ignored unheard as another voice)."""
    out: dict[str, list[float]] = {"addressed": [], "unaddressed": [], "refused": []}
    for line in lines:
        m = _ADDRESSED.search(line)
        if m:
            out["addressed"].append(float(m.group(1)))
            continue
        m = _SCORED.search(line)
        if m:
            out["refused" if "ignored unheard" in m.group(2) else "unaddressed"].append(float(m.group(1)))
    return out


# ---------------------------------------------------------------- enrolment

# Harvard sentences (IEEE 1969, public domain) for their spread of sounds, and
# requests of the kind the listener hears.
SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Claude, what's the weather in London tomorrow?",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "Claude, could you look at the second chapter of my paper and check the argument?",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "Claude, please run the tests and tell me what failed.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "Claude, remind me what we decided about the lecture on Thursday.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "Claude, open my notes on the evolution of conventions.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "Claude, what's on my calendar for next week?",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Claude, summarise the changes since yesterday, briefly please.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Claude, stop for a moment, I need to think about this.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
]


def new_voiceprint(model: str, embeddings: list[np.ndarray], old: Voiceprint | None = None) -> Voiceprint:
    now = time.strftime("%Y-%m-%d %H:%M")
    emb = np.stack(embeddings).astype(np.float32)
    if old is not None and old.model == model:
        return Voiceprint(model, np.concatenate([old.embeddings, emb]), old.created, now)
    return Voiceprint(model, emb, now, now)
