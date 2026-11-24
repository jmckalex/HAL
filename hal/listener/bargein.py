"""Interrupting the voice (duplex "aec"): pure logic, standard library only.

With echo cancellation (duplex.py) the listener keeps listening while HAL's
voice plays, so the user can talk over it. This module decides, without any
audio or process of its own:

    BargeIn.frame()   frame by frame, while the voice plays: is someone else
                      speaking over it? Speech-like frames (VAD) that are also
                      louder than what is left of the voice's echo, for most of
                      a short window: "pause" the voice at once, before the
                      words are known (aim: silent within half a second).
    resolve()         once the utterance has been transcribed: what was it?
                        echo      the voice's own words (compared with the text
                                  it is speaking): resume the voice, drop it
                        noise     no words, or one stray word: resume
                        resume    "go on", "continue": resume
                        stop      a stop phrase ("stop", "that's enough",
                                  "thank you, Claude"): silence the voice; the
                                  words are not passed on
                        request   anything else: silence the voice and pass the
                                  utterance on to the turn logic, without the name
    Spoken            what the voice is saying (the sidecar that
                      hal.speaker.synthesize() writes next to each WAV): its
                      text, its sentences and their times, so a cut can be
                      resumed from the start of the sentence it fell in.

Nothing here logs or stores what the user said; the listener does that under
the usual privacy rules.
"""

from __future__ import annotations

import collections
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Words dropped before a phrase is recognised: names of address, politeness,
# fillers. ("Okay, stop." / "Thank you, Claude." / "Please go on.")
_FILLER = {"please", "ok", "okay", "alright", "right", "yes", "yeah", "no", "now", "just", "um", "uh", "er",
           "erm", "oh", "ah", "hey", "dr", "doctor", "alexander", "hal", "so", "well", "and", "then", "hmm"}


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("’", "'"))


def _names(cfg: dict) -> set[str]:
    return {"claude"} | set(cfg.get("names") or ()) | set(cfg.get("vocative_names") or ())


def _core(text: str, cfg: dict) -> list[str]:
    skip = _FILLER | _names(cfg)
    return [w for w in words(text) if w not in skip]


def _phrase_words(phrases) -> list[list[str]]:
    out = []
    for p in phrases or ():
        w = [x for x in words(p) if x not in _FILLER] or words(p)
        if w:
            out.append(w)
    return out


def _made_of(core: list[str], phrases: list[list[str]], most: int = 3) -> bool:
    """Is core a sequence of at most `most` phrases ("thank you, that's enough")?"""
    if not core:
        return False
    best = [math.inf] * (len(core) + 1)
    best[0] = 0
    for i in range(len(core)):
        if best[i] >= most:
            continue
        for p in phrases:
            if core[i:i + len(p)] == p:
                best[i + len(p)] = min(best[i + len(p)], best[i] + 1)
    return best[len(core)] <= most


def is_stop_phrase(text: str, cfg: dict) -> bool:
    """"Stop.", "That's enough.", "Thank you, Claude.", "Okay, thanks, stop." (barge_in_stop_phrases)."""
    return _made_of(_core(text, cfg), _phrase_words(cfg.get("barge_in_stop_phrases")))


def is_resume_phrase(text: str, cfg: dict) -> bool:
    """"Go on.", "Continue, please.", "Claude, carry on." (barge_in_resume_phrases)."""
    return _made_of(_core(text, cfg), _phrase_words(cfg.get("barge_in_resume_phrases")), most=2)


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (a is short: an utterance)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def echo_score(heard: str, spoken: str) -> tuple[int, int]:
    """(words of `heard` found, in order, in `spoken`; words heard)."""
    h = words(heard)
    return _lcs(h, words(spoken)), len(h)


def is_echo(heard: str, spoken: str, min_ratio: float = 0.6) -> bool:
    """Was `heard` the voice itself? Most of its words, in order, are in what the voice was saying.

    One word alone counts only if the voice was saying it; a stop or resume
    phrase is checked before this (the user's "stop" wins)."""
    found, n = echo_score(heard, spoken)
    if n == 0:
        return False
    if n == 1:
        return found == 1
    return found >= 2 and found / n >= min_ratio


# ---------------------------------------------------------------------------
# What the voice is saying
# ---------------------------------------------------------------------------

def sidecar_path(wav: str | Path) -> Path:
    return Path(f"{wav}.json")


@dataclass
class Spoken:
    """A WAV's text and sentence times (seconds from its start)."""

    text: str
    sentences: list[dict] = field(default_factory=list)   # {"text", "start", "end"}
    seconds: float = 0.0
    wav: str = ""

    @classmethod
    def load(cls, wav: str | Path | None) -> "Spoken | None":
        if not wav:
            return None
        try:
            data = json.loads(sidecar_path(wav).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return None
        sentences = [s for s in data.get("sentences") or [] if isinstance(s, dict) and "text" in s]
        return cls(data["text"], sentences, float(data.get("seconds") or 0.0), str(wav))

    def index_at(self, position: float) -> int:
        """Index of the sentence being spoken at position (seconds); len() if past the end."""
        for i, s in enumerate(self.sentences):
            if position < float(s.get("end", 0.0)):
                return i
        return len(self.sentences)

    def around(self, start: float, end: float, before: float = 2.0, after: float = 1.0) -> str:
        """The text spoken from `before` seconds before start to `after` after end (echo arrives late)."""
        if not self.sentences:
            return self.text
        lo, hi = start - before, end + after
        parts = [s["text"] for s in self.sentences
                 if float(s.get("end", 0.0)) >= lo and float(s.get("start", 0.0)) <= hi]
        return " ".join(parts) or self.text

    def rest_from(self, position: float) -> tuple[int, str]:
        """Where to resume after a cut at position: the start of the sentence it fell in."""
        i = self.index_at(position)
        return i, " ".join(s["text"] for s in self.sentences[i:]).strip()

    def sentence(self, index: int) -> str:
        """The text of sentence `index` ("" past the end)."""
        if 0 <= index < len(self.sentences):
            return str(self.sentences[index].get("text") or "").strip()
        return ""


# ---------------------------------------------------------------------------
# Speech over the voice
# ---------------------------------------------------------------------------

def level_db(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), 1e-9))


class BargeIn:
    """Frame-by-frame detector of the user talking over the voice.

    A frame counts as the user's speech when its VAD probability is at least
    barge_in_vad and its level (after echo cancellation) is at least the floor:
    barge_in_min_db, and no less than barge_in_echo_margin below what the
    voice is rendering at that moment (residual echo follows the voice's own
    level; the user's speech does not). When frames amounting to
    barge_in_speech seconds count within the last barge_in_within seconds,
    frame() returns "pause"."""

    def __init__(self, cfg: dict, frame_seconds: float = 0.032):
        self.cfg = cfg
        size = max(1, round(float(cfg["barge_in_within"]) / frame_seconds))
        self.window: collections.deque = collections.deque(maxlen=size)
        self.need = max(1, min(round(float(cfg["barge_in_speech"]) / frame_seconds), size))
        self.paused = False
        self.triggered_at: float | None = None
        self.overlap_until = -1e9    # the voice was playing (or its echo still ringing) until then

    def reset(self) -> None:
        self.window.clear()

    def floor(self, ref_db: float | None) -> float:
        lo = float(self.cfg["barge_in_min_db"])
        if ref_db is None or ref_db <= -90:
            return lo
        return max(lo, ref_db - float(self.cfg["barge_in_echo_margin"]))

    def counts(self, prob: float, frame_db: float, ref_db: float | None) -> bool:
        return prob >= float(self.cfg["barge_in_vad"]) and frame_db >= self.floor(ref_db)

    def frame(self, t: float, prob: float, frame_db: float, playing: bool, ref_db: float | None = None) -> str | None:
        if playing:
            self.overlap_until = t + float(self.cfg.get("barge_in_tail", 0.5))
        if self.paused or not playing:
            self.window.clear()
            return None
        self.window.append(self.counts(prob, frame_db, ref_db))
        if sum(self.window) >= self.need:
            self.window.clear()
            self.paused = True
            self.triggered_at = t
            return "pause"
        return None

    def overlaps(self, speech_start: float) -> bool:
        """Did an utterance starting at speech_start begin while the voice played (or its echo rang)?"""
        return speech_start <= self.overlap_until

    def settle(self) -> None:
        """The paused voice has been resumed or stopped."""
        self.paused = False
        self.triggered_at = None
        self.window.clear()


def resolve(text: str, spoken_text: str, cfg: dict) -> str:
    """What an utterance said over the voice was: "noise", "stop", "resume", "echo" or "request".

    A single word that is neither a stop nor a resume phrase counts as noise: it is
    what residual echo most often turns into, and one word is not a request."""
    heard = words(text)
    if not heard:
        return "noise"
    if is_stop_phrase(text, cfg):
        return "stop"
    if is_resume_phrase(text, cfg):
        return "resume"
    if len(heard) == 1:
        return "noise"
    if spoken_text and is_echo(text, spoken_text, float(cfg.get("barge_in_echo_ratio", 0.6))):
        return "echo"
    return "request"


@dataclass
class Interrupted:
    """Where the voice was cut off, for "go on" (also written to listen/interrupted.json)."""

    at: float                  # clock time of the cut
    wall: float                # time.time() of the cut
    position: float            # seconds into the WAV
    sentence: int              # index of the sentence it fell in
    sentences: int             # how many there were
    rest: str                  # the text from the start of that sentence
    why: str                   # "a stop phrase", "speech over it", ...
    play_id: object = None     # the helper's id for that playback
    reached: str = ""          # the sentence it was cut in ("" if it had said them all)

    def to_json(self) -> dict:
        return {"time": self.wall, "position": round(self.position, 2), "sentence": self.sentence,
                "sentences": self.sentences, "rest": self.rest, "why": self.why}
