"""Cut a stream of VAD-scored frames into utterances.

push() takes one frame, its speech probability and the time at the end of the
frame, and returns an Utterance when one completes (or a Discarded record for
a burst too short to be speech). The rules follow Silero's own VADIterator,
with a longer end-of-utterance silence suited to spoken requests:

    start   the first frame with probability >= threshold; the pre_roll seconds
            of audio before it are included so soft onsets are never clipped
    end     end_silence seconds without a frame >= threshold, counted from the
            first frame below neg_threshold (frames in between neither reset
            nor start the count, as in Silero); post_roll seconds of that
            silence are kept
    short   fewer than min_speech seconds of speech frames: discarded as noise
    long    at max_utterance seconds the utterance is cut and returned with
            cut=True; the rest continues as a new utterance with continued=True,
            and the caller joins the parts
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

FRAME_SECONDS = 512 / 16000


@dataclass
class Utterance:
    audio: np.ndarray | None
    start: float            # time at the start of the audio (pre-roll included)
    speech_start: float     # start of the first speech frame
    speech_end: float       # end of the last speech frame: "end of speech"
    end: float              # end of the audio (post-roll included)
    speech_seconds: float   # total duration of frames scored as speech
    cut: bool = False       # cut at max_utterance; the next utterance continues it
    continued: bool = False  # continues a cut utterance

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Discarded:
    reason: str             # "short" (noise) or "gated" (the voice started speaking)
    seconds: float


class Segmenter:
    def __init__(self, cfg: dict, frame_seconds: float = FRAME_SECONDS):
        self.dt = frame_seconds
        self.threshold = cfg["vad_threshold"]
        self.neg_threshold = cfg["vad_neg_threshold"]
        self.end_frames = max(1, round(cfg["end_silence"] / self.dt))
        self.pre_frames = max(1, round(cfg["pre_roll"] / self.dt))
        self.post_frames = max(0, round(cfg["post_roll"] / self.dt))
        self.min_speech_frames = max(1, round(cfg["min_speech"] / self.dt))
        self.max_frames = max(self.end_frames + 1, round(cfg["max_utterance"] / self.dt))
        self._preroll: collections.deque = collections.deque(maxlen=self.pre_frames)
        self._clear()
        self._continued = False

    def _clear(self) -> None:
        self._frames: list[np.ndarray] | None = None
        self._first_time = 0.0
        self._speech_start = 0.0
        self._last_speech_end = 0.0
        self._speech_frames = 0
        self._silence_run = 0

    @property
    def in_speech(self) -> bool:
        return self._frames is not None

    def push(self, frame: np.ndarray, prob: float, t_end: float):
        if self._frames is None:
            self._preroll.append(frame)
            if prob < self.threshold:
                return None
            self._frames = list(self._preroll)
            self._preroll.clear()
            self._first_time = t_end - len(self._frames) * self.dt
            self._speech_start = t_end - self.dt
            self._last_speech_end = t_end
            self._speech_frames = 1
            self._silence_run = 0
            return None

        self._frames.append(frame)
        if prob >= self.threshold:
            self._speech_frames += 1
            self._silence_run = 0
            self._last_speech_end = t_end
        elif self._silence_run > 0 or prob < self.neg_threshold:
            self._silence_run += 1
        if self._silence_run >= self.end_frames:
            return self._finish(t_end)
        if len(self._frames) >= self.max_frames:
            return self._cut(t_end)
        return None

    def _finish(self, t_end: float):
        frames = self._frames
        # After a cut at the length limit the buffer can hold only trailing
        # silence, so nothing may be left to keep (this crashed the listener on
        # 16 Nov 2026 during several minutes of continuous background speech).
        keep = max(0, min(len(frames), len(frames) - self._silence_run + self.post_frames))
        continued = self._continued
        result: Utterance | Discarded
        if keep == 0 or (self._speech_frames < self.min_speech_frames and not continued):
            result = Discarded("short", self._speech_frames * self.dt)
        else:
            result = Utterance(np.concatenate(frames[:keep]), self._first_time, self._speech_start,
                               self._last_speech_end, self._first_time + keep * self.dt,
                               self._speech_frames * self.dt, cut=False, continued=continued)
        # The trailing silence can serve as pre-roll for what comes next.
        self._preroll.extend(frames[-self.pre_frames:])
        self._clear()
        self._continued = False
        return result

    def _cut(self, t_end: float) -> Utterance:
        frames = self._frames
        result = Utterance(np.concatenate(frames), self._first_time, self._speech_start, self._last_speech_end,
                           t_end, self._speech_frames * self.dt, cut=True, continued=self._continued)
        self._frames = []
        self._first_time = t_end
        self._speech_start = t_end
        self._speech_frames = 0
        self._continued = True
        return result

    def flush(self, t_end: float):
        """End of input: finish an utterance in progress as if silence followed."""
        if self._frames is None:
            return None
        self._silence_run = min(self._silence_run, len(self._frames))
        return self._finish(t_end)

    def reset(self):
        """Drop everything (the voice started speaking). Returns what was dropped, if anything."""
        dropped = None
        if self._frames is not None:
            dropped = Discarded("gated", len(self._frames) * self.dt)
        self._clear()
        self._preroll.clear()
        self._continued = False
        return dropped
