"""What to do with a transcribed utterance. Pure logic, standard library only.

    submit          addressed to Claude (or accepted without the name, below):
                    type it into the session
    name_only       just the name ("Claude?"): answer "Yes?" (name_only_reply) and
                    accept the next utterance without the name (armed)
    stop_listening  "Claude, stop listening": pause
    ignore          not addressed to Claude
    empty           no words recognised
    partial         one part of an utterance cut at max_utterance; wait for the rest

An utterance needs no name when its speech starts while armed (until
armed_seconds after the "Yes?" finished) or within
follow_up_seconds after the voice finished speaking a reply. The caller can
also say that it needs none because it continues a pending turn (turns.py,
how "continued"), because Claude is still working on a spoken request
(during_turn, how "during-turn") or because it was said over the voice and
interrupted it (duplex "aec", how "barge-in"). With turn_taking "semantic", "submit" means
a fragment of a turn, which the caller holds until the turn is finished.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import addressing

_MAX_CHAIN = 8   # parts of a cut utterance joined at most (8 x max_utterance)


@dataclass
class Decision:
    action: str
    text: str = ""
    how: str = ""          # "named", "armed", "continued", "during-turn", "barge-in" or "follow-up"
    speech_seconds: float = 0.0
    duration: float = 0.0


class Decider:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.armed_until = -1e9
        self._chain: list[str] = []
        self._chain_speech = 0.0
        self._chain_duration = 0.0

    def _reset_chain(self) -> None:
        self._chain, self._chain_speech, self._chain_duration = [], 0.0, 0.0

    def forget(self) -> None:
        """Drop the parts of a cut utterance (its end was discarded)."""
        self._reset_chain()

    def arm(self, until: float) -> None:
        self.armed_until = max(self.armed_until, until)

    def disarm(self) -> None:
        self.armed_until = -1e9

    def armed(self, t: float) -> bool:
        return t <= self.armed_until

    def decide(self, text: str, utt, now: float, reply_end: float = -1e9, *, pending: bool = False,
               during_turn: bool = False, barge_in: bool = False, named_only: bool = False) -> Decision:
        """utt needs .speech_start, .speech_seconds, .duration, .cut, .continued.

        pending: a turn is being held, and this utterance continues it.
        during_turn: Claude is still working on a spoken request.
        barge_in: it was said over the voice and interrupted it (duplex "aec").
        named_only: it was heard over the voice without interrupting it, so it may be
            the voice's residual echo: only the name counts, no window applies."""
        text = (text or "").strip()
        speech, duration = utt.speech_seconds, utt.duration
        if utt.cut:
            if not utt.continued:
                self._reset_chain()
            self._chain.append(text)
            self._chain_speech += speech
            self._chain_duration += duration
            if len(self._chain) < _MAX_CHAIN:
                return Decision("partial", speech_seconds=speech, duration=duration)
            # Very long: decide on what there is; the rest is judged on its own.
            text, speech, duration = " ".join(self._chain), self._chain_speech, self._chain_duration
        elif utt.continued and self._chain:
            text = " ".join(self._chain + [text])
            speech += self._chain_speech
            duration += self._chain_duration
        text = " ".join(text.split())
        self._reset_chain()

        if not text:
            return Decision("empty", speech_seconds=speech, duration=duration)
        m = addressing.match(text, self.cfg)
        if m.addressed:
            if m.command:
                self.disarm()
                return Decision(m.command, m.text, "named", speech, duration)
            if m.name_only:
                self.arm(now + self.cfg["name_only_grace"] + self.cfg["armed_seconds"])
                return Decision("name_only", m.text, "named", speech, duration)
            self.disarm()
            return Decision("submit", m.text, "named", speech, duration)
        follow = self.cfg["follow_up_seconds"]
        how = ""
        if named_only:
            return Decision("ignore", speech_seconds=speech, duration=duration)
        if self.armed(utt.speech_start):
            how = "armed"
        elif pending:
            how = "continued"
        elif during_turn:
            how = "during-turn"
        elif barge_in:
            how = "barge-in"
        elif follow > 0 and 0 <= utt.speech_start - reply_end <= follow:
            how = "follow-up"
        if how:
            self.disarm()
            command = addressing.command_without_name(text, self.cfg)
            return Decision(command or "submit", text, how, speech, duration)
        return Decision("ignore", speech_seconds=speech, duration=duration)
