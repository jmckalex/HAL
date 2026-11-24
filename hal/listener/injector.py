"""Hand addressed text to the pty proxy; hold it while typing would be unsafe.

The proxy decides at the instant of typing (see session.verdict). When it
refuses, the request waits here and is retried:

    dialog      Claude Code shows a permission prompt or other dialog: the voice
                says it needs the user at the keyboard (once per episode)
    draft       unsent text in the prompt box: the voice says so (once)
    keyboard    someone is typing right now: retried quietly
    no-session  the session's hooks haven't reported yet: retried quietly
    ended / exited / unreachable: dropped (the voice says it can't reach the
                session, at most once a minute)

A held request is dropped after hold_seconds. Requests are typed in order.
Each carries an id, so a retry can never type it twice. Standard library only.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

from . import control, logs

_RETRY_QUIET = 0.25
_RETRY_HELD = 0.5


@dataclass
class Pending:
    text: str
    speech_end: float
    created: float
    ident: str = field(default_factory=control.new_id)
    next_try: float = 0.0
    reason: str = ""


class Injector:
    def __init__(self, cfg: dict, *, client_factory=control.ProxyClient.discover, say=lambda text: None,
                 clock=time.monotonic, console=lambda msg: None, log=logs.log):
        self.cfg = cfg
        self.client_factory = client_factory
        self.say = say
        self.clock = clock
        self.console = console
        self.log = log
        self.queue: collections.deque[Pending] = collections.deque()
        self.noticed: str | None = None      # reason already announced in this episode
        self.last_unreachable = -1e9
        self.submitted: list[tuple[str, float]] = []   # (text, latency) for the record
        self.last_typed: float | None = None  # wall-clock time the last request was typed (for during_turn)

    @property
    def holding(self) -> bool:
        return bool(self.queue)

    def submit(self, text: str, speech_end: float) -> None:
        now = self.clock()
        self.queue.append(Pending(text, speech_end, now, next_try=now))
        self.tick()

    def tick(self) -> None:
        now = self.clock()
        while self.queue:
            item = self.queue[0]
            if now - item.created > self.cfg["hold_seconds"]:
                self.queue.popleft()
                self.log(f"dropped a request held {now - item.created:.0f}s ({item.reason})")
                self.console(f"dropped a request held {now - item.created:.0f}s ({item.reason})")
                continue
            if now < item.next_try:
                return
            if not self._attempt(item, now):
                return
            now = self.clock()
        self.noticed = None

    def _attempt(self, item: Pending, now: float) -> bool:
        """True if the item is finished with (typed or dropped)."""
        try:
            client = self.client_factory()
            reply = client.type(item.text, item.ident)
        except control.Unreachable as exc:
            self.queue.popleft()
            self.log(f"could not reach the session ({exc}); request dropped")
            self.console(f"could not reach the Claude Code session ({exc}); request dropped")
            if now - self.last_unreachable > 60:
                self.last_unreachable = now
                self.say(self.cfg["unreachable_notice"])
            return True
        if reply.get("ok"):
            self.queue.popleft()
            self.last_typed = time.time()
            latency = self.clock() - item.speech_end
            self.submitted.append((item.text, latency))
            held = f", held {now - item.created:.1f}s ({item.reason})" if item.reason else ""
            self.log(f"typed into the session {latency:.2f}s after the end of speech{held}")
            self.console(f"typed into Claude Code ({latency:.2f}s after end of speech{held})")
            return True
        reason = str(reply.get("reason") or "refused")
        detail = str(reply.get("detail") or "")
        if reason in ("ended", "exited", "empty", "too-large", "bad-request", "unknown-op"):
            self.queue.popleft()
            self.log(f"request dropped: {reason} {detail}")
            self.console(f"request dropped: {reason} {detail}")
            return True
        if reason != item.reason:
            self.log(f"holding a request: {reason} ({detail})")
            self.console(f"holding: {reason} ({detail})")
        item.reason = reason
        if reason == "dialog" and self.noticed != "dialog":
            self.noticed = "dialog"
            self.say(self.cfg["dialog_notice"])
        elif reason == "draft" and self.noticed not in ("draft", "dialog"):
            self.noticed = "draft"
            self.say(self.cfg["draft_notice"])
        item.next_try = now + (_RETRY_QUIET if reason in ("keyboard", "typing") else _RETRY_HELD)
        return False
