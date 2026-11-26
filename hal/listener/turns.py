"""Semantic turn-taking: has the speaker finished? Standard library only.

With turn_taking "semantic" (the default), a pause ends an utterance, and an
utterance is a fragment of a turn, not a request. An addressed fragment
starts a turn; until the turn is sent, every further utterance joins it
without the name. After each fragment the whole turn so far is judged:

    complete     sent once turn_end_silence seconds of silence have followed it
    incomplete   held for more; sent as it stands after turn_max_wait seconds
                 of silence
    unclear      the judge is asked (the front desk's warm Haiku, hal/desk.py
                 op "turn"); without an answer in time (desk off, not running,
                 too slow) the turn is sent after turn_unclear_wait seconds of
                 silence

Speech that resumes before a turn is sent always joins it, whatever the
verdict was. judge_rules() is the first pass: rules that are cheap and
reliable decide what they can, and everything else is "unclear":

    unfinished  an ellipsis, a comma, colon or dash at the end; a last word
                that can't end a sentence (and, or, but, because, if, the, a,
                my, ...); a preposition at the end without the transcriber's
                full stop (it punctuates from the voice, so "... the pause for"
                with no stop is a sentence broken off); phrases such as "the
                thing is" or "but for the moment"; only filler ("So.", "Um.")
    finished    with a final . ? or !: a question that doesn't break off and
                isn't a preamble ("Can I ask you something?"); a short
                imperative ("Open my thesis.", "Please read me the summary.");
                a greeting, thanks or brief answer ("Thanks, Claude.", "Yes.")

Spoken controls, in a fragment after the first:

    send phrase    ("that's all", "go ahead", ...) as the whole fragment or
                   ending it: the turn is sent at once, without the phrase
    cancel phrase  ("never mind", "cancel that", ...) as the whole fragment or
                   ending it: the turn is discarded. At its start, followed by
                   more ("Scratch that. Open my thesis instead."): what came
                   before is discarded and the rest starts the turn afresh.

In the first fragment of a turn these are ordinary words: "Yes, go ahead."
answers a question and is typed as it is (a send phrase at its end only marks
it finished), and "Claude, never mind." is typed as before.

The name alone followed by a mark of continuation ("Claude," or "Claude...")
opens an empty turn instead of being answered: what follows needs no name. If
nothing follows within turn_max_wait, the empty turn is dropped.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import addressing

# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’][A-Za-z]+)*")

# A sentence can't end on these, whatever the punctuation says.
_OPEN_WORDS = {
    "and", "or", "but", "nor", "because", "cause", "'cause", "if", "unless", "whether", "than", "although",
    "the", "an", "my", "your", "our", "their", "every",
    "um", "uh", "er", "erm", "uhm", "umm", "hmm", "mm",
}
# Prepositions that rarely end a finished sentence. A finished one ("the book
# I was looking for.") gets a full stop from the transcriber, so these mark a
# sentence as broken off only when no stop follows, or when a question asks
# for something without a wh-word ("Could you look at?").
_PREPOSITIONS = {
    "for", "to", "of", "with", "from", "into", "onto", "at", "between", "among", "amongst", "towards",
    "toward", "regarding", "concerning", "including", "via", "per", "upon", "against", "despite", "unlike",
    "versus",
}
# Also particles ("Switch it on.", "Come in."): unfinished only without a stop.
_PARTICLES = {"about", "like", "by", "in", "on", "over", "through", "as"}
_WH = {"what", "which", "who", "whom", "whose", "where", "when", "how", "why"}
# Endings that announce more.
_OPEN_PHRASES = [p.split() for p in (
    "such as", "in other words", "the thing is", "what i mean is", "what i want is", "what i'd like is",
    "what i would like is", "what i'd like you to do is", "the question is", "my question is", "the point is",
    "the problem is", "the idea is", "i was wondering", "let me think", "let me see", "let's see",
    "first of all", "on the one hand", "on the other hand", "as well as", "rather than", "in order to",
    "so that", "sort of", "kind of", "i was thinking", "what i'm thinking is", "how should i put it",
    "how shall i put it", "but for the moment", "and for the moment", "but for now", "and for now",
    "but at the moment", "i'd like to know is", "i want to know is", "i need to know is", "i'm asking is",
    "i'm saying is", "i'm trying to say is", "and then", "but then", "or then", "and also", "and so",
)]
_FILLER_ONLY = {"so", "well", "now", "um", "uh", "er", "erm", "hmm", "and", "but", "also", "then", "anyway",
                "look", "listen", "oh", "ah", "actually", "basically", "like"}
# Questions that only announce the real one.
_PREAMBLE_QUESTION = re.compile(
    r"^(can|could|may|might) i (ask|tell|run|put|bounce|check) |^do you (have|mind)\b|^have you got a\b|"
    r"^(you know what|guess what)\b|^(can|could) (you|we) (do|help) (me )?(something|a favou?r)\b|"
    r"^(are|is) (you|it|this) (ready|okay|ok|alright)\b|\b(tell|show|give|ask|send|find|bring|get) (me|you|us)$")
# Imperative verbs that start a short, finished request.
_IMPERATIVES = {
    "open", "close", "run", "check", "tell", "show", "read", "find", "make", "fix", "add", "remove", "delete",
    "write", "send", "summarize", "summarise", "explain", "search", "play", "start", "create", "give", "list",
    "describe", "compare", "translate", "remind", "schedule", "set", "turn", "switch", "put", "move", "copy",
    "rename", "save", "commit", "push", "pull", "update", "install", "build", "test", "draft", "email", "call",
    "book", "show", "edit", "review", "print", "plot", "draw", "generate", "convert", "calculate", "count",
    "sort", "merge", "split", "restart", "continue", "proceed", "repeat", "undo", "revert", "define", "help",
    "fetch", "download", "upload", "resume", "pause", "stop", "mute", "unmute", "cancel", "clear", "reply",
    "answer", "note", "record", "look", "go", "try", "use", "keep", "change", "do", "let's", "get", "bring",
    "skip", "finish", "submit", "post", "share", "rewrite", "shorten", "lengthen", "expand", "proofread",
}
# Words that can open a longer clause, which a short-request rule can't vouch for.
_SUBORDINATORS = {"whether", "if", "because", "that", "which", "who", "when", "where", "while", "although",
                  "though", "unless", "until", "since", "so", "whose", "whom"}
_REQUEST_OPENERS = [p.split() for p in (
    "please", "can you", "could you", "would you", "will you", "can you please", "could you please",
    "would you please", "i'd like you to", "i would like you to", "i want you to", "i need you to",
)]
# A turn made only of these is a greeting, thanks or brief answer.
_BRIEF = {
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "fine", "right", "alright", "good", "great",
    "perfect", "excellent", "wonderful", "brilliant", "lovely", "splendid", "marvellous", "fantastic", "nice",
    "thanks", "thank", "you", "very", "much", "so", "cheers", "hello", "hi", "hey", "morning", "afternoon",
    "evening", "night", "goodnight", "goodbye", "bye", "well", "done", "that's", "thats", "it's", "its",
    "helpful", "useful", "interesting", "fascinating", "of", "course", "indeed", "agreed", "exactly",
    "correct", "understood", "got", "it", "i", "see", "sounds", "please", "do", "that", "is", "was", "just",
    "what", "needed", "a", "lot", "appreciated", "much", "oh", "ah", "um", "uh", "the", "same", "too",
    "dr", "doctor", "excuse", "me", "sorry",
}
_BRIEF_NEEDS = {"yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "fine", "right", "alright", "good",
                "great", "perfect", "excellent", "wonderful", "brilliant", "lovely", "splendid", "marvellous",
                "fantastic", "nice", "thanks", "thank", "cheers", "hello", "hi", "hey", "morning", "afternoon",
                "evening", "goodnight", "goodbye", "bye", "helpful", "useful", "interesting", "fascinating",
                "indeed", "agreed", "exactly", "correct", "understood", "appreciated", "sorry", "night"}
# Ignored when recognising a spoken control ("No, never mind.", "That's all, thank you.").
_CONTROL_FILLER = {"please", "ok", "okay", "right", "so", "and", "um", "uh", "er", "erm", "oh", "well", "now",
                   "then", "just", "actually", "no", "yes", "yeah", "sorry", "thanks", "thank", "you", "hmm",
                   "alright", "for", "now", "dr", "doctor", "alexander", "oops"}


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(text)]


def _names(cfg: dict) -> set[str]:
    return {"claude"} | set(cfg.get("names") or ()) | set(cfg.get("vocative_names") or ())


def _ends_with(words: list[str], phrase: list[str]) -> bool:
    return len(words) >= len(phrase) and words[-len(phrase):] == phrase


def _last_sentence(text: str) -> str:
    parts = [p for p in re.split(r"(?<=[.?!…])\s+", text.strip()) if _WORD.search(p)]
    return parts[-1] if parts else ""


def _core(words: list[str], cfg: dict) -> list[str]:
    """The words of a sentence after the name and the words that lead up to it."""
    skip = _names(cfg) | set(cfg.get("lead_words") or ()) | set(cfg.get("name_only_words") or ())
    i = 0
    while i < len(words):
        if words[i:i + 3] == ["by", "the", "way"]:
            i += 3
        elif words[i] in skip:
            i += 1
        else:
            break
    rest = words[i:]
    while rest and rest[-1] in _names(cfg):   # "Open my thesis, Claude."
        rest = rest[:-1]
    return rest


def judge_rules(text: str, cfg: dict) -> tuple[str, str]:
    """("complete" | "incomplete" | "unclear", why) for the turn so far."""
    t = " ".join((text or "").split())
    words = _words(t)
    if not words:
        return "incomplete", "nothing said yet"
    if t.endswith(("...", "…", ". . .")):
        return "incomplete", "trails off (…)"
    end = t[-1]
    if end in ",;:-–—":
        return "incomplete", f"ends with {end!r}"
    terminal = end in ".?!"
    question = end == "?"
    last = words[-1]
    last_written = _WORD.findall(t)[-1]
    if last in _OPEN_WORDS or (last == "a" and last_written == "a"):   # "A." is a letter, "a." an article
        return "incomplete", f"ends with {last_written!r}"
    sentence = _last_sentence(t)
    s_words = _words(sentence)
    core = _core(s_words, cfg)
    if last in _PREPOSITIONS or last in _PARTICLES:
        wh = any(w in _WH for w in s_words[:-1])
        if not terminal:
            return "incomplete", f"ends with {last_written!r}"
        if last in _PREPOSITIONS and not wh:
            if question:
                return "incomplete", f"asks with {last_written!r} at the end"
            return "unclear", f"ends with {last_written!r}"
    if not question:
        for phrase in _OPEN_PHRASES:
            if _ends_with(words, phrase):
                return "incomplete", f"ends with {' '.join(phrase)!r}"
    names = _names(cfg)
    content = [w for w in words if w not in names]
    if not content:
        return "incomplete", "nothing after the name"
    if all(w in _FILLER_ONLY for w in content):
        return "incomplete", "only filler so far"
    if not terminal:
        return "unclear", "no final punctuation"
    # Greetings, thanks and brief answers.
    if len(content) <= 8 and all(w in _BRIEF for w in content) and any(w in _BRIEF_NEEDS for w in content):
        return "complete", "a greeting, thanks or brief answer"
    if not core:
        return "unclear", "nothing after the name"
    if question:
        joined = " ".join(core)
        if _PREAMBLE_QUESTION.search(joined):
            return "unclear", "a question that may only announce another"
        if len(core) >= 2:
            return "complete", "a question"
        return "unclear", "a very short question"
    # A short imperative or polite request.
    body = core
    for opener in sorted(_REQUEST_OPENERS, key=len, reverse=True):
        if body[:len(opener)] == opener:
            body = body[len(opener):]
            break
    if (body and body[0] in _IMPERATIVES and len(body) >= 2 and len(core) <= 12
            and not any(w in _SUBORDINATORS for w in body)
            and not re.match(r"^\W*\w+\s*[,—–-]", _strip_to(sentence, body[0]))):
        return "complete", "a short request"
    return "unclear", "a statement"


def _strip_to(sentence: str, word: str) -> str:
    """The sentence from the first occurrence of word (to see what follows the verb)."""
    m = re.search(rf"\b{re.escape(word)}\b", sentence, re.I)
    return sentence[m.start():] if m else sentence


# ---------------------------------------------------------------------------
# Spoken controls: send and cancel phrases
# ---------------------------------------------------------------------------

_CLAUSE = re.compile(r"[^.,;:!?—–…]+")


def _clauses(text: str) -> list[re.Match]:
    return [m for m in _CLAUSE.finditer(text) if _WORD.search(m.group(0))]


def _meaningful(words: list[str], cfg: dict) -> list[str]:
    skip = _CONTROL_FILLER | _names(cfg) | set(cfg.get("name_only_words") or ())
    return [w for w in words if w not in skip]


def _phrase_lists(phrases: list[str], cfg: dict) -> list[list[str]]:
    out = []
    for p in phrases:
        words = _meaningful(_words(p), cfg) or _words(p)
        if words:
            out.append(words)
    return out


def control(text: str, cfg: dict) -> tuple[str | None, str]:
    """A spoken control in a fragment that continues a turn:

        ("send", rest)      ends with a send phrase; rest is what came before it
        ("cancel", "")      is, or ends with, a cancel phrase
        ("restart", rest)   starts with a cancel phrase, followed by rest
        (None, text)        none of these
    """
    clauses = _clauses(text)
    if not clauses:
        return None, text
    sends = _phrase_lists(cfg["turn_send_phrases"], cfg)
    cancels = _phrase_lists(cfg["turn_cancel_phrases"], cfg)
    # The last clause that says something ("That's all, thank you." -> "That's all").
    k = len(clauses) - 1
    while k > 0 and not _meaningful(_words(clauses[k].group(0)), cfg):
        k -= 1
    said = _meaningful(_words(clauses[k].group(0)), cfg)
    head = text[:clauses[k].start()].rstrip()
    head_says = bool(_meaningful(_words(head), cfg))
    if said in cancels:
        return "cancel", ""
    if said in sends:
        return "send", head if head_says else ""
    first = _meaningful(_words(clauses[0].group(0)), cfg)
    if len(clauses) > 1 and first in cancels:
        rest = text[clauses[1].start():].strip()
        if _meaningful(_words(rest), cfg):
            return "restart", rest[:1].upper() + rest[1:]
    return None, text


def ends_with_send_phrase(text: str, cfg: dict) -> bool:
    kind, _ = control(text, cfg)
    return kind == "send"


def opens_with_name(text: str) -> bool:
    """The name alone, with a mark that more is coming: "Claude," "Claude..." or "Claude, um"."""
    t = (text or "").rstrip()
    if t.endswith((",", "...", "…", "-", "–", "—", ":")):
        return True
    words = _words(t)
    return len(words) > 1 and words[-1] in {"um", "uh", "er", "erm", "uhm", "umm", "hmm"}


# ---------------------------------------------------------------------------
# The pending turn
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    text: str
    speech_start: float
    speech_end: float
    speech_seconds: float
    duration: float
    audio: list = field(default_factory=list)   # in-memory samples of its parts (for transcribing afresh)
    words: list | None = None                   # the transcriber's words, with confidences (annotate.HeardWord)
    speaker: str = ""                           # who said it, when not the user (the voice check's mark)


@dataclass
class Turn:
    how: str                                   # how its first fragment was accepted: named, armed, ...
    opened: float                              # clock time
    speech_end: float                          # end of its last speech
    fragments: list[Fragment] = field(default_factory=list)
    text: str = ""                             # the turn so far, as judged
    verdict: str = "incomplete"                # complete | incomplete | unclear | judging
    reason: str = ""
    judging: object = None                     # the judge's handle while it is asked
    judge_started: float = 0.0
    cued: bool = False
    rejoin: bool = True                        # its audio may be transcribed afresh as a whole
    heard: list | None = None                  # words of the turn transcribed afresh, each with its fragment
    interrupted: object = None                 # the voice was cut off by one of its fragments (bargein.Interrupted)

    @property
    def speech_seconds(self) -> float:
        return sum(f.speech_seconds for f in self.fragments)


@dataclass
class Outcome:
    action: str                # "wait", "send", "cancel", "cue", "drop"
    text: str = ""
    turn: Turn | None = None
    why: str = ""


_REJOIN_MAX_SECONDS = 60.0     # transcribing 60 s afresh takes about 0.4 s on an M4 Max


class TurnTaker:
    """Holds the pending turn and decides when it is sent.

    judge: callable(text) -> handle with .done, .kind ("complete" |
    "incomplete" | None), .why, .seconds; None means rules and timers only.
    rejoin: callable(list of audio arrays) -> text, or None."""

    def __init__(self, cfg: dict, *, judge=None, rejoin=None, clock=time.monotonic, log=lambda m: None,
                 console=lambda m: None):
        self.cfg = cfg
        self.judge = judge
        self.rejoin = rejoin
        self.clock = clock
        self.log = log
        self.console = console
        self.turn: Turn | None = None

    @property
    def pending(self) -> bool:
        return self.turn is not None

    # -- input --------------------------------------------------------------

    def open(self, how: str, speech_end: float) -> None:
        """The name alone with a mark of continuation: an empty turn, so what follows needs no name."""
        if self.turn is None:
            self.turn = Turn(how=how, opened=self.clock(), speech_end=speech_end, verdict="incomplete",
                             reason="the name, then a pause")
            self.log("turn: opened by the name alone; waiting for the request")

    def add(self, text: str, *, how: str, speech_start: float, speech_end: float, speech_seconds: float = 0.0,
            duration: float = 0.0, audio=None, words=None, interrupted=None, speaker: str = "") -> Outcome:
        """words: the transcriber's words for this fragment (for the marks, annotate.py);
        interrupted: where this fragment cut off the voice, if it did."""
        frag = Fragment(" ".join((text or "").split()), speech_start, speech_end, speech_seconds, duration,
                        [a for a in (audio or []) if a is not None], words, speaker)
        t = self.turn
        continuing = t is not None and bool(t.fragments)
        if t is not None and not t.fragments and control(frag.text, self.cfg)[0] == "cancel":
            return self._cancel("cancelled right after the name")
        if continuing:
            kind, rest = control(frag.text, self.cfg)
            if kind == "cancel":
                return self._cancel(f"{len(t.fragments)} fragment(s) discarded by a cancel phrase")
            if kind == "send":
                if rest:
                    t.fragments.append(Fragment(rest, speech_start, speech_end, speech_seconds, duration,
                                                words=words, speaker=speaker))
                    t.rejoin = False
                    t.text = self._joined(t)
                t.speech_end = speech_end
                t.interrupted = t.interrupted or interrupted
                return self._send("a send phrase")
            if kind == "restart":
                self.log(f"turn: {len(t.fragments)} fragment(s) discarded; starting afresh")
                self.turn = t = Turn(how=t.how, opened=self.clock(), speech_end=speech_end, rejoin=False)
                frag = Fragment(rest, speech_start, speech_end, speech_seconds, duration, words=words,
                                speaker=speaker)
        if t is None:
            self.turn = t = Turn(how=how, opened=self.clock(), speech_end=speech_end)
        t.interrupted = t.interrupted or interrupted
        t.fragments.append(frag)
        t.speech_end = speech_end
        t.text = self._joined(t)
        t.judging = None
        t.cued = False
        if not continuing and control(t.text, self.cfg)[0] == "send":
            t.verdict, t.reason = "complete", "ends with a send phrase"
        else:
            t.verdict, t.reason = judge_rules(t.text, self.cfg)
        if t.verdict == "unclear" and self.judge is not None:
            try:
                t.judging = self.judge(t.text)
                t.judge_started = self.clock()
                t.verdict = "judging"
            except Exception as exc:  # never let the judge stop the listener
                t.reason = f"{t.reason}; judge failed ({exc!r})"
        self.log(f"turn: fragment {len(t.fragments)} -> {t.verdict} ({t.reason})")
        return Outcome("wait", t.text, t, t.reason)

    def _joined(self, t: Turn) -> str:
        t.heard = None
        texts = " ".join(f.text for f in t.fragments if f.text)
        if not (self.rejoin and t.rejoin and len(t.fragments) >= 2 and self.cfg.get("turn_retranscribe")):
            return texts
        parts = [a for f in t.fragments for a in f.audio]
        if not all(f.audio for f in t.fragments) or sum(len(a) for a in parts) > _REJOIN_MAX_SECONDS * 16000:
            return texts
        try:
            # The rejoiner returns the text, or (text, words) with each word's piece of audio (.part).
            result = self.rejoin(parts)
            fresh, words = result if isinstance(result, tuple) else (result, None)
            fresh = " ".join((fresh or "").split())
        except Exception as exc:
            self.log(f"turn: transcribing afresh failed ({exc!r}); joining the pieces")
            return texts
        if len(_words(fresh)) < max(1, len(_words(texts)) // 2):   # lost most of it: keep the pieces
            return texts
        if words is not None:
            frag_of_part = [i for i, f in enumerate(t.fragments) for _ in f.audio]
            for w in words:
                w.frag = frag_of_part[min(max(getattr(w, "part", 0), 0), len(frag_of_part) - 1)]
            t.heard = list(words)
        m = addressing.match(fresh, self.cfg)
        return m.text if m.addressed else fresh

    def heard_words(self, t: Turn) -> list:
        """The transcriber's words for the turn's text, each with its fragment (annotate.HeardWord):
        those of the turn transcribed afresh, or else the fragments' own (bare words where a
        fragment came without them)."""
        if t.heard is not None:
            return t.heard
        from .annotate import HeardWord

        out = []
        for i, f in enumerate(t.fragments):
            words = f.words if f.words else [HeardWord(w) for w in f.text.split()]
            for w in words:
                out.append(HeardWord(w.text, w.conf, w.start, w.end, getattr(w, "part", 0), i))
        return out

    # -- time ---------------------------------------------------------------

    def tick(self, in_speech: bool) -> Outcome | None:
        t = self.turn
        if t is None:
            return None
        now = self.clock()
        if t.verdict == "judging":
            h = t.judging
            if getattr(h, "done", False):
                kind = getattr(h, "kind", None)
                seconds = getattr(h, "seconds", now - t.judge_started)
                if kind in ("complete", "incomplete"):
                    t.verdict, t.reason = kind, f"judged by the front desk in {seconds:.2f}s"
                else:
                    t.verdict, t.reason = "unclear", f"{t.reason}; no judgement ({getattr(h, 'why', '?')})"
                t.judging = None
                self.log(f"turn: {t.verdict} ({t.reason})")
            elif now - t.judge_started > self.cfg["turn_judge_timeout"] + 0.5:
                t.verdict, t.reason = "unclear", f"{t.reason}; the judge was too slow"
                t.judging = None
                self.log(f"turn: {t.verdict} ({t.reason})")
        silence = now - t.speech_end
        wait = {"complete": self.cfg["turn_end_silence"], "unclear": self.cfg["turn_unclear_wait"],
                "incomplete": self.cfg["turn_max_wait"]}.get(t.verdict, self.cfg["turn_max_wait"])
        if in_speech:
            # Speech has resumed: it will join the turn. Only if it runs on long
            # after the turn was due is the turn sent without it.
            if t.verdict != "judging" and silence > wait + self.cfg["max_utterance"] + self.cfg["end_silence"]:
                return self._due(t, "the wait ran out while speech went on")
            return None
        if t.verdict != "judging" and silence >= wait:
            why = {"complete": "complete", "unclear": f"unclear after {wait:g}s of silence",
                   "incomplete": f"{wait:g}s of silence"}[t.verdict]
            return self._due(t, why)
        if (self.cfg["turn_cue"] != "none" and not t.cued and t.verdict in ("incomplete", "unclear", "judging")
                and silence >= self.cfg["turn_cue_after"]):
            t.cued = True
            return Outcome("cue", turn=t)
        return None

    def _due(self, t: Turn, why: str) -> Outcome:
        if not t.fragments:
            self.turn = None
            self.log("turn: nothing followed the name; dropped")
            return Outcome("drop", turn=t, why="nothing followed the name")
        return self._send(why)

    # -- endings ------------------------------------------------------------

    def _send(self, why: str) -> Outcome:
        t, self.turn = self.turn, None
        if not t.fragments:
            return Outcome("drop", turn=t, why="nothing to send")
        span = t.speech_end - t.fragments[0].speech_start
        self.log(f"turn: sent ({why}); {len(t.fragments)} fragment(s) over {span:.1f}s")
        return Outcome("send", t.text, t, why)

    def _cancel(self, why: str) -> Outcome:
        t, self.turn = self.turn, None
        self.log(f"turn: cancelled ({why})")
        return Outcome("cancel", turn=t, why=why)

    def discard(self, why: str) -> Turn | None:
        t, self.turn = self.turn, None
        if t is not None:
            self.log(f"turn: discarded ({why}); {len(t.fragments)} fragment(s)")
        return t

    def flush(self, why: str) -> Outcome | None:
        """Send what there is now (end of input)."""
        if self.turn is None:
            return None
        return self._due(self.turn, why)

    def describe(self) -> dict | None:
        """For listener.json: no words."""
        t = self.turn
        if t is None:
            return None
        return {"fragments": len(t.fragments), "verdict": t.verdict,
                "silence": round(max(0.0, self.clock() - t.speech_end), 1)}
