"""What the listener types: the request with its label and marks. Standard library only.

build() takes the text of an addressed request (a turn, or with turn_taking
"silence" one utterance) as the listener is about to type it, and returns it
labelled and marked (format: hal/spoken.py; LISTENER.md, "What the model is
told"):

    [spoken 21:14] Claude, ask Clew-boss [heard as "Clue Boss"] to add a lamprey [?] there.

It is given, besides the text:

    heard       the words the transcriber produced for it, each with its
                confidence (Parakeet's, 0 to 1, or None) and the fragment of
                the turn it belongs to. They are lined up with the text, which
                may differ a little (the name rewritten as "Claude", a send
                phrase dropped).
    fragments   (speech_start, speech_end) of each fragment of the turn, for
                pauses
    when        time.time() at which the speech began
    interrupted the sentence the voice was saying when this speech cut it off

and, in turn, with the settings allowing it:

    1. corrections from the lexicon (lexicon.py): aliases, names that sound
       alike, words run together; each marked [heard as "..."], or only
       flagged [? Name] when less certain
    2. the name heard as an ordinary word (a lexicon alias such as "God",
       rewritten by addressing.py): marked [heard as "God"]
    3. [?] after the words the transcriber was least sure of: below
       doubt_threshold, at most two in a sentence, never a short function word
       or a known name
    4. [pause 12s] where a pause of at least pause_mark_seconds separated two
       fragments
    5. the label, with the interrupted sentence if there was one

Nothing here logs or stores anything; it only ever sees addressed speech.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from hal import spoken

from . import lexicon as lexmod

_TOKEN = re.compile(r"^([\"'“‘(]*)(.*?)([.,;:!?…\"'”’)]*)$")
_SENTENCE_END = (".", "?", "!", "…")
_MAX_DOUBTS_PER_SENTENCE = 2
# A single unfamiliar word that sounds like one of the user's names is changed only when the
# transcriber was less sure of it than this; otherwise it is flagged. ("Kolodzen", 0.76 on
# synthetic speech, is changed; "Bob Marley", heard with certainty, is only flagged.)
_SURE = 0.9
# Never marked doubtful: too short to matter, and the transcriber's least certain words.
_FUNCTION_WORDS = set("""
a an the to of and or but in on at for by with from as is it its it's be am are was were been this that these those
i me my you your he him his she her we us our they them their so if not no do does did has have had will would can
could shall should may might must just than then there here what which who whom whose how when where why also
""".split())


@dataclass
class HeardWord:
    """One word as the transcriber produced it."""
    text: str
    conf: float | None = None          # 0..1 (lowest of its tokens), None if unknown
    start: float = 0.0                 # seconds from the start of its audio
    end: float = 0.0
    part: int = 0                      # which piece of audio it came from (a turn transcribed afresh)
    frag: int = 0                      # which fragment of the turn it belongs to


@dataclass
class Annotated:
    text: str                                       # what is typed
    plain: str                                      # the words alone (after corrections)
    corrections: list[tuple[str, str]] = field(default_factory=list)   # (heard, typed)
    flags: list[tuple[str, str]] = field(default_factory=list)         # (word, suggestion)
    doubts: list[str] = field(default_factory=list)
    pauses: list[float] = field(default_factory=list)
    interrupted: bool = False

    def summary(self) -> str:
        """For the log: what was marked (the words are addressed speech, logged as such)."""
        parts = []
        if self.corrections:
            parts.append("corrected " + ", ".join(f"{a!r}->{b!r}" for a, b in self.corrections))
        if self.flags:
            parts.append("flagged " + ", ".join(f"{a!r}?{b!r}" for a, b in self.flags))
        if self.doubts:
            parts.append("doubtful " + ", ".join(repr(d) for d in self.doubts))
        if self.pauses:
            parts.append("pauses " + ", ".join(f"{p:.0f}s" for p in self.pauses))
        if self.interrupted:
            parts.append("interrupted the voice")
        return "; ".join(parts) or "no marks"


@dataclass
class _Tok:
    lead: str
    core: str
    trail: str
    conf: float | None = None
    frag: int | None = None
    was: str | None = None              # the transcriber's word, where the text differs (1 to 1)
    replace: str | None = None          # a correction, on the first token of its span
    span_end: int = -1                  # last token of that span
    gone: bool = False                  # inside a corrected span, after its first token
    marks: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)   # after the trailing punctuation (pauses)
    touched: bool = False               # corrected or flagged: no [?]

    @property
    def norm(self) -> str:
        return lexmod.norm_word(self.core)


def _tokens(text: str) -> list[_Tok]:
    out = []
    for raw in text.split():
        m = _TOKEN.match(raw)
        lead, core, trail = (m.group(1), m.group(2), m.group(3)) if m else ("", raw, "")
        if not core:                      # punctuation alone
            core, trail = trail, ""
        out.append(_Tok(lead, core, trail))
    return out


def _align(toks: list[_Tok], heard: list[HeardWord]) -> None:
    """Give each token the confidence and fragment of the word it was heard as."""
    if not heard:
        return
    a = [t.norm for t in toks]
    b = [lexmod.norm_word(h.text) for h in heard]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                h = heard[j1 + k]
                toks[i1 + k].conf, toks[i1 + k].frag = h.conf, h.frag
                if tag == "replace":
                    toks[i1 + k].was = h.text
        elif tag == "replace":
            confs = [h.conf for h in heard[j1:j2] if h.conf is not None]
            for k in range(i1, i2):
                toks[k].conf = min(confs) if confs else None
                toks[k].frag = heard[j1].frag
        elif tag == "delete":                  # words in the text the transcriber didn't produce
            for k in range(i1, i2):
                toks[k].frag = heard[j1 - 1].frag if j1 > 0 else heard[0].frag


def _sentence_starts(toks: list[_Tok]) -> list[bool]:
    starts, new = [], True
    for t in toks:
        starts.append(new)
        initial = t.trail == "." and len(t.core) == 1 and t.core.isalpha()      # "J. Markdown"
        new = not initial and (t.trail.endswith(_SENTENCE_END)
                               or t.trail.rstrip("\"'”’)").endswith(_SENTENCE_END))
    return starts


def _heard_text(toks: list[_Tok], i: int, j: int) -> str:
    return " ".join(t.lead + t.core + t.trail for t in toks[i:j + 1]).strip(" ,;:")


def _correct(toks: list[_Tok], i: int, j: int, target: str, out: Annotated) -> None:
    orig = " ".join([t.core + t.trail for t in toks[i:j]] + [toks[j].core])     # "J. Markdown"
    possessive = ""
    last = toks[j].core
    if last.lower().endswith(("'s", "’s")) and len(last) > 3 and not target.lower().endswith(("'s", "’s")):
        possessive = last[-2:]
    first = toks[i]
    first.replace, first.span_end = target + possessive, j
    first.marks.append(f"[heard as {spoken.quote(orig, 60)}]")
    first.trail = toks[j].trail
    for t in toks[i:j + 1]:
        t.touched = True
    for t in toks[i + 1:j + 1]:
        t.gone = True
    out.corrections.append((orig, target + possessive))


def _flag(toks: list[_Tok], i: int, j: int, names: list[str], out: Annotated) -> None:
    toks[j].marks.append("[? " + " or ".join(dict.fromkeys(names)) + "]")
    for t in toks[i:j + 1]:
        t.touched = True
    out.flags.append((" ".join(t.core for t in toks[i:j + 1]), " or ".join(dict.fromkeys(names))))


def _free(toks: list[_Tok], i: int, j: int, starts: list[bool]) -> bool:
    """May tokens i..j form one name? Untouched, within one clause."""
    if j >= len(toks):
        return False
    for k in range(i, j + 1):
        t = toks[k]
        if t.touched or t.gone or not t.norm:
            return False
        if k > i and starts[k]:
            return False
        if k < j and t.trail and not (t.trail == "." and len(t.core) == 1):   # "J. Markdown"
            return False
        if k > i and t.lead:
            return False
    return True


def _apply_aliases(toks, lex, starts, out) -> None:
    aliases = sorted((a for a in lex.aliases if a.mode in ("always", "flag")), key=lambda a: -len(a.heard))
    for i in range(len(toks)):
        for a in aliases:
            j = i + len(a.heard) - 1
            if not _free(toks, i, j, starts):
                continue
            words = [t.norm for t in toks[i:j + 1]]
            if words[-1].endswith("'s") and words[-1][:-2] == a.heard[-1]:
                words[-1] = words[-1][:-2]
            if tuple(words) != a.heard:
                continue
            if a.mode == "flag":
                _flag(toks, i, j, [a.target], out)
            else:
                _correct(toks, i, j, a.target, out)
            break


def _unpossessed(word: str) -> str:
    return word[:-2] if word.lower().endswith(("'s", "’s")) and len(word) > 3 else word


def _kind(t: _Tok, initial: bool, lex, dic) -> str:
    core = _unpossessed(t.core)
    letters = lexmod._letters(core)
    if not letters or letters.isdigit():
        return "ordinary"
    if lex.known(core):
        return "lexicon"
    if len(letters) == 1 and core not in ("a", "A", "I"):
        return "name"                      # "J.": a letter
    if dic.ordinary(core):
        if core[:1].isupper() and not initial and core != "I":
            return "known-name"            # an ordinary word the transcriber wrote as a name
        return "ordinary"
    if core.isupper() and 2 <= len(letters) <= 5:
        return "name"                      # "MP", "WASM": an acronym
    if dic.known(core):
        return "known-name"                # Monday, London, Isabel
    return "unknown"


def _apply_lexicon(toks, lex, starts, out, threshold: float | None) -> None:
    dic = lex.dictionary
    own = [e for e in lex.entries if e.source == "file"]
    folders = [e for e in lex.entries if e.source != "file" and not e.ordinary]
    kinds = [_kind(t, starts[k], lex, dic) for k, t in enumerate(toks)]
    # Run together: "ghost script" -> ghostscript.
    for length in (3, 2):
        for i in range(len(toks) - length + 1):
            j = i + length - 1
            if not _free(toks, i, j, starts):
                continue
            e = lex.by_letters("".join(lexmod._letters(t.core) for t in toks[i:j + 1]))
            if e is not None and not e.ordinary and e.name.lower() != "claude":
                _correct(toks, i, j, e.name, out)
    # Sounds alike: every span of one to three words that could be a name, then the
    # best of them applied first (corrections before flags, closer before further).
    found = []
    for length in (3, 2, 1):
        for i in range(len(toks) - length + 1):
            j = i + length - 1
            if not _free(toks, i, j, starts):
                continue
            span = kinds[i:j + 1]
            doubtful = threshold is not None and any(t.conf is not None and t.conf < threshold
                                                     for t in toks[i:j + 1])
            if all(k == "lexicon" for k in span) or (length == 1 and span[0] == "lexicon"):
                continue                       # already one of the user's names
            if all(k == "ordinary" for k in span) and not (length == 1 and doubtful):
                continue
            key = lexmod.phonetic(" ".join(_unpossessed(t.core) for t in toks[i:j + 1]))
            n = lexmod.consonants(key)
            if n < 2:
                continue
            # A folder name is only considered where the transcriber was unsure (below).
            best = sorted(lex.closest(key, entries=own) + (lex.closest(key, entries=folders) if doubtful else []),
                          key=lambda x: (x[0], x[1].source != "file", x[1].name))[:4]
            if not best:
                continue
            d, e = best[0]
            ties = list(dict.fromkeys(x.name for dd, x in best if dd == d))
            unique = len(ties) == 1
            exact = d <= 0.35
            high = d <= 0.65 and n >= 3
            mid = d <= max(1.0, 0.2 * n) and n >= 3
            from_file = e.source == "file"
            unknown = "unknown" in span
            odd = sum(1 for k in span if k != "ordinary")
            # Several words that together look like a name: written as one, or mostly letters and odd words.
            named = length >= 2 and odd >= 1 and (
                odd >= 2 or all(k != "ordinary" or toks[i + m].core[:1].isupper() for m, k in enumerate(span)))
            action = None
            if not from_file:
                # A folder name (one of some 1,400 projects and authors): only for an unfamiliar
                # word or a mangled name the transcriber itself was unsure of. An unfamiliar name
                # heard with confidence is usually right, and an ordinary word is an ordinary word.
                if doubtful and (unknown or named) and n >= 3:
                    if d == 0 and unique:
                        action = "correct"
                    elif high:
                        action = "flag"
            elif n < 3:
                if unknown and length == 1 and d == 0 and unique:
                    action = "correct"
            elif named and unique and (exact or high):
                action = "correct"             # "Clue Boss", "MP ticks WASM": several words, mangled
            elif unknown and unique and (exact or high) and length == 1 and toks[i].conf is not None \
                    and toks[i].conf < _SURE:
                action = "correct"             # "Kolodzen": unfamiliar, and the transcriber unsure
            elif unknown and mid:
                action = "flag"                # "Marley": sounds like Marlow, but heard with confidence
            elif length == 1 and span[0] == "known-name" and exact:
                action = "flag"                # "Isabel": a known name, but one of the user's sounds the same
            elif length == 1 and doubtful and exact and len(lexmod._letters(toks[i].core)) >= 6:
                action = "flag"                # unsure of a longer word that sounds just like one of the user's
            if action:
                found.append((action != "correct", d, -length, i, j, action, e.name, ties[:2]))
    for _, _, _, i, j, action, name, ties in sorted(found):
        if not _free(toks, i, j, starts):
            continue
        if action == "correct":
            _correct(toks, i, j, name, out)
        else:
            _flag(toks, i, j, ties, out)


def _name_marks(toks, cfg, out) -> None:
    """The name heard as an ordinary word through a lexicon alias ("God, ..."): show what was heard."""
    shown = set(cfg.get("calling_names") or ()) | set(cfg.get("alias_names") or ())
    for t in toks:
        if t.was and t.norm == "claude" and lexmod.norm_word(t.was) in shown:
            was = _TOKEN.match(t.was).group(2) or t.was
            t.marks.append(f"[heard as {spoken.quote(was, 40)}]")
            t.touched = True
            out.corrections.append((was, "Claude"))


def _doubts(toks, lex, starts, threshold, out) -> None:
    sentence: list[int] = []

    def close():
        picks = sorted(sentence, key=lambda k: toks[k].conf)[:_MAX_DOUBTS_PER_SENTENCE]
        for k in sorted(picks):
            toks[k].marks.append("[?]")
            out.doubts.append(toks[k].core)
        sentence.clear()

    for k, t in enumerate(toks):
        if starts[k] and sentence:
            close()
        if (t.conf is not None and t.conf < threshold and not t.touched and not t.gone
                and t.norm and t.norm not in _FUNCTION_WORDS and t.norm != "claude"
                and not (lex is not None and lex.known(t.core)) and not t.core.isdigit()):
            sentence.append(k)
    if sentence:
        close()


def _pauses(toks, fragments, seconds, out) -> None:
    if not fragments or len(fragments) < 2 or not any(t.frag is not None for t in toks):
        return
    for f in range(len(fragments) - 1):
        gap = fragments[f + 1][0] - fragments[f][1]
        if gap < seconds:
            continue
        # After the last token of fragment f, if a later fragment's token follows it.
        last = None
        for k, t in enumerate(toks):
            if t.frag is not None and t.frag <= f and not t.gone:
                last = k
        if last is None or not any(t.frag is not None and t.frag > f for t in toks[last + 1:]):
            continue
        toks[last].after.append(f"[pause {round(gap)}s]")
        out.pauses.append(gap)


def _render(toks: list[_Tok]) -> tuple[str, str]:
    typed, plain = [], []
    for t in toks:
        if t.gone:
            continue
        word = t.replace if t.replace is not None else t.core
        marks = (" " + " ".join(t.marks)) if t.marks else ""
        typed.append(f"{t.lead}{word}{marks}{t.trail}" + ("".join(" " + a for a in t.after)))
        plain.append(f"{t.lead}{word}{t.trail}")
    return " ".join(typed), " ".join(plain)


def build(text: str, cfg: dict, *, heard: list[HeardWord] | None = None,
          fragments: list[tuple[float, float]] | None = None, when: float | None = None,
          interrupted: str = "", lex=None, speaker: str = "", verified: bool = False) -> Annotated:
    """The request as typed (see the module docstring). With spoken_label off: the text as it is."""
    text = " ".join(spoken.sanitize(text).split())
    if not cfg.get("spoken_label", True):
        return Annotated(text, text)
    out = Annotated("", "")
    toks = _tokens(text)
    _align(toks, heard or [])
    starts = _sentence_starts(toks)
    threshold = float(cfg.get("doubt_threshold", 0.82)) if cfg.get("doubt_marks", True) else None
    if lex is not None and cfg.get("lexicon", True):
        _apply_aliases(toks, lex, starts, out)
        _apply_lexicon(toks, lex, starts, out, threshold)
        _name_marks(toks, cfg, out)
    if threshold is not None:
        _doubts(toks, lex, starts, threshold, out)
    if cfg.get("pause_marks", True):
        _pauses(toks, fragments, float(cfg.get("pause_mark_seconds", 3.0)), out)
    body, plain = _render(toks)
    note = interrupted if cfg.get("interruption_note", True) else ""
    out.interrupted = bool(note)
    out.text = f"{spoken.label(when, note, speaker, verified)} {body}".strip()
    out.plain = plain
    return out
