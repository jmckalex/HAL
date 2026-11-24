"""Is a transcript addressed to Claude? Pure text rules, standard library only.

Speech-to-text writes "Claude" correctly most of the time, but not always:
on synthetic voices Parakeet also produced "Cloat", "Clothes", "Cloth",
"Cloak" and "a cloud". Two tiers of variants keep false accepts down:

    names           match anywhere as whole words ("Claude", "Klaud", "Cloat"):
                    none of them is an ordinary English word
    vocative_names  ordinary words ("cloud", "clawed", "clothes") that count only
                    where a name is being called: followed by punctuation or the
                    end of the utterance, and either preceded by a comma or
                    preceded only by lead words ("hey", "good morning") since the
                    start of the sentence. "Cloud, what's the weather?",
                    "Thanks, cloud.", "By the way, cloud, ..." and "Hey cloud."
                    match; "it's in the cloud" and "cloud storage" don't.

The matched variant is rewritten to "Claude" in the text that is submitted.

Two more lists come from the lexicon's aliases for the name (lexicon.txt,
"god -> Claude @calling"), which the listener puts in the settings it passes
here at run time:

    alias_names     match anywhere, like names
    calling_names   ordinary words heard for the name ("God", "Todd", "Gaud"),
                    stricter still than vocative_names: in the calling position,
                    and not an exclamation. Not after "oh", "my", "thank",
                    "good", "dear", "for", "by"; not followed by "!"; and not
                    the whole utterance. "God, what's the time?" and "Good
                    morning, God." match; "Oh God.", "Thank God.", "My God!",
                    "God." and "Todd and I" don't. ("God, I'm tired." does:
                    the typed text then shows Claude [heard as "God"].)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A word (letters/digits, inner apostrophes) or one punctuation mark.
_TOKEN = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’][A-Za-z]+)*|[.,;:!?—–]|(?<!\w)-(?!\w)")
_SENTENCE_END = {".", "!", "?", ";", ":"}
# Politeness and filler ignored when recognising a spoken command.
_POLITE = {"please", "now", "thanks", "thank", "you", "for", "a", "while", "right", "just", "can", "could",
           "would", "will", "okay", "ok"}


@dataclass
class Word:
    text: str           # as written
    base: str           # lower case, possessive 's removed
    start: int          # span in the transcript
    end: int
    possessive: str     # "'s" or "" (kept when rewriting)
    before: str = ""    # punctuation since the previous word ("" if none; "^" if first and none)
    after: str = ""     # first punctuation after this word ("" if a word follows directly; "$" at the end)
    sentence_index: int = 0  # position within its sentence


@dataclass
class Match:
    addressed: bool
    name_only: bool = False
    command: str | None = None          # "stop_listening"
    text: str = ""                      # transcript with the matched variants written "Claude"
    variants: list[str] = field(default_factory=list)  # the words that matched, as written


def tokenize(text: str) -> list[Word]:
    words: list[Word] = []
    between = "^"
    sentence_index = 0
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if not tok[0].isalnum():
            if words and not words[-1].after:
                words[-1].after = tok
            between = tok
            if tok in _SENTENCE_END:
                sentence_index = 0
            continue
        lower = tok.lower().replace("’", "'")
        possessive = ""
        if lower.endswith("'s"):
            possessive, lower = tok[-2:], lower[:-2]
        words.append(Word(tok, lower, m.start(), m.end(), possessive, before=between,
                          sentence_index=sentence_index))
        sentence_index += 1
        between = ""
    if words and not words[-1].after:
        words[-1].after = "$"
    return words


# Words that make a following "God" an exclamation, not a name being called ("thank God");
# the first few even with a comma between ("Oh, God."), unlike "Thanks, God.".
_EXCLAIM_BEFORE = {"oh", "o", "my", "thank", "thanks", "good", "dear", "for", "by", "ye", "swear", "to", "honest",
                   "almighty", "of", "a", "the", "and"}
_EXCLAIM_EVEN_WITH_COMMA = {"oh", "o", "my", "dear", "good", "ah", "jesus"}


def is_calling(words: list[Word], i: int, lead_words: set[str]) -> bool:
    """A calling_names word in the calling position, not an exclamation (see the module docstring)."""
    w = words[i]
    if not is_vocative(words, i, lead_words) or w.after == "!":
        return False
    if i > 0 and w.sentence_index > 0:
        before = words[i - 1].base
        if before in _EXCLAIM_EVEN_WITH_COMMA or (before in _EXCLAIM_BEFORE and w.before != ","):
            return False
    return len(words) > 1


def is_vocative(words: list[Word], i: int, lead_words: set[str]) -> bool:
    w = words[i]
    if not w.after:          # another word follows directly: "cloud storage"
        return False
    if w.before == ",":      # "Thanks, cloud." / "By the way, cloud, ..."
        return True
    # Only lead words since the start of the sentence: "Cloud, ...", "Hey cloud."
    return all(words[k].base in lead_words for k in range(i - w.sentence_index, i))


def _meaningful(bases: list[str], cfg: dict) -> list[str]:
    skip = _POLITE | set(cfg["name_only_words"])
    return [b for b in bases if b not in skip]


def _command(bases: list[str], cfg: dict) -> str | None:
    said = _meaningful(bases, cfg)
    for phrase in cfg["stop_phrases"]:
        target = _meaningful([w.base for w in tokenize(phrase)], cfg)
        if target and said == target:
            return "stop_listening"
    return None


def match(text: str, cfg: dict) -> Match:
    """Decide whether text is addressed to Claude; see the module docstring."""
    text = (text or "").strip()
    words = tokenize(text)
    names = set(cfg["names"]) | set(cfg.get("alias_names") or ())
    vocative = set(cfg["vocative_names"])
    calling = set(cfg.get("calling_names") or ())
    lead = set(cfg["lead_words"])
    hits = [i for i, w in enumerate(words)
            if w.base in names or (w.base in vocative and is_vocative(words, i, lead))
            or (w.base in calling and is_calling(words, i, lead))]
    if not hits:
        return Match(False, text=text)

    # Rewrite the matched variants as "Claude", keeping a possessive.
    out, pos = [], 0
    for i in hits:
        w = words[i]
        out += [text[pos:w.start], "Claude" + w.possessive]
        pos = w.end
    out.append(text[pos:])

    rest = [w.base for i, w in enumerate(words) if i not in hits]
    name_only = all(b in set(cfg["name_only_words"]) for b in rest)
    return Match(True, name_only=name_only, command=_command(rest, cfg), text="".join(out),
                 variants=[words[i].text for i in hits])


def command_without_name(text: str, cfg: dict) -> str | None:
    """A command spoken without the name (right after the "Yes?" of name_only_reply)."""
    return _command([w.base for w in tokenize(text)], cfg)
