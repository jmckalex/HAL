"""The user's names and terms, for correcting transcripts. Standard library only.

Parakeet doesn't know the user's names: "Clew-boss" arrives as "Clue Boss", "TeX+"
as "Tech Plus", "Kolodzin" as "Culdran". The lexicon is built from:

    lexicon_file    lexicon.txt in the project, edited by hand: names and terms
                    (with how they are said, where that isn't obvious), and
                    aliases, observed mistakes mapped to what was meant
    lexicon_dirs    the names of the folders directly inside each of these
                    (projects; a paper library's author folders)

Everything is read locally and stays in memory; nothing is written. The file
is re-read when it changes, the folders every ten minutes.

How a transcript is corrected (annotate.py applies it, to addressed speech only):

    alias           "heard -> Meant" in lexicon.txt: always applied. With "?"
                    after it, the word is only flagged ("heard [? Meant]").
                    With "@calling", only where a name is being called; that is
                    for the assistant's own name and is applied by
                    addressing.py ("God, what's the time?").
    sounds alike    a word, or two or three, that sounds like an entry
                    (phonetic(): a rough English pronunciation key, compared
                    with a weighted edit distance). It is changed only when it
                    is not an ordinary word in context: not in the dictionary
                    (/usr/share/dict/words, lower-case entries and their
                    inflections), or written as a name by the transcriber
                    (capitals mid-sentence) and matching a name of two or more
                    parts. A close but less certain match is only flagged.
                    "cloud storage" and "hacking the server" are left alone.
    run together    two or three words that spell an entry when joined
                    ("ghost script" -> ghostscript).

Every change and flag is marked in the typed text, so a wrong one can be caught.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DICTIONARY = Path("/usr/share/dict/words")
PROPER_NAMES = Path("/usr/share/dict/propernames")
_RESCAN_SECONDS = 600.0
_MAX_SPAN = 3

# Words the 1934 dictionary lacks, so that they don't look like misheard names.
_MODERN = set("""
okay ok email emails online offline website websites internet laptop laptops podcast podcasts app apps software
smartphone iphone ipad macbook macos ios wifi bluetooth airpods google github gitlab python javascript typescript
json html css url urls pdf pdfs api apis ai llm llms chatbot chatgpt anthropic openai debug debugging debugger login
logout username password passwords inbox desktop toolbar dropdown checkbox screenshot screenshots download
downloads upload uploads uploaded downloaded workflow workflows repo repos backend frontend codebase dataset datasets
startup startups blog blogs vlog selfie emoji emojis hashtag tweet tweets texting zoom skype teams slack whatsapp
youtube facebook instagram linkedin twitter netflix amazon spotify kindle microsoft apple xcode swift rust golang
latex tex bibtex markdown pandoc homebrew npm node nodejs linux unix ubuntu docker kubernetes terraform aws cloud
sync syncing synced reboot rebooted unzip zip zipped plugin plugins hyperlink hyperlinks spreadsheet spreadsheets
powerpoint keynote dropbox onedrive icloud gmail outlook calendar ok'd okayed fyi etc vs ux ui gui cli sql
""".split())

_VOWELS = "aeiou"
# Pairs of key symbols that sound alike (half the cost of another substitution).
_SIMILAR = {frozenset(p) for p in ("BP", "T0", "S0", "F0", "SX", "JX", "JK", "FV", "VW", "MN", "SZ", "KX", "TD")}
_LETTER_NAMES = {"a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "jee", "h": "aitch",
                 "i": "eye", "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en", "o": "oh", "p": "pee",
                 "q": "cue", "r": "ar", "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "doubleyou",
                 "x": "ex", "y": "why", "z": "zed"}
_FOLD_EXTRA = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "đ": "d", "ł": "l", "Ł": "L",
                             "œ": "oe", "þ": "th", "ð": "th"})


# ---------------------------------------------------------------------------
# The dictionary: is a word an ordinary English word?
# ---------------------------------------------------------------------------

class Dictionary:
    """Lower-case words from /usr/share/dict/words (plus _MODERN), and the capitalised
    entries and first names, which are names but known ones ("Monday", "Arizona", "Isabel")."""

    def __init__(self, words: set[str] | None = None, names: set[str] | None = None):
        if words is None:
            words, names = self._load()
        self.words = set(words) | _MODERN
        self.names = set(names or ())

    @staticmethod
    def _load() -> tuple[set[str], set[str]]:
        words: set[str] = set()
        names: set[str] = set()
        for path, into_names in ((DICTIONARY, False), (PROPER_NAMES, True)):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        w = line.strip()
                        if not w:
                            continue
                        if into_names or not w.islower():
                            names.add(w.lower())
                        else:
                            words.add(w)
            except OSError:
                pass
        return words, names

    def _stem_in(self, w: str, pool: set[str]) -> bool:
        if w in pool:
            return True
        for suffix, repl in (("'s", ""), ("ies", "y"), ("ied", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
                             ("ing", ""), ("ing", "e"), ("ly", ""), ("ily", "y"), ("er", ""), ("er", "e"),
                             ("ers", ""), ("est", ""), ("ness", ""), ("ment", ""), ("ments", ""), ("ings", ""),
                             ("able", ""), ("able", "e"), ("ally", "al"), ("ism", ""), ("ist", ""), ("ists", "")):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                stem = w[: len(w) - len(suffix)] + repl
                if stem in pool:
                    return True
                if len(stem) >= 4 and stem[-1] == stem[-2] and stem[:-1] in pool:   # running -> run
                    return True
        return False

    def ordinary(self, word: str) -> bool:
        """An ordinary (lower-case) English word, or an inflection of one."""
        w = _fold(word).lower().strip("'’-.")
        return bool(w) and (self._stem_in(w, self.words) or w.isdigit())

    def known(self, word: str) -> bool:
        """Ordinary, or a name the dictionary knows (Monday, London, Isabel)."""
        w = _fold(word).lower().strip("'’-.")
        return self.ordinary(word) or self._stem_in(w, self.names)


_DICTIONARY: Dictionary | None = None


def dictionary() -> Dictionary:
    global _DICTIONARY
    if _DICTIONARY is None:
        _DICTIONARY = Dictionary()
    return _DICTIONARY


# ---------------------------------------------------------------------------
# Pronunciation keys
# ---------------------------------------------------------------------------

def _fold(text: str) -> str:
    """Accents off: "Gómez" -> "Gomez", "Gödel" -> "Godel"."""
    text = str(text).translate(_FOLD_EXTRA)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _parts(word: str) -> list[str]:
    """How a written form splits when said: "mp-tikz-wasm" -> mp, tikz, wasm; "BibDesk" ->
    Bib, Desk; "TeX+" -> TeX, plus; "Dynamo.js" -> Dynamo, js."""
    word = _fold(word).replace("+", " plus ").replace("&", " and ")
    out = []
    for chunk in re.split(r"[^A-Za-z0-9']+", word):
        chunk = chunk.strip("'")
        if not chunk:
            continue
        # camelCase and ACRONYMWord boundaries
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z']+|\d+", chunk):
            out.append(piece)
    return out


def _spelled(piece: str) -> bool:
    """Said letter by letter: a single letter, an all-capital acronym of up to three letters
    ("MP", "PDF", "API"), or a short run of consonants ("js", "mp", "btt"). Longer acronyms
    with a vowel are said as words ("WASM", "NASA")."""
    p = piece.lower()
    if not p.isalpha():
        return False
    if len(p) == 1:
        return True
    if piece.isupper() and len(p) <= 3:
        return True
    return len(p) <= 5 and not any(ch in "aeiouy" for ch in p)


def _word_key(w: str) -> str:
    """A rough pronunciation key for one lower-case word: consonant classes, "A" for each vowel sound."""
    w = re.sub(r"[^a-z0-9]", "", w)
    if not w:
        return ""
    if w.isdigit():
        return w
    for a, b in (("kn", "n"), ("gn", "n"), ("pn", "n"), ("wr", "r"), ("ps", "s"), ("pt", "t"), ("wh", "w"),
                 ("x", "s"), ("mc", "mak")):
        if w.startswith(a) and len(w) > len(a):
            w = b + w[len(a):]
            break
    if len(w) > 3 and w.endswith("e") and w[-2] not in _VOWELS and not w.endswith(("le", "re")):
        w = w[:-1]                                     # silent final e: "Anscombe", "Irene"
    out: list[str] = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""
        prev = w[i - 1] if i else ""
        if c.isdigit():
            out.append(c)
            i += 1
            continue
        if c in _VOWELS or (c == "y" and (not nxt or nxt not in _VOWELS) and i > 0):
            out.append("A")
            i += 1
            continue
        if c == prev:                                  # double consonant
            i += 1
            continue
        step, sym = 1, ""
        if c == "c":
            if nxt == "h":
                sym, step = "X", 2
            elif nxt == "k":
                sym, step = "K", 2
            elif nxt in "eiy" and nxt:
                sym = "S"
            else:
                sym = "K"
        elif c == "s":
            if nxt == "h":
                sym, step = "X", 2
            elif nxt == "c" and nxt2 == "h":
                sym, step = "SK", 3
            else:
                sym = "S"
        elif c == "t":
            if nxt == "h":
                sym, step = "0", 2
            elif nxt == "c" and nxt2 == "h":
                sym, step = "X", 3
            elif nxt == "i" and nxt2 in ("a", "o") and nxt2:
                sym = "X"                              # -tion, -tial
            else:
                sym = "T"
        elif c == "d":
            if nxt == "g" and nxt2 in ("e", "i", "y") and nxt2:
                sym, step = "J", 2
            else:
                sym = "T"
        elif c == "p":
            sym, step = ("F", 2) if nxt == "h" else ("P", 1)
        elif c == "g":
            if nxt == "h":
                sym, step = ("K", 2) if i == 0 else ("", 2)   # "ghost"; silent in "night", "Hugh"
            elif nxt == "n" and i + 2 == n:
                sym = ""                               # "sign"
            elif nxt in ("e", "i", "y") and nxt:
                sym = "J"
            else:
                sym = "K"
        elif c == "q":
            sym, step = ("KW", 2) if nxt == "u" else ("K", 1)
        elif c == "x":
            sym = "KS"
        elif c == "z":
            sym = "S"
        elif c == "w":
            sym = "W" if nxt and nxt in _VOWELS else "A"   # "clew", "ow": a vowel
        elif c == "y":
            sym = "Y"
        elif c == "h":
            sym = "H" if (i == 0 and nxt in _VOWELS and nxt) else ""
        elif c == "n" and nxt == "g" and i + 2 == n:
            sym, step = "N", 2                         # "robbing" ~ "Robin"
        elif c == "m" and nxt == "b" and i + 2 == n:
            sym, step = "M", 2                         # "Anscombe"
        elif c == "k":
            sym = "K"
        elif c == "j":
            sym = "J"
        elif c == "v":
            sym = "V"
        else:
            sym = c.upper()                            # b f l m n r
        out.append(sym)
        i += step
    key = "".join(out)
    key = re.sub(r"A+", "A", key)
    return re.sub(r"([^A])\1+", r"\1", key)


def phonetic(text: str) -> str:
    """The pronunciation key of a word or phrase as written (letters of acronyms spelled out)."""
    keys = []
    for piece in _parts(text):
        if _spelled(piece):
            keys.extend(_word_key(_LETTER_NAMES.get(ch, ch)) for ch in piece.lower())
        else:
            keys.append(_word_key(piece.lower()))
    return re.sub(r"([^A])\1+", r"\1", "".join(keys))


def distance(a: str, b: str, limit: float = 99.0) -> float:
    """Weighted edit distance between two keys: a vowel sound costs 0.3 to add or drop,
    sounds alike 0.5 to swap, anything else 1."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = [0.0] * (lb + 1)
    for j in range(1, lb + 1):
        prev[j] = prev[j - 1] + (0.3 if b[j - 1] == "A" else 1.0)
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [prev[0] + (0.3 if ca == "A" else 1.0)] + [0.0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            if ca == cb:
                sub = 0.0
            elif "A" in (ca, cb):
                sub = 1.0
            elif frozenset((ca, cb)) in _SIMILAR:
                sub = 0.5
            else:
                sub = 1.0
            cur[j] = min(prev[j - 1] + sub, prev[j] + (0.3 if ca == "A" else 1.0),
                         cur[j - 1] + (0.3 if cb == "A" else 1.0))
            best = min(best, cur[j])
        if best > limit:
            return best
        prev = cur
    return prev[lb]


def consonants(key: str) -> int:
    return sum(1 for ch in key if ch != "A")


# ---------------------------------------------------------------------------
# The lexicon
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    name: str                                  # as written: "Clew-boss", "TeX+"
    source: str                                # "file", "folder"
    keys: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)   # consonants in each key
    letters: str = ""                          # lower-case letters and digits: "clewboss"
    ordinary: bool = False                     # the name is ordinary words as written ("Pigeonhole", "line break")

    @property
    def parts(self) -> int:
        return len(_parts(self.name))


@dataclass
class Alias:
    heard: tuple[str, ...]                     # normalised words: ("clue", "boss")
    target: str                                # "Clew-boss"
    mode: str = "always"                       # "always", "flag" or "calling"


def norm_word(word: str) -> str:
    """A word as aliases compare it: lower case, accents off, letters, digits and apostrophes."""
    w = _fold(word).lower().replace("’", "'")
    return re.sub(r"[^a-z0-9']", "", w).strip("'")


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _fold(text).lower())


class Lexicon:
    """Entries and aliases, from lexicon_file and the folders in lexicon_dirs (see the module docstring)."""

    def __init__(self, file: str | Path | None = None, dirs=(), *, dictionary_: Dictionary | None = None,
                 clock=time.monotonic, log=lambda m: None):
        self.file = Path(file) if file else None
        self.dirs = [Path(os.path.expanduser(str(d))) for d in dirs or ()]
        self.dictionary = dictionary_ if dictionary_ is not None else dictionary()
        self.clock = clock
        self.log = log
        self.entries: list[Entry] = []
        self.aliases: list[Alias] = []
        self._by_letters: dict[str, Entry] = {}
        self._known: set[str] = set()
        self._file_stamp = None
        self._file_entries: list[Entry] = []
        self._folder_entries: list[Entry] = []
        self._scanned_at = -1e18
        self.version = 0
        self.refresh(force=True)

    # -- loading --------------------------------------------------------------

    def refresh(self, force: bool = False) -> bool:
        """Re-read the file if it changed, rescan the folders every ten minutes. True if anything changed."""
        changed = False
        stamp = None
        if self.file is not None:
            try:
                st = self.file.stat()
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                stamp = None
        if force or stamp != self._file_stamp:
            self._file_stamp = stamp
            self._file_entries, self.aliases = self._read_file()
            changed = True
        if force or self.clock() - self._scanned_at > _RESCAN_SECONDS:
            self._scanned_at = self.clock()
            folders = self._scan()
            if [e.name for e in folders] != [e.name for e in self._folder_entries]:
                self._folder_entries = folders
                changed = True
        if changed:
            self._index()
            self.version += 1
        return changed

    def _entry(self, name: str, source: str, spoken: list[str] | None = None) -> Entry:
        forms = [name] + list(spoken or [])
        keys = list(dict.fromkeys(k for k in (phonetic(f) for f in forms) if k))
        letters = _letters(name)
        # Ordinary: words as written ("Pigeonhole", "line break"); "Clew-boss" and "TeX+" are names.
        ordinary = all(self.dictionary.ordinary(w) and w.replace("'", "").isalpha() for w in name.split())
        return Entry(name, source, keys, [consonants(k) for k in keys], letters, ordinary)

    def _read_file(self) -> tuple[list[Entry], list[Alias]]:
        entries: list[Entry] = []
        aliases: list[Alias] = []
        if self.file is None:
            return entries, aliases
        try:
            text = self.file.read_text(encoding="utf-8")
        except OSError:
            return entries, aliases
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" in line:
                heard, target = (x.strip() for x in line.split("->", 1))
                mode = "always"
                if target.endswith("@calling"):
                    mode, target = "calling", target[: -len("@calling")].strip()
                elif target.endswith("?"):
                    mode, target = "flag", target[:-1].strip()
                words = tuple(w for w in (norm_word(x) for x in heard.split()) if w)
                if words and target:
                    aliases.append(Alias(words, target, mode))
                    if mode != "calling" and not any(e.name == target for e in entries):
                        entries.append(self._entry(target, "file"))
                continue
            name, _, spoken = line.partition("|")
            name = name.strip()
            if not name:
                continue
            forms = [s.strip() for s in spoken.split(",") if s.strip()]
            entries = [e for e in entries if e.name != name]
            entries.append(self._entry(name, "file", forms))
        return entries, aliases

    def _scan(self) -> list[Entry]:
        out = []
        seen = set()
        for d in self.dirs:
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for name in names:
                if name.startswith(".") or name in seen:
                    continue
                try:
                    if not (d / name).is_dir():
                        continue
                except OSError:
                    continue
                seen.add(name)
                shown = name.title() if name.isupper() and len(name) > 4 else name   # "ALFANO" -> "Alfano"
                out.append(self._entry(shown, "folder"))
        return out

    def _index(self) -> None:
        by_name: dict[str, Entry] = {}
        for e in self._folder_entries + self._file_entries:      # the file wins
            by_name[e.name.lower()] = e
        self.entries = list(by_name.values())
        self._by_size: dict[int, list[tuple[str, Entry]]] = {}
        for e in self.entries:
            for k, size in zip(e.keys, e.sizes):
                self._by_size.setdefault(size, []).append((k, e))
        self._by_letters = {}
        self._known = set()
        for e in self.entries:
            if e.letters:
                self._by_letters.setdefault(e.letters, e)
            self._known.add(e.letters)
            for p in _parts(e.name):                   # "Clew" of "Clew-boss", not "line" of "line break"
                if len(p) >= 4 and not self.dictionary.ordinary(p):
                    self._known.add(_letters(p))

    # -- questions ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> str:
        files = sum(1 for e in self.entries if e.source == "file")
        return (f"{len(self.entries)} names ({files} from {self.file.name if self.file else 'no file'}, "
                f"{len(self.entries) - files} folder names), {len(self.aliases)} aliases")

    def known(self, word: str) -> bool:
        """Is this word (or name) one of the lexicon's own, as written?"""
        return _letters(word) in self._known

    def name_aliases(self, name: str = "claude") -> tuple[list[str], list[str]]:
        """Aliases for the assistant's name: (anywhere, calling position only), as single lower-case words."""
        anywhere, calling = [], []
        for a in self.aliases:
            if norm_word(a.target) == name and len(a.heard) == 1:
                (calling if a.mode == "calling" else anywhere).append(a.heard[0])
        return anywhere, calling

    def by_letters(self, letters: str) -> Entry | None:
        return self._by_letters.get(letters)

    def closest(self, key: str, *, entries=None) -> list[tuple[float, Entry]]:
        """Entries ranked by distance to a key: the best ones, ties included (at most a few)."""
        if not key:
            return []
        n = consonants(key)
        allowed = None if entries is None else {id(e) for e in entries}
        found: dict[int, tuple[float, Entry]] = {}
        cutoff = 2.0
        for size in range(max(0, n - 2), n + 3):
            for k, e in self._by_size.get(size, ()):
                if allowed is not None and id(e) not in allowed:
                    continue
                if e.name.lower() == "claude" or e.ordinary:
                    continue                               # the name is addressing's business; "line break"
                                                           # and "Pigeonhole" are words, heard as words
                d = distance(key, k, cutoff)
                if d <= cutoff and (id(e) not in found or d < found[id(e)][0]):
                    found[id(e)] = (d, e)
        best = sorted(found.values(), key=lambda x: (x[0], x[1].source != "file", x[1].name))
        return best[:4]


def load(cfg: dict, *, dictionary_: Dictionary | None = None, log=lambda m: None) -> Lexicon | None:
    """The lexicon the listener settings ask for, or None (spoken_label or lexicon off)."""
    if not cfg.get("spoken_label", True) or not cfg.get("lexicon", True):
        return None
    from . import settings

    path = settings.project_path(cfg.get("lexicon_file") or "lexicon.txt")
    return Lexicon(path, cfg.get("lexicon_dirs") or (), dictionary_=dictionary_, log=log)
