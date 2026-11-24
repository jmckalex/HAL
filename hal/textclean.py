"""Turn a Markdown reply into plain text that sounds right when read aloud.

Pure functions only: no I/O, no globals that change, standard library only.

    to_speech(markdown)              -> cleaned prose ("" if nothing speakable)
    truncate(text, max_chars, tail)  -> text cut at a sentence boundary, plus tail
    pronounce(text)                  -> respell words the voice gets wrong ("HAL" -> "Hal")
    speakable(markdown, ...)         -> all three, in that order

The aim is what a person reading the reply aloud would actually say: code
blocks and tables are skipped, markup is unwrapped to its text, URLs are
dropped, file paths shrink to their file name, and every heading, list item
and paragraph becomes a sentence of its own so the voice pauses between them.
The listener's label and marks ("[spoken 21:14]", "[?]", "[heard as ...]",
hal/spoken.py) are never read aloud, even if a reply quotes them.
"""

from __future__ import annotations

import re
import unicodedata

from .spoken import for_speech

DEFAULT_MAX_CHARS = 600
DEFAULT_OVERFLOW = "The rest is on your screen."

# A code span longer than this many words is a command or expression nobody
# would read out; it is dropped rather than recited.
MAX_CODE_WORDS = 6

_TERMINAL = ".!?:;…"

# --------------------------------------------------------------------------
# Block level
# --------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s+(.*?))?\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_HRULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_BLOCKQUOTE = re.compile(r"^\s*(?:>\s?)+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?(.*)$")
_TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$")
_REF_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
_BOLD_LINE = re.compile(r"^\s*(\*\*|__)[^*_]+\1:?\s*$")
_HTML_ONLY_LINE = re.compile(r"^\s*(?:</?[A-Za-z][\w-]*(?:\s[^<>]*)?/?>\s*)+$")

_GAP = None  # marks removed content (a code block or table) in the block list


def _is_table_separator(line: str) -> bool:
    return "|" in line and "-" in line and bool(_TABLE_SEPARATOR.match(line))


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return len(s) > 1 and s.startswith("|") and s.endswith("|")


def _split_blocks(markdown: str) -> list[str | None]:
    """Split Markdown into speakable blocks; removed content becomes _GAP."""
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str | None] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(_join_lines(para))
            para.clear()

    i, n = 0, len(lines)
    while i < n:
        line = _BLOCKQUOTE.sub("", lines[i], count=1)
        stripped = line.strip()

        # Fenced code block: skip to the closing fence (or the end).
        m = _FENCE_OPEN.match(line)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = m.group(1)
            flush()
            blocks.append(_GAP)
            i += 1
            while i < n:
                close = _BLOCKQUOTE.sub("", lines[i], count=1).strip()
                if close.startswith(fence[0] * len(fence)) and not close.strip(fence[0]):
                    break
                i += 1
            i += 1
            continue

        # HTML comment, possibly over several lines.
        if stripped.startswith("<!--"):
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue

        # Table: a header row followed by a separator row, or rows of |...|.
        if ("|" in line and i + 1 < n and _is_table_separator(lines[i + 1])) or _is_pipe_row(line):
            flush()
            blocks.append(_GAP)
            i += 1
            while i < n and lines[i].strip() and ("|" in lines[i] or _is_table_separator(lines[i])):
                i += 1
            continue

        if not stripped:
            flush()
        elif _SETEXT_UNDERLINE.match(line) and para:
            flush()  # the paragraph above was a heading
        elif _HRULE.match(line) or _REF_DEFINITION.match(line) or _HTML_ONLY_LINE.match(line):
            flush()
        elif m := _ATX_HEADING.match(line):
            flush()
            if m.group(1):
                blocks.append(m.group(1))
        elif m := _LIST_ITEM.match(line):
            flush()
            para.append(m.group(1))
        elif _BOLD_LINE.match(line):
            flush()
            blocks.append(stripped)
        else:
            para.append(stripped)
        i += 1
    flush()
    return blocks


def _join_lines(lines: list[str]) -> str:
    """Join a paragraph's lines. A line break before a capital letter or digit,
    after a line with no closing punctuation, is treated as a sentence break
    (line-per-item text); otherwise it is a soft wrap."""
    out = lines[0].strip().rstrip("\\").rstrip()
    for nxt in lines[1:]:
        nxt = nxt.strip().rstrip("\\").rstrip()
        if not nxt:
            continue
        if out and out[-1] not in _TERMINAL + "," and (nxt[0].isupper() or nxt[0].isdigit()):
            out += "."
        out += " " + nxt
    return out


# --------------------------------------------------------------------------
# Inline level
# --------------------------------------------------------------------------

_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]|\([^)]*\))*(?:\s+\"[^\"]*\")?\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
_AUTOLINK = re.compile(r"<(?:https?|ftp|mailto):[^>]*>", re.I)
_URL = re.compile(r"\b(?:(?:https?|ftp|file)://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"]", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# A link that ended a clause ("see the docs at https://...") takes its
# connecting word with it, so the sentence doesn't end on "at".
_DANGLING_LINK = re.compile(
    r"(?:\s+(?:or|and|at|in|from|to|via|on|see|here))?\s*:?\s*\x02(?=\s*(?:[.,;:!?)]|$))", re.I
)
_HTML_COMMENT = re.compile(r"<!--.*?-->")
_HTML_TAG = re.compile(r"</?[A-Za-z][\w-]*(?:\s[^<>]*)?/?>")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])")
_ITALIC_UNDERSCORE = re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])")
_STRIKE = re.compile(r"~~(.+?)~~")

# Path-like tokens: absolute, home- or dot-relative, or relative with either
# two or more slashes or a file extension on the last part. Must contain a
# letter, so dates like 11/15/2026 are left alone.
_PATH = re.compile(
    r"(?<![\w/.~$}-])"
    r"(?=[^\s]*[A-Za-z])"
    r"(?:"
    r"(?:~|\.{1,2})?(?:/[\w.@+${}-]+)+/?"                # /abs/path, ~/x, ./x, ../x
    r"|[\w.@+${}-]+(?:/[\w.@+${}-]+){2,}/?"              # a/b/c, $VAR/b/c
    r"|[\w.@+${}-]+/[\w@+-]*\.[A-Za-z]\w{0,7}"           # dir/file.ext
    r")"
    r"(?::\d+(?:[-:]\d+)?)?"                             # optional :line or :line:col
    r"(?![\w/])"
)
_FILE_LINE = re.compile(r"\b([\w-]+\.[A-Za-z]\w{0,7}):(\d+)(?:[-:]\d+)?\b")

_ABBREVIATIONS = [
    (re.compile(r"\be\.g\.,?", re.I), "for example,"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\bapprox\.(?=\s)", re.I), "approximately"),
]
_UNITS = [
    (re.compile(r"(?<=\d)\s?[x×](?=\s|$|[,.;:!?)])"), " times"),       # 100x, 3 ×
    (re.compile(r"~\s?(?=\d)"), "about "),                             # ~0.4
    (re.compile(r"(?<=\d)\s?ms\b"), " milliseconds"),
    (re.compile(r"(?<=\d) (?:s|secs?)\b(?!['’])"), " seconds"),        # 0.4 s, 2 sec
    (re.compile(r"(?<![\d'’.])(\d{1,3}(?:\.\d+)?)(?:s|secs?)\b"), r"\1 seconds"),  # 2s, 0.4s (not 1990s, '90s)
    (re.compile(r"(?<=\d)\s?min\b"), " minutes"),
    (re.compile(r"(?<=\d)\s?kHz\b"), " kilohertz"),
    (re.compile(r"(?<=\d)\s?Hz\b"), " hertz"),
    (re.compile(r"(?<=\d)\s?KB\b", re.I), " kilobytes"),
    (re.compile(r"(?<=\d)\s?MB\b"), " megabytes"),
    (re.compile(r"(?<=\d)\s?GB\b"), " gigabytes"),
    (re.compile(r"(?<=\d)\s?TB\b"), " terabytes"),
    (re.compile(r"(?<![\w&])#(?=\d)"), "number "),                    # #42
]
_ARROWS = re.compile(r"\s*(?:->|=>|-->|→|⇒|⟶|⟹|➜|➔)\s*")
_LEFTOVER_SYMBOLS = re.compile(r"[*_#|<>{}\[\]\\^~`]+")
_EMPTY_PARENS = re.compile(r"\(\s*[,.;:]*\s*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…])")
_REPEATED_PUNCT = re.compile(r"([,.;:!?])(?:\s*[,.;:])+")
_WHITESPACE = re.compile(r"\s+")


def _speak_path(match: re.Match) -> str:
    """'/a/b/speaker.py:42' -> 'speaker.py line 42'."""
    token = match.group(0)
    line = ""
    m = re.search(r":(\d+)(?:[-:]\d+)?$", token)
    if m:
        token, line = token[: m.start()], f" line {m.group(1)}"
    parts = [p for p in token.split("/") if p not in ("", ".", "..", "~")]
    return (parts[-1] if parts else "") + line


def _speak_code(code: str) -> str:
    """How to say the content of an inline code span."""
    code = _URL.sub(" ", code).strip()
    if not code:
        return ""
    code = _PATH.sub(_speak_path, code)
    code = _FILE_LINE.sub(r"\1 line \2", code)
    code = re.sub(r"(?<=\w)/(?=\s|$)", "", code)        # state/ -> state
    code = re.sub(r"\(\s*\)", "", code)                 # foo() -> foo
    code = code.replace("&&", " and ").replace("||", " or ")
    code = re.sub(r"->|=>|::", " ", code)
    code = _LEFTOVER_SYMBOLS.sub(" ", code)
    code = _WHITESPACE.sub(" ", code).strip(" .,:;")
    if len(code.split()) > MAX_CODE_WORDS:
        return ""
    return code


def _strip_symbols(text: str) -> str:
    """Drop emoji and pictographs, keep letters, digits and punctuation."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("So", "Cs", "Co", "Cn") or ch in "\u200d\ufe0f\ufe0e•◦▪▸►":  # ZWJ, variation selectors, bullets
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _clean_inline(text: str) -> str:
    codes: list[str] = []

    def stash(spoken: str) -> str:
        codes.append(spoken)
        return f"\x00{len(codes) - 1}\x00"

    escapes: list[str] = []

    def stash_escape(m: re.Match) -> str:
        escapes.append(m.group(1))
        return f"\x01{len(escapes) - 1}\x01"

    text = _CODE_SPAN.sub(lambda m: stash(_speak_code(m.group(2))), text)
    text = _ESCAPE.sub(stash_escape, text)
    text = _HTML_COMMENT.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _REF_LINK.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    text = _AUTOLINK.sub("\x02", text)
    text = _URL.sub("\x02", text)
    text = _EMAIL.sub("\x02", text)
    text = _DANGLING_LINK.sub("", text)
    text = text.replace("\x02", " ")
    text = _HTML_TAG.sub(" ", text)
    for _ in range(2):  # nested emphasis such as ***both***
        text = _BOLD.sub(r"\2", text)
        text = _ITALIC_STAR.sub(r"\1", text)
        text = _ITALIC_UNDERSCORE.sub(r"\1", text)
    text = _STRIKE.sub(r"\1", text)
    # Code spans may have left placeholders inside link text etc.; paths in
    # plain prose get the same treatment as paths in code.
    text = _PATH.sub(_speak_path, text)
    text = _FILE_LINE.sub(r"\1 line \2", text)
    for pattern, spoken in _ABBREVIATIONS:
        text = pattern.sub(spoken, text)
    for pattern, spoken in _UNITS:
        text = pattern.sub(spoken, text)
    text = _ARROWS.sub(" to ", text)
    text = text.replace("&&", " and ")
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " ", text)  # and/or -> and or
    text = re.sub(r"(?<=\w)/(?=[\s.,;:!?)\x00]|$)", "", text)  # a directory's trailing slash
    text = _LEFTOVER_SYMBOLS.sub(" ", text)
    text = re.sub(r"\x01(\d+)\x01", lambda m: escapes[int(m.group(1))], text)
    text = _LEFTOVER_SYMBOLS.sub(" ", text)  # escaped markup characters too
    text = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)
    text = _strip_symbols(text)
    text = _EMPTY_PARENS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return text.lstrip(",.;:!?…–— ").strip()


def _has_words(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


# A "Sources:" / "References" label (plain, bold or a heading), optionally
# followed on the same line by the source itself.
_SOURCES_LABEL = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:sources?|references?|citations?)\s*:?\s*(?:\*\*|__)?\s*:?\s*(.*)$",
    re.I,
)
_LINK_ONLY = re.compile(r"^(?:\s*(?:[-*+•]|\d{1,3}[.)])\s+)?[\s,;|·]*(?:(?:\[[^\]]+\]\([^)]*\)|<?https?://\S+>?)[\s,;|·.]*)+$")


def _drop_sources(markdown: str) -> str:
    """Remove source lists: they stay on screen but are not read aloud.

    A source list is a "Sources"/"References" label followed by list items or
    link-only lines, or a "Source: [link](url)" line. Text after it is kept.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        match = _SOURCES_LABEL.match(lines[i])
        if match:
            rest = match.group(1).strip()
            j = i + 1
            if rest and _LINK_ONLY.match(rest):
                pass  # "Source: [name](url)": drop the line and any items after it
            elif not rest:
                k = j
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k >= len(lines) or not (_LIST_ITEM.match(lines[k]) or _LINK_ONLY.match(lines[k])):
                    out.append(lines[i])
                    i += 1
                    continue
            else:
                out.append(lines[i])
                i += 1
                continue
            while j < len(lines) and (
                not lines[j].strip() or _LIST_ITEM.match(lines[j]) or _LINK_ONLY.match(lines[j])
                or lines[j][:1] in (" ", "\t")  # continuation of a list item
            ):
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def to_speech(markdown: str | None) -> str:
    """Convert a Markdown reply to plain speakable prose ("" if none)."""
    if not markdown or not markdown.strip():
        return ""
    raw_blocks = _split_blocks(_drop_sources(for_speech(markdown)))

    sentences: list[str | None] = []
    for block in raw_blocks:
        if block is _GAP:
            if sentences and sentences[-1] is not _GAP:
                sentences.append(_GAP)
            continue
        text = _clean_inline(block)
        if _has_words(text):
            sentences.append(text)

    out: list[str] = []
    for idx, text in enumerate(sentences):
        if text is _GAP:
            continue
        nxt = sentences[idx + 1] if idx + 1 < len(sentences) else _GAP
        if text.endswith(":") and nxt is _GAP:
            # "Run this:" followed by code we skipped, or nothing at all.
            text = text[:-1].rstrip() + "."
        elif text[-1] not in _TERMINAL:
            text = text.rstrip(",") + "."
        out.append(text)
    return " ".join(out)


# --------------------------------------------------------------------------
# Length cap
# --------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK.split(text) if s.strip()]


def truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS, overflow: str = DEFAULT_OVERFLOW) -> str:
    """Cut text to at most max_chars at a sentence boundary and append overflow.

    If what would be cut is shorter than the overflow message itself, the
    whole text is kept. A first sentence longer than the cap is cut at the
    last clause or word boundary before it. max_chars <= 0 disables the cap.
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    kept, used = "", 0
    for sentence in sentences:
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        kept, used = candidate, used + 1

    if kept:
        remainder = " ".join(sentences[used:])
        if len(remainder) <= len(overflow):
            return text
    else:
        head = text[:max_chars]
        cut = max(head.rfind(", "), head.rfind("; "), head.rfind(": "), head.rfind(" – "), head.rfind(" — "))
        if cut < max_chars // 2:
            cut = head.rfind(" ")
        if cut <= 0:
            cut = max_chars
        kept = head[:cut].rstrip(" ,;:–—") + "."

    return f"{kept} {overflow}".strip() if overflow else kept


# --------------------------------------------------------------------------
# Pronunciation
# --------------------------------------------------------------------------

# Whole-word, case-sensitive respellings applied just before synthesis.
# Piper's espeak-ng front end spells out an all-caps word that sits inside
# mixed-case text ("This is HAL speaking" -> "aitch-ay-el speaking"). That is
# right for API, JSON, URL or CLI, so those are left alone; these entries are
# for all-caps words people say as words. A trailing plural "s" is kept
# (WAVs -> waves); "HAL's", "HAL-9000" and "HAL," match as whole words, while
# "HALT", "SHALL" and lowercase "hal" don't.
PRONUNCIATIONS: dict[str, str] = {
    "HAL": "Hal",                # the name, not H-A-L (espeak also drops the s of "HAL's")
    "WAV": "wave",               # W-A-V
    "RAM": "ram",                # R-A-M
    "DOM": "dom",                # D-O-M
    "ENV": "env",                # E-N-V
    "ONNX": "onyx",              # "onnks"
    "STDIN": "standard in",      # S-T-din
    "STDOUT": "standard out",
    "STDERR": "standard error",
}

_PRONUNCIATION = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, PRONUNCIATIONS), key=len, reverse=True)) + r")(s?)\b"
)


# Titles before a name are written out, so the voice doesn't treat the full
# stop as the end of a sentence and pause: "Dr. Alexander" -> "Doctor Alexander".
TITLES: dict[str, str] = {"Dr": "Doctor", "Prof": "Professor", "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz"}
_TITLE = re.compile(r"\b(" + "|".join(TITLES) + r")\.?(?=\s+[A-Z])")


# The TeX family: the X is a chi, said "tech". "TeX+" is the user's own project,
# said "Tech Plus". Longest names first, so "LaTeX" is not caught by "TeX".
TEX_FAMILY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bTeX\s?\+(?=\W|$)|\bTeX Plus\b"), "Tech Plus"),
    (re.compile(r"\bpdfLaTeX\b"), "P D F Lay-tech"),
    (re.compile(r"\bLuaLaTeX\b"), "Lua Lay-tech"),
    (re.compile(r"\bXeLaTeX\b"), "Zee Lay-tech"),
    (re.compile(r"\bLaTeX\b"), "Lay-tech"),
    (re.compile(r"\bpdfTeX\b"), "P D F Tech"),
    (re.compile(r"\bLuaTeX\b"), "Lua Tech"),
    (re.compile(r"\bXeTeX\b"), "Zee Tech"),
    (re.compile(r"\bBibTeX\b"), "Bib Tech"),
    (re.compile(r"\bTeX\b"), "Tech"),
]


def pronounce(text: str) -> str:
    """Respell words from PRONUNCIATIONS so the voice says them as words."""
    text = _TITLE.sub(lambda m: TITLES[m.group(1)], text)
    for pattern, spoken in TEX_FAMILY:
        text = pattern.sub(spoken, text)
    return _PRONUNCIATION.sub(lambda m: PRONUNCIATIONS[m.group(1)] + m.group(2), text)


def speakable(markdown: str | None, max_chars: int = DEFAULT_MAX_CHARS, overflow: str = DEFAULT_OVERFLOW) -> str:
    """Clean a Markdown reply for speech, apply the length cap, fix pronunciation."""
    return pronounce(truncate(to_speech(markdown), max_chars, overflow))
