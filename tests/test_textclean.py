"""Unit tests for hal.textclean. Run: .venv/bin/python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hal.textclean import (  # noqa: E402
    DEFAULT_OVERFLOW,
    PRONUNCIATIONS,
    pronounce,
    split_sentences,
    speakable,
    to_speech,
    truncate,
)

REALISTIC_REPLY = """\
## Summary

I fixed the race in `hal/speaker.py:88`: the **lock** was released *before* the
player started. All 42 tests pass now (see [the hooks docs](https://code.claude.com/docs/en/hooks.md)).

Here's the change:

```python
with _locked():
    if _read(CURRENT_FILE) != ticket:
        return 0
    player = subprocess.Popen(cmd)
```

| File | Change |
|------|:------:|
| `hal/speaker.py` | hold the lock while starting afplay |
| `tests/test_hook.py` | new race test |

### Next steps
- Run `python -m unittest discover -s tests -v` to confirm
- [ ] Restart Claude Code so the hooks reload
1. Try `/voice tap`
2. Say something

> Synthesis is ~0.4s per reply, e.g. on a cold start.

Details: https://example.com/a/very/long/url?with=query
"""


class Empty(unittest.TestCase):
    def test_nothing_in_nothing_out(self):
        for text in (None, "", "   ", "\n\n", "---", "***", "<!-- note -->", "✅ 🚀"):
            self.assertEqual(to_speech(text), "", repr(text))

    def test_only_code_is_silent(self):
        self.assertEqual(to_speech("```bash\nls -la\n```"), "")

    def test_only_table_is_silent(self):
        self.assertEqual(to_speech("| a | b |\n|---|---|\n| 1 | 2 |"), "")

    def test_only_url_is_silent(self):
        self.assertEqual(to_speech("https://example.com/path"), "")

    def test_speakable_of_empty_is_empty(self):
        self.assertEqual(speakable(""), "")
        self.assertEqual(speakable("```\nx = 1\n```"), "")


class CodeBlocks(unittest.TestCase):
    def test_fenced_block_dropped(self):
        md = "Before.\n\n```python\nprint('hi')\nsecret = 1\n```\n\nAfter."
        self.assertEqual(to_speech(md), "Before. After.")

    def test_tilde_fence_and_longer_fence(self):
        md = "A.\n~~~\ncode\n~~~\nB.\n\n````md\n```\nnested\n```\n````\nC."
        self.assertEqual(to_speech(md), "A. B. C.")

    def test_unclosed_fence_drops_rest(self):
        self.assertEqual(to_speech("Start.\n```\ncode\nmore code"), "Start.")

    def test_fence_inside_list_item(self):
        md = "1. Install:\n\n   ```bash\n   pip install x\n   ```\n2. Run it."
        self.assertEqual(to_speech(md), "Install. Run it.")

    def test_fence_inside_blockquote(self):
        md = "> Quote.\n> ```\n> code\n> ```\n> End."
        self.assertEqual(to_speech(md), "Quote. End.")

    def test_one_line_triple_backticks_is_inline_code(self):
        self.assertEqual(to_speech("Use ```foo``` here."), "Use foo here.")

    def test_colon_before_dropped_code_becomes_period(self):
        self.assertEqual(to_speech("Run this:\n\n```\nls\n```"), "Run this.")
        self.assertEqual(to_speech("Here is the fix:\n```\nx\n```\nIt works."), "Here is the fix. It works.")

    def test_colon_before_list_is_kept(self):
        self.assertEqual(to_speech("Changes:\n- one\n- two"), "Changes: one. two.")


class Tables(unittest.TestCase):
    def test_table_with_pipes_dropped(self):
        md = "Results:\n\n| Test | Result |\n|:-----|-------:|\n| a | ok |\n| b | ok |\n\nAll good."
        self.assertEqual(to_speech(md), "Results. All good.")

    def test_table_without_outer_pipes_dropped(self):
        md = "Intro.\n\nName | Value\n--- | ---\nx | 1\ny | 2\n\nOutro."
        self.assertEqual(to_speech(md), "Intro. Outro.")

    def test_pipe_in_prose_is_not_a_table(self):
        self.assertEqual(to_speech("Pipe it with a | b."), "Pipe it with a b.")


class InlineMarkup(unittest.TestCase):
    def test_emphasis_unwrapped(self):
        self.assertEqual(to_speech("This is **bold**, *italic*, __strong__, _em_ and ***both***."),
                         "This is bold, italic, strong, em and both.")

    def test_strikethrough_unwrapped(self):
        self.assertEqual(to_speech("It is ~~broken~~ fixed."), "It is broken fixed.")

    def test_snake_case_is_not_italic(self):
        self.assertEqual(to_speech("Set max_chars_total and min_value."), "Set max chars total and min value.")

    def test_inline_code_unwrapped(self):
        self.assertEqual(to_speech("Call `stop()` then `speak`."), "Call stop then speak.")

    def test_identifier_in_code_is_spoken_as_words(self):
        self.assertEqual(to_speech("The `last_assistant_message` field."), "The last assistant message field.")

    def test_long_code_span_dropped(self):
        out = to_speech("Run `python -m unittest discover -s tests -v --failfast` to check.")
        self.assertEqual(out, "Run to check.")

    def test_short_command_kept(self):
        self.assertEqual(to_speech("Run `hal off` to mute."), "Run hal off to mute.")

    def test_escaped_markup(self):
        self.assertEqual(to_speech(r"Not \*bold\* and 2 \* 3."), "Not bold and 2 3.")

    def test_html_tags_and_comments_removed(self):
        md = "<details>\n<summary>More</summary>\nHidden <b>text</b> here.<!-- c -->\n</details>"
        self.assertEqual(to_speech(md), "More. Hidden text here.")

    def test_emoji_removed(self):
        self.assertEqual(to_speech("✅ Done! 🎉 All tests pass ⚠️"), "Done! All tests pass.")


class Links(unittest.TestCase):
    def test_link_becomes_text(self):
        self.assertEqual(to_speech("See [the docs](https://x.y/docs) now."), "See the docs now.")

    def test_link_with_parens_in_url(self):
        self.assertEqual(to_speech("Read [this](https://en.wikipedia.org/wiki/HAL_(computer))."), "Read this.")

    def test_image_dropped(self):
        self.assertEqual(to_speech("Look ![diagram](img.png) here."), "Look here.")

    def test_reference_links_and_definitions(self):
        md = "See [the guide][g] and note[^1].\n\n[g]: https://example.com/guide\n[^1]: footnote text"
        self.assertTrue(to_speech(md).startswith("See the guide and note."))

    def test_bare_url_dropped_with_dangling_word(self):
        self.assertEqual(to_speech("See the docs at https://example.com/docs."), "See the docs.")
        self.assertEqual(to_speech("Docs: https://example.com"), "Docs.")

    def test_bare_url_mid_sentence(self):
        self.assertEqual(to_speech("Open https://example.com/x in a browser."), "Open in a browser.")

    def test_autolink_and_www_and_email(self):
        out = to_speech("Write to me@example.com, visit <https://a.b> or www.example.com today.")
        for junk in ("@", "example", "https", "a.b", "<", ">"):
            self.assertNotIn(junk, out)
        self.assertTrue(out.startswith("Write") and out.endswith("today."), out)

    def test_file_link_from_claude_code(self):
        self.assertEqual(to_speech("Edited [speaker.py:42](hal/speaker.py#L42)."), "Edited speaker.py line 42.")


class Paths(unittest.TestCase):
    def test_absolute_path_becomes_file_name(self):
        self.assertEqual(to_speech("Wrote /Users/x/Source/HAL/hal/speaker.py today."),
                         "Wrote speaker.py today.")

    def test_home_path(self):
        self.assertEqual(to_speech("Edit ~/.claude/settings.json first."), "Edit settings.json first.")

    def test_relative_path_in_code(self):
        self.assertEqual(to_speech("In `src/hal/textclean.py` I added tests."), "In textclean.py I added tests.")

    def test_path_with_line_number(self):
        self.assertEqual(to_speech("Bug at `hal/hook.py:120`."), "Bug at hook.py line 120.")
        self.assertEqual(to_speech("Bug at hook.py:120."), "Bug at hook.py line 120.")

    def test_env_var_path(self):
        self.assertEqual(to_speech("Runs $CLAUDE_PROJECT_DIR/bin/hal hook."), "Runs hal hook.")

    def test_directory_trailing_slash(self):
        self.assertEqual(to_speech("Logs are in `state/` and in logs/."), "Logs are in state and in logs.")

    def test_slash_words_are_not_paths(self):
        self.assertEqual(to_speech("Use and/or here."), "Use and or here.")

    def test_dates_are_not_paths(self):
        self.assertEqual(to_speech("Due 11/15/2026."), "Due 11/15/2026.")


class Structure(unittest.TestCase):
    def test_headings_become_sentences(self):
        self.assertEqual(to_speech("# Title\n\n## Sub heading ##\nBody text"), "Title. Sub heading. Body text.")

    def test_setext_heading(self):
        self.assertEqual(to_speech("Overview\n========\nText here"), "Overview. Text here.")

    def test_heading_keeps_question_mark(self):
        self.assertEqual(to_speech("## Why?\nBecause."), "Why? Because.")

    def test_bullets_become_sentences(self):
        md = "- first item\n- second item\n  continued here\n* third\n+ fourth"
        self.assertEqual(to_speech(md), "first item. second item continued here. third. fourth.")

    def test_numbered_and_task_lists(self):
        md = "1. Alpha\n2) Beta\n- [ ] gamma\n- [x] delta"
        self.assertEqual(to_speech(md), "Alpha. Beta. gamma. delta.")

    def test_nested_lists(self):
        self.assertEqual(to_speech("- outer\n  - inner\n    - innermost"), "outer. inner. innermost.")

    def test_blockquote(self):
        self.assertEqual(to_speech("> Quoted line\n> > nested"), "Quoted line nested.")

    def test_horizontal_rules_dropped(self):
        self.assertEqual(to_speech("One.\n\n---\n\nTwo.\n\n* * *\n\nThree."), "One. Two. Three.")

    def test_soft_wrap_joins(self):
        self.assertEqual(to_speech("This sentence was\nwrapped in the middle."), "This sentence was wrapped in the middle.")

    def test_line_per_item_gets_breaks(self):
        self.assertEqual(to_speech("Files: 3\nTests: pass\nTime: 2 min"), "Files: 3. Tests: pass. Time: 2 minutes.")

    def test_bold_line_is_own_sentence(self):
        self.assertEqual(to_speech("**Summary**\nAll fine."), "Summary. All fine.")


class Wording(unittest.TestCase):
    def test_abbreviations(self):
        self.assertEqual(to_speech("Use a hook, e.g. Stop, i.e. the end."),
                         "Use a hook, for example, Stop, that is, the end.")

    def test_units_and_multipliers(self):
        self.assertEqual(to_speech("It took ~0.4s, then 12 ms, at 100x real time."),
                         "It took about 0.4 seconds, then 12 milliseconds, at 100 times real time.")

    def test_decades_are_not_seconds(self):
        self.assertEqual(to_speech("Films of the 1960s."), "Films of the 1960s.")

    def test_sizes_and_rates(self):
        self.assertEqual(to_speech("A 63 MB model at 22050 Hz."), "A 63 megabytes model at 22050 hertz.")

    def test_issue_numbers(self):
        self.assertEqual(to_speech("Fixes #42."), "Fixes number 42.")

    def test_arrows(self):
        self.assertEqual(to_speech("Changed `a` → `b` and x -> y."), "Changed a to b and x to y.")


class Truncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(truncate("One. Two.", 100), "One. Two.")

    def test_cut_at_sentence_boundary(self):
        text = "First sentence here. Second sentence here. Third sentence is quite a lot longer than the rest."
        out = truncate(text, 45, "More on screen.")
        self.assertEqual(out, "First sentence here. Second sentence here. More on screen.")

    def test_never_exceeds_cap_plus_overflow(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(100))
        out = truncate(text, 300)
        self.assertTrue(out.endswith(DEFAULT_OVERFLOW))
        self.assertLessEqual(len(out), 300 + 1 + len(DEFAULT_OVERFLOW))
        body = out[: -len(DEFAULT_OVERFLOW)].strip()
        self.assertTrue(body.endswith("."))

    def test_short_remainder_is_kept(self):
        text = "A" * 50 + ". Ok."
        self.assertEqual(truncate(text, 52), text)

    def test_long_first_sentence_cut_at_clause(self):
        text = "This opening clause is fine, " + "and then it goes on " * 20 + "forever."
        out = truncate(text, 50, "Tail.")
        self.assertEqual(out, "This opening clause is fine. Tail.")

    def test_long_first_sentence_without_clauses_cut_at_word(self):
        text = "word " * 50
        out = truncate(text, 23, "Tail.")
        self.assertEqual(out, "word word word word. Tail.")

    def test_zero_disables_cap(self):
        text = "x. " * 1000
        self.assertEqual(truncate(text, 0), text.strip())

    def test_empty_overflow(self):
        self.assertEqual(truncate("One one. Two two. Three three.", 10, ""), "One one.")

    def test_split_sentences_keeps_closing_quotes(self):
        self.assertEqual(split_sentences('He said "stop." Then left! Why? Ok…'),
                         ['He said "stop."', "Then left!", "Why?", "Ok…"])


class Pronunciation(unittest.TestCase):
    def test_hal_is_a_name(self):
        cases = {
            "Good morning, Dave. I am HAL 9000.": "Good morning, Dave. I am Hal 9000.",
            "HAL": "Hal",
            "HAL-9000 is ready.": "Hal-9000 is ready.",
            "HAL's voice": "Hal's voice",
            "HAL’s voice": "Hal’s voice",
            "Well, HAL, open the doors.": "Well, Hal, open the doors.",
            "This is HAL.": "This is Hal.",
            "(HAL)": "(Hal)",
            '"HAL"': '"Hal"',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pronounce(text), expected)

    def test_only_whole_uppercase_word(self):
        for text in ("HALT", "SHALL", "HALLWAY", "HAL9000", "HALs9", "hal", "Hal", "shall", "halt",
                     "the hal.py file", "bin/hal", "aHAL"):
            with self.subTest(text=text):
                self.assertEqual(pronounce(text), text)

    def test_acronyms_left_for_the_voice_to_spell(self):
        text = "The API returns JSON from a URL; the CLI, SDK and CI use HTTP."
        self.assertEqual(pronounce(text), text)

    def test_other_entries(self):
        self.assertEqual(pronounce("Two WAVs and a WAV file"), "Two waves and a wave file")
        self.assertEqual(pronounce("Read STDIN, write STDOUT and STDERR."),
                         "Read standard in, write standard out and standard error.")
        self.assertEqual(pronounce("The ONNX model, the DOM, the ENV and RAM."),
                         "The onyx model, the dom, the env and ram.")

    def test_table_keys_are_uppercase_words(self):
        for word in PRONUNCIATIONS:
            self.assertRegex(word, r"^[A-Z][A-Z0-9]*$")

    def test_idempotent(self):
        text = "I am HAL 9000. Reading STDIN."
        self.assertEqual(pronounce(pronounce(text)), pronounce(text))

    def test_speakable_applies_pronunciation_after_cleaning(self):
        self.assertEqual(speakable("# HAL\n\nI am **HAL** (`HAL_9000`)."), "Hal. I am Hal (Hal 9000).")
        self.assertEqual(to_speech("I am HAL."), "I am HAL.")  # display-side cleaning leaves it alone


class Realistic(unittest.TestCase):
    def test_realistic_reply(self):
        out = to_speech(REALISTIC_REPLY)
        self.assertEqual(
            out,
            "Summary. I fixed the race in speaker.py line 88: the lock was released before the "
            "player started. All 42 tests pass now (see the hooks docs). Here's the change. "
            "Next steps. Run to confirm. Restart Claude Code so the hooks reload. Try voice tap. "
            "Say something. Synthesis is about 0.4 seconds per reply, for example, on a cold start. Details.",
        )
        for junk in ("```", "|", "http", "**", "`", "_locked", "subprocess", "Popen", "#"):
            self.assertNotIn(junk, out)

    def test_realistic_reply_capped(self):
        out = speakable(REALISTIC_REPLY, max_chars=200)
        self.assertTrue(out.endswith(DEFAULT_OVERFLOW))
        self.assertLessEqual(len(out), 200 + 1 + len(DEFAULT_OVERFLOW))


class TitlesTest(unittest.TestCase):
    def test_title_before_a_name_is_written_out(self):
        self.assertEqual(pronounce("Certainly, Dr. Alexander. I will."), "Certainly, Doctor Alexander. I will.")
        self.assertEqual(pronounce("Good morning, Dr Alexander."), "Good morning, Doctor Alexander.")
        self.assertEqual(pronounce("Ask Prof. Smith."), "Ask Professor Smith.")

    def test_other_uses_are_left_alone(self):
        for text in ("He lives on Elm Dr. near the park.", "See the dr. tomorrow.", "Mr and mrs are titles."):
            self.assertEqual(pronounce(text), text)

    def test_reply_is_one_sentence_for_the_voice(self):
        self.assertEqual(speakable("Understood, Dr. Alexander."), "Understood, Doctor Alexander.")


class TexFamilyTest(unittest.TestCase):
    def test_tex_plus_is_tech_plus(self):
        self.assertEqual(pronounce("TeX+ is paused. I asked TeX+, and TeX Plus replied."),
                         "Tech Plus is paused. I asked Tech Plus, and Tech Plus replied.")

    def test_family_names(self):
        self.assertEqual(pronounce("LaTeX, pdfTeX, LuaTeX and plain TeX; a TeX run-on."),
                         "Lay-tech, P D F Tech, Lua Tech and plain Tech; a Tech run-on.")

    def test_other_words_are_untouched(self):
        for text in ("The text is fine.", "Texas and context.", "A vertex in the graph."):
            self.assertEqual(pronounce(text), text)


class SourcesTest(unittest.TestCase):
    def test_trailing_sources_list_is_not_spoken(self):
        md = "It will rain tomorrow.\n\nSources:\n- [Met Office](https://example.com/a)\n- [BBC Weather](https://example.com/b)\n"
        self.assertEqual(to_speech(md), "It will rain tomorrow.")

    def test_label_variants(self):
        for label in ("**Sources**", "## Sources", "Sources", "References:", "**Sources:**", "Source:"):
            md = f"Done.\n\n{label}\n\n1. [One](https://example.com/1)\n2. https://example.com/2\n"
            self.assertEqual(to_speech(md), "Done.", label)

    def test_single_line_source(self):
        md = "About 9 degrees.\n\nSource: [Open-Meteo forecast](https://open-meteo.com/)"
        self.assertEqual(to_speech(md), "About 9 degrees.")

    def test_text_after_sources_is_kept(self):
        md = "First.\n\nSources:\n- [A](https://example.com/a)\n\nOne more thing."
        self.assertEqual(to_speech(md), "First. One more thing.")

    def test_ordinary_sentences_about_sources_are_kept(self):
        md = "Sources: the two papers disagree on the constant.\n\nThe source is reliable.\n\nSources\n\nI checked three of them."
        out = to_speech(md)
        self.assertIn("the two papers disagree", out)
        self.assertIn("The source is reliable", out)
        self.assertIn("I checked three of them", out)

    def test_sources_only_reply_is_silent(self):
        self.assertEqual(to_speech("Sources:\n- [A](https://example.com/a)"), "")


if __name__ == "__main__":
    unittest.main()
