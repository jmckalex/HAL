"""Spoken requests: the label, the marks and the lexicon.

hal/spoken.py (the label and marks; what the hooks and desk see), annotate.py
(what the listener types), lexicon.py (corrections), addressing.py (the name's
aliases), and every check in hal/hook.py and hal/desk.py with and without the
label. AsrMarksTest runs the real Parakeet on synthesized speech (`say -o` to a
file; no audio device), and the listener over such a file; it is skipped when
.venv-listen, the model, `say` or ffmpeg is missing.

Run: .venv/bin/python -m unittest tests.test_listen_spoken -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal import hook, spoken  # noqa: E402
from hal.listener import addressing, annotate, lexicon, settings, turns  # noqa: E402
from hal.listener.annotate import HeardWord  # noqa: E402
from hal.textclean import speakable  # noqa: E402
from listen_harness import LISTEN_PY, ROOT, make_tmp, rmtree  # noqa: E402
from test_hook import HookTestBase  # noqa: E402
from test_listen_duplex import DuplexRig  # noqa: E402
from test_listen_privacy import Rig  # noqa: E402
from test_listen_turns import track  # noqa: E402

CFG = dict(settings.DEFAULTS)
LABEL = re.compile(r"^\[spoken \d\d:\d\d\] ")
# A small dictionary, so the tests don't depend on /usr/share/dict.
WORDS = set("""
a about add after again agent all an and any anything are argued as ask at bicycle boss break broken build but by
can check claude cloud clue could did do does door draft drive email files for from go god grant had hacking
hard he heading her his i idea robbing in is it keep last library line manual milk my night now of on open
or paper pass plan please plus probability publish repository said say send server shop storage tech that the
there think to todd took was we were what when where whether will with would you your second chapter thesis
weather tomorrow time minutes department sent professor review wrote one book excellent stop thank thanks you
very much good morning yes go ahead that's enough quiet be fine ok okay right sure hello hey hi read again
remind call sister afternoon calendar meeting dean next ten war intention compare morally hold
speaking argument weak played ghost script nice touch tired today went visit grandmother turn
""".split())
NAMES = {"monday", "london", "isabel", "bob", "marley", "sally", "arizona", "thursday", "tuesday", "sunday"}
DIC = lexicon.Dictionary(WORDS, NAMES)

LEXICON_TEXT = textwrap.dedent("""\
    # test lexicon
    Claude
    Kolodzin
    Isobel
    Robin
    Marlow
    Haslanger
    Bicchieri      | bikieri
    Clew-boss      | clew boss
    jmarkdown      | j markdown
    TeX+           | tech plus, tex plus
    ghostscript    | ghost script
    Hacking
    lamprey        -> line break
    idiomin        -> Idi Amin
    clue boss      -> Clew-boss
    j. markdown    -> jmarkdown
    tech plus      -> TeX+
    robbing        -> Robin
    culdran        -> Kolodzin
    kubla          -> Kolodzin ?
    god            -> Claude @calling
    todd           -> Claude @calling
    gaud           -> Claude @calling
""")


def make_lexicon(tmp: Path, folders=(), text=LEXICON_TEXT) -> lexicon.Lexicon:
    (tmp / "lexicon.txt").write_text(text)
    dirs = []
    if folders:
        d = tmp / "folders"
        for name in folders:
            (d / name).mkdir(parents=True, exist_ok=True)
        dirs = [d]
    return lexicon.Lexicon(tmp / "lexicon.txt", dirs, dictionary_=DIC)


def heard(text: str, confs=None, default=0.99) -> list[HeardWord]:
    """The transcriber's words for text: confs maps a word (as written) to its confidence."""
    confs = confs or {}
    return [HeardWord(w, confs.get(w.strip(",.?!"), default)) for w in text.split()]


def body(typed: str) -> str:
    """What follows the label."""
    return LABEL.sub("", typed)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)


# ---------------------------------------------------------------------------
# The label: format, plain(), for_speech(), and that it can't be spoken into existence
# ---------------------------------------------------------------------------

class LabelTest(unittest.TestCase):
    def test_label_says_spoken_and_when(self):
        when = time.mktime((2026, 11, 16, 21, 14, 30, 0, 0, -1))
        self.assertEqual(spoken.label(when), "[spoken 21:14]")
        self.assertEqual(spoken.spoken_at("[spoken 21:14] Claude, hello."), "21:14")

    def test_label_with_interruption_is_short_and_bracket_free(self):
        when = time.mktime((2026, 11, 16, 9, 5, 0, 0, 0, -1))
        text = 'The second section argues [roughly] that "norms" emerge from repeated play, and the third ' \
               'shows how quickly they spread through a population of agents.'
        lab = spoken.label(when, text)
        self.assertTrue(lab.startswith('[spoken 09:05; interrupted the voice during "The second section argues'))
        self.assertTrue(lab.endswith('…"]'))
        inner = lab[1:-1]
        self.assertNotIn("[", inner)
        self.assertNotIn("]", inner)
        self.assertEqual(inner.count('"'), 2)
        self.assertLess(len(lab), 140)

    def test_label_says_who_when_not_the_user(self):
        when = time.mktime((2026, 11, 26, 11, 29, 0, 0, 0, -1))
        self.assertEqual(spoken.label(when, speaker="Robin"), "[spoken 11:29; by Robin]")
        both = spoken.label(when, "Stop there.", speaker="Robin?")
        self.assertEqual(both, '[spoken 11:29; by Robin?; interrupted the voice during "Stop there."]')
        self.assertEqual(spoken.speaker(both + " Claude, wait."), "Robin?")
        self.assertEqual(spoken.speaker("[spoken 11:29; by Robin and the user] Claude, go."), "Robin and the user")
        self.assertIsNone(spoken.speaker("[spoken 11:29] Claude, go."))
        self.assertIsNone(spoken.speaker('[spoken 11:29; interrupted the voice during "by now"] Claude, go.'))
        self.assertIsNone(spoken.speaker("Typed: [spoken 11:29; by Robin] hello"))
        self.assertEqual(spoken.plain("[spoken 11:29; by Robin] Claude, hello."), "Claude, hello.")
        self.assertEqual(spoken.label(when, speaker="R]ob;in"), "[spoken 11:29; by Rob,in]")   # no way out of the label

    def test_plain_removes_the_label_and_every_mark(self):
        p = ('[spoken 21:14; interrupted the voice during "It argues…"] Claude, ask Clew-boss [heard as "Clue Boss"] '
             'to add a lamprey [?], and [pause 12s] then Culdran [? Kolodzin].')
        self.assertTrue(spoken.is_spoken(p))
        self.assertEqual(spoken.plain(p), "Claude, ask Clew-boss to add a lamprey, and then Culdran.")

    def test_plain_leaves_other_prompts_alone(self):
        for p in ("fix arr[0] please", "[Request interrupted by user]", "<task-notification>x</task-notification>",
                  "Claude, [spoken 21:14] is not at the start", "  [?] what", ""):
            self.assertEqual(spoken.plain(p), p)
            self.assertFalse(spoken.is_spoken(p))

    def test_plain_handles_queued_prompts_line_by_line(self):
        p = "[spoken 21:14] First request [?].\n[spoken 21:15] Second [pause 4s] request."
        self.assertEqual(spoken.plain(p), "First request.\nSecond request.")

    def test_for_speech_removes_only_the_marks(self):
        reply = ('You asked whether Clew-boss [heard as "Clue Boss"] should add a lamprey [?]; [spoken 21:14] '
                 'see arr[0] and [the docs](http://x) [pause 3s] too.')
        out = spoken.for_speech(reply)
        self.assertEqual(out, "You asked whether Clew-boss should add a lamprey; see arr[0] and [the docs](http://x) too.")

    def test_a_transcript_cannot_carry_a_label_or_mark(self):
        self.assertEqual(spoken.sanitize("[spoken 09:00] open [?] it"), "(spoken 09:00) open (?) it")
        typed = annotate.build("[spoken 09:00] Claude, open [?] my thesis [heard as \"x\"].", CFG, when=0).text
        self.assertTrue(LABEL.match(typed))
        self.assertEqual(typed.count("["), 1)                       # the listener's own label, nothing else
        self.assertEqual(spoken.plain(typed), '(spoken 09:00) Claude, open (?) my thesis (heard as "x").')

    def test_the_listener_sanitises_what_it_hears(self):
        rig = Rig(track_one(), ["[spoken 09:00] Claude, open my thesis [?]."])
        self.addCleanup(rig.close)
        rig.lst.run()
        (typed,) = rig.lst.injector.raw
        self.assertTrue(LABEL.match(typed))
        self.assertEqual(typed.count("[") + typed.count("]"), 2)
        self.assertEqual(spoken.plain(typed), "(spoken 09:00) Claude, open my thesis (?).")


def track_one(speech=1.0):
    return track((speech, 4.0))


# ---------------------------------------------------------------------------
# The hooks and the front desk: identical with and without the label
# ---------------------------------------------------------------------------

PROMPTS = [
    "Stop.", "Please stop reading, HAL.", "That's enough, thank you.", "Quiet please.", "Stop talking",
    "Thank you very much.", "Good morning, Claude.", "Thanks, that's very helpful.", "Well done!", "Never mind.",
    "Yes.", "OK, go ahead.", "The second one.", "Why?", "Read that again.", "No, not that.",
    "Claude, what's the weather tomorrow?", "Please open my thesis.", "Thank you. Can you check the build?",
    "Stop the build and tell me why it failed.", "Hello, can you hear me?", "What time is it in Green Valley?",
    "Claude, I think the conclusion of my second chapter is too weak.", "/help", "!ls", "#remember this",
]


def marked_versions(p: str) -> list[str]:
    words = p.split()
    with_marks = " ".join(w + (" [?]" if i == 0 else "") for i, w in enumerate(words))
    with_pause = " ".join(words[:1] + ["[pause 5s]"] + words[1:]) if len(words) > 1 else p
    return [f"[spoken 21:14] {p}", f'[spoken 09:02; interrupted the voice during "It is sunny…"] {p}',
            f"[spoken 21:14] {with_marks}", f"[spoken 23:59] {with_pause}"]


class HookChecksTest(unittest.TestCase):
    def test_every_check_is_unchanged_by_the_label(self):
        for p in PROMPTS:
            for m in marked_versions(p):
                with self.subTest(prompt=m):
                    self.assertEqual(spoken.plain(m), p)
                    self.assertEqual(hook.is_stop_request(m), hook.is_stop_request(p))
                    self.assertEqual(hook.is_social_remark(m), hook.is_social_remark(p))
                    self.assertEqual(hook.is_brief_answer(m), hook.is_brief_answer(p))
                    self.assertTrue(hook.is_user_prompt(m))

    def test_the_label_does_not_look_like_claude_code(self):
        self.assertFalse(hook.is_user_prompt("[Request interrupted by user]"))
        self.assertFalse(hook.is_user_prompt("<task-notification>done</task-notification>"))
        self.assertTrue(hook.is_user_prompt("[spoken 21:14] Claude, hello."))
        self.assertFalse(hook.is_user_prompt("[spoken 21:14] "))

    def test_known_answers_still_hold(self):
        self.assertTrue(hook.is_stop_request("[spoken 21:14] Stop, Claude [heard as \"Todd\"]."))
        self.assertTrue(hook.is_social_remark("[spoken 21:14] Thank you [?] very much."))
        self.assertTrue(hook.is_brief_answer("[spoken 21:14] Yes, go ahead."))
        self.assertFalse(hook.is_brief_answer("[spoken 21:14] Please open my thesis."))


class HookCommandTest(HookTestBase):
    """Through the real hook command, as Claude Code runs it."""

    def prompt(self, text, stdout=b""):
        return self.run_hook(self.prompt_payload(text), event="UserPromptSubmit", stdout=stdout)

    def log_lines(self):
        return [line for line in (self.tmp / "state" / "hal.log").read_text().splitlines()
                if "UserPromptSubmit" in line]

    def test_a_spoken_stop_request_is_a_stop_request(self):
        self.prompt("[spoken 21:14] Stop reading, please.")
        self.prompt("Stop reading, please.")
        spoken_line, typed_line = self.log_lines()
        self.assertIn("stop request", spoken_line)
        self.assertEqual(spoken_line.split("hook: ", 1)[1].split(" [")[0], typed_line.split("hook: ", 1)[1].split(" [")[0])
        self.assertTrue((self.tmp / "state" / "skip_next_reply").exists())

    def test_the_last_prompt_is_saved_without_the_label(self):
        self.prompt('[spoken 21:14] Claude, ask Clew-boss [heard as "Clue Boss"] about the lamprey [?].')
        saved = json.loads((self.tmp / "state" / "last_prompt.json").read_text())
        self.assertEqual(saved["text"], "Claude, ask Clew-boss about the lamprey.")

    def test_a_spoken_thank_you_gets_no_acknowledgement(self):
        self.set_config(acknowledge=True, front_desk=False)
        self.prompt("[spoken 21:14] Thank you very much [?].")
        self.prompt("Thank you very much.")
        a, b = self.log_lines()
        self.assertIn("thanks or a greeting, so no acknowledgement", a)
        self.assertIn("; spoken", a)
        self.assertEqual(a.split("hook: ", 1)[1].split("; spoken")[0], b.split("hook: ", 1)[1].split(" [")[0])

    def test_a_spoken_brief_answer_goes_straight_to_claude(self):
        self.set_config(acknowledge=True, front_desk=True)
        self.prompt("[spoken 21:14] Yes, go ahead.")
        (line,) = self.log_lines()
        self.assertIn("brief answer: straight to Claude", line)


class DeskSpokenTest(HookTestBase):
    """The front desk (with the stand-in model) answers a spoken remark as it answers the words."""

    def setUp(self):
        super().setUp()
        self.env["HAL_WARM_MODEL"] = f"{sys.executable} {ROOT / 'tests' / 'fake_warm_model.py'}"
        self.set_config(acknowledge=True, front_desk=True, desk_wait_seconds=1.2, user_name="Dr. Alexander",
                        ack_phrases=["One moment, please."], desk_idle_minutes=5, desk_block=True)

    def tearDown(self):
        subprocess.run([str(ROOT / "bin" / "hal"), "desk", "stop"], env=self.env, capture_output=True, timeout=30)
        super().tearDown()

    def model_calls(self, role):
        try:
            lines = (self.out / "warm_model.log").read_text().splitlines()
        except FileNotFoundError:
            return []
        return [line.split(" ", 2)[2] for line in lines if line.split(" ", 1)[0] == role]

    def test_social_remark_and_turn_judgement(self):
        r = subprocess.run([str(ROOT / "bin" / "hal"), "desk", "start"], env=self.env, capture_output=True,
                           text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        answers = []
        for p in ("[spoken 21:14] Good morning, Claude.", "Good morning, Claude."):
            self.run_hook(self.prompt_payload(p), event="UserPromptSubmit", stdout=None)
            answers.append(json.loads(self.last_stdout))
        self.assertEqual(answers[0], answers[1])
        self.assertEqual(answers[0]["decision"], "block")
        calls = self.model_calls("desk")
        self.assertEqual(calls[:2], ["Good morning, Claude.", "Good morning, Claude."])
        sys.path.insert(0, str(ROOT))
        from hal import desk

        with mock.patch.dict(os.environ, {"HAL_STATE_DIR": self.env["HAL_STATE_DIR"]}):
            got = desk.judge_turn("[spoken 21:14] I was thinking about the unfinished", 4.0,
                                  state_dir=self.env["HAL_STATE_DIR"])
        self.assertEqual(got.get("kind"), "incomplete")
        (turn,) = self.model_calls("turn")
        self.assertNotIn("spoken", turn)
        self.assertIn("I was thinking about the unfinished", turn)


class SpeechTest(unittest.TestCase):
    """The voice never reads a label or mark aloud, even when a reply quotes one."""

    def test_quoted_marks_are_not_spoken(self):
        reply = ('You said "[spoken 21:14] Claude, ask Clew-boss [heard as "Clue Boss"] about the lamprey [?]", '
                 'and after [pause 12s] you asked about Culdran [? Kolodzin].\n\nDetails follow.')
        said = speakable(reply, 0, "")
        for bit in ("spoken", "heard as", "[", "]", "pause 12", "?]"):
            self.assertNotIn(bit, said)
        self.assertIn("Clew-boss", said)

    def test_the_lead_paragraph_path_too(self):
        from hal import speaker

        reply = ("Kolodzin [heard as \"Culdran\"] is free on Tuesday [?]. " * 2).strip() + "\n\n" + "More. " * 120
        text, how = speaker.prepare_reply(reply, dict(spoken_reply="lead", summary_min_chars=350,
                                                      lead_max_chars=450, max_chars=0, overflow_message=""))
        self.assertEqual(how, "lead")
        self.assertNotIn("heard", text)
        self.assertNotIn("[", text)


# ---------------------------------------------------------------------------
# The lexicon
# ---------------------------------------------------------------------------

class LexiconTest(TmpCase):
    def test_file_entries_spoken_forms_and_aliases(self):
        lex = make_lexicon(self.tmp)
        names = {e.name for e in lex.entries}
        self.assertLessEqual({"Kolodzin", "Clew-boss", "TeX+", "line break", "Idi Amin"}, names)
        modes = {(a.heard, a.target, a.mode) for a in lex.aliases}
        self.assertIn((("clue", "boss"), "Clew-boss", "always"), modes)
        self.assertIn((("j", "markdown"), "jmarkdown", "always"), modes)
        self.assertIn((("kubla",), "Kolodzin", "flag"), modes)
        self.assertEqual(lex.name_aliases(), ([], ["god", "todd", "gaud"]))
        self.assertIn("lexicon.txt", lex.describe())

    def test_folders_are_names_and_the_file_is_reread(self):
        lex = make_lexicon(self.tmp, folders=["Haslanger", "Gödel", "code", ".hidden"])
        names = {e.name for e in lex.entries}
        self.assertIn("Gödel", names)
        self.assertNotIn(".hidden", names)
        self.assertTrue(lex.known("Godel"))                       # accents folded
        (self.tmp / "lexicon.txt").write_text(LEXICON_TEXT + "Pemberton\nmarlo -> Marlow\n")
        os.utime(self.tmp / "lexicon.txt", (time.time() + 5, time.time() + 5))
        self.assertTrue(lex.refresh())
        self.assertIn("Pemberton", {e.name for e in lex.entries})
        self.assertIn((("marlo",), "Marlow", "always"), {(a.heard, a.target, a.mode) for a in lex.aliases})
        self.assertFalse(lex.refresh())                              # unchanged

    def test_pronunciation_keys(self):
        same = [("Clue Boss", "Clew-boss"), ("ClueBoss", "Clew-boss"), ("Ediamine", "Idi Amin"),
                ("Isabel", "Isobel"), ("Kolodzen", "Kolodzin"), ("Gomez", "Gómez"), ("Becari", "Bicchieri"),
                ("J. Markdown", "j markdown"), ("Has Langer", "Haslanger")]
        for a, b in same:
            with self.subTest(a=a, b=b):
                self.assertLessEqual(lexicon.distance(lexicon.phonetic(a), lexicon.phonetic(b)), 0.35)
        for a, b in [("Marley", "Matthews"), ("lamprey", "line break"), ("Quidzen", "Kolodzin"), ("Gould", "Could you")]:
            with self.subTest(a=a, b=b):
                self.assertGreater(lexicon.distance(lexicon.phonetic(a), lexicon.phonetic(b)), 0.65)

    def test_dictionary(self):
        self.assertTrue(DIC.ordinary("hacking"))
        self.assertTrue(DIC.ordinary("Robbing"))
        self.assertTrue(DIC.ordinary("clues"))
        self.assertFalse(DIC.ordinary("Kolodzen"))
        self.assertTrue(DIC.known("Monday"))
        self.assertFalse(DIC.ordinary("Monday"))

    def test_load_follows_the_settings(self):
        self.assertIsNone(lexicon.load(dict(CFG, lexicon=False)))
        self.assertIsNone(lexicon.load(dict(CFG, spoken_label=False)))
        (self.tmp / "lex.txt").write_text("Kolodzin\n")
        lex = lexicon.load(dict(CFG, lexicon_file=str(self.tmp / "lex.txt"), lexicon_dirs=[]), dictionary_=DIC)
        self.assertEqual([e.name for e in lex.entries], ["Kolodzin"])


class CorrectionTest(TmpCase):
    def setUp(self):
        super().setUp()
        self.lex = make_lexicon(self.tmp, folders=["Gould", "Held", "Thomas", "Fodor", "Mundy", "Keren"])
        self.cfg = dict(CFG)
        self.cfg["alias_names"], self.cfg["calling_names"] = self.lex.name_aliases()

    def typed(self, text, confs=None, default=0.99, **cfg):
        cfg = dict(self.cfg, **cfg)
        m = addressing.match(text, cfg)
        out = annotate.build(m.text if m.addressed else text, cfg, heard=heard(text, confs, default), lex=self.lex,
                             when=0)
        return body(out.text)

    def test_every_observed_mistake_is_corrected_and_shown(self):
        cases = [
            ("Claude, add a lamprey after the heading.", 'Claude, add a line break [heard as "lamprey"] after'),
            ("What did idiomin do?", 'What did Idi Amin [heard as "idiomin"] do?'),
            ("Ask Clue Boss to check the build.", 'Ask Clew-boss [heard as "Clue Boss"] to check'),
            ("Open the J. Markdown repository.", 'Open the jmarkdown [heard as "J. Markdown"] repository.'),
            ("Ask the Tech Plus agent to build.", 'Ask the TeX+ [heard as "Tech Plus"] agent'),
            ("Send it to robbing.", 'Send it to Robin [heard as "robbing"].'),
            ("What does Culdran think?", 'What does Kolodzin [heard as "Culdran"] think?'),
            ("God, what's the time?", 'Claude [heard as "God"], what\'s the time?'),
            ("Todd, open my thesis.", 'Claude [heard as "Todd"], open my thesis.'),
            ("Good morning, Gaud.", 'Good morning, Claude [heard as "Gaud"].'),
        ]
        for said, expected in cases:
            with self.subTest(said=said):
                self.assertIn(expected, self.typed(said))

    def test_a_flag_only_alias_changes_nothing(self):
        self.assertEqual(self.typed("Ask Kubla about it."), "Ask Kubla [? Kolodzin] about it.")

    def test_names_that_sound_alike(self):
        # Unfamiliar words the transcriber was unsure of: changed. Several words mangled into a name: changed.
        self.assertIn('Kolodzin [heard as "Kolodzen"]', self.typed("What does Kolodzen think?", {"Kolodzen": 0.76}))
        self.assertIn('Clew-boss [heard as "Kluboss"]', self.typed("Ask Kluboss now.", {"Kluboss": 0.87}))
        self.assertIn('Clew-boss\'s [heard as "Clue Boss\'s"] turn', self.typed("It is Clue Boss's turn."))
        self.assertIn('Bicchieri [heard as "Becari"]', self.typed("Did Becari publish?", {"Becari": 0.79}))
        self.assertIn('Idi Amin [heard as "Edi Amin"]', self.typed("What did Edi Amin say?", {"Edi": 0.82}))
        self.assertIn('ghostscript [heard as "ghost script"]', self.typed("Update ghost script now."))
        # Heard with confidence, or a known name: only flagged.
        self.assertIn("Haslinger [? Haslanger]", self.typed("Ask Haslinger now.", {"Haslinger": 0.97}))
        self.assertIn("Marley [? Marlow]", self.typed("Bob Marley played on Monday."))
        self.assertIn("Isabel [? Isobel]", self.typed("Isabel sent the minutes."))
        # Unsure of a longer ordinary word that sounds just like one of his names: flagged.
        self.assertIn("Marlows [? Marlow]", self.typed("Ask the Marlows now.", {"Marlows": 0.6}))

    def test_ordinary_words_are_left_alone(self):
        for said in ("I keep my files in cloud storage now.", "They were hacking the server again.",
                     "Morally speaking, the argument is weak.", "Go to the shop and buy some milk.",
                     "Give me a clue about the boss.", "I need a clue, boss, about the plan.",
                     "Add a line break after the heading.", "The ghost in the script was a nice touch.",
                     "Hacking argued that probability matters, said Marlow.",
                     "Todd and I went to the library.", "Thank God it's Friday.", "Oh God, the build failed."):
            with self.subTest(said=said):
                self.assertEqual(self.typed(said), said)

    def test_an_exclamation_in_the_calling_position_is_taken_as_the_name(self):
        # The known cost of "God" in calling_names: shown, so the model can tell.
        self.assertEqual(self.typed("God, I'm tired today."), 'Claude [heard as "God"], I\'m tired today.')

    def test_cloud_is_never_claude_outside_the_calling_position(self):
        self.assertEqual(self.typed("Put the files in the cloud storage."), "Put the files in the cloud storage.")

    def test_folder_names_only_where_the_transcriber_was_unsure(self):
        # "Could" (unsure) is an ordinary word: never "Gould". "Greene"-like unfamiliar names heard with
        # confidence are left alone; unsure ones may be flagged with a folder name.
        self.assertEqual(self.typed("Hold the door.", {"Hold": 0.6}), "Hold [?] the door.")      # not "Held"
        self.assertEqual(self.typed("Ask Kerene about it."), "Ask Kerene about it.")
        self.assertIn('Keren [heard as "Kerene"]', self.typed("Ask Kerene about it.", {"Kerene": 0.7}))    # same sound
        self.assertIn("Vodor [? Fodor]", self.typed("Ask Vodor about it.", {"Vodor": 0.7}))     # close, not the same

    def test_god_todd_gaud_only_when_calling(self):
        cfg = self.cfg
        for t, addressed in [("God, what's the time?", True), ("Good morning, God.", True), ("Thanks, God.", True),
                             ("Gaud, open my thesis.", True), ("Todd, can you open the draft?", True),
                             ("Oh God.", False), ("Thank God.", False), ("My God!", False), ("God.", False),
                             ("Oh, God.", False), ("Good God, that's awful.", False), ("God knows.", False),
                             ("Todd and I went out.", False), ("For God's sake.", False), ("In God we trust.", False)]:
            with self.subTest(t=t):
                self.assertEqual(addressing.match(t, cfg).addressed, addressed)
        self.assertFalse(addressing.match("God, what's the time?", dict(CFG)).addressed)   # without the lexicon

    def test_lexicon_off(self):
        out = self.typed("Ask Clue Boss about the lamprey.", lexicon=False)
        self.assertEqual(out, "Ask Clue Boss about the lamprey.")


# ---------------------------------------------------------------------------
# Doubtful words, pauses, the interruption, the settings
# ---------------------------------------------------------------------------

class MarksTest(unittest.TestCase):
    def build(self, text, confs=None, **kw):
        cfg = dict(CFG, **kw.pop("cfg", {}))
        return annotate.build(text, cfg, heard=heard(text, confs), when=0, **kw)

    def test_a_clear_sentence_has_no_marks(self):
        out = self.build("Claude, could you open the draft of my paper on the evolution of social norms?")
        self.assertEqual(body(out.text), "Claude, could you open the draft of my paper on the evolution of social norms?")

    def test_doubtful_words_at_most_two_a_sentence(self):
        out = self.build("Please put the meeting with the dean in my calendar.",
                         {"put": 0.7, "meeting": 0.5, "dean": 0.6, "calendar": 0.75, "the": 0.3, "with": 0.4})
        self.assertEqual(body(out.text), "Please put the meeting [?] with the dean [?] in my calendar.")
        self.assertEqual(out.doubts, ["meeting", "dean"])

    def test_each_sentence_gets_its_own_two(self):
        out = self.build("Open the lamp file. Then read the forks again.",
                         {"lamp": 0.5, "file": 0.6, "forks": 0.7, "read": 0.8, "again": 0.9})
        self.assertEqual(body(out.text), "Open the lamp [?] file [?]. Then read [?] the forks [?] again.")

    def test_threshold_and_never_function_words_or_the_name(self):
        out = self.build("Claude, is it on the shelf?", {"Claude": 0.5, "is": 0.5, "it": 0.5, "on": 0.5, "shelf": 0.83})
        self.assertEqual(body(out.text), "Claude, is it on the shelf?")
        self.assertIn("shelf [?]", self.build("Claude, is it on the shelf?", {"shelf": 0.81}).text)

    def test_unknown_confidence_marks_nothing(self):
        out = annotate.build("Open the lamp file.", CFG, heard=[HeardWord(w) for w in "Open the lamp file.".split()],
                             when=0)
        self.assertEqual(body(out.text), "Open the lamp file.")

    def test_pauses_between_fragments(self):
        text = "Claude, compare the two drafts and then send the better one to my editor."
        words = heard(text)
        for w in words[:6]:
            w.frag = 0
        for w in words[6:10]:
            w.frag = 1
        for w in words[10:]:
            w.frag = 2
        frags = [(0.0, 3.0), (5.0, 7.0), (19.0, 21.0)]   # gaps of 2 s and 12 s
        out = annotate.build(text, CFG, heard=words, fragments=frags, when=0)
        self.assertEqual(body(out.text),
                         "Claude, compare the two drafts and then send the better [pause 12s] one to my editor.")
        self.assertEqual(out.pauses, [12.0])
        out = annotate.build(text, dict(CFG, pause_mark_seconds=1.5), heard=words, fragments=frags, when=0)
        self.assertEqual(out.text.count("[pause"), 2)

    def test_interruption_in_the_label(self):
        out = self.build("Stop and tell me about apples.", interrupted="The second sentence is about apples.")
        self.assertTrue(out.text.startswith('[spoken ') and
                        '; interrupted the voice during "The second sentence is about apples."] Stop' in out.text)
        self.assertTrue(out.interrupted)

    def test_each_part_can_be_turned_off(self):
        text = "Open the lamp file."
        confs = {"lamp": 0.5}
        frag_words = heard(text, confs)
        for i, w in enumerate(frag_words):
            w.frag = 0 if i < 2 else 1
        frags = [(0.0, 1.0), (6.0, 7.0)]
        full = annotate.build(text, CFG, heard=frag_words, fragments=frags, when=0, interrupted="It was saying this.")
        self.assertIn("[?]", full.text)
        self.assertIn("[pause 5s]", full.text)
        self.assertIn("interrupted the voice", full.text)
        for key, gone in (("doubt_marks", "[?]"), ("pause_marks", "[pause"), ("interruption_note", "interrupted")):
            out = annotate.build(text, dict(CFG, **{key: False}), heard=frag_words, fragments=frags, when=0,
                                 interrupted="It was saying this.")
            self.assertNotIn(gone, out.text, key)
            self.assertTrue(out.text.startswith("[spoken "))
        off = annotate.build(text, dict(CFG, spoken_label=False), heard=frag_words, fragments=frags, when=0,
                             interrupted="It was saying this.")
        self.assertEqual(off.text, text)

    def test_settings_parse(self):
        tmp = make_tmp()
        self.addCleanup(rmtree, tmp)
        path = tmp / "c.json"
        path.write_text(json.dumps({"listener": {
            "spoken_label": False, "lexicon": "yes", "lexicon_dirs": ["~/Source/Papers", "/Users/X/Books"],
            "doubt_threshold": 0.7, "pause_mark_seconds": -1, "doubt_marks": False, "pause_marks": False,
            "interruption_note": False, "lexicon_file": "names.txt"}}))
        cfg = settings.load(path)
        self.assertEqual((cfg["spoken_label"], cfg["lexicon"], cfg["doubt_marks"], cfg["pause_marks"],
                          cfg["interruption_note"]), (False, True, False, False, False))
        self.assertEqual(cfg["lexicon_dirs"], ["~/Source/Papers", "/Users/X/Books"])     # not lower-cased
        self.assertEqual((cfg["doubt_threshold"], cfg["pause_mark_seconds"], cfg["lexicon_file"]), (0.7, 3.0, "names.txt"))
        for key in ("spoken_label", "lexicon", "doubt_marks", "pause_marks", "interruption_note"):
            self.assertIs(settings.DEFAULTS[key], True)


class TurnWordsTest(unittest.TestCase):
    """The words of a turn transcribed afresh keep their fragments."""

    def test_rejoined_words_know_their_fragment(self):
        def rejoin(parts):
            words = [HeardWord("Claude,", 0.99, part=0), HeardWord("compare", 0.99, part=0),
                     HeardWord("the", 0.99, part=1), HeardWord("drafts.", 0.6, part=2)]
            return "Claude, compare the drafts.", words

        tt = turns.TurnTaker(dict(CFG), rejoin=rejoin, clock=lambda: 0.0)
        tt.add("Claude, compare", how="named", speech_start=0.0, speech_end=1.0, audio=[[0.0] * 800])
        tt.add("the", how="continued", speech_start=6.0, speech_end=6.5, audio=[[0.0] * 800, [0.0] * 800])
        words = tt.heard_words(tt.turn)
        self.assertEqual([w.frag for w in words], [0, 0, 1, 1])

    def test_fragments_without_words_still_line_up(self):
        tt = turns.TurnTaker(dict(CFG, turn_retranscribe=False), clock=lambda: 0.0)
        tt.add("Claude, compare the", how="named", speech_start=0.0, speech_end=1.0,
               words=[HeardWord("Claude,", 0.9), HeardWord("compare", 0.5), HeardWord("the", 0.9)])
        tt.add("drafts.", how="continued", speech_start=6.0, speech_end=6.5)
        words = tt.heard_words(tt.turn)
        self.assertEqual([(w.text, w.conf, w.frag) for w in words],
                         [("Claude,", 0.9, 0), ("compare", 0.5, 0), ("the", 0.9, 0), ("drafts.", None, 1)])


# ---------------------------------------------------------------------------
# The listening loop
# ---------------------------------------------------------------------------

class WordsASR:
    """Transcripts with confidences: items are (text, {word: confidence})."""
    name = "fake-words"

    def __init__(self, items):
        self.items = list(items)

    def transcribe(self, audio):
        return self.transcribe_words(audio)[0]

    def transcribe_words(self, audio):
        if not self.items:
            return "", []
        text, confs = self.items.pop(0)
        return text, heard(text, confs)


class ListenerMarksTest(TmpCase):
    def rig(self, pieces, items, lex=None, **cfg):
        rig = Rig(track(*pieces), [], asr=WordsASR(items), **cfg)
        self.addCleanup(rig.close)
        if lex is not None:
            rig.lst.lexicon = lex
            rig.lst._lexicon_names()
        rig.lst.run()
        return rig

    def test_typed_with_the_label_at_the_time_speech_began(self):
        before = time.time()
        rig = self.rig([(1.0, 4.0)], [("Claude, open my thesis.", {})])
        after = time.time()
        (typed,) = rig.lst.injector.raw
        self.assertIn(typed, {f"{spoken.label(t)} Claude, open my thesis." for t in (before - 10, before, after)})

    def test_a_long_pause_is_marked_where_it_happened(self):
        rig = self.rig([(2.0, 5.0), (1.5, 4.0)], [("Claude, compare the two drafts and", {}),
                                                  ("send the better one to my editor.", {"editor": 0.5})])
        (typed,) = rig.lst.injector.raw
        self.assertEqual(body(typed), "Claude, compare the two drafts and [pause 5s] send the better one to my "
                                      "editor [?].")
        sent = next(e for e in rig.lst.events if e["action"] == "submit")
        self.assertEqual(sent["plain"], "Claude, compare the two drafts and send the better one to my editor.")
        self.assertEqual(rig.lst.counts.doubtful, 1)

    def test_a_short_pause_is_not(self):
        rig = self.rig([(2.0, 2.0), (1.5, 4.0)], [("Claude, compare the two drafts and", {}),
                                                  ("send the better one to my editor.", {})])
        self.assertNotIn("[pause", rig.lst.injector.raw[0])

    def test_corrections_through_the_loop(self):
        lex = make_lexicon(self.tmp)
        rig = self.rig([(2.0, 4.0), (1.5, 4.0)],
                       [("God, ask Clue Boss about the lamprey.", {}), ("Oh God, the build failed.", {})], lex=lex)
        self.assertEqual([body(t) for t in rig.lst.injector.raw],
                         ['Claude [heard as "God"], ask Clew-boss [heard as "Clue Boss"] about the line break '
                          '[heard as "lamprey"].'])
        self.assertEqual(rig.lst.counts.corrected, 3)
        self.assertEqual(rig.lst.counts.ignored, 1)                  # "Oh God, ..." isn't addressed

    def test_silence_mode_is_labelled_too(self):
        rig = self.rig([(1.0, 4.0)], [("Claude, open the lamp file.", {"lamp": 0.5})], turn_taking="silence")
        (typed,) = rig.lst.injector.raw
        self.assertTrue(LABEL.match(typed))
        self.assertIn("lamp [?]", typed)

    def test_label_off_types_exactly_as_before(self):
        lex = make_lexicon(self.tmp)
        rig = self.rig([(2.0, 5.0), (1.5, 4.0)], [("Claude, ask Clue Boss and", {"Boss": 0.3}),
                                                  ("the lamprey.", {})], lex=lex, spoken_label=False)
        self.assertEqual(rig.lst.injector.raw, ["Claude, ask Clue Boss and the lamprey."])

    def test_privacy_unaddressed_speech_is_never_marked_or_written(self):
        lex = make_lexicon(self.tmp)
        rig = self.rig([(1.5, 4.0), (1.0, 4.0)],
                       [("The lamprey Kolodzen told Clue Boss my PIN is 4321.", {"Kolodzen": 0.5}),
                        ("Thanks, Claude.", {})], lex=lex)
        self.assertEqual([spoken.plain(t) for t in rig.lst.injector.raw], ["Thanks, Claude."])
        self.assertEqual((rig.lst.counts.corrected, rig.lst.counts.flagged, rig.lst.counts.doubtful), (0, 0, 0))
        written = "\n".join(p.read_text(errors="replace") for p in rig.tmp.rglob("*") if p.is_file())
        for word in ("lamprey", "Kolodzen", "Kolodzin", "Clue", "Clew", "4321", "line break"):
            self.assertNotIn(word, written)
        status = json.loads((rig.tmp / "state" / "listen" / "listener.json").read_text())
        self.assertEqual(status["lexicon"], len(lex))

    def test_the_interrupted_sentence_reaches_the_model(self):
        r = DuplexRig([(3.0, 1.0)], ["Open the second chapter of my thesis."])
        self.addCleanup(r.close)
        r.run()
        (typed,) = r.lst.injector.raw
        self.assertIn('; interrupted the voice during "The second sentence is about apples and pears."] Open the '
                      'second chapter of my thesis.', typed)
        r2 = DuplexRig([(3.0, 1.0)], ["Open the second chapter of my thesis."], interruption_note=False)
        self.addCleanup(r2.close)
        r2.run()
        self.assertTrue(LABEL.match(r2.lst.injector.raw[0]))
        self.assertNotIn("interrupted", r2.lst.injector.raw[0])
        self.assertNotIn("apples", (r2.tmp / "state" / "listen.log").read_text().split("heard (")[0])

    def test_a_stop_phrase_over_the_voice_types_nothing(self):
        r = DuplexRig([(3.0, 0.6)], ["That's enough."])
        self.addCleanup(r.close)
        r.run()
        self.assertEqual(r.lst.injector.raw, [])


# ---------------------------------------------------------------------------
# The real Parakeet on synthesized speech (files only)
# ---------------------------------------------------------------------------

def _asr_available() -> bool:
    return (LISTEN_PY.exists() and (ROOT / "models" / "parakeet-tdt-0.6b-v2").is_dir()
            and shutil.which("say") is not None and shutil.which("ffmpeg") is not None)


ASR_SCRIPT = r'''
import json, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, sys.argv[1])
from hal.listener.asr import ParakeetASR
from hal.listener import settings, lexicon, annotate, addressing

def synth(text, voice, rate=None, noise=0.0):
    d = tempfile.mkdtemp()
    cmd = ["say", "-v", voice, "-o", f"{d}/a.aiff"] + (["-r", str(rate)] if rate else [])
    subprocess.run(cmd + [text], check=True, timeout=60)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{d}/a.aiff", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    if noise:
        a = a + np.random.default_rng(0).normal(0, noise, len(a)).astype(np.float32)
    return a

cfg = dict(settings.DEFAULTS)
asr = ParakeetASR(cfg["asr_model"])
asr.warmup()
lex = lexicon.Lexicon(sys.argv[2], [])
cfg["alias_names"], cfg["calling_names"] = lex.name_aliases()
out = []
for group, voice, rate, noise, text in json.loads(sys.argv[3]):
    said, words = asr.transcribe_words(synth(text, voice, rate, noise))
    m = addressing.match(said, cfg)
    typed = annotate.build(m.text if m.addressed else said, cfg, heard=words, lex=lex, when=0)
    out.append({"group": group, "text": text, "heard": said, "typed": typed.text, "doubts": typed.doubts,
                "corrections": typed.corrections, "flags": typed.flags,
                "confs": [w.conf for w in words]})
print(json.dumps(out))
'''

CLEAR = ["Claude, what's the weather tomorrow?", "Please explain that to me in more detail.",
         "Claude, remind me to call my sister on Thursday afternoon.",
         "Can you please bring the terminal window up to the front for me?"]
HARD = ["Could you check whether the tests pass after that change?",
        "Please put the meeting with the dean in my calendar for next Tuesday at ten.",
        "I think the conclusion of my second chapter is too weak."]


@unittest.skipUnless(_asr_available(), "needs .venv-listen, the Parakeet model, say and ffmpeg")
class AsrMarksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_tmp()
        (cls.tmp / "lexicon.txt").write_text((ROOT / "lexicon.txt").read_text())
        cases = ([("clear", v, None, 0.0, t) for t in CLEAR for v in ("Samantha", "Daniel")]
                 + [("hard", v, r, n, t) for t in HARD for v, r, n in (("Ralph", 240, 0.0), ("Junior", 250, 0.0),
                                                                        ("Daniel", 320, 0.0))]
                 + [("names", "Samantha", None, 0.0, "Claude, what does Kolodzin think about the plan?"),
                    ("names", "Daniel", None, 0.0, "Claude, please ask Clew-boss to add a line break."),
                    ("names", "Samantha", None, 0.0, "Check the mp-tikz-wasm build for ghostscript errors.")])
        r = subprocess.run([str(LISTEN_PY), "-c", ASR_SCRIPT, str(ROOT), str(cls.tmp / "lexicon.txt"),
                            json.dumps(cases)], capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, HF_HUB_OFFLINE="1", HAL_STATE_DIR=str(cls.tmp / "state")))
        if r.returncode != 0:
            raise AssertionError(r.stderr[-3000:])
        cls.results = json.loads(r.stdout.strip().splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp)

    def group(self, name):
        return [x for x in self.results if x["group"] == name]

    def test_confidences_are_per_word(self):
        for x in self.results:
            self.assertEqual(len(x["confs"]), len(x["heard"].split()))
            self.assertTrue(all(c is None or 0.0 <= c <= 1.0 for c in x["confs"]))

    def test_clear_speech_is_not_marked(self):
        for x in self.group("clear"):
            with self.subTest(text=x["text"]):
                self.assertEqual(x["doubts"], [], x["typed"])
                self.assertEqual(x["corrections"] + x["flags"], [], x["typed"])

    def test_hard_speech_is_marked_sparsely(self):
        hard = self.group("hard")
        self.assertGreaterEqual(sum(1 for x in hard if x["doubts"]), len(hard) // 3,
                                "\n".join(x["typed"] for x in hard))
        for x in hard:
            for sentence in re.split(r"(?<=[.?!])\s", body(x["typed"])):
                self.assertLessEqual(sentence.count("[?]"), 2, x["typed"])

    def test_his_names_come_out_right_or_marked(self):
        for x in self.group("names"):
            with self.subTest(text=x["text"]):
                if x["corrections"] or x["flags"] or x["doubts"]:
                    continue
                for name in ("Kolodzin", "Clew-boss", "mp-tikz-wasm"):
                    if name in x["text"]:
                        self.assertIn(name, x["typed"])


@unittest.skipUnless(_asr_available(), "needs .venv-listen, the Parakeet model, say and ffmpeg")
class PipelineFileTest(unittest.TestCase):
    """bin/hal-listen run --file: real VAD, Parakeet, turns and marks on a synthesized recording."""

    def test_labelled_corrected_and_paused(self):
        tmp = make_tmp()
        self.addCleanup(rmtree, tmp)
        import numpy as np
        import wave

        pieces = [("Daniel", "Claude, please ask Clue Boss to add a line break in the manual.", 4.0),
                  ("Daniel", "Claude, I'd like you to compare the two drafts of my paper, and", 6.0),
                  ("Daniel", "then send the better one to my editor.", 5.0)]
        audio = [np.zeros(16000, np.float32)]
        for voice, text, after in pieces:
            subprocess.run(["say", "-v", voice, "-o", str(tmp / "a.aiff"), text], check=True, timeout=60)
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp / "a.aiff"), "-ac", "1",
                                  "-ar", "16000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
            audio += [np.frombuffer(raw, np.float32), np.zeros(int(after * 16000), np.float32)]
        with wave.open(str(tmp / "speech.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(np.concatenate(audio), -1, 1) * 32767).astype(np.int16).tobytes())
        (tmp / "config.json").write_text(json.dumps({"listener": {
            "turn_judge": False, "lexicon_file": str(ROOT / "lexicon.txt"), "lexicon_dirs": []}}))
        env = dict(os.environ, HAL_STATE_DIR=str(tmp / "state"), HAL_CONFIG=str(tmp / "config.json"),
                   HF_HUB_OFFLINE="1")
        r = subprocess.run([str(ROOT / "bin" / "hal-listen"), "run", "--file", str(tmp / "speech.wav"), "--fast",
                            "--dry-run", "--silent", "--hide-ignored", "--report", str(tmp / "report.json")],
                           capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        report = json.loads((tmp / "report.json").read_text())
        typed = [e["text"] for e in report["events"] if e["action"] == "submit"]
        self.assertEqual(len(typed), 2, r.stdout[-3000:])
        for t in typed:
            self.assertTrue(LABEL.match(t), t)
        self.assertIn('Clew-boss [heard as "', typed[0])
        self.assertRegex(typed[1], r"my paper, and \[pause [56]s\] then send the better one to my editor\.")


if __name__ == "__main__":
    unittest.main()
