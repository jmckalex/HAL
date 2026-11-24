"""The addressing rule and the decider (name alone, arming, follow-up window, long utterances)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hal.listener import addressing, settings  # noqa: E402
from hal.listener.decider import Decider  # noqa: E402

CFG = dict(settings.DEFAULTS)

# Transcripts as Parakeet writes them (punctuated, capitalised). Addressed ones
# include the mis-hearings seen on synthetic voices ("Clothes", "Cloat", ...).
ADDRESSED = [
    "Good morning, Claude.",
    "Claude, what's the weather tomorrow?",
    "By the way, Claude, open my thesis.",
    "Claude, what's the weather to morrow?",
    "Clothes, what's the weather tomorrow?",
    "Cloth, what's the weather tomorrow?",
    "Cloak, stop listening.",
    "Thanks, Claude.",
    "Thanks, cloud.",
    "Hey cloud.",
    "Hey, Claude?",
    "Good morning clothes.",
    "Cloud, read me the last paragraph.",
    "By the way, cloud, open my thesis.",
    "What do you think, Claude?",
    "Can you check that, cloud?",
    "Klaud, open the file.",
    "Claud, are you there?",
    "Cloat.",
    "I asked Claude's opinion yesterday.",  # third-person mention: the rule accepts it
    "Okay Claude, summarize the email.",
    "claude what time is it",
]
NOT_ADDRESSED = [
    "The files are stored in the cloud.",
    "The cat clawed at the door.",
    "He is a clod.",
    "It's in the cloud, I think.",
    "Hey, cloud computing is great.",
    "The cloud.",
    "A cloud.",
    "Cloud storage is cheap these days.",
    "Did you back it up to the cloud?",
    "I need to buy some clothes.",
    "That clause in the contract worries me.",
    "The drain is clogged again.",
    "Lord, what a day.",
    "Oh God, I forgot the milk.",
    "Close the door please.",
    "It is cloudy today.",
    "Claudia called this morning.",
    "Put on your cloak, it's cold.",
    "The cloth is on the table.",
    "Is the claw machine broken?",
    "",
]


class AddressingTest(unittest.TestCase):
    def test_addressed(self):
        missed = [t for t in ADDRESSED if not addressing.match(t, CFG).addressed]
        self.assertEqual(missed, [])

    def test_not_addressed(self):
        accepted = [t for t in NOT_ADDRESSED if addressing.match(t, CFG).addressed]
        self.assertEqual(accepted, [])

    def test_whole_words_only(self):
        for text in ("Claudette is here.", "Cloudflare is down.", "Excluded from the list.", "iCloud is full."):
            self.assertFalse(addressing.match(text, CFG).addressed, text)

    def test_variant_rewritten_as_claude(self):
        self.assertEqual(addressing.match("Cloud, what's the weather?", CFG).text, "Claude, what's the weather?")
        self.assertEqual(addressing.match("Klaud's turn.", CFG).text, "Claude's turn.")
        # Only the vocative "cloud" is rewritten, not the one in "the cloud".
        self.assertEqual(addressing.match("Cloud, upload it to the cloud.", CFG).text,
                         "Claude, upload it to the cloud.")

    def test_whole_utterance_kept(self):
        m = addressing.match("By the way, Claude, open my thesis.", CFG)
        self.assertEqual(m.text, "By the way, Claude, open my thesis.")

    def test_name_only(self):
        for text in ("Claude.", "Claude?", "Hey Claude.", "Hey, Claude?", "Okay Claude", "Cloat.", "Hey cloud."):
            self.assertTrue(addressing.match(text, CFG).name_only, text)
        for text in ("Good morning, Claude.", "Hello Claude.", "Thanks, Claude.", "Claude, open it."):
            self.assertFalse(addressing.match(text, CFG).name_only, text)

    def test_stop_listening(self):
        for text in ("Claude, stop listening.", "Claude, please stop listening for now.", "Stop listening, Claude.",
                     "Hey Claude, stop listening please."):
            self.assertEqual(addressing.match(text, CFG).command, "stop_listening", text)
        for text in ("Claude, stop listening to the radio.", "Claude, why did it stop listening?"):
            self.assertIsNone(addressing.match(text, CFG).command, text)

    def test_configurable_variants(self):
        cfg = dict(CFG, names=["hal"], vocative_names=[])
        self.assertTrue(addressing.match("Hal, open the pod bay doors.", cfg).addressed)
        self.assertFalse(addressing.match("Claude, open the pod bay doors.", cfg).addressed)


def utt(start, speech=1.0, cut=False, continued=False):
    return SimpleNamespace(speech_start=start, speech_seconds=speech, duration=speech + 0.6, cut=cut,
                           continued=continued)


class DeciderTest(unittest.TestCase):
    def test_submit_and_ignore(self):
        d = Decider(CFG)
        self.assertEqual(d.decide("Claude, open my thesis.", utt(0), 1).action, "submit")
        self.assertEqual(d.decide("Open my thesis.", utt(5), 6).action, "ignore")
        self.assertEqual(d.decide("", utt(8), 9).action, "empty")

    def test_name_only_arms_next_utterance(self):
        d = Decider(CFG)
        self.assertEqual(d.decide("Claude?", utt(0), 1.0).action, "name_only")
        # Within armed_seconds the next utterance needs no name ...
        r = d.decide("What's the weather tomorrow?", utt(3.0), 4.0)
        self.assertEqual((r.action, r.how, r.text), ("submit", "armed", "What's the weather tomorrow?"))
        # ... once.
        self.assertEqual(d.decide("And the day after?", utt(5.0), 6.0).action, "ignore")

    def test_arming_expires(self):
        d = Decider(CFG)
        d.decide("Claude.", utt(0), 1.0)
        later = 1.0 + CFG["name_only_grace"] + CFG["armed_seconds"] + 0.1
        self.assertEqual(d.decide("What's the weather?", utt(later), later + 1).action, "ignore")

    def test_arm_extended_while_answering(self):
        d = Decider(CFG)
        d.decide("Claude.", utt(0), 1.0)
        d.arm(20.0)   # the listener extends it while "Yes, Dr. Alexander?" plays
        self.assertEqual(d.decide("What's the weather?", utt(19.0), 20.5).action, "submit")

    def test_stop_listening_after_name(self):
        d = Decider(CFG)
        d.decide("Claude.", utt(0), 1.0)
        self.assertEqual(d.decide("Stop listening.", utt(2.0), 3.0).action, "stop_listening")

    def test_follow_up_window(self):
        d = Decider(dict(CFG, follow_up_seconds=5.0))
        r = d.decide("And tomorrow?", utt(13.0), 14.0, reply_end=10.0)
        self.assertEqual((r.action, r.how), ("submit", "follow-up"))
        self.assertEqual(d.decide("And tomorrow?", utt(16.0), 17.0, reply_end=10.0).action, "ignore")

    def test_follow_up_off_by_default(self):
        d = Decider(CFG)
        self.assertEqual(d.decide("And tomorrow?", utt(10.5), 11.0, reply_end=10.0).action, "ignore")

    def test_long_utterance_parts_are_joined(self):
        d = Decider(CFG)
        self.assertEqual(d.decide("Claude, here is a long dictation", utt(0, 30, cut=True), 31).action, "partial")
        self.assertEqual(d.decide("that keeps going", utt(30, 30, cut=True, continued=True), 61).action, "partial")
        r = d.decide("and ends here.", utt(60, 5, continued=True), 66)
        self.assertEqual(r.action, "submit")
        self.assertEqual(r.text, "Claude, here is a long dictation that keeps going and ends here.")
        self.assertAlmostEqual(r.speech_seconds, 65)

    def test_name_in_a_later_part_counts(self):
        d = Decider(CFG)
        d.decide("So here is what I was thinking about", utt(0, 30, cut=True), 31)
        self.assertEqual(d.decide("the plan, Claude.", utt(30, 3, continued=True), 34).action, "submit")


if __name__ == "__main__":
    unittest.main()
