"""Listening while the voice speaks (duplex "aec"): barge-in, the echo check, resuming, fallback.

Nothing here opens a microphone or plays a sound: the audio helper is replaced
by tests/duplex_standin.py (its protocol, no audio) or by an in-memory source,
and the Swift client is only run against a stand-in socket with a silent WAV.
"""

import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal.listener import bargein, settings  # noqa: E402
from hal.listener.audio import ArraySource  # noqa: E402
from hal.listener.bargein import BargeIn, Spoken, is_echo, is_resume_phrase, is_stop_phrase, resolve  # noqa: E402
from hal.listener.decider import Decider  # noqa: E402
from hal.listener.service import Listener  # noqa: E402
from listen_harness import ROOT, make_tmp, rmtree  # noqa: E402
from test_listen_privacy import FakeASR, FakeClock, FakeGate, FakeInjector, FakeVAD, FakeVoice  # noqa: E402

CFG = dict(settings.DEFAULTS)
STANDIN = Path(__file__).resolve().parent / "duplex_standin.py"
HELPER_BIN = ROOT / "duplex" / "hal-duplex"


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

class PhraseTest(unittest.TestCase):
    def test_stop_phrases(self):
        for text in ["Stop.", "Stop!", "That's enough.", "Thank you, Claude.", "Thanks.", "Okay, thanks, that's enough.",
                     "Claude, stop.", "Hold on.", "Please stop talking.", "Cloud, that will do.", "Thank you very much."]:
            self.assertTrue(is_stop_phrase(text, CFG), text)

    def test_requests_are_not_stop_phrases(self):
        for text in ["Stop the server.", "Thanks for the summary, now open chapter two.", "That's enough sugar.",
                     "Okay.", "Claude.", "", "Wait until the build finishes and then run the tests."]:
            self.assertFalse(is_stop_phrase(text, CFG), text)

    def test_resume_phrases(self):
        for text in ["Go on.", "Continue.", "Claude, continue, please.", "Carry on.", "Sorry, go on.", "Please continue."]:
            self.assertTrue(is_resume_phrase(text, CFG), text)
        for text in ["Continue with the refactor.", "Go on to the next chapter.", "Stop."]:
            self.assertFalse(is_resume_phrase(text, CFG), text)

    def test_the_settings_lists_are_used(self):
        cfg = dict(CFG, barge_in_stop_phrases=["halt"], barge_in_resume_phrases=["proceed"])
        self.assertTrue(is_stop_phrase("Halt, Claude.", cfg))
        self.assertFalse(is_stop_phrase("Stop.", cfg))
        self.assertTrue(is_resume_phrase("Proceed.", cfg))


class EchoTest(unittest.TestCase):
    SPOKEN = "The northern orchards produced more apples than expected this autumn. The farmers were pleased."

    def test_the_voice_heard_back_is_echo(self):
        self.assertTrue(is_echo("produced more apples than expected", self.SPOKEN))
        self.assertTrue(is_echo("the farmers were pleased", self.SPOKEN))
        self.assertTrue(is_echo("orchards produced apples expected", self.SPOKEN))   # words dropped by the canceller

    def test_the_users_words_are_not_echo(self):
        self.assertFalse(is_echo("open the second chapter of my thesis", self.SPOKEN))
        self.assertFalse(is_echo("what about the pears", self.SPOKEN))
        self.assertFalse(is_echo("", self.SPOKEN))

    def test_order_matters(self):
        self.assertFalse(is_echo("pleased farmers were the apples more", self.SPOKEN))

    def test_resolve_order(self):
        self.assertEqual(resolve("", self.SPOKEN, CFG), "noise")
        self.assertEqual(resolve("Stop.", "Use hal stop to stop it.", CFG), "stop")       # the user's stop wins
        self.assertEqual(resolve("Go on.", self.SPOKEN, CFG), "resume")
        self.assertEqual(resolve("more apples than expected", self.SPOKEN, CFG), "echo")
        self.assertEqual(resolve("Open my thesis.", self.SPOKEN, CFG), "request")
        self.assertEqual(resolve("Orchard.", self.SPOKEN, CFG), "noise")                 # one stray word
        self.assertEqual(resolve("more apples than expected", "", CFG), "request")      # text unknown: no echo check


class SpokenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)
        self.wav = self.tmp / "speech-1.wav"
        bargein.sidecar_path(self.wav).write_text(json.dumps({
            "text": "One. Two two. Three three three.", "seconds": 6.0,
            "sentences": [{"text": "One.", "start": 0.0, "end": 1.0}, {"text": "Two two.", "start": 1.0, "end": 3.0},
                          {"text": "Three three three.", "start": 3.0, "end": 6.0}]}))

    def test_positions(self):
        s = Spoken.load(self.wav)
        self.assertEqual([s.index_at(x) for x in (0.0, 0.99, 1.0, 2.5, 5.9, 6.0, 9.0)], [0, 0, 1, 1, 2, 3, 3])
        self.assertEqual(s.rest_from(2.0), (1, "Two two. Three three three."))
        self.assertEqual(s.rest_from(7.0), (3, ""))
        self.assertEqual(s.around(3.5, 4.0, before=1.0, after=0.5), "Two two. Three three three.")
        self.assertEqual(s.around(0.1, 0.2, before=0.0, after=0.0), "One.")

    def test_missing_or_broken(self):
        self.assertIsNone(Spoken.load(self.tmp / "nothing.wav"))
        self.assertIsNone(Spoken.load(None))
        bad = self.tmp / "bad.wav"
        bargein.sidecar_path(bad).write_text("{not json")
        self.assertIsNone(Spoken.load(bad))


class BargeInTest(unittest.TestCase):
    """The frame-by-frame detector: speech-like and loud enough, for most of a short window, while the voice plays."""

    def feed(self, b, frames, playing=True, ref=-20.0, t0=0.0):
        for i, (p, db) in enumerate(frames):
            r = b.frame(t0 + i * 0.032, p, db, playing, ref)
            if r:
                return i
        return None

    def test_pauses_after_enough_speech(self):
        b = BargeIn(CFG)              # 0.2 s of speech within 0.3 s: 6 frames of 9
        self.assertEqual(self.feed(b, [(0.9, -28.0)] * 20), 5)
        self.assertTrue(b.paused)
        self.assertIsNone(self.feed(b, [(0.9, -28.0)] * 20))     # paused: no second trigger
        b.settle()
        self.assertEqual(self.feed(b, [(0.9, -28.0)] * 20), 5)

    def test_not_while_the_voice_is_silent(self):
        self.assertIsNone(self.feed(BargeIn(CFG), [(0.9, -20.0)] * 30, playing=False))

    def test_residual_echo_is_too_quiet(self):
        # Speech-like but 45 dB under what the voice renders: what the canceller left of it.
        self.assertIsNone(self.feed(BargeIn(CFG), [(0.9, -65.0)] * 30, ref=-20.0))
        # The floor follows the voice's level: -45 dBFS is enough when it renders quietly...
        self.assertIsNotNone(self.feed(BargeIn(CFG), [(0.9, -45.0)] * 30, ref=-60.0))
        # ... but not when it renders at -20 dBFS (floor -40).
        self.assertIsNone(self.feed(BargeIn(CFG), [(0.9, -45.0)] * 30, ref=-20.0))

    def test_scattered_frames_do_not_add_up(self):
        frames = [(0.9, -25.0), (0.1, -25.0), (0.1, -25.0)] * 20     # one in three
        self.assertIsNone(self.feed(BargeIn(CFG), frames))

    def test_low_vad_is_not_speech(self):
        self.assertIsNone(self.feed(BargeIn(CFG), [(0.4, -20.0)] * 30))

    def test_overlap_includes_a_tail(self):
        b = BargeIn(CFG)
        b.frame(10.0, 0.0, -90.0, True, -20.0)
        self.assertTrue(b.overlaps(10.3))
        self.assertFalse(b.overlaps(10.6))

    def test_settings_in_seconds(self):
        b = BargeIn(dict(CFG, barge_in_speech=0.1, barge_in_within=0.2))
        self.assertEqual((b.window.maxlen, b.need), (6, 3))


# ---------------------------------------------------------------------------
# The listening loop with an in-memory duplex source
# ---------------------------------------------------------------------------

class FakeDuplex:
    """In-memory audio with the helper's extras: the voice plays from `start` to `end` (clock
    time) unless paused or stopped; commands are recorded and answered with the helper's events."""

    duplex = True

    def __init__(self, audio, clock, *, start, end, wav, active=True):
        self.src = ArraySource(audio, realtime=False, clock=clock)
        self.clock = clock
        self.start_t, self.end_t = start, end
        self.wav = str(wav)
        self.active = active
        self.commands = []
        self._events = []
        self.announced = False
        self.paused = self.stopped = False
        self.paused_for = 0.0
        self._paused_at = None
        self.playing = False
        self.ref_db = None
        self.name = "fake duplex"

    def __getattr__(self, name):          # running, finished, keeps_time, start, stop, time_of ...
        return getattr(self.src, name)

    @property
    def keeps_time(self):
        return True

    def position(self):
        now = self._paused_at if self._paused_at is not None else self.clock()
        return max(0.0, now - self.start_t - self.paused_for)

    def read(self, timeout=0.1):
        item = self.src.read(timeout)
        if item is None:
            return None
        t = item[0]
        if not self.announced and t >= self.start_t:
            self.announced = True
            self._events.append({"event": "play", "id": 7, "seconds": self.end_t - self.start_t, "path": self.wav})
        on = self.start_t <= t < self.end_t + self.paused_for and not self.paused and not self.stopped
        self.playing = on
        self.ref_db = -20.0 if on else None
        return item

    def events(self):
        out, self._events = self._events, []
        return out

    def command(self, line):
        self.commands.append(line)
        if line == "pause" and not self.paused:
            self.paused, self._paused_at = True, self.clock()
            self._events.append({"event": "paused", "id": 7, "position": self.position()})
        elif line == "resume" and self.paused:
            self.paused_for += self.clock() - self._paused_at
            self.paused, self._paused_at = False, None
            self._events.append({"event": "resumed", "id": 7, "position": self.position()})
        elif line == "stop" and not self.stopped:
            pos = self.position()
            self.stopped, self.paused = True, False
            self._events.append({"event": "played", "id": 7, "reason": "stopped", "position": pos})
        return True

    def describe(self):
        return {"mode": "aec" if self.active else "half", "why": ""}


def timeline(total, bursts, level=0.5):
    """Silence of `total` seconds with loud bursts [(start, seconds)] (FakeVAD: loud is speech)."""
    x = np.zeros(int(total * 16000), np.float32)
    for start, secs in bursts:
        x[int(start * 16000):int((start + secs) * 16000)] = level
    return x


class DuplexRig:
    """A Listener over a FakeDuplex; the voice says three sentences from 1000.5 to 1009.5 on its clock."""

    SENTENCES = [("The first sentence is short.", 0.0, 2.0), ("The second sentence is about apples and pears.", 2.0, 5.0),
                 ("The third sentence ends it.", 5.0, 9.0)]

    def __init__(self, bursts, texts, total=16.0, active=True, gate_blocked=None, voice_end=1009.5, **cfg_over):
        self.tmp = make_tmp()
        self._env = mock.patch.dict(os.environ, {"HAL_STATE_DIR": str(self.tmp / "state"),
                                                 "HAL_CONFIG": str(self.tmp / "c.json")})
        self._env.start()
        wav = self.tmp / "state" / "speech-7.wav"
        wav.parent.mkdir(parents=True)
        bargein.sidecar_path(wav).write_text(json.dumps({
            "text": " ".join(s for s, _, _ in self.SENTENCES), "seconds": 9.0,
            "sentences": [{"text": s, "start": a, "end": b} for s, a, b in self.SENTENCES]}))
        cfg = dict(settings.DEFAULTS, duplex="aec", **cfg_over)
        self.clock = clock = FakeClock(1000.0)
        self.source = FakeDuplex(timeline(total, bursts), clock, start=1000.5, end=voice_end, wav=wav, active=active)
        self.stops = []
        self.console = []
        blocked = gate_blocked or (lambda t: 1000.5 <= t <= voice_end)   # the PID gate: the voice is speaking
        self.lst = Listener(cfg, source=self.source, vad=FakeVAD(), asr=FakeASR(texts), gate=FakeGate(blocked),
                            injector=FakeInjector(), decider=Decider(cfg), voice=FakeVoice(), console=self.console.append,
                            clock=clock, stop_voice=lambda: self.stops.append(clock()), linger=3.0)

    def run(self):
        self.lst.run()
        return self

    def actions(self):
        return [e["action"] for e in self.lst.events]

    def files_text(self):
        return "\n".join(p.read_text(errors="replace") for p in self.tmp.rglob("*") if p.is_file())

    def close(self):
        self._env.stop()
        rmtree(self.tmp)


class ListenOverVoiceTest(unittest.TestCase):
    def rig(self, *args, **kw):
        r = DuplexRig(*args, **kw)
        self.addCleanup(r.close)
        return r.run()

    def test_a_stop_phrase_pauses_then_silences_the_voice(self):
        r = self.rig([(3.0, 0.6)], ["That's enough."])
        self.assertEqual(r.source.commands[:2], ["pause", "stop"])
        self.assertEqual(len(r.stops), 1)                       # `hal stop` for what is queued behind it
        self.assertEqual(r.lst.injector.typed, [])              # not passed on
        self.assertIn("barge_in", r.actions())
        self.assertIn("stop_phrase", r.actions())
        # Paused within half a second of the speech starting (FakeVAD: at once; 0.2 s of speech needed).
        barge = next(e for e in r.lst.events if e["action"] == "barge_in")
        self.assertLess(barge["heard_at"] - 1003.0, 0.5)
        # Cut 2.5 s into the voice: in the second sentence, which "go on" would start again.
        cut = r.lst.interrupted
        self.assertEqual((cut.sentence, cut.sentences), (1, 3))
        self.assertTrue(cut.rest.startswith("The second sentence"))
        saved = json.loads((r.tmp / "state" / "listen" / "interrupted.json").read_text())
        self.assertEqual(saved["sentence"], 1)
        self.assertNotIn("enough", r.files_text())              # the stop phrase itself is never written

    def test_go_on_after_a_cut_resumes_from_that_sentence(self):
        r = self.rig([(3.0, 0.6), (7.0, 0.5)], ["Stop.", "Go on, Claude."])
        self.assertEqual(r.lst.voice.said, ["The second sentence is about apples and pears. The third sentence ends it."])
        self.assertEqual(r.lst.injector.typed, [])
        self.assertIsNone(r.lst.interrupted)
        self.assertEqual(r.lst.counts.voice_resumed, 1)

    def test_go_on_too_late_is_an_ordinary_utterance(self):
        r = self.rig([(3.0, 0.6), (9.0, 0.5)], ["Stop.", "Go on, Claude."], resume_seconds=2.0)
        self.assertEqual(r.lst.voice.said, [])
        self.assertEqual(r.lst.injector.typed, ["Go on, Claude."])   # addressed, so typed as usual

    def test_its_own_echo_resumes_the_voice(self):
        r = self.rig([(3.0, 0.6)], ["about apples and pears"])
        self.assertEqual(r.source.commands, ["pause", "resume"])
        self.assertEqual(r.stops, [])
        self.assertEqual(r.lst.injector.typed, [])
        self.assertEqual(r.lst.counts.echo, 1)
        self.assertNotIn("apples and pears", (r.tmp / "state" / "listen.log").read_text())

    def test_a_noise_resumes_the_voice(self):
        r = self.rig([(3.0, 0.22)], [])              # long enough to pause it, too short to be speech
        self.assertEqual(r.source.commands, ["pause", "resume"])
        self.assertEqual(r.lst.injector.typed, [])

    def test_go_on_while_paused_resumes(self):
        r = self.rig([(3.0, 0.5)], ["Sorry, go on."])
        self.assertEqual(r.source.commands, ["pause", "resume"])
        self.assertEqual(r.lst.voice.said, [])

    def test_anything_else_is_a_request_without_the_name(self):
        r = self.rig([(3.0, 1.0)], ["Open the second chapter of my thesis."])
        self.assertEqual(r.source.commands[:2], ["pause", "stop"])
        self.assertEqual(r.lst.injector.typed, ["Open the second chapter of my thesis."])
        frag = next(e for e in r.lst.events if e["action"] == "fragment")
        self.assertEqual(frag["how"], "barge-in")
        self.assertEqual(r.lst.counts.voice_cut, 1)

    def test_quiet_speech_over_the_voice_needs_the_name(self):
        # Too quiet to interrupt (under barge_in_min_db and the echo margin): heard, but not taken as barge-in.
        r = self.rig([(3.0, 1.0)], ["My PIN is 4321."], barge_in_min_db=-3.0, barge_in_echo_margin=0.0)
        self.assertEqual(r.source.commands, [])
        self.assertEqual(r.lst.injector.typed, [])
        self.assertNotIn("4321", r.files_text())               # ignored words are never written

    def test_quiet_speech_over_the_voice_gets_no_window(self):
        # Claude is working on a spoken request (during_turn), which normally needs no name; but
        # speech heard over the voice without interrupting it may be residual echo: name only.
        session = {"turn": "busy", "last_stop": 0.0}
        r = DuplexRig([(3.0, 1.0), (12.0, 1.0)], ["and the apples and pears", "Also check the second chapter."],
                      barge_in_min_db=-3.0, barge_in_echo_margin=0.0)
        self.addCleanup(r.close)
        r.lst.session_state = lambda: session
        r.lst.injector.last_typed = time.time()
        r.run()
        self.assertEqual(r.lst.injector.typed, ["Also check the second chapter."])     # only the one after the voice

    def test_one_stray_word_resumes_the_voice(self):
        r = self.rig([(3.0, 0.5)], ["Orchard."])
        self.assertEqual(r.source.commands, ["pause", "resume"])
        self.assertEqual(r.lst.injector.typed, [])

    def test_named_request_over_the_voice_silences_it(self):
        r = self.rig([(3.0, 1.0)], ["Claude, what time is it?"], barge_in_min_db=-3.0, barge_in_echo_margin=0.0)
        self.assertEqual(r.lst.injector.typed, ["Claude, what time is it?"])
        self.assertIn("stop", r.source.commands)
        self.assertEqual(len(r.stops), 1)

    def test_speech_after_the_voice_is_unaffected(self):
        r = self.rig([(11.5, 0.8)], ["Claude, open my thesis."])
        self.assertEqual(r.source.commands, [])
        self.assertEqual(r.lst.injector.typed, ["Claude, open my thesis."])

    def test_without_cancellation_the_gate_applies_as_before(self):
        r = self.rig([(3.0, 1.0)], ["Claude, open my thesis."], active=False)
        self.assertEqual(r.source.commands, [])
        self.assertEqual(r.lst.injector.typed, [])
        self.assertGreater(r.lst.counts.gated_seconds, 8.0)     # dropped unheard while the voice spoke

    def test_barge_in_off_listens_but_never_interrupts(self):
        r = self.rig([(3.0, 1.0)], ["Claude, open my thesis."], barge_in=False)
        self.assertEqual(r.source.commands, [])
        self.assertEqual(r.lst.injector.typed, ["Claude, open my thesis."])   # heard over the voice

    def test_status_reports_the_mode(self):
        r = self.rig([], [])
        self.assertEqual(r.lst.duplex.describe()["mode"], "aec")


# ---------------------------------------------------------------------------
# DuplexSource with the stand-in helper: start, fallback, device changes
# ---------------------------------------------------------------------------

class FakeMic:
    def __init__(self):
        self.started = self.stopped = 0
        self.name = "MacBook Pro Microphone"

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout=0.1):
        time.sleep(0.005)
        return (time.monotonic(), np.zeros(512, np.float32))


class DuplexSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)
        self._env = mock.patch.dict(os.environ, {"HAL_STATE_DIR": str(self.tmp / "state")})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.helper = self.tmp / "hal-duplex"
        self.helper.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{STANDIN}' \"$@\"\n")
        self.helper.chmod(self.helper.stat().st_mode | stat.S_IXUSR)
        self.log = self.tmp / "standin.log"
        self.mics = []

    def source(self, behaviour, **kw):
        from hal.listener.duplex import DuplexSource

        os.environ["HAL_DUPLEX_STANDIN"] = behaviour
        os.environ["HAL_DUPLEX_STANDIN_LOG"] = str(self.log)

        def mic():
            m = FakeMic()
            self.mics.append(m)
            return m
        src = DuplexSource(dict(CFG, duplex="aec"), helper=kw.pop("helper", self.helper),
                           sock=self.tmp / "state" / "listen" / "duplex.sock", fallback_factory=mic,
                           ready_timeout=kw.pop("ready_timeout", 5.0), **kw)
        self.addCleanup(src.stop)
        return src

    def read_some(self, src, n=5, limit=3.0):
        got, deadline = 0, time.monotonic() + limit
        while got < n and time.monotonic() < deadline:
            if src.read(0.2) is not None:
                got += 1
        return got

    def test_echo_cancellation_on(self):
        src = self.source("vp")
        src.start()
        self.assertTrue(src.active)
        self.assertTrue(src.vp)
        self.assertEqual(self.read_some(src), 5)
        self.assertEqual(self.mics, [])
        args = self.log.read_text().splitlines()[0]
        self.assertIn("serve --socket", args)
        self.assertIn("--mode auto", args)       # voice processing left off with headphones
        self.assertIn("--ducking min", args)
        self.assertIn("--yield-to-other-input", args)

    def test_commands_and_their_events(self):
        src = self.source("vp,playing")
        src.start()
        self.read_some(src, 2)
        self.assertTrue(src.playing)
        self.assertGreater(src.ref_db, -25)
        self.assertTrue(src.command("pause"))
        deadline = time.monotonic() + 2
        evs = []
        while time.monotonic() < deadline and not any(e["event"] == "paused" for e in evs):
            src.read(0.1)
            evs += src.events()
        self.assertTrue(any(e["event"] == "paused" for e in evs))

    def test_headphones_listen_without_cancellation(self):
        src = self.source("headphones")
        src.start()
        self.assertTrue(src.active)
        self.assertFalse(src.vp)
        self.assertEqual(src.describe()["mode"], "headphones")

    def test_no_cancellation_on_speakers_is_half_duplex(self):
        src = self.source("novp")
        src.start()
        self.assertFalse(src.active)
        self.assertIsNotNone(src.proc)            # still the helper: it may come back after a device change
        self.assertIn("echo cancellation unavailable", src.why)
        self.assertEqual(src.describe()["mode"], "half")

    def test_device_change_is_followed(self):
        src = self.source("vp,devices=3")
        src.start()
        self.assertTrue(src.vp)
        self.read_some(src, 6)
        evs = src.events()
        dev = [e for e in evs if e["event"] == "devices"]
        self.assertEqual(len(dev), 1)
        self.assertEqual(dev[0]["output"], "AirPods Pro")
        self.assertTrue(src.active)               # headphones now: still listening over the voice
        self.assertFalse(src.vp)

    def test_missing_helper_falls_back_to_the_microphone(self):
        src = self.source("vp", helper=self.tmp / "absent")
        with mock.patch("hal.listener.logs.log") as log:
            src.start()
        self.assertFalse(src.active)
        self.assertEqual(len(self.mics), 1)
        self.assertEqual(self.mics[0].started, 1)
        self.assertIn("no audio helper", src.why)
        self.assertTrue(any("half-duplex" in c.args[0] for c in log.call_args_list))
        self.assertIsNotNone(src.read(0.1))       # audio from the microphone

    def test_helper_that_exits_at_once(self):
        src = self.source("exit")
        src.start()
        self.assertFalse(src.active)
        self.assertEqual(len(self.mics), 1)
        self.assertIn("exited", src.why)

    def test_helper_that_cannot_open_the_devices(self):
        src = self.source("error")
        src.start()
        self.assertEqual(len(self.mics), 1)
        self.assertIn("could not open the audio devices", src.why)

    def test_helper_that_dies_later(self):
        src = self.source("vp,die=3")
        src.start()
        self.assertTrue(src.active)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not self.mics:
            src.read(0.1)
        self.assertEqual(len(self.mics), 1)
        self.assertFalse(src.active)
        self.assertIn("exited", src.why)

    def test_tries_the_helper_again_later(self):
        clock = FakeClock(100.0)
        src = self.source("exit", clock=clock)
        src.start()
        self.assertEqual(len(self.mics), 1)
        self.assertGreaterEqual(src.retry_at, 160.0)
        os.environ["HAL_DUPLEX_STANDIN"] = "vp"
        clock.t = src.retry_at + 1
        src.read(0.1)                              # time to try again
        self.assertTrue(src.active)
        self.assertEqual(self.mics[0].stopped, 1)

    def test_stop_ends_the_helper(self):
        src = self.source("vp")
        src.start()
        proc = src.proc
        src.stop()
        self.assertIsNotNone(proc.poll())
        self.assertIn("eof", self.log.read_text())
        self.assertFalse(src.running)


# ---------------------------------------------------------------------------
# The speaker side: sentence sidecars, and the player
# ---------------------------------------------------------------------------

class FakeChunk:
    def __init__(self, n):
        self.audio_int16_bytes = b"\0\0" * n


class FakePiper:
    class config:
        sample_rate = 1000

    def __init__(self, chunks):
        self.chunks = chunks

    def synthesize(self, text, syn_config=None):
        return [FakeChunk(n) for n in self.chunks]


class SpeakerSideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)
        from hal import speaker
        self.speaker = speaker

    def test_sidecar_follows_pipers_sentences(self):
        wav = self.tmp / "speech-1.wav"
        secs = self.speaker.synthesize("First one. Second, longer one? Third!", wav, voice=FakePiper([1000, 2500, 500]),
                                       sidecar=True)
        self.assertEqual(secs, 4.0)
        data = json.loads(self.speaker.sidecar(wav).read_text())
        self.assertEqual([(s["text"], s["start"], s["end"]) for s in data["sentences"]],
                         [("First one.", 0.0, 1.0), ("Second, longer one?", 1.0, 3.5), ("Third!", 3.5, 4.0)])

    def test_sidecar_in_proportion_when_the_split_differs(self):
        wav = self.tmp / "speech-2.wav"
        self.speaker.synthesize("Aaaa. Bbbbbbbb.", wav, voice=FakePiper([3000]), sidecar=True)
        s = Spoken.load(wav)
        self.assertEqual(len(s.sentences), 2)
        self.assertAlmostEqual(s.sentences[0]["end"], 3.0 * 5 / 14, places=2)     # 5 of 14 characters of 3 s

    def test_no_sidecar_unless_asked(self):
        wav = self.tmp / "out.wav"
        self.speaker.synthesize("Hello.", wav, voice=FakePiper([100]))
        self.assertFalse(self.speaker.sidecar(wav).exists())

    def test_unlink_removes_both(self):
        wav = self.tmp / "speech-3.wav"
        self.speaker.synthesize("Hello.", wav, voice=FakePiper([100]), sidecar=True)
        self.speaker._unlink_wav(wav)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_cached_acknowledgement_gets_a_sidecar(self):
        cache = self.tmp / "ack-cache"
        with mock.patch.object(self.speaker, "ACK_CACHE", cache):
            cfg = {"length_scale": None}
            wav = self.speaker.ack_wav("Just a moment.", cfg, voice=FakePiper([800]))
            self.assertEqual(Spoken.load(wav).text, "Just a moment.")
            self.speaker.sidecar(wav).unlink()                        # as cached before sidecars existed
            self.assertEqual(self.speaker.ack_wav("Just a moment.", cfg, voice=FakePiper([800])), wav)
            self.assertAlmostEqual(Spoken.load(wav).seconds, 0.8)
        self.assertEqual(sorted(p.name for p in cache.iterdir()), [wav.name, wav.name + ".json"])

    def test_player_is_the_helper_only_while_its_socket_exists(self):
        from hal import config

        root, state = self.tmp / "root", self.tmp / "state"
        (root / "duplex").mkdir(parents=True)
        helper = root / "duplex" / "hal-duplex"
        helper.write_text("#!/bin/sh\n")
        helper.chmod(0o755)
        (state / "listen").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k != "HAL_PLAYER"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(config, "ROOT", root), \
                mock.patch.object(config, "STATE_DIR", state):
            self.assertEqual(config.player_command(), ["/usr/bin/afplay"])
            sock = state / "listen" / "duplex.sock"
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(s.close)
            short = Path("/tmp") / f"hal-dx-test-{os.getpid()}.sock"     # a socket file at the expected path
            s.bind(str(short))
            self.addCleanup(lambda: short.unlink(missing_ok=True))
            os.replace(short, sock)
            self.assertEqual(config.player_command(), [str(helper), "play", "--socket", str(sock)])
            with mock.patch.dict(os.environ, {"HAL_PLAYER": "/bin/echo"}):
                self.assertEqual(config.player_command(), ["/bin/echo"])     # tests' fake player wins
            helper.chmod(0o644)
            self.assertEqual(config.player_command(), ["/usr/bin/afplay"])  # no helper binary


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)

    def load(self, block):
        p = self.tmp / "c.json"
        p.write_text(json.dumps({"listener": block}))
        return settings.load(p)

    def test_off_by_default(self):
        cfg = settings.load(self.tmp / "none.json")
        self.assertEqual(cfg["duplex"], "half")
        self.assertTrue(cfg["barge_in"])

    def test_values_checked(self):
        cfg = self.load({"duplex": "AEC", "aec_ducking": "max", "barge_in_min_db": -45, "barge_in_speech": 5,
                         "barge_in_echo_margin": -3, "barge_in_stop_phrases": ["Halt"], "aec_on_headphones": "yes"})
        self.assertEqual(cfg["duplex"], "aec")
        self.assertEqual(cfg["aec_ducking"], "max")
        self.assertEqual(cfg["barge_in_min_db"], -45.0)
        self.assertEqual(cfg["barge_in_speech"], cfg["barge_in_within"])      # capped at the window
        self.assertEqual(cfg["barge_in_echo_margin"], settings.DEFAULTS["barge_in_echo_margin"])
        self.assertEqual(cfg["barge_in_stop_phrases"], ["halt"])
        self.assertFalse(cfg["aec_on_headphones"])
        self.assertEqual(self.load({"duplex": "full"})["duplex"], "half")

    def test_decider_takes_barge_in_without_the_name(self):
        class U:
            speech_start, speech_seconds, duration, cut, continued = 0.0, 1.0, 1.2, False, False

        d = Decider(dict(CFG)).decide("Open the second chapter.", U(), 0.0, barge_in=True)
        self.assertEqual((d.action, d.how), ("submit", "barge-in"))
        d = Decider(dict(CFG)).decide("Open the second chapter.", U(), 0.0)
        self.assertEqual(d.action, "ignore")


# ---------------------------------------------------------------------------
# The Swift helper's `play` client, against a stand-in socket (silent WAV; no audio device opened)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.access(HELPER_BIN, os.X_OK), "duplex/hal-duplex not built (duplex/build.sh)")
class SwiftClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(rmtree, self.tmp)
        self.wav = self.tmp / "silence.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 1600)
        self.sock_path = Path("/tmp") / f"hal-dx-client-{os.getpid()}.sock"
        self.addCleanup(lambda: self.sock_path.unlink(missing_ok=True))

    def serve(self, reply_started=True):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.sock_path))
        srv.listen(1)
        self.addCleanup(srv.close)
        got = {}

        def run():
            c, _ = srv.accept()
            data = b""
            while not data.endswith(b"\n"):
                data += c.recv(100)
            got["request"] = data.decode()
            if reply_started:
                c.sendall(b"started 1 0.100\n")
                time.sleep(0.2)
                c.sendall(b"done end 0.100\n")
            else:
                c.sendall(b"error cannot read the file\n")
            got["eof"] = c.recv(10) == b""
            c.close()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return got, t

    def test_play_hands_the_file_to_the_helper_and_waits(self):
        got, t = self.serve()
        started = time.monotonic()
        r = subprocess.run([str(HELPER_BIN), "play", "--socket", str(self.sock_path), str(self.wav)], timeout=10)
        t.join(2)
        self.assertEqual(r.returncode, 0)
        self.assertGreaterEqual(time.monotonic() - started, 0.2)
        self.assertEqual(got["request"], f"play {self.wav}\n")
        self.assertTrue(got["eof"])

    def test_probe_prints_the_default_devices(self):
        out = subprocess.run([str(HELPER_BIN), "probe"], capture_output=True, text=True, timeout=10).stdout
        info = json.loads(out)
        self.assertLessEqual({"input", "output", "output_kind", "headphones", "other_input"}, set(info))


if __name__ == "__main__":
    unittest.main()
