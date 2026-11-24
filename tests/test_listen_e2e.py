"""End to end on synthesized speech: audio file -> listener -> pty proxy -> program, with the real hooks.

A track of synthesized utterances (macOS `say`), some addressed to Claude and
some not, is fed through the listener exactly as microphone audio would be
(real-time pacing, Silero VAD, Parakeet speech-to-text, the decision rules,
the voice gate). Typed lines go through the pty proxy into a stand-in for
claude, which fires the real HAL hooks (bin/hal hook: spoken acknowledgement
and reply) and the listener's state hook, as Claude Code would. Everything
"plays" through tests/fake_player.py into a temporary state directory, so
nothing is heard and the real state/ is untouched.

Takes about a minute, so it runs only with HAL_LISTEN_E2E=1:

    HAL_LISTEN_E2E=1 .venv/bin/python -m unittest tests.test_listen_e2e -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hal import spoken  # noqa: E402

from listen_harness import (HAL_LISTEN, LISTEN_PY, ROOT, Terminal, hook, isolated_env, make_tmp,  # noqa: E402
                            proxy_argv, proxy_info, rmtree, standin_argv, write_config)

FAKE_HOOKS = Path(__file__).resolve().parent / "listen_fake_hooks.py"

# (start second, voice, text, expected): expected is the line typed into the
# session, None if nothing should be typed, or a tag for other outcomes.
SCRIPT = [
    (1.0, "Samantha", "Good morning, Claude.", "Good morning, Claude."),
    (4.6, "Daniel", "Claude, you should not hear this while you are talking.", None),   # during ack + reply
    (12.0, "Daniel", "The files are stored in the cloud.", None),
    (15.5, "Karen", "Claude, what's the weather tomorrow?", "Claude, what's the weather tomorrow?"),
    (26.5, "Moira", "We should visit grandmother on Sunday.", None),
    (30.0, "Samantha", "Claude?", "name_only"),
    (35.0, "Samantha", "Open my thesis.", "Open my thesis."),                           # armed: no name needed
    (45.5, "Samantha", "By the way, Claude, open my thesis.", "By the way, Claude, open my thesis."),
    (57.5, "Karen", "Claude, stop listening.", "stop_listening"),
    (66.0, "Samantha", "Claude, are you still there?", None),                          # paused
]
# Each typed line is acknowledged and answered aloud (by the real hooks), and the
# gate stays closed until about 8 s after the end of the speech that caused it,
# so the utterances that should be heard start at least 8.5 s after the previous one.
TRACK_SECONDS = 70.0


def synth(text, voice):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["say", "-v", voice, "-o", f"{d}/a.aiff", text], check=True, timeout=60)
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", f"{d}/a.aiff", "-f", "s16le",
                              "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16)


def norm(text):
    """The words of a line, without the listener's label and marks (hal/spoken.py)."""
    return " ".join(re.findall(r"[a-z']+", spoken.plain(text).lower()))


def ready():
    return (os.environ.get("HAL_LISTEN_E2E") == "1" and LISTEN_PY.exists() and shutil.which("say")
            and shutil.which("ffmpeg") and (ROOT / "models" / "hal.onnx").exists()
            and (ROOT / "models" / "silero_vad.onnx").exists())


@unittest.skipUnless(ready(), "set HAL_LISTEN_E2E=1 (needs .venv-listen, the models, say and ffmpeg)")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = make_tmp("hal-e2e-")
        track = np.zeros(int(TRACK_SECONDS * 16000), np.int16)
        cls.spans = []
        for start, voice, text, expected in SCRIPT:
            audio = synth(text, voice)
            i = int(start * 16000)
            track[i:i + len(audio)] = audio
            nz = np.nonzero(np.abs(audio) > 300)[0]
            cls.spans.append((start + nz[0] / 16000, start + nz[-1] / 16000))
        cls.wav = cls.tmp / "track.wav"
        with wave.open(str(cls.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(track.tobytes())
        write_config(cls.tmp, {"autostart": False, "name_only_reply": "Yes, Dr. Alexander?"}, acknowledge=True,
                     spoken_reply="full", max_chars=600)
        cls.env = isolated_env(cls.tmp, HAL_VOICE="1", STANDIN_OUT=cls.tmp / "lines.jsonl",
                               STANDIN_ON_LINE=f"{sys.executable} {FAKE_HOOKS}")
        cls.term = Terminal(proxy_argv(*standin_argv()), env=cls.env)
        try:
            cls.term.wait_for(r"READY")
            cls.proxy = proxy_info(cls.tmp / "state")
            hook(cls.env, {"hook_event_name": "SessionStart", "source": "startup", "session_id": "e2e"},
                 cls.proxy["pid"])
            report = cls.tmp / "report.json"
            started = time.monotonic()
            cls.run_result = subprocess.run(
                [str(HAL_LISTEN), "run", "--file", str(cls.wav), "--hide-ignored", "--report", str(report),
                 "--linger", "10"], env=cls.env, capture_output=True, text=True, timeout=240)
            cls.elapsed = time.monotonic() - started
            cls.term.pump(1.0)
            cls.report = json.loads(report.read_text()) if report.exists() else {}
        except Exception:
            cls.term.close()
            raise
        lines = cls.tmp / "lines.jsonl"
        cls.lines = [json.loads(l) for l in lines.read_text().splitlines()] if lines.exists() else []
        cls.log = (cls.tmp / "state" / "listen.log").read_text() if (cls.tmp / "state" / "listen.log").exists() else ""

    @classmethod
    def tearDownClass(cls):
        try:
            cls.term.send("exit 0\r")
            cls.term.wait_exit(10)
        finally:
            cls.term.close()
            if os.environ.get("HAL_LISTEN_E2E_KEEP") != "1":
                rmtree(cls.tmp)

    def test_0_listener_ran(self):
        self.assertEqual(self.run_result.returncode, 0, self.run_result.stderr[-2000:])

    def test_typed_exactly_the_addressed_utterances(self):
        expected = [e for _, _, _, e in SCRIPT if e and e not in ("name_only", "stop_listening")]
        typed = [l["line"] for l in self.lines]
        self.assertEqual([norm(t) for t in typed], [norm(e) for e in expected],
                         f"typed {typed}\nlistener output:\n{self.run_result.stdout[-3000:]}")

    def test_name_alone_answered_and_stop_listening_paused(self):
        said = self.report.get("said", [])
        self.assertIn("Yes, Dr. Alexander?", said)
        self.assertTrue(any("stop listening" in s for s in said), said)
        self.assertTrue((self.tmp / "state" / "listen" / "paused").exists())

    def test_voice_gate_dropped_speech_during_the_reply(self):
        played = (self.tmp / "played" / "events.log")
        self.assertTrue(played.exists() and "start" in played.read_text(), "the HAL voice never played")
        # Utterance 2 lies wholly inside the acknowledgement and reply to utterance 1: it is never heard.
        self.assertGreater(self.report["counts"]["gated_seconds"], 10)
        self.assertFalse(any("hear this" in l["line"] for l in self.lines))
        self.assertFalse(any("hear this" in e.get("text", "") for e in self.report["events"]))

    def test_latency(self):
        latencies = [lat for _, lat in self.report.get("submitted", [])]
        self.assertEqual(len(latencies), 4)
        self.assertLess(max(latencies), 2.0, latencies)

    def test_unaddressed_words_not_logged(self):
        for word in ("stored", "grandmother", "Sunday"):
            self.assertNotIn(word, self.log)
        self.assertNotIn("grandmother", self.run_result.stdout)   # --hide-ignored
        self.assertIn("ignored an utterance not addressed to Claude", self.log)

    def test_zz_timing_table(self):
        """Not a check: prints what happened to each utterance, for the record."""
        t0 = self.report.get("audio_t0") or 0.0
        events = self.report.get("events", [])
        t0_wall = self.report.get("audio_t0_wall") or 0.0
        rows = []
        for (start, voice, text, expected), (s, e) in zip(SCRIPT, self.spans):
            ev = next((x for x in events if "speech_end" in x and s - 0.5 <= x["speech_end"] - t0 <= e + 1.0), None)
            action = ev["action"] if ev else "not heard (voice speaking, or paused)"
            typed = None
            if ev and ev["action"] in ("submit", "fragment"):   # the stand-in's wall-clock receipt of that line
                got = next((l["t"] for l in self.lines if norm(l["line"]) == norm(ev["text"])), None)
                typed = None if got is None else got - t0_wall - (ev["speech_end"] - t0)
            rows.append(f"  {s:5.1f}-{e:5.1f}s  {voice:9s} {text[:44]:44s} -> {action:15s}"
                        + (f" received {typed:.2f}s after end of speech" if typed is not None else ""))
        print("\n" + "\n".join(rows))
        print(f"  submitted latencies (listener clock): {[round(l, 2) for _, l in self.report.get('submitted', [])]}")
        print(f"  speech-to-text: {[round(x['asr'] * 1000) for x in events if 'asr' in x]} ms; "
              f"counts {self.report.get('counts')}; run took {self.elapsed:.0f}s")


# Turns (turn_taking "semantic"): one request spoken in four pieces with pauses of 3, 10 and 30 s
# between them is typed as one line; short finished requests go promptly; speech after the turn is
# answered needs the name again. The front desk runs (in the temporary state directory) to judge
# what the rules can't; its model is tests/fake_warm_model.py, or the real Haiku with
# HAL_LISTEN_E2E_HAIKU=1 (a handful of short calls). The long request is in Karen's voice: with
# Samantha's, Parakeet writes the final "paper and" of the first piece as "paper Anne?", a finished
# question, and the turn is sent too early (a known limit: the rules read the transcript).
TURN_SCRIPT = [
    (1.0, "Samantha", "Claude, what's the weather tomorrow?", "q"),
    (12.0, "Daniel", "Claude, open my thesis.", "open"),
    (23.0, "Samantha", "Claude, I think the conclusion of my paper is too weak.", "judged"),
    (42.0, "Karen", "Claude, could you look at the draft of my paper and", "long"),
    (None, "Karen", "check whether the argument in the second section is stronger than", "long"),     # +3 s
    (None, "Karen", "the objection raised by the second reviewer, because", "long"),                  # +10 s
    (None, "Karen", "the reviewer misread the main claim?", "long"),                                   # +30 s
    (None, "Daniel", "The files are stored in the cloud.", None),                                        # +12 s
]
TURN_PAUSES = [3.0, 10.0, 30.0, 12.0]


@unittest.skipUnless(ready(), "set HAL_LISTEN_E2E=1 (needs .venv-listen, the models, say and ffmpeg)")
class TurnsEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.haiku = os.environ.get("HAL_LISTEN_E2E_HAIKU") == "1"
        cls.tmp = make_tmp("hal-e2e-turns-")
        pieces, t, previous_end = [], 0.0, 0.0
        pauses = iter(TURN_PAUSES)
        for start, voice, text, tag in TURN_SCRIPT:
            audio = synth(text, voice)
            nz = np.nonzero(np.abs(audio) > 300)[0]
            if start is None:   # a set pause after the end of the previous piece's speech
                start = previous_end + next(pauses) - nz[0] / 16000
            s, e = start + nz[0] / 16000, start + nz[-1] / 16000
            pieces.append((start, audio, s, e, text, tag))
            previous_end = e
        total = pieces[-1][3] + 12.0
        track = np.zeros(int(total * 16000), np.int16)
        for start, audio, *_ in pieces:
            i = int(start * 16000)
            track[i:i + len(audio)] = audio
        cls.spans = [(s, e, text, tag) for _, _, s, e, text, tag in pieces]
        cls.wav = cls.tmp / "track.wav"
        with wave.open(str(cls.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(track.tobytes())
        write_config(cls.tmp, {"autostart": False}, acknowledge=True, spoken_reply="full", max_chars=600,
                     front_desk=True, desk_summaries=False, user_name="Dr. Alexander", desk_idle_minutes=10)
        extra = {} if cls.haiku else {
            "HAL_WARM_MODEL": f"{sys.executable} {Path(__file__).resolve().parent / 'fake_warm_model.py'}"}
        cls.env = isolated_env(cls.tmp, HAL_VOICE="1", STANDIN_OUT=cls.tmp / "lines.jsonl",
                               STANDIN_ON_LINE=f"{sys.executable} {FAKE_HOOKS}", **extra)
        hal = str(ROOT / "bin" / "hal")
        started = subprocess.run([hal, "desk", "start"], env=cls.env, capture_output=True, text=True, timeout=30)
        if started.returncode != 0:
            raise AssertionError(f"desk did not start: {started.stdout} {started.stderr}")
        cls.desk_ready = cls._wait_turn_model()
        cls.term = Terminal(proxy_argv(*standin_argv()), env=cls.env)
        try:
            cls.term.wait_for(r"READY")
            cls.proxy = proxy_info(cls.tmp / "state")
            hook(cls.env, {"hook_event_name": "SessionStart", "source": "startup", "session_id": "e2e"},
                 cls.proxy["pid"])
            report = cls.tmp / "report.json"
            began = time.monotonic()
            cls.run_result = subprocess.run(
                [str(HAL_LISTEN), "run", "--file", str(cls.wav), "--hide-ignored", "--report", str(report),
                 "--linger", "10"], env=cls.env, capture_output=True, text=True, timeout=400)
            cls.elapsed = time.monotonic() - began
            cls.term.pump(1.0)
            cls.report = json.loads(report.read_text()) if report.exists() else {}
        except Exception:
            cls.term.close()
            subprocess.run([hal, "desk", "stop"], env=cls.env, capture_output=True, timeout=30)
            raise
        lines = cls.tmp / "lines.jsonl"
        cls.lines = [json.loads(l) for l in lines.read_text().splitlines()] if lines.exists() else []
        log = cls.tmp / "state" / "listen.log"
        cls.log = log.read_text() if log.exists() else ""
        hal_log = cls.tmp / "state" / "hal.log"
        cls.hal_log = hal_log.read_text() if hal_log.exists() else ""

    @classmethod
    def _wait_turn_model(cls):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            p = subprocess.run([str(ROOT / "bin" / "hal"), "desk", "status", "--json"], env=cls.env,
                               capture_output=True, text=True)
            info = json.loads(p.stdout or "null") or {}
            if (info.get("turn_model") or {}).get("alive"):
                time.sleep(2.0 if cls.haiku else 0.2)   # let the real model finish starting
                return True
            time.sleep(0.2)
        return False

    @classmethod
    def tearDownClass(cls):
        try:
            cls.term.send("exit 0\r")
            cls.term.wait_exit(10)
        finally:
            cls.term.close()
            subprocess.run([str(ROOT / "bin" / "hal"), "desk", "stop"], env=cls.env, capture_output=True, timeout=30)
            if os.environ.get("HAL_LISTEN_E2E_KEEP") != "1":
                rmtree(cls.tmp)

    def test_0_listener_ran(self):
        self.assertTrue(self.desk_ready, "the desk's turn model never came up")
        self.assertEqual(self.run_result.returncode, 0, self.run_result.stderr[-2000:])

    def test_one_line_per_request(self):
        typed = [norm(l["line"]) for l in self.lines]
        long = " ".join(text for _, _, text, tag in self.spans if tag == "long")
        expected = [norm(t) for _, _, t, tag in self.spans if tag in ("q", "open", "judged")] + [norm(long)]
        self.assertEqual(len(typed), 4, f"typed {typed}\n{self.run_result.stdout[-4000:]}")
        self.assertEqual(typed[:3], expected[:3])
        # Transcribed afresh as a whole, the long request may differ from the pieces in a word or two.
        import difflib
        ratio = difflib.SequenceMatcher(None, typed[3], expected[3]).ratio()
        self.assertGreater(ratio, 0.9, (typed[3], expected[3]))

    def test_long_request_sent_once_after_the_last_piece(self):
        sent = [e for e in self.report.get("events", []) if e["action"] == "submit"]
        self.assertEqual([e["fragments"] for e in sent], [1, 1, 1, 4])
        t0 = self.report["audio_t0"]
        last_end = self.spans[6][1]
        self.assertGreater(sent[3]["at"] - t0, last_end, "not before the last piece was spoken")

    def test_judge_was_asked(self):
        self.assertIn("judged by the front desk", self.log)
        self.assertRegex(self.hal_log, r"desk: turn of \d+ words judged complete")

    def test_after_the_turn_the_name_is_needed_again(self):
        self.assertFalse(any("stored" in l["line"] for l in self.lines))
        self.assertNotIn("stored", self.log)

    def test_zz_timing_table(self):
        """Not a check: what happened to each piece, for the record."""
        t0 = self.report.get("audio_t0") or 0.0
        t0_wall = self.report.get("audio_t0_wall") or 0.0
        events = self.report.get("events", [])
        rows = []
        for s, e, text, tag in self.spans:
            ev = next((x for x in events if "speech_end" in x and x["action"] != "submit"
                       and s - 0.5 <= x["speech_end"] - t0 <= e + 1.0), None)
            what = ev["action"] if ev else "not heard"
            if ev and ev["action"] == "fragment":
                what = f"fragment: {ev.get('verdict')} ({ev.get('why')})"
            rows.append(f"  {s:6.1f}-{e:6.1f}s  {text[:52]:52s} -> {what}")
        rows.append("  typed:")
        for ev in (x for x in events if x["action"] == "submit"):
            got = next((l["t"] for l in self.lines if norm(l["line"]) == norm(ev["text"])), None)
            after = None if got is None else got - t0_wall - (ev["speech_end"] - t0)
            rows.append(f"    {ev['fragments']} fragment(s), {ev['why']}: received "
                        f"{after:.2f}s after the end of the last speech: {ev['text'][:70]!r}"
                        if after is not None else f"    {ev}")
        print(f"\n  model: {'real Haiku' if self.haiku else 'stand-in'}\n" + "\n".join(rows))
        print(f"  latencies (listener clock): {[round(l, 2) for _, l in self.report.get('submitted', [])]}; "
              f"counts {self.report.get('counts')}; run took {self.elapsed:.0f}s")
        print("  desk: " + "; ".join(l.split("desk: ", 1)[1] for l in self.hal_log.splitlines() if "desk: turn" in l))


if __name__ == "__main__":
    unittest.main()
