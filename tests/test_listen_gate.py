"""The half-duplex gate: audio is discarded while the HAL voice speaks, and for a tail after."""

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hal.listener import settings  # noqa: E402
from hal.listener.gate import SpeakingGate  # noqa: E402
from listen_harness import ROOT, isolated_env, make_tmp, rmtree, write_config  # noqa: E402
from test_listen_privacy import FakeClock, Rig  # noqa: E402


class FakePids:
    """PID files whose 'processes' are alive while listed in self.alive."""

    def __init__(self, tmp: Path):
        self.reply = tmp / "speaker.pid"
        self.ack = tmp / "ack.pid"
        self.alive = set()

    def speak(self, path: Path, pid: int):
        path.write_text(json.dumps({"pid": pid}))
        self.alive.add(pid)

    def finish(self, path: Path):
        pid = json.loads(path.read_text())["pid"]
        self.alive.discard(pid)
        path.unlink()


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.pids = FakePids(self.tmp)
        self.clock = FakeClock(100.0)
        self.own = False
        self.gate = SpeakingGate([self.pids.reply, self.pids.ack], tail=0.6, clock=self.clock, poll_interval=0.05,
                                 own_speaking=lambda: self.own, is_worker=lambda pid: True,
                                 alive=lambda pid: pid in self.pids.alive)

    def tearDown(self):
        rmtree(self.tmp)

    def test_open_when_silent(self):
        self.assertFalse(self.gate.blocked(100.0))

    def test_blocked_while_speaking_and_for_the_tail(self):
        self.pids.speak(self.pids.reply, 501)
        self.clock.t = 101.0
        self.assertTrue(self.gate.blocked(101.0))
        self.pids.finish(self.pids.reply)
        self.clock.t = 103.0
        self.assertFalse(self.gate.poll(force=True))
        self.assertTrue(self.gate.blocked(102.5))      # captured while speaking (processed late)
        self.assertTrue(self.gate.blocked(103.5))      # within the tail
        self.assertFalse(self.gate.blocked(103.7))     # after the tail
        self.assertFalse(self.gate.blocked(100.5))     # before it started

    def test_acknowledgement_also_blocks(self):
        self.pids.speak(self.pids.ack, 502)
        self.clock.t = 100.2
        self.assertTrue(self.gate.blocked(100.2))

    def test_dead_or_foreign_pid_is_not_speaking(self):
        self.pids.reply.write_text(json.dumps({"pid": 999999}))
        self.assertFalse(self.gate.poll(force=True))
        gate = SpeakingGate([self.pids.reply], tail=0.6, clock=self.clock, is_worker=lambda pid: False,
                            alive=lambda pid: True)
        self.assertFalse(gate.poll(force=True))

    def test_reply_end_only_for_replies_not_our_own_speech(self):
        self.pids.speak(self.pids.ack, 503)            # acknowledgement
        self.clock.t = 101
        self.gate.poll(force=True)
        self.pids.finish(self.pids.ack)
        self.clock.t = 102
        self.gate.poll(force=True)
        self.assertLess(self.gate.reply_end, 0)
        self.own = True                                 # "Yes, Dr. Alexander?"
        self.pids.speak(self.pids.reply, 504)
        self.clock.t = 103
        self.gate.poll(force=True)
        self.pids.finish(self.pids.reply)
        self.own = False
        self.clock.t = 104
        self.gate.poll(force=True)
        self.assertLess(self.gate.reply_end, 0)
        self.pids.speak(self.pids.reply, 505)          # Claude's reply
        self.clock.t = 105
        self.gate.poll(force=True)
        self.pids.finish(self.pids.reply)
        self.clock.t = 110
        self.gate.poll(force=True)
        self.assertEqual(self.gate.reply_end, 110)


class ListenerGateTest(unittest.TestCase):
    """The listening loop drops an utterance that the voice interrupts, and never transcribes it."""

    def test_partial_utterance_discarded(self):
        speech = np.concatenate([np.zeros(16000, np.float32), np.full(32000, 0.5, np.float32),
                                 np.zeros(32000, np.float32)])
        # Speech from 1.0 to 3.0 s; the voice starts 1.5 s in and (with its tail) lasts past the speech.
        rig = Rig(speech, texts=["Claude, this should never be transcribed."],
                  blocked=lambda t: 1001.5 <= t <= 1003.5)
        try:
            rig.lst.run()
            self.assertEqual(rig.lst.asr.calls, 0)
            self.assertEqual([e["action"] for e in rig.lst.events], ["discarded"])
            self.assertEqual(rig.lst.injector.typed, [])
        finally:
            rig.close()

    def test_speech_after_the_tail_is_heard(self):
        speech = np.concatenate([np.zeros(16000, np.float32), np.full(16000, 0.5, np.float32),
                                 np.zeros(48000, np.float32)])
        rig = Rig(speech, texts=["Claude, hello."], blocked=lambda t: t < 1000.9)   # voice ends before speech
        try:
            rig.lst.run()
            self.assertEqual(rig.lst.injector.typed, ["Claude, hello."])
        finally:
            rig.close()


@unittest.skipUnless((ROOT / "models" / "hal.onnx").exists() and (ROOT / ".venv" / "bin" / "python").exists(),
                     "needs the HAL voice model and .venv")
class RealSpeakerTest(unittest.TestCase):
    """The gate against a real HAL speech worker (played by the silent fake player)."""

    def test_gate_follows_hal_say(self):
        tmp = make_tmp()
        write_config(tmp)
        env = isolated_env(tmp)
        old = os.environ.get("HAL_STATE_DIR")
        os.environ["HAL_STATE_DIR"] = env["HAL_STATE_DIR"]
        try:
            gate = SpeakingGate(settings.speaker_pid_files(), tail=0.6, poll_interval=0.02)
            self.assertFalse(gate.poll(force=True))
            proc = subprocess.Popen([str(ROOT / "bin" / "hal"), "say", "--raw", "--wait", "Yes, Doctor Alexander?"],
                                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seen, deadline = False, time.monotonic() + 20
            while proc.poll() is None and time.monotonic() < deadline:
                seen = seen or gate.poll(force=True)
                time.sleep(0.02)
            proc.wait(10)
            time.sleep(0.1)
            self.assertTrue(seen, "the gate never saw the voice speaking")
            self.assertFalse(gate.poll(force=True))
            self.assertTrue(gate.blocked(time.monotonic()))            # tail
            time.sleep(0.7)
            self.assertFalse(gate.blocked(time.monotonic()))
            events = (tmp / "played" / "events.log").read_text()
            self.assertIn("start", events)
            self.assertIn("end", events)
        finally:
            if old is None:
                os.environ.pop("HAL_STATE_DIR", None)
            else:
                os.environ["HAL_STATE_DIR"] = old
            rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
