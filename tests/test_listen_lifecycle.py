"""Background listener lifecycle: autostart, status, one at a time, pause/resume, stop, exit with the session.

The listener here hears an audio file instead of the microphone
(HAL_LISTEN_FILE) and uses a list of transcripts instead of the speech model
(HAL_LISTEN_FAKE_ASR), so the test opens no microphone and runs in seconds.
"""

import json
import os
import subprocess
import sys
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from listen_harness import (HAL_LISTEN, LISTEN_PY, ROOT, Terminal, hook, isolated_env, make_tmp,  # noqa: E402
                            proxy_argv, proxy_info, rmtree, standin_argv, write_config)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@unittest.skipUnless(LISTEN_PY.exists() and (ROOT / "models" / "silero_vad.onnx").exists(),
                     "needs .venv-listen and models/silero_vad.onnx")
class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        # 40 s of audio: one loud "utterance" at 1 s (the fake transcript names Claude), then silence.
        audio = np.zeros(40 * 16000, np.int16)
        t = np.arange(int(1.2 * 16000)) / 16000
        audio[16000:16000 + len(t)] = (8000 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.int16)
        self.wav = self.tmp / "a.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())
        (self.tmp / "answers.txt").write_text("Claude, lifecycle test.\n")
        write_config(self.tmp, {"autostart": True, "keyboard_quiet": 0.2, "vad_threshold": 0.3,
                                "vad_neg_threshold": 0.15})
        self.env = isolated_env(self.tmp, HAL_LISTEN_FILE=self.wav, HAL_LISTEN_FAKE_ASR=self.tmp / "answers.txt",
                                STANDIN_OUT=self.tmp / "lines.jsonl")
        self.term = None

    def tearDown(self):
        if self.term:
            self.term.close()
        subprocess.run([str(HAL_LISTEN), "stop"], env=self.env, capture_output=True, timeout=20)
        rmtree(self.tmp)

    def cli(self, *args):
        p = subprocess.run([str(HAL_LISTEN), *args], env=self.env, capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout + p.stderr

    def listener(self, state=None, timeout=15):
        path = self.tmp / "state" / "listen" / "listener.json"
        deadline = time.monotonic() + timeout
        info = {}
        while time.monotonic() < deadline:
            try:
                info = json.loads(path.read_text())
            except (OSError, ValueError):
                info = {}
            if info.get("pid") and pid_alive(info["pid"]) and (state is None or info.get("state") == state):
                return info
            if self.term:
                self.term.pump(0.1)
            else:
                time.sleep(0.1)
        self.fail(f"no listener in state {state!r}: {info}\n{self.log()}")

    def log(self):
        p = self.tmp / "state" / "listen.log"
        return p.read_text() if p.exists() else ""

    def test_lifecycle(self):
        self.term = Terminal(proxy_argv(*standin_argv()), env=self.env)
        self.term.wait_for(r"READY")
        proxy = proxy_info(self.tmp / "state")
        hook(self.env, {"hook_event_name": "SessionStart", "source": "startup"}, proxy["pid"])

        # Autostarted by the proxy; it serves that session.
        info = self.listener("listening")
        self.assertEqual(info["proxy_pid"], proxy["pid"])
        # Typed with the listener's label (hal/spoken.py).
        self.term.wait_for(r'LINE "\[spoken \d\d:\d\d\] Claude, lifecycle test."', timeout=20)

        code, out = self.cli("status")
        self.assertIn(f"listener: running (pid {info['pid']})", out)
        self.assertIn("can type: yes", out)

        code, out = self.cli("start")                  # only one listener at a time
        self.assertIn("already running", out)

        self.cli("pause")
        self.listener("paused")
        self.assertIn("paused", self.cli("status")[1])
        self.cli("resume")
        self.listener("listening")

        code, out = self.cli("stop")
        self.assertEqual(code, 0, out)
        self.assertFalse(pid_alive(info["pid"]))
        self.assertIn("listener: not running", self.cli("status")[1])

        code, out = self.cli("start")                   # by hand this time
        self.assertEqual(code, 0, out)
        again = self.listener("listening")
        self.assertNotEqual(again["pid"], info["pid"])

        # The session ends: the listener goes with it (the proxy stops it on the way out).
        self.term.send("exit 0\r")
        self.term.wait_exit()
        self.assertTrue(self.gone(again["pid"]), self.log())

    def gone(self, pid, timeout=5.0):
        deadline = time.monotonic() + timeout
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        return not pid_alive(pid)

    def test_listener_exits_when_the_session_dies(self):
        """Even if the proxy is killed outright and can't stop it, the listener notices."""
        self.term = Terminal(proxy_argv(*standin_argv()), env=self.env)
        self.term.wait_for(r"READY")
        proxy = proxy_info(self.tmp / "state")
        info = self.listener("listening")
        os.kill(proxy["pid"], 9)
        self.assertTrue(self.gone(info["pid"]), self.log())
        self.assertIn("has exited; stopping", self.log())

    def test_start_without_a_session_refuses(self):
        code, out = self.cli("start")
        self.assertEqual(code, 1)
        self.assertIn("no Claude Code session", out)


if __name__ == "__main__":
    unittest.main()
