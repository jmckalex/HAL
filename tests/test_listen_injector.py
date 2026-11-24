"""The injector: typing addressed text into the session, and holding it while that would be unsafe.

The proxy's replies are driven by the real session-state reducer and verdict,
so these tests cover the injector state machine for an idle session, a busy
one, and one showing a dialog.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hal.listener import control, session, settings  # noqa: E402
from hal.listener.injector import Injector  # noqa: E402

CFG = dict(settings.DEFAULTS)
BASH = {"command": "make"}


class FakeProxy:
    """Answers like the pty proxy, from a session state and keyboard it holds."""

    def __init__(self, clock):
        self.clock = clock
        self.state = session.apply_event(session.empty(), {"hook_event_name": "SessionStart"}, 0)
        self.keyboard = session.Keyboard()
        self.typed = []          # (id, text)
        self.requests = []
        self.reachable = True
        self.drop_first_reply = False

    def event(self, name, **fields):
        self.state = session.apply_event(self.state, dict(hook_event_name=name, **fields), self.clock())

    def type(self, text, ident):
        self.requests.append(ident)
        if not self.reachable:
            raise control.Unreachable("connection refused")
        if any(i == ident for i, _ in self.typed):
            return {"ok": True, "typed": len(text), "duplicate": True}
        ok, reason, detail = session.verdict(self.state, self.keyboard, self.clock(), CFG)
        if not ok:
            return {"ok": False, "reason": reason, "detail": detail}
        self.typed.append((ident, text))
        if self.drop_first_reply:
            self.drop_first_reply = False
            raise control.Unreachable("timed out")
        return {"ok": True, "typed": len(text)}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class InjectorTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.proxy = FakeProxy(self.clock)
        self.said = []
        self.logged = []
        self.inj = Injector(CFG, client_factory=lambda: self, say=self.said.append, clock=self.clock,
                            log=self.logged.append)

    # The injector calls client_factory().type(); this test case plays the client.
    def type(self, text, ident):
        return self.proxy.type(text, ident)

    def advance(self, seconds, step=0.25):
        end = self.clock.t + seconds
        while self.clock.t < end:
            self.clock.t += step
            self.inj.tick()

    def texts(self):
        return [t for _, t in self.proxy.typed]

    def test_idle_session_types_at_once(self):
        self.inj.submit("Claude, open my thesis.", speech_end=99.0)
        self.assertEqual(self.texts(), ["Claude, open my thesis."])
        self.assertFalse(self.inj.holding)
        self.assertEqual(self.said, [])
        self.assertAlmostEqual(self.inj.submitted[0][1], 1.0)   # latency from end of speech

    def test_busy_session_types_too(self):
        self.proxy.event("UserPromptSubmit", prompt="earlier")
        self.inj.submit("Claude, and another thing.", speech_end=99.5)
        self.assertEqual(self.texts(), ["Claude, and another thing."])

    def test_dialog_holds_and_announces_once(self):
        self.proxy.event("PermissionRequest", tool_name="Bash", tool_input=BASH)
        self.inj.submit("Claude, first.", speech_end=99.0)
        self.inj.submit("Claude, second.", speech_end=99.5)
        self.advance(5)
        self.assertEqual(self.texts(), [])
        self.assertTrue(self.inj.holding)
        self.assertEqual(self.said, [CFG["dialog_notice"]])     # said once, not per request or retry
        self.proxy.event("PostToolUse", tool_name="Bash", tool_input=BASH)
        self.advance(1)
        self.assertEqual(self.texts(), ["Claude, first.", "Claude, second."])   # in order
        self.assertFalse(self.inj.holding)

    def test_dialog_answered_at_the_keyboard(self):
        self.proxy.event("PermissionRequest", tool_name="Bash", tool_input=BASH)
        self.inj.submit("Claude, check the build.", speech_end=99.0)
        self.advance(2)
        self.proxy.keyboard.feed(b"\r", self.clock.t)            # the user approves
        self.advance(0.5)
        self.assertEqual(self.texts(), [])                        # still typing-quiet after the keystroke
        self.advance(CFG["keyboard_quiet"] + 0.5)
        self.assertEqual(self.texts(), ["Claude, check the build."])

    def test_new_episode_announces_again(self):
        self.proxy.event("PermissionRequest", tool_name="Bash", tool_input=BASH)
        self.inj.submit("Claude, one.", speech_end=99.0)
        self.advance(1)
        self.proxy.event("Stop")
        self.advance(1)
        self.proxy.event("PermissionRequest", tool_name="Bash", tool_input=BASH)
        self.inj.submit("Claude, two.", speech_end=self.clock.t)
        self.advance(1)
        self.assertEqual(self.said, [CFG["dialog_notice"], CFG["dialog_notice"]])

    def test_draft_holds_with_its_own_notice(self):
        self.proxy.keyboard.feed(b"half a sentence", self.clock.t)
        self.inj.submit("Claude, hello.", speech_end=99.0)
        self.advance(3)
        self.assertEqual(self.said, [CFG["draft_notice"]])
        self.proxy.keyboard.feed(b"\r", self.clock.t)
        self.advance(CFG["keyboard_quiet"] + 0.5)
        self.assertEqual(self.texts(), ["Claude, hello."])

    def test_typing_at_keyboard_is_waited_out_quietly(self):
        self.proxy.keyboard.feed(b"\x1b[A", self.clock.t)        # arrow key: no draft
        self.inj.submit("Claude, hello.", speech_end=99.0)
        self.assertEqual(self.texts(), [])
        self.advance(CFG["keyboard_quiet"] + 0.3)
        self.assertEqual(self.texts(), ["Claude, hello."])
        self.assertEqual(self.said, [])

    def test_held_request_expires(self):
        self.proxy.event("PermissionRequest", tool_name="Bash", tool_input=BASH)
        self.inj.submit("Claude, too late.", speech_end=99.0)
        self.advance(CFG["hold_seconds"] + 1, step=1.0)
        self.assertFalse(self.inj.holding)
        self.proxy.event("Stop")
        self.advance(1)
        self.assertEqual(self.texts(), [])
        self.assertTrue(any("dropped" in line for line in self.logged))

    def test_unreachable_drops_and_announces_rarely(self):
        self.proxy.reachable = False
        self.inj.submit("Claude, one.", speech_end=99.0)
        self.inj.submit("Claude, two.", speech_end=99.0)
        self.assertFalse(self.inj.holding)
        self.assertEqual(self.said, [CFG["unreachable_notice"]])

    def test_retry_uses_the_same_id(self):
        # The proxy typed the line but the reply was lost: the injector's client
        # (control.ProxyClient.type) retries with the same id and is answered from
        # the record, so the line is typed once.
        client = control.ProxyClient("/nonexistent")
        calls = []

        def request(obj):
            calls.append(obj["id"])
            if len(calls) == 1:
                raise control.Unreachable("timed out")
            return {"ok": True, "duplicate": True}

        client.request = request
        self.assertTrue(client.type("hello", "abc")["ok"])
        self.assertEqual(calls, ["abc", "abc"])

    def test_session_ended_drops(self):
        self.proxy.event("SessionEnd", reason="prompt_input_exit")
        self.inj.submit("Claude, hello.", speech_end=99.0)
        self.assertFalse(self.inj.holding)
        self.assertEqual(self.texts(), [])


if __name__ == "__main__":
    unittest.main()
