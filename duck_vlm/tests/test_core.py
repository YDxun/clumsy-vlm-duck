from __future__ import annotations

import tempfile
import unittest

from duck_vlm.actions import TOKENS, available_tokens, parse_action_token
from duck_vlm.config import LoopConfig, PluginConfig, VLMConfig
from duck_vlm.interpreter import DuckActionInterpreter
from duck_vlm.loop import DuckVlmLoop
from duck_vlm.plugins import RecoveryPlugin
from duck_vlm.recorder import export_alpaca
from duck_vlm.types import DuckState
from duck_vlm.vlm import VlmReply


class FakeKick:
    def __init__(self):
        self.name = None
        self.phase = None
        self.events = []

    def request(self, name, available, fallen=False):
        if name not in available:
            return {"status": "rejected", "message": "missing policy"}
        self.name = name
        self.phase = "preparing"
        return {"status": "accepted"}

    def finish(self, status, message):
        self.name = None
        self.phase = None


class FakeVlm:
    def __init__(self):
        self.config = VLMConfig(mode="dry_run", model="fake")

    def configure(self, values):
        self.config.update(values)

    def decide(self, observation, allowed=None, mode=None):
        return VlmReply("STOP", "STOP", "test", "fake", 0.001, False)


class ActionsTest(unittest.TestCase):
    def test_exact_and_recovered_tokens(self):
        self.assertEqual(parse_action_token("FWD"), "FWD")
        self.assertEqual(parse_action_token("```text\nTURN_LEFT\n```"), "TURN_L")
        self.assertEqual(parse_action_token('{"action":"kick right"}'), "KICK_R")
        self.assertEqual(parse_action_token("I choose 前进"), "FWD")

    def test_invalid_token(self):
        with self.assertRaises(ValueError):
            parse_action_token("teleport", allowed=TOKENS)

    def test_kick_token_is_offered_when_policy_exists(self):
        tokens = available_tokens({"ball_kick_left": object()})
        self.assertIn("KICK_L", tokens)
        self.assertNotIn("KICK_R", tokens)


class InterpreterTest(unittest.TestCase):
    def test_motion_is_bounded(self):
        kick = FakeKick()
        interp = DuckActionInterpreter(kick, {"ball_kick_left": object()})
        first = interp.start("FWD")
        self.assertFalse(first.done)
        self.assertAlmostEqual(float(first.command[0]), 0.35)
        done = None
        for _ in range(40):
            done = interp.tick(0.02)
            if done.done:
                break
        self.assertIsNotNone(done)
        self.assertTrue(done.done)

    def test_skill_requires_policy(self):
        kick = FakeKick()
        interp = DuckActionInterpreter(kick, {})
        result = interp.start("KICK_L")
        self.assertTrue(result.done)
        self.assertIn("missing", result.note)


class RecoveryTest(unittest.TestCase):
    def test_repeated_stop_is_recovered(self):
        plugin = RecoveryPlugin(True)
        obs = type('Obs', (), {'task': 'find ball', 'recent_actions': ['STOP', 'STOP', 'STOP', 'STOP']})()
        self.assertEqual(plugin.recover_token('STOP', obs), 'TURN_R')

    def test_explicit_stop_task_is_not_overridden(self):
        plugin = RecoveryPlugin(True)
        obs = type('Obs', (), {'task': 'stop moving', 'recent_actions': ['STOP', 'STOP', 'STOP', 'STOP']})()
        self.assertEqual(plugin.recover_token('STOP', obs), 'STOP')


class LoopTest(unittest.TestCase):
    def test_loop_records_transition_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            kick = FakeKick()
            cfg = LoopConfig(max_steps=2, run_dir=tmp, plugins=PluginConfig(affordance=False))
            loop = DuckVlmLoop(
                vlm=FakeVlm(),
                state_provider=lambda target, t: DuckState(sim_time=t, target_name=target),
                image_provider=lambda: b"fake-jpeg",
                kick=kick,
                policies={},
                loop_config=cfg,
                vlm_config=FakeVlm().config,
            )
            loop.start("test stop", "ball", auto_run=True, record=True)
            for i in range(80):
                loop.tick(i * 0.02, 0.02)
                if loop.step_index >= 2 or not loop.running:
                    break
            self.assertGreaterEqual(loop.step_index, 1)
            session = loop.recorder.session_dir
            self.assertIsNotNone(session)
            out = export_alpaca(session)
            self.assertTrue(out.exists())
            self.assertIn('"STOP"', out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()