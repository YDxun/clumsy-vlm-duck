from __future__ import annotations

import tempfile
import unittest

from duck_vlm.actions import TOKENS, available_tokens, parse_action_token
from duck_vlm.config import LoopConfig, PluginConfig, VLMConfig
from duck_vlm.interpreter import DuckActionInterpreter
from duck_vlm.plugins import RecoveryPlugin, SearchAlignPlugin
from duck_vlm.recorder import EpisodeRecorder, export_alpaca
from duck_vlm.types import DuckState, VlmObservation
from duck_vlm.loop import DuckVlmLoop
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


class HeadActionTest(unittest.TestCase):
    def setUp(self):
        self.interp = DuckActionInterpreter(FakeKick(), {})

    def _run(self, token):
        self.interp.start(token)
        while self.interp.busy:
            tick = self.interp.tick(0.02)
        return tick

    def test_head_tokens_are_registered(self):
        self.assertIn("LOOK_DOWN", TOKENS)
        self.assertIn("HEAD_CENTER", TOKENS)
        self.assertIn("LOOK_DOWN", available_tokens({"ball_kick_left": object()}))

    def test_look_down_latches_after_completion(self):
        tick = self._run("LOOK_DOWN")
        self.assertIn("head", tick.note)
        targets = self.interp.head_targets()
        self.assertIsNotNone(targets)
        # head_pitch positive verified to pitch the head down
        self.assertGreater(targets["head_pitch"], 0.0)

    def test_head_center_restores_nominal(self):
        self._run("LOOK_DOWN")
        self._run("HEAD_CENTER")
        self.assertEqual(self.interp.head_targets()["head_pitch"], 0.0)

    def test_cancel_clears_head_pose(self):
        self._run("LOOK_DOWN")
        self.interp.cancel(reset_action=True)
        self.assertIsNone(self.interp.head_targets())

    def test_head_action_emits_no_velocity(self):
        tick = self.interp.start("LOOK_DOWN")
        self.assertEqual(list(tick.command), [0.0, 0.0, 0.0])


class SearchAlignTest(unittest.TestCase):
    def _obs(self, bearing, visible, rng, task_id="walk_to_ball"):
        state = DuckState(sim_time=0.0, x=0.0, y=0.0, z=0.12, heading_rad=0.0,
                          linear_speed_mps=0.0, angular_speed_rps=0.0, upright=1.0,
                          fallen=False, target_name="ball")
        state.target_bearing_rad = bearing
        state.target_range_m = rng
        state.target_visible = visible
        return VlmObservation(task="t", task_id=task_id, target="ball", image_jpeg=b"",
                              state=state, step_index=0, recent_actions=[])

    def test_blind_turns_sweep_one_way(self):
        plugin = SearchAlignPlugin(True)
        out = [plugin.resolve_token(tok, self._obs(None, False, None), {})
               for tok in ("TURN_L", "TURN_R", "TURN_L", "TURN_R")]
        self.assertEqual(len(set(out)), 1, f"oscillated while blind: {out}")

    def test_blind_sweep_is_bounded_then_releases_control(self):
        plugin = SearchAlignPlugin(True)
        obs = self._obs(None, False, None)
        out = [plugin.resolve_token("TURN_L", obs, {}) for _ in range(10)]
        self.assertEqual(set(out[:6]), {"TURN_R"}, "sweep should be single-direction")
        self.assertEqual(out[-1], "TURN_L", "model must regain control after the sweep"
                            " so it can explore instead of spinning forever")

    def test_blind_does_not_hijack_stop(self):
        plugin = SearchAlignPlugin(True)
        self.assertEqual(plugin.resolve_token("STOP", self._obs(None, False, None), {}), "STOP")

    def test_visible_far_turn_becomes_forward(self):
        plugin = SearchAlignPlugin(True)
        self.assertEqual(plugin.resolve_token("TURN_L", self._obs(0.10, True, 1.5), {}), "FWD")

    def test_visible_close_turn_still_aligns(self):
        plugin = SearchAlignPlugin(True)
        self.assertEqual(plugin.resolve_token("TURN_L", self._obs(-0.10, True, 0.25), {}), "TURN_R")


class DualViewRecordingTest(unittest.TestCase):
    def test_saves_secondary_view_and_composite(self):
        try:
            from PIL import Image
        except Exception:
            self.skipTest("PIL unavailable")
        import io as _io
        with tempfile.TemporaryDirectory() as tmp:
            rec = EpisodeRecorder(tmp)
            rec.start("t", "ball", "zero_shot")
            def jpg(color):
                buf = _io.BytesIO()
                Image.new("RGB", (320, 240), color).save(buf, format="JPEG")
                return buf.getvalue()
            paths = rec.save_extra_views(jpg((10, 20, 30)), [("scene", jpg((90, 90, 90)))], 0)
            self.assertIn("scene", paths)
            self.assertIn("dual", paths)
            session = rec.session_dir
            dual = session / paths["dual"]
            self.assertTrue(dual.exists())
            self.assertEqual(Image.open(dual).size, (640, 240))
            rec.finish({})


if __name__ == "__main__":
    unittest.main()