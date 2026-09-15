from __future__ import annotations

import math
import tempfile
import unittest

from duck_vlm.actions import TOKENS, available_tokens, parse_action_token
from duck_vlm.config import LoopConfig, PluginConfig, VLMConfig
from duck_vlm.interpreter import DuckActionInterpreter
from duck_vlm.plugins import ExplorePlugin, RecoveryPlugin, SearchAlignPlugin
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
    def test_three_panel_live_composite(self):
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
            path = rec.save_live_dual(jpg((1, 2, 3)), jpg((40, 40, 40)), jpg((200, 200, 200)))
            self.assertTrue(path)
            with Image.open(rec.session_dir / path) as img:
                size = img.size
            self.assertEqual(size, (960, 240),
                             "duck-cam | third-person | whole-floor panels")
            rec.finish({})

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
            with Image.open(dual) as img:
                self.assertEqual(img.size, (640, 240))
            rec.finish({})


class SceneMapTest(unittest.TestCase):
    def test_home_scene_has_one_doorway_at_origin(self):
        from duck_vlm.map import doorways
        doors = doorways("duck_scenes/scenes/duck_home_v1/scene.xml")
        self.assertEqual(len(doors), 1, f"expected the mid-wall doorway, got {doors}")
        cx, cy = doors[0]["center"]
        self.assertAlmostEqual(cx, 0.0, places=2)
        self.assertAlmostEqual(cy, 0.0, places=2)
        self.assertEqual(doors[0]["through"], "x")
        self.assertGreater(doors[0]["width"], 1.0)

    def test_single_room_scenes_have_no_doorway(self):
        from duck_vlm.map import doorways
        self.assertEqual(doorways("duck_scenes/scenes/duck_workspace_v1/scene.xml"), [])

    def test_coverage_grid_covers_floor(self):
        from duck_vlm.map import coverage_waypoints, floor_bounds
        bounds = floor_bounds("duck_scenes/scenes/duck_home_v1/scene.xml")
        self.assertIsNotNone(bounds)
        pts = coverage_waypoints("duck_scenes/scenes/duck_home_v1/scene.xml")
        self.assertGreater(len(pts), 8)
        min_x, max_x, min_y, max_y = bounds
        for x, y in pts:
            self.assertGreater(x, min_x)
            self.assertLess(x, max_x)
            self.assertGreater(y, min_y)
            self.assertLess(y, max_y)


class ExplorePluginTest(unittest.TestCase):
    def _obs(self, x, y, heading, visible=False, task_id="search_ball_across_rooms"):
        state = DuckState(sim_time=0.0, x=x, y=y, z=0.12, heading_rad=heading,
                          linear_speed_mps=0.0, angular_speed_rps=0.0, upright=1.0,
                          fallen=False, target_name="ball")
        state.target_visible = visible
        state.target_range_m = 1.0 if visible else None
        state.target_bearing_rad = 0.0 if visible else None
        return VlmObservation(task="t", task_id=task_id, target="ball", image_jpeg=b"",
                              state=state, step_index=0, recent_actions=[])

    def _plugin(self):
        plugin = ExplorePlugin(True)
        plugin.set_waypoints([(0.0, 0.0), (1.0, 0.0)])   # doorway then through it
        return plugin

    def test_walks_forward_when_waypoint_is_ahead(self):
        p = self._plugin()
        obs = self._obs(-1.0, 0.0, 0.0)                  # facing +x toward door
        self.assertEqual(p.resolve_token("TURN_R", obs, {}), "FWD")

    def test_turns_toward_waypoint_when_misaligned(self):
        p = self._plugin()
        obs = self._obs(-1.0, 0.0, math.pi / 2)          # facing +y, door is +x
        self.assertEqual(p.resolve_token("FWD", obs, {}), "TURN_R")

    def test_yields_when_target_visible(self):
        p = self._plugin()
        obs = self._obs(0.5, 0.0, 0.0, visible=True)
        self.assertEqual(p.resolve_token("TURN_L", obs, {}), "TURN_L")

    def test_never_hijacks_skill_tokens(self):
        p = self._plugin()
        obs = self._obs(0.5, 0.0, 0.0)
        self.assertEqual(p.resolve_token("KICK_R", obs, {}), "KICK_R")

    def test_advances_past_reached_waypoints(self):
        p = self._plugin()
        obs = self._obs(0.05, 0.0, 0.0)                  # standing inside the doorway
        out = p.resolve_token("STOP", obs, {})
        self.assertEqual(out, "FWD", "should head for the through-point next")

    def test_does_not_explore_right_after_losing_a_close_target(self):
        p = self._plugin()
        seen = self._obs(0.40, 0.0, 0.0, visible=True)
        seen.step_index = 0
        self.assertEqual(p.resolve_token("TURN_L", seen, {}), "TURN_L")
        for step in range(1, 6):
            # STOP is a locomotion token explore would rewrite if it engaged.
            lost = self._obs(0.44, 0.0, 0.0)      # ball slipped below the camera
            lost.step_index = step
            self.assertEqual(p.resolve_token("STOP", lost, {}), "STOP",
                             "explore must not walk away from a just-seen target")

    def test_explores_again_once_the_hold_expires(self):
        p = self._plugin()
        seen = self._obs(0.40, 0.0, 0.0, visible=True)
        seen.step_index = 0
        p.resolve_token("STOP", seen, {})
        for step in range(1, 20):
            lost = self._obs(0.60, 0.0, 0.0)
            lost.step_index = step
            out = p.resolve_token("STOP", lost, {})
            if out in ("FWD", "TURN_L", "TURN_R"):
                return
        self.fail("explore never re-engaged after the seen-hold expired")

    def test_skips_pose_and_orbit_tasks(self):
        p = self._plugin()
        for task_id in ("walk_turn_stop", "rough_ground_walk", "recover_after_fall",
                        "narrow_corridor", "orbit_stranger"):
            obs = self._obs(-1.0, 0.0, 0.0, task_id=task_id)
            self.assertEqual(p.resolve_token("TURN_R", obs, {}), "TURN_R",
                             f"explore must not hijack {task_id}")
            self.assertNotIn("EXPLORE", p.before_decision(obs, {}).subgoal or "")

    def test_subgoal_verbalizes_exploration(self):
        p = self._plugin()
        obs = self._obs(-1.0, 0.0, 0.0)
        obs = p.before_decision(obs, {})
        self.assertIn("EXPLORE", obs.subgoal)


if __name__ == "__main__":
    unittest.main()