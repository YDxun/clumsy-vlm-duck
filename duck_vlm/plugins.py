"""Ablation-friendly plugins mounted around the duck perceive-reason-act loop."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import io
import math
from typing import Any, Deque

from .config import PluginConfig
from .types import DuckState, VlmObservation

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional outside simulation image path
    Image = None
    ImageDraw = None


class DuckPlugin:
    name = "plugin"

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        return observation

    def after_step(self, token: str, before: DuckState, after: DuckState, ok: bool, context: dict[str, Any]) -> None:
        return None

    def resolve_token(self, token: str, observation: VlmObservation, context: dict[str, Any]) -> str:
        return token

    def reset(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


class ProprioceptionPlugin(DuckPlugin):
    name = "proprioception"

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if self.enabled:
            observation.proprio_text = (
                "PROPRIOCEPTION: " + observation.state.prompt_text()
                + f" | current_action={context.get('action_token') or 'none'}"
                + f" | last_result={context.get('last_result') or 'none'}"
            )
        return observation


class ActionMemoryPlugin(DuckPlugin):
    name = "action_memory"

    def __init__(self, enabled: bool = True, maxlen: int = 8):
        super().__init__(enabled)
        self.recent: Deque[str] = deque(maxlen=maxlen)

    def observe(self, token: str) -> None:
        if self.enabled and token:
            self.recent.appendleft(token.upper())

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if self.enabled:
            observation.recent_actions = list(self.recent)
            counts = Counter(self.recent)
            repeated = counts.most_common(1)[0] if counts else ("", 0)
            anti = ""
            if repeated[1] >= 3:
                anti = f" Warning: {repeated[0]} has repeated {repeated[1]} times; avoid an action loop."
            observation.history_text = (
                "RECENT_ACTION_MEMORY: " + (", ".join(self.recent) or "(none)") + anti
            )
        return observation

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "recent": list(self.recent)}


class SubgoalPlugin(DuckPlugin):
    name = "subgoal"
    NEAR_SEEN_M = 0.60   # only suggest LOOK_DOWN if it was seen this close

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.subgoal = ""
        self._last_seen_range: float | None = None

    def reset(self) -> None:
        self.subgoal = ""
        self._last_seen_range = None

    def _seen_near(self, state: DuckState) -> bool:
        """True when the target was last seen close enough to slip under the camera."""
        if state.target_visible and state.target_range_m is not None:
            self._last_seen_range = float(state.target_range_m)
        return (self._last_seen_range is not None
                and self._last_seen_range <= self.NEAR_SEEN_M)

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if not self.enabled:
            return observation
        task = (observation.task or "").lower()
        state = observation.state
        task_id = observation.task_id or ""
        if task_id == "walk_turn_stop":
            import math as _math
            distance = _math.hypot(0.8 - state.x, -0.1 - state.y)
            yaw_error = (_math.radians(-90.0) - state.heading_rad + _math.pi) % (2 * _math.pi) - _math.pi
            if distance > 0.35:
                self.subgoal = (f"Move into the target region near (0.8,-0.1). Current distance is {distance:.2f} m. Walk forward while maintaining heading.")
            elif abs(_math.degrees(yaw_error)) > 20.0:
                direction = "right" if yaw_error < 0 else "left"
                self.subgoal = (f"You are inside the target region. Turn {direction} toward -90 deg. Current heading is {_math.degrees(state.heading_rad):.0f} deg, error is {_math.degrees(yaw_error):.0f} deg.")
            else:
                self.subgoal = "Target position and heading are satisfied. Output STOP and remain stable."
            observation.subgoal = self.subgoal
            return observation
        if task_id in ("walk_to_ball", "come_to_owner", "go_to_beacon", "rough_ground_walk"):
            if state.target_range_m is None:
                tail = ("It was last seen close, so it may be under the chin: LOOK_DOWN."
                        if self._seen_near(state) else
                        "It was not seen nearby, so do not stare at the floor: explore instead.")
                self.subgoal = ("Target is not on the duck-cam, so its distance is unknown. Scan "
                                "with TURN_L/TURN_R, then FWD to a new vantage point (through a "
                                f"doorway or along open floor) if scanning does not reveal it. {tail}")
            elif state.target_range_m > 0.4:
                self.subgoal = f"Approach target; current range is {state.target_range_m:.2f} m."
            elif state.target_range_m > 0.18:
                self.subgoal = (f"Target is close ({state.target_range_m:.2f} m) and can drop below "
                                f"the camera. LOOK_DOWN to keep it in view, then finish.")
            else:
                self.subgoal = f"Target is very close ({state.target_range_m:.2f} m). Slow down and stop."
            observation.subgoal = self.subgoal
            return observation
        if any(word in task for word in ("踢", "kick", "球", "ball")):
            if state.target_range_m is None or not state.target_visible:
                tail = ("It was last seen close, so it has likely dropped below the camera: "
                        "LOOK_DOWN to see the ground."
                        if self._seen_near(state) else
                        "It was not seen nearby: scan and explore rather than staring at the floor.")
                self.subgoal = ("Ball is not on the duck-cam. Scan with TURN_L/TURN_R, then FWD to a "
                                f"new vantage point if scanning does not find it. {tail}")
            elif state.target_range_m > 0.35:
                self.subgoal = "Approach the ball until it is close and centered."
            elif abs(state.target_bearing_rad or 0.0) > 0.20:
                self.subgoal = "Fine-align the ball with the selected foot."
            else:
                self.subgoal = "Kick the ball, then verify it moved."
        elif any(word in task for word in ("过来", "come", "召唤", "summon", "主人", "owner")):
            self.subgoal = "Locate the target and approach it, then stop at a comfortable distance."
        elif any(word in task for word in ("跳舞", "dance", "开心")):
            self.subgoal = "Stabilize, perform the dance skill, then stop."
        elif any(word in task for word in ("翻滚", "roll", "trick")):
            self.subgoal = "Stabilize, perform one forward roll, then recover."
        else:
            self.subgoal = "Make observable progress toward the task and finish only when the target condition is met."
        observation.subgoal = self.subgoal
        return observation


class SearchAlignPlugin(DuckPlugin):
    """Keep the duck converging on the target instead of spinning in place.

    The state sensor only publishes range/bearing while the duck-cam actually
    holds the target, so this plugin never pretends to know where an unseen
    target is. It covers three cases:

    * target on camera and far -> a turn is wasted motion, walk forward instead;
    * target on camera and close -> align by turning toward the published bearing;
    * target off camera -> sweep consistently in the direction it was last seen,
      which needs no privileged information and stops the L/R oscillation.
    """

    name = "search_align"
    ALIGN_RAD = 0.55          # "straight ahead" band, ~31 deg
    SEEN_HOLD_STEPS = 20      # retained for compatibility with recorded runs
    MAX_SWEEP_STEPS = 6       # bounded sweep before handing control back

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self._turn_dir: str | None = None
        self._search_dir: str | None = None
        self._blind_steps = 0

    def reset(self) -> None:
        self._turn_dir = None
        self._search_dir = None
        self._blind_steps = 0

    def resolve_token(self, token: str, observation: VlmObservation, context: dict[str, Any]) -> str:
        if not self.enabled or observation.task_id == "walk_turn_stop":
            return token
        state = observation.state
        bearing = state.target_bearing_rad
        task_id = observation.task_id or ""
        if bearing is None:
            # Nothing on camera. Sweep one way so the duck scans instead of
            # flipping left/right, but only for a bounded number of steps:
            # after that the model must be free to explore (e.g. walk through a
            # doorway) rather than spin forever on the spot.
            self._turn_dir = None
            self._blind_steps += 1
            if token in ("TURN_L", "TURN_R"):
                if self._blind_steps > self.MAX_SWEEP_STEPS:
                    return token
                if self._search_dir is None:
                    self._search_dir = "TURN_R"
                return self._search_dir
            return token
        self._blind_steps = 0
        # Remember which side the target was last on so the sweep continues that way.
        self._search_dir = "TURN_L" if bearing > 0 else "TURN_R"
        rng = state.target_range_m
        if token in ("TURN_L", "TURN_R"):
            if (rng is None or rng > 0.45) and "orbit" not in task_id and "corridor" not in task_id:
                return "FWD"   # already facing it: closing the gap beats turning
            direction = "TURN_L" if bearing > 0 else "TURN_R"
            self._turn_dir = direction
            return direction
        return token

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "turn_dir": self._turn_dir,
                "search_dir": self._search_dir}

class ExplorePlugin(DuckPlugin):
    """Head for the nearest opening and keep covering new ground when blind.

    Only the floor plan is used (a map the robot may legitimately hold); the
    target's hidden position is never consulted. This replaces the old
    "spin on the spot until something appears" behaviour, which could never
    leave the room it started in.
    """

    name = "explore"
    ARRIVE_M = 0.30
    ALIGN_RAD = 0.35
    LOCO = ("FWD", "BACK", "STRAFE_L", "STRAFE_R", "TURN_L", "TURN_R", "STOP", "STAND")
    # Tasks whose goal is a pose, a specific gait, or an orbit: walking to a
    # doorway would actively fight the objective, so leave them alone.
    SKIP_TASKS = ("walk_turn_stop", "rough_ground_walk", "recover_after_fall",
                  "narrow_corridor", "orbit_stranger")
    # If the goal was on camera very recently, losing it is a local perception
    # problem (it dropped below the chin), not a reason to walk off exploring.
    SEEN_HOLD_STEPS = 8
    # After the goal has been lost for a while, first go back to where it was last
    # seen and look around, instead of resuming a long tour past it.
    SEARCH_ARRIVE_M = 0.35
    SEARCH_SWEEP_STEPS = 4

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.waypoints: list[tuple[float, float]] = []
        self.index = 0
        self._hold = 0
        self._last_step: int | None = None
        self._engaged = False
        self._last_seen_xy: tuple[float, float] | None = None
        self._searching = False
        self._swept = 0
        self._sweep_dir = "TURN_L"

    def set_waypoints(self, waypoints: Any) -> None:
        self.waypoints = [(float(w[0]), float(w[1])) for w in (waypoints or [])]
        self.index = 0

    def reset(self) -> None:
        self.index = 0
        self._hold = 0
        self._last_step = None
        self._engaged = False
        self._last_seen_xy = None
        self._searching = False
        self._swept = 0

    def _seen_hold(self, observation: VlmObservation) -> int:
        """Count down the "goal was just visible" hold; idempotent per step."""
        step = observation.step_index
        if self._last_step != step:
            self._last_step = step
            if observation.state.target_visible:
                self._hold = self.SEEN_HOLD_STEPS
                # Remember OUR OWN position at the sighting (odometry, not the
                # target ground truth) so we can come back and search here.
                self._last_seen_xy = (observation.state.x, observation.state.y)
                self._searching = False
                self._swept = 0
            elif self._hold > 0:
                self._hold -= 1
        return self._hold

    def _pending(self, state: DuckState) -> tuple[float, float] | None:
        i = self.index
        while i < len(self.waypoints):
            wx, wy = self.waypoints[i]
            if math.hypot(wx - state.x, wy - state.y) <= self.ARRIVE_M:
                i += 1
                continue
            return (wx, wy)
        return None

    def _active(self, observation: VlmObservation) -> bool:
        if not self.enabled or not self.waypoints:
            return False
        # Arm/decay the hold first: while the goal is visible this records that
        # it was just seen, so losing it does not immediately trigger a trek.
        hold = self._seen_hold(observation)
        if observation.state.target_visible:
            self._engaged = False
            return False
        if (observation.task_id or "") in self.SKIP_TASKS:
            self._engaged = False
            return False
        active = hold == 0
        if active and not self._engaged:
            # Resume the tour in order. The waypoint list is built doorway-first,
            # so staying on it guarantees the duck heads for the opening rather
            # than drifting to whatever cell happens to be nearest.
            self._pending(observation.state)
            if self._last_seen_xy is not None:
                self._searching = True
                self._swept = 0
        self._engaged = active
        return active

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if not self._active(observation) or observation.state.target_visible:
            return observation
        st = observation.state
        if self._searching and self._last_seen_xy is not None:
            sx, sy = self._last_seen_xy
            if math.hypot(sx - st.x, sy - st.y) > self.SEARCH_ARRIVE_M:
                hint = (f"SEARCH: the goal was last seen near ({sx:.2f},{sy:.2f}); go back there "
                        f"and look around before exploring further.")
            else:
                hint = ("SEARCH: you are back where the goal was last seen. Sweep with "
                        "TURN_L/TURN_R and use LOOK_DOWN/LOOK_DOWN_MORE if it may be under the chin.")
            observation.subgoal = f"{observation.subgoal} {hint}".strip() if observation.subgoal else hint
            return observation
        wp = self._pending(observation.state)
        if wp is None:
            return observation
        hint = (f"EXPLORE: the goal is not visible. Head for the next opening/waypoint at "
                f"({wp[0]:.2f},{wp[1]:.2f}) and keep moving to cover new ground.")
        observation.subgoal = f"{observation.subgoal} {hint}".strip() if observation.subgoal else hint
        return observation

    def resolve_token(self, token: str, observation: VlmObservation, context: dict[str, Any]) -> str:
        if not self._active(observation):
            return token
        state = observation.state
        if state.target_visible:
            # Do NOT rewind the exploration tour: if the goal slips out of view
            # again we must carry on from the ground we already covered instead
            # of marching back to the first waypoint (behind us).
            self._pending(state)
            return token
        if token not in self.LOCO:
            return token   # never hijack a skill token
        # 1) local search around the last sighting takes priority over the tour
        if self._searching and self._last_seen_xy is not None:
            sx, sy = self._last_seen_xy
            d = math.hypot(sx - state.x, sy - state.y)
            if d > self.SEARCH_ARRIVE_M:
                ang = math.atan2(sy - state.y, sx - state.x) - state.heading_rad
                ang = (ang + math.pi) % (2 * math.pi) - math.pi
                return ("TURN_L" if ang > 0 else "TURN_R") if abs(ang) > self.ALIGN_RAD else "FWD"
            if self._swept < self.SEARCH_SWEEP_STEPS:
                self._swept += 1
                return self._sweep_dir
            self._searching = False
        while self.index < len(self.waypoints):
            wx, wy = self.waypoints[self.index]
            dx, dy = wx - state.x, wy - state.y
            if math.hypot(dx, dy) <= self.ARRIVE_M:
                self.index += 1
                continue
            ang = math.atan2(dy, dx) - state.heading_rad
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) > self.ALIGN_RAD:
                return "TURN_L" if ang > 0 else "TURN_R"
            return "FWD"
        return token

    def directive(self, observation: VlmObservation) -> str | None:
        """Action this plugin can take on its own, or None to defer to the VLM."""
        if not self._active(observation):
            return None
        return self.resolve_token("FWD", observation, {})

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "index": self.index,
                "waypoints": len(self.waypoints), "seen_hold": self._hold,
                "searching": self._searching}


class KickStationPlugin(DuckPlugin):
    """Pick the stance before a kick: stand behind the object, on the far side from
    the target zone, facing it - then let the model choose the kicking foot.

    Geometry comes from two legitimate sources: the object's position is only used
    while the duck-cam actually sees it (the sensor withholds it otherwise), and the
    zone centre is static scene metadata. Nothing here reads a hidden target.
    """

    name = "kick_station"
    # task id -> (object body, zone name)
    STATION_TASKS = {
        "kick_ball_to_zone": ("ball", "zone_green"),
        "retrieve_ball": ("ball", "zone_green"),
        "push_ball_around_obstacle": ("ball", "zone_green"),
        "push_red_cube": ("obj_cube_red", "zone_blue"),
    }
    STANDOFF_M = 0.28
    ARRIVE_M = 0.15
    ALIGN_RAD = 0.22
    SIDE_ROUTE_M = 0.45
    BALL_CLEAR_M = 0.18

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.zones: dict[str, tuple[float, float]] = {}

    def set_zones(self, zones: dict[str, Any]) -> None:
        out = {}
        for k, v in (zones or {}).items():
            try:
                out[str(k)] = (float(v[0]), float(v[1]))
            except Exception:
                continue
        self.zones = out

    def reset(self) -> None:
        return None

    def _plan(self, observation: VlmObservation):
        """Return (approach_xy, direction_to_zone) or None when we must defer."""
        task = observation.task_id or ""
        pair = self.STATION_TASKS.get(task)
        if pair is None:
            return None
        state = observation.state
        obj = state.target_world_xyz
        if obj is None:
            return None            # blind: not our business, let explore/model handle
        zone = self.zones.get(pair[1])
        if zone is None:
            return None
        dx, dy = zone[0] - obj[0], zone[1] - obj[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        ux, uy = dx / norm, dy / norm
        approach = (obj[0] - ux * self.STANDOFF_M, obj[1] - uy * self.STANDOFF_M)
        return approach, (ux, uy), (obj[0], obj[1])

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        plan = self._plan(observation)
        if plan is None:
            return observation
        approach, (ux, uy), _obj = plan
        hint = (f"KICK STATION: stand behind the object on the far side from the zone, at "
                f"({approach[0]:.2f},{approach[1]:.2f}), then face the direction to the zone "
                f"({ux:+.2f},{uy:+.2f}) and kick it through.")
        observation.subgoal = f"{observation.subgoal} {hint}".strip() if observation.subgoal else hint
        return observation

    def directive(self, observation: VlmObservation) -> str | None:
        """Approach + fine alignment, resolved without a VLM round-trip."""
        if not self.enabled:
            return None
        plan = self._plan(observation)
        if plan is None:
            return None
        approach, (ux, uy), obj = plan
        state = observation.state
        dx, dy = approach[0] - state.x, approach[1] - state.y
        dist = math.hypot(dx, dy)
        if dist > self.ARRIVE_M:
            # Route around the object if the straight line would shove it the wrong
            # way (which happens whenever we are already on the zone side of it).
            gx, gy = dx, dy
            bx, by = obj[0] - state.x, obj[1] - state.y
            bd = math.hypot(bx, by)
            if bd > 1e-6:
                proj = (bx * dx + by * dy) / (dist if dist > 1e-6 else 1.0)
                if 0.0 < proj < dist and abs(bx * dy - by * dx) / (bd * dist if dist > 1e-6 else 1.0) < 1.0:
                    side = 1.0 if (bx * uy - by * ux) > 0 else -1.0
                    gx = obj[0] + (-uy) * side * self.SIDE_ROUTE_M - state.x
                    gy = obj[1] + (ux) * side * self.SIDE_ROUTE_M - state.y
            ang = math.atan2(gy, gx) - state.heading_rad
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            return ("TURN_L" if ang > 0 else "TURN_R") if abs(ang) > 0.35 else "FWD"
        # at the stance: face exactly along object -> zone
        err = math.atan2(uy, ux) - state.heading_rad
        err = (err + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > self.ALIGN_RAD:
            return "TURN_L" if err > 0 else "TURN_R"
        return None      # lined up: let the model pick the foot and kick

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "zones": sorted(self.zones)}


class AffordancePlugin(DuckPlugin):
    name = "affordance"

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if not self.enabled:
            return observation
        state = observation.state
        if state.target_range_m is None:
            observation.affordance_text = f"Target {state.target_name} is not detected."
        else:
            observation.affordance_text = (
                f"Target {state.target_name}: {state.target_range_m:.2f}m away, "
                f"bearing {state.target_bearing_rad:.2f}rad, "
                f"visible={'yes' if state.target_visible else 'no'}."
            )
            if state.target_uv:
                observation.affordance_text += (
                    f" Its projected head-camera pixel is "
                    f"({state.target_uv[0]:.0f},{state.target_uv[1]:.0f}); the image center is the current heading."
                )
        observation.image_jpeg = self._annotate(observation.image_jpeg, state)
        return observation

    @staticmethod
    def _annotate(jpeg: bytes, state: DuckState) -> bytes:
        if Image is None or ImageDraw is None or not jpeg:
            return jpeg
        try:
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            draw = ImageDraw.Draw(image)
            w, h = image.size
            draw.line((w // 2 - 6, h // 2, w // 2 + 6, h // 2), fill=(80, 220, 255), width=2)
            draw.line((w // 2, h // 2 - 6, w // 2, h // 2 + 6), fill=(80, 220, 255), width=2)
            if state.target_uv:
                u, v = state.target_uv
                if -10 <= u <= w + 10 and -10 <= v <= h + 10:
                    r = 9
                    draw.ellipse((u - r, v - r, u + r, v + r), outline=(255, 80, 30), width=3)
                    draw.line((u - r - 5, v, u + r + 5, v), fill=(255, 80, 30), width=1)
                    draw.line((u, v - r - 5, u, v + r + 5), fill=(255, 80, 30), width=1)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=82)
            return buf.getvalue()
        except Exception:
            return jpeg


class RecoveryPlugin(DuckPlugin):
    name = "recovery"

    def __init__(self, enabled: bool = True, repeat_threshold: int = 3):
        super().__init__(enabled)
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.last_distance: float | None = None
        self.no_progress_count = 0

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if not self.enabled:
            return observation
        hints: list[str] = []
        recent = observation.recent_actions
        if len(recent) >= self.repeat_threshold and len(set(recent[:self.repeat_threshold])) == 1:
            hints.append(f"You repeated {recent[0]} {self.repeat_threshold} times. Change strategy.")
        if self.no_progress_count >= 3:
            hints.append("The last actions made little progress. Re-observe the target and use a different direction or skill.")
        observation.recovery_hint = ("RECOVERY: " + " ".join(hints)) if hints else ""
        return observation

    def after_step(self, token: str, before: DuckState, after: DuckState, ok: bool, context: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if not ok:
            self.no_progress_count += 1
            return
        if before.target_range_m is not None and after.target_range_m is not None:
            delta = before.target_range_m - after.target_range_m
            if delta < 0.015 and token in ("FWD", "BACK", "STRAFE_L", "STRAFE_R", "TURN_L", "TURN_R"):
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0
            self.last_distance = after.target_range_m

    def recover_token(self, token: str, observation: VlmObservation) -> str:
        """Break obvious no-op loops unless the task itself is to stay still."""
        if not self.enabled:
            return token
        task = (observation.task or '').lower()
        if any(word in task for word in ('stop', 'doze', 'sleep', '??', '??')):
            return token
        recent = observation.recent_actions
        window = recent[: max(4, self.repeat_threshold + 1)]
        if len(window) >= 4 and len(set(window)) == 1 and token == window[0]:
            if token in ('STOP', 'STAND'):
                return 'TURN_R'
            if token in ('FWD', 'BACK', 'STRAFE_L', 'STRAFE_R'):
                return 'TURN_L'
        if token in ('TURN_L', 'TURN_R') and self._alternating_turns(recent):
            state = observation.state
            bearing = state.target_bearing_rad
            if bearing is not None and abs(bearing) > 0.25:
                return 'TURN_L' if bearing > 0 else 'TURN_R'
            if state.target_visible:
                return 'FWD'
        return token

    @staticmethod
    def _alternating_turns(recent: list[str]) -> bool:
        if len(recent) < 4:
            return False
        w = list(recent[:4])
        return set(w) == {'TURN_L', 'TURN_R'} and w[0] == w[2] and w[1] == w[3] and w[0] != w[1]

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "no_progress_count": self.no_progress_count}


class HumanTakeoverPlugin(DuckPlugin):
    name = "human_takeover"

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.pending: Deque[str] = deque()

    def push(self, token: str) -> None:
        if self.enabled:
            self.pending.append(token.upper())

    def pop(self) -> str | None:
        return self.pending.popleft() if self.enabled and self.pending else None

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "pending": list(self.pending)}


@dataclass
class PluginSuite:
    config: PluginConfig = field(default_factory=PluginConfig)
    plugins: list[DuckPlugin] = field(default_factory=list)
    by_name: dict[str, DuckPlugin] = field(default_factory=dict)

    @classmethod
    def build(cls, config: PluginConfig | None = None) -> "PluginSuite":
        cfg = config or PluginConfig()
        plugins: list[DuckPlugin] = [
            ProprioceptionPlugin(cfg.proprioception),
            ActionMemoryPlugin(cfg.action_memory),
            SubgoalPlugin(cfg.subgoal),
            AffordancePlugin(cfg.affordance),
            RecoveryPlugin(cfg.recovery),
            SearchAlignPlugin(cfg.search_align),
            ExplorePlugin(cfg.explore),
            KickStationPlugin(cfg.kick_station),
            HumanTakeoverPlugin(cfg.human_takeover),
        ]
        return cls(cfg, plugins, {plugin.name: plugin for plugin in plugins})

    def get(self, name: str) -> DuckPlugin | None:
        return self.by_name.get(name)

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        for plugin in self.plugins:
            observation = plugin.before_decision(observation, context)
        return observation

    def resolve_token(self, token: str, observation: VlmObservation, context: dict[str, Any]) -> str:
        for plugin in self.plugins:
            token = plugin.resolve_token(token, observation, context)
        return token

    def set_waypoints(self, waypoints: Any) -> None:
        for plugin in self.plugins:
            if isinstance(plugin, ExplorePlugin):
                plugin.set_waypoints(waypoints)

    def set_zones(self, zones: dict[str, Any]) -> None:
        for plugin in self.plugins:
            if isinstance(plugin, KickStationPlugin):
                plugin.set_zones(zones)

    def explore_directive(self, observation: VlmObservation) -> str | None:
        for plugin in self.plugins:
            if isinstance(plugin, ExplorePlugin):
                return plugin.directive(observation)
        return None

    def station_directive(self, observation: VlmObservation) -> str | None:
        for plugin in self.plugins:
            if isinstance(plugin, KickStationPlugin):
                return plugin.directive(observation)
        return None

    # Only these tasks have "get to the target" as the objective, so walking at a
    # visible target is unambiguously correct. Pose tasks (walk_turn_stop), object
    # tasks (kick/push) and gaits must not be second-guessed here: walking at the
    # ball would satisfy walk_turn_stop's distance but destroy its heading goal.
    APPROACH_TASKS = ("walk_to_ball", "come_to_owner", "go_to_beacon",
                      "search_ball_across_rooms", "rough_ground_walk")

    def reactive_directive(self, observation: VlmObservation) -> str | None:
        """Unambiguous long-range approach, resolved without a VLM round-trip.

        When the goal is plainly on camera and still far away, the correct action
        is fully determined (walk at it, or turn toward it). Spending ~1 s of the
        episode budget asking a VLM for that leaves less time to actually travel.
        Anything subtle - final approach, kicking, stopping, choosing between
        objects, or a lost target - still goes to the model.
        """
        if not self.get("explore") or not self.get("search_align"):
            return None
        if (observation.task_id or "") not in self.APPROACH_TASKS:
            return None
        state = observation.state
        if not state.target_visible or state.target_bearing_rad is None:
            return None
        rng = state.target_range_m
        if rng is None:
            return None
        bearing = state.target_bearing_rad
        if rng >= 0.45:
            if abs(bearing) < 0.35:
                return "FWD"
            return "TURN_L" if bearing > 0 else "TURN_R"
        # Final approach band (0.20~0.45 m): still unambiguous when we are
        # squarely on the target, and each VLM round-trip here costs more than
        # the step it would order. Kick/push tasks keep their own stance planner.
        if rng < 0.20 or abs(bearing) > 0.30:
            return None
        return "FWD"

    def reset(self) -> None:
        for plugin in self.plugins:
            plugin.reset()

    def after_step(self, token: str, before: DuckState, after: DuckState, ok: bool, context: dict[str, Any]) -> None:
        for plugin in self.plugins:
            plugin.after_step(token, before, after, ok, context)

    def configure(self, cfg: PluginConfig) -> None:
        self.config = cfg
        for plugin in self.plugins:
            if hasattr(plugin, "enabled"):
                plugin.enabled = bool(getattr(cfg, plugin.name))

    def status(self) -> dict[str, Any]:
        return {plugin.name: plugin.status() for plugin in self.plugins}
