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
