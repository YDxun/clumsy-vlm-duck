"""Ablation-friendly plugins mounted around the duck perceive-reason-act loop."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import io
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

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.subgoal = ""

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        if not self.enabled:
            return observation
        task = (observation.task or "").lower()
        state = observation.state
        if any(word in task for word in ("踢", "kick", "球", "ball")):
            if state.target_range_m is None or not state.target_visible:
                self.subgoal = "Find the ball in the head camera and turn toward it."
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
        return token

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
            HumanTakeoverPlugin(cfg.human_takeover),
        ]
        return cls(cfg, plugins, {plugin.name: plugin for plugin in plugins})

    def get(self, name: str) -> DuckPlugin | None:
        return self.by_name.get(name)

    def before_decision(self, observation: VlmObservation, context: dict[str, Any]) -> VlmObservation:
        for plugin in self.plugins:
            observation = plugin.before_decision(observation, context)
        return observation

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