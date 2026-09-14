"""Runtime data contracts for the duck VLM harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DuckState:
    sim_time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading_rad: float = 0.0
    linear_speed_mps: float = 0.0
    angular_speed_rps: float = 0.0
    upright: float = 1.0
    fallen: bool = False
    target_name: str = "ball"
    target_visible: bool = False
    target_range_m: float | None = None
    target_bearing_rad: float | None = None
    target_elevation_rad: float | None = None
    target_uv: tuple[float, float] | None = None
    target_world_xyz: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "heading_rad": self.heading_rad,
            "linear_speed_mps": self.linear_speed_mps,
            "angular_speed_rps": self.angular_speed_rps,
            "upright": self.upright,
            "fallen": self.fallen,
            "target_name": self.target_name,
            "target_visible": self.target_visible,
            "target_range_m": self.target_range_m,
            "target_bearing_rad": self.target_bearing_rad,
            "target_elevation_rad": self.target_elevation_rad,
            "target_uv": self.target_uv,
            "target_world_xyz": self.target_world_xyz,
        }

    def prompt_text(self) -> str:
        parts = [
            f"body=({self.x:.2f},{self.y:.2f},{self.z:.2f})",
            f"heading={self.heading_rad:.2f}rad",
            f"speed={self.linear_speed_mps:.2f}m/s",
            f"turn_rate={self.angular_speed_rps:.2f}rad/s",
            f"upright={self.upright:.2f}",
            "fallen=" + ("yes" if self.fallen else "no"),
        ]
        if self.target_range_m is None:
            parts.append(f"target({self.target_name})=not_found")
        else:
            parts.append(
                f"target({self.target_name})="
                f"range {self.target_range_m:.2f}m, "
                f"bearing {self.target_bearing_rad:.2f}rad, "
                f"visible={'yes' if self.target_visible else 'no'}"
            )
        return " | ".join(parts)


@dataclass
class VlmObservation:
    task: str
    target: str = "ball"
    image_jpeg: bytes = b""
    state: DuckState = field(default_factory=DuckState)
    task_id: str = ""
    step_index: int = 0
    recent_actions: list[str] = field(default_factory=list)
    history_text: str = ""
    proprio_text: str = ""
    subgoal: str = ""
    affordance_text: str = ""
    recovery_hint: str = ""
    extra_views: list[tuple[str, bytes]] = field(default_factory=list)
    prompt: str = ""
    prompt_version: str = "duck_v0"
    plugin_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_image: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task": self.task,
            "task_id": self.task_id,
            "target": self.target,
            "step_index": self.step_index,
            "recent_actions": list(self.recent_actions),
            "history_text": self.history_text,
            "proprio_text": self.proprio_text,
            "subgoal": self.subgoal,
            "affordance_text": self.affordance_text,
            "recovery_hint": self.recovery_hint,
            "state": self.state.to_dict(),
            "prompt_version": self.prompt_version,
            "prompt": self.prompt,
            "plugin_state": self.plugin_state,
            "extra_view_names": [name for name, _ in self.extra_views],
        }
        if include_image:
            data["image_bytes"] = len(self.image_jpeg)
        return data


@dataclass
class DecisionRecord:
    step_index: int
    token: str
    source: str
    raw_response: str = ""
    provider: str = ""
    model: str = ""
    latency_s: float | None = None
    parse_recovered: bool = False
    note: str = ""
    image_path: str = ""
    observation: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""


@dataclass
class ActionResult:
    token: str
    ok: bool
    note: str
    sim_time: float
    state: DuckState = field(default_factory=DuckState)
