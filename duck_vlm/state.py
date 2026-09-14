"""MuJoCo state adapter for VLM prompts and reward/recording data."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .types import DuckState


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = float(quat[0])
    xyz = np.asarray(quat[1:4], dtype=np.float64)
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


_TARGET_ALIASES = {
    "ball": "ball", "sphere": "ball", "orange ball": "ball", "足球": "ball", "球": "ball",
    "beacon": "beacon", "marker": "beacon", "信标": "beacon", "目标": "beacon",
    "owner": "owner", "person": "owner", "主人": "owner", "人": "owner",
}


class DuckStateSensor:
    def __init__(self, sim: Any):
        self.sim = sim
        self._target_cache: dict[str, int | None] = {}

    def resolve_target(self, target_name: str) -> int | None:
        key = (target_name or "ball").strip().lower()
        canonical = _TARGET_ALIASES.get(key, key)
        if canonical in self._target_cache:
            return self._target_cache[canonical]
        body_id: int | None = None
        if canonical == "ball":
            body_id = int(getattr(self.sim, "ball_body", -1))
        if canonical == "owner":
            for candidate in ("owner", "person", "human"):
                try:
                    import mujoco
                    body_id = int(mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, candidate))
                except Exception:
                    body_id = -1
                if body_id is not None and body_id >= 0:
                    break
        if body_id is None or body_id < 0:
            try:
                import mujoco
                body_id = int(mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, canonical))
            except Exception:
                body_id = -1
        if body_id is not None and body_id < 0:
            body_id = None
        self._target_cache[canonical] = body_id
        return body_id

    def snapshot(self, target_name: str = "ball", sim_time: float = 0.0) -> DuckState:
        d = self.sim.data
        trunk_id = int(self.sim.trunk_id)
        trunk = np.asarray(d.xpos[trunk_id], dtype=np.float64).copy()
        quat = np.asarray(d.xquat[trunk_id], dtype=np.float64).copy()
        xmat = np.asarray(d.xmat[trunk_id], dtype=np.float64).reshape(3, 3)
        heading = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
        gravity = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        speed = float(np.linalg.norm(np.asarray(d.qvel[:2], dtype=np.float64)))
        gyro_adr = int(getattr(self.sim, "imu_gyro_adr", 0))
        gyro = np.asarray(d.sensordata[gyro_adr:gyro_adr + 3], dtype=np.float64)
        state = DuckState(
            sim_time=float(sim_time), x=float(trunk[0]), y=float(trunk[1]), z=float(trunk[2]),
            heading_rad=float(heading), linear_speed_mps=speed,
            angular_speed_rps=float(np.linalg.norm(gyro)), upright=float(-gravity[2]),
            fallen=bool((-gravity[2]) < 0.5), target_name=target_name or "ball",
        )
        target_id = self.resolve_target(target_name)
        if target_id is None:
            return state
        target_world = np.asarray(d.xpos[target_id], dtype=np.float64).copy()
        target_local = _quat_rotate_inverse(quat, target_world - trunk)
        state.target_world_xyz = tuple(float(x) for x in target_world[:3])
        state.target_range_m = float(np.linalg.norm(target_local))
        state.target_bearing_rad = float(math.atan2(target_local[1], target_local[0]))
        state.target_elevation_rad = float(math.atan2(target_local[2], math.hypot(target_local[0], target_local[1])))
        state.target_uv = self._project_to_headcam(target_world)
        state.target_visible = bool(state.target_uv is not None and target_local[0] > 0.03)
        return state

    def _project_to_headcam(self, world_xyz: np.ndarray) -> tuple[float, float] | None:
        try:
            from duck_play.perception.camera import DuckHeadCam
            return DuckHeadCam(width=320, height=240, pitch_up_deg=-20, fwd_m=0.09, up_m=0.03).project(world_xyz, self.sim)
        except Exception:
            return None