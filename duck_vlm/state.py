"""MuJoCo state adapter for VLM prompts and reward/recording data."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .scenes import SceneSpec
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
    def __init__(self, sim: Any, scene: SceneSpec | None = None):
        self.sim = sim
        self.scene = scene
        self._target_cache: dict[str, int | None] = {}

    def resolve_target(self, target_name: str) -> int | None:
        key = (target_name or "ball").strip().lower()
        canonical = _TARGET_ALIASES.get(key, key)
        if self.scene is not None:
            resolved = self.scene.resolve_body(target_name)
            if resolved:
                canonical = resolved
        cache_key = f"{target_name}|{canonical}"
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]
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
        self._target_cache[cache_key] = body_id
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
        in_front = bool(target_local[0] > 0.03)
        uv = self._project_to_headcam(target_world) if in_front else None
        if uv is not None and self._is_occluded(target_world, target_id):
            uv = None
        state.target_uv = uv
        state.target_visible = uv is not None
        if state.target_visible:
            # The head camera has the target, so a detector + depth sensor could
            # report the same range/bearing. This is the only case where the
            # privileged simulator state is allowed into the prompt.
            state.target_world_xyz = tuple(float(x) for x in target_world[:3])
            state.target_range_m = float(np.linalg.norm(target_local))
            state.target_bearing_rad = float(math.atan2(target_local[1], target_local[0]))
            state.target_elevation_rad = float(math.atan2(target_local[2], math.hypot(target_local[0], target_local[1])))
        else:
            # Occluded or out of frame: there is no depth/range sensor on the duck,
            # so withhold ground truth instead of leaking the answer.
            state.target_world_xyz = None
            state.target_range_m = None
            state.target_bearing_rad = None
            state.target_elevation_rad = None
        return state
    def _is_occluded(self, target_world: np.ndarray, target_id: int) -> bool:
        """True when scene geometry blocks the head camera's line of sight."""
        try:
            import mujoco
            from duck_play.perception.camera import DuckHeadCam
            cam = DuckHeadCam(width=320, height=240, pitch_up_deg=-20, fwd_m=0.09, up_m=0.03)
            origin = np.asarray(cam.pose(self.sim)[0], dtype=np.float64)
        except Exception:
            return False
        direction = np.asarray(target_world, dtype=np.float64) - origin
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return False
        direction = direction / dist
        geomid = np.array([-1], dtype=np.int32)
        try:
            hit = float(mujoco.mj_ray(self.sim.model, self.sim.data, origin, direction,
                                      None, 1, -1, geomid))
        except Exception:
            return False
        if hit < 0.0 or hit >= dist - 0.02:
            return False
        if hit < 0.05:
            return False  # the duck's own head/neck shell
        gid = int(geomid[0])
        if gid < 0:
            return False
        return int(self.sim.model.geom_bodyid[gid]) != int(target_id)

    def _project_to_headcam(self, world_xyz: np.ndarray) -> tuple[float, float] | None:
        try:
            from duck_play.perception.camera import DuckHeadCam
            return DuckHeadCam(width=320, height=240, pitch_up_deg=-20, fwd_m=0.09, up_m=0.03).project(world_xyz, self.sim)
        except Exception:
            return None