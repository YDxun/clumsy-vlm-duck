"""Server adapter that mounts DuckVlmLoop into the existing cockpit process."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import LoopConfig, VLMConfig
from .loop import DuckVlmLoop
from .scenes import catalog
from .intent import pick_target
from .state import DuckStateSensor, reset_to_spawn
from .task_manager import TaskManager
from .vlm import VlmDecisionClient


class DuckVlmHost:
    def __init__(self, server: Any):
        self.server = server
        self.scene = getattr(server, "scene_spec", None)
        if self.scene is None and getattr(server, "scene_id", None):
            self.scene = catalog().load(server.scene_id)
        self.sensor = DuckStateSensor(server.sim, self.scene)
        self.task_manager = TaskManager(server.sim, self.scene)
        self.vlm = VlmDecisionClient(VLMConfig.from_env())
        self.loop = DuckVlmLoop(
            vlm=self.vlm,
            state_provider=self._state,
            image_provider=self.capture_headcam,
            extra_image_provider=self.capture_main,
            overview_image_provider=self.capture_overview,
            kick=server.kick,
            policies=server.policies.sessions,
            reset_callback=server.sim.reset,
            spawn_setter=lambda task: reset_to_spawn(server.sim, task),
            loop_config=LoopConfig(),
            vlm_config=self.vlm.config,
            task_manager=self.task_manager,
        )
        self._install_waypoints()
        self._install_zones()

    def _install_zones(self) -> None:
        """Publish static zone centres so the kick-station planner can aim."""
        meta = getattr(self.scene, "metadata", None) or {}
        zones: dict[str, tuple[float, float]] = {}
        for z in meta.get("zones") or []:
            centre = z.get("center") or z.get("centre")
            if not isinstance(centre, (list, tuple)) or len(centre) < 2:
                continue
            try:
                xy = (float(centre[0]), float(centre[1]))
            except Exception:
                continue
            for name in (z.get("id"), z.get("body"), *(z.get("semantic_labels") or [])):
                if name:
                    zones[str(name)] = xy
        try:
            self.loop.plugins.set_zones(zones)
        except Exception:
            return

    def _install_waypoints(self) -> None:
        """Derive exploration waypoints (doorways + floor coverage) from the scene map.

        Doorways come first, each followed by a point just beyond it, then a coarse
        coverage grid anchored on the far side so the duck keeps entering new areas.
        """
        xml = getattr(self.scene, "xml", None)
        if xml is None:
            return
        try:
            from .map import coverage_waypoints, doorways
        except Exception:
            return
        spawn = (0.0, 0.0)
        try:
            meta = (getattr(self.scene, "metadata", {}) or {}).get("robot", {}) or {}
            xy = (meta.get("spawn") or {}).get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                spawn = (float(xy[0]), float(xy[1]))
        except Exception:
            spawn = (0.0, 0.0)
        try:
            doors = sorted(doorways(xml),
                           key=lambda d: (d["center"][0] - spawn[0]) ** 2 + (d["center"][1] - spawn[1]) ** 2)
            wps: list[tuple[float, float]] = []
            for d in doors:
                cx, cy = d["center"]
                wps.append((cx, cy))
                push = 1.0
                if d["through"] == "x":
                    sgn = 1.0 if cx >= spawn[0] else -1.0
                    wps.append((cx + sgn * push, cy))
                else:
                    sgn = 1.0 if cy >= spawn[1] else -1.0
                    wps.append((cx, cy + sgn * push))
            anchor = wps[-1] if wps else spawn
            wps.extend(coverage_waypoints(xml, from_xy=anchor))
            self.loop.plugins.set_waypoints(wps)
        except Exception:
            return

    def infer_target(self, task_text: str) -> tuple[str, str | None]:
        """Choose the body the sensor tracks, from the sentence itself.

        Returns (body_name, the label that matched). Falls back to the scene ball so
        an unparsed sentence still behaves like the previous default rather than
        tracking nothing.
        """
        scene = self.scene
        if scene is None:
            return "ball", None
        try:
            hits = scene.scan_text(task_text)
        except Exception:
            return "ball", None
        if not hits:
            return "ball", None
        # For "kick the ball into the green zone" the longer match is the zone, but
        # the thing to act on is the ball, so prefer a non-zone entity there.
        return pick_target(hits, task_text)

    def _state(self, target: str, sim_time: float):
        return self.sensor.snapshot(target, sim_time)

    def capture_headcam(self) -> bytes:
        jpg = bytes(getattr(self.server, "last_headcam_jpg", b"") or b"")
        age = time.monotonic() - float(getattr(self.server, "last_headcam_at", 0.0))
        if jpg and age <= self.loop.loop_config.stale_image_s:
            return jpg
        # The renderer is normally owned by headcam_loop. This fallback is used
        # for headless/manual API calls before the first streamed frame arrives.
        # Every EGL call must go through the server's single render worker.
        def _render():
            self.server._sync_render_state()
            return self.server.render_headcam()
        jpg = self.server.render_exec.submit(_render).result(timeout=5.0)
        self.server.last_headcam_jpg = jpg
        self.server.last_headcam_at = time.monotonic()
        return jpg

    def recording(self) -> bool:
        return bool(self.loop.recorder.active)

    def record_demo_frame(self) -> str:
        """Append one duck-cam | whole-floor-overview frame to the demo video."""
        if not self.loop.recorder.active:
            return ""
        head = bytes(getattr(self.server, "last_headcam_jpg", b"") or b"")
        track = bytes(getattr(self.server, "last_main_jpg", b"") or b"")
        scene = bytes(getattr(self.server, "last_scene_jpg", b"") or b"")
        return self.loop.recorder.save_live_dual(head, track, scene)

    def capture_main(self) -> bytes:
        """Third-person tracking view: the duck fills the frame."""
        jpg = bytes(getattr(self.server, "last_main_jpg", b"") or b"")
        age = time.monotonic() - float(getattr(self.server, "last_main_at", 0.0))
        if jpg and age <= self.loop.loop_config.stale_image_s:
            return jpg
        def _render():
            self.server._sync_render_state()
            return self.server.render_main()
        jpg = self.server.render_exec.submit(_render).result(timeout=5.0)
        self.server.last_main_jpg = jpg
        self.server.last_main_at = time.monotonic()
        return jpg

    def capture_overview(self) -> bytes:
        """Whole-floor overview: the complete map with the duck on it."""
        jpg = bytes(getattr(self.server, "last_scene_jpg", b"") or b"")
        age = time.monotonic() - float(getattr(self.server, "last_scene_at", 0.0))
        if jpg and age <= self.loop.loop_config.stale_image_s:
            return jpg
        def _render():
            self.server._sync_render_state()
            return self.server.render_overview()
        jpg = self.server.render_exec.submit(_render).result(timeout=5.0)
        self.server.last_scene_jpg = jpg
        self.server.last_scene_at = time.monotonic()
        return jpg

    def head_override(self) -> dict[str, float] | None:
        """Latched head pitch/yaw offsets, applied by the sim each control step."""
        return self.loop.head_override()

    def tick(self, sim_time: float) -> np.ndarray:
        return self.loop.tick(sim_time, dt=0.02)

    def configure(self, values: dict[str, Any]) -> dict[str, Any]:
        return self.loop.set_config(values)

    def control(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        action = (action or "status").lower()
        payload = payload or {}
        if isinstance(payload.get('config'), dict):
            self.loop.set_config(payload['config'])
        vlm_values = dict(payload.get('vlm') or {})
        if payload.get('mode'):
            vlm_values['mode'] = payload['mode']
        if vlm_values:
            self.loop.set_config({'vlm': vlm_values})
        if isinstance(payload.get('plugins'), dict):
            self.loop.set_config({'plugins': payload['plugins']})
        if action == "start":
            task_text = str(payload.get("task") or "explore the scene")
            explicit = str(payload.get("target") or "").strip()
            if explicit:
                target, matched_on = explicit, None
            else:
                target, matched_on = self.infer_target(task_text)
            state = self.loop.start(task_text, target,
                                    auto_run=bool(payload.get("auto_run", True)),
                                    record=payload.get("record"))
            state["target_inferred_from"] = matched_on
            state["target_auto"] = not bool(explicit)
            return state
        if action == "pause":
            return self.loop.pause()
        if action == "resume":
            return self.loop.resume()
        if action == "step":
            return self.loop.step_once()
        if action == "stop":
            return self.loop.stop(str(payload.get("reason") or "user_stop"))
        return self.loop.status()

    def human_action(self, token: str) -> dict[str, Any]:
        ok = self.loop.enqueue_human(token)
        status = self.loop.status()
        status["accepted"] = ok
        return status

    def status(self) -> dict[str, Any]:
        status = self.loop.status()
        status["scene_id"] = getattr(self.scene, "scene_id", None)
        status["scene_tasks"] = [t.get("id") for t in self.scene.list_tasks()] if self.scene else []
        return status
