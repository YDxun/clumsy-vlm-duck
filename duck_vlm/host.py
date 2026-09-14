"""Server adapter that mounts DuckVlmLoop into the existing cockpit process."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import LoopConfig, VLMConfig
from .loop import DuckVlmLoop
from .state import DuckStateSensor
from .vlm import VlmDecisionClient


class DuckVlmHost:
    def __init__(self, server: Any):
        self.server = server
        self.sensor = DuckStateSensor(server.sim)
        self.vlm = VlmDecisionClient(VLMConfig.from_env())
        self.loop = DuckVlmLoop(
            vlm=self.vlm,
            state_provider=self._state,
            image_provider=self.capture_headcam,
            kick=server.kick,
            policies=server.policies.sessions,
            reset_callback=server.sim.reset,
            loop_config=LoopConfig(),
            vlm_config=self.vlm.config,
        )

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
            return self.loop.start(str(payload.get("task") or "explore the scene"),
                                   str(payload.get("target") or "ball"),
                                   auto_run=bool(payload.get("auto_run", True)),
                                   record=payload.get("record"))
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
        return self.loop.status()
