"""Adapter around the scene pack's execution-based TaskEvaluator."""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

from .scenes import SceneSpec


class TaskManager:
    def __init__(self, sim: Any, scene: SceneSpec | None):
        self.sim = sim
        self.scene = scene
        self._module = self._load_evaluator_module(scene)
        self.evaluator = None
        self.task = None
        self.last: dict[str, Any] = {
            "active": False, "task_id": None, "success": False,
            "failure": False, "reason": "", "progress": 0.0,
        }

    @staticmethod
    def _load_evaluator_module(scene: SceneSpec | None):
        if scene is None:
            return None
        path = scene.root.parent.parent / "validation_core.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("duck_scene_validation_core", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @property
    def available(self) -> bool:
        return self.scene is not None and self._module is not None

    def start(self, task_text: str, target: str | None = None) -> dict[str, Any] | None:
        if not self.available or self.scene is None:
            return None
        task = self.scene.match_task(task_text)
        if task is None:
            # Free-form prompt: reuse whichever curated task grades the same object, so
            # the episode still gets truth-based scoring instead of running blind.
            try:
                task = self.scene.match_task_semantic(task_text, target)
            except Exception:
                task = None
        if task is None:
            self.evaluator = None
            self.task = None
            self.last = {"active": False, "task_id": None, "success": False,
                         "failure": False, "reason": "no_matching_task", "progress": 0.0}
            return None
        self.task = task
        self.evaluator = self._module.TaskEvaluator(self.sim.model, self.sim.data, task, dt=0.02)
        self.last = {"active": True, "task_id": task.get("id"), "success": False,
                     "failure": False, "reason": "", "progress": 0.0}
        return task

    def step(self, dt: float = 0.02) -> dict[str, Any]:
        if self.evaluator is None:
            return self.last
        success, failure, reason = self.evaluator.step(dt)
        self.last = {"active": True, "task_id": (self.task or {}).get("id"),
                     "success": bool(success), "failure": bool(failure),
                     "reason": str(reason or ""),
                     "progress": float(getattr(self.evaluator, "success_timer", 0.0))}
        return self.last

    def stop(self) -> None:
        if self.evaluator is not None:
            self.last = dict(self.last)
            self.last["active"] = False

    def status(self) -> dict[str, Any]:
        return dict(self.last)