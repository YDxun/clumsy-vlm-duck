"""Ground discrete duck action tokens into the existing ONNX/LocalKick stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actions import ACTION_SPECS


@dataclass
class ExecutionTick:
    command: np.ndarray
    done: bool
    note: str
    token: str


class DuckActionInterpreter:
    """Execute exactly one token at a time.

    Motion tokens hold a bounded velocity command for a short calibrated
    duration. Skill tokens delegate to LocalKick/Official ONNX policies and
    complete when the underlying policy lifecycle returns to idle.
    """

    def __init__(self, kick: Any, policies: dict[str, Any] | None = None):
        self.kick = kick
        self.policies = policies or {}
        self.token: str | None = None
        self._elapsed = 0.0
        self._duration = 0.0
        self._command = np.zeros(3, dtype=np.float32)
        self._started_skill = False
        self._request_ok = False

    def update_policies(self, policies: dict[str, Any]) -> None:
        self.policies = policies or {}

    @property
    def busy(self) -> bool:
        return self.token is not None

    def available(self, policies: dict[str, Any] | None = None) -> tuple[str, ...]:
        if policies is None:
            return tuple(ACTION_SPECS)
        available = set(policies)
        return tuple(
            token for token, spec in ACTION_SPECS.items()
            if spec.policy is None or spec.policy in available
        )

    def start(self, token: str, fallen: bool = False) -> ExecutionTick:
        token = token.upper().strip()
        spec = ACTION_SPECS.get(token)
        if spec is None:
            return ExecutionTick(np.zeros(3, dtype=np.float32), True, f"unknown token {token}", token)
        self.token = token
        self._elapsed = 0.0
        self._started_skill = False
        self._request_ok = False
        if spec.terminal:
            self.token = None
            return ExecutionTick(np.zeros(3, dtype=np.float32), True, "task marked done", token)
        if spec.policy is not None:
            if fallen and spec.request in ("kickL", "kickR"):
                self.token = None
                return ExecutionTick(np.zeros(3, dtype=np.float32), True, "duck is down", token)
            request_name = spec.request or spec.policy
            result = self.kick.request(request_name, self.policies, fallen=fallen)
            self._request_ok = result.get("status") == "accepted"
            if not self._request_ok:
                note = str(result.get("message") or result.get("reason") or "action rejected")
                self.token = None
                return ExecutionTick(np.zeros(3, dtype=np.float32), True, note, token)
            return ExecutionTick(np.zeros(3, dtype=np.float32), False, "skill requested", token)
        self._duration = float(spec.duration_s)
        self._command = np.asarray(spec.command, dtype=np.float32).copy()
        return ExecutionTick(self._command.copy(), False, "motion started", token)

    def tick(self, dt: float) -> ExecutionTick:
        if self.token is None:
            return ExecutionTick(np.zeros(3, dtype=np.float32), True, "idle", "")
        token = self.token
        spec = ACTION_SPECS[token]
        if spec.policy is not None:
            active_name = getattr(self.kick, "name", None)
            if active_name is not None:
                self._started_skill = True
                phase = getattr(self.kick, "phase", None)
                return ExecutionTick(np.zeros(3, dtype=np.float32), False, f"skill:{phase or 'running'}", token)
            if self._started_skill or not self._request_ok:
                self.token = None
                note = "completed" if self._request_ok else "rejected"
                return ExecutionTick(np.zeros(3, dtype=np.float32), True, note, token)
            return ExecutionTick(np.zeros(3, dtype=np.float32), False, "skill:pending", token)

        self._elapsed += max(0.0, float(dt))
        if self._elapsed >= self._duration - 1e-9:
            self.token = None
            return ExecutionTick(np.zeros(3, dtype=np.float32), True, "motion completed", token)
        return ExecutionTick(self._command.copy(), False, "motion running", token)

    def cancel(self, reset_action: bool = False) -> None:
        if reset_action and getattr(self.kick, "name", None) is not None:
            self.kick.finish("cancelled", "VLM loop cancelled the action.")
        self.token = None
        self._elapsed = 0.0
        self._started_skill = False
        self._request_ok = False

    def status(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "elapsed_s": round(self._elapsed, 3),
            "duration_s": self._duration,
            "command": [round(float(x), 3) for x in self._command],
            "skill": getattr(self.kick, "name", None),
            "phase": getattr(self.kick, "phase", None),
        }