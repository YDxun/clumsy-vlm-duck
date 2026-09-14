"""Non-blocking perceive -> VLM -> one discrete action -> repeat controller."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import time
from typing import Any, Callable

import numpy as np

from .actions import ACTION_SPECS, available_tokens
from .config import LoopConfig, VLMConfig
from .interpreter import DuckActionInterpreter, ExecutionTick
from .plugins import ActionMemoryPlugin, HumanTakeoverPlugin, PluginSuite, RecoveryPlugin
from .recorder import EpisodeRecorder
from .task_manager import TaskManager
from .types import ActionResult, DecisionRecord, DuckState, VlmObservation
from .vlm import VlmDecisionClient, VlmReply, render_prompt

StateProvider = Callable[[str, float], DuckState]
ImageProvider = Callable[[], bytes]


class DuckVlmLoop:
    """One action token per model step, with VLM inference off the 50 Hz thread."""

    def __init__(self, *, vlm: VlmDecisionClient | None = None, state_provider: StateProvider,
                 image_provider: ImageProvider, kick: Any, policies: dict[str, Any] | None = None,
                 reset_callback: Callable[[], None] | None = None,
                 loop_config: LoopConfig | None = None, vlm_config: VLMConfig | None = None,
                 task_manager: TaskManager | None = None):
        self.vlm = vlm or VlmDecisionClient()
        self.state_provider = state_provider
        self.image_provider = image_provider
        self.kick = kick
        self.policies = policies or {}
        self.reset_callback = reset_callback or (lambda: None)
        self.loop_config = loop_config or LoopConfig()
        self.vlm_config = vlm_config or getattr(self.vlm, "config", VLMConfig.from_env())
        self.task_manager = task_manager
        self.plugins = PluginSuite.build(self.loop_config.plugins)
        self.interpreter = DuckActionInterpreter(kick, self.policies)
        self.recorder = EpisodeRecorder(self.loop_config.run_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duck-vlm")
        self.running = False
        self.paused = False
        self.single_step = False
        self.task = ""
        self.active_task_id = ""
        self.target = "ball"
        self.step_index = 0
        self.last_token = ""
        self.last_result = ""
        self.last_error = ""
        self.last_latency_s: float | None = None
        self.last_reply: VlmReply | None = None
        self.started_at = 0.0
        self._fall_since: float | None = None
        self._pending: Future | None = None
        self._pending_observation: VlmObservation | None = None
        self._pending_context: dict[str, Any] = {}
        self._before_state: DuckState | None = None
        self._result_history: list[ActionResult] = []

    def set_config(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        values = values or {}
        loop_values = {k: v for k, v in values.items() if k not in ("vlm", "plugins")}
        if "plugins" in values:
            loop_values["plugins"] = values["plugins"]
        self.loop_config.update(loop_values)
        if "plugins" in values:
            self.plugins.configure(self.loop_config.plugins)
        if "vlm" in values:
            self.vlm.configure(values["vlm"])
            self.vlm_config = self.vlm.config
        return self.status()

    def start(self, task: str, target: str = "ball", *, auto_run: bool = True,
              record: bool | None = None) -> dict[str, Any]:
        self.task = (task or "explore the scene").strip()
        self.target = (target or "ball").strip()
        self.running = True
        self.paused = not auto_run
        self.single_step = not auto_run
        self.step_index = 0
        self.last_token = ""
        self.last_result = "started"
        self.last_error = ""
        self.last_reply = None
        self.started_at = time.monotonic()
        self._fall_since = None
        self._before_state = None
        self._result_history.clear()
        action_memory = self.plugins.get("action_memory")
        if isinstance(action_memory, ActionMemoryPlugin):
            action_memory.recent.clear()
        self.interpreter.cancel(reset_action=True)
        self._pending = None
        self._pending_observation = None
        self.active_task_id = ""
        if self.task_manager is not None:
            task_def = self.task_manager.start(self.task)
            if task_def:
                self.active_task_id = str(task_def.get("id") or "")
        should_record = self.loop_config.auto_start_recording if record is None else bool(record)
        if should_record and not self.recorder.active:
            self.recorder.start(self.task, self.target, self.vlm_config.mode,
                                {"loop": asdict(self.loop_config), "vlm": self.vlm_config.public()})
        self.recorder.record_event({"type": "control", "action": "start", "task": self.task, "target": self.target})
        return self.status()

    def pause(self) -> dict[str, Any]:
        self.paused = True
        self.recorder.record_event({"type": "control", "action": "pause"})
        return self.status()

    def resume(self) -> dict[str, Any]:
        self.paused = False
        self.single_step = False
        self.recorder.record_event({"type": "control", "action": "resume"})
        return self.status()

    def step_once(self) -> dict[str, Any]:
        if not self.running:
            self.start(self.task or "explore the scene", self.target, auto_run=False)
        self.paused = False
        self.single_step = True
        self.recorder.record_event({"type": "control", "action": "step_once"})
        return self.status()

    def stop(self, reason: str = "user_stop") -> dict[str, Any]:
        self.interpreter.cancel(reset_action=True)
        if self._pending is not None:
            self._pending.cancel()
        self._pending = None
        self._pending_observation = None
        self.running = False
        if self.task_manager is not None:
            self.task_manager.stop()
        self.last_result = reason
        self.recorder.record_event({"type": "control", "action": "stop", "reason": reason})
        self.recorder.finish(self.status())
        return self.status()

    def enqueue_human(self, token: str) -> bool:
        plugin = self.plugins.get("human_takeover")
        if not isinstance(plugin, HumanTakeoverPlugin) or not plugin.enabled:
            return False
        token = (token or "").upper().strip()
        if token not in ACTION_SPECS:
            return False
        plugin.push(token)
        if not self.running:
            self.start(self.task or "human demonstration", self.target, auto_run=True)
        self.paused = False
        return True

    def tick(self, sim_time: float, dt: float = 0.02) -> np.ndarray:
        zero = np.zeros(3, dtype=np.float32)
        if not self.running:
            return zero
        state = self.state_provider(self.target, sim_time)
        if state.fallen:
            self.interpreter.cancel(reset_action=True)
            if self._fall_since is None:
                self._fall_since = sim_time
                self.last_result = "fallen"
                self.recorder.record_event({"type": "fall", "sim_time": sim_time, "state": state.to_dict()})
            if (self.loop_config.plugins.fall_recovery
                    and sim_time - self._fall_since >= self.loop_config.fall_reset_delay_s):
                self.reset_callback()
                self._fall_since = None
                self.last_result = "fall_recovered"
                self.recorder.record_event({"type": "fall_recovery", "sim_time": sim_time})
            return zero
        self._fall_since = None
        if self.task_manager is not None:
            task_state = self.task_manager.step(dt)
            if task_state.get("success") or task_state.get("failure"):
                self.interpreter.cancel(reset_action=True)
                self.running = False
                self.last_result = "task_success" if task_state.get("success") else "task_failed"
                self.recorder.record_event({"type": "task_end", "state": task_state, "sim_time": sim_time})
                self.recorder.finish(self.status())
                return zero
        if self.interpreter.busy:
            tick = self.interpreter.tick(dt)
            if tick.done:
                self._finish_action(tick, sim_time)
            return tick.command
        if self._pending is not None:
            if not self._pending.done():
                return zero
            future = self._pending
            observation = self._pending_observation
            context = self._pending_context
            self._pending = None
            self._pending_observation = None
            self._pending_context = {}
            if observation is None:
                return zero
            try:
                reply = future.result()
                recovery_plugin = self.plugins.get('recovery')
                if isinstance(recovery_plugin, RecoveryPlugin):
                    recovered = recovery_plugin.recover_token(reply.token, observation)
                    if recovered != reply.token:
                        reply.raw_text = reply.raw_text + f'\n[runtime-recovery:{reply.token}->{recovered}]'
                        reply.token = recovered
                        reply.parse_recovered = True
                self.last_reply = reply
                self.last_latency_s = reply.latency_s
                self.last_error = ""
                record = DecisionRecord(
                    step_index=self.step_index, token=reply.token,
                    source=("dry_run" if reply.provider == "dry_run" else "vlm"),
                    raw_response=reply.raw_text, provider=reply.provider, model=reply.model,
                    latency_s=reply.latency_s, parse_recovered=reply.parse_recovered,
                    note="vlm_decision",
                    image_path=self.recorder.save_image(observation.image_jpeg, self.step_index),
                    observation=observation.to_dict(), prompt=observation.prompt)
                self.recorder.record_decision(record, observation)
                return self._begin_action(reply.token, observation.state, sim_time)
            except Exception as exc:
                self.last_error = str(exc)
                self.last_result = "vlm_error"
                self.recorder.record_event({"type": "vlm_error", "error": str(exc), "step": self.step_index})
                return self._begin_action(self.loop_config.fallback_token, observation.state, sim_time,
                                          source="fallback", raw=str(exc))
        if self.paused and not self.single_step:
            return zero
        human = self.plugins.get("human_takeover")
        if isinstance(human, HumanTakeoverPlugin) and human.enabled:
            token = human.pop()
            if token:
                observation = self._make_observation(state)
                record = DecisionRecord(
                    step_index=self.step_index, token=token, source="human", raw_response="",
                    provider="human", model="keyboard", latency_s=0.0, note="human_override",
                    image_path=self.recorder.save_image(observation.image_jpeg, self.step_index),
                    observation=observation.to_dict(), prompt=observation.prompt)
                self.recorder.record_decision(record, observation)
                return self._begin_action(token, state, sim_time, source="human")
        if self._pending is None:
            self._begin_decision(state)
        return zero

    def _make_observation(self, state: DuckState) -> VlmObservation:
        image = self.image_provider() or b""
        observation = VlmObservation(task=self.task, task_id=self.active_task_id, target=self.target, image_jpeg=image,
                                     state=state, step_index=self.step_index, recent_actions=[])
        context = {"action_token": self.last_token, "last_result": self.last_result,
                   "allowed_tokens": available_tokens(self.policies)}
        observation = self.plugins.before_decision(observation, context)
        observation.plugin_state = self.plugins.status()
        observation.prompt = render_prompt(observation, available_tokens(self.policies), self.vlm_config.mode, self.vlm_config.prompt_version)
        return observation

    def _begin_decision(self, state: DuckState) -> None:
        observation = self._make_observation(state)
        allowed = tuple(available_tokens(self.policies))
        self._pending_observation = observation
        self._pending_context = {"action_token": self.last_token, "last_result": self.last_result,
                                 "allowed_tokens": allowed, "plugins": self.plugins.status()}
        self.last_result = "thinking"
        self._pending = self.executor.submit(self.vlm.decide, observation, allowed, self.vlm_config.mode)
        self.recorder.record_event({"type": "decision_requested", "step": self.step_index,
                                    "mode": self.vlm_config.mode, "allowed_tokens": list(allowed)})

    def _begin_action(self, token: str, before: DuckState, sim_time: float, *,
                      source: str = "vlm", raw: str = "") -> np.ndarray:
        self.last_token = token
        self._before_state = before
        tick = self.interpreter.start(token, fallen=before.fallen)
        self.last_result = tick.note
        self.recorder.record_event({"type": "action_started", "step": self.step_index,
                                    "token": token, "source": source, "raw": raw})
        if tick.done:
            self._finish_action(tick, sim_time)
            return np.zeros(3, dtype=np.float32)
        return np.asarray(tick.command, dtype=np.float32)

    def _finish_action(self, tick: ExecutionTick, sim_time: float) -> None:
        after = self.state_provider(self.target, sim_time)
        before = self._before_state or after
        memory = self.plugins.get("action_memory")
        if isinstance(memory, ActionMemoryPlugin):
            memory.observe(tick.token or self.last_token)
        result = ActionResult(token=tick.token or self.last_token,
                              ok=tick.note not in ("rejected", "duck is down"),
                              note=tick.note, sim_time=sim_time, state=after)
        self._result_history.append(result)
        self.plugins.after_step(result.token, before, after, result.ok, {"source": "vlm"})
        self.recorder.record_result(self.step_index, result)
        self.last_result = result.note
        self.last_token = result.token
        self.step_index += 1
        if result.token == "DONE":
            self.running = False
            self.last_result = "done"
            self.recorder.record_event({"type": "episode_done", "step": self.step_index})
            self.recorder.finish(self.status())
            return
        if self.step_index >= self.loop_config.max_steps:
            self.running = False
            self.last_result = "max_steps"
            self.recorder.record_event({"type": "episode_done", "step": self.step_index, "reason": "max_steps"})
            self.recorder.finish(self.status())
            return
        if self.single_step:
            self.paused = True
            self.single_step = False

    def status(self) -> dict[str, Any]:
        if not self.running:
            phase = "DONE" if self.last_result in ("done", "max_steps") else "IDLE"
        elif self.interpreter.busy:
            phase = "EXECUTING"
        elif self.paused:
            phase = "PAUSED"
        elif self._pending is not None:
            phase = "THINKING"
        else:
            phase = "READY"
        return {
            "phase": phase, "running": self.running, "paused": self.paused,
            "task": self.task, "task_id": self.active_task_id, "target": self.target, "mode": self.vlm_config.mode,
            "allowed_tokens": list(available_tokens(self.policies)),
            "provider": self.vlm_config.provider,
            "model": self.vlm_config.ft_model if self.vlm_config.mode == "finetuned" else self.vlm_config.model,
            "step": self.step_index, "max_steps": self.loop_config.max_steps,
            "last_action": self.last_token, "last_result": self.last_result,
            "last_error": self.last_error,
            "latency_ms": None if self.last_latency_s is None else round(self.last_latency_s * 1000),
            "pending": self._pending is not None, "interpreter": self.interpreter.status(),
            "plugins": self.plugins.status(), "recording": self.recorder.active,
            "task_eval": self.task_manager.status() if self.task_manager is not None else None,
            "session_dir": str(self.recorder.session_dir) if self.recorder.session_dir else "",
            "recent_results": [asdict(item) for item in self._result_history[-8:]],
        }
