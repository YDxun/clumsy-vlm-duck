"""VLM clients for zero-shot and fine-tuned duck action-token inference."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Sequence
import urllib.error
import urllib.request

from .actions import ACTION_SPECS, available_tokens, parse_action_token, token_menu, token_semantics
from .config import PROMPT_DIR, VLMConfig
from .types import VlmObservation


@dataclass
class VlmReply:
    token: str
    raw_text: str
    provider: str
    model: str
    latency_s: float
    parse_recovered: bool = False


def _load_prompt(version: str, mode: str) -> str:
    name = "finetuned.txt" if mode == "finetuned" else "zero_shot.txt"
    path = PROMPT_DIR / name
    text = path.read_text(encoding="utf-8")
    if version and version not in ("duck_v0", "v0"):
        override = PROMPT_DIR / version / name
        if override.exists():
            text = override.read_text(encoding="utf-8")
    return text


def render_prompt(observation: VlmObservation, allowed: Sequence[str], mode: str, version: str) -> str:
    template = _load_prompt(version, mode)
    if mode == "finetuned":
        # Preserve the exact trained contract: only task, target, and recent moves.
        values = {
            "{task}": observation.task,
            "{task_id}": observation.task_id or "(none)",
            "{target}": observation.target,
            "{recent_actions}": ", ".join(observation.recent_actions[:5]) or "(none)",
        }
    else:
        values = {
            "{task}": observation.task,
            "{task_id}": observation.task_id or "(none)",
            "{target}": observation.target,
            "{state}": observation.state.prompt_text(),
            "{subgoal}": observation.subgoal or "(none)",
            "{recent_actions}": ", ".join(observation.recent_actions[:5]) or "(none)",
            "{affordance}": observation.affordance_text or "(none)",
            "{proprio}": observation.proprio_text,
            "{recovery}": observation.recovery_hint,
            "{token_menu}": token_menu(allowed),
            "{token_semantics}": token_semantics(allowed),
        }
    out = template
    for key, value in values.items():
        out = out.replace(key, str(value))
    return out.strip() + "\n"


def _endpoint(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("VLM base_url is empty")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"VLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"VLM connection failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VLM returned non-JSON response: {raw[:500]}") from exc


def _openai_text(data: dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI-compatible response: {str(data)[:500]}") from exc
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        content = "\n".join(parts)
    return str(content or "").strip()


def _gemini_text(data: dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:500]}") from exc
    return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()


class VlmDecisionClient:
    """One-step image-conditioned action-token decision client."""

    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig.from_env()

    def configure(self, values: dict[str, Any]) -> None:
        self.config.update(values)

    def decide(
        self,
        observation: VlmObservation,
        allowed: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> VlmReply:
        mode = (mode or self.config.mode).lower()
        if mode == "finetuned":
            active = self.config.active()
            provider = "openai"
        else:
            active = self.config.active()
            provider = str(active["provider"]).lower()

        # A dry run is useful for smoke tests and browser demos without a key.
        if mode == "dry_run" or not active.get("api_key") and provider not in ("ollama", "local"):
            token = self._dry_run_token(observation, allowed)
            return VlmReply(token, token, "dry_run", active.get("model", "dry-run"), 0.0, False)

        prompt = render_prompt(observation, allowed or available_tokens(), mode, self.config.prompt_version)
        observation.prompt = prompt
        t0 = time.monotonic()
        if provider == "gemini":
            raw = self._gemini(prompt, observation.image_jpeg, active)
        elif provider in ("openai", "vllm", "openai_compatible", "dashscope"):
            raw = self._openai(prompt, observation.image_jpeg, active)
        else:
            raise ValueError(f"unsupported VLM provider: {provider}")
        latency = time.monotonic() - t0
        try:
            token = parse_action_token(raw, allowed=allowed, fallback=None)
            recovered = token not in raw.strip().upper().split()
        except ValueError:
            token = parse_action_token(raw, allowed=allowed, fallback="STOP")
            recovered = True
        return VlmReply(token, raw, provider, str(active.get("model", "")), latency, recovered)

    def _openai(self, prompt: str, image_jpeg: bytes, active: dict[str, Any]) -> str:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + b64,
                    "detail": self.config.image_detail,
                },
            },
        ]
        payload: dict[str, Any] = {
            "model": active["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": float(active.get("temperature", 0.0)),
            "max_tokens": int(active.get("max_tokens", 48)),
        }
        if self.config.reasoning_effort and self.config.mode != "finetuned":
            payload["reasoning_effort"] = self.config.reasoning_effort
        headers = {"Content-Type": "application/json", "User-Agent": "duck-vlm/0.1"}
        key = active.get("api_key") or ""
        if key:
            headers["Authorization"] = "Bearer " + str(key)
        data = _post_json(_endpoint(str(active["base_url"])), payload, headers, float(active.get("timeout_s", 35)))
        return _openai_text(data)

    def _gemini(self, prompt: str, image_jpeg: bytes, active: dict[str, Any]) -> str:
        base = str(active["base_url"]).rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        model = str(active["model"])
        url = f"{base}/models/{model}:generateContent?key={active.get('api_key', '')}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode("ascii")}},
                ],
            }],
            "generationConfig": {
                "temperature": float(active.get("temperature", 0.0)),
                "maxOutputTokens": int(active.get("max_tokens", 48)),
            },
        }
        data = _post_json(url, payload, {"Content-Type": "application/json", "User-Agent": "duck-vlm/0.1"}, float(active.get("timeout_s", 35)))
        return _gemini_text(data)

    @staticmethod
    def _dry_run_token(observation: VlmObservation, allowed: Sequence[str] | None = None) -> str:
        allowed_set = set(allowed or available_tokens())
        state = observation.state
        if "DONE" in allowed_set and observation.task.lower().startswith("stop"):
            return "DONE"
        if state.fallen and "STAND" in allowed_set:
            return "STAND"
        bearing = state.target_bearing_rad
        if state.target_range_m is not None and bearing is not None:
            if state.target_range_m < 0.30:
                token = "KICK_L" if bearing >= 0.0 else "KICK_R"
                if token in allowed_set:
                    return token
                return "STOP" if "STOP" in allowed_set else "STAND"
            if abs(bearing) > 0.35:
                token = "TURN_L" if bearing > 0 else "TURN_R"
                if token in allowed_set:
                    return token
            return "FWD" if "FWD" in allowed_set else "STOP"
        if state.target_visible:
            return "FWD" if "FWD" in allowed_set else "STOP"
        return "TURN_L" if "TURN_L" in allowed_set else "STOP"


def prompt_payload_for_training(observation: VlmObservation, allowed: Sequence[str], mode: str = "finetuned") -> str:
    """Return the exact language prompt, with <image> for LLaMA-Factory data."""
    return "<image>\n" + render_prompt(observation, allowed, mode, "duck_v0")
