"""Configuration for the duck VLM action-token harness."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

from .actions import TOKENS

PACKAGE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PACKAGE_DIR / "prompts"
DEFAULT_RUN_DIR = PACKAGE_DIR / "runs"


@dataclass
class PluginConfig:
    proprioception: bool = True
    action_memory: bool = True
    subgoal: bool = True
    affordance: bool = True
    recovery: bool = True
    fall_recovery: bool = True
    human_takeover: bool = True
    search_align: bool = True
    explore: bool = True
    kick_station: bool = True
    extra_views: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in (values or {}).items():
            if hasattr(self, key):
                setattr(self, key, bool(value))


@dataclass
class LoopConfig:
    max_steps: int = 40
    action_settle_s: float = 0.20
    fall_reset_delay_s: float = 1.2
    decision_timeout_s: float = 35.0
    stale_image_s: float = 0.45
    max_retries: int = 1
    repeat_warning_count: int = 3
    fallback_token: str = "STOP"
    auto_start_recording: bool = True
    run_dir: str = str(DEFAULT_RUN_DIR)
    plugins: PluginConfig = field(default_factory=PluginConfig)

    def update(self, values: dict[str, Any]) -> None:
        values = values or {}
        for key in (
            "max_steps",
            "action_settle_s",
            "fall_reset_delay_s",
            "decision_timeout_s",
            "stale_image_s",
            "max_retries",
            "repeat_warning_count",
            "fallback_token",
            "auto_start_recording",
            "run_dir",
        ):
            if key in values:
                setattr(self, key, values[key])
        if isinstance(values.get("plugins"), dict):
            self.plugins.update(values["plugins"])
        self.max_steps = max(1, int(self.max_steps))
        self.max_retries = max(0, int(self.max_retries))
        self.repeat_warning_count = max(2, int(self.repeat_warning_count))
        self.fallback_token = str(self.fallback_token).upper()
        if self.fallback_token not in TOKENS:
            self.fallback_token = "STOP"


@dataclass
class VLMConfig:
    provider: str = "openai"
    mode: str = "zero_shot"  # zero_shot | finetuned | dry_run
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 35.0
    temperature: float = 0.0
    max_tokens: int = 48
    reasoning_effort: str = "low"
    image_detail: str = "high"
    ft_base_url: str = ""
    ft_model: str = ""
    ft_api_key: str = ""
    prompt_version: str = "duck_v0"

    @classmethod
    def from_env(cls) -> "VLMConfig":
        llm_url = os.environ.get("DUCKAGENT_LLM_URL", "")
        llm_key = os.environ.get("DUCKAGENT_LLM_KEY") or os.environ.get("DASHSCOPE_KEY", "")
        vlm_model = os.environ.get("DUCKAGENT_VLM_MODEL") or "qwen3-vl-plus"
        return cls(
            provider=os.environ.get("DUCK_VLM_PROVIDER", "openai"),
            mode=os.environ.get("DUCK_VLM_MODE", "zero_shot"),
            base_url=os.environ.get("DUCK_VLM_BASE_URL", llm_url),
            model=os.environ.get("DUCK_VLM_MODEL", vlm_model),
            api_key=os.environ.get("DUCK_VLM_API_KEY", llm_key),
            timeout_s=float(os.environ.get("DUCK_VLM_TIMEOUT_S", os.environ.get("DUCKAGENT_TIMEOUT_S", "35"))),
            temperature=float(os.environ.get("DUCK_VLM_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("DUCK_VLM_MAX_TOKENS", "48")),
            reasoning_effort=os.environ.get("DUCK_VLM_REASONING_EFFORT", "low"),
            image_detail=os.environ.get("DUCK_VLM_IMAGE_DETAIL", "high"),
            ft_base_url=os.environ.get("DUCK_VLM_FT_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")),
            ft_model=os.environ.get("DUCK_VLM_FT_MODEL", "duck-vlm-lora"),
            ft_api_key=os.environ.get("DUCK_VLM_FT_API_KEY", "EMPTY"),
            prompt_version=os.environ.get("DUCK_VLM_PROMPT_VERSION", "duck_v0"),
        )

    def update(self, values: dict[str, Any]) -> None:
        for key, value in (values or {}).items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.provider = str(self.provider).lower().strip()
        self.mode = str(self.mode).lower().strip()
        if self.mode not in ("zero_shot", "finetuned", "dry_run"):
            raise ValueError(f"unsupported VLM mode: {self.mode}")

    def active(self) -> dict[str, Any]:
        if self.mode == "finetuned":
            return {
                "provider": "openai",
                "base_url": self.ft_base_url,
                "model": self.ft_model,
                "api_key": self.ft_api_key,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "timeout_s": self.timeout_s,
            }
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "reasoning_effort": self.reasoning_effort,
            "image_detail": self.image_detail,
        }

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = "<set>" if self.api_key else ""
        data["ft_api_key"] = "<set>" if self.ft_api_key else ""
        return data