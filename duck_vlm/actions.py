"""Discrete action vocabulary shared by prompts, interpreters, GUI and training.

The VLM never emits velocities or joint targets. It emits exactly one token
from this module; :mod:`duck_vlm.interpreter` grounds that token in the duck's
existing walking ONNX policy or LocalKick state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ActionSpec:
    token: str
    description: str
    kind: str = "motion"  # motion | skill | control | terminal
    duration_s: float = 0.0
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    policy: str | None = None
    request: str | None = None
    terminal: bool = False
    # Joint deltas applied on top of the nominal pose while a head action runs.
    head_delta: tuple[tuple[str, float], ...] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ACTION_SPECS: dict[str, ActionSpec] = {
    "FWD": ActionSpec("FWD", "Walk forward one short step", "motion", 0.55, (0.35, 0.0, 0.0)),
    "BACK": ActionSpec("BACK", "Walk backward one short step", "motion", 0.55, (-0.35, 0.0, 0.0)),
    "STRAFE_L": ActionSpec("STRAFE_L", "Step sideways to the duck's left", "motion", 0.40, (0.0, 0.22, 0.0)),
    "STRAFE_R": ActionSpec("STRAFE_R", "Step sideways to the duck's right", "motion", 0.40, (0.0, -0.22, 0.0)),
    "TURN_L": ActionSpec("TURN_L", "Turn left in place", "motion", 0.55, (0.0, 0.0, 1.5)),
    "TURN_R": ActionSpec("TURN_R", "Turn right in place", "motion", 0.55, (0.0, 0.0, -1.5)),
    "STOP": ActionSpec("STOP", "Stop and stand for one short interval", "control", 0.30),
    "STAND": ActionSpec("STAND", "Hold a stable standing pose", "control", 0.70),
    # Head actions: the duck keeps standing, but the head joints are driven to a
    # fixed offset so the duck-cam can actually look at the ground in front of it.
    # head_pitch increases = look down, neck_pitch increases = look up (verified FK).
    "LOOK_DOWN": ActionSpec("LOOK_DOWN", "Keep standing and pitch the head down to see the ground just in front of the duck", "head", 1.20, head_delta=(("neck_pitch", 0.0), ("head_pitch", 0.60), ("head_yaw", 0.0), ("head_roll", 0.0))),
    "HEAD_CENTER": ActionSpec("HEAD_CENTER", "Return the head to the neutral forward pose so the duck looks ahead again", "head", 0.90, head_delta=(("neck_pitch", 0.0), ("head_pitch", 0.0), ("head_yaw", 0.0), ("head_roll", 0.0))),
    # Covers the very close blind spot: +0.60 rad sees 0.15~0.45 m, +1.00 rad sees
    # 0.10~0.22 m, so the two stages together leave only <0.10 m unseen.
    "LOOK_DOWN_MORE": ActionSpec("LOOK_DOWN_MORE", "Keep standing and pitch the head further down to see a target almost under the chin", "head", 1.20, head_delta=(("neck_pitch", 0.0), ("head_pitch", 1.00), ("head_yaw", 0.0), ("head_roll", 0.0))),
    "KICK_L": ActionSpec("KICK_L", "Run the left-foot ball-kick policy", "skill", 0.0, policy="ball_kick_left", request="kickL"),
    "KICK_R": ActionSpec("KICK_R", "Run the right-foot ball-kick policy", "skill", 0.0, policy="ball_kick_right", request="kickR"),
    "ROLL": ActionSpec("ROLL", "Run the roulade forward-roll policy", "skill", 0.0, policy="roulade"),
    "DANCE": ActionSpec("DANCE", "Run the happy-hop dance policy", "skill", 0.0, policy="happy_hop"),
    "DONE": ActionSpec("DONE", "Declare the task complete and stop", "terminal", 0.0, terminal=True),
}

TOKENS: tuple[str, ...] = tuple(ACTION_SPECS)
MOTION_TOKENS: tuple[str, ...] = tuple(
    token for token, spec in ACTION_SPECS.items() if spec.kind == "motion"
)
SKILL_TOKENS: tuple[str, ...] = tuple(
    token for token, spec in ACTION_SPECS.items() if spec.kind == "skill"
)

_ALIASES: dict[str, str] = {
    "FORWARD": "FWD",
    "MOVE_FORWARD": "FWD",
    "MOVE_FWD": "FWD",
    "前进": "FWD",
    "向前": "FWD",
    "BACKWARD": "BACK",
    "MOVE_BACK": "BACK",
    "后退": "BACK",
    "LEFT": "STRAFE_L",
    "MOVE_LEFT": "STRAFE_L",
    "左移": "STRAFE_L",
    "RIGHT": "STRAFE_R",
    "MOVE_RIGHT": "STRAFE_R",
    "右移": "STRAFE_R",
    "TURN_LEFT": "TURN_L",
    "LEFT_TURN": "TURN_L",
    "左转": "TURN_L",
    "TURN_RIGHT": "TURN_R",
    "RIGHT_TURN": "TURN_R",
    "右转": "TURN_R",
    "HALT": "STOP",
    "WAIT": "STOP",
    "停止": "STOP",
    "站立": "STAND",
    "LOW_HEAD": "LOOK_DOWN",
    "LOOK_DOWN": "LOOK_DOWN",
    "低头": "LOOK_DOWN",
    "看地面": "LOOK_DOWN",
    "HEAD_UP": "HEAD_CENTER",
    "RAISE_HEAD": "HEAD_CENTER",
    "抬头": "HEAD_CENTER",
    "DEEP_LOOK_DOWN": "LOOK_DOWN_MORE",
    "LOOK_DOWN_DEEP": "LOOK_DOWN_MORE",
    "更深低头": "LOOK_DOWN_MORE",
    "头回正": "HEAD_CENTER",
    "KICK_LEFT": "KICK_L",
    "KICKLEFT": "KICK_L",
    "左脚踢": "KICK_L",
    "KICK_RIGHT": "KICK_R",
    "KICKRIGHT": "KICK_R",
    "右脚踢": "KICK_R",
    "ROULADE": "ROLL",
    "FORWARD_ROLL": "ROLL",
    "翻滚": "ROLL",
    "HAPPY_HOP": "DANCE",
    "DANCING": "DANCE",
    "跳舞": "DANCE",
    "FINISH": "DONE",
    "FINISHED": "DONE",
    "COMPLETE": "DONE",
    "完成": "DONE",
}


def available_tokens(policies: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return tokens executable with the supplied policy bank.

    Basic motion/control tokens are always available. Skill tokens are offered
    only when their ONNX policy exists, preventing the language interface from
    drifting away from the executable action set.
    """
    if policies is None:
        return TOKENS
    available = set(policies)
    out: list[str] = []
    for token in TOKENS:
        spec = ACTION_SPECS[token]
        if spec.policy is None or spec.policy in available:
            out.append(token)
    return tuple(out)


def token_menu(tokens: Sequence[str] | None = None) -> str:
    return ", ".join(tokens or TOKENS)


def token_semantics(tokens: Sequence[str] | None = None) -> str:
    return "\n".join(f"- {token}: {ACTION_SPECS[token].description}" for token in (tokens or TOKENS))


def _normalise(raw: str) -> str:
    value = (raw or "").strip()
    value = re.sub(r"^```(?:text)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
    value = re.sub(r"^[\[\]{}()<>\"']+|[\]{}()<>\"']+$", "", value).strip()
    return value.upper().replace("-", "_").replace(" ", "_")


def parse_action_token(
    raw: str,
    allowed: Sequence[str] | None = None,
    fallback: str | None = None,
) -> str:
    """Parse a model reply and recover one allowed token.

    Strict enough for training, tolerant enough for hosted VLMs that wrap the
    answer in a sentence, JSON, or a markdown fence.
    """
    allowed_set = set(allowed or TOKENS)
    text = (raw or "").strip()
    if not text:
        if fallback:
            return fallback
        raise ValueError("empty VLM action output")

    direct = _normalise(text)
    direct = _ALIASES.get(direct, direct)
    if direct in allowed_set:
        return direct

    # JSON keys commonly used by function-calling / hosted VLMs.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("token", "action", "answer", "decision"):
                direct = _normalise(str(obj.get(key, "")))
                direct = _ALIASES.get(direct, direct)
                if direct in allowed_set:
                    return direct
    except Exception:
        pass

    normalised = _normalise(text)
    alias_hits = [token for token in allowed_set if token in normalised]
    if len(alias_hits) == 1:
        return alias_hits[0]

    # Search original text with word boundaries and longest-token-first.
    for token in sorted(allowed_set, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", text.upper()):
            return token

    for alias, token in sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if token in allowed_set and alias in text.upper():
            return token

    # Last resort: one unique token after broad text containment.
    hits = [token for token in allowed_set if token.upper() in text.upper()]
    if len(hits) == 1:
        return hits[0]

    if fallback:
        return fallback
    raise ValueError(f"invalid action token {raw!r}; allowed={sorted(allowed_set)}")