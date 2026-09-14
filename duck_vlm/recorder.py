"""GUMI-style episode recording and LLaMA-Factory export."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import Any

from .types import DecisionRecord, VlmObservation

try:
    from PIL import Image
except Exception:  # pragma: no cover - imaging is optional
    Image = None


def _slug(text: str, limit: int = 36) -> str:
    clean = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", (text or "task").strip())
    clean = clean.strip("_") or "task"
    return clean[:limit]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class EpisodeRecorder:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.session_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.events_path: Path | None = None
        self.transitions_path: Path | None = None
        self._closed = True

    @property
    def active(self) -> bool:
        return self.session_dir is not None and not self._closed

    def start(self, task: str, target: str, mode: str, config: dict[str, Any] | None = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.run_root / f"{stamp}_{_slug(task)}"
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_dir / "events.jsonl"
        self.transitions_path = self.session_dir / "transitions.jsonl"
        self._closed = False
        self.record_event({"type": "session_start", "task": task, "target": target,
                           "mode": mode, "config": _jsonable(config or {})})
        return self.session_dir

    def _require(self) -> tuple[Path, Path]:
        if not self.active or self.session_dir is None or self.events_path is None:
            raise RuntimeError("recorder is not active")
        return self.session_dir, self.events_path

    def save_image(self, image_jpeg: bytes, step_index: int) -> str:
        # Recording is optional: episodes started with record=False must still run.
        if not self.active or self.session_dir is None or self.frames_dir is None:
            return ""
        path = self.frames_dir / f"{step_index:05d}.jpg"
        path.write_bytes(image_jpeg)
        return str(path.relative_to(self.session_dir))

    def save_extra_views(self, head_jpeg: bytes, extra_views: Any,
                         step_index: int) -> dict[str, str]:
        """Persist secondary views plus a side-by-side duck-cam/scene composite.

        Returns a mapping of view name -> path relative to the session dir.
        """
        if not self.active or self.session_dir is None or not extra_views:
            return {}
        session_dir = self.session_dir
        main_dir = session_dir / "frames_main"
        dual_dir = session_dir / "frames_dual"
        try:
            main_dir.mkdir(exist_ok=True)
            dual_dir.mkdir(exist_ok=True)
        except Exception:
            return {}
        out: dict[str, str] = {}
        tiles: list[bytes] = []
        for name, data in extra_views:
            if not data:
                continue
            path = main_dir / f"{step_index:05d}.jpg"
            try:
                path.write_bytes(bytes(data))
                out[str(name)] = str(path.relative_to(session_dir))
                tiles.append(bytes(data))
            except Exception:
                continue
        if Image is not None and head_jpeg and tiles:
            try:
                head = Image.open(io.BytesIO(bytes(head_jpeg))).convert("RGB")
                canvas = Image.new("RGB", (head.width * (1 + len(tiles)), head.height), (0, 0, 0))
                canvas.paste(head, (0, 0))
                for i, data in enumerate(tiles, start=1):
                    tile = Image.open(io.BytesIO(data)).convert("RGB")
                    canvas.paste(tile.resize((head.width, head.height)), (i * head.width, 0))
                dual = dual_dir / f"{step_index:05d}.jpg"
                canvas.save(dual, format="JPEG", quality=85)
                out["dual"] = str(dual.relative_to(session_dir))
            except Exception:
                pass
        return out

    def record_event(self, event: dict[str, Any]) -> None:
        if not self.active or self.events_path is None:
            return
        event = dict(event)
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")

    def record_decision(self, record: DecisionRecord, observation: VlmObservation) -> None:
        if not self.active or self.transitions_path is None:
            return
        row = {
            "step": int(record.step_index),
            "image": record.image_path,
            "task": observation.task,
            "target": observation.target,
            "action": record.token,
            "source": record.source,
            "raw_response": record.raw_response,
            "provider": record.provider,
            "model": record.model,
            "latency_s": record.latency_s,
            "parse_recovered": record.parse_recovered,
            "note": record.note,
            "prompt": observation.prompt,
            "prompt_version": observation.prompt_version,
            "state": observation.state.to_dict(),
            "recent_actions": list(observation.recent_actions),
            "subgoal": observation.subgoal,
            "affordance": observation.affordance_text,
            "proprio": observation.proprio_text,
            "recovery": observation.recovery_hint,
            "plugins": dict(observation.plugin_state),
        }
        views = self.save_extra_views(observation.image_jpeg, observation.extra_views,
                                      record.step_index)
        if views:
            row["extra_view_files"] = views
            row["image_dual"] = views.get("dual", "")
        with self.transitions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def record_result(self, step_index: int, result: Any) -> None:
        self.record_event({"type": "action_result", "step": step_index, "result": _jsonable(result)})

    def finish(self, summary: dict[str, Any] | None = None) -> Path | None:
        if self.session_dir is None:
            return None
        if not self._closed:
            self.record_event({"type": "session_end", "summary": _jsonable(summary or {})})
            (self.session_dir / "session.json").write_text(
                json.dumps(_jsonable(summary or {}), ensure_ascii=False, indent=2), encoding="utf-8")
        self._closed = True
        return self.session_dir


def export_alpaca(session_dir: str | Path, output: str | Path | None = None) -> Path:
    session = Path(session_dir)
    transitions = session / "transitions.jsonl"
    if not transitions.exists():
        raise FileNotFoundError(transitions)
    out = Path(output) if output else session / "llamafactory.json"
    rows: list[dict[str, Any]] = []
    with transitions.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image = session / str(row.get("image", ""))
            if not image.exists():
                continue
            prompt = str(row.get("prompt") or "").strip()
            if not prompt.startswith("<image>"):
                prompt = "<image>\n" + prompt
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(row.get("action") or "STOP")},
                ],
                "images": [str(image.resolve())],
                "metadata": {"step": row.get("step"), "task": row.get("task"),
                             "source": row.get("source"), "subgoal": row.get("subgoal")},
            })
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
