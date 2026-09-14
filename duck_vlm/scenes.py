"""Scene-pack discovery and metadata-aware entity resolution."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _default_root() -> Path | None:
    env = os.environ.get("DUCK_SCENES_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("/root/microduck_sim/duck_scenes/scenes"),
        PACKAGE_ROOT / "duck_scenes" / "scenes",
        PACKAGE_ROOT / ".tmp_scenes_validated_20260914" / "scenes",
    ])
    for path in candidates:
        if path.exists():
            return path
    return None


def _norm(text: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text or "").strip().lower())


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    root: Path
    xml: Path
    metadata_path: Path
    tasks_path: Path
    metadata: dict[str, Any]
    tasks: dict[str, Any]

    def list_tasks(self) -> list[dict[str, Any]]:
        return list(self.tasks.get("tasks") or [])

    def task(self, task_id: str) -> dict[str, Any] | None:
        for item in self.list_tasks():
            if item.get("id") == task_id:
                return item
        return None

    def match_task(self, text: str) -> dict[str, Any] | None:
        query = _norm(text)
        if not query:
            return None
        exact = self.task(str(text).strip())
        if exact:
            return exact
        for item in self.list_tasks():
            if query == _norm(item.get("id")):
                return item
            for key in ("instruction_zh", "instruction_en"):
                value = _norm(item.get(key))
                if value and (value in query or query in value):
                    return item
        return None

    def resolve_body(self, query: str) -> str | None:
        q = _norm(query)
        if not q:
            return None
        robot = self.metadata.get("robot") or {}
        if q in (_norm(robot.get("body_name")), _norm(robot.get("root_body"))):
            return robot.get("body_name") or robot.get("root_body")
        beacon = self.metadata.get("beacon") or {}
        if q in ("beacon", "target beacon", "信标") or q == _norm(beacon.get("body")):
            return beacon.get("body")
        entries: list[tuple[str, str]] = []
        for obj in self.metadata.get("objects") or []:
            body = obj.get("body") or obj.get("id")
            for name in [obj.get("id"), body, *(obj.get("semantic_labels") or [])]:
                entries.append((_norm(name), body))
        for zone in self.metadata.get("zones") or []:
            body = zone.get("body") or zone.get("id")
            for name in [zone.get("id"), body, *(zone.get("semantic_labels") or [])]:
                entries.append((_norm(name), body))
        for name, body in entries:
            if q == name:
                return body
        for name, body in sorted(entries, key=lambda item: len(item[0]), reverse=True):
            if name and name in q:
                return body
        return None

    def resolve_zone(self, query: str) -> str | None:
        q = _norm(query)
        for zone in self.metadata.get("zones") or []:
            names = [zone.get("id"), zone.get("body"), *(zone.get("semantic_labels") or [])]
            if q in {_norm(name) for name in names}:
                return zone.get("body") or zone.get("id")
        return None

    def zone_center(self, query: str) -> list[float] | None:
        q = _norm(query)
        for zone in self.metadata.get("zones") or []:
            names = [zone.get("id"), zone.get("body"), *(zone.get("semantic_labels") or [])]
            if q in {_norm(name) for name in names}:
                center = zone.get("center")
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    return [float(center[0]), float(center[1]), float(center[2]) if len(center) > 2 else 0.0]
        return None


class SceneCatalog:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else _default_root()

    def _require(self) -> Path:
        if self.root is None or not self.root.exists():
            raise FileNotFoundError("Duck scene root not found; set DUCK_SCENES_ROOT")
        return self.root

    def list_ids(self) -> list[str]:
        root = self._require()
        return sorted(path.parent.name for path in root.glob("*/scene.xml"))

    def load(self, scene_id: str) -> SceneSpec:
        root = self._require()
        scene_dir = root / scene_id
        xml = scene_dir / "scene.xml"
        metadata_path = scene_dir / "metadata.yaml"
        tasks_path = scene_dir / "tasks.yaml"
        for path in (xml, metadata_path, tasks_path):
            if not path.exists():
                raise FileNotFoundError(path)
        if yaml is None:
            raise RuntimeError(f"PyYAML is required for scene metadata: {_YAML_ERROR}")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}
        return SceneSpec(scene_id, scene_dir, xml, metadata_path, tasks_path, metadata, tasks)


_DEFAULT_CATALOG: SceneCatalog | None = None


def catalog(root: str | Path | None = None) -> SceneCatalog:
    global _DEFAULT_CATALOG
    if root is not None:
        return SceneCatalog(root)
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = SceneCatalog()
    return _DEFAULT_CATALOG


def resolve_scene_xml(scene_id: str | None) -> Path | None:
    if not scene_id:
        return None
    return catalog().load(scene_id).xml