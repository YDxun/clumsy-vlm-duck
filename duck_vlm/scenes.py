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


# Users rarely phrase an object the way the scene file does: the pack says
# "红色方块" and a person says "红方块". Rather than enumerate aliases per object,
# accept a match when the colour word and the noun both appear in the sentence.
_COLORS = {
    "red": ("red", "红"), "blue": ("blue", "蓝"), "green": ("green", "绿"),
    "orange": ("orange", "橙"), "yellow": ("yellow", "黄"),
}
_NOUNS = {
    "cube": ("cube", "block", "box", "方块", "立方体"),
    "ball": ("ball", "sphere", "球"),
    "cup": ("cup", "cylinder", "杯", "圆柱", "筒"),
    "zone": ("zone", "area", "区域"),
    "beacon": ("beacon", "marker", "信标", "标记"),
}


def label_in_text(label: str, text: str) -> bool:
    """True when the text names this entity, exactly or loosely."""
    token = _norm(label)
    if token and token in _norm(text):
        return True
    return _loose_entity_match(label, text)


def _loose_entity_match(label: str, text: str) -> bool:
    """True when the label's colour and noun occur *next to each other*.

    Adjacency matters: "把红方块推到蓝色区域" mentions a red cube and a blue zone,
    so a bag-of-words test would also match "蓝色方块" and pick the wrong object.
    Comparing on a space-stripped copy catches "red cube"/"红方块" while keeping
    the colour bound to its own noun.
    """
    lab = _norm(label)
    compact = re.sub(r"\s+", "", _norm(text))
    if not lab or not compact:
        return False
    have_colors = [c for c, ws in _COLORS.items() if any(w in lab for w in ws)]
    have_nouns = [n for n, ws in _NOUNS.items() if any(w in lab for w in ws)]
    if not have_colors or not have_nouns:
        return False
    for c in have_colors:
        for n in have_nouns:
            for cw in _COLORS[c]:
                for nw in _NOUNS[n]:
                    if (cw + nw) in compact or (nw + cw) in compact:
                        return True
    return False


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
        for person in self.metadata.get("people") or []:
            body = person.get("body") or person.get("id")
            for name in [person.get("id"), body, *(person.get("semantic_labels") or [])]:
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

    def entity_names(self) -> list[tuple[str, str, str]]:
        """(label, body, kind) for every addressable entity, longest label first.

        The kind lets a caller separate the thing to act on from the place to put it.
        """
        out: list[tuple[str, str, str]] = []
        for kind, key in (("object", "objects"), ("person", "people"), ("zone", "zones")):
            for item in self.metadata.get(key) or []:
                body = item.get("body") or item.get("id")
                if not body:
                    continue
                for name in (item.get("id"), body, *(item.get("semantic_labels") or [])):
                    if name:
                        out.append((str(name), str(body), kind))
        beacon = self.metadata.get("beacon") or {}
        if beacon.get("body"):
            for name in (beacon.get("body"), *(beacon.get("semantic_labels") or [])):
                if name:
                    out.append((str(name), str(beacon["body"]), "beacon"))
        out.sort(key=lambda nbk: len(_norm(nbk[0])), reverse=True)
        return out

    def scan_text(self, text: str) -> list[tuple[str, str, str]]:
        """Entities named in the text, most specific label first."""
        q = _norm(text)
        if not q:
            return []
        hits: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for label, body, kind in self.entity_names():
            if body in seen:
                continue
            if label_in_text(label, text):
                seen.add(body)
                hits.append((label, body, kind))
        return hits

    def match_task_semantic(self, text: str, target: str | None = None) -> dict[str, Any] | None:
        """Fallback for free-form prompts: reuse the curated task that grades the same
        object (and zone, when the sentence names one), so the episode is still scored
        by the scene evaluator instead of running blind."""
        from .intent import resolve_intent
        intent = resolve_intent("", text)
        # Only approach/manipulate can be graded by the curated tasks. A gait, pose,
        # orbit or recovery sentence has no matching success condition, and grading it
        # with, say, walk_to_ball would report a meaningless pass.
        if intent not in ("approach", "manipulate"):
            return None
        zone = None
        for _label, body, kind in self.scan_text(text):
            if kind == "zone":
                zone = body
                break
        best: dict[str, Any] | None = None
        best_score = 0
        for task in self.list_tasks():
            refs = self._task_refs(task)
            score = 0
            if target and target in refs["targets"]:
                score += 2
            if zone and zone in refs["zones"]:
                score += 1
            if not score:
                continue
            if intent == "manipulate":
                if not refs["zones"]:
                    continue          # a kick/push that grades nothing moving is the wrong task
                score += 1
            if score > best_score:
                best, best_score = task, score
        return best if best_score >= 2 else None

    def _task_refs(self, task: dict[str, Any]) -> dict[str, set[str]]:
        targets: set[str] = set()
        zones: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "body_in_zone":
                    if node.get("body"):
                        targets.add(str(node["body"]))
                    if node.get("zone"):
                        zones.add(str(node["zone"]))
                elif node.get("type") and node.get("target"):
                    targets.add(str(node["target"]))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(task.get("success"))
        return {"targets": targets, "zones": zones}

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