# -*- coding: utf-8 -*-
"""Static scene map: doorway/opening extraction from the scene XML.

A floor plan is legitimate prior knowledge for a mobile robot (it is not the
target's hidden state), so the exploration plugin is allowed to use it when the
duck cannot see its goal.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

MIN_OPENING_M = 0.50
_COORD_EPS = 0.08


def _floats(text: str | None) -> list[float]:
    if not text:
        return []
    out = []
    for piece in text.replace(",", " ").split():
        try:
            out.append(float(piece))
        except ValueError:
            continue
    return out


def wall_segments(xml_path: str | Path) -> list[dict]:
    """Wall box geoms from the scene XML as (axis, constant, start, end)."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return []
    segs: list[dict] = []
    for geom in root.iter("geom"):
        name = (geom.get("name") or "").lower()
        if "wall" not in name or geom.get("type") != "box":
            continue
        pos = _floats(geom.get("pos"))
        size = _floats(geom.get("size"))
        if len(pos) < 2 or len(size) < 2:
            continue
        sx, sy = size[0], size[1]
        if sx <= sy:      # thin in x -> runs along y at constant x
            segs.append({"axis": "x", "at": pos[0], "lo": pos[1] - sy, "hi": pos[1] + sy, "name": name})
        else:             # thin in y -> runs along x at constant y
            segs.append({"axis": "y", "at": pos[1], "lo": pos[0] - sx, "hi": pos[0] + sx, "name": name})
    return segs


def doorways(xml_path: str | Path, min_width_m: float = MIN_OPENING_M) -> list[dict]:
    """Gaps between wall segments on the same line -> doorway centres."""
    segs = wall_segments(xml_path)
    groups: dict[tuple[str, int], list[dict]] = {}
    for s in segs:
        groups.setdefault((s["axis"], round(s["at"] / _COORD_EPS)), []).append(s)

    out: list[dict] = []
    for (axis, _key), items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda s: s["lo"])
        at = sum(s["at"] for s in items) / len(items)
        for left, right in zip(items, items[1:]):
            gap = right["lo"] - left["hi"]
            if gap < min_width_m:
                continue
            mid = (left["hi"] + right["lo"]) / 2.0
            if axis == "x":     # wall at x=at, gap runs along y, pass through along x
                out.append({"center": (at, mid), "through": "x", "width": gap})
            else:
                out.append({"center": (mid, at), "through": "y", "width": gap})
    out.sort(key=lambda d: (-d["width"], d["center"]))
    return out


def exploration_waypoints(xml_path: str | Path, from_xy: tuple[float, float] = (0.0, 0.0),
                          push_m: float = 1.0) -> list[tuple[float, float]]:
    """Doorway centres ordered by distance, each followed by a point beyond it."""
    doors = doorways(xml_path)
    doors.sort(key=lambda d: (d["center"][0] - from_xy[0]) ** 2 + (d["center"][1] - from_xy[1]) ** 2)
    out: list[tuple[float, float]] = []
    for d in doors:
        cx, cy = d["center"]
        out.append((cx, cy))
        # step through the opening toward the far side, away from the start point
        if d["through"] == "x":
            sign = 1.0 if cx >= from_xy[0] else -1.0
            out.append((cx + sign * push_m, cy))
        else:
            sign = 1.0 if cy >= from_xy[1] else -1.0
            out.append((cx, cy + sign * push_m))
    return out


def floor_bounds(xml_path: str | Path) -> tuple[float, float, float, float] | None:
    """(min_x, max_x, min_y, max_y) of the floor geom, if one is declared."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return None
    for geom in root.iter("geom"):
        if (geom.get("name") or "").lower() != "floor":
            continue
        size = _floats(geom.get("size"))
        pos = _floats(geom.get("pos")) or [0.0, 0.0]
        if len(size) < 2:
            continue
        cx = pos[0] if pos else 0.0
        cy = pos[1] if len(pos) > 1 else 0.0
        return (cx - size[0], cx + size[0], cy - size[1], cy + size[1])
    return None


def coverage_waypoints(xml_path: str | Path, step_m: float = 1.30, inset_m: float = 0.45,
                       from_xy: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Coarse lawn-mower grid over the floor so the duck keeps covering new ground."""
    bounds = floor_bounds(xml_path)
    if bounds is None:
        return []
    min_x, max_x, min_y, max_y = bounds
    min_x += inset_m; max_x -= inset_m
    min_y += inset_m; max_y -= inset_m
    if max_x <= min_x or max_y <= min_y:
        return []
    pts: list[tuple[float, float]] = []
    y = min_y
    row = 0
    while y <= max_y + 1e-9:
        xs = []
        x = min_x
        while x <= max_x + 1e-9:
            xs.append(x)
            x += step_m
        if row % 2:
            xs.reverse()
        for x in xs:
            pts.append((round(x, 3), round(y, 3)))
        y += step_m
        row += 1
    pts.sort(key=lambda pt: (pt[0] - from_xy[0]) ** 2 + (pt[1] - from_xy[1]) ** 2)
    return pts
