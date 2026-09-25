"""Scene calibration support: reference frame, editable shapes <-> scene.json, validation (RUNBOOK P1.1).

Contract:
- `make_reference(video, n)` returns the per-pixel median of `n` frames spread over the whole video at
  native resolution: moving traffic disappears and the empty road remains (configs/reference.jpg).
- `LAYERS` defines every drawable layer: its shape kind and the attributes asked after drawing. The
  browser tool (scripts/calibrate_scene.py) edits a flat list of shapes
  `{"layer": str, "pts": [[x, y], ...], "attrs": {str: str}}`.
- `shapes_to_scene(shapes, ...)` builds the SPEC §4 scene dict; `scene_to_shapes(scene)` is its inverse,
  so an existing scene.json can be edited again. Coordinates are native-resolution pixels.
- `validate_scene(scene)` lists problems (missing ids, dangling references, degenerate homography);
  an empty list means the scene is consistent.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roadwatch.video import probe, read_window

# kind: polygon (>= 3 points, Enter closes), polyline (>= 2, Enter ends), line (2), rect (2 corners),
# arrow (2: tail, head), point (1). "multi": whether a layer holds several shapes; "fields": attributes.
LAYERS: dict[str, dict[str, Any]] = {
    "carriageway": {"kind": "polygon", "multi": False, "fields": []},
    "sidewalks": {"kind": "polygon", "multi": True, "fields": []},
    "islands": {"kind": "polygon", "multi": True, "fields": []},  # refuges and the median: pavement
    "parking_zones": {"kind": "polygon", "multi": True, "fields": []},
    "bus_stops": {"kind": "polygon", "multi": True, "fields": []},  # buses dwelling here are not stopped
    "intersection": {"kind": "polygon", "multi": False, "fields": []},
    "lanes": {
        "kind": "polygon",
        "multi": True,
        "fields": ["id", "direction_group", "approach", "signal", "allowed_exits"],
    },
    "lane_directions": {"kind": "arrow", "multi": True, "fields": ["lane"]},
    "exits": {"kind": "polygon", "multi": True, "fields": ["id"]},
    "stop_lines": {"kind": "line", "multi": True, "fields": ["id", "lanes", "signal"]},
    "crosswalks": {"kind": "polygon", "multi": True, "fields": ["id"]},
    "solid_lines": {"kind": "polyline", "multi": True, "fields": ["id"]},
    "signals": {"kind": "rect", "multi": True, "fields": ["id", "lamp"]},
    "no_u_turn_zones": {"kind": "polygon", "multi": True, "fields": []},
    "homography": {"kind": "point", "multi": True, "fields": ["world_x", "world_y"]},
}
LAMPS = ("red", "yellow", "green")
# layers stored as a plain list of polygons in scene.json (SPEC §4, §12.42)
POLYGON_LIST_LAYERS = ("sidewalks", "islands", "parking_zones", "bus_stops", "no_u_turn_zones")
_COLLINEAR_REL_AREA = 1e-3  # triangle area / squared extent below this counts as three points on a line
_ARROW_PX = 200.0  # length of the arrow drawn for an existing lane direction when re-editing


def make_reference(video_path: str | Path, n: int = 49) -> np.ndarray:
    """Per-pixel median of `n` frames evenly spread over the video (BGR uint8, native size)."""
    meta = probe(video_path)
    if meta.n_frames < 1:
        raise RuntimeError(f"{video_path}: no frames")
    frames = []
    for idx in np.unique(np.linspace(0, meta.n_frames - 1, n).round().astype(int)):
        window = read_window(video_path, idx / meta.fps, (idx + 1) / meta.fps)
        frame = next((img for _, _, img in window), None)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"{video_path}: could not decode any frame")
    stack = np.stack(frames)
    out = np.empty(stack.shape[1:], dtype=np.uint8)
    for r in range(0, stack.shape[1], 64):  # row strips keep the float median small in memory
        out[r : r + 64] = np.median(stack[:, r : r + 64], axis=0).round().astype(np.uint8)
    return out


def _pts(points: Any) -> list[list[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in points]


def _split(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _rect_xywh(pts: list[list[float]]) -> list[float]:
    (x0, y0), (x1, y1) = pts[:2]
    return [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]


def shapes_to_scene(
    shapes: list[dict[str, Any]],
    image_size: tuple[int, int],
    reference_frame: str = "configs/reference.jpg",
    u_turn_prohibited_everywhere: bool = False,
) -> dict[str, Any]:
    """Build the SPEC §4 scene dict from the tool's shapes (layers without shapes are left out)."""
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shape in shapes:
        if shape["layer"] not in LAYERS:
            raise ValueError(f"unknown layer {shape['layer']!r}")
        by_layer[shape["layer"]].append({"pts": _pts(shape["pts"]), "attrs": dict(shape.get("attrs") or {})})

    scene: dict[str, Any] = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "reference_frame": reference_frame,
    }
    for layer in ("carriageway", "intersection"):
        if by_layer[layer]:
            scene[layer] = by_layer[layer][-1]["pts"]
    for layer in POLYGON_LIST_LAYERS:
        if by_layer[layer]:
            scene[layer] = [s["pts"] for s in by_layer[layer]]

    arrows = {}
    for s in by_layer["lane_directions"]:
        (x0, y0), (x1, y1) = s["pts"][:2]
        norm = float(np.hypot(x1 - x0, y1 - y0))
        if norm > 0:
            arrows[s["attrs"].get("lane", "")] = [round((x1 - x0) / norm, 4), round((y1 - y0) / norm, 4)]
    lanes, groups = [], defaultdict(list)
    for s in by_layer["lanes"]:
        a = s["attrs"]
        lane: dict[str, Any] = {"id": a.get("id", ""), "polygon": s["pts"]}
        if lane["id"] in arrows:
            lane["direction"] = arrows[lane["id"]]
        for key in ("approach", "signal"):
            if a.get(key):
                lane[key] = a[key]
        if a.get("allowed_exits"):
            lane["allowed_exits"] = _split(a["allowed_exits"])
        if a.get("direction_group"):
            groups[a["direction_group"]].append(lane["id"])
        lanes.append(lane)
    if lanes:
        scene["lanes"] = lanes
    if groups:
        scene["directions"] = [{"id": g, "lanes": ids} for g, ids in groups.items()]

    for layer, key in (("exits", "polygon"), ("crosswalks", "polygon"), ("solid_lines", "polyline")):
        if by_layer[layer]:
            scene[layer] = [{"id": s["attrs"].get("id", ""), key: s["pts"]} for s in by_layer[layer]]
    if by_layer["stop_lines"]:
        scene["stop_lines"] = []
        for s in by_layer["stop_lines"]:
            line: dict[str, Any] = {
                "id": s["attrs"].get("id", ""),
                "line": s["pts"][:2],
                "lanes": _split(s["attrs"].get("lanes", "")),
            }
            if s["attrs"].get("signal"):
                line["signal"] = s["attrs"]["signal"]
            scene["stop_lines"].append(line)

    signals: dict[str, dict[str, Any]] = {}
    for s in by_layer["signals"]:
        sid = s["attrs"].get("id", "")
        signals.setdefault(sid, {"id": sid})[s["attrs"].get("lamp", "")] = _rect_xywh(s["pts"])
    if signals:
        scene["signals"] = list(signals.values())

    if by_layer["homography"]:
        scene["homography"] = {
            "image_pts": [s["pts"][0] for s in by_layer["homography"]],
            "world_pts": [
                [float(s["attrs"].get("world_x", "nan")), float(s["attrs"].get("world_y", "nan"))]
                for s in by_layer["homography"]
            ],
        }
    scene["u_turn_prohibited_everywhere"] = bool(u_turn_prohibited_everywhere)
    return scene


def scene_to_shapes(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Inverse of `shapes_to_scene`: the editable shapes of an existing scene."""
    shapes: list[dict[str, Any]] = []

    def add(layer: str, pts: Any, **attrs: Any) -> None:
        shapes.append({"layer": layer, "pts": _pts(pts), "attrs": {k: str(v) for k, v in attrs.items()}})

    for layer in ("carriageway", "intersection"):
        if scene.get(layer):
            add(layer, scene[layer])
    for layer in POLYGON_LIST_LAYERS:
        for poly in scene.get(layer) or []:
            add(layer, poly)

    group_of = {lane_id: d["id"] for d in scene.get("directions") or [] for lane_id in d.get("lanes", [])}
    for lane in scene.get("lanes") or []:
        add(
            "lanes",
            lane["polygon"],
            id=lane["id"],
            direction_group=group_of.get(lane["id"], ""),
            approach=lane.get("approach", ""),
            signal=lane.get("signal", ""),
            allowed_exits=",".join(lane.get("allowed_exits", [])),
        )
        if lane.get("direction"):
            tail = np.asarray(lane["polygon"], dtype=np.float64).mean(axis=0)
            head = tail + _ARROW_PX * np.asarray(lane["direction"], dtype=np.float64)
            add("lane_directions", [tail, head], lane=lane["id"])

    for layer, key in (("exits", "polygon"), ("crosswalks", "polygon"), ("solid_lines", "polyline")):
        for item in scene.get(layer) or []:
            add(layer, item[key], id=item["id"])
    for sl in scene.get("stop_lines") or []:
        add(
            "stop_lines",
            sl["line"],
            id=sl["id"],
            lanes=",".join(sl.get("lanes", [])),
            signal=sl.get("signal", ""),
        )
    for sig in scene.get("signals") or []:
        for lamp in LAMPS:
            if sig.get(lamp):
                x, y, w, h = sig[lamp]
                add("signals", [[x, y], [x + w, y + h]], id=sig["id"], lamp=lamp)
    h = scene.get("homography")
    if h:
        for (x, y), (wx, wy) in zip(h["image_pts"], h["world_pts"], strict=True):
            add("homography", [[x, y]], world_x=wx, world_y=wy)
    return shapes


def validate_scene(scene: dict[str, Any]) -> list[str]:
    """Human-readable problems of a scene dict; [] when it is consistent."""
    problems: list[str] = []
    size = scene.get("image_size")
    if not (isinstance(size, list) and len(size) == 2 and min(size) > 0):
        problems.append("image_size must be [width, height]")

    ids: dict[str, set[str]] = {}
    for layer in ("lanes", "exits", "crosswalks", "stop_lines", "solid_lines", "signals"):
        seen: set[str] = set()
        for item in scene.get(layer) or []:
            item_id = item.get("id", "")
            if not item_id:
                problems.append(f"{layer}: an item has no id")
            elif item_id in seen:
                problems.append(f"{layer}: duplicate id {item_id!r}")
            seen.add(item_id)
        ids[layer] = seen

    for layer in ("carriageway", "intersection"):
        if scene.get(layer) and len(scene[layer]) < 3:
            problems.append(f"{layer}: a polygon needs at least 3 points")
    for lane in scene.get("lanes") or []:
        if lane.get("signal") and lane["signal"] not in ids["signals"]:
            problems.append(f"lane {lane['id']}: unknown signal {lane['signal']!r}")
        for exit_id in lane.get("allowed_exits", []):
            if exit_id not in ids["exits"]:
                problems.append(f"lane {lane['id']}: unknown exit {exit_id!r}")
    for sl in scene.get("stop_lines") or []:
        for lane_id in sl.get("lanes", []):
            if lane_id not in ids["lanes"]:
                problems.append(f"stop line {sl['id']}: unknown lane {lane_id!r}")
        if sl.get("signal") and sl["signal"] not in ids["signals"]:
            problems.append(f"stop line {sl['id']}: unknown signal {sl['signal']!r}")
    for sig in scene.get("signals") or []:
        missing = [lamp for lamp in LAMPS if lamp not in sig]
        if missing:
            problems.append(f"signal {sig['id']}: missing lamp box(es) {', '.join(missing)}")
    for d in scene.get("directions") or []:
        for lane_id in d.get("lanes", []):
            if lane_id not in ids["lanes"]:
                problems.append(f"direction {d['id']}: unknown lane {lane_id!r}")

    h = scene.get("homography")
    if h:
        img = np.asarray(h.get("image_pts", []), dtype=np.float64)
        world = np.asarray(h.get("world_pts", []), dtype=np.float64)
        if len(img) < 4 or len(img) != len(world):
            problems.append("homography: needs at least 4 image points, each with world coordinates")
        elif not np.isfinite(world).all():
            problems.append("homography: every point needs numeric world_x and world_y (metres)")
        elif not _has_general_quad(img, world) or cv2.findHomography(img, world, 0)[0] is None:
            problems.append("homography: points are degenerate (need 4 with no three on one line)")
    return problems


def _collinear(pts: np.ndarray) -> bool:
    """Whether three points are (nearly) on one line, relative to the size of the whole point set."""
    a, b, c = pts
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    return area < _COLLINEAR_REL_AREA * max(np.ptp(pts, axis=0).max(), 1e-9) ** 2


def _has_general_quad(img: np.ndarray, world: np.ndarray) -> bool:
    """Some 4 points have no three collinear, both in the image and on the ground plane.

    cv2.findHomography does not reject such inputs: it returns a meaningless matrix instead.
    """
    for quad in combinations(range(len(img)), 4):
        if not any(_collinear(pts[list(tri)]) for pts in (img, world) for tri in combinations(quad, 3)):
            return True
    return False
