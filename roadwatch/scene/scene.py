"""Hand-calibrated scene geometry (SPEC §4, RUNBOOK P1.1).

Contract:
- `Scene.load(path)` reads configs/scene.json; a missing file gives an empty scene, so every rule that
  needs geometry disables itself instead of guessing (SPEC §12.7).
- `Scene.has(layer)` tells whether a layer (e.g. "lanes", "stop_lines", "homography") is present.
- Geometry helpers work on pixel coordinates of the native resolution (3840x2160 for the samples):
  `point_in(layer, pts)`, `region_of(layer, pts)`, `lane_of(pts)`, `crosses(line, p0, p1)`, and the
  homography pair `to_world(pts)` / `to_image(pts)` (metres on the road plane).
- `transformed(H, image_size)` maps every layer through a pixel homography, e.g. from the reference
  frame into one video's framing (roadwatch/scene/registration.py). The road-plane mapping is composed
  exactly (`homography.matrix`), not refitted from moved points.

Polygon layers come in three shapes (SPEC §4): one polygon (`carriageway`, `intersection`), a list of
polygons (`sidewalks`, `parking_zones`, `no_u_turn_zones`), or a list of objects with `id` and `polygon`
(`lanes`, `exits`, `crosswalks`). All helpers accept any of them.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roadwatch.config import SCENE_PATH


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting: boolean mask of which (N, 2) points lie inside the (M, 2) polygon."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        straddles = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (x < x_cross)
    return inside


def _orient(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Sign of the cross product (b - a) x (p - a): which side of line a->b each point p is on."""
    return np.sign(
        (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])
    )


def segments_cross(line: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Whether each step p0[i] -> p1[i] crosses the 2-point segment `line`.

    Half-open: a step that lands exactly on the line counts, a step that leaves it does not, so a
    track sampled exactly on the line is counted once.
    """
    a, b = (np.asarray(v, dtype=np.float64) for v in np.asarray(line, dtype=np.float64).reshape(2, 2))
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    s0, s1 = _orient(a, b, p0), _orient(a, b, p1)
    changes_side = (s0 != 0) & (s1 != s0)
    within_segment = _orient(p0, p1, a) * _orient(p0, p1, b) <= 0
    return changes_side & within_segment


@dataclass
class Scene:
    layers: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = SCENE_PATH) -> Scene:
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def has(self, layer: str) -> bool:
        return bool(self.layers.get(layer))

    def polygons(self, layer: str) -> list[tuple[str, np.ndarray]]:
        """(id, (M, 2) polygon) for every polygon of a layer; ids are `<layer>_<i>` when unnamed."""
        value = self.layers.get(layer) or []
        if not value:
            return []
        if isinstance(value[0], dict):
            return [(str(item["id"]), np.asarray(item["polygon"], dtype=np.float64)) for item in value]
        if isinstance(value[0][0], (int, float)):  # a single polygon: [[x, y], ...]
            return [(f"{layer}_0", np.asarray(value, dtype=np.float64))]
        return [(f"{layer}_{i}", np.asarray(poly, dtype=np.float64)) for i, poly in enumerate(value)]

    def point_in(self, layer: str, pts: np.ndarray) -> np.ndarray:
        """Boolean mask: which (N, 2) points lie inside any polygon of `layer` (all False if absent)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        mask = np.zeros(len(pts), dtype=bool)
        for _, poly in self.polygons(layer):
            mask |= points_in_polygon(pts, poly)
        return mask

    def region_of(self, layer: str, pts: np.ndarray) -> np.ndarray:
        """Id of the first polygon of `layer` containing each (N, 2) point, or "" outside all of them."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ids = np.full(len(pts), "", dtype=object)
        for region_id, poly in self.polygons(layer):
            free = ids == ""
            ids[free & points_in_polygon(pts, poly)] = region_id
        return ids

    def lane_of(self, pts: np.ndarray) -> np.ndarray:
        """Lane id for each (N, 2) point, or "" outside every lane."""
        return self.region_of("lanes", pts)

    def lane_directions(self) -> dict[str, np.ndarray]:
        """Image-space unit direction vector of every lane that has one."""
        out = {}
        for lane in self.layers.get("lanes") or []:
            d = np.asarray(lane.get("direction") or (0.0, 0.0), dtype=np.float64)
            norm = np.hypot(*d)
            if norm > 0:
                out[str(lane["id"])] = d / norm
        return out

    def crosses(self, line: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Whether each step p0[i] -> p1[i] crosses the 2-point `line`."""
        return segments_cross(line, p0, p1)

    def scaled_to(self, image_size: tuple[int, int]) -> Scene:
        """This scene in the pixels of a resized copy of its image (itself if the size already matches
        or the scene records no `image_size`)."""
        size = self.layers.get("image_size")
        if not size or (int(size[0]), int(size[1])) == (int(image_size[0]), int(image_size[1])):
            return self
        return self.transformed(np.diag([image_size[0] / size[0], image_size[1] / size[1], 1.0]), image_size)

    def transformed(self, H: np.ndarray, image_size: tuple[int, int]) -> Scene:
        """This scene with every pixel coordinate mapped through the 3x3 homography `H`.

        Polygons, lines and polylines map point by point; lamp boxes become the bounding box of their
        mapped corners; lane directions are re-measured at the lane's centroid; the road-plane
        homography becomes `world_from_old @ inv(H)` exactly. Non-geometric attributes are kept.
        """
        H = np.asarray(H, dtype=np.float64)
        layers = json.loads(json.dumps(self.layers))  # deep copy of plain JSON data
        layers["image_size"] = [int(image_size[0]), int(image_size[1])]
        for layer in ("carriageway", "intersection"):
            if layers.get(layer):
                layers[layer] = _map_list(H, layers[layer])
        for layer in ("sidewalks", "islands", "parking_zones", "bus_stops", "no_u_turn_zones"):
            if layers.get(layer):
                layers[layer] = [_map_list(H, poly) for poly in layers[layer]]
        for layer, key in (("exits", "polygon"), ("crosswalks", "polygon"), ("solid_lines", "polyline")):
            for item in layers.get(layer) or []:
                item[key] = _map_list(H, item[key])
        for lane in layers.get("lanes") or []:
            poly = np.asarray(lane["polygon"], dtype=np.float64)
            if lane.get("direction"):
                centre = poly.mean(axis=0)
                a, b = _map(H, np.stack([centre, centre + 100.0 * np.asarray(lane["direction"], float)]))
                d = (b - a) / max(float(np.hypot(*(b - a))), 1e-12)
                lane["direction"] = [round(float(d[0]), 6), round(float(d[1]), 6)]
            lane["polygon"] = _map_list(H, poly)
        for line in layers.get("stop_lines") or []:
            line["line"] = _map_list(H, line["line"])
        for sig in layers.get("signals") or []:
            for lamp in ("red", "yellow", "green"):
                if sig.get(lamp):
                    x, y, w, h = sig[lamp]
                    corners = _map(H, np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], float))
                    lo, hi = corners.min(axis=0), corners.max(axis=0)
                    sig[lamp] = [round(float(v), 2) for v in (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])]
        if layers.get("homography"):
            world_from_new = self._homography[0] @ np.linalg.inv(H)
            layers["homography"]["image_pts"] = _map_list(H, layers["homography"]["image_pts"])
            layers["homography"]["matrix"] = (world_from_new / world_from_new[2, 2]).tolist()
        return Scene(layers)

    @functools.cached_property
    def _homography(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.layers.get("homography")
        if not h:
            raise ValueError("scene has no homography; metric rules must be disabled (SPEC §12.7)")
        if h.get("matrix") is not None:  # composed by `transformed`, exact
            H = np.asarray(h["matrix"], dtype=np.float64)
            return H, np.linalg.inv(H)
        img = np.asarray(h["image_pts"], dtype=np.float64)
        world = np.asarray(h["world_pts"], dtype=np.float64)
        H, _ = cv2.findHomography(img, world, 0)
        if H is None:
            raise ValueError("homography points are degenerate (collinear or repeated)")
        return H, np.linalg.inv(H)

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> road-plane metres (N, 2) via the homography."""
        return _apply_homography(self._homography[0], pts)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        """Road-plane metres (N, 2) -> pixels (N, 2)."""
        return _apply_homography(self._homography[1], pts)


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts.copy()
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def _map(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return _apply_homography(H, pts)


def _map_list(H: np.ndarray, pts: Any) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in _map(H, np.asarray(pts, dtype=np.float64))]
