"""Draw scene layers on an image (scripts/render_scene.py now, roadwatch/render.py later).

Contract: `draw_scene(img, scene, alpha)` returns a copy of the BGR image with every present layer drawn
at native-resolution coordinates: filled polygons blended at `alpha`, lines and outlines opaque, and
ids as labels. OpenCV's built-in Hershey font only, so nothing is ever fetched (CLAUDE.md rule 3).
"""

from __future__ import annotations

import cv2
import numpy as np

from roadwatch.scene.scene import Scene

# BGR colour per layer; the browser tool uses the same colours (scripts/calibrate_scene.html).
LAYER_COLOURS: dict[str, tuple[int, int, int]] = {
    "carriageway": (128, 128, 128),
    "sidewalks": (180, 130, 70),
    "islands": (150, 200, 120),
    "parking_zones": (200, 80, 200),
    "bus_stops": (40, 180, 240),
    "intersection": (0, 200, 255),
    "lanes": (80, 200, 80),
    "exits": (255, 200, 0),
    "crosswalks": (255, 255, 255),
    "no_u_turn_zones": (60, 60, 230),
    "stop_lines": (0, 0, 255),
    "solid_lines": (0, 255, 255),
    "signals": (0, 140, 255),
    "homography": (255, 0, 255),
}
_FILL_ORDER = (
    "carriageway",
    "sidewalks",
    "islands",
    "parking_zones",
    "bus_stops",
    "intersection",
    "lanes",
    "exits",
    "crosswalks",
    "no_u_turn_zones",
)


def _ipts(pts: object) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _label(img: np.ndarray, text: str, at: np.ndarray, colour: tuple[int, int, int], scale: float) -> None:
    if not text:
        return
    org = (int(at[0]), int(at[1]))
    thick = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_scene(img: np.ndarray, scene: Scene, alpha: float = 0.25) -> np.ndarray:
    """Copy of `img` with the scene's layers drawn on it (see the module docstring)."""
    out = img.copy()
    scale = max(0.5, img.shape[1] / 1920)  # text and line size follow the resolution
    line_w = max(1, int(round(2 * scale)))

    fill = out.copy()
    for layer in _FILL_ORDER:
        for _, poly in scene.polygons(layer):
            cv2.fillPoly(fill, [_ipts(poly)], LAYER_COLOURS[layer])
    cv2.addWeighted(fill, alpha, out, 1 - alpha, 0, dst=out)

    for layer in _FILL_ORDER:
        colour = LAYER_COLOURS[layer]
        for region_id, poly in scene.polygons(layer):
            cv2.polylines(out, [_ipts(poly)], True, colour, line_w, cv2.LINE_AA)
            if not region_id.startswith(f"{layer}_"):  # named regions only (lanes, exits, crosswalks)
                _label(out, region_id, np.asarray(poly).mean(axis=0), colour, scale)

    for lane_id, direction in scene.lane_directions().items():
        poly = dict(scene.polygons("lanes"))[lane_id]
        tail = np.asarray(poly).mean(axis=0)
        head = tail + direction * 150 * scale
        cv2.arrowedLine(
            out,
            tuple(_ipts(tail)[0, 0]),
            tuple(_ipts(head)[0, 0]),
            LAYER_COLOURS["lanes"],
            line_w * 2,
            cv2.LINE_AA,
            tipLength=0.25,
        )

    for item in scene.layers.get("stop_lines") or []:
        cv2.polylines(out, [_ipts(item["line"])], False, LAYER_COLOURS["stop_lines"], line_w * 2, cv2.LINE_AA)
        _label(out, item["id"], np.asarray(item["line"])[0], LAYER_COLOURS["stop_lines"], scale)
    for item in scene.layers.get("solid_lines") or []:
        cv2.polylines(
            out, [_ipts(item["polyline"])], False, LAYER_COLOURS["solid_lines"], line_w, cv2.LINE_AA
        )
        _label(out, item["id"], np.asarray(item["polyline"])[0], LAYER_COLOURS["solid_lines"], scale)
    for sig in scene.layers.get("signals") or []:
        for lamp, colour in (("red", (0, 0, 255)), ("yellow", (0, 255, 255)), ("green", (0, 255, 0))):
            if sig.get(lamp):
                x, y, w, h = (int(round(v)) for v in sig[lamp])
                cv2.rectangle(out, (x, y), (x + w, y + h), colour, line_w)
        first = next((sig[lamp] for lamp in ("red", "yellow", "green") if sig.get(lamp)), None)
        if first:
            _label(
                out, sig["id"], np.array([first[0], first[1] - 8 * scale]), LAYER_COLOURS["signals"], scale
            )
    h = scene.layers.get("homography")
    if h:
        for (x, y), (wx, wy) in zip(h["image_pts"], h["world_pts"], strict=True):
            cv2.circle(
                out, (int(round(x)), int(round(y))), line_w * 4, LAYER_COLOURS["homography"], -1, cv2.LINE_AA
            )
            _label(
                out,
                f"({wx:g}, {wy:g}) m",
                np.array([x + 10 * scale, y]),
                LAYER_COLOURS["homography"],
                scale * 0.8,
            )
    return out
