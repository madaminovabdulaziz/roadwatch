"""Annotated video for the website and demo (WEBSITE_SPEC "Visual rules", RUNBOOK P2.3).

Contract: `render_video(video_path, tt, events, risk, scene, out_path, ...)` writes an H.264 MP4
(libx264, yuv420p, +faststart; FFmpeg through PyAV) with class-coloured boxes, track id + speed, 2 s
footprint trails, semi-transparent scene layers, active-event banners, a risk bar and the signal state.
It never uses Ultralytics `.plot()` or any helper that fetches fonts or assets (CLAUDE.md rule 3).
`tt` is in the pixels of `video_path`; scene layers are in the scene's `image_size` pixels and must be
scaled to the video size (the demo renders on a 720p re-encode).

Details:
- Output height `height` (default 720), even width, the source frame rate; `t0`/`t1` cut a window
  (per-event gallery clips). Track boxes exist only on processed frames (every `stride`-th): between
  them the last boxes are held, which lags at most stride-1 frames.
- Speeds come from `add_kinematics` when the table has no `speed` column and the scene is metric.
- `blur_faces` blurs the top fifth of every person box (privacy on the public website). Plates are not
  blurred: no plate detector is shipped, and at 720p from this camera they are not legible.
- Colours: configs/palette.json (shared with the website).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd

from roadwatch.config import CONFIG_DIR
from roadwatch.features import add_kinematics
from roadwatch.scene.light import SignalTimeline, state_at
from roadwatch.scene.overlay import draw_scene
from roadwatch.scene.scene import Scene
from roadwatch.video import probe, read_window

TRAIL_SEC = 2.0
FACE_FRAC = 0.2  # top share of a person box treated as the face region
_STATE_BGR = {"red": (0, 0, 230), "yellow": (0, 210, 255), "green": (0, 200, 0), "unknown": (120, 120, 120)}


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def load_palette() -> tuple[dict[str, tuple[int, int, int]], dict[str, tuple[int, int, int]]]:
    """(event class -> BGR, object class -> BGR) from configs/palette.json."""
    data = json.loads((CONFIG_DIR / "palette.json").read_text(encoding="utf-8"))
    return (
        {k: _hex_bgr(v) for k, v in data["classes"].items()},
        {k: _hex_bgr(v) for k, v in data["objects"].items()},
    )


def _scaled_scene(scene: Scene, width: int, height: int) -> Scene:
    """The scene with every coordinate scaled from its image_size to (width, height)."""
    size = scene.layers.get("image_size")
    if not size:
        return scene
    sx, sy = width / size[0], height / size[1]

    def scale(value: Any, key: str = "") -> Any:
        if key in (
            "id",
            "lanes",
            "signal",
            "allowed_exits",
            "approach",
            "direction",
            "world_pts",
            "image_size",
        ):
            return value
        if isinstance(value, dict):
            return {k: scale(v, k) for k, v in value.items()}
        if isinstance(value, list) and len(value) == 4 and key in ("red", "yellow", "green"):
            return [value[0] * sx, value[1] * sy, value[2] * sx, value[3] * sy]
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return [value[0] * sx, value[1] * sy]
        if isinstance(value, list):
            return [scale(v, key) for v in value]
        return value

    # top-level layers are all geometry (a layer called "lanes" is polygons); the id-list keys above only
    # apply inside objects (stop_lines[].lanes, directions[].lanes)
    keep = ("image_size", "u_turn_prohibited_everywhere", "reference_frame")
    layers = {k: v if k in keep else scale(v) for k, v in scene.layers.items()}
    layers["image_size"] = [width, height]
    return Scene(layers)


def _label(
    img: np.ndarray, text: str, org: tuple[int, int], colour: tuple[int, int, int], scale: float
) -> None:
    thick = max(1, round(scale * 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


class _Frames:
    """Per processed frame: rows of the (scaled) track table, plus trail lookups."""

    def __init__(self, tt: pd.DataFrame) -> None:
        self.tt = tt
        self.frames = np.unique(tt["frame"].to_numpy()) if len(tt) else np.array([], dtype=int)
        self.by_frame = {int(f): g for f, g in tt.groupby("frame", sort=True)} if len(tt) else {}
        self.by_track = {int(k): g for k, g in tt.groupby("track_id", sort=True)} if len(tt) else {}

    def at(self, idx: int) -> pd.DataFrame | None:
        k = np.searchsorted(self.frames, idx, side="right") - 1
        return self.by_frame[int(self.frames[k])] if k >= 0 else None


def render_video(
    video_path: str | Path,
    tt: pd.DataFrame,
    events: list[list],
    risk: list[list[float]],
    scene: Scene,
    out_path: str | Path,
    *,
    height: int = 720,
    t0: float = 0.0,
    t1: float = math.inf,
    blur_faces: bool = True,
    signal_timeline: SignalTimeline | None = None,
    crf: int = 23,
    preset: str = "veryfast",
) -> Path:
    """Write the annotated MP4 (see the module docstring) and return its path.

    `crf` / `preset` are libx264's: the live demo encodes while the user waits (veryfast); the website's
    full-length samples trade encoding time for half the size (scripts/render_samples.py).
    """
    meta = probe(video_path)
    out_h = min(height, meta.height) // 2 * 2
    out_w = max(2, round(meta.width * out_h / meta.height / 2) * 2)
    k = out_w / meta.width
    event_colours, object_colours = load_palette()
    # calibration points are for the Approach page, not for viewers of the video
    small_scene = _scaled_scene(
        Scene({k: v for k, v in scene.layers.items() if k != "homography"}), out_w, out_h
    )
    signals = signal_timeline or {}

    if len(tt) and "speed" not in tt.columns and scene.has("homography"):
        tt = add_kinematics(_to_scene(tt, meta.width, meta.height, scene), scene)
        tt = _from_scene(tt, meta.width, meta.height, scene)
    view = tt.copy()
    for col in ("x1", "y1", "x2", "y2", "fx", "fy"):
        view[col] = view[col] * k
    lookup = _Frames(view)
    risk_t = np.array([p[0] for p in risk], dtype=np.float64)
    risk_v = np.array([p[1] for p in risk], dtype=np.float64)
    scale = out_h / 720

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(meta.fps).limit_denominator(1001)
    with av.open(str(out_path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = out_w, out_h, "yuv420p"
        stream.options = {"preset": preset, "crf": str(crf)}
        base = (
            draw_scene(np.zeros((out_h, out_w, 3), np.uint8), small_scene, alpha=0.18)
            if scene.layers
            else None
        )
        layer_mask = base.any(axis=2) if base is not None else None
        for n, (idx, t, img) in enumerate(read_window(video_path, t0, t1, size=(out_w, out_h))):
            if base is not None:
                img[layer_mask] = cv2.addWeighted(img, 0.65, base, 0.35, 0)[layer_mask]
            _draw_tracks(img, lookup, idx, t, object_colours, scale, blur_faces)
            _draw_events(img, events, t, event_colours, scale)
            _draw_risk(img, risk_t, risk_v, t, scale)
            _draw_signals(img, signals, t, scale)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = n, 1 / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return out_path


def _to_scene(tt: pd.DataFrame, width: int, height: int, scene: Scene) -> pd.DataFrame:
    size = scene.layers.get("image_size") or [width, height]
    out = tt.copy()
    for col, s in (("x1", size[0] / width), ("x2", size[0] / width), ("fx", size[0] / width)):
        out[col] = out[col] * s
    for col, s in (("y1", size[1] / height), ("y2", size[1] / height), ("fy", size[1] / height)):
        out[col] = out[col] * s
    return out


def _from_scene(tt: pd.DataFrame, width: int, height: int, scene: Scene) -> pd.DataFrame:
    size = scene.layers.get("image_size") or [width, height]
    out = tt.copy()
    for col in ("x1", "x2", "fx"):
        out[col] = out[col] * width / size[0]
    for col in ("y1", "y2", "fy"):
        out[col] = out[col] * height / size[1]
    return out


def _draw_tracks(img, lookup: _Frames, idx: int, t: float, colours, scale: float, blur_faces: bool) -> None:
    rows = lookup.at(idx)
    if rows is None:
        return
    h, w = img.shape[:2]
    for r in rows.itertuples():
        colour = colours.get(str(r.cls), colours["other"])
        x1, y1, x2, y2 = (int(round(v)) for v in (r.x1, r.y1, r.x2, r.y2))
        if blur_faces and str(r.cls) == "person":
            fy2 = y1 + max(2, round((y2 - y1) * FACE_FRAC))
            xa, xb, ya, yb = max(0, x1), min(w, x2), max(0, y1), min(h, fy2)
            if xb - xa > 1 and yb - ya > 1:
                img[ya:yb, xa:xb] = cv2.GaussianBlur(
                    img[ya:yb, xa:xb], (0, 0), sigmaX=max(2.0, (xb - xa) / 4)
                )
        trail = lookup.by_track[int(r.track_id)]
        trail = trail[(trail["t"] <= t) & (trail["t"] >= t - TRAIL_SEC)]
        if len(trail) > 1:
            pts = trail[["fx", "fy"]].to_numpy().round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, colour, max(1, round(2 * scale)), cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(1, round(2 * scale)))
        text = f"{r.track_id}"
        speed = getattr(r, "speed", float("nan"))
        if isinstance(speed, float) and math.isfinite(speed) and str(r.cls) != "person":
            text += f" {speed * 3.6:.0f} km/h"
        _label(img, text, (x1, max(12, y1 - 4)), colour, 0.45 * scale)


def _draw_events(img, events: list[list], t: float, colours, scale: float) -> None:
    active = [e for e in events if e[0] <= t <= e[1]]
    for i, (_, _, label) in enumerate(active):
        colour = colours.get(label, (200, 200, 200))
        y = round((12 + 34 * i) * scale)
        cv2.rectangle(img, (round(10 * scale), y), (round(260 * scale), y + round(28 * scale)), colour, -1)
        _label(
            img,
            label.replace("_", " ").upper(),
            (round(18 * scale), y + round(20 * scale)),
            (255, 255, 255),
            0.6 * scale,
        )


def _draw_risk(img, risk_t: np.ndarray, risk_v: np.ndarray, t: float, scale: float) -> None:
    if not len(risk_t):
        return
    r = float(np.interp(t, risk_t, risk_v))
    w = img.shape[1]
    x0, y0, bw, bh = w - round(230 * scale), round(14 * scale), round(200 * scale), round(18 * scale)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (40, 40, 40), -1)
    colour = (0, round(255 * (1 - r)), round(255 * min(1.0, 2 * r)))  # green -> red
    cv2.rectangle(img, (x0, y0), (x0 + round(bw * r), y0 + bh), colour, -1)
    cv2.line(img, (x0 + bw // 2, y0 - 3), (x0 + bw // 2, y0 + bh + 3), (255, 255, 255), 1)
    _label(img, f"risk {r:.2f}", (x0, y0 + bh + round(18 * scale)), (255, 255, 255), 0.5 * scale)


def _draw_signals(img, signals: SignalTimeline, t: float, scale: float) -> None:
    for i, (sid, segs) in enumerate(sorted(signals.items())):
        state = state_at(segs, t)
        centre = (img.shape[1] - round(30 * scale), round((80 + 30 * i) * scale))
        cv2.circle(
            img, centre, round(10 * scale), _STATE_BGR.get(state, _STATE_BGR["unknown"]), -1, cv2.LINE_AA
        )
        _label(
            img,
            sid,
            (centre[0] - round(60 * scale), centre[1] + round(5 * scale)),
            (255, 255, 255),
            0.5 * scale,
        )
