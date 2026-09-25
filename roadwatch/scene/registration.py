"""Per-video alignment of the hand-calibrated scene (SPEC §12.34).

Every scene coordinate was clicked on the reference frame (configs/reference.jpg), but the tripod is set
up again for every recording: on the samples, C3902 sits 83-94 px right and 33-91 px up of the C3896
reference (4K px, zoom 2 % different) and C3905 up to 56 px off. A stop line 100 px off is 1-2 m on
the road, so every video is registered to the reference frame and the scene is mapped into it.

Contract:
- `estimate(frames, cfg)` -> `Registration`: the homography from video pixels to reference pixels (native
  pixels of each), fitted on SIFT features of the per-pixel median of `frames` (moving traffic drops
  out), CLAHE-normalised against lighting changes (dusk), Lowe ratio test, RANSAC with a fixed seed.
  `confident` = enough inliers, a high enough inlier share and a plausible camera move (no frame corner
  moves more than `max_corner_shift_frac` of the width).
- `align_scene(scene, reg, video_size)` -> the scene in video pixels. A missing or unconfident
  registration leaves the scene as clicked: the organizers state the test videos come from the same
  camera and angle, so "no confident fit" means few usable features, not a different camera.
- `register_video(path)` (Part A) samples `frames` frames spread over the video. Part B registers on
  frames it is given (`OnlineRegistration`), so it stays causal.
- Deterministic: SIFT, the matcher and RANSAC (cv2.setRNGSeed) give the same result for the same input.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roadwatch.config import REPO_ROOT, load_thresholds
from roadwatch.scene.scene import Scene
from roadwatch.video import probe, read_window

log = logging.getLogger(__name__)

_RANSAC_SEED = 0
_CLAHE = {"clipLimit": 2.0, "tileGridSize": (8, 8)}
_MIN_RANSAC_MATCHES = 4


@dataclass(frozen=True)
class Registration:
    video_to_ref: np.ndarray  # 3x3, native video px -> native reference px
    inliers: int
    matches: int
    corner_shift_px: float  # largest displacement of a video frame corner, reference px
    confident: bool

    def summary(self) -> dict[str, Any]:
        return {
            "video_to_ref": np.round(self.video_to_ref, 9).tolist(),
            "inliers": self.inliers,
            "matches": self.matches,
            "corner_shift_px": round(self.corner_shift_px, 1),
            "confident": self.confident,
        }

    @classmethod
    def from_summary(cls, data: dict[str, Any]) -> Registration:
        return cls(
            np.asarray(data["video_to_ref"], dtype=np.float64),
            int(data["inliers"]),
            int(data["matches"]),
            float(data["corner_shift_px"]),
            bool(data["confident"]),
        )


def reference_path(scene: Scene) -> Path | None:
    """The frame the scene was clicked on, or None if the scene names none (nothing to register to)."""
    ref = scene.layers.get("reference_frame")
    return REPO_ROOT / ref if ref else None


def can_register(scene: Scene) -> bool:
    ref = reference_path(scene)
    return ref is not None and ref.exists()


@functools.lru_cache(maxsize=4)
def _reference_features(path: str, work_width: int, n_features: int) -> tuple[Any, np.ndarray, float]:
    """SIFT keypoints/descriptors of the reference frame at work size, and its native->work scale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read the reference frame {path}")
    scale = work_width / img.shape[1]
    work = cv2.resize(img, (work_width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = _sift(n_features).detectAndCompute(_normalise(work), None)
    return keypoints, descriptors, scale


def _sift(n_features: int) -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=n_features)


def _normalise(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(**_CLAHE).apply(gray)


def to_work_gray(frame_bgr: np.ndarray, work_width: int) -> np.ndarray:
    """A frame as grayscale at the registration working width (what `estimate` consumes)."""
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (work_width, round(h * work_width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def estimate(
    work_frames: list[np.ndarray],
    native_size: tuple[int, int],
    scene: Scene,
    cfg: dict[str, Any] | None = None,
) -> Registration | None:
    """Fit video -> reference on grayscale work-size frames of one video; None if nothing matched."""
    cfg = cfg or load_thresholds()["registration"]
    if not work_frames:
        return None
    ref_kp, ref_desc, ref_scale = _reference_features(
        str(reference_path(scene)), cfg["work_width"], cfg["sift_features"]
    )
    background = np.median(np.stack(work_frames), axis=0).astype(np.uint8)
    kp, desc = _sift(cfg["sift_features"]).detectAndCompute(_normalise(background), None)
    if desc is None or ref_desc is None or len(kp) < _MIN_RANSAC_MATCHES:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, ref_desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg["ratio"] * p[1].distance]
    if len(good) < _MIN_RANSAC_MATCHES:
        return None
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(_RANSAC_SEED)
    H_work, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg["ransac_px"])
    if H_work is None:
        return None

    vid_scale = cfg["work_width"] / native_size[0]
    to_work = np.diag([vid_scale, vid_scale, 1.0])
    from_ref_work = np.diag([1 / ref_scale, 1 / ref_scale, 1.0])
    H = from_ref_work @ H_work @ to_work
    H /= H[2, 2]
    w, h = native_size
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    shift = float(np.max(np.hypot(*(moved - corners).T)))
    inliers = int(mask.sum())
    confident = (
        inliers >= cfg["min_inliers"]
        and inliers >= cfg["min_inlier_ratio"] * len(good)
        and shift <= cfg["max_corner_shift_frac"] * w
    )
    return Registration(H, inliers, len(good), shift, bool(confident))


def align_scene(scene: Scene, reg: Registration | None, video_size: tuple[int, int]) -> Scene:
    """The scene in this video's pixels (unchanged without a confident registration)."""
    if reg is None or not reg.confident:
        return scene
    return scene.transformed(np.linalg.inv(reg.video_to_ref), video_size)


def register_video(
    video_path: str | Path, scene: Scene, cfg: dict[str, Any] | None = None
) -> Registration | None:
    """Part A: register a video from `frames` frames spread over it (read by seeking)."""
    cfg = cfg or load_thresholds()["registration"]
    meta = probe(video_path)
    work_h = round(meta.height * cfg["work_width"] / meta.width)
    frames = []
    for frac in np.linspace(0.05, 0.95, cfg["frames"]):
        t0 = float(frac * meta.duration)
        window = read_window(video_path, t0, t0 + 2.0, size=(cfg["work_width"], work_h), skip_nonref=True)
        try:
            first = next(iter(window), None)
        finally:
            window.close()
        if first is not None:
            frames.append(cv2.cvtColor(first[2], cv2.COLOR_BGR2GRAY))
    return estimate(frames, (meta.width, meta.height), scene, cfg)


def scene_for_video(video_path: str | Path, scene: Scene | None = None) -> tuple[Scene, Registration | None]:
    """Part A: the scene aligned to one video, and the registration used (None if it failed)."""
    scene = scene if scene is not None else Scene.load()
    if not can_register(scene):
        return scene, None
    meta = probe(video_path)
    try:
        reg = register_video(video_path, scene)
    except Exception:
        log.exception("%s: registration failed; using the scene as clicked", meta.video_id)
        return scene, None
    _log(meta.video_id, reg)
    return align_scene(scene, reg, (meta.width, meta.height)), reg


def _log(video_id: str, reg: Registration | None) -> None:
    if reg is None:
        log.warning("%s: no registration (too few features); using the scene as clicked", video_id)
    elif reg.confident:
        log.info(
            "%s: scene aligned (%d/%d inliers, corners moved <= %.0f px)",
            video_id,
            reg.inliers,
            reg.matches,
            reg.corner_shift_px,
        )
    else:
        log.warning(
            "%s: registration not confident (%d/%d inliers, %.0f px); using the scene as clicked",
            video_id,
            reg.inliers,
            reg.matches,
            reg.corner_shift_px,
        )


class OnlineRegistration:
    """Part B: register from frames seen so far (causal); gives up after `online_max_frames` tries."""

    def __init__(self, scene: Scene, native_size: tuple[int, int], cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_thresholds()["registration"]
        self.scene = scene
        self.native_size = native_size
        self.frames: list[np.ndarray] = []
        self.next_t = 0.0
        self.done = not can_register(scene)
        self.result: Registration | None = None
        if not self.done:  # load the reference now: step() must not touch files
            _reference_features(str(reference_path(scene)), self.cfg["work_width"], self.cfg["sift_features"])

    def offer(self, frame_bgr: np.ndarray, t_sec: float) -> Scene | None:
        """Take a frame if one is due; return the aligned scene once the fit is confident."""
        if self.done or t_sec < self.next_t:
            return None
        self.next_t = t_sec + self.cfg["online_retry_sec"]
        self.frames.append(to_work_gray(frame_bgr, self.cfg["work_width"]))
        self.result = estimate(self.frames, self.native_size, self.scene, self.cfg)
        if self.result is not None and self.result.confident:
            self.done = True
            return align_scene(self.scene, self.result, self.native_size)
        if len(self.frames) >= self.cfg["online_max_frames"]:
            self.done = True
        return None
