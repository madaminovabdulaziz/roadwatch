"""Scene registration (SPEC §12.34): camera moves recovered, geometry mapped exactly, Part B causal.

Synthetic "videos" are the real reference frame warped by a known homography, with moving boxes
(traffic) painted on each frame, and a dark variant for dusk.
"""

from __future__ import annotations

import builtins

import cv2
import numpy as np
import pytest

from roadwatch.config import load_thresholds
from roadwatch.scene import registration
from roadwatch.scene.registration import OnlineRegistration, align_scene, can_register, estimate
from roadwatch.scene.scene import Scene

NATIVE = (3840, 2160)
CFG = load_thresholds()["registration"]
WORK = CFG["work_width"]
S = WORK / NATIVE[0]  # native -> work scale (both images are 16:9 4K)

# a tripod re-set like C3902's: ~90 px right, ~60 px up, 2 % zoom, a little rotation (native px)
CAMERA_MOVE = np.array([[0.98, -0.006, 110.0], [0.006, 0.98, -20.0], [0.0, 0.0, 1.0]])


def scene_with_reference() -> Scene:
    return Scene(
        {
            "image_size": list(NATIVE),
            "reference_frame": "configs/reference.jpg",
            "carriageway": [[500, 800], [3500, 800], [3500, 2100], [500, 2100]],
            "lanes": [
                {
                    "id": "n1",
                    "polygon": [[1000, 900], [1300, 900], [1300, 1500], [1000, 1500]],
                    "direction": [0, 1],
                }
            ],
            "stop_lines": [{"id": "sl", "line": [[1500, 1000], [2500, 1000]], "lanes": ["n1"]}],
            "crosswalks": [{"id": "cw", "polygon": [[800, 1200], [2600, 1200], [2600, 1400], [800, 1400]]}],
            "solid_lines": [{"id": "s1", "polyline": [[900, 900], [1400, 1700], [1600, 2100]]}],
            "signals": [{"id": "L1", "red": [2000, 600, 18, 18], "green": [2000, 642, 18, 18]}],
            "sidewalks": [[[100, 100], [400, 100], [400, 400]]],
            "homography": {
                "image_pts": [[1000, 1000], [2000, 1000], [2200, 1800], [800, 1800]],
                "world_pts": [[0, 0], [10, 0], [10, 20], [0, 20]],
            },
        }
    )


def reference_work() -> np.ndarray:
    img = cv2.imread(str(registration.REPO_ROOT / "configs" / "reference.jpg"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (WORK, round(img.shape[0] * S)), interpolation=cv2.INTER_AREA)


def video_frames(ref_to_video: np.ndarray, n: int = 5, dark: bool = False) -> list[np.ndarray]:
    """Work-size grayscale frames of a camera whose native pixels are ref_to_video @ reference pixels."""
    to_work = np.diag([S, S, 1.0])
    h_work = to_work @ ref_to_video @ np.linalg.inv(to_work)
    base = cv2.warpPerspective(reference_work(), h_work, (WORK, round(NATIVE[1] * S)))
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        f = base.copy()
        for _ in range(25):  # traffic: boxes in different places in every frame
            x, y = int(rng.integers(0, WORK - 120)), int(rng.integers(300, 1000))
            cv2.rectangle(f, (x, y), (x + 110, y + 60), int(rng.integers(0, 255)), -1)
        if dark:
            f = (255 * (f / 255.0) ** 1.8 * 0.45).astype(np.uint8)
        frames.append(f)
    return frames


def moved(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


PROBES = np.array([[1920, 1080], [600, 1800], [3200, 600], [1920, 300]], dtype=np.float64)


@pytest.mark.parametrize("dark", [False, True])
def test_a_camera_move_is_recovered_to_a_pixel(dark: bool) -> None:
    reg = estimate(video_frames(CAMERA_MOVE, dark=dark), NATIVE, scene_with_reference())

    assert reg is not None and reg.confident
    video_to_ref_true = np.linalg.inv(CAMERA_MOVE)
    err = np.abs(moved(reg.video_to_ref, PROBES) - moved(video_to_ref_true, PROBES)).max()
    assert err < 2.0, f"max probe error {err:.2f} px (native)"


def test_an_unchanged_camera_gives_identity() -> None:
    reg = estimate(video_frames(np.eye(3)), NATIVE, scene_with_reference())
    assert reg is not None and reg.confident
    assert np.abs(moved(reg.video_to_ref, PROBES) - PROBES).max() < 1.0


def test_a_different_scene_is_not_confident() -> None:
    noise = [np.random.default_rng(i).integers(0, 255, (1080, WORK), dtype=np.uint8) for i in range(3)]
    reg = estimate(noise, NATIVE, scene_with_reference())
    assert reg is None or not reg.confident
    assert align_scene(scene_with_reference(), reg, NATIVE).layers == scene_with_reference().layers


def test_transformed_scene_maps_every_layer_and_keeps_world_coordinates() -> None:
    scene = scene_with_reference()
    aligned = scene.transformed(CAMERA_MOVE, NATIVE)

    for layer, key in (("stop_lines", "line"), ("crosswalks", "polygon"), ("solid_lines", "polyline")):
        np.testing.assert_allclose(
            aligned.layers[layer][0][key], moved(CAMERA_MOVE, scene.layers[layer][0][key]), atol=0.01
        )
    np.testing.assert_allclose(
        aligned.layers["carriageway"], moved(CAMERA_MOVE, scene.layers["carriageway"]), atol=0.01
    )
    np.testing.assert_allclose(
        aligned.layers["sidewalks"][0], moved(CAMERA_MOVE, scene.layers["sidewalks"][0]), atol=0.01
    )
    red = aligned.layers["signals"][0]["red"]
    centre = moved(CAMERA_MOVE, [[2009, 609]])[0]
    assert red[0] < centre[0] < red[0] + red[2] and red[1] < centre[1] < red[1] + red[3]
    assert np.hypot(*aligned.layers["lanes"][0]["direction"]) == pytest.approx(1.0)

    # a point on the road has the same world coordinates seen from either framing
    ref_pts = np.array([[1500, 1300], [2100, 1700], [900, 1100]], dtype=np.float64)
    np.testing.assert_allclose(
        aligned.to_world(moved(CAMERA_MOVE, ref_pts)), scene.to_world(ref_pts), atol=1e-6
    )
    assert scene.layers["stop_lines"][0]["line"] == [[1500, 1000], [2500, 1000]]  # the original is untouched


def test_scenes_without_a_reference_frame_are_not_registered() -> None:
    assert not can_register(Scene())
    assert not can_register(Scene({"homography": {"image_pts": [[0, 0]], "world_pts": [[0, 0]]}}))
    assert can_register(scene_with_reference())


def test_online_registration_is_causal_and_reads_no_files_after_init(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = scene_with_reference()
    online = OnlineRegistration(scene, NATIVE)

    def no_files(*args: object, **kwargs: object) -> None:
        raise AssertionError("OnlineRegistration.offer must not touch files")

    frame = cv2.cvtColor(cv2.resize(video_frames(CAMERA_MOVE, n=1)[0], NATIVE), cv2.COLOR_GRAY2BGR)
    monkeypatch.setattr(builtins, "open", no_files)
    monkeypatch.setattr(cv2, "imread", no_files)

    aligned = online.offer(frame, 0.0)

    assert aligned is not None and online.done
    np.testing.assert_allclose(
        aligned.layers["stop_lines"][0]["line"],
        moved(CAMERA_MOVE, scene.layers["stop_lines"][0]["line"]),
        atol=3.0,
    )


def test_online_registration_waits_between_tries_and_gives_up() -> None:
    online = OnlineRegistration(scene_with_reference(), NATIVE)
    noise = np.random.default_rng(1).integers(0, 255, (NATIVE[1], NATIVE[0], 3), dtype=np.uint8)
    t, tries = 0.0, 0
    while not online.done:
        before = len(online.frames)
        assert online.offer(noise, t) is None
        tries += len(online.frames) > before
        t += 0.5
    assert tries == CFG["online_max_frames"]
    assert t >= CFG["online_retry_sec"] * (CFG["online_max_frames"] - 1)
