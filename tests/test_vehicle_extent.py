"""Front/rear bumper points from the ground-plane vehicle model (SPEC §12.35) against a pinhole camera.

A camera like the samples' (12 m up, pitched 35 degrees down, 4K) projects a real 3-D car box; the
detector box is the bounding box of its projection. The road-plane homography is the camera's exact
ground mapping, so the only error left is the model's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roadwatch.features import add_kinematics, vehicle_corners
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES

F, CX, CY, CAM_H, PITCH = 2800.0, 1920.0, 1080.0, 12.0, np.radians(35)
R = np.array([[1, 0, 0], [0, -np.sin(PITCH), -np.cos(PITCH)], [0, np.cos(PITCH), -np.sin(PITCH)]])
C = np.array([0.0, 0.0, CAM_H])
CAR_L, CAR_W, CAR_H = 4.6, 1.8, 1.5  # thresholds.yaml car dims; height only shapes the image box
FPS, STEP = 30000 / 1001, 3


def project(P: np.ndarray) -> np.ndarray:
    pc = (R @ (P - C).T).T
    return np.stack([F * pc[:, 0] / pc[:, 2] + CX, F * pc[:, 1] / pc[:, 2] + CY], axis=1)


def ground_scene() -> Scene:
    world = np.array([[-10, 15], [10, 15], [10, 70], [-10, 70]], dtype=np.float64)
    image = project(np.c_[world, np.zeros(4)])
    return Scene({"homography": {"image_pts": image.tolist(), "world_pts": world.tolist()}})


def car_track(start: tuple[float, float], heading_deg: float, speed: float = 8.0, secs: float = 2.0):
    """TrackTable of a car driving straight, and the true front/rear bumper centres per row."""
    h = np.array([np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))])
    n = np.array([-h[1], h[0]])
    rows, fronts, rears = [], [], []
    for k in range(int(secs * FPS / STEP)):
        t = k * STEP / FPS
        centre = np.asarray(start, dtype=np.float64) + speed * t * h
        corners = [centre + a * CAR_L / 2 * h + b * CAR_W / 2 * n for a in (-1, 1) for b in (-1, 1)]
        pts = np.array([[*c, z] for c in corners for z in (0.0, CAR_H)])
        uv = project(pts)
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        rows.append(
            {
                "frame": k * STEP,
                "t": t,
                "track_id": 1,
                "cls": "car",
                "conf": 0.9,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "fx": (x1 + x2) / 2,
                "fy": y2,
            }
        )
        fronts.append(centre + CAR_L / 2 * h)
        rears.append(centre - CAR_L / 2 * h)
    tt = pd.DataFrame(rows).astype(TRACK_DTYPES)
    return tt, np.array(fronts), np.array(rears)


# heading 90 = world +y = away from the camera; 270 = toward it; 0/180 = across the view
CASES = {
    "toward": ((3.0, 55.0), 270.0),
    "away": ((3.0, 20.0), 90.0),
    "across": ((-8.0, 30.0), 0.0),
    "diagonal": ((-6.0, 25.0), 45.0),
    "toward far": ((-2.0, 80.0), 270.0),
}


@pytest.mark.parametrize("case", CASES)
def test_front_and_rear_bumpers_are_within_half_a_metre(case: str) -> None:
    start, heading = CASES[case]
    scene = ground_scene()
    tt, true_front, true_rear = car_track(start, heading)

    kin = add_kinematics(tt, scene).sort_values("t")
    front = scene.to_world(kin[["front_x", "front_y"]].to_numpy(np.float64))
    rear = scene.to_world(kin[["rear_x", "rear_y"]].to_numpy(np.float64))

    assert np.abs(front - true_front).max() < 0.5, f"front error {np.abs(front - true_front).max():.2f} m"
    assert np.abs(rear - true_rear).max() < 0.5, f"rear error {np.abs(rear - true_rear).max():.2f} m"


def test_the_old_half_box_height_rule_was_metres_off_toward_the_camera() -> None:
    """Regression guard for SPEC §12.35: the footprint IS the front when driving toward the camera."""
    scene = ground_scene()
    tt, true_front, _ = car_track((3.0, 55.0), 270.0)
    footprint = scene.to_world(tt[["fx", "fy"]].to_numpy(np.float64))
    half_h = (tt["y2"] - tt["y1"]).to_numpy() / 2
    old_front = scene.to_world(tt[["fx", "fy"]].to_numpy(np.float64) + np.c_[np.zeros(len(tt)), half_h])
    assert np.abs(footprint - true_front).max() < 0.5
    assert np.abs(old_front - true_front).max() > 2.0


def test_corners_span_the_car_and_persons_are_points() -> None:
    scene = ground_scene()
    tt, true_front, true_rear = car_track((-8.0, 30.0), 0.0)
    person = tt.iloc[:3].assign(track_id=2, cls="person")
    kin = add_kinematics(pd.concat([tt, person]).astype(TRACK_DTYPES), scene)

    cars = kin[kin["cls"] == "car"].sort_values("t")
    world = scene.to_world(vehicle_corners(cars, scene).reshape(-1, 2)).reshape(-1, 4, 2)
    widths = np.linalg.norm(world[:, 0] - world[:, 1], axis=1)
    lengths = np.linalg.norm(world[:, 1] - world[:, 2], axis=1)
    np.testing.assert_allclose(widths, CAR_W, atol=0.05)
    np.testing.assert_allclose(lengths, CAR_L, atol=0.05)

    persons = kin[kin["cls"] == "person"]
    np.testing.assert_allclose(persons[["front_x", "front_y"]].to_numpy(), persons[["fx", "fy"]].to_numpy())
    np.testing.assert_allclose(persons[["rear_x", "rear_y"]].to_numpy(), persons[["fx", "fy"]].to_numpy())


def test_without_a_homography_front_and_rear_are_the_footprint() -> None:
    tt, _, _ = car_track((3.0, 55.0), 270.0)
    kin = add_kinematics(tt, Scene())
    np.testing.assert_allclose(kin[["front_x", "front_y"]].to_numpy(), kin[["fx", "fy"]].to_numpy())
