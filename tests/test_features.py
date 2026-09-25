"""Kinematics on synthetic tracks whose true speed/acceleration are known (RUNBOOK P1.3, SPEC §9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roadwatch.config import load_thresholds
from roadwatch.features import add_kinematics, pair_features
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES

FPS = 30000 / 1001
STRIDE = 3
PX_PER_M = 10.0
BOX_H = 20.0
TOL = 1e-3  # footprints are float32 in the TrackTable, so ~1e-6 m of rounding, amplified by derivatives
SCENE = Scene(
    {
        "homography": {
            "image_pts": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        "carriageway": [[0, 0], [5000, 0], [5000, 5000], [0, 5000]],
        "lanes": [
            {"id": "n1", "polygon": [[0, 0], [5000, 0], [5000, 5000], [0, 5000]], "direction": [0, -1]}
        ],
    }
)


def frames(seconds: float, start: float = 0.0) -> np.ndarray:
    """Frame indices at the Part A stride covering [start, start + seconds]."""
    first = int(np.ceil(start * FPS / STRIDE)) * STRIDE
    return np.arange(first, int((start + seconds) * FPS) + 1, STRIDE)


def track(
    track_id: int, frame: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, cls: str = "car"
) -> pd.DataFrame:
    """Rows of one track whose footprint is at (x_m, y_m) metres (10 px per metre)."""
    fx, fy = np.asarray(x_m) * PX_PER_M, np.asarray(y_m) * PX_PER_M
    return pd.DataFrame(
        {
            "frame": frame,
            "t": frame / FPS,
            "track_id": track_id,
            "cls": cls,
            "conf": 0.9,
            "x1": fx - 10,
            "y1": fy - BOX_H,
            "x2": fx + 10,
            "y2": fy,
            "fx": fx,
            "fy": fy,
        }
    )


def table(*tracks: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(tracks, ignore_index=True).astype(TRACK_DTYPES)


def rows_of(kin: pd.DataFrame, track_id: int) -> pd.DataFrame:
    return kin[kin["track_id"] == track_id].reset_index(drop=True)


def test_constant_velocity_offline_is_exact() -> None:
    f = frames(5.0)
    t = f / FPS
    kin = rows_of(add_kinematics(table(track(1, f, 20 + 10 * t, 30 + 0 * t)), SCENE), 1)
    np.testing.assert_allclose(kin["speed"], 10.0, atol=TOL)
    np.testing.assert_allclose(kin["accel"], 0.0, atol=TOL)
    np.testing.assert_allclose(kin["heading_deg"], 0.0, atol=1e-6)
    np.testing.assert_allclose(kin["yaw_rate"], 0.0, atol=1e-6)
    np.testing.assert_allclose(kin["X"], 20 + 10 * t, atol=1e-6)
    assert kin["kin_valid"].all()


def test_braking_gives_signed_accel_and_linear_speed() -> None:
    f = frames(4.0)
    t = f / FPS
    kin = rows_of(add_kinematics(table(track(1, f, 50 - (15 * t - 1.5 * t**2), 20 + 0 * t)), SCENE), 1)
    np.testing.assert_allclose(kin["speed"], 15 - 3 * t, atol=TOL)
    np.testing.assert_allclose(kin["accel"], -3.0, atol=TOL)
    np.testing.assert_allclose(kin["heading_deg"].abs(), 180.0, atol=1e-6)  # driving towards -X


def test_gaps_from_lost_frames_keep_exact_speed() -> None:
    f = frames(5.0)
    keep = np.ones(len(f), dtype=bool)
    keep[[10, 11, 12, 25, 31]] = False
    t = f[keep] / FPS
    kin = rows_of(add_kinematics(table(track(1, f[keep], 0 * t + 5, 5 + 8 * t)), SCENE), 1)
    np.testing.assert_allclose(kin["speed"], 8.0, atol=TOL)
    np.testing.assert_allclose(kin["heading_deg"], 90.0, atol=1e-6)


def test_turn_gives_yaw_rate() -> None:
    f = frames(6.0)
    t = f / FPS
    omega = np.radians(20.0)
    kin = rows_of(
        add_kinematics(table(track(1, f, 100 + 20 * np.cos(omega * t), 100 + 20 * np.sin(omega * t))), SCENE),
        1,
    )
    inner = kin.iloc[10:-10]
    np.testing.assert_allclose(inner["yaw_rate"], 20.0, atol=0.5)
    np.testing.assert_allclose(inner["speed"], 20 * omega, rtol=0.01)


def test_stationary_track_holds_no_heading() -> None:
    f = frames(3.0)
    kin = rows_of(add_kinematics(table(track(1, f, 0 * f + 10.0, 0 * f + 10.0)), SCENE), 1)
    np.testing.assert_allclose(kin["speed"], 0.0, atol=1e-9)
    assert kin["heading_deg"].isna().all()
    assert (kin["yaw_rate"] == 0).all()


def test_short_track_is_not_kin_valid() -> None:
    f = frames(0.5)
    t = f / FPS
    kin = add_kinematics(table(track(1, f, 10 * t, 0 * t + 5)), SCENE)
    assert not kin["kin_valid"].any()


def test_no_homography_leaves_metric_columns_nan() -> None:
    f = frames(3.0)
    t = f / FPS
    kin = add_kinematics(table(track(1, f, 10 * t, 0 * t + 5)), Scene({"lanes": SCENE.layers["lanes"]}))
    assert not kin["kin_valid"].any()
    assert kin["speed"].isna().all() and kin["X"].isna().all()
    assert (kin["lane_id"] == "n1").all()
    assert (kin["obj_id"] == kin["track_id"]).all()


def test_online_is_causal_and_converges() -> None:
    f = frames(5.0)
    t = f / FPS
    base = table(track(1, f, 10 * t, 0 * t + 5))
    kin = rows_of(add_kinematics(base, SCENE, mode="online"), 1)
    assert kin["speed"].iloc[-10:].sub(10.0).abs().max() < 0.1

    changed = base.copy()
    changed.loc[changed.index[30:], "fx"] += 200.0  # rewrite the future
    kin2 = rows_of(add_kinematics(changed, SCENE, mode="online"), 1)
    cols = ["speed", "accel", "heading_deg", "yaw_rate"]
    pd.testing.assert_frame_equal(kin[cols].iloc[:30], kin2[cols].iloc[:30])


def test_zone_columns_and_vehicle_extent() -> None:
    # A car crossing the image sideways (+x) in a top-down view: the footprint is the middle of its
    # near long side, so the car's centre is half its width behind it (away from the camera, -y) and
    # the bumpers are half its length ahead and behind along the heading (SPEC §12.35).
    length, width = load_thresholds()["kinematics"]["vehicle_dims_m"]["car"]
    f = frames(2.0)
    t = f / FPS
    kin = add_kinematics(table(track(1, f, 10 + 5 * t, 0 * t + 20)), SCENE)
    assert kin["on_road"].all() and not kin["in_crosswalk"].any() and not kin["on_sidewalk"].any()
    np.testing.assert_allclose(kin["front_x"], kin["fx"] + length / 2 * PX_PER_M, atol=1e-3)
    np.testing.assert_allclose(kin["rear_x"], kin["fx"] - length / 2 * PX_PER_M, atol=1e-3)
    np.testing.assert_allclose(kin["front_y"], kin["fy"] - width / 2 * PX_PER_M, atol=1e-3)
    np.testing.assert_allclose(kin["rear_y"], kin["fy"] - width / 2 * PX_PER_M, atol=1e-3)


def test_position_persistence_links_stationary_id_switch() -> None:
    cfg = load_thresholds()["kinematics"]
    f1, f2 = frames(5.0), frames(4.0, start=5.0 + cfg["persistence_max_gap_sec"] / 2)
    far, moving = frames(4.0, start=6.0), frames(4.0, start=6.0)
    kin = add_kinematics(
        table(
            track(1, f1, 0 * f1 + 30.0, 0 * f1 + 30.0),
            track(2, f2, 0 * f2 + 30.5, 0 * f2 + 30.0),  # same parked car, new id after occlusion
            track(3, far, 0 * far + 60.0, 0 * far + 30.0),  # another parked car, 30 m away
            track(4, moving, 30.0 + 10 * (moving / FPS - 6), 0 * moving + 30.0),  # passes by, moving
        ),
        SCENE,
    )
    obj = kin.groupby("track_id")["obj_id"].first().to_dict()
    assert obj == {1: 1, 2: 1, 3: 3, 4: 4}


def test_output_sorted_by_time_then_track() -> None:
    f = frames(2.0)
    t = f / FPS
    kin = add_kinematics(table(track(2, f, 10 * t, 0 * t + 5), track(1, f, 10 * t, 0 * t + 9)), SCENE)
    assert kin[["t", "track_id"]].equals(kin[["t", "track_id"]].sort_values(["t", "track_id"]))


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_empty_table(mode: str) -> None:
    empty = np.array([], dtype=np.int64)
    kin = add_kinematics(table(track(1, empty, empty * 1.0, empty * 1.0)), SCENE, mode=mode)
    assert len(kin) == 0 and {"speed", "obj_id", "lane_id"} <= set(kin.columns)


def test_pair_features_head_on_ttc() -> None:
    rows = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4],
            "cls": ["car", "car", "person", "car"],
            "X": [0.0, 20.0, 0.0, 100.0],
            "Y": [0.0, 0.0, 5.0, 0.0],
            "vx": [5.0, -5.0, -1.0, -20.0],
            "vy": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pairs = pair_features(rows)
    head_on = pairs[(pairs["track_a"] == 1) & (pairs["track_b"] == 2)].iloc[0]
    assert head_on["closing_speed"] == pytest.approx(10.0)
    assert head_on["ttc"] == pytest.approx(2.0)
    assert not ((pairs["track_a"] == 4) | (pairs["track_b"] == 4)).any()  # 100 m away
    assert not ((pairs["track_a"] == 1) & (pairs["track_b"] == 3)).any()  # not converging
    assert pairs["ttc"].is_monotonic_increasing


def test_pair_features_keeps_only_real_conflicts() -> None:
    # Adjacent-lane overtaking closes fast but the paths stay 3.5 m apart; a follower braking to stop
    # behind a stopped car never reaches it; the same follower not braking does.
    rows = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5, 6],
            "cls": ["car"] * 6,
            "X": [0.0, 10.0, 100.0, 115.0, 200.0, 215.0],
            "Y": [0.0, 3.5, 0.0, 0.0, 0.0, 0.0],
            "vx": [15.0, 5.0, 10.0, 0.0, 10.0, 0.0],
            "vy": [0.0] * 6,
            "accel": [0.0, 0.0, -4.0, 0.0, 0.0, 0.0],
            "heading_deg": [0.0] * 6,
        }
    )
    pairs = pair_features(rows)
    keys = set(zip(pairs["track_a"], pairs["track_b"], strict=True))
    assert (1, 2) not in keys  # overtaking in the next lane: closest approach 3.5 m
    stopping = pairs[(pairs["track_a"] == 3) & (pairs["track_b"] == 4)].iloc[0]
    assert stopping["ttc"] == np.inf  # stops within 12.5 m of its 15 m gap: no contact predicted ...
    assert stopping["ttc_cv"] == pytest.approx(1.5)  # ... though it would be 1.5 s away without braking
    row = pairs[(pairs["track_a"] == 5) & (pairs["track_b"] == 6)].iloc[0]
    assert row["ttc"] == pytest.approx(1.5) and row["d_cpa"] == pytest.approx(0.0, abs=1e-9)
    assert pairs["ttc"].is_monotonic_increasing


def test_braking_aware_ttc_when_braking_is_not_enough() -> None:
    # 20 m/s toward a stopped car 20 m ahead, braking at 4 m/s^2: 20 t - 2 t^2 = 20 -> t = 1.127 s
    rows = pd.DataFrame(
        {
            "track_id": [1, 2],
            "cls": ["car", "car"],
            "X": [0.0, 20.0],
            "Y": [0.0, 0.0],
            "vx": [20.0, 0.0],
            "vy": [0.0, 0.0],
            "accel": [-4.0, 0.0],
            "heading_deg": [0.0, 0.0],
        }
    )
    assert pair_features(rows)["ttc"].iloc[0] == pytest.approx(5 - 15**0.5, rel=1e-6)


def test_pair_features_needs_two_rows() -> None:
    assert pair_features(
        pd.DataFrame({"track_id": [1], "cls": ["car"], "X": [0.0], "Y": [0.0], "vx": [1.0], "vy": [0.0]})
    ).empty
