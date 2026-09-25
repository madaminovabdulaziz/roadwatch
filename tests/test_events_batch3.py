"""Batch 3 rules on synthetic tracks: accident, near_miss, road_obstacle (SPEC §5, §9).

Uses the batch 1 scene (10 px per metre, road y = 20-60 m, eastbound lanes e1/e2, sidewalks).
"""

from __future__ import annotations

import numpy as np

from roadwatch.events import accident, near_miss, road_obstacle
from roadwatch.features import add_kinematics
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES
from tests.test_events_batch1 import FPS, SCENE, cfg, ctx, frames, kin, near, track


def stop_by(t: np.ndarray, x0: float, v: float, t_stop: float, decel: float) -> np.ndarray:
    """Position along a line: speed v until t_stop, then braking at `decel` to a standstill."""
    brake = np.clip(t - t_stop, 0, v / decel)
    return x0 + v * np.minimum(t, t_stop) + v * brake - 0.5 * decel * brake**2


# ---------------------------------------------------------------------------------------- accident
def t_bone(t_end: float = 20.0):
    """Car 1 eastbound at 8 m/s in e1; car 2 northbound at 10 m/s hits its side at x = 130 m, t = 6 s;
    both stop within about a second and stay."""
    f = frames(0, t_end)
    t = f / FPS
    x1 = stop_by(t, 82, 8, 6.0, 10)
    y2 = 110 - stop_by(t, 0, 10, 6.0, 12)  # reaches y = 50 m at t = 6
    a = track(1, f, x1, 45)
    b = track(2, f, 130, y2 - 3.8, w_px=20, h_px=30)
    return a, b


def test_side_collision_is_an_accident() -> None:
    a, b = t_bone()
    segs = accident.detect(kin(a, b), SCENE, ctx(20), cfg("accident"))
    assert len(segs) == 1 and segs[0].track_ids == (1, 2)
    assert "decel" in segs[0].meta["shocks"]
    assert abs(segs[0].start - 6.0) <= 0.3  # first contact, before stride-1 refinement
    assert 6.5 <= segs[0].end <= 8.0  # both at rest about a second after the impact


def test_pedestrian_knocked_down_is_an_accident() -> None:
    f = frames(0, 12)
    t = f / FPS
    car = track(1, f, 70 + 8 * t, 45)  # passes x = 110 m at t = 5 without braking
    walker = track(2, f, 110.5, 44.3, cls="person", w_px=8, h_px=17)
    lying = t >= 5.2  # the box turns from upright (17 x 8 px) to lying (8 x 17 px) after the hit
    walker.loc[lying, "x1"], walker.loc[lying, "x2"] = 1105 - 8.5, 1105 + 8.5
    walker.loc[lying, "y1"] = walker.loc[lying, "y2"] - 8
    segs = accident.detect(kin(car, walker), SCENE, ctx(12), cfg("accident"))
    assert len(segs) == 1 and segs[0].meta["shocks"] == ["fall"]


def test_occlusion_without_shock_is_not_an_accident() -> None:
    f = frames(0, 12)
    t = f / FPS
    near_lane = track(1, f, 60 + 8 * t, 45)  # overtaken in the image by a car in the next lane
    far_lane = track(2, f, 50 + 10 * t, 45.8)
    parked = [track(3, f, 200, 55), track(4, f, 201, 55.5)]  # touching boxes, nobody moves
    assert accident.detect(kin(near_lane, far_lane, *parked), SCENE, ctx(12), cfg("accident")) == []


def test_single_vehicle_crash_off_the_road() -> None:
    f = frames(0, 12)
    t = f / FPS
    x = stop_by(t, 50, 14, 4.0, 14)  # 14 m/s to 0 in 1 s, off the carriageway (y = 70 m)
    segs = accident.detect(kin(track(1, f, x, 70)), SCENE, ctx(12), cfg("accident"))
    assert len(segs) == 1 and segs[0].meta == {"single": True}
    assert abs(segs[0].start - 4.0) <= 0.4


def test_hard_stop_in_a_lane_is_not_a_single_vehicle_crash() -> None:
    f = frames(0, 12)
    t = f / FPS
    x = stop_by(t, 50, 14, 4.0, 14)
    assert accident.detect(kin(track(1, f, x, 45)), SCENE, ctx(12), cfg("accident")) == []


# ---------------------------------------------------------------------------------------- near_miss
def emergency_brake(decel: float, x_start: float = 60, t_brake: float = 1.92):
    f = frames(0, 10)
    t = f / FPS
    a = track(1, f, stop_by(t, x_start, 12, t_brake, decel), 45)
    b = track(2, f, 100 + 0 * t, 45)  # a car stopped ahead in the same lane
    return kin(a, b)


def test_hard_braking_behind_a_stopped_car_is_a_near_miss() -> None:
    segs = near_miss.detect(emergency_brake(6.0), SCENE, ctx(10), cfg("near_miss"))
    assert len(segs) == 1 and segs[0].track_ids == (1, 2)
    # braking starts at 1.92 s (smoothing makes it look a little earlier); stopped 5 m short at 3.92 s
    assert abs(segs[0].start - 1.92) <= 0.4 and abs(segs[0].end - 3.92) <= 0.4


def test_gentle_stop_or_contact_is_not_a_near_miss() -> None:
    gentle = emergency_brake(3.0, x_start=40, t_brake=0.5)  # plans the stop early, no TTC alarm
    assert near_miss.detect(gentle, SCENE, ctx(10), cfg("near_miss")) == []
    crash = emergency_brake(6.0, t_brake=2.6)  # brakes too late and reaches the other car
    assert near_miss.detect(crash, SCENE, ctx(10), cfg("near_miss")) == []


def test_a_firm_but_planned_stop_is_not_a_near_miss() -> None:
    # 12 m/s, braking at 4.5 m/s^2 from 1.7 s before reaching the stopped car, stopping 4 m short: firm,
    # but begun while the car ahead was > 1.5 s away. Its TTC falls below 1.5 s during the stop and the
    # deceleration passes 4 m/s^2, which used to be enough; imminence is now judged at the reaction.
    firm = emergency_brake(4.5, x_start=100 - 12 * 1.7 - 12 * 1.92, t_brake=1.92)
    assert near_miss.detect(firm, SCENE, ctx(10), cfg("near_miss")) == []


def test_turning_past_a_waiting_pedestrian_is_not_a_near_miss() -> None:
    # A car turns at 7 m/s on a 12 m radius (33 deg/s, above the 25 deg/s "evasive" yaw rate) past a
    # pedestrian standing 3 m outside its path. The radial TTC of this pass-by is 0.8 s, but the paths
    # never come within 3 m: no conflict (closest point of approach).
    f = frames(0, 8)
    t = f / FPS
    cx, cy, radius = 120.0, 55.0, 12.0
    ang = -np.pi / 2 + (7.0 / radius) * t  # starts heading east at (120, 43), turns north-east
    car = track(1, f, cx + radius * np.cos(ang), cy + radius * np.sin(ang))
    walker = track(2, f, cx + 15.0 * np.cos(0.4), cy + 15.0 * np.sin(0.4), cls="person", w_px=8, h_px=17)
    assert near_miss.detect(kin(car, walker), SCENE, ctx(8), cfg("near_miss")) == []


# ---------------------------------------------------------------------------------------- road_obstacle
def test_animal_on_the_road_with_a_short_occlusion() -> None:
    f1, f2 = frames(0, 3), frames(4, 8)  # hidden between 3 and 4 s (< gone_sec)
    dog = [
        track(1, f1, 120, 45, cls="dog", w_px=10, h_px=8),
        track(1, f2, 120, 45, cls="dog", w_px=10, h_px=8),
    ]
    segs = road_obstacle.detect(kin(*dog), SCENE, ctx(10), cfg("road_obstacle"))
    assert len(segs) == 1 and near(segs[0], 0.0, 8.0) and segs[0].score == 1.0


def test_carried_bags_brief_or_off_road_objects_are_not_obstacles() -> None:
    f = frames(0, 8)
    t = f / FPS
    walker = track(1, f, 50 + 1.4 * t, 45, cls="person", w_px=8, h_px=17)
    backpack = track(2, f, 50 + 1.4 * t, 44.5, cls="backpack", w_px=5, h_px=5)
    pavement_dog = track(3, f, 120, 17, cls="dog", w_px=10, h_px=8)
    fb = frames(0, 0.5)
    blip = track(4, fb, 150, 45, cls="suitcase", w_px=6, h_px=6)
    assert (
        road_obstacle.detect(kin(walker, backpack, pavement_dog, blip), SCENE, ctx(8), cfg("road_obstacle"))
        == []
    )


def leaving_through_the_bottom(t_end: float = 6.0):
    """A car driving down the image at 10 m/s (off the lanes) and out through the bottom edge (y = 100 m).

    The detector clips boxes to the frame, so once the car reaches the edge its footprint (box bottom)
    freezes at the border while the car is still moving: the only thing that looks like a stop. Near
    the camera a box includes the roof, so it stays clipped for about a second (here 10 m of box at
    10 m/s); from 0.6 s on, the unfixed kinematics reported a single-vehicle crash here.
    """
    f = frames(0, t_end)
    t = f / FPS
    y = 50 + 10 * t  # metres; the bottom edge (1000 px) is reached at t = 5
    rows = track(1, f, 250, y, w_px=40, h_px=100)
    rows["y2"] = np.minimum(rows["y2"], 1000.0)
    rows["fy"] = rows["y2"]
    return rows[rows["y1"] < 1000.0].reset_index(drop=True)  # gone once the whole box is out


def test_leaving_through_the_frame_edge_is_not_a_crash() -> None:
    scene = Scene({**SCENE.layers, "image_size": [3000, 1000]})
    tt = add_kinematics(leaving_through_the_bottom().astype(TRACK_DTYPES), scene)
    assert tt.loc[tt["y2"] >= 999.0, "at_edge"].all()
    assert not tt.loc[tt["at_edge"], "kin_valid"].any()
    assert accident.detect(tt, scene, ctx(6), cfg("accident")) == []
