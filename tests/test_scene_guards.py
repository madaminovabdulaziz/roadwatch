"""Scene-aware false-positive guards for batch-1 rules (SPEC §12.42).

Each negative case is a situation the review of 2026-09-25 reproduced as a false event on this junction
(people waiting at kerbs and islands, riders without a detected bike, cars waiting in the intersection
or at the bus stop, cars hugging the centre line); each positive case checks the guard is not so broad
that it hides real events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from roadwatch.events import jaywalking, stopped_vehicle, wrong_way
from roadwatch.features import add_kinematics
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES
from tests.test_events_batch1 import FPS, SCENE, braking_car, cfg, ctx, frames, rect, track

GUARD_SCENE = Scene(
    {
        **SCENE.layers,
        "intersection": rect(200, 20, 240, 60),
        "islands": [rect(120, 38, 130, 42)],  # a refuge on the centre line
        "bus_stops": [rect(170, 56, 190, 60)],
    }
)


def kin(*parts: pd.DataFrame) -> pd.DataFrame:
    return add_kinematics(pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES), GUARD_SCENE)


def person(tid: int, f: np.ndarray, x, y) -> pd.DataFrame:
    return track(tid, f, x, y, cls="person", w_px=8, h_px=17)


# ---------------------------------------------------------------------------------------- jaywalking
def test_waiting_at_the_kerb_is_not_jaywalking() -> None:
    f = frames(0, 12)
    at_kerb = person(1, f, 60, 20.3)  # feet 0.3 m past the kerb for 12 s (footprints are noisy)
    assert jaywalking.detect(kin(at_kerb), GUARD_SCENE, ctx(12), cfg("jaywalking")) == []


def test_walking_right_beside_the_zebra_is_not_jaywalking() -> None:
    f = frames(0, 25)
    t = f / FPS
    beside = person(1, f, 106.5, 17 + 2 * t)  # 0.5 m outside the painted stripes, all the way across
    assert jaywalking.detect(kin(beside), GUARD_SCENE, ctx(25), cfg("jaywalking")) == []


def test_standing_on_a_traffic_island_is_not_jaywalking() -> None:
    f = frames(0, 12)
    assert jaywalking.detect(kin(person(1, f, 125, 40)), GUARD_SCENE, ctx(12), cfg("jaywalking")) == []


def test_a_fast_rider_without_a_detected_bike_is_not_jaywalking() -> None:
    f = frames(0, 8)
    t = f / FPS
    rider = person(1, f, 20 + 8 * t, 45)  # 8 m/s along the lane: nobody walks that fast
    assert jaywalking.detect(kin(rider), GUARD_SCENE, ctx(8), cfg("jaywalking")) == []


def test_a_rider_whose_bike_is_detected_only_sometimes_is_not_jaywalking() -> None:
    f = frames(0, 8)
    t = f / FPS
    rider = person(1, f, 20 + 2.5 * t, 45)  # slow enough to pass the speed test
    bike = track(2, f, 20 + 2.5 * t, 45.2, cls="bicycle", w_px=20, h_px=15)
    bike = bike[np.arange(len(bike)) % 2 == 0]  # the bicycle box is missed in every other frame
    assert jaywalking.detect(kin(rider, bike), GUARD_SCENE, ctx(8), cfg("jaywalking")) == []


def test_running_straight_across_the_road_is_still_jaywalking() -> None:
    f = frames(0, 12)
    t = f / FPS
    runner = person(1, f, 50, 17 + 4.5 * t)  # 4.5 m/s straight across: running, not riding
    segs = jaywalking.detect(kin(runner), GUARD_SCENE, ctx(12), cfg("jaywalking"))
    assert len(segs) == 1 and abs(segs[0].start - 3 / 4.5) <= 0.15 and abs(segs[0].end - 43 / 4.5) <= 0.15


def test_crossing_mid_block_is_still_jaywalking() -> None:
    f = frames(0, 25)
    t = f / FPS
    segs = jaywalking.detect(kin(person(1, f, 50, 17 + 2 * t)), GUARD_SCENE, ctx(25), cfg("jaywalking"))
    assert len(segs) == 1  # steps well into the road, far from the zebra and the island


# ---------------------------------------------------------------------------------------- stopped_vehicle
def test_waiting_inside_the_intersection_is_not_a_stopped_vehicle() -> None:
    tt = kin(braking_car(1, 220, 45, t_brake=2, t_go=25, t_end=30))  # e.g. a left-turner yielding
    assert stopped_vehicle.detect(tt, GUARD_SCENE, ctx(30), cfg("stopped_vehicle")) == []


def test_a_bus_at_the_bus_stop_is_not_a_stopped_vehicle() -> None:
    bus = braking_car(1, 180, 58, t_brake=2, t_go=40, t_end=45)
    bus["cls"] = "bus"
    assert stopped_vehicle.detect(kin(bus), GUARD_SCENE, ctx(45), cfg("stopped_vehicle")) == []


def test_a_breakdown_in_a_lane_is_still_a_stopped_vehicle() -> None:
    tt = kin(braking_car(1, 145, 55, t_brake=2, t_go=25, t_end=30))
    assert len(stopped_vehicle.detect(tt, GUARD_SCENE, ctx(30), cfg("stopped_vehicle"))) == 1


# ---------------------------------------------------------------------------------------- wrong_way
def test_hugging_the_centre_line_is_not_wrong_way() -> None:
    f = frames(0, 8)
    t = f / FPS
    hugging = track(1, f, 20 + 10 * t, 39.7)  # eastbound, footprint 0.3 m into westbound w2
    assert wrong_way.detect(kin(hugging), GUARD_SCENE, ctx(8), cfg("wrong_way")) == []


def test_driving_in_the_oncoming_lane_is_still_wrong_way() -> None:
    f = frames(0, 8)
    t = f / FPS
    oncoming = track(1, f, 20 + 10 * t, 35)  # eastbound in the middle of westbound w2
    assert len(wrong_way.detect(kin(oncoming), GUARD_SCENE, ctx(8), cfg("wrong_way"))) == 1


# ---------------------------------------------------------------------------------------- scene layers
def test_islands_and_bus_stops_move_with_the_scene() -> None:
    shift = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    moved = GUARD_SCENE.transformed(shift, (3000, 1000))
    for layer in ("islands", "bus_stops"):
        before = np.asarray(GUARD_SCENE.layers[layer][0])
        np.testing.assert_allclose(np.asarray(moved.layers[layer][0]), before + [15.0, -7.0], atol=0.01)


# ------------------------------------------------------------------------------- stopped_vehicle: queues
def test_the_tail_of_a_long_red_queue_is_queued() -> None:
    # C3902's red phases: the queue reaches 60+ m back from the stop line (x = 95 m), and at that
    # distance the detector misses some of the cars in it, so nobody is found stopped just ahead.
    tail = braking_car(1, 35, 45, t_brake=2, t_go=40, t_end=40)  # 60 m upstream, lane e1 (signal L1)
    red = {"L1": [(0.0, 40.0, "red")]}
    assert stopped_vehicle.detect(kin(tail), GUARD_SCENE, ctx(40, red), cfg("stopped_vehicle")) == []
    green = {"L1": [(0.0, 40.0, "green")]}  # the same car stopped there with a green light is an event
    assert len(stopped_vehicle.detect(kin(tail), GUARD_SCENE, ctx(40, green), cfg("stopped_vehicle"))) == 1


def test_a_column_stuck_past_the_stop_line_behind_a_waiting_truck_is_queued() -> None:
    # C3905 1:17-1:54: cars that entered on green stand on the zebra past the stop line, outside any
    # lane, behind a long truck waiting at the exit. Each has a stopped vehicle just ahead of it.
    no_lanes_past_line = Scene(
        {
            **GUARD_SCENE.layers,
            "lanes": [
                {**lane, "polygon": rect(0, lane["polygon"][0][1] / 10, 95, lane["polygon"][2][1] / 10)}
                for lane in GUARD_SCENE.layers["lanes"]
            ],
        }
    )
    truck = braking_car(1, 112, 45, t_brake=2, t_go=40, t_end=40)
    truck["cls"] = "truck"
    car = braking_car(2, 102, 45, t_brake=2.5, t_go=40, t_end=40)  # 10 m behind the truck, past the line
    tt = add_kinematics(pd.concat([truck, car], ignore_index=True).astype(TRACK_DTYPES), no_lanes_past_line)
    waiting = (tt["track_id"] == 2) & (tt["t"] > 10)
    assert (tt.loc[waiting, "lane_id"] == "").all()  # stopped past the line: in no lane
    segs = stopped_vehicle.detect(tt, no_lanes_past_line, ctx(40), cfg("stopped_vehicle"))
    assert 2 not in {tid for s in segs for tid in s.track_ids}
