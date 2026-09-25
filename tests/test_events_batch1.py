"""Batch 1 rules on synthetic tracks: stopped_vehicle, wrong_way, jaywalking, congestion (SPEC §5, §9).

Scene: 10 px per metre. A road from y = 20 m to 60 m; eastbound lanes e1 (40-50 m) and e2 (50-60 m),
westbound lanes w2 (30-40 m) and w1 (20-30 m); sidewalks on both edges; a crosswalk at x = 100-106 m;
an eastbound stop line at x = 95 m with signal L1; a parking bay at x = 150-170 m, y = 56-60 m.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from roadwatch.config import class_cfg
from roadwatch.events import congestion, jaywalking, stopped_vehicle, wrong_way
from roadwatch.events.base import VideoContext
from roadwatch.features import add_kinematics
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES, VideoMeta

FPS = 30000 / 1001
STRIDE = 3
PX = 10.0  # pixels per metre


def rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0 * PX, y0 * PX], [x1 * PX, y0 * PX], [x1 * PX, y1 * PX], [x0 * PX, y1 * PX]]


SCENE = Scene(
    {
        "image_size": [3000, 1000],
        "homography": {"image_pts": rect(0, 0, 10, 10), "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        "carriageway": rect(0, 20, 300, 60),
        "sidewalks": [rect(0, 15, 300, 20), rect(0, 60, 300, 65)],
        "crosswalks": [{"id": "cw", "polygon": rect(100, 20, 106, 60)}],
        "parking_zones": [rect(150, 56, 170, 60)],
        "lanes": [
            {"id": "e1", "polygon": rect(0, 40, 300, 50), "direction": [1, 0], "signal": "L1"},
            {"id": "e2", "polygon": rect(0, 50, 300, 60), "direction": [1, 0], "signal": "L1"},
            {"id": "w2", "polygon": rect(0, 30, 300, 40), "direction": [-1, 0]},
            {"id": "w1", "polygon": rect(0, 20, 300, 30), "direction": [-1, 0]},
        ],
        "directions": [{"id": "EB", "lanes": ["e1", "e2"]}, {"id": "WB", "lanes": ["w1", "w2"]}],
        "stop_lines": [
            {"id": "sl_E", "line": [[950, 400], [950, 600]], "lanes": ["e1", "e2"], "signal": "L1"}
        ],
    }
)


def frames(t0: float, t1: float) -> np.ndarray:
    return np.arange(int(np.ceil(t0 * FPS / STRIDE)) * STRIDE, int(t1 * FPS) + 1, STRIDE)


def track(
    tid: int, f: np.ndarray, x_m, y_m, cls: str = "car", w_px: float = 40, h_px: float = 30
) -> pd.DataFrame:
    fx = np.broadcast_to(np.asarray(x_m, dtype=float) * PX, f.shape)
    fy = np.broadcast_to(np.asarray(y_m, dtype=float) * PX, f.shape)
    return pd.DataFrame(
        {
            "frame": f,
            "t": f / FPS,
            "track_id": tid,
            "cls": cls,
            "conf": 0.9,
            "x1": fx - w_px / 2,
            "y1": fy - h_px,
            "x2": fx + w_px / 2,
            "y2": fy,
            "fx": fx,
            "fy": fy,
        }
    )


def kin(*parts: pd.DataFrame) -> pd.DataFrame:
    return add_kinematics(pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES), SCENE)


def ctx(duration: float, timeline: dict | None = None) -> VideoContext:
    return VideoContext(VideoMeta("v.mp4", FPS, 3000, 1000, round(duration * FPS)), STRIDE, timeline or {})


def cfg(label: str, **params: float) -> dict:
    c = copy.deepcopy(class_cfg(label))
    c["params"].update(params)
    return c


def spans(segments) -> list[tuple[float, float]]:
    return sorted((round(s.start, 2), round(s.end, 2)) for s in segments)


def near(segment, start: float, end: float, tol: float = 0.2) -> bool:
    return abs(segment.start - start) <= tol and abs(segment.end - end) <= tol


# ---------------------------------------------------------------------------------------- wrong_way
def test_wrong_way_starts_at_lane_entry_and_ends_at_last_sighting() -> None:
    f = frames(0, 9)
    t = f / FPS
    y = np.clip(25 + 20 * (t - 4), 25, 45)  # westbound in w1, swerves into e1 between t = 4 and 5
    tt = kin(track(1, f, 150 - 8 * t, y))
    segs = wrong_way.detect(tt, SCENE, ctx(10), cfg("wrong_way"))
    assert len(segs) == 1 and segs[0].label == "wrong_way"
    assert near(segs[0], 4.75, t[-1])  # entered e1 (y = 40 m) at t = 4.75
    assert segs[0].score > 0.9


# Real tracks have holes: ByteTrack writes no row for a frame where the detection was missed, and the
# pacer thins frames on a slow machine. A hole shorter than events.max_gap_sec must not split an event.
HOLES = {
    "every 4th detection missed": lambda n: np.arange(n) % 4 != 3,
    "every 2nd frame": lambda n: np.arange(n) % 2 == 0,
}


@pytest.mark.parametrize("keep", HOLES.values(), ids=HOLES.keys())
def test_wrong_way_survives_missed_detections(keep) -> None:
    f = frames(0, 9)
    t = f / FPS
    y = np.clip(25 + 20 * (t - 4), 25, 45)
    rows = track(1, f, 150 - 8 * t, y)
    segs = wrong_way.detect(kin(rows[keep(len(rows))]), SCENE, ctx(10), cfg("wrong_way"))
    assert len(segs) == 1 and near(segs[0], 4.75, t[-1], tol=0.35), segs


def test_wrong_way_negative_cases() -> None:
    f = frames(0, 8)
    t = f / FPS
    correct = track(1, f, 20 + 10 * t, 45)  # eastbound in an eastbound lane
    slow = track(2, f, 150 - 1.0 * t, 55)  # westbound but at walking pace
    brief = track(3, frames(0, 0.6), 150 - 8 * frames(0, 0.6) / FPS, 45)  # 0.6 s against the lane
    assert wrong_way.detect(kin(correct, slow, brief), SCENE, ctx(10), cfg("wrong_way")) == []


# ---------------------------------------------------------------------------------------- jaywalking
def test_jaywalking_from_stepping_on_to_leaving_the_road() -> None:
    f = frames(0, 25)
    t = f / FPS
    tt = kin(track(1, f, 50, 17 + 2 * t, cls="person", w_px=8, h_px=17))
    segs = jaywalking.detect(tt, SCENE, ctx(25), cfg("jaywalking"))
    assert len(segs) == 1 and near(segs[0], 1.5, 21.5)


@pytest.mark.parametrize("keep", HOLES.values(), ids=HOLES.keys())
def test_jaywalking_survives_missed_detections(keep) -> None:
    f = frames(0, 25)
    t = f / FPS
    rows = track(1, f, 50, 17 + 2 * t, cls="person", w_px=8, h_px=17)
    segs = jaywalking.detect(kin(rows[keep(len(rows))]), SCENE, ctx(25), cfg("jaywalking"))
    assert len(segs) == 1 and near(segs[0], 1.5, 21.5, tol=0.35), segs


def test_jaywalking_negative_cases() -> None:
    f = frames(0, 25)
    t = f / FPS
    on_crosswalk = track(1, f, 103, 17 + 2 * t, cls="person", w_px=8, h_px=17)
    rider = track(2, f, 20 + 8 * t, 45, cls="person", w_px=8, h_px=17)
    motorbike = track(3, f, 20 + 8 * t, 45.2, cls="motorcycle", w_px=20, h_px=15)
    in_bus = track(4, f, 60, 55, cls="person", w_px=6, h_px=10)
    bus = track(5, f, 60, 55.3, cls="bus", w_px=100, h_px=40)
    fb = frames(0, 3)
    step_out = track(6, fb, 70, 19.5 + 0.5 * np.sin(np.pi * fb / FPS / 3), cls="person", w_px=8, h_px=17)
    segs = jaywalking.detect(
        kin(on_crosswalk, rider, motorbike, in_bus, bus, step_out), SCENE, ctx(25), cfg("jaywalking")
    )
    assert segs == []


# ---------------------------------------------------------------------------------------- stopped_vehicle
def braking_car(tid: int, x_stop: float, y: float, t_brake: float, t_go: float, t_end: float) -> pd.DataFrame:
    """10 m/s, brakes at 2 m/s^2 from t_brake (stops 5 s later), waits, then pulls away at 2 m/s^2."""
    f = frames(0, t_end)
    t = f / FPS
    x_brake = x_stop - 25.0
    x = np.where(
        t < t_brake,
        x_brake - 10 * (t_brake - t),
        np.where(t < t_brake + 5, x_brake + 10 * (t - t_brake) - (t - t_brake) ** 2, x_stop),
    )
    x = np.where(t > t_go, x_stop + (t - t_go) ** 2, x)
    return track(tid, f, x, y)


def test_stopped_vehicle_start_and_end() -> None:
    tt = kin(braking_car(1, 145, 55, t_brake=2, t_go=25, t_end=30))
    segs = stopped_vehicle.detect(tt, SCENE, ctx(30), cfg("stopped_vehicle"))
    assert len(segs) == 1
    # below 0.5 m/s at 2 + 4.75 s; above 1.0 m/s from 25.5 s (and it stays above)
    assert near(segs[0], 6.75, 25.5)


def test_short_stop_is_not_an_event() -> None:
    tt = kin(braking_car(1, 145, 55, t_brake=2, t_go=15, t_end=20))  # stopped ~8 s
    assert stopped_vehicle.detect(tt, SCENE, ctx(20), cfg("stopped_vehicle")) == []


def test_vehicle_waiting_behind_a_stopped_vehicle_is_queued() -> None:
    first = braking_car(1, 145, 45, t_brake=2, t_go=40, t_end=40)
    second = braking_car(2, 137, 45, t_brake=2.5, t_go=40, t_end=40)  # stops 8 m behind the first
    segs = stopped_vehicle.detect(kin(first, second), SCENE, ctx(40), cfg("stopped_vehicle"))
    assert [s.track_ids for s in segs] == [(1,)]


@pytest.mark.parametrize(
    ("timeline", "expected"),
    [({"L1": [(0.0, 40.0, "red")]}, 0), ({"L1": [(0.0, 40.0, "green")]}, 1), (None, 0)],
)
def test_stop_line_queue_depends_on_the_signal(timeline, expected: int) -> None:
    car = braking_car(1, 85, 45, t_brake=2, t_go=40, t_end=40)  # waits 10 m before the stop line
    segs = stopped_vehicle.detect(kin(car), SCENE, ctx(40, timeline), cfg("stopped_vehicle"))
    assert len(segs) == expected


def test_parked_vehicle_is_not_an_event() -> None:
    tt = kin(braking_car(1, 160, 58, t_brake=2, t_go=40, t_end=40))
    assert stopped_vehicle.detect(tt, SCENE, ctx(40), cfg("stopped_vehicle")) == []


def test_stopped_hysteresis_ignores_creeping() -> None:
    t = np.arange(0, 20, 0.1)
    speed = np.where(t < 2, 5.0, 0.2)
    speed[(t > 8) & (t < 8.5)] = 1.5  # creeps forward for 0.5 s, less than resume_hold_sec
    stopped = stopped_vehicle.stopped_hysteresis(t, speed, class_cfg("stopped_vehicle")["params"])
    assert not stopped[t < 2].any() and stopped[t >= 2].all()


# ---------------------------------------------------------------------------------------- congestion
def jam(duration: float, n_per_lane: int = 2, moving: bool = False, lanes=(45, 55)) -> list[pd.DataFrame]:
    f = frames(0, duration)
    t = f / FPS
    out, tid = [], 1
    for y in lanes:
        for k in range(n_per_lane):
            x = 150 + 12 * k + (8 * t if moving else 0 * t)
            out.append(track(tid, f, x, y))
            tid += 1
    return out


def test_long_jam_is_congestion() -> None:
    segs = congestion.detect(kin(*jam(70)), SCENE, ctx(80), cfg("congestion"))
    assert len(segs) == 1 and segs[0].meta["direction"] == "EB"
    assert segs[0].start == 0.0 and abs(segs[0].end - 70) <= 1.0


def test_short_jam_needs_green_persistence() -> None:
    queue = kin(*jam(30))
    assert congestion.detect(queue, SCENE, ctx(40), cfg("congestion")) == []  # a 30 s red queue
    timeline = {"L1": [(0.0, 15.0, "red"), (15.0, 40.0, "green")]}
    segs = congestion.detect(queue, SCENE, ctx(40, timeline), cfg("congestion"))
    assert len(segs) == 1 and segs[0].start == 0.0 and abs(segs[0].end - 30) <= 1.0


def test_moving_or_partial_traffic_is_not_congestion() -> None:
    assert congestion.detect(kin(*jam(70, moving=True)), SCENE, ctx(80), cfg("congestion")) == []
    assert congestion.detect(kin(*jam(70, lanes=(45,))), SCENE, ctx(80), cfg("congestion")) == []
