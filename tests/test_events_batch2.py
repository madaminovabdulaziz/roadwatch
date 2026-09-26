"""Batch 2 rules on synthetic tracks: red_light, stop_line, failure_to_yield, solid_line_crossing,
illegal_u_turn, illegal_turn (SPEC §5, §9).

Scene (10 px per metre): an east-west road y = 20-60 m (eastbound e1 40-50 m, e2 50-60 m) meeting a
north-south road at x = 100-140 m (the intersection). Eastbound stop line at x = 95 m (signal L1),
a crosswalk at x = 96-100 m before the intersection, exits east (E_out) and south (S_out); e1 may only
go straight, e2 straight or right. A no-U-turn zone at x = 200-260 m.
"""

from __future__ import annotations

import numpy as np
import pytest

from roadwatch.config import load_thresholds
from roadwatch.events import (
    failure_to_yield,
    illegal_turn,
    illegal_u_turn,
    red_light,
    solid_line_crossing,
    stop_line,
)
from roadwatch.scene.scene import Scene
from tests.test_events_batch1 import FPS, PX, braking_car, cfg, frames, rect, track

SCENE = Scene(
    {
        "image_size": [3000, 1000],
        "homography": {"image_pts": rect(0, 0, 10, 10), "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        "carriageway": rect(0, 0, 300, 100),
        "intersection": rect(100, 20, 140, 60),
        "crosswalks": [{"id": "cw", "polygon": rect(96, 20, 100, 60)}],
        "lanes": [
            {
                "id": "e1",
                "polygon": rect(0, 40, 100, 50),
                "direction": [1, 0],
                "signal": "L1",
                "allowed_exits": ["E_out"],
            },
            {
                "id": "e2",
                "polygon": rect(0, 50, 100, 60),
                "direction": [1, 0],
                "signal": "L1",
                "allowed_exits": ["E_out", "S_out"],
            },
            {"id": "far", "polygon": rect(140, 20, 300, 60), "direction": [1, 0]},
        ],
        "exits": [
            {"id": "E_out", "polygon": rect(140, 40, 160, 60)},
            {"id": "S_out", "polygon": rect(115, 64, 135, 80)},
        ],
        "stop_lines": [
            {"id": "sl_E", "line": [[950, 400], [950, 600]], "lanes": ["e1", "e2"], "signal": "L1"}
        ],
        "signals": [{"id": "L1", "red": [0, 0, 5, 5], "yellow": [0, 6, 5, 5], "green": [0, 12, 5, 5]}],
        "no_u_turn_zones": [rect(200, 20, 260, 60)],
    }
)
RED = {"L1": [(0.0, 30.0, "red")]}


def kin(*parts):
    import pandas as pd

    from roadwatch.features import add_kinematics
    from roadwatch.types import TRACK_DTYPES

    return add_kinematics(pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES), SCENE)


def ctx(duration: float, timeline: dict | None = None):
    from roadwatch.events.base import VideoContext
    from roadwatch.types import VideoMeta

    return VideoContext(VideoMeta("v.mp4", FPS, 3000, 1000, round(duration * FPS)), 3, timeline or {})


def near(seg, start: float, end: float, tol: float = 0.2) -> bool:
    return abs(seg.start - start) <= tol and abs(seg.end - end) <= tol


def straight(tid: int, x0: float, y: float, v: float, t1: float, cls: str = "car"):
    f = frames(0, t1)
    return track(tid, f, x0 + v * f / FPS, y, cls=cls)


# ---------------------------------------------------------------------------------------- red_light
# The ground-plane vehicle model (SPEC §12.35) puts a car's bumpers 2.3 m (half of 4.6 m) ahead of and
# behind its centre. In these top-down scenes the footprint is the box bottom-centre, the centre is
# 0.9 m (half the width) above it, and for eastbound cars front x = footprint x + 2.3, rear x = - 2.3.


def test_red_light_from_crossing_to_leaving_the_intersection() -> None:
    tt = kin(straight(1, 50, 45, 10, 12))
    segs = red_light.detect(tt, SCENE, ctx(12, RED), cfg("red_light"))
    assert len(segs) == 1
    # the front bumper (52.3 + 10 t) crosses x = 95 m at t = 4.27; the rear bumper (47.7 + 10 t)
    # clears the intersection (x = 140 m) at t = 9.23
    assert near(segs[0], 4.27, 9.23, tol=0.06)


def test_jumping_the_red_from_the_head_of_the_queue_is_red_light() -> None:
    # The most common violation here: the first car waits with its front bumper 0.2 m behind the line,
    # pulls away while the light is still red and drives through. It crosses the line slowly (< 1 m/s),
    # which must not be mistaken for "stopped past the line" (stop_line's case).
    half_len = load_thresholds()["kinematics"]["vehicle_dims_m"]["car"][0] / 2
    x_stop = 95.0 - half_len - 0.2  # footprint position while waiting (front bumper 0.2 m before x = 95)
    tt = kin(braking_car(1, x_stop, 45, t_brake=1, t_go=10, t_end=20))
    segs = red_light.detect(tt, SCENE, ctx(20, RED), cfg("red_light"))
    t_cross = 10 + 0.2**0.5  # front: 95 - 0.2 + (t - 10)^2 = 95
    t_clear = 10 + (140 - x_stop + half_len) ** 0.5  # rear: x_stop - half_len + (t - 10)^2 = 140
    assert len(segs) == 1 and near(segs[0], t_cross, t_clear, tol=0.1), (segs, t_cross, t_clear)


def test_crossing_then_stopping_before_the_intersection_is_not_red_light() -> None:
    # front 2 m past the line (x = 97), stops on the crosswalk, goes on at green: stop_line, not red_light
    tt = kin(braking_car(1, 94.7, 45, t_brake=2, t_go=25, t_end=30))
    assert red_light.detect(tt, SCENE, ctx(30, RED), cfg("red_light")) == []


def test_a_red_runner_whose_track_breaks_after_the_line_is_still_red_light() -> None:
    # C3905 0:33.5: a delivery rider jumps the red and the tracker loses (re-ids) it 1.1 s after the
    # line, on the zebra, still accelerating: it never "enters the intersection" under its first id.
    # Lost while moving is not stopped: the event ends when a vehicle would have cleared the junction.
    f = frames(0, 5.8)
    t = f / FPS
    wait = 95.0 - 2.1 / 2 - 0.2  # footprint while waiting: the 2.1 m motorcycle's front 0.2 m before the line
    rider = track(1, f, np.where(t < 4.0, wait, wait + 0.5 * 3.0 * (t - 4.0) ** 2), 45, cls="motorcycle")
    segs = red_light.detect(kin(rider), SCENE, ctx(20, RED), cfg("red_light"))
    assert len(segs) == 1
    assert segs[0].end - segs[0].start == pytest.approx(
        cfg("red_light")["params"]["lost_crossing_sec"], abs=0.2
    )


def test_a_short_red_glitch_is_not_a_red_phase() -> None:
    glitch = {"L1": [(0.0, 3.8, "green"), (3.8, 5.3, "red"), (5.3, 30.0, "green")]}  # a 1.5 s "red"
    assert red_light.detect(kin(straight(1, 50, 45, 10, 12)), SCENE, ctx(12, glitch), cfg("red_light")) == []


@pytest.mark.parametrize(
    "timeline",
    [{"L1": [(0.0, 30.0, "green")]}, {"L1": [(0.0, 4.2, "green"), (4.2, 30.0, "red")]}, None],
    ids=["green", "just-turned-red", "no-signal"],
)
def test_red_light_negatives(timeline) -> None:
    assert (
        red_light.detect(kin(straight(1, 50, 45, 10, 12)), SCENE, ctx(12, timeline), cfg("red_light")) == []
    )


def test_stopping_before_the_line_is_not_red_light() -> None:
    tt = kin(braking_car(1, 92, 45, t_brake=1, t_go=40, t_end=20))
    assert red_light.detect(tt, SCENE, ctx(20, RED), cfg("red_light")) == []


# ---------------------------------------------------------------------------------------- stop_line
def test_stop_line_from_stop_to_green() -> None:
    timeline = {"L1": [(0.0, 20.0, "red"), (20.0, 40.0, "green")]}
    tt = kin(braking_car(1, 95.5, 45, t_brake=2, t_go=25, t_end=30))  # front 2 m past the line
    segs = stop_line.detect(tt, SCENE, ctx(30, timeline), cfg("stop_line"))
    assert len(segs) == 1 and near(segs[0], 6.75, 20.0)


@pytest.mark.parametrize(("x_stop", "timeline"), [(92.0, RED), (95.5, {"L1": [(0.0, 30.0, "green")]})])
def test_stop_line_negatives(x_stop: float, timeline: dict) -> None:
    tt = kin(braking_car(1, x_stop, 45, t_brake=2, t_go=40, t_end=30))
    assert stop_line.detect(tt, SCENE, ctx(30, timeline), cfg("stop_line")) == []


# ---------------------------------------------------------------------------------------- failure_to_yield
def pedestrian(tid: int, x: float, y0: float, vy: float, t1: float):
    f = frames(0, t1)
    return track(tid, f, x, y0 + vy * f / FPS, cls="person", w_px=8, h_px=17)


def test_vehicle_through_occupied_crosswalk() -> None:
    # car x = 60 + 4 t: its front bumper reaches the crosswalk (x = 96) at t = 8.43, its rear bumper
    # clears it (x = 100) at t = 10.58; the pedestrian stands in its path (2.9 m from its axis)
    tt = kin(straight(1, 60, 45, 4, 15), pedestrian(2, 98, 47, 0, 15))
    segs = failure_to_yield.detect(tt, SCENE, ctx(15), cfg("failure_to_yield"))
    assert len(segs) == 1 and near(segs[0], 8.43, 10.58, tol=0.06) and segs[0].score == 1.0


def test_a_pedestrian_far_along_the_same_crosswalk_is_not_in_the_path() -> None:
    tt = kin(straight(1, 60, 45, 4, 15), pedestrian(2, 98, 22, 0, 15))  # 22 m from the car's axis
    assert failure_to_yield.detect(tt, SCENE, ctx(15), cfg("failure_to_yield")) == []


def test_pedestrian_stepping_onto_the_crosswalk_counts_with_lower_score() -> None:
    # car in lane e2 near the kerb (y = 59.5, axis 58.6); pedestrian beyond the kerb at y = 61, 2.4 m
    # from the axis, i.e. in its path (conflict_lateral_m 3 m since SPEC §12.49)
    tt = kin(straight(1, 60, 59.5, 4, 15), pedestrian(2, 98, 61.0, -0.05, 15))  # 1 m out, standing
    segs = failure_to_yield.detect(tt, SCENE, ctx(15), cfg("failure_to_yield"))
    assert len(segs) == 0  # 0.05 m/s is standing, not stepping onto it
    tt = kin(straight(1, 60, 59.5, 4, 15), pedestrian(2, 98, 65.8, -0.5, 15))  # 1 m out at 9.6 s, walking in
    segs = failure_to_yield.detect(tt, SCENE, ctx(15), cfg("failure_to_yield"))
    assert len(segs) == 1 and segs[0].score == 0.7


@pytest.mark.parametrize(
    "others",
    [(), ("far",), ("away",), ("slow",)],
    ids=["nobody", "pedestrian-10m-away", "walking-away", "vehicle-crawling"],
)
def test_failure_to_yield_negatives(others: tuple) -> None:
    parts = [straight(1, 60, 45, 1.0 if "slow" in others else 4, 15)]
    if "far" in others:
        parts.append(pedestrian(2, 110, 75, 0, 15))
    if "away" in others:
        parts.append(pedestrian(2, 98, 60.5, 1.0, 15))
    if "slow" in others:
        parts.append(pedestrian(2, 98, 56, 0, 15))
    assert failure_to_yield.detect(kin(*parts), SCENE, ctx(15), cfg("failure_to_yield")) == []


# ---------------------------------------------------------------------------------------- solid_line_crossing
LINE_SCENE = Scene(
    {
        "homography": {"image_pts": rect(0, 0, 10, 10), "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        "solid_lines": [{"id": "sol", "polyline": [[500, 0], [500, 1000]]}],  # x = 50 m, along the road
        "intersection": rect(0, 0, 1, 1),
    }
)


def lane_change(t_start: float, secs: float = 2.0):
    """A 4.6 x 1.8 m car driving up the image at 10 m/s, moving from x = 47 m to 53 m over `secs`.

    The rows are drawn from the true rectangle (top-down: box = its bounding box, footprint = box
    bottom-centre); also returns when its first and its last corner cross x = 50 m (the oracle).
    """
    f = frames(0, 8)
    t = f / FPS

    def path(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s_ = np.clip((tt - t_start) / secs, 0.0, 1.0)
        return 47 + 6 * (3 * s_**2 - 2 * s_**3), 90 - 10 * tt  # smoothstep lateral move

    def corners(tt: np.ndarray) -> np.ndarray:
        x, y = path(tt)
        dt = 1e-3
        x2, y2 = path(tt + dt)
        h = np.stack([x2 - x, y2 - y], axis=-1)
        h /= np.linalg.norm(h, axis=-1, keepdims=True)
        n = np.stack([-h[..., 1], h[..., 0]], axis=-1)
        c = np.stack([x, y], axis=-1)
        return np.stack([c + a * 2.3 * h + b * 0.9 * n for a in (-1, 1) for b in (-1, 1)], axis=-2)

    cs = corners(t)
    x1, x2 = cs[..., 0].min(axis=1), cs[..., 0].max(axis=1)
    y1, y2 = cs[..., 1].min(axis=1), cs[..., 1].max(axis=1)
    rows = track(1, f, (x1 + x2) / 2, y2)
    rows["x1"], rows["x2"], rows["y1"], rows["y2"] = x1 * PX, x2 * PX, y1 * PX, y2 * PX
    fine = np.arange(0, 8, 1e-3)
    fc = corners(fine)[..., 0]
    first, last = fine[np.argmax((fc > 50).any(axis=1))], fine[np.argmax((fc > 50).all(axis=1))]
    return rows, first, last


def line_kin(*parts):
    import pandas as pd

    from roadwatch.features import add_kinematics
    from roadwatch.types import TRACK_DTYPES

    return add_kinematics(pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES), LINE_SCENE)


def test_solid_line_crossing_from_first_to_last_wheel() -> None:
    rows, first, last = lane_change(2.0)
    segs = solid_line_crossing.detect(line_kin(rows), LINE_SCENE, ctx(8), cfg("solid_line_crossing"))
    assert len(segs) == 1 and near(segs[0], first, last, tol=0.15), (segs, first, last)


def test_touching_or_parallel_driving_is_not_a_crossing() -> None:
    f = frames(0, 8)
    t = f / FPS
    touch = track(
        1, f, 48.5 + 0.8 * np.sin(np.pi * t / 8), 90 - 10 * t, w_px=20, h_px=15
    )  # right wheel on the line
    parallel = track(2, f, 45, 90 - 10 * t, w_px=20, h_px=15)
    segs = solid_line_crossing.detect(
        line_kin(touch, parallel), LINE_SCENE, ctx(8), cfg("solid_line_crossing")
    )
    assert segs == []


# ---------------------------------------------------------------------------------------- illegal_u_turn
def u_turn_track(
    x_turn: float, t_turn: float = 3.0, radius: float = 6.0, v: float = 6.0, t_end: float = 12.0
):
    """East at v, a half circle (turning left, towards -y) starting at t_turn, then west."""
    f = frames(0, t_end)
    t = f / FPS
    omega = v / radius
    x0 = x_turn - v * t_turn
    phase = np.clip((t - t_turn) * omega, 0, np.pi)
    after = np.clip(t - t_turn - np.pi / omega, 0, None)
    x = np.where(t < t_turn, x0 + v * t, x_turn + radius * np.sin(phase) - v * after)
    y = np.where(t < t_turn, 50.0, 50.0 - radius * (1 - np.cos(phase)))
    return track(1, f, x, y), t_turn, t_turn + np.pi / omega


def test_u_turn_in_a_no_u_turn_zone() -> None:
    part, t0, t1 = u_turn_track(220)
    segs = illegal_u_turn.detect(kin(part), SCENE, ctx(12), cfg("illegal_u_turn"))
    assert len(segs) == 1
    # Savitzky-Golay smoothing (1.2 s window) lets the yaw rate rise ~0.3 s before the turn starts
    assert abs(segs[0].start - t0) <= 0.4 and abs(segs[0].end - t1) <= 0.4


def test_u_turn_outside_the_zone_only_counts_when_prohibited_everywhere() -> None:
    part, _, _ = u_turn_track(170)
    tt = kin(part)
    assert illegal_u_turn.detect(tt, SCENE, ctx(12), cfg("illegal_u_turn")) == []
    everywhere = Scene({**SCENE.layers, "u_turn_prohibited_everywhere": True})
    assert len(illegal_u_turn.detect(tt, everywhere, ctx(12), cfg("illegal_u_turn"))) == 1


def test_quarter_turn_is_not_a_u_turn() -> None:
    f = frames(0, 10)
    t = f / FPS
    phase = np.clip((t - 3) * 1.0, 0, np.pi / 2)
    x = np.where(t < 3, 212 + 6 * t, 230 + 6 * np.sin(phase))
    y = np.where(t < 3, 50.0, 50 - 6 * (1 - np.cos(phase)) - 6 * np.clip(t - 3 - np.pi / 2, 0, None))
    assert illegal_u_turn.detect(kin(track(1, f, x, y)), SCENE, ctx(10), cfg("illegal_u_turn")) == []


# ---------------------------------------------------------------------------------------- illegal_turn
def right_turn(tid: int, y: float, t_end: float = 14.0):
    """East along `y` at 8 m/s, a quarter circle right (towards +y) from x = 110 m, then south at x = 127."""
    f = frames(0, t_end)
    t = f / FPS
    v, r = 8.0, 62.0 - y + 0.0
    x_turn, t_turn = 110.0, (110.0 - 60.0) / 8.0
    omega = v / r
    phase = np.clip((t - t_turn) * omega, 0, np.pi / 2)
    after = np.clip(t - t_turn - (np.pi / 2) / omega, 0, None)
    x = np.where(t < t_turn, 60 + v * t, x_turn + r * np.sin(phase))
    yy = np.where(t < t_turn, y, 62 - r * np.cos(phase) + v * after)
    return track(tid, f, x, yy)


def test_right_turn_from_a_straight_only_lane() -> None:
    tt = kin(right_turn(1, 45))  # e1 allows E_out only
    segs = illegal_turn.detect(tt, SCENE, ctx(14), cfg("illegal_turn"))
    assert len(segs) == 1 and segs[0].meta == {"from": "e1", "to": "S_out"}
    # the turn starts at t = 6.25 (smoothing lets the yaw rate rise a little earlier); it reaches S_out
    # (y = 64 m) at 6.25 + 3.34 + 0.25 = 9.84 s with a steady heading
    assert abs(segs[0].start - 6.25) <= 0.4 and abs(segs[0].end - 9.84) <= 0.3


def test_allowed_movements_are_not_illegal() -> None:
    assert (
        illegal_turn.detect(kin(right_turn(1, 55)), SCENE, ctx(14), cfg("illegal_turn")) == []
    )  # e2 may turn
    assert illegal_turn.detect(kin(straight(2, 60, 45, 8, 14)), SCENE, ctx(14), cfg("illegal_turn")) == []


def test_turn_end_ignores_unknown_yaw_before_the_peak() -> None:
    # Slow samples have no heading, so their yaw rate is NaN; the peak must be the real one (30 deg/s),
    # not the NaN, or the turn would "end" at the 8 deg/s dip before the vehicle has even turned.
    from roadwatch.events.illegal_turn import _turn_end

    t = np.arange(6) * 0.1
    yaw = np.array([12.0, np.nan, 8.0, 30.0, 25.0, 5.0])
    heading = np.array([0.0, np.nan, 5.0, 40.0, 70.0, 90.0])
    end = _turn_end(t, yaw, heading, 0, 4, cfg("illegal_turn")["params"], gap=0.15)
    assert end == pytest.approx(0.45)  # midway between the last turning sample (0.4) and the calm one (0.5)
