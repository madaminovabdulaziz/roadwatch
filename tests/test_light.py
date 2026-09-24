"""Signal state from synthetic lamp frames, and the flow fallback (RUNBOOK P1.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roadwatch.features import add_kinematics
from roadwatch.scene.light import UNKNOWN, SignalStateEstimator, flow_timeline, state_at
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES

FPS = 10.0
# (start, end, lit lamp) of one signal cycle; the scene is calibrated at 2x the frame size
CYCLE = [(0.0, 5.0, "red"), (5.0, 10.0, "green"), (10.0, 11.0, "yellow"), (11.0, 15.0, "red")]
BOXES = {"red": [20, 20, 20, 20], "yellow": [20, 50, 20, 20], "green": [20, 80, 20, 20]}  # native px
LIT_BGR = {"red": (40, 40, 250), "yellow": (40, 220, 250), "green": (120, 230, 40)}
SCENE = Scene({"image_size": [200, 240], "signals": [{"id": "L1", **BOXES}]})


def lit_at(t: float) -> str:
    t = t % CYCLE[-1][1]
    return next(lamp for t0, t1, lamp in CYCLE if t0 <= t < t1)


def frame(lit: str | None) -> np.ndarray:
    img = np.full((120, 100, 3), 30, np.uint8)  # half of the native 200x240
    for lamp, (x, y, w, h) in BOXES.items():
        colour = LIT_BGR[lamp] if lamp == lit else (45, 45, 45)
        img[y // 2 : (y + h) // 2, x // 2 : (x + w) // 2] = colour
    return img


def frames(seconds: float, glitch_at: float | None = None):
    for i in range(int(seconds * FPS)):
        t = i / FPS
        lit = None if glitch_at is not None and abs(t - glitch_at) < 1e-9 else lit_at(t)
        yield i, t, frame(lit)


def test_offline_timeline_follows_the_cycle() -> None:
    segs = SignalStateEstimator(SCENE).timeline(frames(15.0))["L1"]
    assert [s for _, _, s in segs] == ["red", "green", "yellow", "red"]
    for (t0, _, _), (true_t0, _, _) in zip(segs[1:], CYCLE[1:], strict=True):
        assert abs(t0 - true_t0) <= 0.3
    assert segs[0][0] == 0.0 and segs[-1][1] == 14.9


def test_single_frame_glitch_is_filtered() -> None:
    segs = SignalStateEstimator(SCENE).timeline(frames(15.0, glitch_at=2.0))["L1"]
    assert [s for _, _, s in segs] == ["red", "green", "yellow", "red"]


def test_online_is_unknown_until_calibrated_then_correct() -> None:
    est = SignalStateEstimator(SCENE)
    states = [(t, est.update(img, t)["L1"]) for _, t, img in frames(30.0)]
    assert states[0][1] == UNKNOWN  # nothing seen yet: no lamp has shown on and off
    second_cycle = [(t, s) for t, s in states if t >= 15.0]
    near_change = lambda t: min(abs((t % 15.0) - c[0]) for c in CYCLE) < 0.6  # noqa: E731
    wrong = [(t, s) for t, s in second_cycle if not near_change(t) and s != lit_at(t)]
    assert wrong == []


def test_online_depends_only_on_the_past() -> None:
    a, b = SignalStateEstimator(SCENE), SignalStateEstimator(SCENE)
    seq = list(frames(20.0))
    out_a = [a.update(img, t) for _, t, img in seq]
    out_b = [b.update(img, t) for _, t, img in seq[:120]]
    assert out_a[:120] == out_b


def test_missing_lamp_box_and_no_signals() -> None:
    partial = Scene(
        {"image_size": [200, 240], "signals": [{"id": "L2", "red": BOXES["red"], "green": BOXES["green"]}]}
    )
    segs = SignalStateEstimator(partial).timeline(frames(15.0))["L2"]
    assert {s for _, _, s in segs} <= {"red", "green", UNKNOWN}
    assert SignalStateEstimator(Scene()).timeline(frames(1.0)) == {}
    assert SignalStateEstimator(SCENE).timeline(iter(())) == {"L1": []}


def test_state_at() -> None:
    segs = [(0.0, 5.0, "red"), (5.0, 9.0, "green")]
    assert state_at(segs, 0.0) == "red"
    assert state_at(segs, 4.99) == "red"
    assert state_at(segs, 5.0) == "green"
    assert state_at(segs, 9.5) == UNKNOWN
    assert state_at(segs, -1.0) == UNKNOWN
    assert state_at([], 1.0) == UNKNOWN


def _track(track_id: int, t: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> pd.DataFrame:
    fx, fy = x_m * 10, y_m * 10
    return pd.DataFrame(
        {
            "frame": np.round(t * 30).astype(int),
            "t": t,
            "track_id": track_id,
            "cls": "car",
            "conf": 0.9,
            "x1": fx - 10,
            "y1": fy - 1,
            "x2": fx + 10,
            "y2": fy,
            "fx": fx,
            "fy": fy,
        }
    )


def test_flow_fallback_green_when_crossing_red_when_waiting() -> None:
    # 10 px per metre. Signal A: eastbound lane, stop line at x = 50 m. Signal B: northbound lane
    # (y decreasing), stop line at y = 50 m. A's car crosses at ~t = 5 s while B's car waits.
    scene = Scene(
        {
            "homography": {
                "image_pts": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
            "lanes": [
                {"id": "a", "polygon": [[0, 300], [1000, 300], [1000, 340], [0, 340]], "direction": [1, 0]},
                {"id": "b", "polygon": [[700, 0], [740, 0], [740, 1000], [700, 1000]], "direction": [0, -1]},
            ],
            "stop_lines": [
                {"id": "sa", "line": [[500, 300], [500, 340]], "lanes": ["a"], "signal": "A"},
                {"id": "sb", "line": [[700, 500], [740, 500]], "lanes": ["b"], "signal": "B"},
            ],
        }
    )
    t = np.arange(0, 10, 0.1)
    tt = pd.concat(
        [_track(1, t, 25 + 5 * t, 0 * t + 32), _track(2, t, 0 * t + 72, 0 * t + 55)], ignore_index=True
    ).astype(TRACK_DTYPES)
    timeline = flow_timeline(add_kinematics(tt, scene), scene)
    assert state_at(timeline["A"], 5.0) == "green"
    assert state_at(timeline["B"], 5.0) == "red"
    assert state_at(timeline["A"], 1.0) == UNKNOWN
