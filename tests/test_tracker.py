"""OnlineTracker: stable ids, per-group trackers with globally unique ids, occlusion gaps, determinism."""

from __future__ import annotations

import numpy as np

from roadwatch.perception.tracker import OnlineTracker
from roadwatch.types import FrameDetections

FPS = 10.0  # 29.97 fps with stride 3
CAR, PERSON, MOTORCYCLE = 2, 0, 3


def dets(idx: int, boxes: list[tuple[float, float, float, float]], classes: list[int], conf: float = 0.9):
    return FrameDetections(
        frame_idx=idx,
        t=idx / FPS,
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        conf=np.full(len(boxes), conf, dtype=np.float32),
        cls=np.array(classes, dtype=np.int64),
    )


def car_at(idx: int) -> tuple[float, float, float, float]:
    x = 100.0 + 40.0 * idx  # 40 px per frame, a 200 x 120 px car
    return (x, 500.0, x + 200.0, 620.0)


def run(frames: list[FrameDetections]) -> list:
    tracker = OnlineTracker(FPS)
    return [tracker.update(f) for f in frames]


def test_a_moving_car_keeps_one_id_from_its_first_frame() -> None:
    out = run([dets(i, [car_at(i)], [CAR]) for i in range(15)])
    assert [len(f.track_id) for f in out] == [1] * 15
    assert {int(f.track_id[0]) for f in out} == {1}
    np.testing.assert_allclose(out[7].xyxy[0], car_at(7))  # the detector's box, not a smoothed one


def test_rider_and_motorcycle_are_tracked_separately_with_unique_ids() -> None:
    rider, bike = (500.0, 300.0, 560.0, 420.0), (495.0, 340.0, 565.0, 440.0)
    out = run([dets(i, [rider, bike], [PERSON, MOTORCYCLE]) for i in range(5)])
    last = out[-1]
    assert len(last.track_id) == 2
    assert len(set(last.track_id.tolist())) == 2
    assert sorted(last.cls.tolist()) == [PERSON, MOTORCYCLE]
    assert last.track_id.tolist() == sorted(last.track_id.tolist())


def test_a_short_occlusion_keeps_the_id() -> None:
    frames = [dets(i, [car_at(i)], [CAR]) for i in range(8)]
    frames += [dets(i, [], []) for i in range(8, 12)]  # 0.4 s hidden, inside the 1 s lost-track buffer
    frames += [dets(i, [car_at(i)], [CAR]) for i in range(12, 18)]
    out = run(frames)
    assert all(len(f.track_id) == 0 for f in out[8:12])
    assert {int(f.track_id[0]) for f in out if len(f.track_id)} == {1}


def test_low_confidence_boxes_continue_tracks_but_never_start_them() -> None:
    weak = 0.15  # between conf_min (0.1) and track_activation_threshold (0.25)
    started = run(
        [dets(0, [car_at(0)], [CAR])] + [dets(i, [car_at(i)], [CAR], conf=weak) for i in range(1, 6)]
    )
    never = run([dets(i, [car_at(i)], [CAR], conf=weak) for i in range(6)])
    assert all(len(f.track_id) == 1 for f in started)
    assert all(len(f.track_id) == 0 for f in never)


def test_identical_input_gives_identical_ids() -> None:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(20):
        jitter = rng.normal(0, 2, size=(3, 4))
        boxes = [
            tuple(np.add(car_at(i), jitter[0])),
            (900.0, 200.0, 950.0, 330.0),
            (1500.0, 800.0, 1700.0, 900.0),
        ]
        frames.append(dets(i, boxes, [CAR, PERSON, CAR]))
    first, second = run(frames), run(frames)
    assert [f.track_id.tolist() for f in first] == [f.track_id.tolist() for f in second]


def test_a_bus_entering_at_the_frame_edge_stays_in_the_output() -> None:
    # Replica of a bus entering C3902's right edge (720p, 10 Hz): the box grows leftwards while its right
    # side is pinned at the border. ByteTrack keeps the track, but its Kalman box lags the growing
    # width; supervision then matched tracks back to detections by IoU > 0.5 with that lagging box, so
    # the bus vanished from the output although the tracker had updated it with this very detection.
    BUS = 5
    boxes = [(1279.0 - 60.0 - 20.0 * i, 295.0 - 1.8 * i, 1279.3, 408.0) for i in range(15)]
    out = run([dets(i, [b], [BUS]) for i, b in enumerate(boxes)])
    assert [len(f.track_id) for f in out] == [1] * 15
    assert {int(f.track_id[0]) for f in out} == {1}
    for f, b in zip(out, boxes, strict=True):
        np.testing.assert_allclose(f.xyxy[0], b, atol=1e-3)
