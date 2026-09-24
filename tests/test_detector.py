"""Detector: letterbox geometry, DFL decoding, coordinate mapping, class filter, group NMS, batching.

A stand-in network plants detections at chosen anchors, so every expected box is known exactly and
no weights are needed. The real network is covered by the parity check in scripts/fetch_weights.py
and by tests/test_offline.py when weights are present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roadwatch.config import load_thresholds
from roadwatch.perception.detector import Detector, Letterbox

STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_CLASSES = 80
CAR, MOTORCYCLE, TRUCK, PERSON, CHAIR = 2, 3, 7, 0, 56
CERTAIN = 10.0  # class logit -> probability 0.99995


class PlantedHead(torch.nn.Module):
    """Raw-head outputs with detections planted at (level, gx, gy): class logit and DFL distances."""

    def __init__(self, plants: list[tuple[int, int, int, int, float, tuple[int, int, int, int]]]) -> None:
        super().__init__()
        self.plants = plants

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        grids = [(height // s, width // s) for s in STRIDES]
        offsets = np.cumsum([0] + [h * w for h, w in grids])
        box = torch.zeros(batch, 4 * REG_MAX, int(offsets[-1]))
        cls = torch.full((batch, NUM_CLASSES, int(offsets[-1])), -20.0)
        for level, gx, gy, class_id, logit, ltrb in self.plants:
            anchor = int(offsets[level]) + gy * grids[level][1] + gx
            for side, bins in enumerate(ltrb):
                box[:, side * REG_MAX + bins, anchor] = 50.0  # one-hot DFL: distance == bins exactly
            cls[:, class_id, anchor] = logit
        return box.to(x.dtype), cls.to(x.dtype)


def make_detector(plants: list, **overrides: object) -> Detector:
    cfg = load_thresholds()["perception"]
    options = {
        "strides": STRIDES,
        "reg_max": REG_MAX,
        "num_classes": NUM_CLASSES,
        "device": "cpu",
        "half": False,
        "imgsz": 960,
        "conf_min": cfg["conf_min"],
        "nms_iou": cfg["nms_iou"],
        "max_det": cfg["max_det"],
        "batch_size": 4,
        "keep_classes": {int(k): v for k, v in cfg["keep_classes"].items()},
        "groups": cfg["tracker_groups"],
    }
    return Detector(PlantedHead(plants), **{**options, **overrides})


def frame(width: int = 1920, height: int = 1080, idx: int = 0, fps: float = 30.0) -> tuple:
    return idx, idx / fps, np.zeros((height, width, 3), np.uint8)


def expected_native_box(gx: int, gy: int, ltrb: tuple, stride: int, pad: tuple, scale: float) -> np.ndarray:
    """Planted box in input pixels -> minus letterbox padding -> native pixels."""
    x1, y1 = (gx + 0.5 - ltrb[0]) * stride, (gy + 0.5 - ltrb[1]) * stride
    x2, y2 = (gx + 0.5 + ltrb[2]) * stride, (gy + 0.5 + ltrb[3]) * stride
    return (np.array([x1, y1, x2, y2]) - np.array([pad[0], pad[1]] * 2)) * scale


@pytest.mark.parametrize(
    ("size", "imgsz", "expected"),
    [
        ((3840, 2160), 960, Letterbox(960, 540, 960, 544, 0, 2)),
        ((1920, 1080), 1280, Letterbox(1280, 720, 1280, 736, 0, 8)),
        ((1080, 1920), 960, Letterbox(540, 960, 544, 960, 2, 0)),
        ((1000, 999), 640, Letterbox(640, 639, 640, 640, 0, 0)),
    ],
)
def test_letterbox_matches_ultralytics_geometry(size: tuple, imgsz: int, expected: Letterbox) -> None:
    assert Letterbox.fit(*size, imgsz, 32) == expected


def test_frame_size_for_is_the_resized_frame() -> None:
    assert make_detector([]).frame_size_for(3840, 2160) == (960, 540)


def test_planted_box_decodes_to_exact_native_coordinates() -> None:
    ltrb = (2, 3, 4, 5)
    detector = make_detector([(0, 50, 30, CAR, CERTAIN, ltrb)])

    out = detector.predict([frame()], native_size=(3840, 2160))[0]

    assert out.cls.tolist() == [CAR]
    assert out.conf[0] == pytest.approx(torch.sigmoid(torch.tensor(CERTAIN)).item(), abs=1e-6)
    expected = expected_native_box(50, 30, ltrb, stride=8, pad=(0, 2), scale=4.0)
    np.testing.assert_allclose(out.xyxy[0], expected, atol=1e-3)
    assert out.xyxy.dtype == np.float32 and out.conf.dtype == np.float32 and out.cls.dtype == np.int64


def test_4k_frames_are_resized_and_give_the_same_native_boxes() -> None:
    detector = make_detector([(1, 20, 10, PERSON, CERTAIN, (1, 2, 1, 2))])
    from_1080p = detector.predict([frame(1920, 1080)], native_size=(3840, 2160))[0]
    from_4k = detector.predict([frame(3840, 2160)])[0]  # Part B: native size is the frame's own
    np.testing.assert_allclose(from_4k.xyxy, from_1080p.xyxy, atol=1e-3)


def test_classes_outside_keep_classes_are_dropped() -> None:
    detector = make_detector([(0, 50, 30, CHAIR, CERTAIN, (2, 2, 2, 2))])
    assert len(detector.predict([frame()])[0].cls) == 0


def test_confidence_floor() -> None:
    weak = float(torch.logit(torch.tensor(0.05)))  # below perception.conf_min
    detector = make_detector([(0, 50, 30, CAR, weak, (2, 2, 2, 2))])
    assert len(detector.predict([frame()])[0].cls) == 0


def test_nms_merges_duplicates_within_a_group_but_not_across_groups() -> None:
    big = (6, 6, 6, 6)  # 96 px boxes: neighbouring anchors overlap with IoU ~0.85
    car_and_truck = make_detector([(0, 50, 30, CAR, CERTAIN, big), (0, 51, 30, TRUCK, 8.0, big)])
    rider = make_detector([(0, 50, 30, PERSON, CERTAIN, big), (0, 51, 30, MOTORCYCLE, 8.0, big)])

    assert car_and_truck.predict([frame()])[0].cls.tolist() == [CAR]
    assert sorted(rider.predict([frame()])[0].cls.tolist()) == [PERSON, MOTORCYCLE]


def test_batches_larger_than_batch_size_keep_order_and_timestamps() -> None:
    detector = make_detector([(2, 5, 5, CAR, CERTAIN, (1, 1, 1, 1))], batch_size=4)
    batch = [frame(idx=i) for i in range(10)]
    out = detector.predict(batch)
    assert [(d.frame_idx, d.t) for d in out] == [(i, t) for i, t, _ in batch]
    assert all(d.cls.tolist() == [CAR] for d in out)


def test_mixed_frame_sizes_in_one_call_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_detector([]).predict([frame(1920, 1080), frame(1280, 720)])


def test_boxes_are_clipped_to_the_frame() -> None:
    detector = make_detector([(0, 0, 0, CAR, CERTAIN, (5, 5, 1, 1))])  # extends past the top-left corner
    box = detector.predict([frame()], native_size=(3840, 2160))[0].xyxy[0]
    assert box[0] == 0.0 and box[1] == 0.0
