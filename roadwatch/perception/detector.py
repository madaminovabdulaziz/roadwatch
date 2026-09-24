"""Object detector: YOLO11m with COCO weights (SPEC §3, §12.18-12.20).

Contract:
- `Detector.load()` returns the process-wide instance: weights are read once from weights/ by local
  path (never downloaded), seeds and deterministic flags are set, and one warm-up batch runs.
- `Detector.predict(batch, native_size)` takes `(frame_idx, t_sec, frame_bgr)` tuples from one video
  and returns one `FrameDetections` per frame: boxes in native pixels (`native_size`, default the
  frame's own size), only classes in `perception.keep_classes`, sorted by confidence.
- FP16 on CUDA, FP32 on CPU; input size, thresholds and batch size come from thresholds.yaml.

Weights: weights/<model>.torchscript is the fused YOLO11 network traced *without* its box decoding
(scripts/fetch_weights.py); weights/<model>.json holds its strides, DFL bins and class names. The traced
graph has no device-, dtype- or shape-specific constants, so one file runs on CPU or CUDA, in FP32 or
FP16, at any batch size and any input size that is a multiple of 32. Decoding (anchor grid, DFL,
xyxy) and NMS happen here in float32 and reproduce Ultralytics' predictions (parity check in
fetch_weights.py). The `ultralytics` package is never imported at inference (SPEC §12.12).

Pre-processing is Ultralytics' letterbox: long side resized to `imgsz`, centred padding to multiples
of 32 with grey 114, RGB, /255. Part A decodes frames straight to `frame_size_for(...)`, so they need
no resize; bigger frames (Part B receives the harness's 4K frames) are resized with INTER_AREA.

NMS runs per tracker group, not per class: a car also scored as a truck is one object, while a rider
and their motorcycle stay two objects (SPEC §12.20).
"""

from __future__ import annotations

import functools
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from roadwatch.config import WEIGHTS_DIR, get_device, load_thresholds, seed_everything
from roadwatch.types import Frame, FrameDetections

log = logging.getLogger(__name__)

PAD_VALUE = 114  # Ultralytics letterbox grey
_WARM_UP_FRAME = (1920, 1080)  # a 16:9 frame, the samples' aspect ratio


@dataclass(frozen=True)
class Letterbox:
    """Where a frame of a given size sits inside the network input."""

    new_w: int  # frame size after resizing (long side = imgsz)
    new_h: int
    in_w: int  # network input size (multiples of the largest stride)
    in_h: int
    pad_x: int  # left / top padding
    pad_y: int

    @classmethod
    def fit(cls, frame_w: int, frame_h: int, imgsz: int, stride: int) -> Letterbox:
        scale = imgsz / max(frame_w, frame_h)
        new_w, new_h = round(frame_w * scale), round(frame_h * scale)
        in_w, in_h = math.ceil(new_w / stride) * stride, math.ceil(new_h / stride) * stride
        return cls(new_w, new_h, in_w, in_h, (in_w - new_w) // 2, (in_h - new_h) // 2)


@functools.lru_cache(maxsize=16)
def _anchor_grid(
    in_h: int, in_w: int, strides: tuple[int, ...], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor centres (2, A) in grid units and their stride (1, A), level by level, row-major."""
    points, scales = [], []
    for stride in strides:
        h, w = in_h // stride, in_w // stride
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32) + 0.5,
            torch.arange(w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        points.append(torch.stack((xs, ys)).view(2, -1))
        scales.append(torch.full((1, h * w), float(stride)))
    return torch.cat(points, 1).to(device), torch.cat(scales, 1).to(device)


def decode_raw(
    box_dist: torch.Tensor, logits: torch.Tensor, in_h: int, in_w: int, strides: Sequence[int], reg_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw head outputs -> (xyxy in input pixels (B, 4, A), class probabilities (B, nc, A)), float32.

    `box_dist` is (B, 4 * reg_max, A): per side, logits over `reg_max` distance bins (DFL); the
    distance is the softmax-weighted mean bin, in units of the level's stride.
    """
    batch, _, n_anchors = box_dist.shape
    anchors, scale = _anchor_grid(in_h, in_w, tuple(int(s) for s in strides), str(box_dist.device))
    if anchors.shape[1] != n_anchors:
        raise ValueError(f"{n_anchors} anchors from the model, {anchors.shape[1]} expected for {in_w}x{in_h}")
    probs = box_dist.float().view(batch, 4, reg_max, n_anchors).softmax(2)
    bins = torch.arange(reg_max, dtype=torch.float32, device=box_dist.device).view(1, 1, reg_max, 1)
    ltrb = (probs * bins).sum(2)
    xyxy = torch.cat((anchors - ltrb[:, :2], anchors + ltrb[:, 2:]), 1) * scale
    return xyxy, logits.float().sigmoid()


class Detector:
    """Batched YOLO11 inference with group-aware NMS."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        strides: Sequence[int],
        reg_max: int,
        num_classes: int,
        device: str,
        half: bool,
        imgsz: int,
        conf_min: float,
        nms_iou: float,
        max_det: int,
        batch_size: int,
        keep_classes: dict[int, str] | None,
        groups: dict[str, list[str]] | None,
    ) -> None:
        """`keep_classes=None` keeps all classes; `groups=None` runs plain per-class NMS."""
        self.device = device
        self.dtype = torch.float16 if half and device.startswith("cuda") else torch.float32
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.strides = [int(s) for s in strides]
        self.reg_max = reg_max
        self.imgsz = imgsz
        self.conf_min = conf_min
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.batch_size = batch_size

        keep = keep_classes or {i: str(i) for i in range(num_classes)}
        self.keep_ids = torch.tensor(sorted(keep), dtype=torch.long, device=device)
        group_index = {name: g for g, members in enumerate((groups or {}).values()) for name in members}
        group_of = torch.full((num_classes,), -1, dtype=torch.long)
        for class_id, name in keep.items():
            # Classes outside every group (and all classes when groups is None) get their own NMS group.
            group_of[class_id] = group_index.get(name, len(group_index) + class_id)
        self.group_of = group_of.to(device)

    @classmethod
    def from_config(
        cls,
        *,
        weights_dir: Path = WEIGHTS_DIR,
        device: str | None = None,
        imgsz: int | None = None,
        half: bool | None = None,
    ) -> Detector:
        """Build from configs/thresholds.yaml and the exported weights; seeds and warms up.

        `device`, `imgsz` and `half` override the config (benchmarks and ablations).
        """
        cfg = load_thresholds()["perception"]
        model_path = weights_dir / f"{cfg['model']}.torchscript"
        meta_path = model_path.with_suffix(".json")
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"{model_path} or {meta_path.name} is missing: run `bash weights/download.sh` "
                "(or rebuild with scripts/fetch_weights.py)"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        device = device or get_device()
        seed_everything()
        model = torch.jit.load(str(model_path), map_location=device)
        detector = cls(
            model,
            strides=meta["strides"],
            reg_max=meta["reg_max"],
            num_classes=meta["nc"],
            device=device,
            half=cfg["half"] if half is None else half,
            imgsz=imgsz or cfg["imgsz"],
            conf_min=cfg["conf_min"],
            nms_iou=cfg["nms_iou"],
            max_det=cfg["max_det"],
            batch_size=cfg["batch_size"],
            keep_classes={int(k): v for k, v in cfg["keep_classes"].items()},
            groups=cfg["tracker_groups"],
        )
        detector.warm_up()
        log.info("detector %s on %s (%s), imgsz %d", cfg["model"], device, detector.dtype, detector.imgsz)
        return detector

    @classmethod
    def load(cls) -> Detector:
        """The shared instance used by Part A and Part B (built on first use)."""
        return _shared_detector()

    def frame_size_for(self, width: int, height: int) -> tuple[int, int]:
        """(width, height) to decode frames at so that `predict` needs no resize."""
        box = Letterbox.fit(width, height, self.imgsz, max(self.strides))
        return box.new_w, box.new_h

    def warm_up(self) -> None:
        """Run the batch sizes used later once, so the first real call is not slow."""
        width, height = self.frame_size_for(*_WARM_UP_FRAME)
        blank = np.zeros((height, width, 3), np.uint8)
        for size in sorted({1, self.batch_size}):
            self.predict([(0, 0.0, blank)] * size)

    @torch.inference_mode()
    def predict(
        self, batch: Sequence[Frame], native_size: tuple[int, int] | None = None
    ) -> list[FrameDetections]:
        out: list[FrameDetections] = []
        for start in range(0, len(batch), self.batch_size):
            out.extend(self._predict_chunk(batch[start : start + self.batch_size], native_size))
        return out

    def _predict_chunk(
        self, chunk: Sequence[Frame], native_size: tuple[int, int] | None
    ) -> list[FrameDetections]:
        frame_h, frame_w = chunk[0][2].shape[:2]
        if any(img.shape[:2] != (frame_h, frame_w) for _, _, img in chunk):
            raise ValueError("all frames in one predict() call must have the same size")
        box = Letterbox.fit(frame_w, frame_h, self.imgsz, max(self.strides))
        images = np.stack([self._resize(img, box) for _, _, img in chunk])

        x = torch.from_numpy(images).to(self.device)
        x = x.permute(0, 3, 1, 2).flip(1).to(self.dtype).div_(255)  # NHWC BGR uint8 -> NCHW RGB [0, 1]
        right, bottom = box.in_w - box.new_w - box.pad_x, box.in_h - box.new_h - box.pad_y
        x = F.pad(x, (box.pad_x, right, box.pad_y, bottom), value=PAD_VALUE / 255)
        box_dist, logits = self.model(x)
        xyxy, probs = decode_raw(box_dist, logits, box.in_h, box.in_w, self.strides, self.reg_max)

        native_w, native_h = native_size or (frame_w, frame_h)
        to_native = torch.tensor(
            [native_w / box.new_w, native_h / box.new_h] * 2, dtype=torch.float32, device=self.device
        )
        offset = torch.tensor([box.pad_x, box.pad_y] * 2, dtype=torch.float32, device=self.device)
        limits = torch.tensor([native_w, native_h] * 2, dtype=torch.float32, device=self.device)
        return [
            self._select(xyxy[i], probs[i], offset, to_native, limits, idx, t_sec)
            for i, (idx, t_sec, _) in enumerate(chunk)
        ]

    @staticmethod
    def _resize(img: np.ndarray, box: Letterbox) -> np.ndarray:
        if img.shape[:2] == (box.new_h, box.new_w):
            return img
        shrink = box.new_w < img.shape[1]
        return cv2.resize(
            img, (box.new_w, box.new_h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
        )

    def _select(
        self,
        xyxy: torch.Tensor,
        probs: torch.Tensor,
        offset: torch.Tensor,
        to_native: torch.Tensor,
        limits: torch.Tensor,
        frame_idx: int,
        t_sec: float,
    ) -> FrameDetections:
        """One image: best kept class per anchor, confidence floor, group-aware NMS, native pixels."""
        conf, best = probs[self.keep_ids].max(0)
        mask = conf >= self.conf_min
        boxes, conf, cls = xyxy[:, mask].T, conf[mask], self.keep_ids[best[mask]]
        keep = torchvision.ops.batched_nms(boxes, conf, self.group_of[cls], self.nms_iou)[: self.max_det]
        boxes = ((boxes[keep] - offset) * to_native).clamp_(min=torch.zeros_like(limits), max=limits)
        return FrameDetections(
            frame_idx=frame_idx,
            t=t_sec,
            xyxy=boxes.cpu().numpy().astype(np.float32),
            conf=conf[keep].cpu().numpy().astype(np.float32),
            cls=cls[keep].cpu().numpy().astype(np.int64),
        )


@functools.cache
def _shared_detector() -> Detector:
    return Detector.from_config()


def detector_info(detector: Detector) -> dict[str, Any]:
    """Summary for logs and benchmarks."""
    return {
        "device": detector.device,
        "dtype": str(detector.dtype).replace("torch.", ""),
        "imgsz": detector.imgsz,
        "batch_size": detector.batch_size,
        "conf_min": detector.conf_min,
        "nms_iou": detector.nms_iou,
    }
