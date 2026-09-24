"""Dev only: build weights/yolo11m.torchscript + .json from the official Ultralytics checkpoint.

Runs in the export venv (requirements-export.txt), because `ultralytics` must never share the runtime
venv (SPEC §12.12). Steps:
  1. download yolo11m.pt and check its sha256;
  2. fuse Conv+BN and trace the network *without* its box decoding: outputs are the raw DFL box logits
     (B, 64, A) and class logits (B, 80, A), so the file has no device/dtype/shape-specific constants;
  3. write the TorchScript, a metadata JSON and SHA256SUMS;
  4. parity: roadwatch's Detector (our decoding + NMS) must reproduce Ultralytics' own predictions on
     real frames (--parity-video). Results are stored in the metadata JSON.
The submission never runs this script: weights/download.sh fetches the published files.

Usage:
  uv venv --python 3.11 .venv-export
  uv pip install --python .venv-export/bin/python -r requirements-export.txt
  .venv-export/bin/python scripts/fetch_weights.py --parity-video samples/C3902.MP4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import WEIGHTS_DIR  # noqa: E402
from roadwatch.perception.detector import Detector  # noqa: E402

MODEL = "yolo11m"
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
SOURCE_SHA256 = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
TRACE_SHAPE = (1, 3, 544, 960)  # any multiple of 32 works; the traced graph is shape-generic
PARITY = {"imgsz": 960, "conf": 0.25, "iou": 0.7, "max_det": 300}
PARITY_MIN_IOU = 0.99
PARITY_MAX_CONF_DIFF = 1e-3


class RawYolo(torch.nn.Module):
    """A fused Ultralytics DetectionModel that stops before box decoding."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.layers, self.save = model.model, set(model.save)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        saved: list[torch.Tensor | None] = []
        for layer in self.layers[:-1]:  # same routing as Ultralytics' BaseModel._predict_once
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else saved[j] for j in layer.f]
                )
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        head = self.layers[-1]
        out = head.forward_head([x if j == -1 else saved[j] for j in head.f], **head.one2many)
        return out["boxes"], out["scores"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(dest: Path) -> None:
    if dest.exists() and sha256(dest) == SOURCE_SHA256:
        return
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, dest)
    if sha256(dest) != SOURCE_SHA256:
        raise SystemExit(f"{dest} sha256 mismatch; expected {SOURCE_SHA256}")


def export(source: Path, out: Path) -> dict:
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    det_model = YOLO(str(source)).model.float().fuse().eval()
    head = det_model.model[-1]
    raw = RawYolo(det_model).eval()
    with torch.inference_mode():
        example = torch.rand(*TRACE_SHAPE)
        eager = raw(example)
        traced = torch.jit.trace(raw, example)
        if not all(torch.equal(a, b) for a, b in zip(eager, traced(example), strict=True)):
            raise SystemExit("traced model differs from the eager model")
    traced.save(str(out))
    return {
        "model": MODEL,
        "format": "TorchScript; fused YOLO11 without box decoding (roadwatch.perception.detector decodes)",
        "input": "RGB float [0, 1], NCHW; H and W multiples of 32; any batch size",
        "outputs": ["box_dist (B, 4 * reg_max, A): DFL logits per side", "class_logits (B, nc, A)"],
        "strides": [int(s) for s in head.stride.tolist()],
        "reg_max": int(head.reg_max),
        "nc": int(head.nc),
        "names": {int(k): v for k, v in det_model.names.items()},
        "source": {"url": SOURCE_URL, "sha256": SOURCE_SHA256, "licence": "AGPL-3.0 (Ultralytics)"},
        "exported_with": {"ultralytics": ultralytics_version, "torch": torch.__version__},
    }


def parity_frames(video: Path, count: int, span_sec: float, size: tuple[int, int]) -> list[np.ndarray]:
    """`count` frames spread over the first `span_sec` seconds, resized like Part A's decode."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = min(total, int(fps * span_sec))
    frames = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * span // count)
        ok, img = cap.read()
        if ok:
            frames.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise SystemExit(f"could not read frames from {video}")
    return frames


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def check_parity(
    model_path: Path, meta: dict, source: Path, video: Path, count: int, span_sec: float
) -> dict:
    """Compare Ultralytics predictions with roadwatch's Detector on the same frames (CPU, FP32)."""
    from ultralytics import YOLO

    ours = Detector(
        torch.jit.load(str(model_path), map_location="cpu"),
        strides=meta["strides"],
        reg_max=meta["reg_max"],
        num_classes=meta["nc"],
        device="cpu",
        half=False,
        imgsz=PARITY["imgsz"],
        conf_min=PARITY["conf"],
        nms_iou=PARITY["iou"],
        max_det=PARITY["max_det"],
        batch_size=1,
        keep_classes=None,
        groups=None,
    )
    reference = YOLO(str(source))
    size = ours.frame_size_for(3840, 2160)
    rows, worst_iou, worst_conf = [], 1.0, 0.0
    for i, img in enumerate(parity_frames(video, count, span_sec, size)):
        ref = reference.predict(img, device="cpu", verbose=False, **PARITY)[0].boxes
        ref_xyxy, ref_conf, ref_cls = ref.xyxy.numpy(), ref.conf.numpy(), ref.cls.numpy().astype(int)
        mine = ours.predict([(i, 0.0, img)])[0]
        matched = 0
        if len(ref_xyxy) and len(mine.xyxy):
            iou = box_iou(ref_xyxy, mine.xyxy) * (ref_cls[:, None] == mine.cls[None, :])
            best = iou.argmax(1)
            for r, m in enumerate(best):
                worst_iou = min(worst_iou, float(iou[r, m]))
                worst_conf = max(worst_conf, abs(float(ref_conf[r] - mine.conf[m])))
                matched += bool(iou[r, m] >= PARITY_MIN_IOU)
        row = {"frame": i, "ultralytics": len(ref_xyxy), "roadwatch": len(mine.xyxy), "matched": matched}
        rows.append(row)
        print("parity", ", ".join(f"{k} {v}" for k, v in row.items()))
    ok = (
        all(r["ultralytics"] == r["roadwatch"] == r["matched"] for r in rows)
        and worst_iou >= PARITY_MIN_IOU
        and worst_conf <= PARITY_MAX_CONF_DIFF
    )
    return {
        "video": video.name,
        "settings": PARITY,
        "frames": rows,
        "min_iou": round(worst_iou, 6),
        "max_conf_diff": float(f"{worst_conf:.2e}"),
        "passed": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity-video", type=Path, help="real footage for the parity check (recommended)")
    ap.add_argument("--frames", type=int, default=6, help="frames used by the parity check")
    ap.add_argument(
        "--span-sec", type=float, default=60.0, help="parity frames come from the first N seconds"
    )
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    source = WEIGHTS_DIR / f"{MODEL}.pt"
    model_path = WEIGHTS_DIR / f"{MODEL}.torchscript"
    meta_path = model_path.with_suffix(".json")

    download_source(source)
    meta = export(source, model_path)
    meta["sha256"] = sha256(model_path)
    if args.parity_video:
        meta["parity"] = check_parity(model_path, meta, source, args.parity_video, args.frames, args.span_sec)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in (model_path, meta_path))
    (WEIGHTS_DIR / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(sums, end="")
    if args.parity_video and not meta["parity"]["passed"]:
        print("PARITY FAILED: roadwatch's decoding does not reproduce Ultralytics", file=sys.stderr)
        return 1
    print(f"wrote {model_path.name}, {meta_path.name}, SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
