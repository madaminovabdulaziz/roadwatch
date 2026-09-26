"""Create or update the Hugging Face Space that hosts the live demo, in one command.

Uploads exactly what demo/Dockerfile needs (roadwatch/, configs/, demo/, the two weight files the
detector loads, solution.py, evaluate.py) with the Dockerfile at the Space's root and a README whose
front matter makes it a Docker Space on port 7860. The upload goes through the Hub API, which stores
the 81 MB TorchScript file as a large file by itself (a plain `git push` to a Space rejects files over
10 MB without Git LFS). The Space then builds the image and starts it; the build log is on its page.

Needs a Hugging Face token with write access (https://huggingface.co/settings/tokens) in HF_TOKEN, and
the weights in weights/ (bash weights/download.sh).

Usage: HF_TOKEN=... uv run --with huggingface_hub python demo/deploy_space.py --space <user>/roadwatch-demo
       python demo/deploy_space.py --stage /tmp/space   # the same files locally: docker build /tmp/space
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ("yolo11m.torchscript", "yolo11m.json", "SHA256SUMS")
SPACE_README = """---
title: RoadWatch demo
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Backend of the RoadWatch live demo (WIUT Hackathon 2026): upload a road-camera video, get the events,
the accident-risk curve and an annotated video back. API: `/api/health`, `/api/jobs`.
"""


def stage(dst: Path) -> None:
    """Copy the Space's files into `dst` (sources only: no caches, bytecode or videos)."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.mp4", "*.MP4", "*.bak", "*.tmp")
    for folder in ("roadwatch", "configs", "demo"):
        shutil.copytree(REPO_ROOT / folder, dst / folder, ignore=ignore)
    (dst / "weights").mkdir()
    for name in WEIGHTS:
        src = REPO_ROOT / "weights" / name
        if not src.exists():
            sys.exit(f"weights/{name} is missing: run bash weights/download.sh first")
        shutil.copy2(src, dst / "weights" / name)
    for name in ("solution.py", "evaluate.py"):
        shutil.copy2(REPO_ROOT / name, dst / name)
    shutil.copy2(REPO_ROOT / "demo" / "Dockerfile", dst / "Dockerfile")
    (dst / "README.md").write_text(SPACE_README, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="<user or org>/<space name>, e.g. me/roadwatch-demo")
    ap.add_argument("--stage", type=Path, help="only write the Space's files to this new folder")
    ap.add_argument("--private", action="store_true", help="create the Space private (judges cannot open it)")
    args = ap.parse_args()
    if args.stage:
        args.stage.mkdir(parents=True)
        stage(args.stage)
        print(f"staged in {args.stage}; check it with: docker build -t roadwatch-demo {args.stage}")
        return 0
    if not args.space:
        ap.error("--space is required (or --stage to only write the files)")
    token = os.environ.get("HF_TOKEN")
    if not token:
        ap.error("set HF_TOKEN to a Hugging Face token with write access")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.space,
            repo_type="space",
            commit_message="Deploy the RoadWatch demo backend",
            delete_patterns=["roadwatch/**", "configs/**", "demo/**"],  # drop files removed since last time
        )
    owner, name = args.space.split("/", 1)
    print(f"uploaded; the Space builds now: https://huggingface.co/spaces/{args.space}")
    print(f"API once it runs: https://{owner}-{name}.hf.space/api/health".replace("_", "-").lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
