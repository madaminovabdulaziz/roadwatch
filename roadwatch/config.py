"""Configuration, repository paths, seeding and device selection.

Contract:
- `load_thresholds()` returns configs/thresholds.yaml parsed once per process (treat it as read-only).
- `class_cfg(label)` returns one class block; `enabled_classes()` lists labels with `enabled: true`.
- `seed_everything()` fixes random/numpy/torch seeds and the deterministic CUDA flags (CLAUDE.md rule 6).
- `get_device()` returns "cuda" when available, else "cpu"; the ROADWATCH_DEVICE env var overrides it.
All paths resolve from the repository root, never from the current working directory.
"""

from __future__ import annotations

import functools
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
WEIGHTS_DIR = REPO_ROOT / "weights"
CACHE_DIR = REPO_ROOT / "cache"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"
SCENE_PATH = CONFIG_DIR / "scene.json"


@functools.cache
def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    """Parse the tunables file once; the returned dict is shared, so never mutate it."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_cfg(label: str) -> dict[str, Any]:
    """Config block of one event class: enabled, min_score, merge_gap, min_dur, params."""
    return load_thresholds()["classes"][label]


def enabled_classes() -> list[str]:
    """Labels whose `enabled` flag is true, in config order."""
    return [label for label, cfg in load_thresholds()["classes"].items() if cfg.get("enabled", False)]


def seed_everything(seed: int | None = None) -> None:
    """Seed every RNG we use and force deterministic, TF32-free CUDA math."""
    seed = load_thresholds()["seed"] if seed is None else seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> str:
    """Torch device string for inference."""
    override = os.environ.get("ROADWATCH_DEVICE")
    if override:
        return override

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
