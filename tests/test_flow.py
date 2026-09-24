"""Lane directions and flow field from synthetic tracks (RUNBOOK P1.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from roadwatch.scene.flow import flow_field, lane_directions_from_tracks, step_velocities
from roadwatch.scene.scene import Scene
from roadwatch.types import TRACK_DTYPES

SCENE = Scene(
    {
        "lanes": [
            {"id": "east", "polygon": [[0, 0], [2000, 0], [2000, 100], [0, 100]]},
            {"id": "west", "polygon": [[0, 100], [2000, 100], [2000, 200], [0, 200]]},
            {"id": "mixed", "polygon": [[0, 200], [2000, 200], [2000, 300], [0, 300]]},
            {"id": "empty", "polygon": [[0, 900], [100, 900], [100, 1000], [0, 1000]]},
        ]
    }
)


def track(track_id: int, x0: float, y: float, vx: float, cls: str = "car", n: int = 50) -> pd.DataFrame:
    t = np.arange(n) / 10
    fx = x0 + vx * t
    return pd.DataFrame(
        {
            "frame": np.arange(n) * 3,
            "t": t,
            "track_id": track_id,
            "cls": cls,
            "conf": 0.9,
            "x1": fx - 20,
            "y1": y - 40,
            "x2": fx + 20,
            "y2": y,
            "fx": fx,
            "fy": y + 0 * t,
        }
    )


def table(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True).astype(TRACK_DTYPES)


def test_step_velocities_per_track_only_road_users() -> None:
    steps = step_velocities(
        table(track(1, 0, 50, 200), track(2, 1000, 50, -100), track(3, 0, 50, 300, cls="person"))
    )
    assert set(steps["track_id"]) == {1, 2}
    np.testing.assert_allclose(steps.loc[steps["track_id"] == 1, "vx"], 200, atol=1e-3)
    np.testing.assert_allclose(steps["vy"], 0, atol=1e-6)
    assert len(steps) == 2 * 49  # no step joins two tracks


def test_lane_directions_and_consistency() -> None:
    tt = table(
        track(1, 0, 50, 300),
        track(2, 100, 60, 250),
        track(3, 1900, 150, -300),
        track(4, 0, 250, 300),
        track(5, 1900, 260, -300),
        track(6, 500, 150, 300, cls="person"),  # pedestrians never vote
    )
    d = lane_directions_from_tracks(tt, SCENE).set_index("lane_id")
    np.testing.assert_allclose(d.loc["east", ["dx", "dy"]].to_numpy(float), [1, 0], atol=1e-6)
    np.testing.assert_allclose(d.loc["west", ["dx", "dy"]].to_numpy(float), [-1, 0], atol=1e-6)
    assert d.loc["east", "ok"] and d.loc["west", "ok"]
    assert d.loc["mixed", "consistency"] < 0.1 and not d.loc["mixed", "ok"]
    assert d.loc["empty", "n"] == 0 and not d.loc["empty", "ok"]


def test_slow_jitter_is_ignored() -> None:
    d = lane_directions_from_tracks(table(track(1, 500, 50, 5)), SCENE).set_index("lane_id")
    assert d.loc["east", "n"] == 0


def test_flow_field_cells() -> None:
    field = flow_field(table(track(1, 0, 50, 300), track(2, 0, 60, 300)), 2000, 300)
    assert field["cell_px"] > 0 and field["cells"]
    assert all(c["vx"] == 300.0 and c["vy"] == 0.0 for c in field["cells"])
    assert all(0 <= c["x"] <= 2000 and 0 <= c["y"] <= 300 for c in field["cells"])
