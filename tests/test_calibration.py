"""Scene calibration: reference median, shapes <-> scene.json, validation, overlay, tool server (P1.1)."""

from __future__ import annotations

import copy
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

from roadwatch.config import REPO_ROOT, SCENE_PATH
from roadwatch.scene.calibration import make_reference, scene_to_shapes, shapes_to_scene, validate_scene
from roadwatch.scene.overlay import draw_scene
from roadwatch.scene.scene import Scene
from tests.conftest import TINY_SIZE

SCENE = {
    "image_size": [400, 300],
    "reference_frame": "configs/reference.jpg",
    "carriageway": [[0.0, 100.0], [400.0, 100.0], [400.0, 300.0], [0.0, 300.0]],
    "sidewalks": [[[0.0, 80.0], [400.0, 80.0], [400.0, 100.0], [0.0, 100.0]]],
    "intersection": [[150.0, 100.0], [250.0, 100.0], [250.0, 200.0], [150.0, 200.0]],
    "lanes": [
        {
            "id": "n1",
            "polygon": [[200.0, 200.0], [300.0, 200.0], [300.0, 300.0], [200.0, 300.0]],
            "direction": [0.0, -1.0],
            "approach": "S",
            "signal": "L1",
            "allowed_exits": ["N_out"],
        },
        {
            "id": "s1",
            "polygon": [[100.0, 200.0], [200.0, 200.0], [200.0, 300.0], [100.0, 300.0]],
            "direction": [0.0, 1.0],
        },
    ],
    "directions": [{"id": "NB", "lanes": ["n1"]}, {"id": "SB", "lanes": ["s1"]}],
    "exits": [{"id": "N_out", "polygon": [[200.0, 100.0], [300.0, 100.0], [300.0, 120.0], [200.0, 120.0]]}],
    "stop_lines": [{"id": "sl_S", "line": [[200.0, 210.0], [300.0, 210.0]], "lanes": ["n1"], "signal": "L1"}],
    "crosswalks": [
        {"id": "cw_S", "polygon": [[200.0, 215.0], [300.0, 215.0], [300.0, 235.0], [200.0, 235.0]]}
    ],
    "solid_lines": [{"id": "sol1", "polyline": [[200.0, 240.0], [200.0, 300.0]]}],
    "signals": [
        {
            "id": "L1",
            "red": [10.0, 10.0, 5.0, 5.0],
            "yellow": [10.0, 16.0, 5.0, 5.0],
            "green": [10.0, 22.0, 5.0, 5.0],
        }
    ],
    "no_u_turn_zones": [[[150.0, 100.0], [250.0, 100.0], [250.0, 200.0]]],
    "homography": {
        "image_pts": [[100.0, 300.0], [300.0, 300.0], [260.0, 150.0], [140.0, 150.0]],
        "world_pts": [[0.0, 0.0], [7.0, 0.0], [7.0, 30.0], [0.0, 30.0]],
    },
    "u_turn_prohibited_everywhere": False,
}


def test_reference_median_removes_moving_objects(tiny_video: Path) -> None:
    ref = make_reference(tiny_video, n=9)
    assert ref.shape == (TINY_SIZE[1], TINY_SIZE[0], 3) and ref.dtype == np.uint8
    # a red box moves across the black frame; the median keeps the empty background
    assert (ref[:, :, 2] > 128).mean() < 0.02


def test_scene_shapes_round_trip() -> None:
    shapes = scene_to_shapes(SCENE)
    assert shapes_to_scene(shapes, (400, 300)) == SCENE
    assert {s["layer"] for s in shapes} >= {"lanes", "lane_directions", "signals", "homography"}


def test_shapes_to_scene_normalises_and_splits() -> None:
    shapes = [
        {
            "layer": "lanes",
            "pts": [[0, 0], [10, 0], [10, 10]],
            "attrs": {"id": "a", "allowed_exits": " x , y ", "direction_group": "G"},
        },
        {"layer": "lane_directions", "pts": [[0, 0], [3, 4]], "attrs": {"lane": "a"}},
        {"layer": "carriageway", "pts": [[0, 0], [1, 0], [1, 1]], "attrs": {}},
    ]
    scene = shapes_to_scene(shapes, (100, 50), u_turn_prohibited_everywhere=True)
    assert scene["lanes"][0]["direction"] == [0.6, 0.8]
    assert scene["lanes"][0]["allowed_exits"] == ["x", "y"]
    assert scene["directions"] == [{"id": "G", "lanes": ["a"]}]
    assert scene["u_turn_prohibited_everywhere"] is True
    assert "stop_lines" not in scene and "homography" not in scene
    with pytest.raises(ValueError):
        shapes_to_scene([{"layer": "nope", "pts": [[0, 0]], "attrs": {}}], (1, 1))


def test_validate_accepts_consistent_scene() -> None:
    assert validate_scene(SCENE) == []


def test_validate_reports_broken_references() -> None:
    bad = copy.deepcopy(SCENE)
    bad["lanes"][1]["id"] = "n1"
    bad["stop_lines"][0]["lanes"] = ["zz"]
    bad["lanes"][0]["allowed_exits"] = ["nowhere"]
    del bad["signals"][0]["green"]
    bad["homography"]["image_pts"] = [[0, 0], [1, 1], [2, 2], [3, 3]]
    problems = " | ".join(validate_scene(bad))
    for expected in (
        "duplicate id 'n1'",
        "unknown lane 'zz'",
        "unknown exit 'nowhere'",
        "missing lamp",
        "degenerate",
    ):
        assert expected in problems


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="configs/scene.json not calibrated yet")
def test_committed_scene_is_consistent() -> None:
    assert validate_scene(json.loads(SCENE_PATH.read_text(encoding="utf-8"))) == []


def test_overlay_draws_without_touching_input() -> None:
    img = np.zeros((300, 400, 3), np.uint8)
    out = draw_scene(img, Scene(SCENE))
    assert out.shape == img.shape and img.sum() == 0 and out.sum() > 0


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool_url(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    tool = _load_script("calibrate_scene")
    reference = tmp_path / "reference.jpg"
    cv2.imwrite(str(reference), np.zeros((300, 400, 3), np.uint8))
    scene_path = tmp_path / "scene.json"
    server = tool.ThreadingHTTPServer(
        ("127.0.0.1", 0), tool.make_handler(tool.CalibrationApp(reference, scene_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", scene_path
    finally:
        server.shutdown()
        server.server_close()


def _post(url: str, body: bytes) -> dict:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_tool_serves_page_state_and_saves(tool_url: tuple[str, Path]) -> None:
    url, scene_path = tool_url
    with urllib.request.urlopen(url + "/", timeout=10) as r:
        assert b"scene calibration" in r.read()
    with urllib.request.urlopen(url + "/api/state", timeout=10) as r:
        state = json.loads(r.read())
    assert state["image_size"] == [400, 300] and state["shapes"] == [] and "lanes" in state["layers"]

    res = _post(url + "/api/save", json.dumps({"shapes": scene_to_shapes(SCENE)}).encode())
    assert res["problems"] == []
    saved = json.loads(scene_path.read_text(encoding="utf-8"))
    assert {k: v for k, v in saved.items() if k != "reference_frame"} == {
        k: v for k, v in SCENE.items() if k not in ("reference_frame", "image_size")
    } | {"image_size": [400, 300]}

    _post(url + "/api/save", json.dumps({"shapes": []}).encode())
    assert scene_path.with_suffix(".json.bak").exists()  # the previous scene is kept

    with pytest.raises(urllib.error.HTTPError) as err:
        _post(url + "/api/save", b"{not json")
    assert err.value.code == 400
