"""Scene geometry helpers: point-in-polygon, layer shapes, segment crossing, homography (SPEC §9)."""

from __future__ import annotations

import numpy as np
import pytest

from roadwatch.scene.scene import Scene, points_in_polygon, segments_cross

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
L_SHAPE = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]  # concave


def test_points_in_polygon_square_and_concave() -> None:
    pts = np.array([[5, 5], [15, 5], [-1, 5], [2, 8], [8, 8]])
    assert points_in_polygon(pts, SQUARE).tolist() == [True, False, False, True, True]
    assert points_in_polygon(pts, L_SHAPE).tolist() == [False, False, False, True, False]


def test_layer_shapes_single_list_and_named() -> None:
    scene = Scene(
        {
            "carriageway": SQUARE,
            "sidewalks": [SQUARE, [[20, 0], [30, 0], [30, 10], [20, 10]]],
            "crosswalks": [
                {"id": "cw_S", "polygon": SQUARE},
                {"id": "cw_N", "polygon": [[20, 0], [30, 0], [30, 10]]},
            ],
        }
    )
    pts = np.array([[5, 5], [25, 2], [50, 50]])
    assert scene.point_in("carriageway", pts).tolist() == [True, False, False]
    assert scene.point_in("sidewalks", pts).tolist() == [True, True, False]
    assert scene.region_of("crosswalks", pts).tolist() == ["cw_S", "cw_N", ""]
    assert scene.point_in("intersection", pts).tolist() == [False, False, False]
    assert [pid for pid, _ in scene.polygons("sidewalks")] == ["sidewalks_0", "sidewalks_1"]


def test_lane_of_and_directions() -> None:
    scene = Scene(
        {"lanes": [{"id": "n1", "polygon": SQUARE, "direction": [0, -2]}, {"id": "s1", "polygon": L_SHAPE}]}
    )
    assert scene.lane_of(np.array([[5, 5], [2, 8], [40, 40]])).tolist() == ["n1", "n1", ""]
    dirs = scene.lane_directions()
    assert list(dirs) == ["n1"]
    np.testing.assert_allclose(dirs["n1"], [0, -1])


def test_segments_cross() -> None:
    line = [[0, 5], [10, 5]]
    p0 = np.array([[5, 0], [5, 0], [15, 0], [5, 0], [5, 5], [5, 6]])
    p1 = np.array([[5, 10], [5, 4], [15, 10], [6, 0], [5, 10], [5, 5]])
    # crossing, stays on one side, beyond the segment end, parallel, leaves the line, lands on the line
    assert segments_cross(line, p0, p1).tolist() == [True, False, False, False, False, True]


def test_landing_exactly_on_line_counts_once() -> None:
    line = [[0, 5], [10, 5]]
    path = np.array([[5, 3], [5, 5], [5, 7]])
    assert segments_cross(line, path[:-1], path[1:]).sum() == 1


def test_homography_scale_and_round_trip() -> None:
    scene = Scene({"homography": {"image_pts": SQUARE, "world_pts": [[0, 0], [1, 0], [1, 1], [0, 1]]}})
    np.testing.assert_allclose(scene.to_world(np.array([[5, 5], [10, 0]])), [[0.5, 0.5], [1, 0]], atol=1e-9)

    perspective = Scene(
        {
            "homography": {
                "image_pts": [[1000, 2000], [2800, 2000], [2300, 900], [1500, 900]],
                "world_pts": [[0, 0], [7, 0], [7, 30], [0, 30]],
            }
        }
    )
    pts = np.array([[1200.0, 1900.0], [2000.0, 1200.0], [2500.0, 1500.0]])
    np.testing.assert_allclose(perspective.to_image(perspective.to_world(pts)), pts, atol=1e-6)
    np.testing.assert_allclose(perspective.to_world(np.array([[2300, 900]])), [[7, 30]], atol=1e-6)


def test_empty_scene() -> None:
    scene = Scene()
    assert not scene.has("homography")
    assert scene.lane_of(np.array([[1, 1]])).tolist() == [""]
    with pytest.raises(ValueError):
        scene.to_world(np.array([[1, 1]]))
