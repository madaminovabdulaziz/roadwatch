"""The website's report is filled from the generated files, never typed by hand (scripts/write_report.py)."""

from __future__ import annotations

from scripts import write_report


def test_alarms_are_counted_as_evaluate_counts_them() -> None:
    # two runs 1 s apart merge into one alarm; a run 3 s later is a second one
    risk = [[0.0, 0.1], [1.0, 0.6], [1.1, 0.7], [2.1, 0.6], [5.2, 0.9], [6.0, 0.2]]
    assert write_report.alarms(risk, 0.5) == 2
    assert write_report.alarms([[0.0, 0.1]], 0.5) == 0


def test_report_numbers_come_from_the_files() -> None:
    metrics = {
        "videos": 1,
        "minutes": 2.0,
        "gt_events": 39,
        "pred_events": 42,
        "score_a": 0.4706,
        "per_class": {"stop_line": {"f1_mean": 1.0}, "stopped_vehicle": {"f1_mean": 0.0}},
    }
    pred = {
        "videos": {"a.mp4": {"events": [], "risk": [[0.0, 0.05], [0.1, 0.9], [0.2, 0.05]]}},
        "log": {"a.mp4": {"duration": 120.0, "total_sec": 360.0}},
    }
    facts = write_report.facts(metrics, pred)
    assert facts["alarms"] == 1 and facts["minutes"] == 2.0 and facts["ratios"] == "a 3.00x"
    text = " ".join(p for s in write_report.report(facts)["sections"] for p in s["paragraphs"])
    assert "Score A 0.471 (42 predicted events)" in text
    assert "We labelled 1 of the 1 sample videos ourselves (2.0 min, 39 events)" in text
    assert "stop_line 1.00" in text and "count as 0: stopped_vehicle" in text
    assert "1 alarm in 2.0 min" in text and "  " not in text
    assert "inside the budget." in text

    official = {"log": {"a.mp4": {"duration": 120.0, "total_sec": 324.0}}}
    sections = write_report.report(write_report.facts(metrics, pred, official))["sections"]
    text = " ".join(p for s in sections for p in s["paragraphs"])
    assert "the official run on the same machine took a 2.70x." in text


def test_the_gallery_shows_the_event_that_matches_a_label() -> None:
    from scripts.render_samples import gallery_picks

    events = {
        "a.mp4": [[2.0, 4.0, "jaywalking"], [10.0, 20.0, "jaywalking"], [5.0, 6.0, "red_light"]],
        "b.mp4": [[1.0, 3.0, "jaywalking"]],
    }
    gt = {"a.mp4": {"events": [[11.0, 20.0, "jaywalking"]]}}
    picks = gallery_picks(events, gt)
    assert picks["jaywalking"] == ("a.mp4", [10.0, 20.0, "jaywalking"])  # not the first one, at 2 s
    assert picks["red_light"] == ("a.mp4", [5.0, 6.0, "red_light"])  # unlabelled class: its first event
    assert gallery_picks(events, {})["jaywalking"] == ("a.mp4", [2.0, 4.0, "jaywalking"])


def test_event_map_draws_each_path_in_its_class_colour() -> None:
    import numpy as np

    from scripts import event_heat

    assert event_heat._bgr("#ff8000") == (0, 128, 255)
    bg = np.full((1080, 1920, 3), 200, np.uint8)
    path = np.array([[100.0, 900.0], [1800.0, 900.0]])
    img = event_heat.draw([("jaywalking", path)], bg, {"jaywalking": "#00ff00"})
    assert img.shape == (1080, 1920, 3)
    b, g, r = img[900, 1000].tolist()
    assert g > 180 and b < 120 and r < 120  # the path, green over the dimmed frame


def test_error_analysis_confusion_and_boundary_offsets() -> None:
    from scripts import write_ablations

    gt = {
        "a.mp4": {
            "events": [[10.0, 20.0, "jaywalking"], [30.0, 32.0, "red_light"], [50.0, 55.0, "stop_line"]]
        }
    }
    videos = {
        "a.mp4": {
            "events": [
                [10.5, 19.0, "jaywalking"],  # right class, starts 0.5 s late, ends 1 s early
                [30.0, 32.0, "stop_line"],  # the red-light runner read as a stop-line violation
                [80.0, 82.0, "jaywalking"],  # on unlabelled time
            ]
        }
    }
    errors = write_ablations.error_analysis(gt, videos)
    c = errors["confusion"]
    cell = {
        (g, p): n
        for g, row in zip(c["labelled"], c["counts"], strict=False)
        for p, n in zip(c["predicted"], row, strict=False)
    }
    assert cell[("jaywalking", "jaywalking")] == 1 and cell[("red_light", "stop_line")] == 1
    assert cell[("stop_line", "none")] == 1 and cell[("none", "jaywalking")] == 1
    assert errors["boundaries"]["jaywalking"] == {"n": 1, "start_median_sec": 0.5, "end_median_sec": -1.0}
