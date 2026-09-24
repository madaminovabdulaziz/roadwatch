"""Registry of event rules: one module per official class, in the order of solution.CLASSES."""

from __future__ import annotations

from roadwatch.events import (
    accident,
    congestion,
    failure_to_yield,
    fire_smoke,
    illegal_turn,
    illegal_u_turn,
    jaywalking,
    near_miss,
    red_light,
    road_obstacle,
    solid_line_crossing,
    stop_line,
    stopped_vehicle,
    wrong_way,
)
from roadwatch.events.base import EventRule

RULES: dict[str, EventRule] = {
    rule.LABEL: rule
    for rule in (
        accident,
        near_miss,
        red_light,
        wrong_way,
        illegal_u_turn,
        stopped_vehicle,
        jaywalking,
        failure_to_yield,
        illegal_turn,
        solid_line_crossing,
        stop_line,
        congestion,
        road_obstacle,
        fire_smoke,
    )
}
