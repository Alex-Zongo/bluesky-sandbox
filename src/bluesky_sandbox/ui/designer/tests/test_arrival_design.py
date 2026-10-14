"""A route step's target arrival time in a design: its slack and RTA built and
generated on the step - never passed to the Waypoint."""

from __future__ import annotations

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.builder import build_scenario

from .test_designer import _example_design_spec


def test_a_route_steps_arrival_slack_and_rta_ride_on_the_step():
    spec = _example_design_spec()
    spec.queryables["fix"] = {"type": "waypoint", "lat": 52.0, "lon": 4.5}
    step = {"waypoint": "fix", "arrival_slack_s": {"type": "range", "low": 0, "high": 120}, "rta": True}
    spec.spawn = {**spec.spawn, "routes": {"r": [step]}}
    spawn = build_scenario(spec).support().spawn
    assert spawn.routes["r"] == [{"waypoint": "fix", "arrival_slack_s": (0.0, 120.0), "rta": True}]
    files = codegen.generate_task(spec, "Timed")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "'arrival_slack_s': (0, 120)" in scenario and "'rta': True" in scenario
    assert "arrival_slack_s=" not in scenario  # not a Waypoint argument
