"""A spawn region's route rule in a design: the body of ``assign_route(request,
rng, ctx)`` its code writes. Built for the preview and the environment,
generated into the package, kept as ``route_<region>`` in a design folder's
scenario code, and completed and checked in the editor."""

from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pytest

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import BuildError, build_scenario
from bluesky_sandbox.ui.designer.code_intel import code_intel
from bluesky_sandbox.ui.designer.diagnostics import diagnostics
from bluesky_sandbox.ui.designer.folder import load_design, save_design

from .test_designer import _example_design_spec

# North of the middle to the north exit, south of it to the south one: the
# route depends on where the aircraft appears, as a list cannot.
RULE = 'return ["north"] if request.at.lat_deg > MIDDLE else ["south"]\n'


def _design(**region) -> S.DesignSpec:
    spec = _example_design_spec()
    spec.scenario_setup = "MIDDLE = 52.0"
    spec.queryables["north"] = {"type": "waypoint", "lat": 53.0, "lon": 4.8}
    spec.queryables["south"] = {"type": "waypoint", "lat": 51.0, "lon": 4.8}
    spec.spawn["regions"][0] = {**spec.spawn["regions"][0], "name": "entry", "route_rule": RULE, **region}
    return S.DesignSpec.from_json(spec.to_json())


def _routes(spawn) -> list[tuple[float, tuple[str, ...]]]:
    return [(p.request.at.lat_deg, tuple(p.request.route)) for p in spawn]


def test_it_routes_each_aircraft_as_it_appears():
    from bluesky_sandbox.sim.spawn import PlanContext

    episode = build_scenario(_design()).sample(np.random.default_rng(0))
    planned = episode.spawn.plan_episode(np.random.default_rng(0), PlanContext(queryables=episode.queryables, spawn=episode.spawn))
    assert planned
    for lat, route in _routes(planned):
        assert route == (("north",) if lat > 52.0 else ("south",))


def test_the_environment_flies_them_on_it():
    from bluesky_sandbox.env import BlueskyEnv
    from bluesky_sandbox.ui.designer.builder import build_design_config

    spec = _design()
    env = BlueskyEnv(scenario=build_scenario(spec), config=build_design_config(spec))
    try:
        env.reset(seed=0)
        assert env.spawn_log
        for record in env.spawn_log:
            assert record.route == (("north",) if record.lat_deg > 52.0 else ("south",))
    finally:
        env.close()


def test_a_region_with_a_rule_needs_a_name_for_its_code_and_no_route_too():
    with pytest.raises(BuildError, match="identifier"):
        build_scenario(_design(name="the entry"))
    with pytest.raises(BuildError, match="route or a route rule"):
        build_scenario(_design(route=["north"]))


def test_a_mistake_in_it_names_the_region_and_its_line():
    from bluesky_sandbox.ui.designer.preview import scenario_preview

    spec = _design(route_rule="x = 1\nreturn [nowhere]\n")
    with pytest.raises(BuildError, match="route rule of spawn region entry.*line 2"):
        scenario_preview(spec, seed=0)


def test_the_editor_knows_its_parameters_and_checks_its_names():
    params = {p["name"] for p in code_intel(_design())["scopes"]["route_rule:entry"]["params"]}
    assert params == {"request", "rng", "ctx"}
    problems = diagnostics(_design(route_rule="return [nowhere]\n"))
    assert any("nowhere" in p["message"] for p in problems["route_rule:entry"])


def test_a_design_folder_keeps_it_as_scenario_code(tmp_path):
    spec = _design()
    save_design(spec, tmp_path / "design")
    saved = json.loads((tmp_path / "design" / "design.json").read_text())
    assert "route_rule" not in saved["spawn"]["regions"][0]
    assert "def route_entry(request, rng, ctx):" in (tmp_path / "design" / "code" / "scenario.py").read_text()
    again = load_design(tmp_path / "design")
    assert again.spawn["regions"][0]["route_rule"].strip() == RULE.strip()


def test_the_generated_package_routes_the_same(tmp_path):
    files = codegen.generate_task(_design(), "Ruled")
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    text = files[scenario_py]
    assert "def _route_entry(request, rng, ctx):" in text
    assert "assign_route=_route_entry," in text
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    (region,) = cls().sample(np.random.default_rng(0)).spawn.regions
    assert region.assign_route is module._route_entry
