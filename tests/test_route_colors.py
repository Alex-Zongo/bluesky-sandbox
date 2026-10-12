"""A route's color: its own, or the next of the defaults - the same in every
driver and on the designer's map - kept through whatever rewrites its steps."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import RegionBounds
from bluesky_sandbox.sim.bounds.coordinates import LatLon
from bluesky_sandbox.sim.bounds.footprints import PointFootprint
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.scenario.geometry_json import route_color_map
from bluesky_sandbox.sim.scenario.transforms import rotate_spawn
from bluesky_sandbox.sim.spawn import (
    DEFAULT_ROUTE_COLORS,
    Route,
    SpawnConfig,
    SpawnRegion,
    expand_route_paths,
    route_colors,
)
from bluesky_sandbox.ui.designer.spec import _route_dump, _route_load

_QUERYABLES = {
    "SIE": Waypoint(lat=39.0955, lon=-74.80033333),
    "BOTON": Waypoint(lat=39.414472, lon=-74.454744, render_label=False),
    "HOGGS": Waypoint(lat=39.582847, lon=-74.270575, alt_ft=18_000),
}
_ROUTES = {"WEST": ["SIE", "BOTON"], "ARRIVAL": Route([{"route": "WEST"}, "HOGGS"], "white")}


def test_a_route_is_its_steps_and_one_without_a_color_takes_the_next_default():
    assert expand_route_paths(_ROUTES["ARRIVAL"], _ROUTES) == [["SIE", "BOTON", "HOGGS"]]
    # WEST the first default; ARRIVAL its own, so an inline route the second.
    assert route_colors(_ROUTES, extra=1) == [DEFAULT_ROUTE_COLORS[0], "white", DEFAULT_ROUTE_COLORS[1]]


def test_a_design_writes_and_reads_a_routes_color():
    assert _route_dump(_ROUTES["ARRIVAL"]) == {"type": "route", "steps": [{"route": "WEST"}, "HOGGS"], "color": "white"}
    assert _route_dump(Route(["SIE"])) == ["SIE"]  # none: as before, its steps
    assert _route_load(_route_dump(_ROUTES["ARRIVAL"])) == _ROUTES["ARRIVAL"]
    assert _route_load(["SIE"]) == ["SIE"]


def _spawn(**kwargs) -> SpawnConfig:
    region = SpawnRegion(
        bounds=RegionBounds(PointFootprint(LatLon(39.0, -74.9))),
        n_aircraft=1,
        params={"alt_ft": 20_000, "spd_kts": 280, "hdg_deg": 40},
        route="ARRIVAL",
        **kwargs,
    )
    return SpawnConfig(regions=[region], routes=dict(_ROUTES), aircraft_type="B738")


def test_a_rotated_episode_keeps_each_routes_color():
    rotated = rotate_spawn(_spawn(), (39.0, -74.9), 30.0)
    assert rotated.routes["ARRIVAL"].color == "white"


def test_the_map_is_told_each_routes_color_by_its_key():
    assert route_color_map(_spawn()) == {"route:WEST": DEFAULT_ROUTE_COLORS[0], "route:ARRIVAL": "white"}


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=_spawn(), queryables=dict(_QUERYABLES), max_aircraft=1)


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("pygame")
    config = EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeg()])
    env = BlueskyEnv(scenario=_Scenario(), config=config, render_mode="rgb_array")
    yield env
    env.close()


def test_the_drivers_draw_each_route_and_its_spawn_line_in_its_color(env):
    env.reset(seed=0)
    env.step({a: np.array([40.0], dtype=np.float32) for a in env.agents})
    drawn = env._driver.defined_route_polylines()
    assert [(len(r.points), r.color) for r in drawn] == [(2, DEFAULT_ROUTE_COLORS[0]), (3, "white"), (2, "white")]


def test_a_waypoint_with_its_label_off_is_not_labelled_on_a_route(env):
    env.reset(seed=0)
    acid = env._driver.tracked_acid()
    shown = {round(wp["lat"], 4): wp["render_label"] for wp in env._driver.aircraft_route_waypoints(acid)}
    assert shown == {39.0955: True, 39.4145: False, 39.5828: True}
