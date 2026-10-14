"""A design that spawns on a point - an arrival from over a fix - with no
airspace: the point drawn as a point, the view fit to what the design places,
and the design's route drawn from where its aircraft appear."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.bounds.coordinates import LatLon
from bluesky_sandbox.sim.bounds.footprints import PointFootprint
from bluesky_sandbox.sim.bounds.motion import Drift, MovingFootprint
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.scenario.geometry_json import spawn_region_geometry
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui.display.overlays import BoundsResource, Point, Polygon
from bluesky_sandbox.ui.drivers.common.route_heights import route_heights
from bluesky_sandbox.ui.drivers.human_driver import DefinedRoute, RoutePoint

pytest.importorskip("pygame")

_FIX = LatLon(39.0955, -74.8003)
_QUERYABLES = {
    "BOTON": Waypoint(lat=39.414472, lon=-74.454744),
    "HOGGS": Waypoint(lat=39.582847, lon=-74.270575),
}


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            bounds=RegionBounds(PointFootprint(_FIX, "SIE")),
            n_aircraft=1,
            params={"alt_ft": (19_000, 24_000), "spd_kts": 280, "hdg_deg": 40},
            route="ARRIVAL",
            name="SIE",
        )
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[region], routes={"ARRIVAL": ["BOTON", {"waypoint": "HOGGS", "alt_ft": 18_000}]}, aircraft_type="B738"),
            queryables=dict(_QUERYABLES),
            max_aircraft=1,
        )


def test_a_point_is_drawn_as_a_point_and_an_area_as_its_outline():
    (point,) = BoundsResource(RegionBounds(PointFootprint(_FIX, "SIE")), "green", "SIE").render_primitives()
    assert isinstance(point, Point) and (point.lat, point.lon) == (_FIX.lat_deg, _FIX.lon_deg)
    # A spawn on it, over a band of altitudes: the line between them.
    spawn = BoundsResource(RegionBounds(PointFootprint(_FIX)), "green", "SIE", alt_range_override=(19_000.0, 24_000.0))
    assert next(iter(spawn.render_primitives())).alt_range == (19_000.0, 24_000.0)
    (box,) = BoundsResource(RegionBounds(BoxFootprint(39.0, 40.0, -75.0, -74.0)), "red", "box").render_primitives()
    assert isinstance(box, Polygon)
    # A moving point is a point where it is now.
    moving = MovingFootprint(PointFootprint(_FIX), (Drift(heading_deg=90.0, speed_kts=60.0),))
    assert moving.is_point and not BoxFootprint(39.0, 40.0, -75.0, -74.0).is_point
    (drifted,) = BoundsResource(RegionBounds(moving), "green", "storm").render_primitives()
    assert isinstance(drifted, Point)


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeg()])
    env = BlueskyEnv(scenario=_Scenario(), config=config, render_mode="rgb_array")
    yield env
    env.close()


def test_it_renders_with_the_view_fit_to_the_spawn_and_the_waypoints(env):
    env.reset(seed=0)
    env.step({a: np.array([40.0], dtype=np.float32) for a in env.agents})
    frame = env.render()
    assert frame is not None and frame.ndim == 3 and frame.any()


def test_the_designs_route_is_drawn_from_its_spawn_at_its_heights(env):
    env.reset(seed=0)
    polylines = env._driver.defined_route_polylines()
    # BOTON has no altitude: drawn level with HOGGS's, the first there is.
    route = [
        RoutePoint(39.414472, -74.454744, 18_000.0, True, "BOTON"),
        RoutePoint(39.582847, -74.270575, 18_000.0, False, "HOGGS"),
    ]
    assert polylines[0] == DefinedRoute(route, "cyan")  # the first default
    # From the middle of the band over SIE aircraft spawn into, to where the
    # route begins - in its route's color.
    assert polylines[1] == DefinedRoute([RoutePoint(_FIX.lat_deg, _FIX.lon_deg, 21_500.0, True), route[0]], "cyan")
    # Each fix named once, where it is drawn; the spawn's center is no fix.
    assert env._driver.defined_route_fixes() == [(p, "cyan") for p in route]


def test_a_fix_with_no_altitude_is_drawn_between_those_that_have_one():
    # CAMRN FIVE: HOGGS expects FL180, CAMRN 11,000 ft; BOTON, PANZE and KARRS none.
    fixes = [(39.414472, -74.454744), (39.582847, -74.270575), (39.676, -74.168194), (39.840872, -73.985989), (40.017303, -73.861058)]
    heights, inferred = route_heights(fixes, [None, 18_000, None, None, 11_000])
    assert inferred == [True, False, True, True, False]
    assert heights[0] == 18_000  # level with the first constraint, before it
    assert 18_000 > heights[2] > heights[3] > 11_000  # on the descent, by distance
    # With no altitude anywhere, nothing is made up.
    assert route_heights(fixes[:2], [None, None]) == ([None, None], [False, False])


def test_a_spawn_is_shown_over_the_altitudes_it_spawns_at():
    (region,) = _Scenario().support().spawn.regions
    drawn = spawn_region_geometry(0, region)
    assert (drawn["alt_min_ft"], drawn["alt_max_ft"]) == (19_000.0, 24_000.0)
