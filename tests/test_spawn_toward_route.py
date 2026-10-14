"""Arrivals over a fix: each aircraft spawned around it headed at it - its
route's first fix - and the fix flown over, not by, before the route turns."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import RegionBounds
from bluesky_sandbox.sim.bounds.coordinates import LatLon
from bluesky_sandbox.sim.bounds.derived import AnnularSectorFootprint
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion, TowardRoute, route_heading
from bluesky_sandbox.ui.designer.spec import dump_value, load_value

_SIE = LatLon(39.0955, -74.80033333)
_QUERYABLES = {
    "SIE": Waypoint(lat=_SIE.lat_deg, lon=_SIE.lon_deg),
    "BOTON": Waypoint(lat=39.414472, lon=-74.454744),
}


def _arc(**kwargs) -> SpawnRegion:
    """Southwest of SIE, 2.5-3.5 nm out, over 90 degrees."""
    return SpawnRegion(
        RegionBounds(AnnularSectorFootprint(_SIE, 2.5, 3.5, 220.0, 45.0)),
        n_aircraft=4,
        params={"alt_ft": 20_000, "spd_kts": 280, "hdg_deg": TowardRoute()},
        **kwargs,
    )


def test_the_heading_is_the_course_to_the_routes_first_fix():
    fix = {"FIX": Waypoint(lat=0.0, lon=0.0)}
    assert route_heading(-1.0, 0.0, ["FIX"], fix) == pytest.approx(0.0)
    assert route_heading(0.0, -1.0, [{"waypoint": "FIX", "alt_ft": 9000}], fix) == pytest.approx(90.0)
    with pytest.raises(ValueError, match="route"):
        route_heading(0.0, 0.0, None, fix)


def test_it_is_a_heading_and_needs_a_route():
    with pytest.raises(ValueError, match="hdg_deg"):
        SpawnRegion(RegionBounds(AnnularSectorFootprint(_SIE, 2.5, 3.5, 220.0, 45.0)), n_aircraft=1, params={"alt_ft": TowardRoute(), "spd_kts": 280})
    with pytest.raises(ValueError, match="no route"):
        SpawnConfig(regions=[_arc()], aircraft_type="B738")


def test_a_design_writes_it_and_reads_it():
    assert dump_value(TowardRoute()) == {"type": "toward_route"}
    assert load_value({"type": "toward_route"}) == TowardRoute()


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[_arc(route=[{"waypoint": "SIE", "flyover": True}, "BOTON"])], aircraft_type="B738"),
            queryables=dict(_QUERYABLES),
            max_aircraft=4,
        )


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeg()])
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    yield env
    env.close()


def test_each_aircraft_spawns_headed_at_the_fix_and_flies_over_it(env):
    env.reset(seed=0)
    assert bs.traf.ntraf == 4
    for i in range(bs.traf.ntraf):
        at = (float(bs.traf.lat[i]), float(bs.traf.lon[i]))
        assert float(bs.traf.hdg[i]) == pytest.approx(route_heading(*at, ["SIE"], _QUERYABLES), abs=0.5)
        route = bs.traf.ap.route[i]
        # SIE flown over; the route after it by, as BlueSky flies by default.
        assert list(route.wpflyby) == [False, True]
    closest = np.full(bs.traf.ntraf, np.inf)
    for _ in range(40):
        env.step({})
        for i in range(bs.traf.ntraf):
            north = (float(bs.traf.lat[i]) - _SIE.lat_deg) * 60.0
            east = (float(bs.traf.lon[i]) - _SIE.lon_deg) * 60.0 * np.cos(np.radians(_SIE.lat_deg))
            closest[i] = min(closest[i], np.hypot(north, east))
    assert (closest < 0.3).all()  # nm: over SIE, not cutting inside it
