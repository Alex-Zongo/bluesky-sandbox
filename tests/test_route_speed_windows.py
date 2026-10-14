"""A route's crossing of a fix: windows about what is flown there - how far
below and above its altitude and speed an aircraft may be: "at or above",
"N KT or less", a +/- band - checked against each aircraft's own crossing;
and a speed flown with no altitude."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import RegionBounds
from bluesky_sandbox.sim.bounds.coordinates import LatLon
from bluesky_sandbox.sim.bounds.footprints import PointFootprint
from bluesky_sandbox.sim.fixes import Crossing, RouteFix
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion, expand_route_paths


def test_a_window_reaches_some_way_each_side():
    with pytest.raises(ValueError, match=">= 0"):
        Crossing(alt_ft=5000, alt_below_ft=-100)
    with pytest.raises(ValueError, match=">= 0"):
        expand_route_paths([{"waypoint": "A", "alt_ft": 5000, "alt_below_ft": -100}], {})


class _Scenario:
    def __init__(self, fixes, route):
        self.fixes, self.route = fixes, route

    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            RegionBounds(PointFootprint(LatLon(60.0, 10.0))),
            n_aircraft=1,
            params={"alt_ft": 9000, "spd_kts": 240, "hdg_deg": 90},
            route=self.route,
        )
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[region], aircraft_type="B738"),
            queryables=dict(self.fixes),
            max_aircraft=1,
        )


def _env(fixes, route):
    config = EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeg()], performance_model="openap")
    env = BlueskyEnv(scenario=_Scenario(fixes, route), config=config)
    env.reset(seed=0)
    while not bs.traf.ntraf:
        env.step({})
    return env


def test_the_window_is_what_is_checked():
    env = _env({"A": Waypoint(lat=60.0, lon=10.4)}, ["A"])
    try:
        assert float(bs.traf.alt[0]) / ft == pytest.approx(9000, abs=50)
        assert float(bs.traf.cas[0]) / kts == pytest.approx(240, abs=1)

        def check(**window):
            fix = RouteFix(lat=60.0, lon=10.4, crossing=Crossing(**window))
            current = Waypoint(lat=60.0, lon=10.4).current_state(0, fix)
            return current.within_altitude, current.within_speed

        # At 9000 FT and 240 KT:
        assert check(alt_ft=8000, alt_above_ft=1500) == (True, True)  # within 1500 above 8000
        assert check(alt_ft=9500, alt_below_ft=0) == (False, True)  # not at or above 9500
        assert check(alt_ft=8000, alt_below_ft=0) == (True, True)  # at or above 8000: open above
        assert check(speed_kts=250, speed_above_kts=0) == (True, True)  # 250 KT or less
        assert check(speed_kts=250, speed_below_kts=0) == (True, False)  # 250 KT or more
        assert check(alt_ft=5000) == (True, True)  # no window: no check
    finally:
        env.close()


def test_a_speed_with_no_altitude_is_flown():
    env = _env({"A": Waypoint(lat=60.0, lon=10.4)}, [{"waypoint": "A", "speed_kts": 220}])
    try:
        route = bs.traf.ap.route[0]
        assert route.wpalt[0] < 0 and float(route.wpspd[0]) / kts == pytest.approx(220, abs=0.1)
    finally:
        env.close()


def test_the_drivers_show_each_fix_ahead_with_its_windows():
    import importlib.util

    if importlib.util.find_spec("pygame") is None:
        pytest.skip("needs pygame")
    from bluesky_sandbox.interface.task import WaypointReadoutKey as K
    from bluesky_sandbox.ui.drivers.common.readouts import AircraftReadoutMixin

    fixes = {"A": Waypoint(lat=60.0, lon=10.4), "B": Waypoint(lat=60.0, lon=10.8)}
    route = [
        {"waypoint": "A", "speed_kts": 250, "speed_above_kts": 0},
        {"waypoint": "B", "alt_ft": 5000, "alt_below_ft": 0},
    ]
    config = EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeg()], performance_model="openap")
    env = BlueskyEnv(scenario=_Scenario(fixes, route), config=config, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        while not bs.traf.ntraf:
            env.step({})
        route = env._driver.aircraft_route_waypoints(bs.traf.id[0])
        # Both ahead of it: each with its route's windows there.
        assert [wp["constraints"] for wp in route] == [{K.SPEED_ABOVE_KTS: 0}, {K.ALT_BELOW_FT: 0}]
    finally:
        env.close()
    fmt = AircraftReadoutMixin.format_window
    assert (fmt(0, None, "FT"), fmt(None, 0, "KT", quantity="speed"), fmt(500, 500, "FT")) == (
        "AT OR ABOVE",
        "OR LESS",
        "-500/+500 FT",
    )
