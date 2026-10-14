"""A fix's crossing is its route's: two routes may cross one fix differently,
each aircraft is checked against its own route's crossing there, and an
aircraft not routed through a waypoint is checked against its place alone.
A crossing given as a range is drawn once per episode; a route passing a fix
twice is checked at the occurrence ahead; a replaced route gets its arrival
times anew; and only a step asking for an RTA hands BlueSky one."""

from __future__ import annotations

import math

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.fixes import ROUTES, fix_at
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec, RandomizedScenario
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_FIXES = {
    "MERGE": Waypoint(lat=52.0, lon=5.0, reach_radius_nm=None),
    "EXIT": Waypoint(lat=52.0, lon=6.0),
    "ELSEWHERE": Waypoint(lat=53.0, lon=6.0),
}


def _region(name: str, lat: float, route, alt_ft: float = 12_000) -> SpawnRegion:
    return SpawnRegion(
        RegionBounds(BoxFootprint(lat - 0.001, lat + 0.001, 4.0, 4.001)),
        n_aircraft=1,
        params={"alt_ft": alt_ft, "spd_kts": 250, "hdg_deg": 90},
        aircraft_type="B738",
        route=route,
        name=name,
    )


class _Scenario:
    def __init__(self, *regions: SpawnRegion) -> None:
        self.spawn = SpawnConfig(regions=list(regions), conflict_free_spawn=False)

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=self.spawn, queryables=_FIXES, max_aircraft=4)


def _env(scenario, seed: int = 0) -> BlueskyEnv:
    env = BlueskyEnv(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    env.reset(seed=seed)
    return env


def _query(env, acid: str, name: str):
    return env.agent_context(bs.traf.id.index(acid)).query(name)


def test_two_routes_cross_one_fix_each_their_own_way():
    high = [{"waypoint": "MERGE", "alt_ft": 10_000, "alt_below_ft": 0}, "EXIT"]
    low = [{"waypoint": "MERGE", "alt_ft": 9_000, "alt_above_ft": 0}, "EXIT"]
    env = _env(_Scenario(_region("HIGH", 52.0, high), _region("LOW", 52.01, low)))
    try:
        by_region = {env._aircraft_region[acid]: acid for acid in bs.traf.id}
        at_high, at_low = (_query(env, by_region[k], "MERGE") for k in (0, 1))
        # Both at 12,000 FT: at or above 10,000 holds, at or below 9,000 does not.
        assert at_high.target.crossing.alt_ft == 10_000 and at_low.target.crossing.alt_ft == 9_000
        assert at_high.current.alt_diff_ft == pytest.approx(2_000, abs=50)
        assert at_low.current.alt_diff_ft == pytest.approx(3_000, abs=50)
        assert at_high.current.within_altitude and not at_low.current.within_altitude
        # The batched read agrees, aircraft by aircraft.
        batch = env.query_batch("MERGE", [bs.traf.id.index(by_region[k]) for k in (0, 1)])
        assert batch.values("current.alt_diff_ft") == pytest.approx([2_000, 3_000], abs=50)
    finally:
        env.close()


def test_an_aircraft_not_routed_through_a_waypoint_is_checked_on_its_place_alone():
    env = _env(_Scenario(_region("R", 52.0, [{"waypoint": "MERGE", "alt_ft": 9_000}])))
    try:
        (acid,) = bs.traf.id
        elsewhere = _query(env, acid, "ELSEWHERE")
        assert elsewhere.target.crossing.alt_ft is None
        assert math.isnan(elsewhere.current.alt_diff_ft) and elsewhere.current.within_altitude
        assert elsewhere.current.distance_nm > 50
    finally:
        env.close()


def test_a_crossing_range_is_drawn_once_an_episode_for_every_aircraft():
    route = [{"waypoint": "MERGE", "alt_ft": (8_000, 11_000)}, "EXIT"]
    scenario = RandomizedScenario(
        airspace_bounds=None,
        spawn=SpawnConfig(
            regions=[_region("A", 52.0, "ARR"), _region("B", 52.01, "ARR")],
            routes={"ARR": route},
            conflict_free_spawn=False,
        ),
        queryables=_FIXES,
    )
    drawn = []
    for seed in range(4):
        env = _env(scenario, seed)
        try:
            alts = {ROUTES.route(i).fixes[0].crossing.alt_ft for i in range(bs.traf.ntraf)}
            assert len(alts) == 1  # one draw, shared
            drawn.append(alts.pop())
        finally:
            env.close()
    assert all(8_000 <= alt <= 11_000 for alt in drawn) and len(set(drawn)) > 1


def test_a_fix_passed_twice_is_checked_at_the_occurrence_ahead():
    route = [{"waypoint": "MERGE", "alt_ft": 12_000}, "EXIT", {"waypoint": "MERGE", "alt_ft": 6_000}]
    env = _env(_Scenario(_region("R", 52.0, route)))
    try:
        (acid,) = bs.traf.id
        assert _query(env, acid, "MERGE").target.crossing.alt_ft == 12_000
        for _ in range(400):  # out past MERGE, on to EXIT
            env.step({})
            if int(bs.traf.ap.route[bs.traf.id.index(acid)].iactwp) >= 1:
                break
        env.step({})
        assert _query(env, acid, "MERGE").target.crossing.alt_ft == 6_000
    finally:
        env.close()


def test_a_replaced_route_gets_its_arrival_times_anew_and_only_rta_steps_an_rta():
    env = _env(_Scenario(_region("R", 52.0, [{"waypoint": "MERGE", "arrival_slack_s": 0.0}])))
    try:
        (acid,) = bs.traf.id
        idx = bs.traf.id.index(acid)
        assert fix_at(idx, 0).due_s is not None
        assert float(bs.traf.ap.route[idx].wprta[0]) < 0  # a target, no RTA
        env.replace_aircraft_route(
            acid, ["EXIT", {"waypoint": "MERGE", "arrival_slack_s": 60.0, "rta": True}]
        )
        idx = bs.traf.id.index(acid)
        first, second = fix_at(idx, 0), fix_at(idx, 1)
        assert first.due_s is None and second.due_s is not None
        route = bs.traf.ap.route[idx]
        assert float(route.wprta[0]) < 0
        assert float(route.wprta[1]) == pytest.approx(second.due_s)
    finally:
        env.close()


def test_a_route_edited_behind_the_record_is_read_from_bluesky():
    env = _env(_Scenario(_region("R", 52.0, [{"waypoint": "MERGE", "alt_ft": 9_000}])))
    try:
        (acid,) = bs.traf.id
        bs.stack.stack(f"ADDWPT {acid} 52.0 6.5 7000")
        bs.stack.process()
        idx = bs.traf.id.index(acid)
        assert fix_at(idx, 0).crossing.alt_ft == pytest.approx(9_000)
        added = fix_at(idx, 1)
        assert added.crossing.alt_ft == pytest.approx(7_000) and added.lon == pytest.approx(6.5)
        assert float(bs.traf.alt[idx]) / ft == pytest.approx(12_000, abs=100)
    finally:
        env.close()


def test_a_step_crossing_needs_whole_window_sides():
    from bluesky_sandbox.sim.spawn import expand_route_paths  # noqa: PLC0415

    with pytest.raises(ValueError, match="True or False"):
        expand_route_paths([{"waypoint": "A", "rta": 1}], {})
    with pytest.raises(ValueError, match="finite number"):
        expand_route_paths([{"waypoint": "A", "alt_ft": np.nan}], {})
