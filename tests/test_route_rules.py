"""A spawn region routed by a rule - ``assign_route(request, rng, ctx)`` -
gives each of its aircraft the route it returns for the aircraft as drawn: as
the episode is planned, when a steady-state region tops up, and drawn as a
:class:`RegionSource`. A spawn source's rule takes the same arguments."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import Mixture, PlanContext, RegionSource, SpawnConfig, SpawnRegion, TowardRoute

QUERYABLES = {"north": Waypoint(lat=52.5, lon=4.5), "south": Waypoint(lat=51.5, lon=4.5)}


def by_side(request, rng, ctx):
    """North of 52 N to the north fix, else the south one; a draw from ``rng``
    on the way, as a rule drawing its own point would."""
    rng.random()
    return ["north"] if request.at.lat_deg > 52.0 else ["south"]


def _region(**kwargs):
    return SpawnRegion(
        RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 4.6)),
        n_aircraft=kwargs.pop("n_aircraft", 6),
        params={"alt_ft": 10_000, "spd_kts": 250, "hdg_deg": TowardRoute()},
        aircraft_type="B738",
        assign_route=kwargs.pop("assign_route", by_side),
        name="mid",
        **kwargs,
    )


def _ctx(spawn):
    return PlanContext(queryables=QUERYABLES, spawn=spawn)


def test_each_aircraft_of_the_region_flies_the_route_its_rule_gives():
    spawn = SpawnConfig(regions=[_region()])
    planned = spawn.plan_episode(np.random.default_rng(0), _ctx(spawn))
    assert len(planned) == 6
    for p in planned:
        assert p.request.route == (["north"] if p.request.at.lat_deg > 52.0 else ["south"])
    assert {tuple(p.request.route) for p in planned} == {("north",), ("south",)}


def test_a_region_takes_its_route_or_a_rule_not_both():
    with pytest.raises(ValueError, match="route or a rule"):
        _region(route=["north"])


def test_a_rule_routes_a_region_heading_toward_its_route():
    # TowardRoute needs a route: the rule is one.
    SpawnConfig(regions=[_region()])
    with pytest.raises(ValueError, match="no route"):
        SpawnConfig(regions=[_region(assign_route=None)])


def test_a_region_drawn_as_a_source_is_routed_by_its_rule():
    spawn = SpawnConfig(regions=[], sources=[Mixture([RegionSource(_region())], n=4)])
    planned = spawn.plan_episode(np.random.default_rng(1), _ctx(spawn))
    assert len(planned) == 4
    for p in planned:
        assert p.request.route == (["north"] if p.request.at.lat_deg > 52.0 else ["south"])


class _Scenario:
    def __init__(self, region):
        self.spawn = SpawnConfig(regions=[region], conflict_free_spawn=False)

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=self.spawn, queryables=QUERYABLES, max_aircraft=self.spawn.max_aircraft())


@pytest.mark.parametrize("maintain", [False, True])
def test_the_env_flies_each_aircraft_on_its_rules_route_spawned_and_topped_up(maintain):
    env = BlueskyEnv(
        scenario=_Scenario(_region(n_aircraft=3, maintain=maintain)),
        config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]),
    )
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step({})
        assert len(env.spawn_log) >= 3
        for record in env.spawn_log:
            assert record.route == (("north",) if record.lat_deg > 52.0 else ("south",))
    finally:
        env.close()
