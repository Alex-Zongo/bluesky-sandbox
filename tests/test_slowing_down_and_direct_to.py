"""What a point-merge controller clears: a speed reduction counted from the
speed the aircraft is cleared to (``ApSpdDeltaKts``), offered as reductions
only (``StepNormalizer(steps_up=0)``), and a turn direct to a fix of its route
once it is past another (``DirectToWaypoint``)."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields.actions import ApSpdDeltaKts, DirectToWaypoint
from bluesky_sandbox.interface.fields.grid import Grid
from bluesky_sandbox.interface.wrappers.observations.normalizer import StepNormalizer
from bluesky_sandbox.sim.bounds import LatLon, PointFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

# Due east along 52 N: a leg's start, a fix on it, its end.
QUERYABLES = {
    "LEG": Waypoint(lat=52.0, lon=4.0),
    "MID": Waypoint(lat=52.0, lon=4.5),
    "END": Waypoint(lat=52.0, lon=5.0),
}


class _Scenario:
    def __init__(self):
        region = SpawnRegion(
            RegionBounds(PointFootprint(LatLon(52.0, 3.97))),
            n_aircraft=1,
            params={"alt_ft": 10_000, "spd_kts": 250, "hdg_deg": 90.0},
            aircraft_type="B738",
            route=["LEG", "MID", "END"],
        )
        self.spawn = SpawnConfig(regions=[region], conflict_free_spawn=False)

    def sample(self, _rng):
        return EpisodeSpec(airspace_bounds=None, spawn=self.spawn, queryables=QUERYABLES, max_aircraft=1)

    def support(self):
        return self.sample(None)


@pytest.fixture
def env():
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    env.reset(seed=0)
    try:
        yield env
    finally:
        env.close()


def _active(idx=0):
    return int(bs.traf.ap.route[idx].iactwp)


def test_a_reduction_counts_from_the_speed_cleared_not_the_one_flown(env):
    bs.stack.stack(f"SPD {bs.traf.id[0]} 220")
    env.step({})
    # Still slowing from 250: the reduction is from 220.
    assert bs.traf.cas[0] / kts > 225
    assert ApSpdDeltaKts().nominal(0) == pytest.approx(220.0, abs=0.01)


def test_reductions_only_offers_no_step_up():
    steps = StepNormalizer(steps_each_way=2, steps_up=0, include_zero=False)
    assert steps.steps() == [-2, -1]
    assert StepNormalizer(steps_each_way=2, include_zero=False).steps() == [-2, -1, 1, 2]
    with pytest.raises(ValueError, match="steps_up"):
        StepNormalizer(steps_each_way=1, steps_up=-1)


def test_a_reduction_on_the_grid_lands_on_the_next_speed_down(env):
    reductions = StepNormalizer(steps_each_way=2, steps_up=0, include_zero=False)
    action = ApSpdDeltaKts(grid=Grid(10, on="target"), normalizer=reductions)
    # Under VNAV at its 250 KT: index 1 is one step down.
    assert action.normalizer.denormalize(action, [1.0], 0) == pytest.approx(-10.0)


def test_direct_to_waits_until_past_the_leg_start(env):
    direct = DirectToWaypoint(waypoint="END", after=("LEG",))
    assert _active() == 0 and not direct.current_switch_state(0)
    direct.set(0, 1.0)  # before LEG: nothing
    env.step({})
    assert _active() == 0
    for _ in range(20):
        if _active() > 0:
            break
        env.step({})
    assert _active() == 1  # past LEG, flying to MID
    direct.set(0, 1.0)
    env.step({})
    assert _active() == 2 and direct.current_switch_state(0)


def test_direct_to_needs_its_waypoint():
    with pytest.raises(ValueError, match="needs the waypoint"):
        DirectToWaypoint()
