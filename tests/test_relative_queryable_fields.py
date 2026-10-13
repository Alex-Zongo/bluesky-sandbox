"""A field that reads the design's queryables works inside one built of it:
an intruder's range to a fix less the ownship's - who is nearer the merge
point - is bound to the env with the field it is made of."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import bind_field
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_MP = Waypoint(lat=60.0, lon=10.8)
_RANGE = obs.WaypointDistanceNm(query_name="MP", low=0, high=100)


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            RegionBounds(BoxFootprint(59.5, 59.9, 10.0, 10.4)),
            n_aircraft=2,
            params={"alt_ft": 10_000, "spd_kts": 250, "hdg_deg": 45},
        )
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[region], aircraft_type="B738"),
            queryables={"MP": _MP},
            max_aircraft=2,
        )


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(
        dt=5.0,
        obs_fields=[_RANGE],
        intruder_obs_fields=[_RANGE.relative_to_own(low=-50, high=50)],
        action_fields=[act.HdgDeg()],
    )
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    yield env
    env.close()


def test_binding_reaches_the_fields_a_field_is_built_of(env):
    relative = _RANGE.relative_to_own()
    bound = bind_field(relative, env)
    assert bound.left.env is env and bound.right.env is env
    assert relative.left.env is None  # the configured field is left as it was
    assert bind_field("not a field", env) == "not a field"


def test_an_intruders_range_to_a_fix_less_the_ownships(env):
    env.reset(seed=0)
    env.step({a: np.array([45.0], dtype=np.float32) for a in env.agents})
    relative = bind_field(_RANGE.relative_to_own(), env)
    ranges = [bind_field(_RANGE, env).get(i) for i in range(2)]
    assert relative.get_pair(0, 1) == pytest.approx(ranges[1] - ranges[0])
    assert bs.traf.ntraf == 2
