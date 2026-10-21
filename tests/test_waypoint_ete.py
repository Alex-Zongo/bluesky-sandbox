"""``WaypointEteS``: the time to a named fix along the aircraft's route, at its
ground speed - the legs still to fly, not the straight line - and 0 once the
fix is behind it."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import nm
from bluesky.tools.geo import kwikdist

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields.observations import queryable as qobs

from test_slowing_down_and_direct_to import QUERYABLES, _Scenario


@pytest.fixture
def env():
    fields = [qobs.WaypointEteS(query_name="END"), qobs.WaypointEteS(query_name="LEG")]
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=fields, action_fields=[]))
    env.reset(seed=0)
    try:
        yield env
    finally:
        env.close()


def _along_nm(*fixes):
    lat, lon = float(bs.traf.lat[0]), float(bs.traf.lon[0])
    total = 0.0
    for name in fixes:
        wp = QUERYABLES[name]
        total += kwikdist(lat, lon, wp.lat, wp.lon)
        lat, lon = wp.lat, wp.lon
    return total


def test_it_is_the_legs_still_to_fly_at_the_ground_speed(env):
    env.step({})
    (agent,) = env.agents
    raw = env.raw_observation(agent)["ownship"]
    gs = float(bs.traf.gs[0])
    assert raw["end_ete_s"] == pytest.approx(_along_nm("LEG", "MID", "END") * nm / gs, rel=1e-3)
    assert raw["leg_ete_s"] == pytest.approx(_along_nm("LEG") * nm / gs, rel=1e-3)


def test_a_fix_behind_reads_zero(env):
    for _ in range(30):
        env.step({})
        if int(bs.traf.ap.route[0].iactwp) > 0:
            break
    env.step({})
    (agent,) = env.agents
    assert env.raw_observation(agent)["ownship"]["leg_ete_s"] == 0.0
