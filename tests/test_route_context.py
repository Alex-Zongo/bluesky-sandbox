"""An aircraft's route through its context (``context.route``): every fix in
order, the one it is flying to, and the one before - passed, or skipped by a
direct-to."""

from __future__ import annotations

import bluesky as bs
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields.actions import DirectToWaypoint

from test_slowing_down_and_direct_to import QUERYABLES, _Scenario


@pytest.fixture
def env():
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    env.reset(seed=0)
    try:
        yield env
    finally:
        env.close()


def _is(fix, name):
    assert (fix.lat, fix.lon) == pytest.approx((QUERYABLES[name].lat, QUERYABLES[name].lon))


def _past_leg(env):
    for _ in range(20):
        if int(bs.traf.ap.route[0].iactwp) > 0:
            return
        env.step({})
    raise AssertionError("never passed LEG")


def test_the_route_is_every_fix_in_order(env):
    fixes = env.agent_context(0).route.fixes
    assert len(fixes) == 3
    for fix, name in zip(fixes, ("LEG", "MID", "END")):
        _is(fix, name)


def test_before_its_first_fix_none_is_behind_it(env):
    route = env.agent_context(0).route
    _is(route.active, "LEG")
    assert route.previous is None


def test_past_a_fix_it_is_the_one_behind(env):
    _past_leg(env)
    route = env.agent_context(0).route
    _is(route.active, "MID")
    _is(route.previous, "LEG")


def test_a_fix_a_direct_to_skipped_is_behind_it_too(env):
    _past_leg(env)
    DirectToWaypoint(waypoint="END", after=("LEG",)).set(0, 1.0)
    env.step({})
    route = env.agent_context(0).route
    _is(route.active, "END")
    _is(route.previous, "MID")
