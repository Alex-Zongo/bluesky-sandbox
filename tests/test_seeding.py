"""A seeded reset starts the episodes' random stream and an unseeded reset
continues it, as Gymnasium's do: a vector env seeded once - SuperSuit resets
its copies between episodes with no seed - plays the same episodes every run."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv

from test_slowing_down_and_direct_to import _Scenario


@pytest.fixture
def make():
    envs = []

    def build():
        env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
        envs.append(env)
        return env

    yield build
    for env in envs:
        env.close()


def _stream(env, seeds):
    """The first draw of the env's stream after each reset with ``seeds``."""
    draws = []
    for seed in seeds:
        env.reset(seed=seed)
        draws.append(float(env.rng.random()))
    return draws


def test_an_unseeded_reset_continues_the_seeded_stream(make):
    assert _stream(make(), [0, None, None]) == _stream(make(), [0, None, None])


def test_a_seeded_reset_starts_the_stream_afresh(make):
    first, *_rest, again = _stream(make(), [0, None, 0])
    assert first == again


def test_unseeded_episodes_differ_from_each_other(make):
    draws = _stream(make(), [0, None, None])
    assert len(set(draws)) == 3
