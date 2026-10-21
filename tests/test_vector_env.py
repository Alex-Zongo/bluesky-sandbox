"""Copies of a task env as one vector env, each copy in its own process."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.integrations import (
    sb3_vec_env,
    vec_env,
    wrap_parallel_env,
)
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers import StepNormalizer
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

pytest.importorskip("supersuit")


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
                    n_aircraft=3,
                    params={"alt_ft": (8_000, 30_000), "spd_kts": (230, 270)},
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )
        return EpisodeSpec(airspace_bounds=None, spawn=spawn, queryables={}, max_aircraft=3)


def make_env(render_mode=None, **drawing):
    """Module-level, so a worker process can import it."""
    env = BlueskyEnv(
        scenario=_Scenario(),
        render_mode=render_mode,
        **drawing,
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt(), obs.LatDeg()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def _step(v, n=2):
    for _ in range(n):
        actions = np.stack([v.action_space.sample() for _ in range(v.num_envs)])
        out = v.step(actions)
    return out


def test_one_process_is_built_here():
    v = vec_env(make_env)
    try:
        observations, _ = v.reset(seed=0)
        assert v.num_envs == 3
        _step(v)
    finally:
        v.close()


def test_processes_each_run_their_own_episode():
    v = vec_env(make_env, n_processes=2)
    try:
        observations, _ = v.reset(seed=0)
        assert v.num_envs == 6
        own = np.asarray(observations)
        # Worker i is seeded seed + i: the two copies' aircraft differ.
        assert not np.allclose(own[:3], own[3:])
        _step(v)
    finally:
        v.close()


def test_a_process_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        vec_env(make_env, n_processes=0)


def test_sb3_takes_the_seed_it_is_given():
    pytest.importorskip("stable_baselines3")
    first = sb3_vec_env(make_env)
    try:
        first.seed(7)
        a = first.reset()
    finally:
        first.close()
    second = sb3_vec_env(make_env)
    try:
        second.seed(7)
        b = second.reset()
    finally:
        second.close()
    np.testing.assert_allclose(a, b)


def test_a_padded_critic_intruder_block_widens_each_row():
    from gymnasium.spaces import Box, Dict  # noqa: PLC0415

    from bluesky_sandbox import critic_obs, critic_observation_space  # noqa: PLC0415

    space = Dict(
        {
            "ownship": Box(-1, 1, (3,)),
            "intruders": Box(-1, 1, (4, 2)),
            "critic_intruders": Box(0, 5, (4, 1)),
        }
    )
    widened = critic_observation_space(space)
    assert set(widened.spaces) == {"ownship", "intruders"}
    assert widened["intruders"].shape == (4, 3)
    batch = {k: np.stack([s.sample() for _ in range(6)]) for k, s in space.spaces.items()}
    assert critic_obs(batch)["intruders"].shape == (6, 4, 3)


class _LateScenario(_Scenario):
    """Its aircraft spawn a minute in: no agents at all before then."""

    def support(self):
        spec = super().support()
        (region,) = spec.spawn.regions
        late = type(region)(**{**region.__dict__, "spawn_time": 60.0})
        spawn = type(spec.spawn)(regions=[late], aircraft_type="B744", conflict_free_spawn=False)
        return type(spec)(**{**spec.__dict__, "spawn": spawn})


def make_late_env(render_mode=None):
    env = BlueskyEnv(
        scenario=_LateScenario(),
        render_mode=render_mode,
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def test_an_episode_whose_traffic_spawns_later_is_not_reset_before_it():
    import bluesky as bs  # noqa: PLC0415

    v = vec_env(make_late_env)
    try:
        v.reset(seed=0)
        for _ in range(20):  # 100 s: past the spawn at 60 s
            _, _, terms, truncs, infos = v.step(
                np.stack([v.action_space.sample() for _ in range(v.num_envs)])
            )
        # Had every empty slot counted as done, the env would reset each step
        # and the clock never pass zero.
        assert bs.sim.simt >= 60.0
        assert sum(not i.get("_padded", False) for i in infos) == 3
    finally:
        v.close()


# A switch and a choice: a Dict of a binary and a discrete part.
_SWITCH_AND_CHOICE = [
    act.AutopilotLnav(),
    act.SpdDeltaKts(normalizer=StepNormalizer(10.0, 3, include_zero=False)),
]


def make_discrete_env(render_mode=None, action_fields=None):
    # The SB3 package's own wrappers.py, as the designer writes it.
    from bluesky_sandbox.ui.designer.package_files.wrappers import (  # noqa: PLC0415
        MultiDiscreteActions,
    )

    env = BlueskyEnv(
        scenario=_Scenario(),
        render_mode=render_mode,
        config=EnvConfig(
            dt=5.0, obs_fields=[obs.AltFt()], action_fields=action_fields or _SWITCH_AND_CHOICE
        ),
    )
    return MultiDiscreteActions(wrap_parallel_env(env, max_agents=3))


def test_switches_and_choices_are_one_multidiscrete_split_back_for_the_env():
    pytest.importorskip("stable_baselines3")
    from gymnasium.spaces import MultiDiscrete  # noqa: PLC0415

    env = make_discrete_env()
    try:
        space = env.action_space(None)
        assert isinstance(space, MultiDiscrete) and list(space.nvec) == [2, 6]
        parts = env._parts_of([1, 4])
        assert list(parts) == ["binary", "discrete"]
        assert parts["binary"].tolist() == [1] and parts["discrete"].tolist() == [4]
        assert [s.name for s in env.action_layout()["discrete"]] == [
            s.name for part in ("binary", "discrete") for s in env.env.action_layout()[part]
        ]
    finally:
        env.close()


def test_a_continuous_action_among_switches_cannot_be_one_multidiscrete():
    pytest.importorskip("stable_baselines3")
    with pytest.raises(TypeError, match="switches and choices only"):
        make_discrete_env(action_fields=[*_SWITCH_AND_CHOICE, act.HdgDeltaDeg()])


def test_sb3_trains_switches_and_choices():
    sb3 = pytest.importorskip("stable_baselines3")
    v = sb3_vec_env(make_discrete_env)
    try:
        model = sb3.PPO("MlpPolicy", v, n_steps=8, batch_size=12, n_epochs=1, seed=0)
        model.learn(total_timesteps=24)
        action, _ = model.predict(v.reset(), deterministic=True)
        assert action.shape == (3, 2)
    finally:
        v.close()


class _CostlyEnv(BlueskyEnv):
    """Every step of every aircraft costs 1: an aircraft's episode costs its
    steps flown, until it is truncated 200 s after it spawns."""

    def truncated(self, obs, action, context, info, rng):
        return info["time_in_env"] >= 200.0

    def cost(self, obs, action, terminated, truncated, context, info, rng):
        return 1.0


def make_costly_env(render_mode=None):
    env = _CostlyEnv(
        scenario=_LateScenario(),
        render_mode=render_mode,
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def test_ppo_lagrangian_counts_an_aircraft_episode_from_spawn_to_done():
    pytest.importorskip("stable_baselines3")
    from bluesky_sandbox.ui.designer.package_files.ppo_lagrangian import PPOLagrangian  # noqa: PLC0415

    finished = []

    class _Watched(PPOLagrangian):
        def _update_lambda(self):
            finished.extend(self._finished_costs)
            super()._update_lambda()

    v = sb3_vec_env(make_costly_env)
    try:
        # Past the time limit: the three spawn at 60 s and fly until it.
        model = _Watched("MlpPolicy", v, cost_limit=0.0, n_steps=60, batch_size=60, n_epochs=1, seed=0)
        model.learn(180)
    finally:
        v.close()
    # The slots waiting to spawn are no episode: each episode is an aircraft's,
    # its steps from spawn to the limit, the same for all three.
    assert len(finished) == 3 and len(set(finished)) == 1
    assert finished[0] >= 25
    assert model.lagrangian_lambda > 0


class _FinishingEnv(BlueskyEnv):
    """Each aircraft terminates 50 s after it spawns, rewarded 10 and costing
    1 for it - left out of the env's next observations, as a done agent is."""

    def terminated(self, obs, action, context, info, rng):
        return info["time_in_env"] >= 50.0

    def reward(self, obs, action, terminated, truncated, context, info, rng):
        return 10.0 if terminated else 0.0

    def cost(self, obs, action, terminated, truncated, context, info, rng):
        return 1.0 if terminated else 0.0


def make_finishing_env(render_mode=None):
    env = _FinishingEnv(
        scenario=_LateScenario(),
        render_mode=render_mode,
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def test_an_agent_done_this_step_gives_its_reward_cost_and_done():
    v = vec_env(make_finishing_env)
    try:
        v.reset(seed=0)
        for _ in range(30):  # past the spawn at 60 s and 50 s more
            _, rewards, terms, _, infos = v.step(
                np.stack([v.action_space.sample() for _ in range(v.num_envs)])
            )
            if terms.any() and rewards.any():
                break
        assert rewards.tolist() == [10.0, 10.0, 10.0]
        assert all(i["_agent_done"] and i["cost"] == 1.0 for i in infos)
    finally:
        v.close()


def test_sb3_sees_each_agent_episode_and_no_padded_one():
    pytest.importorskip("stable_baselines3")
    from bluesky_sandbox.ui.designer.package_files.wrappers import AgentEpisodes  # noqa: PLC0415

    v = AgentEpisodes(sb3_vec_env(make_finishing_env))
    try:
        v.reset()
        episodes = []
        for _ in range(30):  # past the spawn at 60 s and 50 s more
            *_, infos = v.step(np.stack([v.action_space.sample() for _ in range(v.num_envs)]))
            episodes += [i["episode"] for i in infos if "episode" in i]
            if episodes:
                break
    finally:
        v.close()
    # Each aircraft's, spawn to done: 50 s of 5 s steps, its 10 at the end.
    assert [(e["r"], e["l"]) for e in episodes] == [(10.0, 10)] * 3


def test_masked_ppo_learns_only_from_slots_that_hold_an_agent():
    pytest.importorskip("stable_baselines3")
    from bluesky_sandbox.ui.designer.package_files.masked_ppo import MaskedPPO  # noqa: PLC0415
    from bluesky_sandbox.ui.designer.package_files.wrappers import AgentEpisodes  # noqa: PLC0415

    v = AgentEpisodes(sb3_vec_env(make_finishing_env))
    try:
        # 200 s of 5 s steps: the three spawn at 60 s and are done 50 s later,
        # their slots padded before and after.
        model = MaskedPPO("MlpPolicy", v, n_steps=40, batch_size=40, n_epochs=1, seed=0)
        model.learn(120)
        buffer = model.rollout_buffer
        # Live exactly where the observation is an aircraft's, not padding
        # (trained on, the buffer is flattened slot by slot).
        own = np.asarray(buffer.observations).reshape(3, 40, -1).swapaxes(0, 1)
        np.testing.assert_array_equal(buffer.live, np.abs(own).sum(axis=-1) > 0)
        assert 0 < buffer.live.sum() < buffer.live.size
        # The update takes the live transitions and nothing else.
        assert sum(len(batch.actions) for batch in buffer.get(7)) == buffer.live.sum()
        assert model.logger.name_to_value["train/live_fraction"] == pytest.approx(buffer.live.mean())
    finally:
        v.close()


def test_an_env_reporting_no_padding_is_trained_on_whole():
    pytest.importorskip("stable_baselines3")
    from bluesky_sandbox.ui.designer.package_files.masked_ppo import MaskedPPO  # noqa: PLC0415
    from stable_baselines3.common.env_util import make_vec_env  # noqa: PLC0415

    model = MaskedPPO("MlpPolicy", make_vec_env("CartPole-v1", n_envs=2), n_steps=16, batch_size=16, seed=0)
    model.learn(32)
    assert model.rollout_buffer.live.all()
