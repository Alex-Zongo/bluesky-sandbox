"""PPO-Lagrangian: PPO kept under a cost limit by a multiplier that rises while
the episodes cost too much, and a cost critic of its own."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from bluesky_sandbox.ui.designer.package_files.ppo_lagrangian import (  # noqa: E402
    PPOLagrangian,
    combined_advantages,
)

STEPS = 20  # an episode's length


class _Tempting(gym.Env):
    """Each step, action 1 earns 1 and costs 1; action 0 earns and costs
    nothing. Unconstrained, every step is a 1: an episode's cost is 20."""

    observation_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, cost_shape=()):
        self.cost_shape = cost_shape

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        took = float(action)
        info = {"cost": np.full(self.cost_shape, took) if self.cost_shape else took}
        obs = np.array([self.t / STEPS], np.float32)
        return obs, took, self.t >= STEPS, False, info


def _model(cost_limit: float, **kwargs) -> PPOLagrangian:
    vec = DummyVecEnv([_Tempting] * 4)
    kwargs = {"n_steps": 64, "batch_size": 64, "n_epochs": 4, "seed": 0, **kwargs}
    return PPOLagrangian("MlpPolicy", vec, cost_limit=cost_limit, **kwargs)


def _episode_cost(model) -> float:
    env = _Tempting()
    obs, _ = env.reset()
    total = done = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        total += info["cost"]
    return total


def test_over_its_limit_the_policy_is_held_under_it():
    # Unconstrained it would cost 20; the multiplier rises until it does not.
    model = _model(cost_limit=5.0, lambda_lr=0.5).learn(3_000)
    assert model.lagrangian_lambda > 0
    assert _episode_cost(model) <= 8


def test_under_its_limit_the_multiplier_stays_zero():
    model = _model(cost_limit=STEPS).learn(2_000)
    assert model.lagrangian_lambda == 0.0
    assert _episode_cost(model) == STEPS  # nothing holds it back


def _stepped(model, *episode_costs) -> float:
    """The multiplier after one update on these finished episodes."""
    model.set_logger(configure(None, []))
    model._finished_costs = list(episode_costs)
    model._update_lambda()
    return model.lagrangian_lambda


def test_the_multiplier_moves_at_its_rate_whatever_the_cost_units():
    # Adam's step is about lambda_lr, 5 over the limit or 500 over.
    a = _stepped(_model(cost_limit=5.0, lambda_lr=0.5), 9.0, 11.0)
    b = _stepped(_model(cost_limit=5.0, lambda_lr=0.5), 505.0)
    assert a == pytest.approx(0.5, rel=1e-3) and b == pytest.approx(a, rel=1e-3)


def test_the_multiplier_is_capped_and_never_below_zero():
    model = _model(cost_limit=50.0, lambda_lr=2.0, max_lambda=3.0)
    for _ in range(5):
        _stepped(model, 95.0)  # 45 over
    assert model.lagrangian_lambda == 3.0
    for _ in range(20):
        _stepped(model, 5.0)  # 45 under
    assert model.lagrangian_lambda == 0.0


def test_the_cost_critic_learns_the_cost_to_go():
    # Unconstrained, every step is a 1: its cost is its reward, so the cost
    # critic learns what the value function does.
    model = _model(cost_limit=STEPS, n_epochs=10, learning_rate=1e-3).learn(6_000)
    for t in (0, 6, 12, 18):
        obs = model.policy.obs_to_tensor(np.array([t / STEPS], np.float32))[0]
        with th.no_grad():
            cost_to_go = model.policy.predict_cost_values(obs).item()
            value = model.policy.predict_values(obs).item()
        assert cost_to_go == pytest.approx(value, abs=1.0)
    assert model.policy.predict_cost_values(model.policy.obs_to_tensor(np.zeros(1, np.float32))[0]).item() > 15


def test_a_saved_model_loads_with_its_multiplier_and_cost_critic(tmp_path):
    model = _model(cost_limit=1.0).learn(1_000)
    model.save(tmp_path / "lag")
    again = PPOLagrangian.load(tmp_path / "lag")
    assert again.lagrangian_lambda == model.lagrangian_lambda
    for a, b in zip(model.policy.cost_critic.parameters(), again.policy.cost_critic.parameters()):
        th.testing.assert_close(a.detach().cpu(), b.detach().cpu())


def test_one_cost_per_constraint_is_refused():
    model = PPOLagrangian(
        "MlpPolicy", DummyVecEnv([lambda: _Tempting(cost_shape=(2,))]), n_steps=8, batch_size=8
    )
    with pytest.raises(ValueError, match="one per constraint"):
        model.learn(8)


def test_its_tensorboard_runs_are_named_for_it(tmp_path):
    _model(cost_limit=1.0, tensorboard_log=str(tmp_path)).learn(256)
    assert [p.name for p in tmp_path.iterdir()] == ["PPOLagrangian_1"]


def test_the_advantages_combine_as_omnisafe_combines_them():
    rng = np.random.default_rng(0)
    reward, cost = rng.normal(3, 2, (8, 4)), rng.normal(40, 15, (8, 4))
    live = rng.random((8, 4)) > 0.3
    out = combined_advantages(reward, cost, live, lam=0.5)
    # Over the live transitions: A_r standardized, A_c only centered.
    a_r = (reward - reward[live].mean()) / (reward[live].std() + 1e-8)
    a_c = cost - cost[live].mean()
    np.testing.assert_allclose(out, (a_r - 0.5 * a_c) / 1.5, rtol=1e-5)
    assert out.dtype == np.float32
    # At lambda 0, the reward's alone - standardized.
    assert abs(combined_advantages(reward, cost, live, 0.0)[live].std() - 1.0) < 1e-5


def test_it_does_not_normalize_the_combined_advantage_again():
    assert _model(cost_limit=1.0).normalize_advantage is False
    assert _model(cost_limit=1.0, normalize_advantage=True).normalize_advantage is True
