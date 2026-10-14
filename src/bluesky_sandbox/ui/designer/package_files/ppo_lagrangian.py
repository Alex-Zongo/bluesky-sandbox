"""PPO-Lagrangian: PPO kept under a cost limit.

This task's own copy, written by the designer - change it as you like - on
:class:`.masked_ppo.MaskedPPO`, its other file. ``train.py`` trains with it,
and ``evaluate.py`` loads what it saved.

Each agent's info carries its step's ``cost`` - a design's ``cost`` hook. The
constraint is on an agent's episode cost, its expected sum ``J_c``::

    maximize J_r  subject to  J_c <= cost_limit

solved at the saddle point of the Lagrangian ``J_r - lambda (J_c - cost_limit)``:

* a cost critic ``V_c`` - its own network, trained on the cost returns as the
  value function is on the rewards - gives each step a cost advantage ``A_c``,
  by GAE as ``A_r`` is;
* PPO updates the policy on ``(A_r - lambda A_c) / (1 + lambda)``, as
  OmniSafe's PPOLag does: over the rollout, ``A_r`` standardized - its mean
  taken off, divided by its standard deviation - and ``A_c`` only centered,
  its mean taken off; no other normalization (``normalize_advantage`` off);
* ``lambda`` rises while the episodes' cost is over the limit and falls while
  it is under, never below 0: gradient ascent on the Lagrangian, by Adam. Adam
  scales each step by the gradient's own running size, so ``lambda_lr`` moves
  ``lambda`` at about the same rate whatever the cost's units - seconds,
  counts or a weighted sum.

An agent's episode is its life, spawn to done: the stable-ID slots
(``_padded`` in their info) of agents not yet spawned or already done are no
episode of any agent's.
"""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import numpy as np
import torch as th
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
from torch import nn
from torch.nn import functional as F

from .masked_ppo import MaskedPPO

__all__ = ["CostActorCriticPolicy", "CostMultiInputActorCriticPolicy", "PPOLagrangian"]


class _CostCritic(nn.Module):
    """``V_c``: an observation's expected cost-to-go."""

    def __init__(self, extractor: nn.Module, head: nn.Module, observation_space, normalize_images: bool):
        super().__init__()
        self.extractor = extractor
        self.head = head
        self.observation_space = observation_space
        self.normalize_images = normalize_images

    def forward(self, obs) -> th.Tensor:
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.head(self.extractor(obs))


class _CostCriticPolicy:
    """An actor-critic policy with a cost critic beside its value function:
    its own features extractor (of the policy's class) and layers (the value
    function's), with its own optimizer - training it moves nothing PPO
    trains."""

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Built after the policy's optimizer, so outside it.
        extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        arch = self.net_arch.get("vf", []) if isinstance(self.net_arch, dict) else list(self.net_arch)
        layers: list[nn.Module] = []
        width = extractor.features_dim
        for size in arch:
            layers += [nn.Linear(width, size), self.activation_fn()]
            width = size
        out = nn.Linear(width, 1)
        if self.ortho_init:
            extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for layer in layers:
                layer.apply(partial(self.init_weights, gain=np.sqrt(2)))
            out.apply(partial(self.init_weights, gain=1))
        self.cost_critic = _CostCritic(
            extractor, nn.Sequential(*layers, out), self.observation_space, self.normalize_images
        )
        self.cost_optimizer = self.optimizer_class(
            self.cost_critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def predict_cost_values(self, obs) -> th.Tensor:
        """``V_c`` of each observation."""
        return self.cost_critic(obs)


class CostActorCriticPolicy(_CostCriticPolicy, ActorCriticPolicy):
    """``MlpPolicy`` with a cost critic."""


class CostMultiInputActorCriticPolicy(_CostCriticPolicy, MultiInputActorCriticPolicy):
    """``MultiInputPolicy`` with a cost critic."""


def _step_cost(info: dict[str, Any]) -> float:
    cost = np.asarray(info.get("cost", 0.0), dtype=np.float64)
    if cost.size != 1:
        raise ValueError(
            f"PPOLagrangian keeps one cost under its limit; this env's has {cost.size}, "
            "one per constraint. Return their sum, or one weighted to the limit."
        )
    return float(cost.reshape(()))


def _rows(obs, start: int, stop: int):
    """Rows ``start:stop`` of observations flattened to one batch dim."""
    if isinstance(obs, dict):
        return {key: value[start:stop] for key, value in obs.items()}
    return obs[start:stop]


def combined_advantages(
    reward: np.ndarray, cost: np.ndarray, live: np.ndarray, lam: float
) -> np.ndarray:
    """``(A_r - lam A_c) / (1 + lam)``, as OmniSafe's PPOLag combines them:
    ``A_r`` standardized and ``A_c`` centered over the rollout - here over its
    live transitions, the ones trained on."""
    if not live.any():
        return reward.astype(np.float32)
    reward = (reward - reward[live].mean()) / (reward[live].std() + 1e-8)
    cost = cost - cost[live].mean()
    return ((reward - lam * cost) / (1.0 + lam)).astype(np.float32)


def _flat(obs):
    """A rollout's observations, (steps, envs, ...) as (steps * envs, ...)."""
    if isinstance(obs, dict):
        return {key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}
    return obs.reshape(-1, *obs.shape[2:])


class _Multiplier(nn.Module):
    """The Lagrange multiplier, a parameter Adam steps."""

    def __init__(self, initial: float) -> None:
        super().__init__()
        self.value = nn.Parameter(th.tensor(float(initial)))


class PPOLagrangian(MaskedPPO):
    """PPO with each agent's episode cost kept under ``cost_limit``, trained
    only on the slots that hold an agent (:class:`MaskedPPO`).

    ``lambda_lr`` is Adam's learning rate on the multiplier, about how far it
    moves each update; ``initial_lambda`` starts it, and ``max_lambda`` caps
    it. ``cost_gamma`` discounts the cost as
    ``gamma`` does the reward (``None``: ``gamma``). The rest is PPO's.

    Logs ``lagrangian/lambda``, ``lagrangian/episode_cost`` (the mean cost of
    the episodes the rollout finished: ``J_c``) and ``train/cost_value_loss``.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": CostActorCriticPolicy,
        "MultiInputPolicy": CostMultiInputActorCriticPolicy,
    }
    policy: _CostCriticPolicy

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env,
        cost_limit: float = 0.0,
        lambda_lr: float = 0.035,
        initial_lambda: float = 0.0,
        max_lambda: float | None = None,
        cost_gamma: float | None = None,
        **kwargs: Any,
    ) -> None:
        if cost_limit < 0:
            raise ValueError(f"cost_limit must be at least 0, got {cost_limit}.")
        if initial_lambda < 0:
            raise ValueError(f"initial_lambda must be at least 0, got {initial_lambda}.")
        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.initial_lambda = float(initial_lambda)
        self.max_lambda = max_lambda
        self.cost_gamma = cost_gamma
        # The combined advantage is PPO's as it is, as in OmniSafe: no
        # normalizing it again per minibatch.
        kwargs.setdefault("normalize_advantage", False)
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.policy, _CostCriticPolicy):
            raise TypeError(
                "PPOLagrangian needs a policy with a cost critic: CostActorCriticPolicy "
                f"or CostMultiInputActorCriticPolicy, got {type(self.policy).__name__}."
            )
        self._episode_costs = np.zeros(self.n_envs)
        self._finished_costs: list[float] = []
        self.multiplier = _Multiplier(self.initial_lambda)
        self.lambda_optimizer = th.optim.Adam(self.multiplier.parameters(), lr=self.lambda_lr)

    @property
    def lagrangian_lambda(self) -> float:
        """The Lagrange multiplier, ``lambda``."""
        return float(self.multiplier.value)

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "_episode_costs",
            "_finished_costs",
            "_costs",
            "_cost_advantages",
            "_cost_returns",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return [*state_dicts, "policy.cost_optimizer", "multiplier", "lambda_optimizer"], variables

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1,
              tb_log_name: str = "PPOLagrangian", reset_num_timesteps: bool = True,
              progress_bar: bool = False):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------ rollouts
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        self._costs = np.zeros((n_rollout_steps, env.num_envs), dtype=np.float32)
        if not super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps):
            return False
        self._cost_advantages, self._cost_returns = self._cost_gae(rollout_buffer)
        return True

    def _record_step(self, locals_: dict[str, Any]) -> None:
        super()._record_step(locals_)
        step, infos, dones = locals_["n_steps"], locals_["infos"], locals_["dones"]
        gamma = self.gamma if self.cost_gamma is None else self.cost_gamma
        for i, info in enumerate(infos):
            cost = _step_cost(info)
            self._costs[step, i] = cost
            # Cut off by a time limit: what it would have cost from there on.
            if (
                dones[i]
                and info.get("terminal_observation") is not None
                and info.get("TimeLimit.truncated", False)
            ):
                terminal = self.policy.obs_to_tensor(info["terminal_observation"])[0]
                with th.no_grad():
                    self._costs[step, i] += gamma * float(self.policy.predict_cost_values(terminal)[0])
            if info.get("_padded", False):
                continue
            self._episode_costs[i] += cost
            if info.get("_agent_done", bool(dones[i])):
                self._finished_costs.append(float(self._episode_costs[i]))
                self._episode_costs[i] = 0.0

    def _cost_gae(self, buffer) -> tuple[np.ndarray, np.ndarray]:
        """The rollout's cost advantages and returns, by GAE as the reward's."""
        gamma = self.gamma if self.cost_gamma is None else self.cost_gamma
        n_steps, n_envs = self._costs.shape
        obs = _flat(buffer.observations)
        values = np.empty(n_steps * n_envs, dtype=np.float32)
        with th.no_grad():
            for start in range(0, n_steps * n_envs, self.batch_size):
                stop = start + self.batch_size
                batch = obs_as_tensor(_rows(obs, start, stop), self.device)
                values[start:stop] = self.policy.predict_cost_values(batch).cpu().numpy().flatten()
            last = self.policy.predict_cost_values(obs_as_tensor(self._last_obs, self.device))
        values = values.reshape(n_steps, n_envs)
        last_values = last.cpu().numpy().flatten()
        advantages = np.zeros_like(values)
        gae = np.zeros(n_envs, dtype=np.float32)
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                non_terminal = 1.0 - self._last_episode_starts.astype(np.float32)
                next_values = last_values
            else:
                non_terminal = 1.0 - buffer.episode_starts[step + 1]
                next_values = values[step + 1]
            delta = self._costs[step] + gamma * next_values * non_terminal - values[step]
            gae = delta + gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae
        return advantages, advantages + values

    # ------------------------------------------------------------ updates
    def train(self) -> None:
        self._update_lambda()
        buffer = self.rollout_buffer
        buffer.advantages = combined_advantages(
            buffer.advantages, self._cost_advantages, buffer.live, self.lagrangian_lambda
        )
        # Before PPO's update, while the rollout is still (steps, envs): the cost
        # critic shares nothing with what PPO trains.
        self._train_cost_critic(buffer)
        super().train()

    def _update_lambda(self) -> None:
        """Adam's step on ``lambda``, from the episodes the rollout finished:
        up while ``J_c`` is over the limit, down while under, kept between 0
        and ``max_lambda``. With none finished, ``lambda`` stays."""
        finished, self._finished_costs = self._finished_costs, []
        self.logger.record("lagrangian/episodes", len(finished))
        if finished:
            episode_cost = float(np.mean(finished))
            # The Lagrangian's gradient in lambda is J_c - cost_limit: ascend it.
            self.lambda_optimizer.zero_grad()
            loss = -self.multiplier.value * (episode_cost - self.cost_limit)
            loss.backward()
            self.lambda_optimizer.step()
            with th.no_grad():
                self.multiplier.value.clamp_(min=0.0, max=self.max_lambda)
            self.logger.record("lagrangian/episode_cost", episode_cost)
        self.logger.record("lagrangian/lambda", self.lagrangian_lambda)

    def _train_cost_critic(self, buffer) -> None:
        """``V_c`` toward the rollout's cost returns: PPO's epochs and batches."""
        self.policy.set_training_mode(True)
        optimizer = self.policy.cost_optimizer
        self._update_learning_rate(optimizer)
        obs = _flat(buffer.observations)
        returns = th.as_tensor(self._cost_returns.reshape(-1), device=self.device)
        # The live transitions alone, as PPO's update takes them; still (steps,
        # envs) here, flattened step by step.
        live = np.flatnonzero(buffer.live.reshape(-1))
        if not len(live):
            return
        losses = []
        for _ in range(self.n_epochs):
            order = np.random.permutation(live)
            for start in range(0, len(order), self.batch_size):
                rows = order[start : start + self.batch_size]
                batch = {k: v[rows] for k, v in obs.items()} if isinstance(obs, dict) else obs[rows]
                predicted = self.policy.predict_cost_values(obs_as_tensor(batch, self.device)).flatten()
                loss = F.mse_loss(predicted, returns[rows])
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
                optimizer.step()
                losses.append(loss.item())
        # Its gradients would count in PPO's clipping of the policy's.
        optimizer.zero_grad(set_to_none=True)
        self.logger.record("train/cost_value_loss", float(np.mean(losses)))
