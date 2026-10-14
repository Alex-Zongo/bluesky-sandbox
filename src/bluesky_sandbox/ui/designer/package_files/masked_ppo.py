"""PPO that learns only from the agent slots that hold an agent.

This task's own copy, written by the designer - change it as you like.
``train.py`` trains with it, and ``evaluate.py`` loads what it saved.

A multi-agent env made a vector env has a fixed pool of agent slots, and a
slot with no agent in it - not yet spawned, or already done - still steps,
padded: a zero observation, a zero reward. SB3's PPO would train on those as
on any other transition: work spent on padding, and value targets pulled
toward zero.

:class:`MaskedPPO` reads each slot's ``info["_next_padded"]`` - whether the
observation returned for it is padding, as
:class:`~bluesky_sandbox.interface.wrappers.StableIDsParallelWrapper` reports -
and leaves those transitions out of its update. An env that reports none is
trained on whole, as SB3's PPO would.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

__all__ = ["LiveDictRolloutBuffer", "LiveRolloutBuffer", "MaskedPPO"]


class _LiveRows:
    """A rollout buffer that yields only its live transitions: ``live``,
    (steps, envs), is True where the observation held an agent."""

    def reset(self) -> None:
        super().reset()
        self.live = np.ones((self.buffer_size, self.n_envs), dtype=bool)

    def live_rows(self) -> np.ndarray:
        """The live transitions' indices in the buffer once flattened - env by
        env, as SB3 flattens it."""
        return np.flatnonzero(self.live.swapaxes(0, 1).reshape(-1))

    def get(self, batch_size: int | None = None):
        assert self.full, ""
        if not self.generator_ready:
            # Flattens the rollout as SB3 does; the one sample it gives is not used.
            next(super().get(batch_size=1))
        rows = np.random.permutation(self.live_rows())
        if batch_size is None:
            batch_size = len(rows)
        for start in range(0, len(rows), batch_size):
            yield self._get_samples(rows[start : start + batch_size])


class LiveRolloutBuffer(_LiveRows, RolloutBuffer):
    """SB3's ``RolloutBuffer``, yielding only live transitions."""


class LiveDictRolloutBuffer(_LiveRows, DictRolloutBuffer):
    """SB3's ``DictRolloutBuffer``, yielding only live transitions."""


class _Recording(BaseCallback):
    """The rollout's callback, with each step recorded before it."""

    def __init__(self, inner: BaseCallback, record) -> None:
        super().__init__()
        self.inner = inner
        self.record = record

    def update_locals(self, locals_: dict[str, Any]) -> None:
        self.locals = locals_
        self.inner.update_locals(locals_)

    def on_rollout_start(self) -> None:
        self.inner.on_rollout_start()

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        self.record(self.locals)
        return self.inner.on_step()

    def on_rollout_end(self) -> None:
        self.inner.on_rollout_end()


def _next_live(infos) -> np.ndarray:
    """Whether each slot's next observation holds an agent."""
    return np.array([not info.get("_next_padded", False) for info in infos], dtype=bool)


class MaskedPPO(PPO):
    """PPO trained only on the transitions of slots that hold an agent.

    Logs ``train/live_fraction``, the share of the rollout trained on, and
    computes ``train/explained_variance`` over those transitions alone.
    """

    def _setup_model(self) -> None:
        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = (
                LiveDictRolloutBuffer
                if isinstance(self.observation_space, spaces.Dict)
                else LiveRolloutBuffer
            )
        super()._setup_model()
        if not issubclass(self.rollout_buffer_class, _LiveRows):
            raise TypeError(
                f"{type(self).__name__} needs a rollout buffer that yields only live "
                f"transitions (LiveRolloutBuffer), got {self.rollout_buffer_class.__name__}."
            )
        self._obs_live = np.ones(self.n_envs, dtype=bool)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_obs_live"]

    def _setup_learn(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        resetting = reset_num_timesteps or self._last_obs is None
        out = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        if resetting:
            # The env was just reset: which first observations hold an agent.
            infos = getattr(self.env, "reset_infos", None) or []
            self._obs_live = (
                _next_live(infos) if len(infos) == self.n_envs else np.ones(self.n_envs, dtype=bool)
            )
        return out

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        return super().collect_rollouts(
            env, _Recording(callback, self._record_step), rollout_buffer, n_rollout_steps
        )

    def _record_step(self, locals_: dict[str, Any]) -> None:
        """Whether the step's observations held an agent; then the next ones'."""
        locals_["rollout_buffer"].live[locals_["n_steps"]] = self._obs_live
        self._obs_live = _next_live(locals_["infos"])

    def train(self) -> None:
        buffer = self.rollout_buffer
        live = buffer.live.mean()
        self.logger.record("train/live_fraction", float(live))
        if not live:
            # Nothing to learn from: every slot was empty the whole rollout.
            return
        super().train()
        # SB3's, over the live transitions alone: padding is easy to predict.
        rows = buffer.live_rows()
        self.logger.record(
            "train/explained_variance",
            float(explained_variance(buffer.values.flatten()[rows], buffer.returns.flatten()[rows])),
        )
