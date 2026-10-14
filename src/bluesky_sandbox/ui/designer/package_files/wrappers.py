"""How the env is fitted to Stable-Baselines3: its actions, and the episodes
SB3 logs.

This task's own copy, written by the designer - change it as you like.
``train.py``'s ``make_env`` and ``train`` put these around the env.
"""

from __future__ import annotations

import time

import numpy as np
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper


class MultiDiscreteActions(BaseParallelWrapper):
    """The env's ``Dict`` of a ``binary`` and a ``discrete`` part as one
    ``MultiDiscrete``: each switch a choice of 2, then the choices.

    For a trainer that takes a ``MultiDiscrete`` but no ``Dict`` action space,
    such as Stable-Baselines3. Each action is split back into the parts before
    it reaches the env. A design with continuous actions has no such space:
    its actions cannot be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        space = env.action_space(None)
        kinds = {"binary": MultiBinary, "discrete": MultiDiscrete}
        if not (
            isinstance(space, Dict)
            and all(isinstance(part, kinds.get(key, ())) for key, part in space.spaces.items())
        ):
            raise TypeError(
                "MultiDiscreteActions takes an action space of switches and choices "
                f"only, got {space!r}."
            )
        binary = space.spaces.get("binary")
        discrete = space.spaces.get("discrete")
        self._n_binary = 0 if binary is None else int(np.prod(binary.shape))
        nvec = [2] * self._n_binary + ([] if discrete is None else list(discrete.nvec))
        self._parts = list(space.spaces)
        self._space = MultiDiscrete(nvec)

    def action_space(self, agent):
        return self._space

    def action_layout(self, agent=None):
        layout = self.env.action_layout(agent)
        return {"discrete": [*layout.get("binary", []), *layout.get("discrete", [])]}

    def _parts_of(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        split = {
            "binary": action[: self._n_binary].astype(np.int8),
            "discrete": action[self._n_binary :],
        }
        return {part: split[part] for part in self._parts}

    def step(self, actions):
        return self.env.step({agent: self._parts_of(a) for agent, a in actions.items()})


class AgentEpisodes(VecEnvWrapper):
    """Each agent's episode - its life, spawn to done - in its info as SB3's
    ``Monitor`` puts one (``info["episode"]``: its return ``r``, its steps
    ``l``, and the seconds since this was made ``t``), so SB3 logs
    ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean``. The agent slots of
    agents not yet spawned or already done (``_padded``) are no episode."""

    def __init__(self, venv):
        super().__init__(venv)
        self._start = time.time()
        self._returns = np.zeros(self.num_envs)
        self._lengths = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        self._returns[:] = 0.0
        self._lengths[:] = 0
        observations = self.venv.reset()
        # The reset's infos - which slots hold an agent - for the trainer.
        self.reset_infos = list(getattr(self.venv, "reset_infos", self.reset_infos))
        return observations

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        infos = list(infos)
        for i, info in enumerate(infos):
            if info.get("_padded", False):
                continue
            self._returns[i] += float(rewards[i])
            self._lengths[i] += 1
            if info.get("_agent_done", bool(dones[i])):
                infos[i] = {
                    **info,
                    "episode": {
                        "r": float(self._returns[i]),
                        "l": int(self._lengths[i]),
                        "t": round(time.time() - self._start, 6),
                    },
                }
                self._returns[i] = 0.0
                self._lengths[i] = 0
        return observations, rewards, dones, infos
