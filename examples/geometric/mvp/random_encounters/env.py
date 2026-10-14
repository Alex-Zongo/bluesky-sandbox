"""Environment wrapper for this generated task."""

from __future__ import annotations

from bluesky_sandbox.env import BlueskyEnv

from .config import CONFIG
from .scenario import RandomEncountersScenario
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluesky_sandbox import BaseAgentInfo, BaseObs
    from numpy import ndarray
    from numpy.random import Generator
    from .task_types import TaskAgentStepContext


class RandomEncountersEnv(BlueskyEnv):
    """Generated environment class with task behavior in Python code."""

    def __init__(
        self,
        *,
        scenario: RandomEncountersScenario | None = None,
        render_mode=None,
        realtime: bool = False,
        views=None,
        frame_driver=None,
    ) -> None:
        scenario = RandomEncountersScenario() if scenario is None else scenario
        super().__init__(
            config=CONFIG,
            scenario=scenario,
            render_mode=render_mode,
            realtime=realtime,
            views=views,
            frame_driver=frame_driver,
        )

    def define_task_info_providers(self):
        return ()

    # Env hooks. reward / terminated / truncated are always present; other
    # @overridable hooks appear only when customized (else inherit the base).
    def reward(
        self,
        obs: BaseObs,
        action: ndarray | None,
        terminated: bool,
        truncated: bool,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: Generator,
    ) -> float | ndarray:
        # Every minute flown costs 0.1, every second in loss of separation 1:
        # through the encounter safely, and without a long detour.
        return -0.1 * self.config.dt / 60.0 - float(context.separation.los.time.during_step_s)

    def terminated(
        self,
        obs: BaseObs,
        action: ndarray | None,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: Generator,
    ) -> bool:
        # Out of the airspace: through the encounter.
        return not info["in_airspace"]

    def truncated(
        self,
        obs: BaseObs,
        action: ndarray | None,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: Generator,
    ) -> bool:
        return False

    def cost(
        self,
        obs: BaseObs,
        action: ndarray | None,
        terminated: bool,
        truncated: bool,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: Generator,
    ) -> float | ndarray:
        # Seconds in loss of separation this step.
        return float(context.separation.los.time.during_step_s)


def main() -> None:
    """Smoke rollout: print the task spec, then step with random actions.

    This is a multi-agent (PettingZoo parallel) environment - ``step`` takes a
    dict of per-agent actions and returns dicts keyed by agent.
    """
    env = RandomEncountersEnv(render_mode="pygame")
    env.reset(seed=0)
    if env.render_mode is not None:
        env.render()

    # Task spec: the agent set and per-agent observation / action spaces.
    print(f"agents ({len(env.agents)}): {list(env.agents)}")
    for agent in env.agents[:1]:
        print(f"observation_space[{agent}]: {env.observation_space(agent)}")
        print(f"action_space[{agent}]:      {env.action_space(agent)}")

    import numpy as np

    steps = 0
    total_reward = 0.0
    total_cost = 0.0
    for _ in range(100):
        if env.episode_done:
            env.reset()
            if env.render_mode is not None:
                env.render()
            continue
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        # This design defines a cost: each agent's is in its info.
        costs = {agent: info["cost"] for agent, info in infos.items()}
        # A reward or cost may be a number or an array of components.
        total_reward += sum(float(np.sum(r)) for r in rewards.values())
        total_cost += sum(float(np.sum(c)) for c in costs.values())
        if env.render_mode is not None:
            env.render()
        steps += 1
    env.close()
    print(f"ran {steps} steps OK: total reward {total_reward:.3f}, total cost {total_cost:.3f}")
