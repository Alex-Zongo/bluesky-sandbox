"""Environment wrapper for this generated task."""

from __future__ import annotations

from typing import Unpack

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv, EnvOptions

from .config import CONFIG
from .scenario import PointMergeScenario
from bluesky_sandbox import (
    AircraftControlState,
)
from .setup import (
    CLEARANCE_COST,
    LANDED_NM,
    SHAPING_GAMMA,
    TIME_COST_PER_MIN,
    bs,
    kts,
    np,
    potential,
    slow_from_m,
    spacing_score,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from bluesky_sandbox import AircraftControlState, BaseAgentInfo, BaseObs
    from collections.abc import Mapping
    from typing import Any
    from .task_types import TaskAgentStepContext


class PointMergeEnvKwargs(EnvOptions, total=False):
    """What :class:`PointMergeEnv` takes, as keywords - to pass on, typed,
    with ``**env_kwargs: Unpack[PointMergeEnvKwargs]``."""

    config: EnvConfig
    scenario: PointMergeScenario


class PointMergeEnv(BlueskyEnv):
    """Generated environment class with task behavior in Python code.

    ``config`` defaults to this task's (``CONFIG``) - pass a changed copy,
    ``dataclasses.replace(CONFIG, dt=5.0)``, to run it otherwise; ``options``
    are how it is shown and paced (``render_mode``, ``views``, ...)."""

    def __init__(
        self,
        *,
        config: EnvConfig = CONFIG,
        scenario: PointMergeScenario | None = None,
        **options: Unpack[EnvOptions],
    ) -> None:
        scenario = PointMergeScenario() if scenario is None else scenario
        super().__init__(config=config, scenario=scenario, **options)

    def define_task_info_providers(self):
        return ()

    # Env hooks. reward / terminated / truncated are always present; other
    # @overridable hooks appear only when customized (else inherit the base).
    def reward(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        terminated: bool,
        truncated: bool,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> float | np.ndarray:
        # Delay: every step flown.
        r = -TIME_COST_PER_MIN * self.config.dt / 60.0
        # Workload: a speed reduction given this step.
        if context.applied["ap_spd_delta_kts"]:
            r -= CLEARANCE_COST
        # Spacing: scored once, as the aircraft crosses VALPU.
        if terminated:
            now = float(bs.sim.simt)
            gap = None if self.last_over_merge_s is None else now - self.last_over_merge_s
            self.last_over_merge_s = now
            r += spacing_score(gap)
        # Shaping: the potential's change, with the potential 0 at the flight's end.
        now = 0.0 if terminated else potential(context.acidx, self.config.lookahead_s)
        r += SHAPING_GAMMA * now - self.potentials.get(context.acid, 0.0)
        self.potentials[context.acid] = now
        return r

    def terminated(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> bool:
        # The flight ends at VALPU, the IAF, where the approach takes over.
        return context.query("VALPU").step.reached

    def truncated(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> bool:
        # Never: every flight reaches VALPU.
        return False

    def cost(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        terminated: bool,
        truncated: bool,
        context: TaskAgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> float | np.ndarray:
        # Seconds in loss of separation this step: the constrained quantity (limit 0).
        return float(context.separation.los.time.during_step_s)

    def on_agent_done(
        self,
        acid: str,
        info: BaseAgentInfo,
        terminated: bool,
        truncated: bool,
    ) -> AircraftControlState:
        # Past VALPU the approach takes over: the aircraft flies on as background
        # traffic that later arrivals must stay clear of. Any other end removes it.
        return AircraftControlState.BACKGROUND if terminated else AircraftControlState.DELETE

    def on_aircraft_spawned(self, callsign: str, route: list[str] | None) -> None:
        # BlueSky leaves VNAV off for a route given after creation: switch it on to
        # fly the published altitudes. Clearing the current speed turns VNAV speed off,
        # since on_sim_step manages speed instead: VNAV crosses fixes too fast.
        bs.stack.stack(f"VNAV {callsign} ON")
        idx = bs.traf.id.index(callsign)
        bs.stack.stack(f"SPD {callsign} {bs.traf.cas[idx] / kts:.0f}")

    def on_episode_reset(
        self,
        seed: int | None,
        options: Mapping[str, Any] | None,
    ) -> None:
        # Per-episode state: when the last aircraft crossed VALPU, for the spacing
        # score, and each aircraft's last potential, for the shaping.
        self.last_over_merge_s = None
        self.potentials = {}

    def on_sim_step(self) -> None:
        # Every simulator substep, so neither landing nor slowing is late by a step.
        for idx, acid in enumerate(bs.traf.id):
            context = self.agent_context(idx)
            # At the threshold the aircraft has landed.
            if context.query("RW01L").current.distance_nm < LANDED_NM:
                self.mark_aircraft_for_deletion(acid)
                continue
            # Cross the active fix no faster than its published speed, slowing just in
            # time; a lower cleared speed is kept.
            active = context.route.active
            published = None if active is None else active.crossing.speed_kts
            if (
                published is not None
                and bs.traf.selspd[idx] / kts > published + 0.5
                and bs.traf.ap.dist2wp[idx] < slow_from_m(idx, published)
            ):
                bs.stack.stack(f"SPD {acid} {published:.0f}")


def main() -> None:
    """Smoke rollout: print the task spec, then step with random actions.

    This is a multi-agent (PettingZoo parallel) environment - ``step`` takes a
    dict of per-agent actions and returns dicts keyed by agent.
    """
    env = PointMergeEnv(render_mode="pygame")
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
