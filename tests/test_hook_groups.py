"""Each environment hook says which stage of the loop it runs in - its
``HookGroup`` - and the designer groups its hook picker by it."""

from __future__ import annotations

from bluesky_sandbox.core.base_environment import HookGroup
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.ui.designer import catalog

E, A, S, C, O, D = (
    HookGroup.EPISODE,
    HookGroup.AIRCRAFT,
    HookGroup.STEP,
    HookGroup.CONTEXT,
    HookGroup.OUTCOME,
    HookGroup.DIAGNOSTICS,
)
STAGES = {
    # Reset.
    "on_episode_loaded": E,
    "on_episode_reset": E,
    # Spawn, and Transition.
    "on_before_spawn": A,
    "on_after_spawn": A,
    "on_aircraft_spawned": A,
    "define_initial_aircraft_control_state": A,
    "on_agent_done": A,
    # Apply actions, and Simulate.
    "on_before_step": S,
    "on_agent_action": S,
    "on_sim_step": S,
    # Observe.
    "on_before_agent_contexts": C,
    "define_agent_context": C,
    # Evaluate.
    **{name: O for name in ("reward", "cost", "terminated", "truncated")},
    **{f"{name}_batch": O for name in ("reward", "cost", "terminated", "truncated")},
    # Outside the loop.
    "define_task_info_providers": D,
    "define_aircraft_readouts": D,
    "define_waypoint_readouts": D,
}


def test_every_hook_runs_in_its_stage():
    hooks = {
        name: getattr(BlueskyEnv, name)
        for name in dir(BlueskyEnv)
        if getattr(getattr(BlueskyEnv, name, None), "__overridable__", False)
    }
    assert {name: fn.__hook_group__ for name, fn in hooks.items()} == STAGES


def test_the_picker_groups_the_hooks_by_stage():
    assert {hook["name"]: hook["category"] for hook in catalog.hooks()} == {
        name: group.value for name, group in STAGES.items()
    }
