"""``EnvOptions`` names every keyword ``BlueskyEnv`` passes on beside its config
and scenario, so a task env forwarding ``**options: Unpack[EnvOptions]`` is
typed for each."""

from __future__ import annotations

import inspect

from bluesky_sandbox import EnvOptions
from bluesky_sandbox.core.base_environment import BlueskyBaseEnvironment


def test_the_options_are_the_base_environments_keywords():
    taken = set(inspect.signature(BlueskyBaseEnvironment.__init__).parameters) - {"self", "config", "scenario"}
    assert set(EnvOptions.__annotations__) == taken
