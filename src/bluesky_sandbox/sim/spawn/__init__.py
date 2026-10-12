"""Spawn configuration: where aircraft appear and what route they fly.

Split into :mod:`~bluesky_sandbox.sim.spawn.routes` (the route-spec grammar)
and :mod:`~bluesky_sandbox.sim.spawn.regions` (spawn volumes and the episode
spawn queue); everything is re-exported
here, so the import path is unchanged.
"""

from __future__ import annotations

from .heading import TowardRoute, route_heading
from .regions import (
    SpawnConfig,
    SpawnRegion,
    param_alt_range,
)
from .sources import (
    Mixture,
    PlanContext,
    PlannedSource,
    PlannedSpawn,
    RegionSource,
    Replay,
    SpawnRequest,
    SpawnSource,
    nearest_entry,
)
from .routes import (
    DEFAULT_ROUTE_COLORS,
    Route,
    RouteSpec,
    RouteStep,
    expand_route_paths,
    map_route_steps,
    resolve_route,
    route_colors,
    route_steps,
    route_step_name,
    route_step_names,
    sample_route_path,
)

__all__ = [
    "Mixture",
    "PlanContext",
    "PlannedSource",
    "PlannedSpawn",
    "RegionSource",
    "Replay",
    "SpawnRequest",
    "SpawnSource",
    "TowardRoute",
    "nearest_entry",
    "RouteSpec",
    "RouteStep",
    "SpawnConfig",
    "SpawnRegion",
    "expand_route_paths",
    "DEFAULT_ROUTE_COLORS",
    "Route",
    "map_route_steps",
    "route_colors",
    "route_steps",
    "param_alt_range",
    "resolve_route",
    "route_heading",
    "route_step_name",
    "route_step_names",
    "sample_route_path",
]
