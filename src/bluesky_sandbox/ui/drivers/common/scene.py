"""What a driver frames: the airspace, or - a design without one - what the
design places, where it spawns and its waypoints."""

from __future__ import annotations

from typing import Any

__all__ = ["scene_bounds"]


def scene_bounds(env: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((lat_min, lat_max), (lon_min, lon_max))`` to frame: the airspace's
    bounding box, else the spawn regions' with every waypoint in - a spawn may
    be one point, on a fix, so the spawn alone may have no extent."""
    airspace = env.episode_airspace_bounds
    if airspace is not None:
        return airspace.bounding_box
    spawn_bb = env.episode_spawn.resolved_bounds
    lats, lons = list(spawn_bb["lat_deg"]), list(spawn_bb["lon_deg"])
    for queryable in (env.episode_queryables or {}).values():
        lat, lon = getattr(queryable, "lat", None), getattr(queryable, "lon", None)
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))
    return (min(lats), max(lats)), (min(lons), max(lons))
