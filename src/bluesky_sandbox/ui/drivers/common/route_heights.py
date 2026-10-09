"""The height of each point of a route, for drivers that draw routes in 3D.

A fix with an altitude constraint is drawn at it. One without has no height of
its own: it is drawn between the points on either side that have one, by
distance along the route - and, before the first or after the last, level
with it - and marked inferred, so a driver can draw it apart from a
constraint. The designer's map draws routes the same way (``routeHeights``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["route_heights"]


def route_heights(
    points: Sequence[tuple[float, float]],
    alts_ft: Sequence[float | None],
) -> tuple[list[float | None], list[bool]]:
    """Each point's altitude, ft - its own, or inferred - and whether it is
    inferred. With no altitude anywhere along the route, none is inferred:
    every point stays ``None``."""
    along = [0.0]
    for (lat0, lon0), (lat1, lon1) in zip(points, points[1:], strict=False):
        cos_lat = math.cos(math.radians((lat0 + lat1) / 2.0))
        along.append(along[-1] + math.hypot((lon1 - lon0) * cos_lat, lat1 - lat0))
    known = [i for i, a in enumerate(alts_ft) if a is not None]
    if not known:
        return list(alts_ft), [False] * len(alts_ft)
    heights: list[float | None] = []
    for i, alt in enumerate(alts_ft):
        if alt is not None:
            heights.append(float(alt))
            continue
        before = max((k for k in known if k < i), default=None)
        after = min((k for k in known if k > i), default=None)
        if before is None:
            heights.append(float(alts_ft[after]))
        elif after is None:
            heights.append(float(alts_ft[before]))
        else:
            span = along[after] - along[before]
            t = (along[i] - along[before]) / span if span else 0.0
            heights.append(float(alts_ft[before]) + (float(alts_ft[after]) - float(alts_ft[before])) * t)
    return heights, [a is None for a in alts_ft]
