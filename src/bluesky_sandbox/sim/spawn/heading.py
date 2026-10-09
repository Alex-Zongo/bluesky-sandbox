"""A spawn heading taken from the aircraft's route: toward its first fix.

A region's ``hdg_deg`` is a number, a range or a distribution - drawn apart
from where the aircraft appears. :class:`TowardRoute` instead points each
aircraft, wherever it appears, at the first fix of its route: one spread over
an arc around a fix all fly at it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .routes import route_step_name

__all__ = ["TowardRoute", "is_toward_route", "route_heading"]


@dataclass(frozen=True)
class TowardRoute:
    """Marker for a spawn heading toward the first fix of the aircraft's
    route, from where it appears (``SpawnRegion.params['hdg_deg']``)."""


def is_toward_route(value: object) -> bool:
    return isinstance(value, TowardRoute)


def route_heading(lat_deg: float, lon_deg: float, route: Sequence[Any] | None, queryables: Mapping[str, Any]) -> float:
    """The initial true course, deg, from ``(lat_deg, lon_deg)`` to the first
    fix of ``route`` - its steps resolved, as the aircraft will fly them."""
    if not route:
        raise ValueError("A spawn heading toward the route needs the aircraft to have a route.")
    name = route_step_name(route[0])
    fix = queryables.get(name)
    lat, lon = getattr(fix, "lat", None), getattr(fix, "lon", None)
    if lat is None or lon is None:
        raise ValueError(f"A spawn heading toward the route needs a position for its first fix {name!r}.")
    phi0, phi1 = math.radians(lat_deg), math.radians(float(lat))
    dlon = math.radians(float(lon) - lon_deg)
    y = math.sin(dlon) * math.cos(phi1)
    x = math.cos(phi0) * math.sin(phi1) - math.sin(phi0) * math.cos(phi1) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0
