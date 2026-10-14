"""Reading an aircraft's route, fix by fix (:func:`~bluesky_sandbox.sim.fixes.fix_at`).

Shared by the active-route-waypoint observations and the route-relative actions,
so both read the same fix.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vtas2cas
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.fixes import fix_at


def _route_index(idx: int, offset: int = 0) -> int | None:
    """The route index of the fix ``offset`` legs from the active one, or
    ``None`` when there is none there - as :func:`_active_route_waypoint`."""
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    try:
        iact = int(routes[idx].iactwp)
        nwp = int(routes[idx].nwp or 0)
    except (TypeError, ValueError):
        return None
    target = iact + offset
    return target if iact >= 0 and 0 <= target < nwp else None


def _active_route_waypoint(
    idx: int,
    offset: int = 0,
) -> tuple[float, float, float | None, float | None] | None:
    """Return ``(lat, lon, alt_m, spd_ms)`` for a route fix relative to the
    aircraft's active leg, or ``None`` when there is no usable fix there.

    ``offset`` selects which fix: ``0`` (default) is the active leg the
    autopilot is currently flying to; ``1`` is the *next* leg (e.g. the exit
    fix while still working a merge fix), ``2`` the one after, etc. ``None``
    whenever ``iactwp + offset`` falls outside the route (no next leg on a
    single-leg route, or beyond the final fix) - the same "no usable waypoint"
    convention offset ``0`` already used for a routeless aircraft.

    ``alt_m`` and ``spd_ms`` are ``None`` when the waypoint carries no altitude
    or speed constraint. A target arrival time is a speed constraint too, only
    a derived one: where the fix has no speed gate but a time to be over it,
    ``spd_ms`` is the CAS that still meets that time from where the aircraft is
    (:func:`_scheduled_cas_ms`). So every reader of a fix's speed - the
    route-relative speed actions, whose zero is then "on schedule", and the
    speed observations - agrees on it. Shared by the active-route-waypoint observation fields
    and the deviation-from-nominal action fields (which always use ``offset=0``
    - actions command the leg actually being flown, never a future one) so
    both read the same target.
    """
    target = _route_index(idx, offset)
    fix = None if target is None else fix_at(idx, target)
    if fix is None or not (np.isfinite(fix.lat) and np.isfinite(fix.lon)):
        return None
    crossing = fix.crossing
    alt = None if crossing.alt_ft is None else crossing.alt_ft * ft
    if crossing.speed_kts is not None:
        speed = crossing.speed_kts * kts
    else:
        speed = _scheduled_cas_ms(idx, fix.due_s, offset)
    return (fix.lat, fix.lon, alt, speed)


def _scheduled_cas_ms(idx: int, due: float | None, offset: int) -> float | None:
    """The CAS that gets the aircraft at ``idx`` over the route fix ``offset``
    legs ahead at its arrival time ``due``, or ``None`` with no time: the
    along-route distance left over the time to go, in the wind the aircraft
    flies in now. Overdue, its maximum. Late, that is faster; early, slower -
    and below its minimum speed, only a longer path loses the rest."""
    dist_nm = None if due is None else _route_along_distance_nm(idx, offset)
    if dist_nm is None:
        return None
    time_to_go = due - float(bs.sim.simt)
    if time_to_go <= 0.0:
        return float(bs.traf.perf.vmax[idx])
    gs_needed = dist_nm * nm / time_to_go
    tas_needed = gs_needed + float(bs.traf.tas[idx]) - float(bs.traf.gs[idx])
    return float(vtas2cas(max(tas_needed, 0.0), float(bs.traf.alt[idx])))


def _route_along_distance_nm(idx: int, offset: int) -> float | None:
    """Distance from the aircraft to the route fix at ``iactwp + offset``, nm,
    measured ALONG the remaining legs.

    Direct great-circle range to the active fix, then leg by leg out to the
    target fix - the distance actually flown, not the straight line to a fix
    two legs ahead. Reduces to the plain range at ``offset=0``, where it equals
    :class:`ActiveRouteWaypointDistanceNm`. ``None`` whenever there is no
    usable fix at that offset, mirroring :func:`_active_route_waypoint`'s
    validation and convention.
    """
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    route = routes[idx]
    try:
        iact = int(route.iactwp)
        nwp = int(route.nwp or 0)
    except (TypeError, ValueError):
        return None
    target = iact + offset
    if iact < 0 or target < 0 or target >= nwp:
        return None
    lat = float(bs.traf.lat[idx])
    lon = float(bs.traf.lon[idx])
    total_nm = 0.0
    for leg in range(iact, target + 1):
        try:
            wp_lat = float(route.wplat[leg])
            wp_lon = float(route.wplon[leg])
        except (IndexError, TypeError, ValueError):
            return None
        if not (np.isfinite(wp_lat) and np.isfinite(wp_lon)):
            return None
        _qdr, dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        total_nm += float(dist)
        lat, lon = wp_lat, wp_lon
    return total_nm
