"""How each aircraft crosses the fixes of its route.

A route step pairs a waypoint - a place - with a :class:`Crossing`: the
altitude and speed flown there, the windows about them, whether it is flown
over, and whether the aircraft's FMS flies its arrival time (an RTA). At spawn
each step resolves, for that aircraft, into a :class:`RouteFix` - the place as
numbers, the crossing, and the time it is due there - and the whole route is
kept in :data:`ROUTES`, the one record of it every query, field and driver
reads. BlueSky is handed what it can fly (``ADDWPT``, ``RTA``); nothing is read
back from it but for routes the sandbox did not make.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.sim.aircraft_uids import live_aircraft_uids
from bluesky_sandbox.sim.performance.speeds import as_cas_ms, within_speed_window

__all__ = [
    "CROSSING_KEYS",
    "Crossing",
    "ROUTES",
    "ResolvedRoute",
    "RouteFix",
    "check_window",
    "fix_at",
    "in_window",
]

#: A route step's keys that make its crossing - the same names as
#: :class:`Crossing`'s fields.
CROSSING_KEYS = (
    "alt_ft",
    "alt_below_ft",
    "alt_above_ft",
    "speed_kts",
    "speed_below_kts",
    "speed_above_kts",
    "flyover",
    "rta",
)


def in_window(value, target, below, above):
    """Whether ``value`` (or each of an array) lies in a window about
    ``target``: no more than ``below`` under it, no more than ``above`` over
    it. A ``None`` or NaN target is no constraint, and a ``None`` or NaN side
    is open; each may be an array, one per value."""
    value = np.asarray(value, dtype=np.float64)
    if target is None:
        return np.ones(value.shape, dtype=bool)
    target = np.asarray(target, dtype=np.float64)
    inside = np.ones(np.broadcast(value, target).shape, dtype=bool)
    for side, reach in ((-1.0, below), (1.0, above)):
        if reach is None:
            continue
        reach = np.asarray(reach, dtype=np.float64)
        outside = (value - target) * side > reach
        inside &= ~(outside & np.isfinite(reach))
    return inside | ~np.isfinite(target)


def check_window(below: float | None, above: float | None, unit: str) -> None:
    """A window's sides are how far below and above its value it reaches: >= 0."""
    for side, reach in (("below", below), ("above", above)):
        if reach is not None and not (isinstance(reach, (int, float)) and reach >= 0):
            raise ValueError(
                f"Crossing window: how far {side} its value it reaches ({unit}) "
                f"must be >= 0, got {reach!r}."
            )


@dataclass(frozen=True)
class Crossing:
    """How a route crosses one of its fixes.

    ``alt_ft`` / ``speed_kts`` - the altitude and CAS flown there, handed to
    BlueSky with the waypoint. Their windows - ``alt_below_ft`` /
    ``alt_above_ft`` and ``speed_below_kts`` / ``speed_above_kts`` - say how far
    below and above them the aircraft may be, an unset side open:
    ``alt_below_ft=0`` alone is "at or above", ``speed_above_kts=0`` alone
    "N KT or less", both sides 500 a +/-500 FT band, neither no check. Speed is
    checked in CAS below the CAS/Mach crossover and in Mach above it, the
    window converted at the aircraft's altitude, as aircraft are controlled
    there.

    ``flyover`` - flown over, crossing the fix before turning, rather than by
    it (BlueSky's default, turning early onto the next leg). ``rta`` - the
    aircraft's FMS flies its arrival time there (a BlueSky RTA), where the step
    gives it one (``arrival_slack_s``); without it the time is a target only,
    for the controller.
    """

    alt_ft: float | None = None
    alt_below_ft: float | None = None
    alt_above_ft: float | None = None
    speed_kts: float | None = None
    speed_below_kts: float | None = None
    speed_above_kts: float | None = None
    flyover: bool = False
    rta: bool = False

    def __post_init__(self) -> None:
        check_window(self.alt_below_ft, self.alt_above_ft, "ft")
        check_window(self.speed_below_kts, self.speed_above_kts, "kt")

    @classmethod
    def from_step(cls, step: Mapping[str, Any]) -> Crossing:
        """The crossing a route step states: its :data:`CROSSING_KEYS`."""
        return cls(**{key: step[key] for key in CROSSING_KEYS if key in step})

    def within_altitude(self, alt_ft):
        """Whether ``alt_ft`` (or each of an array) is in its altitude window."""
        return in_window(alt_ft, self.alt_ft, self.alt_below_ft, self.alt_above_ft)

    def within_speed(self, acidx: int) -> bool:
        """Whether aircraft ``acidx`` flies its speed window - always, with no
        speed stated."""
        if self.speed_kts is None:
            return True
        return within_speed_window(
            acidx, self.speed_kts * kts, self.speed_below_kts, self.speed_above_kts
        )


@dataclass(frozen=True)
class RouteFix:
    """One fix of one aircraft's route, resolved for it: where (a navdb fix
    by ``waypoint`` name, or a position), how it crosses it, and when it is
    due there (simulator seconds, as ``bs.sim.simt``; ``None`` for no time)."""

    lat: float
    lon: float
    waypoint: str | None = None
    crossing: Crossing = field(default_factory=Crossing)
    due_s: float | None = None


@dataclass(frozen=True)
class ResolvedRoute:
    """An aircraft's route as resolved at spawn: each step's design name, its
    fix, and its arrival slack (seconds, a distribution, or ``None``) that its
    due time is drawn from."""

    names: tuple[str, ...]
    fixes: tuple[RouteFix, ...]
    arrival_slacks: tuple[Any, ...] = ()


def _key(idx: int) -> Hashable:
    """Aircraft ``idx``'s uid - or its callsign, without a tracker."""
    uids = live_aircraft_uids()
    if uids is not None and len(uids) == len(bs.traf.id):
        return int(uids[int(idx)])
    return str(bs.traf.id[int(idx)])


class AircraftRoutes:
    """Each live aircraft's :class:`ResolvedRoute`, following the aircraft, not
    its callsign - BlueSky reuses a deleted aircraft's callsign.

    Set when an aircraft spawns or its route is replaced; forgotten when it is
    deleted; cleared on every environment reset."""

    def __init__(self) -> None:
        # uid -> (callsign when set, route)
        self._routes: dict[Hashable, tuple[str, ResolvedRoute]] = {}

    def set(self, acid: str, route: ResolvedRoute | None) -> None:
        key = _key(bs.traf.id.index(acid))
        if route is None:
            self._routes.pop(key, None)
        else:
            self._routes[key] = (acid, route)

    def route(self, idx: int) -> ResolvedRoute | None:
        """Aircraft ``idx``'s route (BlueSky traffic index), or ``None``."""
        entry = self._routes.get(_key(idx))
        return None if entry is None else entry[1]

    def forget(self, acid: str) -> None:
        """Drop what despawned aircraft ``acid`` left behind."""
        for key in [k for k, (owner, _r) in self._routes.items() if owner == acid]:
            del self._routes[key]

    def clear(self) -> None:
        self._routes.clear()


#: Every aircraft's resolved route - one simulator per process, as BlueSky's.
ROUTES = AircraftRoutes()


def fix_at(idx: int, route_index: int) -> RouteFix | None:
    """Fix ``route_index`` of aircraft ``idx``'s route: as resolved for it, or
    - for a route the sandbox did not make (a scenario's, raw ``ADDWPT``), or
    one edited since - as BlueSky holds it: its position, altitude, speed and
    RTA, with no windows. ``None`` past the route's end."""
    flown = bs.traf.ap.route[int(idx)]
    if not 0 <= route_index < int(flown.nwp or 0):
        return None
    route = ROUTES.route(idx)
    if route is not None and len(route.fixes) == int(flown.nwp):
        fix = route.fixes[route_index]
        # Unless the route was edited behind the record (a hook's ADDWPT).
        if abs(fix.lat - float(flown.wplat[route_index])) < 1e-6 and abs(
            fix.lon - float(flown.wplon[route_index])
        ) < 1e-6:
            return fix
    alt_m = float(flown.wpalt[route_index])
    alt_ft = alt_m / ft if math.isfinite(alt_m) and alt_m >= 0.0 else None
    spd = float(flown.wpspd[route_index])
    speed_kts = None
    if math.isfinite(spd) and spd >= 0.0:
        # A Mach (a scenario's route) as its CAS at the fix's altitude.
        at_m = float(bs.traf.alt[int(idx)]) if alt_ft is None else alt_ft * ft
        speed_kts = float(as_cas_ms(spd, at_m)) / kts
    rta = float(flown.wprta[route_index]) if route_index < len(flown.wprta) else -999.0
    return RouteFix(
        lat=float(flown.wplat[route_index]),
        lon=float(flown.wplon[route_index]),
        waypoint=str(flown.wpname[route_index]),
        crossing=Crossing(alt_ft=alt_ft, speed_kts=speed_kts, rta=rta >= 0.0),
        due_s=rta if rta >= 0.0 else None,
    )
