"""Waypoint queryables: an aircraft's progress against a named fix."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky.tools.aero import ft as _M_PER_FT
from bluesky.tools.aero import nm as _M_PER_NM
from bluesky.tools.geo import qdrdist

from bluesky_sandbox.sim.bounds.coordinates import LatLon
from bluesky_sandbox.sim.fixes import RouteFix, fix_at

from bluesky_sandbox.interface.task import (
    QueryableTemporalStateUnavailable,
    StepTime,
    UnavailableStepTime,
)
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from .base import _MIN_DYNAMIC_SCALE, _ensure_navdb_loaded, _require_bound_query_result


@dataclass(frozen=True)
class WaypointCurrent:
    """Current aircraft-relative waypoint state."""

    distance_nm: float
    bearing_deg: float
    track_error_deg: float
    alt_diff_ft: float
    speed_diff_kts: float
    within_lateral: bool
    within_altitude: bool
    within_speed: bool
    satisfied: bool
    # CAS/Mach crossover speed extras (default keeps back-compat construction):
    # ``speed_diff_mach`` is current minus target Mach, and ``speed_in_mach`` is
    # True when the aircraft is above the crossover altitude (constraint in Mach).
    speed_diff_mach: float = math.nan
    speed_in_mach: bool = False

    @property
    def distance_m(self) -> float:
        return self.distance_nm * _M_PER_NM

    @property
    def alt_diff_m(self) -> float:
        return self.alt_diff_ft * _M_PER_FT


@dataclass(frozen=True)
class WaypointRoute:
    """Relationship between a waypoint queryable and the aircraft route."""

    index: int | None = None
    active: bool = False
    reached: bool = False
    future: bool = False


@dataclass(frozen=True)
class WaypointStep:
    """Waypoint facts accumulated during the current env step - kept only with
    ``track_temporal_state=True`` on the waypoint; without it, ``satisfied``
    and ``reached`` read False and the minimums raise."""

    satisfied: bool = False
    reached: bool = False
    min_distance_nm: float = math.inf
    min_abs_alt_diff_ft: float = math.inf


@dataclass(frozen=True)
class UnavailableWaypointStep:
    """Waypoint step placeholder for query results without temporal tracking."""

    satisfied: bool = False
    reached: bool = False

    @property
    def min_distance_nm(self) -> float:
        raise QueryableTemporalStateUnavailable(
            "Waypoint step minimum distance requires "
            "track_temporal_state=True on the waypoint."
        )

    @property
    def min_abs_alt_diff_ft(self) -> float:
        raise QueryableTemporalStateUnavailable(
            "Waypoint step minimum altitude difference requires "
            "track_temporal_state=True on the waypoint."
        )


@dataclass
class WaypointResult:
    """Structured waypoint query result for one aircraft."""

    _queryable: Any | None = field(default=None, repr=False, compare=False)
    _acidx: int | None = field(default=None, repr=False, compare=False)
    _target_cache: RouteFix | None = field(
        default=None, repr=False, compare=False
    )
    _current_cache: WaypointCurrent | None = field(
        default=None, repr=False, compare=False
    )
    _route_cache: WaypointRoute | None = field(
        default=None, repr=False, compare=False
    )
    step: WaypointStep | UnavailableWaypointStep = field(
        default_factory=UnavailableWaypointStep
    )
    time: StepTime | UnavailableStepTime = field(default_factory=UnavailableStepTime)

    @classmethod
    def for_aircraft(
        cls,
        queryable: Any,
        acidx: int,
        *,
        target: RouteFix | None = None,
        current: WaypointCurrent | None = None,
        route: WaypointRoute | None = None,
        step: WaypointStep | UnavailableWaypointStep | None = None,
        time: StepTime | UnavailableStepTime | None = None,
    ) -> WaypointResult:
        """Build a waypoint result for one aircraft."""
        return cls(
            _queryable=queryable,
            _acidx=acidx,
            _target_cache=target,
            _current_cache=current,
            _route_cache=route,
            step=step if step is not None else UnavailableWaypointStep(),
            time=time if time is not None else UnavailableStepTime(),
        )

    @property
    def target(self) -> RouteFix:
        """This aircraft's fix at the waypoint: as its route crosses it, or -
        not routed through it - the place, with no crossing."""
        if self._target_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._target_cache = queryable.fix_for(acidx)
        return self._target_cache

    @property
    def current(self) -> WaypointCurrent:
        """Current aircraft-relative waypoint state."""
        if self._current_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._current_cache = queryable.current_state(acidx, self.target)
        return self._current_cache

    @property
    def route(self) -> WaypointRoute:
        """Current route relationship for this waypoint."""
        if self._route_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._route_cache = queryable.route_state(acidx)
        return self._route_cache

    @property
    def aircraft_altitude_ceiling_ft(self) -> float:
        """Aircraft performance ceiling in feet for this query's aircraft."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        return float(bs.traf.perf.hmax[acidx] / ft)

    @property
    def altitude_error_scale_ft(self) -> float:
        """Symmetric altitude-error scale matching active waypoint obs bounds."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        alt_ft = self.target.crossing.alt_ft
        nominal_ft = alt_ft if alt_ft is not None else float(bs.traf.alt[acidx] / ft)
        ceiling_ft = self.aircraft_altitude_ceiling_ft
        return max(
            abs(float(nominal_ft)),
            abs(ceiling_ft - float(nominal_ft)),
            _MIN_DYNAMIC_SCALE,
        )

    @property
    def speed_error_scale_kts(self) -> float:
        """Symmetric CAS-error scale matching active waypoint obs bounds."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        speed_kts = self.target.crossing.speed_kts
        nominal_kts = speed_kts if speed_kts is not None else float(bs.traf.cas[acidx] / kts)
        lo_kts = float(bs.traf.perf.vmin[acidx] / kts)
        hi_kts = float(bs.traf.perf.vmax[acidx] / kts)
        return max(
            abs(float(nominal_kts) - lo_kts),
            abs(hi_kts - float(nominal_kts)),
            _MIN_DYNAMIC_SCALE,
        )

    @property
    def speed_error_normalized(self) -> float:
        """Regime-aware speed-error magnitude in ``[0, 1]`` (0 = on target).

        Uses the CAS/Mach crossover state: CAS error / CAS scale below the
        crossover altitude, Mach error / Mach scale above it - so the shaping
        magnitude is comparable across regimes and on the same axis the aircraft
        is controlled. This is the crossover-aware analogue of feeding
        ``speed_diff_kts`` / :attr:`speed_error_scale_kts` into ``err``. Returns 0
        when the target carries no speed constraint.
        """
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        speed_kts = self.target.crossing.speed_kts
        if speed_kts is None:
            return 0.0
        state = crossover_speed_state(acidx, speed_kts * kts)
        return abs(state.normalized_error)


def _point_of(at: Any) -> tuple[LatLon, str | None]:
    """A waypoint's ``at``: its position, and the navdb fix it names, if any."""
    if isinstance(at, LatLon):
        return at, None
    footprint = getattr(at, "footprint", at)
    # A placed or moving point: the point under its layers.
    inner = footprint
    while not hasattr(inner, "fix") and hasattr(inner, "footprint"):
        inner = inner.footprint
    if hasattr(footprint, "center_point"):
        return footprint.center_point(), getattr(inner, "fix", None)
    raise TypeError(f"Waypoint at= takes a point (bounds, footprint or LatLon), not {type(at).__name__}")


@dataclass
class Waypoint:
    """A place aircraft are evaluated against: a BlueSky navdb fix or an
    ad-hoc point.

    Lifecycle is per-airspace, not per-flight: configure once on episode
    queryables and every aircraft is evaluated against it each step.
    ``query()`` returns a :class:`WaypointResult` carrying current geometry,
    route state, and step timing. A merge point is just one use of it.

    A waypoint is only where: how it is crossed - the altitude and speed flown
    there, their windows, fly-over, an RTA - is a route's, stated on each route
    step through it (:class:`~bluesky_sandbox.sim.fixes.Crossing`), so two
    routes may cross one fix differently. A query answers for the aircraft
    asking: against its own route's crossing there, or - an aircraft not
    routed through it - geometry alone, with no altitude or speed check.

    Parameters
    ----------
    at:
        Where it is: a point - a :class:`RegionBounds` of a
        :class:`PointFootprint` (a design's named point), a footprint, or a
        :class:`LatLon`.  Its position is the point's; a point at a navdb fix
        (``PointFootprint.at_fix("EKROS")``) names it.  A scenario hands each
        episode's draw of the point - placed, in a group - on to it.
        Mutually exclusive with :attr:`lat` / :attr:`lon` and :attr:`waypoint`.
    lat, lon:
        Waypoint position in degrees.  Mutually exclusive with
        :attr:`waypoint`.
    waypoint:
        Name of a real BlueSky navdb waypoint (e.g. ``"EKROS"``,
        ``"VBG14"``).  Resolved to ``(lat, lon)`` at construction time
        via the BlueSky navigation database.  Mutually exclusive with
        :attr:`lat` / :attr:`lon`.
    reach_radius_nm:
        How near the aircraft must be to it, laterally, for ``within_lateral``
        (and ``satisfied``); ``None`` - any distance.
    color:
        Display color name passed through to drivers.  Defaults to ``"cyan"``.
    tsas_region:
        Optional name of another episode queryable
        (must be a :class:`QueryRegion`) used as a TSAS-strip filter:
        only aircraft inside that region appear on this waypoint's
        sequencing strip.  Lookup is by-name so the cone is configured
        once and reused by task code and the TSAS view. ``None`` means no
        filter - every aircraft is listed.
    render_shape:
        Whether the waypoint marker is drawn on map/world views.  ``True``
        by default.  Set ``False`` to keep the waypoint out of the map while
        still allowing it to be used by query logic, routes, and TSAS.
    render_tsas:
        Whether the waypoint gets a TSAS sequencing strip.  ``None`` means
        follow ``render_shape``.
    render_label:
        Whether the waypoint's name is drawn next to its map marker.
        ``True`` by default; set ``False`` to draw the marker without
        text.  Does not affect the TSAS strip header.
    track_temporal_state:
        Whether this waypoint is sampled every physics substep to provide
        ``during_step``, accumulated ``time``, and step-minimum values.
    """

    lat: float | None = None
    lon: float | None = None
    result_type: ClassVar[type[WaypointResult]] = WaypointResult
    waypoint: str | None = None
    reach_radius_nm: float | None = 1.0
    color: str = "cyan"
    tsas_region: str | None = None
    render_shape: bool = True
    render_tsas: bool | None = None
    render_label: bool = True
    track_temporal_state: bool = False
    at: Any = None

    @staticmethod
    def _acid(acidx: int) -> str | None:
        try:
            return str(bs.traf.id[acidx])
        except (IndexError, TypeError, ValueError):
            return None

    def __post_init__(self) -> None:
        self._place()

    def _place(self) -> None:
        if self.at is not None:
            # Where its point is - by name, at a fix.
            center, fix = _point_of(self.at)
            self.lat, self.lon = center.lat_deg, center.lon_deg
            self.waypoint = fix
            return
        latlon_set = self.lat is not None and self.lon is not None
        if self.waypoint is not None:
            if latlon_set:
                raise ValueError(
                    "Waypoint: pass either `waypoint=...` or `lat=`/`lon=`, not both."
                )
            _ensure_navdb_loaded()
            idx = bs.navdb.getwpidx(self.waypoint)
            if idx < 0:
                raise ValueError(
                    f"Waypoint: {self.waypoint!r} not found in BlueSky navdb."
                )
            self.lat = float(bs.navdb.wplat[idx])
            self.lon = float(bs.navdb.wplon[idx])
        elif not latlon_set:
            raise ValueError(
                "Waypoint: must specify either `waypoint=...` or both "
                "`lat=` and `lon=`."
            )
    def fix_for(self, acidx: int, route_idx: int | None = None) -> RouteFix:
        """Aircraft ``acidx``'s fix here: its route's, where its route passes
        here (at ``route_idx``, or found on it); else the place, with no
        crossing."""
        route_idx = self._route_index(acidx, route_idx)
        fix = None if route_idx is None else fix_at(acidx, route_idx)
        if fix is not None:
            return fix
        return RouteFix(lat=float(self.lat), lon=float(self.lon), waypoint=self.waypoint)

    def _route_index(self, acidx: int, route_idx: int | None = None) -> int | None:
        if route_idx is not None:
            return route_idx
        routes = bs.traf.ap.route
        if acidx < 0 or acidx >= len(routes):
            return None
        route = routes[acidx]
        try:
            nwp = int(route.nwp or 0)
        except (TypeError, ValueError):
            return None
        if nwp <= 0:
            return None

        target_names = {str(self.waypoint).upper()} if self.waypoint else set()
        tgt_lat, tgt_lon = float(self.lat), float(self.lon)
        names = route.wpname
        lats = route.wplat
        lons = route.wplon
        for wp_idx in range(nwp):
            try:
                raw_name = names[wp_idx]
                wp_name = (
                    raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                ).upper()
            except (IndexError, TypeError, ValueError):
                wp_name = ""
            if wp_name and wp_name in target_names:
                return wp_idx

            try:
                wp_lat = float(lats[wp_idx])
                wp_lon = float(lons[wp_idx])
            except (IndexError, TypeError, ValueError):
                continue
            if not (math.isfinite(wp_lat) and math.isfinite(wp_lon)):
                continue
            _qdr, dist = qdrdist(wp_lat, wp_lon, tgt_lat, tgt_lon)
            if dist < 0.05:
                return wp_idx
        return None

    def route_state(
        self,
        acidx: int,
        route_idx: int | None = None,
    ) -> WaypointRoute:
        route_idx = self._route_index(acidx, route_idx)
        if route_idx is None:
            return WaypointRoute()

        routes = bs.traf.ap.route
        route = routes[acidx]
        try:
            active_idx = -1 if route.iactwp is None else int(route.iactwp)
        except (TypeError, ValueError):
            active_idx = -1

        reached = route_idx < active_idx
        try:
            if acidx in set(bs.traf.ap.idxreached):
                just_reached = active_idx - 1 if bool(bs.traf.swlnav[acidx]) else active_idx
                reached = reached or route_idx == just_reached
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        active = route_idx == active_idx
        future = route_idx >= active_idx and not reached if active_idx >= 0 else False
        return WaypointRoute(
            index=route_idx,
            active=active,
            reached=reached,
            future=future,
        )

    def reached_during_substep(self, acidx: int, route_idx: int | None = None) -> bool:
        """Return true only when this waypoint was passed in the current substep."""
        route_idx = self._route_index(acidx, route_idx)
        if route_idx is None:
            return False
        routes = bs.traf.ap.route
        route = routes[acidx]
        try:
            active_idx = -1 if route.iactwp is None else int(route.iactwp)
        except (TypeError, ValueError):
            active_idx = -1
        try:
            if acidx not in set(bs.traf.ap.idxreached):
                return False
            just_reached = active_idx - 1 if bool(bs.traf.swlnav[acidx]) else active_idx
            return route_idx == just_reached
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def current_state(self, acidx: int, fix: RouteFix | None = None) -> WaypointCurrent:
        """Aircraft ``acidx``'s state here, against ``fix`` - its own fix here
        (:meth:`fix_for`) when not given: the geometry to the fix, and the
        altitude and speed against its crossing."""
        lat_deg = float(bs.traf.lat[acidx])
        lon_deg = float(bs.traf.lon[acidx])
        alt_ft = float(bs.traf.alt[acidx] / ft)
        fix = fix or self.fix_for(acidx)
        crossing = fix.crossing
        qdr, dist = qdrdist(lat_deg, lon_deg, fix.lat, fix.lon)
        alt_diff_ft = math.nan if crossing.alt_ft is None else alt_ft - crossing.alt_ft
        track_error = (
            float(qdr) - float(bs.traf.trk[acidx]) + 540.0
        ) % 360.0 - 180.0
        speed_diff = (
            math.nan
            if crossing.speed_kts is None
            else float(bs.traf.cas[acidx] / _M_PER_NM * 3600.0) - crossing.speed_kts
        )
        # Regime-aware speed state (CAS below the crossover altitude, Mach above),
        # computed whenever the crossing states a speed so the readout and the
        # window switch regimes for *any* sampled speed/altitude.
        speed_diff_mach = math.nan
        speed_in_mach = False
        if crossing.speed_kts is not None:
            crossover = crossover_speed_state(acidx, crossing.speed_kts * kts)
            speed_diff_mach = crossover.mach_diff
            speed_in_mach = crossover.in_mach
        within_lateral = (
            True
            if self.reach_radius_nm is None
            else float(dist) <= self.reach_radius_nm
        )
        within_altitude = bool(crossing.within_altitude(alt_ft))
        within_speed = crossing.within_speed(acidx)
        return WaypointCurrent(
            distance_nm=float(dist),
            bearing_deg=float(qdr),
            track_error_deg=track_error,
            alt_diff_ft=float(alt_diff_ft),
            speed_diff_kts=float(speed_diff),
            within_lateral=within_lateral,
            within_altitude=within_altitude,
            within_speed=within_speed,
            satisfied=within_lateral and within_altitude and within_speed,
            speed_diff_mach=float(speed_diff_mach),
            speed_in_mach=bool(speed_in_mach),
        )
