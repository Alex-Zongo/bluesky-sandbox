"""Types for this task's code, generated from its design.

Editors and type checkers read these to complete and check the design's
keys - observation parts and fields, action fields, queryables. Nothing
here runs: hooks receive the library's objects, and these subclasses only
describe what those objects hold for this task. Regenerate the package
to update them.
"""

# Narrowing the library's types, in methods without bodies, is the point:
# say so to type checkers.
# pyright: reportIncompatibleVariableOverride=false, reportIncompatibleMethodOverride=false
# mypy: disable-error-code="assignment, override, empty-body"

from __future__ import annotations

import numpy as np
from bluesky_sandbox import AgentStepContext, RegionBounds, StepBatch, Waypoint, WaypointResult
from bluesky_sandbox.core.services import QueryBatch
from typing import Any, Literal, TypedDict, overload


class Queryables(TypedDict):
    ADOPI: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    NIVDU: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    BELGU: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    VANUK: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    RIPAM: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    ROXIM: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    VALPU: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM400: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM402: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM404: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM405: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM406: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM407: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM408: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM409: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM410: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM455: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    GM456: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    OGRAS: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    XIKLA: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""
    RW01L: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""


class ObsOwnship(TypedDict):
    valpu_distance_nm: float
    """float · nm - WaypointDistanceNm: Distance from ownship to a waypoint query result."""
    valpu_ete_s: float
    """float · s - WaypointEteS: Time to this waypoint along the aircraft's route, s: the legs still to
fly to it - the direct range to the active fix, then fix by fix - at the
current ground speed. 0 once it is reached, or with it off the route: pair
with :class:WaypointRouteReached where that matters."""
    valpu_bearing_deg: float
    """float · deg - WaypointBearingDeg: Bearing from ownship to a waypoint query result."""
    gm405_route_reached: float
    """float · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    gm410_route_reached: float
    """float · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_route_active: float
    """float · switch - WaypointRouteActive: Whether this waypoint is the aircraft's active route waypoint."""
    alt_ft: float
    """float · ft - AltFt: Aircraft altitude in feet."""
    cas_kts: float
    """float · kts - CasKts: Calibrated airspeed in knots."""
    ap_cas_kts: float
    """float · kts - ApCasKts: Autopilot selected calibrated airspeed in knots."""
    vs_ftmin: float
    """float · ft/min - VsFtMin: Vertical speed in ft/min."""


class ObsIntruders(TypedDict):
    acid: np.ndarray
    """ndarray (n_intruders,) str - Each intruder row's callsign."""
    relative_valpu_distance_nm: np.ndarray
    """ndarray (n_intruders,) · nm - Difference: Difference between an intruder field and an ownship field."""
    relative_valpu_ete_s: np.ndarray
    """ndarray (n_intruders,) · s - Difference: Difference between an intruder field and an ownship field."""
    valpu_route_active: np.ndarray
    """ndarray (n_intruders,) · switch - WaypointRouteActive: Whether this waypoint is the aircraft's active route waypoint."""
    gm405_route_reached: np.ndarray
    """ndarray (n_intruders,) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    gm410_route_reached: np.ndarray
    """ndarray (n_intruders,) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_route_reached: np.ndarray
    """ndarray (n_intruders,) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_bearing_deg: np.ndarray
    """ndarray (n_intruders,) · deg - WaypointBearingDeg: Bearing from ownship to a waypoint query result."""
    relative_alt_ft: np.ndarray
    """ndarray (n_intruders,) · ft - Difference: Difference between an intruder field and an ownship field."""
    relative_ap_alt_ft: np.ndarray
    """ndarray (n_intruders,) · ft - Difference: Difference between an intruder field and an ownship field."""
    active_route_waypoint_alt_diff_ft: np.ndarray
    """ndarray (n_intruders,) · ft - ActiveRouteWaypointAltDiffFt: Ownship altitude minus its active route waypoint altitude, ft (0 if none)."""
    relative_cas_kts: np.ndarray
    """ndarray (n_intruders,) · kts - Difference: Difference between an intruder field and an ownship field."""
    dist_to_own_nm: np.ndarray
    """ndarray (n_intruders,) · nm - DistToOwnNm: Ownship-relative intruder distance in nautical miles."""
    in_conf: np.ndarray
    """ndarray (n_intruders,) - InConf: 1.0 when BlueSky's conflict detection flags the pair in conflict now."""
    conflict_tcpa_s: np.ndarray
    """ndarray (n_intruders,) · s - ConflictTcpaS: Time to horizontal CPA (s), for every intruder; negative past it."""
    conflict_horizontal_dist_at_cpa_nm: np.ndarray
    """ndarray (n_intruders,) · nm - ConflictHorizontalDistAtCpaNm: Predicted *horizontal* separation at CPA, in nm, for every intruder."""
    conflict_vertical_sep_at_cpa_ft: np.ndarray
    """ndarray (n_intruders,) · ft - ConflictVerticalSepAtCpaFt: Predicted minimum *vertical* separation over the 3-D conflict window, in ft."""


class Obs(TypedDict):
    ownship: ObsOwnship
    """10 fields"""
    intruders: ObsIntruders
    """16 fields"""


class State(TypedDict):
    pass


class Action(TypedDict):
    direct_to_waypoint: float
    """float · 0 or 1 - DirectToWaypoint: Clear the aircraft direct to a fix of its route (1), or let it fly on
along the route (0): the fix the route's step waypoint names, as the
design names its waypoints. With after, only once it is past one of
those fixes - a sequencing leg's first, say - and in any case before the
fix itself; elsewhere 1 does nothing. A clearance: 0 does not take it back."""
    ap_spd_delta_kts: float
    """float · a choice, 0..4 - ApSpdDeltaKts: Change the autopilot's selected calibrated airspeed by a delta, knots:
from the speed the aircraft is cleared to - its procedure's, under VNAV, or
the one it holds - not from the CAS it flies now, which lags it while the
aircraft accelerates or slows. With high=0 it only ever slows."""
    action_mask: float
    """float · 0 or 1 - ActionMask: Mask another action: 1 skips it this step, 0 lets it through."""


class Shapes(TypedDict):
    ADOPI: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    NIVDU: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    BELGU: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    VANUK: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    RIPAM: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    ROXIM: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    VALPU: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM400: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM402: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM404: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM405: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM406: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM407: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM408: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM409: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM410: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM455: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    GM456: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    OGRAS: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    XIKLA: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    RW01L: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""
    ADOPI_ENTRY: RegionBounds
    """AnnularSector - Bounds built from a horizontal footprint and an altitude band."""
    BELGU_ENTRY: RegionBounds
    """AnnularSector - Bounds built from a horizontal footprint and an altitude band."""
    RIPAM_ENTRY: RegionBounds
    """AnnularSector - Bounds built from a horizontal footprint and an altitude band."""


class ObsBatchOwnship(TypedDict):
    valpu_distance_nm: np.ndarray
    """ndarray (n_agents,) · nm - WaypointDistanceNm: Distance from ownship to a waypoint query result."""
    valpu_ete_s: np.ndarray
    """ndarray (n_agents,) · s - WaypointEteS: Time to this waypoint along the aircraft's route, s: the legs still to
fly to it - the direct range to the active fix, then fix by fix - at the
current ground speed. 0 once it is reached, or with it off the route: pair
with :class:WaypointRouteReached where that matters."""
    valpu_bearing_deg: np.ndarray
    """ndarray (n_agents,) · deg - WaypointBearingDeg: Bearing from ownship to a waypoint query result."""
    gm405_route_reached: np.ndarray
    """ndarray (n_agents,) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    gm410_route_reached: np.ndarray
    """ndarray (n_agents,) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_route_active: np.ndarray
    """ndarray (n_agents,) · switch - WaypointRouteActive: Whether this waypoint is the aircraft's active route waypoint."""
    alt_ft: np.ndarray
    """ndarray (n_agents,) · ft - AltFt: Aircraft altitude in feet."""
    cas_kts: np.ndarray
    """ndarray (n_agents,) · kts - CasKts: Calibrated airspeed in knots."""
    ap_cas_kts: np.ndarray
    """ndarray (n_agents,) · kts - ApCasKts: Autopilot selected calibrated airspeed in knots."""
    vs_ftmin: np.ndarray
    """ndarray (n_agents,) · ft/min - VsFtMin: Vertical speed in ft/min."""


class ObsBatchIntruders(TypedDict):
    acid: np.ndarray
    """ndarray (n_agents, n_intruders) str - Each intruder row's callsign."""
    relative_valpu_distance_nm: np.ndarray
    """ndarray (n_agents, n_intruders) · nm - Difference: Difference between an intruder field and an ownship field."""
    relative_valpu_ete_s: np.ndarray
    """ndarray (n_agents, n_intruders) · s - Difference: Difference between an intruder field and an ownship field."""
    valpu_route_active: np.ndarray
    """ndarray (n_agents, n_intruders) · switch - WaypointRouteActive: Whether this waypoint is the aircraft's active route waypoint."""
    gm405_route_reached: np.ndarray
    """ndarray (n_agents, n_intruders) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    gm410_route_reached: np.ndarray
    """ndarray (n_agents, n_intruders) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_route_reached: np.ndarray
    """ndarray (n_agents, n_intruders) · switch - WaypointRouteReached: Whether this waypoint has been reached in the aircraft's route."""
    valpu_bearing_deg: np.ndarray
    """ndarray (n_agents, n_intruders) · deg - WaypointBearingDeg: Bearing from ownship to a waypoint query result."""
    relative_alt_ft: np.ndarray
    """ndarray (n_agents, n_intruders) · ft - Difference: Difference between an intruder field and an ownship field."""
    relative_ap_alt_ft: np.ndarray
    """ndarray (n_agents, n_intruders) · ft - Difference: Difference between an intruder field and an ownship field."""
    active_route_waypoint_alt_diff_ft: np.ndarray
    """ndarray (n_agents, n_intruders) · ft - ActiveRouteWaypointAltDiffFt: Ownship altitude minus its active route waypoint altitude, ft (0 if none)."""
    relative_cas_kts: np.ndarray
    """ndarray (n_agents, n_intruders) · kts - Difference: Difference between an intruder field and an ownship field."""
    dist_to_own_nm: np.ndarray
    """ndarray (n_agents, n_intruders) · nm - DistToOwnNm: Ownship-relative intruder distance in nautical miles."""
    in_conf: np.ndarray
    """ndarray (n_agents, n_intruders) - InConf: 1.0 when BlueSky's conflict detection flags the pair in conflict now."""
    conflict_tcpa_s: np.ndarray
    """ndarray (n_agents, n_intruders) · s - ConflictTcpaS: Time to horizontal CPA (s), for every intruder; negative past it."""
    conflict_horizontal_dist_at_cpa_nm: np.ndarray
    """ndarray (n_agents, n_intruders) · nm - ConflictHorizontalDistAtCpaNm: Predicted *horizontal* separation at CPA, in nm, for every intruder."""
    conflict_vertical_sep_at_cpa_ft: np.ndarray
    """ndarray (n_agents, n_intruders) · ft - ConflictVerticalSepAtCpaFt: Predicted minimum *vertical* separation over the 3-D conflict window, in ft."""


class ObsBatch(TypedDict):
    ownship: ObsBatchOwnship
    """10 fields"""
    intruders: ObsBatchIntruders
    """16 fields"""


class StateBatch(TypedDict):
    pass


class ActionBatch(TypedDict):
    direct_to_waypoint: np.ndarray
    """ndarray (n_agents,) · 0 or 1 - DirectToWaypoint: Clear the aircraft direct to a fix of its route (1), or let it fly on
along the route (0): the fix the route's step waypoint names, as the
design names its waypoints. With after, only once it is past one of
those fixes - a sequencing leg's first, say - and in any case before the
fix itself; elsewhere 1 does nothing. A clearance: 0 does not take it back."""
    ap_spd_delta_kts: np.ndarray
    """ndarray (n_agents,) · a choice, 0..4 - ApSpdDeltaKts: Change the autopilot's selected calibrated airspeed by a delta, knots:
from the speed the aircraft is cleared to - its procedure's, under VNAV, or
the one it holds - not from the CAS it flies now, which lags it while the
aircraft accelerates or slows. With high=0 it only ever slows."""
    action_mask: np.ndarray
    """ndarray (n_agents,) · 0 or 1 - ActionMask: Mask another action: 1 skips it this step, 0 lets it through."""


class TaskAgentStepContext(AgentStepContext):
    """``AgentStepContext``, typed for this task: the members that hold
    the design's keys, as this design has them."""

    queryables: Queryables
    obs: Obs
    state: State
    action: Action
    applied: Action
    shapes: Shapes
    def bounds(self, name: Literal['ADOPI', 'NIVDU', 'BELGU', 'VANUK', 'RIPAM', 'ROXIM', 'VALPU', 'GM400', 'GM402', 'GM404', 'GM405', 'GM406', 'GM407', 'GM408', 'GM409', 'GM410', 'GM455', 'GM456', 'OGRAS', 'XIKLA', 'RW01L', 'ADOPI_ENTRY', 'BELGU_ENTRY', 'RIPAM_ENTRY']) -> RegionBounds: ...
    @overload
    def query(self, name: Literal['ADOPI']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['NIVDU']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['BELGU']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['VANUK']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['RIPAM']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['ROXIM']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['VALPU']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM400']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM402']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM404']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM405']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM406']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM407']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM408']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM409']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM410']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM455']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['GM456']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['OGRAS']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['XIKLA']) -> WaypointResult: ...
    @overload
    def query(self, name: Literal['RW01L']) -> WaypointResult: ...
    def query(self, name: str) -> Any: ...
    @overload
    def queryable(self, name: Literal['ADOPI']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['NIVDU']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['BELGU']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['VANUK']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['RIPAM']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['ROXIM']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['VALPU']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM400']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM402']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM404']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM405']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM406']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM407']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM408']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM409']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM410']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM455']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['GM456']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['OGRAS']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['XIKLA']) -> Waypoint: ...
    @overload
    def queryable(self, name: Literal['RW01L']) -> Waypoint: ...
    def queryable(self, name: str) -> Any: ...
    def shape(self, name: Literal['ADOPI', 'NIVDU', 'BELGU', 'VANUK', 'RIPAM', 'ROXIM', 'VALPU', 'GM400', 'GM402', 'GM404', 'GM405', 'GM406', 'GM407', 'GM408', 'GM409', 'GM410', 'GM455', 'GM456', 'OGRAS', 'XIKLA', 'RW01L', 'ADOPI_ENTRY', 'BELGU_ENTRY', 'RIPAM_ENTRY']) -> RegionBounds: ...


class TaskStepBatch(StepBatch):
    """``StepBatch``, typed for this task: the members that hold
    the design's keys, as this design has them."""

    obs: ObsBatch
    state: StateBatch
    action: ActionBatch
    def bounds(self, name: Literal['ADOPI', 'NIVDU', 'BELGU', 'VANUK', 'RIPAM', 'ROXIM', 'VALPU', 'GM400', 'GM402', 'GM404', 'GM405', 'GM406', 'GM407', 'GM408', 'GM409', 'GM410', 'GM455', 'GM456', 'OGRAS', 'XIKLA', 'RW01L', 'ADOPI_ENTRY', 'BELGU_ENTRY', 'RIPAM_ENTRY']) -> RegionBounds: ...
    def query(self, name: Literal['ADOPI', 'NIVDU', 'BELGU', 'VANUK', 'RIPAM', 'ROXIM', 'VALPU', 'GM400', 'GM402', 'GM404', 'GM405', 'GM406', 'GM407', 'GM408', 'GM409', 'GM410', 'GM455', 'GM456', 'OGRAS', 'XIKLA', 'RW01L']) -> QueryBatch: ...
    def shape(self, name: Literal['ADOPI', 'NIVDU', 'BELGU', 'VANUK', 'RIPAM', 'ROXIM', 'VALPU', 'GM400', 'GM402', 'GM404', 'GM405', 'GM406', 'GM407', 'GM408', 'GM409', 'GM410', 'GM455', 'GM456', 'OGRAS', 'XIKLA', 'RW01L', 'ADOPI_ENTRY', 'BELGU_ENTRY', 'RIPAM_ENTRY']) -> RegionBounds: ...
    def context(self, k: int) -> TaskAgentStepContext: ...
