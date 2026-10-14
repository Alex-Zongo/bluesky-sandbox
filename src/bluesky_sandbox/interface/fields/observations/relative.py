"""Another aircraft relative to the ownship: its position and motion, the
closest point of approach, and shared route fixes.

Pair fields, so ``intruder_obs_fields`` only. Position and motion come from the
shared pair geometry (:func:`~bluesky_sandbox.sim.geometry.conflict.conflict_geometry`),
on the flat-earth projection BlueSky's conflict detection measures on, so they
describe the encounter the detector sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.geometry.conflict import ConflictView

from .._pairs import _BroadcastPairs, _fix_projection, _track_frame
from .._reference import one_pair_fix_projection, one_pair_motion, one_pair_track_frame
from ..base import ObsMeta, ObsQuantity, PairObsField, Unit


def _future_tcpa_h(along, cross, v_along, v_cross):
    """Time to the horizontal CPA from now on, in hours (nm over kts): a
    passed one reads now."""
    v2 = np.maximum(v_along * v_along + v_cross * v_cross, 1e-9)
    return np.maximum(-(along * v_along + cross * v_cross) / v2, 0.0)


def _one_tcpa_s(along, cross, v_along, v_cross):
    v2 = max(v_along * v_along + v_cross * v_cross, 1e-9)
    return max(-(along * v_along + cross * v_cross) / v2, 0.0)


@dataclass(frozen=True)
class DistToOwnNm(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder distance in nautical miles."""

    meta = ObsMeta("dist_to_own_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "distance nautical miles"] = 0.0
    high: Annotated[float, "distance nautical miles"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return ConflictView(own, other).horiz_dist_now_nm

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, _ve, _vn = one_pair_motion(own_idx, other_idx)
        return math.hypot(east, north) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ClosingRateKts(_BroadcastPairs, PairObsField):
    """Ownship-intruder horizontal closing rate, in knots.

    Positive means the pair is closing horizontally; negative means opening.
    """

    meta = ObsMeta("closing_rate_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "speed knots"] = -1000.0
    high: Annotated[float, "speed knots"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, cross, v_along, v_cross = _track_frame(own, other)
        dist = np.maximum(np.hypot(along, cross), 1e-9)
        return -(along * v_along + cross * v_cross) / dist

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range_m = max(math.hypot(east, north), 1e-6)
        return -(east * v_east + north * v_north) / range_m / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BearingRateDegPerSec(_BroadcastPairs, PairObsField):
    """Ownship-intruder bearing rate, in degrees per second.

    Positive means the bearing from ownship to intruder rotates clockwise.
    """

    meta = ObsMeta(
        "bearing_rate_deg_per_sec",
        Unit.DEG_PER_SEC,
        ObsQuantity.BEARING,
        is_pair=True,
    )
    low: Annotated[float, "degrees per second"] = -10.0
    high: Annotated[float, "degrees per second"] = 10.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # The track frame mirrors east/north, so the clockwise rate reads
        # along x v_cross - cross x v_along there. In SI: BlueSky's knot is
        # 0.514444 m/s, not exactly a nautical mile per hour.
        along, cross, v_along, v_cross = _track_frame(own, other)
        along, cross, v_along, v_cross = along * nm, cross * nm, v_along * kts, v_cross * kts
        dist2 = np.maximum(along * along + cross * cross, 1e-6)
        return np.degrees((along * v_cross - cross * v_along) / dist2)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range2 = max(east * east + north * north, 1e-6)
        return math.degrees((north * v_east - east * v_north) / range2)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAlongTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ALONG the own track, nm (ahead +).

    Track-frame Cartesian. Unlike range x bearing, this gives relative position
    directly - no multiplicative decode whose bearing resolution scales with
    range. Pairs with :class:`RelPosCrossTrackNm`.
    """

    meta = ObsMeta(
        "rel_pos_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return _track_frame(own, other)[0]

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[0] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosCrossTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ACROSS the own track, nm (right +).

    Track-frame Cartesian companion to :class:`RelPosAlongTrackNm`.
    """

    meta = ObsMeta(
        "rel_pos_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return _track_frame(own, other)[1]

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[1] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelAlongTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ALONG the own track, kts.

    Track-frame Cartesian relative velocity (negative = intruder falling behind /
    ownship overtaking along-track). Together with the cross component this is the
    same information as closing rate + bearing rate, but singularity-free and
    without the range-dependent scaling.
    """

    meta = ObsMeta("rel_vel_along_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "along-track kts"] = -1000.0
    high: Annotated[float, "along-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return _track_frame(own, other)[2]

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[2] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelCrossTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ACROSS the own track, kts (right +).

    Track-frame Cartesian companion to :class:`RelVelAlongTrackKts`.
    """

    meta = ObsMeta("rel_vel_cross_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "cross-track kts"] = -1000.0
    high: Annotated[float, "cross-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return _track_frame(own, other)[3]

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[3] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaAlongTrackNm(_BroadcastPairs, PairObsField):
    """Along-track relative position at predicted CPA, nm (ahead +).

    Cartesian replacement for the scalar horizontal-miss `dcpa`: together with the
    cross component it preserves the miss magnitude AND adds the pass direction.
    A CPA already passed reads the position now.
    """

    meta = ObsMeta(
        "rel_pos_at_cpa_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, cross, v_along, v_cross = _track_frame(own, other)
        return along + v_along * _future_tcpa_h(along, cross, v_along, v_cross)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        return (along + v_along * _one_tcpa_s(along, cross, v_along, v_cross)) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaCrossTrackNm(_BroadcastPairs, PairObsField):
    """Cross-track relative position at predicted CPA, nm (right +).

    The sign is which side the intruder passes at closest approach - the key cue
    for turn direction. ``|along, cross|`` == the horizontal miss dcpa.
    """

    meta = ObsMeta(
        "rel_pos_at_cpa_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, cross, v_along, v_cross = _track_frame(own, other)
        return cross + v_cross * _future_tcpa_h(along, cross, v_along, v_cross)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        return (cross + v_cross * _one_tcpa_s(along, cross, v_along, v_cross)) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVsFtMin(_BroadcastPairs, PairObsField):
    """Intruder vertical speed minus ownship vertical speed, in ft/min."""

    meta = ObsMeta(
        "rel_vs_ft_min",
        Unit.FT_PER_MIN,
        ObsQuantity.VERTICAL_SPEED,
        is_pair=True,
    )
    low: Annotated[float, "vertical speed feet per minute"] = -6000.0
    high: Annotated[float, "vertical speed feet per minute"] = 6000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return ConflictView(own, other).rel_vs_ft_min

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return (float(bs.traf.vs[other_idx]) - float(bs.traf.vs[own_idx])) * 60.0 / ft

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixApproachDistNm(_BroadcastPairs, PairObsField):
    """How near an intruder's *observed* path passes the ownship's active fix, nm.

    Projects the intruder's current position/velocity to its closest approach
    of the ownship's own fix (``route_offset`` 0 = active leg, 1 = next leg,
    e.g. the shared exit while still working a merge fix). Small => that
    aircraft is funnelling into my fix (a merge threat); large => crossing or
    unrelated. Captures the shared-destination geometry that pairwise
    ownship<->intruder CPA fields miss (same-heading in-trail traffic has low
    closing rate yet converges at the fix). Non-private: see :func:`_fix_projection`.
    Sentinel ``high`` (far) when the ownship has no usable fix.
    """

    meta = ObsMeta(
        "intruder_fix_approach_dist_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "distance nm lower bound"] = 0.0
    high: Annotated[float, "distance nm upper bound (also the no-fix sentinel)"] = 100.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        cpa_nm, _eta, _own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, cpa_nm, float(self.high))

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return float(self.high) if projection is None else projection[0]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixArrivalDeltaS(_BroadcastPairs, PairObsField):
    """Signed queue order at the ownship's fix: intruder ETA minus ownship ETA, s.

    Both ETAs come from projecting *observed* motion onto the ownship's own fix
    (``route_offset`` 0 = active, 1 = next). Negative => the intruder reaches
    my fix *before* me (I should slot behind it); positive => I am ahead. This
    is the temporal-sequencing signal for a point merge - who arrives first -
    that lets the policy learn to decelerate and space in-trail rather than
    turn (which cannot separate traffic converging on a shared point).
    Non-private: see :func:`_fix_projection`. ``0`` (no order) when the ownship
    has no usable fix. Pair with :class:`IntruderFixApproachDistNm` so a
    consumer knows *whether* the intruder is heading to the fix at all.
    """

    meta = ObsMeta(
        "intruder_fix_arrival_delta_s", Unit.S, ObsQuantity.TIME, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "time s lower bound"] = -600.0
    high: Annotated[float, "time s upper bound"] = 600.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, intr_eta, own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, intr_eta - own_eta, 0.0)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[1] - projection[2]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixVerticalSepFt(_BroadcastPairs, PairObsField):
    """Signed altitude gap at the ownship's fix: intruder minus ownship, ft.

    Both aircraft are projected to the fix by their observed vertical speed
    (current alt + VS x ETA), and the signed difference is taken there. Near 0
    => they will be *co-altitude* at the merge (a genuine conflict that lateral
    turning cannot separate - your same-altitude merge case); large magnitude
    => vertically clear, no sequencing needed regardless of horizontal
    approach. This is the vertical companion to
    :class:`IntruderFixApproachDistNm` (horizontal) and
    :class:`IntruderFixArrivalDeltaS` (temporal); together they predict a merge
    conflict against the ownship's fix the way the pairwise
    ``ConflictHorizontalDistAtCpaNm`` / ``ConflictVerticalSepAtCpaFt`` /
    ``ConflictTcpaS`` triple predicts a pairwise one. Signed so the policy
    knows which way to separate. Non-private: see :func:`_fix_projection`.
    ``0`` when the ownship has no usable fix - read alongside the approach
    field, whose sentinel flags that state.
    """

    meta = ObsMeta(
        "intruder_fix_vertical_sep_ft", Unit.FT, ObsQuantity.ALTITUDE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "altitude ft lower bound"] = -5000.0
    high: Annotated[float, "altitude ft upper bound"] = 5000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, _eta, _own_eta, vsep_ft, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, vsep_ft, 0.0)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[3]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in degrees.

    The true bearing from ownship to the intruder, signed in ``[-180, 180]``
    (0 = north, +90 = east, -90 = west).
    """

    meta = ObsMeta(
        "brg_from_own_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = ConflictView(own, other)
        return np.degrees(np.arctan2(view.rel_east_now_nm, view.rel_north_now_nm))

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return _one_bearing_deg(own_idx, other_idx)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnRelTrkDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in the ownship track frame, degrees.

    The true bearing from ownship to the intruder minus the ownship track, so
    ``0`` means the intruder is dead ahead and ``+/-180`` directly behind. This
    is the egocentric (body-frame) polar angle, matching
    :class:`ActiveRouteWaypointTrackErrorDeg` for the waypoint; pair with
    :class:`DistToOwnNm` for a full egocentric polar intruder position. Unlike
    :class:`BrgFromOwnDeg` (an absolute compass bearing), this rotates with the
    ownship heading.
    """

    meta = ObsMeta(
        "brg_from_own_rel_trk_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, cross, _va, _vc = _track_frame(own, other)
        return np.degrees(np.arctan2(cross, along))

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        relative = _one_bearing_deg(own_idx, other_idx) - float(bs.traf.trk[own_idx])
        return (relative + 180.0) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def _one_bearing_deg(own_idx: int, other_idx: int) -> float:
    qdr, _dist = kwikqdrdist(
        float(bs.traf.lat[own_idx]),
        float(bs.traf.lon[own_idx]),
        float(bs.traf.lat[other_idx]),
        float(bs.traf.lon[other_idx]),
    )
    return (float(qdr) + 180.0) % 360.0 - 180.0
