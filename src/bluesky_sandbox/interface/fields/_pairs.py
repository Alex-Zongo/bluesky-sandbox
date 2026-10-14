"""Pair geometry: one ownship, or many, against other aircraft.

Pair helpers take ``own`` and ``other`` INDEX ARRAYS that broadcast against
each other: a scalar ownship against a 1-D intruder list (``get_pairs``), or an
``(k, 1)`` column of ownships against a ``(1, n)`` row of every aircraft
(``get_pair_matrix``). One computation serves both, so they cannot disagree.
"""

from __future__ import annotations

from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import nm
from bluesky.tools.geo import qdrdist

from bluesky_sandbox.sim.geometry.conflict import ConflictView

from ._common import _M_TO_FT, _indices_array, _traf_array
from ._route import _active_route_waypoint
from .base import ObsField


def _pair_index_grid(own_indices: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(k, 1)`` ownships and a ``(1, n)`` row of every live aircraft."""
    own = _indices_array(own_indices).reshape(-1, 1)
    every = np.arange(int(bs.traf.ntraf), dtype=np.intp).reshape(1, -1)
    return own, every


def _per_own(own: np.ndarray, value) -> np.ndarray:
    """``value(own_idx)`` for each ownship, shaped like ``own`` to broadcast."""
    return np.array(
        [float(value(int(o))) for o in own.ravel()], dtype=np.float64
    ).reshape(own.shape)


class _BroadcastPairs:
    """A pair field defined by one broadcasting function, :meth:`_pairs`.

    ``get_pair``, ``get_pairs`` and ``get_pair_matrix`` are all that function
    evaluated on different index shapes, so they agree by construction.
    """

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        return self._pairs(_indices_array(own_idx), _indices_array(other_indices))

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        own, every = _pair_index_grid(own_indices)
        return self._pairs(own, every)


def _per_aircraft(field: ObsField, idx: np.ndarray) -> np.ndarray:
    """An ownship field's value at each index of ``idx``, shaped like it."""
    flat = np.asarray(field.get_many(idx.ravel()), dtype=np.float64)
    return flat.reshape(idx.shape + flat.shape[1:])


def _track_frame(own: np.ndarray, other: np.ndarray):
    """``(along, cross, v_along, v_cross)``: the intruder's position (nm) and
    velocity (kts) relative to the ownship, in its TRACK frame - along = ahead,
    cross = to the right - from the shared pair geometry. Rotates with the own
    track, so an encounter reads the same at any heading."""
    view = ConflictView(own, other)
    trk = np.radians(_traf_array("trk"))[own]
    s, c = np.sin(trk), np.cos(trk)

    def rotate(east, north):
        return east * s + north * c, east * c - north * s

    along, cross = rotate(view.rel_east_now_nm, view.rel_north_now_nm)
    v_along, v_cross = rotate(view.rel_vel_east_kts, view.rel_vel_north_kts)
    return along, cross, v_along, v_cross


def _fix_projection(
    own_idx: Any,
    other_indices: Any,
    route_offset: int,
    own_eta_mode: str = "projection",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project observed motion onto the *ownship's* active fix (a merge signal).

    Returns ``(approach_dist_nm, intruder_eta_s, own_eta_s, vsep_at_fix_ft,
    has_fix)``, broadcast over the ``own`` / ``other`` index arrays
    where each intruder's straight-line motion is projected to the closest
    approach of the ownship's own fix ``F`` (offset ``0`` = active leg, ``1`` =
    next leg): how near its current trajectory passes ``F`` (horizontal), when
    (ETA), and the signed altitude gap at the merge (intruder minus ownship,
    each projected to its own fix arrival by its vertical speed). ``has_fix`` is
    False for an ownship with no usable fix there; its other values are
    placeholders the caller replaces.

    Privacy: uses only ``F`` (the ownship's *own* plan, offset 0/1 both being
    own legs) and each intruder's *observed* position/velocity/altitude/VS
    (radar/ADS-B grade) - never the intruder's flight plan. This is the
    observable counterpart of reading ``ActiveRouteWaypoint*`` on the intruder,
    which would expose unshared intent and is therefore critic-only.

    ``own_eta_mode`` selects how the OWNSHIP's own ETA is formed, and the two
    sides are deliberately NOT symmetric because the information is not:

    * ``"projection"`` (default, and what every task before v52 used) applies
      the same kinematic closest-approach projection to the ownship. Since
      ``t* = (range / groundspeed) * cos(theta)`` for ``theta`` the angle
      between track and bearing-to-fix, the ownship's own ETA is scaled by its
      instantaneous heading: measured on a v51 rollout it reads 0.998x the
      range/speed ETA while tracking the fix, but 0.29x at 45-90 deg off and
      0.0 beyond 90 deg. That collapse is an artifact - the ownship is *not*
      going to fly straight forever, route guidance turns it back - and it is
      driven by the ownship's own ACTION, which is the observation-feedback
      shape that caused the ``PrevActionNorm`` hysteresis loop.
    * ``"route"`` uses ``range / groundspeed`` for the ownship instead: no
      ``cos`` factor, no clamp. Legitimate precisely because the ownship's plan
      is its OWN knowledge - it knows it is going to ``F``. Stable under
      maneuvering, and monotone in groundspeed, so decelerating to open a gap
      raises the ownship ETA cleanly (speed is the instrument that resolves an
      in-trail merge, so this is the channel that has to stay readable).

    The INTRUDER side keeps the kinematic projection under both modes, and must:
    ``range / groundspeed`` for an intruder would assert "it is flying to MY
    fix", which is exactly the intent this field exists to avoid assuming. Its
    ``cos`` collapse is the honest reading - "this one is not going to my fix" -
    and it is what makes a large ``IntruderFixApproachDistNm`` meaningful.

    Mixing the two models does not corrupt the difference where it is used: a
    genuine merge partner is by construction tracking the fix, so its ``theta``
    is near 0 and its projection equals its range/speed ETA anyway (measured
    0.998x within 5 deg, 0.987x within 20 deg). The models only diverge once the
    intruder is off-bearing to the fix, where the approach-distance gate has
    already flagged the pair as irrelevant.
    """
    if own_eta_mode not in ("projection", "route"):
        raise ValueError(
            f"own_eta_mode must be 'projection' or 'route', got {own_eta_mode!r}."
        )
    own = _indices_array(own_idx)
    other = _indices_array(other_indices)
    fixes = [_active_route_waypoint(int(o), route_offset) for o in own.ravel()]
    has_fix = np.array([fix is not None for fix in fixes]).reshape(own.shape)
    # A fix-less ownship projects onto (0, 0) - finite arithmetic, masked out.
    fix_lat = np.array(
        [0.0 if fix is None else float(fix[0]) for fix in fixes], dtype=np.float64
    ).reshape(own.shape)
    fix_lon = np.array(
        [0.0 if fix is None else float(fix[1]) for fix in fixes], dtype=np.float64
    ).reshape(own.shape)
    lat, lon = _traf_array("lat"), _traf_array("lon")
    trk, gs = _traf_array("trk"), _traf_array("gs")
    alt, vs = _traf_array("alt"), _traf_array("vs")

    def _eta_and_cpa(idx: np.ndarray):
        # Position of each aircraft relative to its ownship's fix, east/north m.
        qdr, dist_nm = qdrdist(fix_lat, fix_lon, lat[idx], lon[idx])
        qdrrad = np.radians(np.asarray(qdr, dtype=np.float64))
        dist_m = np.asarray(dist_nm, dtype=np.float64) * nm
        r_e = dist_m * np.sin(qdrrad)
        r_n = dist_m * np.cos(qdrrad)
        trkrad = np.radians(trk[idx])
        v_e = gs[idx] * np.sin(trkrad)
        v_n = gs[idx] * np.cos(trkrad)
        v2 = np.maximum(v_e * v_e + v_n * v_n, 1e-6)
        # Future closest approach only (t* clamped >= 0): an aircraft receding
        # from the fix reads its current distance, not a past pass.
        tstar = np.maximum(-(r_e * v_e + r_n * v_n) / v2, 0.0)
        cpa_e = r_e + v_e * tstar
        cpa_n = r_n + v_n * tstar
        cpa_nm = np.sqrt(cpa_e * cpa_e + cpa_n * cpa_n) / nm
        return tstar, cpa_nm

    intr_eta, intr_cpa_nm = _eta_and_cpa(other)
    if own_eta_mode == "route":
        # The ownship knows its own plan, so its ETA is a property of the route
        # and the speed it is flying - not of the heading it happens to hold
        # this step. See the docstring for why the two sides differ.
        _own_qdr, own_dist_nm = qdrdist(fix_lat, fix_lon, lat[own], lon[own])
        own_eta_s = (
            np.asarray(own_dist_nm, dtype=np.float64) * nm / np.maximum(gs[own], 1e-3)
        )
    else:
        own_eta_s, _own_cpa = _eta_and_cpa(own)
    # Altitude each aircraft is projected to hold when it reaches the fix
    # (current alt + VS x its own ETA); the merge conflict is where these
    # coincide. Signed intruder-minus-ownship, in feet.
    own_alt_at_fix = alt[own] + vs[own] * own_eta_s
    intr_alt_at_fix = alt[other] + vs[other] * intr_eta
    vsep_ft = (intr_alt_at_fix - own_alt_at_fix) * _M_TO_FT
    return intr_cpa_nm, intr_eta, own_eta_s, vsep_ft, has_fix
