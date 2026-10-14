"""Conflict fields, per intruder: BlueSky's conflict detection itself
(:func:`~bluesky_sandbox.sim.geometry.conflict.detection`), and the shared pair
geometry's closest approach and conflict window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from bluesky.tools.aero import nm

from bluesky_sandbox.sim.geometry.conflict import (
    ConflictView,
    detection,
    windowed_min_hsep_nm,
    windowed_min_vsep_ft,
    windowed_signed_vsep_at_entry_ft,
)
from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m as _cd_hpz_m
from bluesky_sandbox.sim.geometry.conflict import cd_lookahead_s as _cd_lookahead_s
from bluesky_sandbox.sim.geometry.conflict import cd_rpz_m as _cd_rpz_m

from .._common import _M_TO_FT
from .._pairs import _BroadcastPairs, _per_own
from ..base import ObsMeta, ObsQuantity, PairObsField, Unit


@dataclass(frozen=True)
class _ConflictGeomPairField(_BroadcastPairs, PairObsField):
    """Intruder field read from the shared pair geometry
    (:class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView`): every
    intruder, every step, conflict or not."""

    _geom_attr: ClassVar[str] = ""

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = ConflictView(own, other)
        return np.asarray(getattr(view, self._geom_attr), dtype=np.float64)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return float(getattr(ConflictView(own_idx, other_idx), self._geom_attr))


@dataclass(frozen=True)
class _WindowedConflictPairField(_ConflictGeomPairField):
    """Conflict field whose value is measured over a 3-D conflict *window*.

    Adds an optional zone override. The window defaults to the live CD
    ``rpz``/``hpz``, which is right whenever the cost grades the true protected
    zone. Tasks that grade a *buffered* zone - a shaped margin wider than the PZ,
    so resolutions are not trained to the PZ edge - must widen the observation's
    window to match, or the marginal encounters the buffer exists to charge are
    reported to the policy as clean misses (they fall to the field's safe-miss
    branch) while the cost bills them.

    Overriding here rather than via ``config.pz_radius_nm`` / ``pz_height_ft`` is
    deliberate: those move BlueSky's CD zone, which is what ``bs.traf.cd.lospairs``
    - and therefore any loss-of-separation cost channel - is defined against.
    Widening the CD zone to fix an observation would silently redefine what counts
    as a LoS. The observation window and the LoS predicate are different things and
    are configured separately.

    Bound defaults still follow the CD zone, so an override wants explicit
    ``low``/``high`` alongside it - otherwise the normalizer re-clips away the very
    band the wider window just exposed.
    """

    rpz_nm: Annotated[
        float | None, "window horizontal radius nm; None = CD rpz at runtime"
    ] = None
    vpz_ft: Annotated[
        float | None, "window vertical half-height ft; None = CD hpz at runtime"
    ] = None

    def _detection(self, lookahead_s: float | None = None):
        """BlueSky's detection in this field's zone (and ``lookahead_s``)."""
        self._window_zone()
        if lookahead_s is not None and lookahead_s <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} lookahead_s must be positive, got {lookahead_s}."
            )
        return detection(self.rpz_nm, self.vpz_ft, lookahead_s)

    def _window_zone(self) -> tuple[float, float]:
        """``(rpz_nm, vpz_ft)`` for the window: overrides, else the live CD zone."""
        rpz = _cd_rpz_m() / nm if self.rpz_nm is None else float(self.rpz_nm)
        vpz = _cd_hpz_m() * _M_TO_FT if self.vpz_ft is None else float(self.vpz_ft)
        if rpz <= 0.0 or vpz <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} window zone must be positive, "
                f"got rpz_nm={rpz}, vpz_ft={vpz}."
            )
        return rpz, vpz


@dataclass(frozen=True)
class ConflictHorizontalDistAtCpaNm(_ConflictGeomPairField):
    """Predicted *horizontal* separation at CPA, in nm, for every intruder.

    Bounds are dynamic: unless given, ``(0, CD rpz)`` - so a clipped normalizer
    grades the danger band and saturates every safer miss at the PZ edge,
    matching the cost's ``r_h`` support.
    """

    meta = ObsMeta(
        "conflict_horizontal_dist_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    _geom_attr: ClassVar[str] = "dcpa_nm"
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictVerticalSepAtCpaFt(_WindowedConflictPairField):
    """Predicted minimum *vertical* separation over the 3-D conflict window, in ft.

    Reads the SAME windowed measure the cost's penetration term (``r_v``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_vsep_ft` - the rel-alt
    line at its in-window minimum, NOT the vertical sep at the *horizontal* CPA
    instant. The CPA-instant reading collapsed to ~0 for altitude-crossing traffic
    whose vertical crossing is offset in time from the horizontal CPA, hiding
    developing vertical conflicts from the policy that the cost was charging; the
    windowed minimum is exactly what the cost sees. A safe miss (no valid 3-D
    window) falls back to the classic CPA-instant value. Bounds are dynamic: unless
    given, ``(0, CD hpz)`` - a clipped normalizer grades the danger band and
    saturates every safe pair at the PZ edge.
    """

    meta = ObsMeta(
        "conflict_vertical_sep_at_cpa_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "altitude ft lower bound; None = 0"] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_vsep_ft(ConflictView(own, other), *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, np.array([other_idx]))
        return float(windowed_min_vsep_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_hpz_m() * _M_TO_FT))


@dataclass(frozen=True)
class ConflictHorizontalSepAtCpaNm(_WindowedConflictPairField):
    """Predicted minimum *horizontal* separation over the 3-D conflict window, nm.

    The horizontal twin of :class:`ConflictVerticalSepAtCpaFt`, reading the SAME
    windowed measure the cost's penetration term (``r_h``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_hsep_nm` - the
    separation hyperbola at its in-window minimum, NOT the miss distance at the
    unconstrained CPA.

    **Why this is not just ``dcpa``.** :class:`ConflictHorizontalDistAtCpaNm` and
    the :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm` pair
    all report ``dcpa``, the miss over ALL time. The cost charges the pair only
    while it is inside both bands at once, so when the horizontal CPA falls
    outside that window the two diverge - always with ``h_min >= dcpa``, i.e. the
    ``dcpa`` reading is the more alarming one. A policy steering on it maneuvers
    for encounters the cost never bills, and pays for the detour in track miles.

    This is the horizontal half of the same correction
    :class:`ConflictVerticalSepAtCpaFt` made vertically; that one was the more
    urgent because sampling vertical separation at the *horizontal* CPA instant
    was first-order wrong (the wrong axis's extremum time), whereas ``dcpa`` is
    only window-truncation wrong. Keep the signed ``RelPosAtCpa*`` pair alongside
    this field - they carry the pass DIRECTION, which this magnitude does not.

    Bounds are dynamic: unless given, ``(0, CD rpz)`` - a clipped normalizer
    grades the danger band and saturates every safer miss at the PZ edge, matching
    the cost's ``r_h`` support. Pass ``rpz_nm``/``vpz_ft`` (and matching
    ``low``/``high``) when the cost grades a buffered zone rather than the CD one.
    """

    meta = ObsMeta(
        "conflict_horizontal_sep_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_hsep_nm(ConflictView(own, other), *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, np.array([other_idx]))
        return float(windowed_min_hsep_nm(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictSignedVerticalSepAtEntryFt(_WindowedConflictPairField):
    """SIGNED vertical separation, in ft, as the 3-D conflict window opens.

    Positive = intruder ABOVE ownship. The vertical counterpart of the signed
    horizontal :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm`
    pair, and the missing half of :class:`ConflictVerticalSepAtCpaFt`: that field
    grades HOW CLOSE the pair comes vertically (matching the cost's ``r_v`` by
    construction) but is an absolute value, so on its own it never says which way
    to go. This one supplies the side.

    Sampled at window entry rather than at the in-window minimum on purpose - the
    minimum of a crossing pair sits at the rel-alt zero, where the sign is
    undefined and flips, so a signed *minimum* would be blank exactly where the
    direction matters. See
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_signed_vsep_at_entry_ft`.

    Bounds are dynamic: unless given, ``(-CD hpz, +CD hpz)``. Pair with
    :class:`SymmetricNormalizer` so the whole PZ band spans ``[-1, 1]`` and every
    safe pair saturates at the correct end - unlike a wide ``relative_alt_ft``,
    where the danger band is squeezed into a few percent of the input range.
    """

    meta = ObsMeta(
        "conflict_signed_vertical_sep_at_entry_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude ft lower bound; None = -CD PZ height at runtime"
    ] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = +CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = ConflictView(own, other)
        return windowed_signed_vsep_at_entry_ft(view, *self._window_zone())

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, np.array([other_idx]))
        return float(windowed_signed_vsep_at_entry_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        hpz_ft = _cd_hpz_m() * _M_TO_FT
        return self._dynamic_or_configured_bounds(lambda: (-hpz_ft, hpz_ft))


@dataclass(frozen=True)
class ConflictTcpaS(_ConflictGeomPairField):
    """Time to horizontal CPA (s), for every intruder; negative past it.

    Bounds are dynamic: unless given, ``(-lookahead, +lookahead)``.
    """

    meta = ObsMeta(
        "conflict_tcpa_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    _geom_attr: ClassVar[str] = "tcpa_s"
    low: Annotated[
        float | None, "time s lower bound; None = -CD lookahead at runtime"
    ] = None
    high: Annotated[
        float | None, "time s upper bound; None = +CD lookahead at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (-_cd_lookahead_s(), _cd_lookahead_s())
        )


@dataclass(frozen=True)
class ConflictTlosS(_WindowedConflictPairField):
    """Time until the pair enters the protected zone (BlueSky ``tLOS``), s.

    BlueSky's conflict detection, run on the current state
    (:func:`~bluesky_sandbox.sim.geometry.conflict.detection`). A pair not in
    conflict reads the high bound (the lookahead); one already in the zone, 0.
    Bounds are dynamic: unless given, ``(0, lookahead)``.
    """

    meta = ObsMeta(
        "conflict_tlos_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[float | None, "time s lower bound; None = 0"] = None
    high: Annotated[
        float | None, "time s upper bound; None = CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        tlos = self._detection().tlos_s[own, other]
        return np.clip(tlos, 0.0, _per_own(own, lambda o: self.bounds(o)[1]))

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        tlos = float(self._detection().tlos_s[own_idx, other_idx])
        return min(max(tlos, 0.0), self.bounds(own_idx)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class InConf(_WindowedConflictPairField):
    """1.0 when BlueSky's conflict detection flags the pair in conflict now.

    BlueSky's own ``detect``, run on the current state
    (:func:`~bluesky_sandbox.sim.geometry.conflict.detection`): the pair is
    predicted to enter the protected zone within the lookahead, or is inside it.
    A dedicated flag, because :class:`ConflictTlosS` clips "no conflict" to its
    high bound, where a conflict entering exactly at the horizon also sits.

    Pair it with :class:`InLosNow`: this one is the approach CD alerts on, that
    one the breach. A pair in the zone is in conflict too, so ``in_los_now = 1``
    comes with ``in_conf = 1``; the reverse gap is the warning time a
    resolution has to act in.

    ``rpz_nm`` / ``vpz_ft`` / ``lookahead_s`` default to BlueSky's and override
    together with the cost's zone - see :class:`_WindowedConflictPairField` for
    why a buffered zone belongs here and not in ``config.pz_radius_nm``. Leave
    ``normalizer`` unset: the value is already 0/1.
    """

    meta = ObsMeta("in_conf", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    lookahead_s: Annotated[
        float | None, "detection horizon s; None = CD lookahead at runtime"
    ] = None
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return self._detection(self.lookahead_s).in_conf[own, other].astype(np.float32)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return float(self._detection(self.lookahead_s).in_conf[own_idx, other_idx])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class InLosNow(_BroadcastPairs, PairObsField):
    """1.0 when the intruder is inside the ownship's protected zone now - a loss
    of separation, as BlueSky's conflict detection scores it on the current
    state (:func:`~bluesky_sandbox.sim.geometry.conflict.detection`).

    A dedicated flag because a policy penalized for LoS should see it directly:
    continuous features bury it in a few percent of their range, or - as
    :class:`ConflictTlosS`, which reads 0 once inside - at one end of it. Leave
    ``normalizer`` unset: the value is already 0/1.
    """

    meta = ObsMeta("in_los_now", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return detection().in_los[own, other].astype(np.float32)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return float(detection().in_los[own_idx, other_idx])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()
