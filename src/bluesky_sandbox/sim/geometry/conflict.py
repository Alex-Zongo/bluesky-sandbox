"""Pairwise geometry and conflict detection, once per step.

Two things, each computed once per sim time and cached:

* :func:`conflict_geometry` - every pair's straight-line motion: where ``j`` is
  relative to ``i`` now, how fast, and their closest point of approach (CPA).
  Positions come from BlueSky's ``kwikqdrdist_matrix``, the flat-earth
  projection its conflict detection measures on, so a field built on them
  describes the encounter the detector sees.
* :func:`detection` - BlueSky's conflict detection itself: the configured
  ``CDMETHOD``'s ``detect`` run on the current state, giving which pairs are in
  conflict and in loss of separation, and when each conflict enters the zone.
  Its rule is BlueSky's, never re-derived here.

Arrays are ``N x N`` in ``bs.traf`` order, ``[i, j]`` = ``j`` seen from ``i``.
:class:`ConflictView` slices them by aircraft - an ownship's row over the
others for hooks (``context.conflicts``), any index arrays for fields.

Geometry, for each pair:

- ``rel_east_now_nm`` / ``rel_north_now_nm``  signed position of ``j`` (nm).
- ``horiz_dist_now_nm``                       horizontal distance (nm).
- ``rel_vel_east_kts`` / ``rel_vel_north_kts`` velocity of ``j`` relative to ``i`` (kts).
- ``rel_hspd_kts``                            horizontal relative speed (kts).
- ``tcpa_s``                                  time to the horizontal CPA (s), negative past it.
- ``dcpa_nm``                                 horizontal distance at that CPA (nm).
- ``rel_alt_now_ft`` / ``dalt_now_ft``         altitude of ``j`` above ``i`` (ft), signed / unsigned.
- ``rel_vs_ft_min``                           vertical speed of ``j`` relative to ``i`` (ft/min).
- ``vsep_at_cpa_ft``                          vertical separation at the horizontal CPA (ft).
"""

from __future__ import annotations

from types import SimpleNamespace

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft as _FT
from bluesky.tools.aero import kts as _KTS
from bluesky.tools.aero import nm as _NM
from bluesky.tools.geo import kwikqdrdist_matrix

_M_TO_FT = 1.0 / _FT
_MS_TO_KTS = 1.0 / _KTS
_MS_TO_FTMIN = _M_TO_FT * 60.0

_CACHE: dict = {}


# ---- BlueSky CD settings ------------------------------------------ #
# The live protected zone and lookahead, as CD actually applies them.
# ``bs.settings.asas_*`` is only the config-file default: the sandbox
# publishes ``EnvConfig.pz_radius_nm`` / ``pz_height_ft`` / ``lookahead_s``
# through ZONER / ZONEDH / DTLOOK, which update CD but not settings. Read
# these, not settings, or spawn clearance disagrees with detection.

def cd_lookahead_s() -> float:
    """Conflict-detection lookahead horizon, in seconds.

    Prefers the CD's applied default ``bs.traf.cd.dtlookahead_def`` - a scalar
    set at init from ``asas_dtlookahead`` and updated by the ``DTLOOK`` command
    (how the sandbox applies ``config.lookahead_s``), so the horizon tracks the
    scenario's actual setting rather than the config-file default. Falls back to
    ``bs.settings.asas_dtlookahead`` before CD/traffic exist. Uses the scalar
    default, not the per-aircraft ``cd.dtlookahead`` array (empty pre-traffic).
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    look = getattr(cd, "dtlookahead_def", None)
    if look is None:
        look = getattr(bs.settings, "asas_dtlookahead", 300.0)
    look = float(look)
    return look if look > 0.0 else 300.0


def cd_rpz_m() -> float:
    """Horizontal protected-zone radius, in meters (BlueSky CD ``rpz``).

    Prefers the CD's applied default ``bs.traf.cd.rpz_def`` (meters) - set from
    ``asas_pzr`` and updated by ``ZONER`` (how the sandbox applies
    ``config.pz_radius_nm``) - falling back to ``bs.settings.asas_pzr`` before CD
    exists. A detected conflict has ``dcpa < rpz``, so this is the natural cap on
    horizontal distance at CPA.
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    rpz = getattr(cd, "rpz_def", None)
    if rpz is None:
        rpz = float(getattr(bs.settings, "asas_pzr", 5.0)) * _NM
    rpz = float(rpz)
    return rpz if rpz > 0.0 else 5.0 * _NM


def cd_hpz_m() -> float:
    """Vertical protected-zone height, in meters (BlueSky CD ``hpz``).

    Prefers the CD's applied default ``bs.traf.cd.hpz_def`` (meters) - set from
    ``asas_pzh`` and updated by ``ZONEDH`` (how the sandbox applies
    ``config.pz_height_ft``) - falling back to ``bs.settings.asas_pzh`` before CD
    exists. It is the minimum vertical separation (a vertical loss of separation
    is ``|dalt| < hpz``).
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    hpz = getattr(cd, "hpz_def", None)
    if hpz is None:
        hpz = float(getattr(bs.settings, "asas_pzh", 1000.0)) * _FT
    hpz = float(hpz)
    return hpz if hpz > 0.0 else 1000.0 * _FT



def bluesky_asas_dt_s() -> float | None:
    """BlueSky's default conflict-detection interval, ``bs.settings.asas_dt`` (s).

    Unlike the zone and lookahead above there is no live CD attribute to prefer:
    detection runs on BlueSky's ``asas`` timer, and ``bs.sim.reset()`` returns
    that timer to this setting every episode. BlueSky registers the setting when
    ``bs.init`` imports its traffic module, and ``settings.cfg`` may override it,
    so before either has happened there is nothing to read. That returns
    ``None`` rather than a guess at BlueSky's value.
    """
    value = getattr(bs.settings, "asas_dt", None)
    return None if value is None else float(value)


def _step_key() -> tuple:
    return (getattr(getattr(bs, "sim", None), "simt", None), tuple(bs.traf.id))


def _cached(name):
    """What :func:`_store` kept under ``name`` this sim time and traffic set,
    else ``None``. A reset returns the clock to 0 with the same callsigns, so
    the runtime calls :func:`invalidate_conflict_geometry` on every reset."""
    hit = _CACHE.get(name)
    return hit[1] if hit is not None and hit[0] == _step_key() else None


def _store(name, value):
    _CACHE[name] = (_step_key(), value)
    return value


def invalidate_conflict_geometry() -> None:
    """Drop everything cached; the next read recomputes."""
    _CACHE.clear()


def conflict_geometry() -> SimpleNamespace:
    """Every pair's straight-line motion this step (see the module docstring)."""
    return _cached("geometry") or _store("geometry", _compute())


def _compute() -> SimpleNamespace:
    n = len(bs.traf.id)
    lat = np.asarray(bs.traf.lat, dtype=np.float64)
    lon = np.asarray(bs.traf.lon, dtype=np.float64)
    if n:
        qdr, dist = kwikqdrdist_matrix(lat, lon, lat, lon)
        qdr = np.radians(np.asarray(qdr, dtype=np.float64))
        dist = np.asarray(dist, dtype=np.float64) * _NM  # m
    else:
        qdr = dist = np.zeros((0, 0))
    dx, dy = dist * np.sin(qdr), dist * np.cos(qdr)  # j relative to i, m

    trk = np.radians(np.asarray(bs.traf.trk, dtype=np.float64))
    gs = np.asarray(bs.traf.gs, dtype=np.float64)
    u, v = gs * np.sin(trk), gs * np.cos(trk)
    du, dv = u[None, :] - u[:, None], v[None, :] - v[:, None]
    # As BlueSky's StateBased.detect: the relative speed floored, the miss
    # distance from the Pythagorean form.
    dv2 = np.maximum(du * du + dv * dv, 1e-6)
    tcpa = -(du * dx + dv * dy) / dv2
    dcpa = np.sqrt(np.abs(dist * dist - tcpa * tcpa * dv2))

    alt = np.asarray(bs.traf.alt, dtype=np.float64)
    vs = np.asarray(bs.traf.vs, dtype=np.float64)
    rel_alt = alt[None, :] - alt[:, None]
    rel_vs = vs[None, :] - vs[:, None]

    geom = SimpleNamespace(
        n=n,
        rel_east_now_nm=dx / _NM,
        rel_north_now_nm=dy / _NM,
        horiz_dist_now_nm=dist / _NM,
        rel_vel_east_kts=du * _MS_TO_KTS,
        rel_vel_north_kts=dv * _MS_TO_KTS,
        rel_hspd_kts=np.sqrt(dv2) * _MS_TO_KTS,
        tcpa_s=tcpa,
        dcpa_nm=dcpa / _NM,
        rel_alt_now_ft=rel_alt * _M_TO_FT,
        dalt_now_ft=np.abs(rel_alt) * _M_TO_FT,
        rel_vs_ft_min=rel_vs * _MS_TO_FTMIN,
        vsep_at_cpa_ft=np.abs(rel_alt + rel_vs * np.maximum(tcpa, 0.0)) * _M_TO_FT,
    )
    for value in vars(geom).values():
        if isinstance(value, np.ndarray):
            np.fill_diagonal(value, 0.0)
    return geom


def detection(
    rpz_nm: float | None = None,
    vpz_ft: float | None = None,
    lookahead_s: float | None = None,
) -> SimpleNamespace:
    """BlueSky's conflict detection on the current state, as ``N x N`` arrays:
    ``in_conf`` and ``in_los`` (bool), and ``tlos_s`` - when each conflict
    enters the protected zone (BlueSky's ``tLOS``), ``inf`` for no conflict.

    The configured detector's own ``detect``, run now rather than read from its
    last ``asas`` tick. With no argument it scores BlueSky's zone and lookahead
    (per aircraft, as ZONER / ZONEDH / DTLOOK set them); each argument replaces
    one for every aircraft - a wider zone to shape against, a shorter horizon
    to show - still under BlueSky's rule.
    """
    name = ("detection", rpz_nm, vpz_ft, lookahead_s)
    return _cached(name) or _store(name, _detect(bs.traf, rpz_nm, vpz_ft, lookahead_s))


def _detect(traffic, rpz_nm=None, vpz_ft=None, lookahead_s=None) -> SimpleNamespace:
    """BlueSky's ``detect`` over ``traffic`` (``bs.traf``, or one standing in
    for it), laid out by index."""
    cd = bs.traf.cd
    n = len(traffic.id)

    def zone(override, live, scale):
        return np.asarray(live, dtype=np.float64)[:n] if override is None else np.full(n, float(override) * scale)

    confpairs, lospairs, *_rest, tlos = cd.detect(
        traffic,
        traffic,
        zone(rpz_nm, cd.rpz, _NM),
        zone(vpz_ft, cd.hpz, _FT),
        zone(lookahead_s, cd.dtlookahead, 1.0),
    )
    index = {acid: i for i, acid in enumerate(traffic.id)}
    found = SimpleNamespace(
        in_conf=np.zeros((n, n), dtype=bool),
        in_los=np.zeros((n, n), dtype=bool),
        tlos_s=np.full((n, n), np.inf),
    )
    for (a, b), t in zip(confpairs, tlos):
        found.in_conf[index[a], index[b]] = True
        found.tlos_s[index[a], index[b]] = t
    for a, b in lospairs:
        found.in_los[index[a], index[b]] = True
    return found


class ConflictView:
    """The pair arrays sliced by aircraft index: ``view.tcpa_s`` is
    ``conflict_geometry().tcpa_s[own, others]``.

    ``ConflictView(i)`` is ownship ``i``'s row over every other aircraft, in
    ``bs.traf`` order - ``context.conflicts`` in a hook, aligned with the
    observation's intruder rows. ``own`` and ``others`` may also be index
    arrays that broadcast, as the pair fields use it.
    """

    __slots__ = ("_geom", "_other", "_own")

    def __init__(self, own, others=None) -> None:
        self._geom = conflict_geometry()
        if others is None:
            others = np.array([j for j in range(self._geom.n) if j != own], dtype=np.intp)
        self._own = own
        self._other = others

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._geom, name)[self._own, self._other]


def _conflict_window(view, rpz_nm: float, vpz_ft: float) -> tuple[np.ndarray, ...]:
    """``(valid, tinconf, toutconf, d0, dvs)``: when each pair is inside both the
    ``rpz_nm`` and the ``vpz_ft`` band at once, if ever - ``d0`` / ``dvs`` the
    signed relative altitude (ft) and its rate (ft/s). Unlike
    :func:`detection` it is not cut at a lookahead, and gives the window's end:
    what the windowed measures below sample within."""
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    dcpa = np.asarray(view.dcpa_nm, dtype=np.float64)

    swhor = dcpa < rpz_nm
    vrel_nm_s = np.asarray(view.rel_hspd_kts, dtype=np.float64) / 3600.0
    dtinhor = np.sqrt(np.maximum(0.0, rpz_nm**2 - dcpa**2)) / np.maximum(vrel_nm_s, 1e-9)
    tinhor = np.where(swhor, tcpa - dtinhor, np.inf)
    touthor = np.where(swhor, tcpa + dtinhor, -np.inf)

    d0 = np.asarray(view.rel_alt_now_ft, dtype=np.float64)
    dvs = np.asarray(view.rel_vs_ft_min, dtype=np.float64) / 60.0
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # as BlueSky
    t_hi = (vpz_ft - d0) / dvs
    t_lo = (-vpz_ft - d0) / dvs

    tinconf = np.maximum(tinhor, np.minimum(t_hi, t_lo))
    toutconf = np.minimum(touthor, np.maximum(t_hi, t_lo))
    valid = swhor & (tinconf <= toutconf) & (toutconf > 0.0)
    return valid, tinconf, toutconf, d0, dvs


def windowed_min_vsep_ft(view: ConflictView, rpz_nm: float, vpz_ft: float) -> np.ndarray:
    """Minimum ``|relative altitude|`` (ft) over the shared 3-D conflict window.

    A *derived* quantity (parameterized by ``rpz``/``vpz``), and the single shared implementation
    of the vertical separation the cost's penetration term (``r_v``) grades: the
    signed rel-alt line ``|d0 + dvs*t|`` sampled at its in-window minimum - the
    vertical zero-crossing ``-d0/dvs`` clipped into ``[max(tinconf, 0), toutconf]``
    (:func:`_conflict_window`).

    Supersedes ``vsep_at_cpa_ft`` (vertical sep at the *horizontal* CPA instant)
    for conflict pairs: that measure reads ~0 for altitude-crossing traffic whose
    vertical crossing is offset in time from the horizontal CPA. Where no valid
    3-D window exists (a safe miss) it falls back to ``vsep_at_cpa_ft`` so a
    horizontally-distant pair keeps its classic, sensible reading.

    UNSIGNED by construction, and necessarily so: at the in-window minimum of a
    crossing pair the signed line is ~0, so a signed version of *this* measure
    would carry no direction exactly where direction matters. Escape direction
    comes from :func:`windowed_signed_vsep_at_entry_ft` instead.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, toutconf, d0, dvs = _conflict_window(view, rpz_nm, vpz_ft)

    # Vertical sep at its in-window minimum (the rel-alt zero-crossing clamped
    # into the shared window), exactly the cost's ``v_min``.
    w0 = np.maximum(tinconf, 0.0)
    tv = np.clip(-d0 / dvs, w0, toutconf)
    v_min = np.abs(d0 + dvs * tv)
    return np.where(valid, v_min, np.asarray(view.vsep_at_cpa_ft, dtype=np.float64))


def windowed_min_hsep_nm(view: ConflictView, rpz_nm: float, vpz_ft: float) -> np.ndarray:
    """Minimum horizontal separation (nm) over the shared 3-D conflict window.

    The horizontal twin of :func:`windowed_min_vsep_ft`, and the single shared
    implementation of the separation the cost's penetration term (``r_h``)
    grades: the hyperbola ``sqrt(dcpa^2 + (vrel*(t - tcpa))^2)`` sampled at its
    in-window minimum - ``tcpa`` clipped into ``[max(tinconf, 0), toutconf]``
    (the SAME window the vertical measures key off).

    Differs from plain ``dcpa`` only by that clip, but the clip is the whole
    point: ``dcpa`` is the miss over ALL time, whereas the cost charges the pair
    only while it is simultaneously inside both bands. When the horizontal CPA
    falls outside that window - the pair is already separating horizontally by
    the time it converges vertically, or it leaves the vertical band before
    reaching the horizontal CPA - the two diverge, always with
    ``h_min >= dcpa``. Reading ``dcpa`` there overstates the horizontal threat
    relative to what the cost actually charges, inviting maneuvers that earn
    nothing on the cost channel and are paid for in track miles.

    Where no valid 3-D window exists (a safe miss) it falls back to ``dcpa_nm``,
    mirroring :func:`windowed_min_vsep_ft`'s fallback to ``vsep_at_cpa_ft``.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, toutconf, _d0, _dvs = _conflict_window(view, rpz_nm, vpz_ft)

    dcpa = np.asarray(view.dcpa_nm, dtype=np.float64)
    vrel_nm_s = np.asarray(view.rel_hspd_kts, dtype=np.float64) / 3600.0
    # Horizontal sep at its in-window minimum (tcpa clamped into the shared
    # window), exactly the cost's ``h_min``.
    w0 = np.maximum(tinconf, 0.0)
    th = np.clip(tcpa, w0, toutconf)
    h_min = np.sqrt(dcpa**2 + (vrel_nm_s * (th - tcpa)) ** 2)
    return np.where(valid, h_min, dcpa)


def windowed_signed_vsep_at_entry_ft(
    view: ConflictView, rpz_nm: float, vpz_ft: float
) -> np.ndarray:
    """SIGNED relative altitude (ft) at entry to the shared 3-D conflict window.

    Positive = the intruder is ABOVE ownship as the encounter opens. The vertical
    counterpart of the signed horizontal ``RelPosAtCpa*`` primitives, and the
    direction half of what :func:`windowed_min_vsep_ft` reports the magnitude of:
    together they say "you will pass 200 ft apart, with the intruder above".

    Sampled at window ENTRY (``max(tinconf, 0)``), not at the in-window minimum,
    precisely because the minimum of a crossing pair sits AT the rel-alt zero
    where the sign is undefined and flips. Entry is the last instant at which the
    geometry still has an unambiguous side, so it is the instant that answers
    "climb or descend?". Where no valid 3-D window exists (a safe miss) it falls
    back to the signed rel-alt at the horizontal CPA, mirroring
    :func:`windowed_min_vsep_ft`'s fallback to ``vsep_at_cpa_ft``.
    """
    tcpa = np.asarray(view.tcpa_s, dtype=np.float64)
    if tcpa.size == 0:
        return tcpa
    valid, tinconf, _toutconf, d0, dvs = _conflict_window(view, rpz_nm, vpz_ft)

    entry = d0 + dvs * np.maximum(tinconf, 0.0)
    miss = d0 + dvs * np.maximum(tcpa, 0.0)
    return np.where(valid, entry, miss)
