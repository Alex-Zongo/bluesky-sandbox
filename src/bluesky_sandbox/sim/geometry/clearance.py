"""Spawn clearance: is a *candidate* state clear of the live traffic?

Two moments, one zone:

* :func:`inside_separation_zone` - the *present* position only, which is what a
  steady-state ``maintain`` top-up needs: within the zone of any aircraft, on
  BlueSky's flat-earth distance (``kwikdist``).
* :func:`predicted_conflict` - over the lookahead too, so the aircraft is not on
  course to conflict either. The candidate is not in ``bs.traf`` yet, so it is
  put alongside the traffic and BlueSky's conflict detection is asked
  (:func:`~.conflict._detect`) - the rule that will score it once it flies.

Both default every margin to the live CD values via :func:`~.conflict.cd_rpz_m`
/ :func:`~.conflict.cd_hpz_m` / :func:`~.conflict.cd_lookahead_s` - *not*
``bs.settings``. ``EnvConfig.pz_radius_nm`` / ``pz_height_ft`` / ``lookahead_s``
reach CD through ZONER / ZONEDH / DTLOOK and never write back to settings, so
reading settings would clear spawns against a zone the detector does not use.
"""

from __future__ import annotations

from types import SimpleNamespace

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vcas2tas
from bluesky.tools.geo import kwikdist

from bluesky_sandbox.sim.weather import WindField

from .conflict import _detect, cd_hpz_m, cd_lookahead_s, cd_rpz_m

__all__ = ["inside_separation_zone", "predicted_conflict"]

_CANDIDATE = "\0candidate"


def inside_separation_zone(
    lat_deg: float,
    lon_deg: float,
    alt_ft: float,
    *,
    sep_nm: float | None = None,
    sep_ft: float | None = None,
) -> bool:
    """Whether any live aircraft shares this point's separation zone - inside
    the horizontal radius **and** the vertical band, as BlueSky scores a loss of
    separation. Each margin defaults to the live CD zone; a non-positive one
    means no zone, so nothing can breach it."""
    n = int(bs.traf.ntraf)
    rpz_nm, vpz_ft = _zone(sep_nm, sep_ft)
    if n == 0 or rpz_nm <= 0.0 or vpz_ft <= 0.0:
        return False
    dist_nm = kwikdist(
        float(lat_deg),
        float(lon_deg),
        np.asarray(bs.traf.lat[:n], dtype=np.float64),
        np.asarray(bs.traf.lon[:n], dtype=np.float64),
    )
    dalt_ft = np.abs(np.asarray(bs.traf.alt[:n], dtype=np.float64) / ft - float(alt_ft))
    return bool(np.any((dist_nm < rpz_nm) & (dalt_ft < vpz_ft)))


def predicted_conflict(
    lat_deg: float,
    lon_deg: float,
    alt_ft: float,
    hdg_deg: float,
    cas_kts: float,
    *,
    sep_nm: float | None = None,
    sep_ft: float | None = None,
    lookahead_s: float | None = None,
    wind: WindField | None = None,
) -> bool:
    """Whether a candidate spawn state, flown level and straight, is in conflict
    with the live traffic within ``lookahead_s`` - or already inside its zone.

    Each margin defaults to the live CD value, so an unset spawn requirement is
    exactly what the detector will score once the episode runs. Passing a value
    smaller than CD's own zone clears spawns that CD already counts as conflicts
    - deliberate for a conflict-seeded scenario, a mistake otherwise;
    ``SpawnConfig.region_spawn_separation`` warns about it rather than silently
    obliging.

    ``wind`` is the airspace's wind field. Traffic ground speeds include it, so
    the candidate's air vector gets its *mean* added: gusts are zero-mean and
    decorrelate within a step, so extrapolating one over the lookahead would
    assert a persistence it lacks.
    """
    alt_m = float(alt_ft) * ft
    tas = float(vcas2tas(float(cas_kts) * kts, alt_m))
    trk = np.radians(float(hdg_deg))
    east, north = tas * np.sin(trk), tas * np.cos(trk)
    if wind is not None and not wind.is_still:
        wind_n, wind_e = wind.mean_ne_ms()
        north += wind_n
        east += wind_e
    # Inside the zone already is checked apart: BlueSky's compiled detector
    # flags nothing for a pair at one spot with one velocity (its CPA is 0/0).
    if inside_separation_zone(lat_deg, lon_deg, alt_ft, sep_nm=sep_nm, sep_ft=sep_ft):
        return True
    n = int(bs.traf.ntraf)
    rpz_nm, vpz_ft = _zone(sep_nm, sep_ft)
    if n == 0 or rpz_nm <= 0.0 or vpz_ft <= 0.0:
        return False

    def plus(name, value):
        return np.append(np.asarray(getattr(bs.traf, name), dtype=np.float64)[:n], value)

    traffic = SimpleNamespace(
        ntraf=n + 1,
        id=[*bs.traf.id[:n], _CANDIDATE],
        lat=plus("lat", lat_deg),
        lon=plus("lon", lon_deg),
        alt=plus("alt", alt_m),
        trk=plus("trk", float(np.degrees(np.arctan2(east, north)) % 360.0)),
        gs=plus("gs", float(np.hypot(east, north))),
        vs=plus("vs", 0.0),
    )
    look = cd_lookahead_s() if lookahead_s is None else float(lookahead_s)
    found = _detect(traffic, rpz_nm, vpz_ft, look).in_conf
    return bool(found[-1].any() or found[:, -1].any())


def _zone(sep_nm: float | None, sep_ft: float | None) -> tuple[float, float]:
    """``(rpz_nm, vpz_ft)``: the given margins, else the live CD zone."""
    rpz_nm = cd_rpz_m() / nm if sep_nm is None else float(sep_nm)
    vpz_ft = cd_hpz_m() / ft if sep_ft is None else float(sep_ft)
    return rpz_nm, vpz_ft
