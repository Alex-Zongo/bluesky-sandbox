"""Static environment config for this generated task."""

from __future__ import annotations

from bluesky_sandbox.config import EnvConfig

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields import queryables as qobs
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    CircularNormalizer, MinMaxNormalizer, PowerNormalizer, SignedPowerNormalizer, StepNormalizer, SymmetricNormalizer,
)

CONFIG = EnvConfig(
    obs_fields=list((
            qobs.WaypointDistanceNm(query_name='VALPU', low=0, high=80, normalizer=PowerNormalizer(power=2, clipped=True)),
            qobs.WaypointEteS(query_name='VALPU', low=0, high=2400, normalizer=MinMaxNormalizer(clipped=True)),
            qobs.WaypointBearingDeg(query_name='VALPU', normalizer=CircularNormalizer()),
            qobs.WaypointRouteReached(query_name='GM405'),
            qobs.WaypointRouteReached(query_name='GM410'),
            qobs.WaypointRouteActive(query_name='VALPU'),
            obs.AltFt(low=4000, high=19000, normalizer=MinMaxNormalizer(clipped=True)),
            obs.CasKts(low=180, high=320, normalizer=MinMaxNormalizer(clipped=True)),
            obs.ApCasKts(low=180, high=320, normalizer=MinMaxNormalizer(clipped=True)),
            obs.VsFtMin(normalizer=SymmetricNormalizer(clipped=True)),
        )),
    intruder_obs_fields=list((
            qobs.WaypointDistanceNm(query_name='VALPU').relative_to_own(low=-80, high=80, normalizer=SignedPowerNormalizer(power=2, clipped=True)),
            qobs.WaypointEteS(query_name='VALPU').relative_to_own(low=-2400, high=2400, normalizer=SymmetricNormalizer(clipped=True)),
            qobs.WaypointRouteActive(query_name='VALPU'),
            qobs.WaypointRouteReached(query_name='GM405'),
            qobs.WaypointRouteReached(query_name='GM410'),
            qobs.WaypointRouteReached(query_name='VALPU'),
            qobs.WaypointBearingDeg(query_name='VALPU', normalizer=CircularNormalizer()),
            obs.AltFt().relative_to_own(low=-5000, high=5000, normalizer=SymmetricNormalizer(clipped=True)),
            obs.ApAltFt().relative_to_own(low=-5000, high=5000, normalizer=SymmetricNormalizer(clipped=True)),
            obs.ActiveRouteWaypointAltDiffFt(low=-5000, high=5000, normalizer=SymmetricNormalizer(clipped=True)),
            obs.CasKts().relative_to_own(low=-100, high=100, normalizer=SymmetricNormalizer(clipped=True)),
            obs.DistToOwnNm(low=0, high=120, normalizer=PowerNormalizer(power=2, clipped=True)),
            obs.InConf(),
            obs.ConflictTcpaS(normalizer=SymmetricNormalizer(clipped=True)),
            obs.ConflictHorizontalDistAtCpaNm(low=0, high=10, normalizer=MinMaxNormalizer(clipped=True)),
            obs.ConflictVerticalSepAtCpaFt(normalizer=MinMaxNormalizer(clipped=True)),
        )),
    action_fields=list((
            act.DirectToWaypoint(waypoint='VALPU', after=['GM405', 'GM410']),
            act.Clearance(act.ApSpdDeltaKts(low=-50, high=0, command_floor=200, grid=act.Grid(10.0, on='target'), normalizer=StepNormalizer(steps_each_way=5, steps_up=0, include_zero=False)), lock='captured'),
        )),
    dt=10,
    simdt=0.5,
    asas_dt=None,
    cd_method='CSTATEBASED',
    reso_method=None,
    pz_radius_nm=3,
    pz_height_ft=999,
    lookahead_s=300,
    performance_model='openap',
    wind_dir_deg=270,
    wind_kts=0,
    turbulence_kts=0,
    gust_tau_s=30,
)
