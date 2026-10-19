"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint, BooleanFootprint, BoxFootprint, ConstantAltitudeBand,
    DiskFootprint, LatLon, LinearAltitudeBand, PointFootprint, PolygonFootprint,
    RadialAltitudeBand, RegionBounds, SectorFootprint, VertexAltitudeBand,
    GeneratedFootprint, Blob, ConvexPolygon, VoronoiSectors, PlacedFootprint, InRegion,
    MovingFootprint, Drift, Spin, Grow,
)
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.spawn import (
    PlanContext, PlannedSource, SpawnConfig, SpawnRegion, SpawnRequest, nearest_entry,
)
from bluesky.tools.geo import kwikqdrdist, qdrpos

ALT_FT = 20_000
CAS_KTS = 280.0
AC_TYPE = "A320"
ROUTES = {"STAR_W": "W1", "STAR_E": "E1"}   # each route, by its entry fix
SPAWN_BEFORE_NM = 5.0             # spawn this far before the entry fix, on the inbound line
IN_TRAIL_S = (90.0, 150.0)        # gaps between consecutive aircraft on one branch
PER_BRANCH = (2, 4)               # aircraft per branch, inclusive



def _point(center, bearing_deg, dist_nm):
    """The point dist_nm from center, on that bearing from it."""
    lat, lon = qdrpos(center.lat_deg, center.lon_deg, bearing_deg, dist_nm)
    return LatLon(float(lat), float(lon))


# Spawn sources: each plans its aircraft at reset.
def _plan_star(rng, ctx):
    """Spawn source 'star': this episode's aircraft."""
    # This episode's aircraft, each a SpawnRequest with its time: read any data
    # (scenario setup can load a file once), draw from any distribution.
    # ctx: the episode's shapes, queryables, airspace and spawn config.

    merge = ctx.shapes["MERGE"].center
    out = []

    for route, entry in ROUTES.items():
        fix = ctx.shapes[entry].center
        hdg, _ = kwikqdrdist(fix.lat_deg, fix.lon_deg, merge.lat_deg, merge.lon_deg)  # inbound track
        at = _point(fix, hdg + 180.0, SPAWN_BEFORE_NM)      # on the inbound line, before the fix
        t = rng.uniform(0.0, 120.0)                           # this branch's first spawn
        for k in range(int(rng.integers(PER_BRANCH[0], PER_BRANCH[1] + 1))):
            out.append(SpawnRequest(at=at, alt_ft=ALT_FT, spd_kts=CAS_KTS, hdg_deg=hdg, actype=AC_TYPE,
                                    callsign=f"{route[-1]}{k + 1:02d}", route=route, time_s=t))
            t += rng.uniform(*IN_TRAIL_S)
    return out


SPAWN_SOURCES = [
    PlannedSource(_plan_star, name='star', when_blocked='defer', max_aircraft=10),
]



class MergeScenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {
            'airspace': RegionBounds(DiskFootprint(LatLon(56, 2), radius_nm=60, n_vertices=72), None),
    'exit': RegionBounds(PointFootprint(LatLon(56, 2)), None),
    'MERGE': RegionBounds(PointFootprint(LatLon(56, 2)), None),
    'W1': RegionBounds(PointFootprint(LatLon(56.4687, 1.1462)), None),
    'E1': RegionBounds(PointFootprint(LatLon(56.4687, 2.8538)), None),
    'IAF': RegionBounds(PointFootprint(LatLon(55.5831, 2)), None),
    'MERGE_REGION': RegionBounds(BoxFootprint(55.4971322835021, 56.10088203413953, 1.789618424743665, 2.218314535095246), None)
        }
        transform = None
        rotation = None
        groups = None
        if transform and transform.get("rotation"):
            rot = transform["rotation"]
            pivot = rot.get("pivot")
            rotation = {"angle": rot["angle_deg"], "pivot": tuple(pivot) if pivot else None}
        if transform and transform.get("groups"):
            groups = tuple(transform["groups"])
        super().__init__(
            airspace_bounds=REGIONS['airspace'],
            spawn=SpawnConfig(regions=[], sources=SPAWN_SOURCES, routes={'STAR_W': ['W1', 'MERGE', 'IAF'], 'STAR_E': ['E1', 'MERGE', 'IAF']}),
            queryables={
            'exit': Waypoint(at=REGIONS['exit'], reach_radius_nm=2, color='cyan', render_shape=False, render_label=False, track_temporal_state=False),
    'MERGE': Waypoint(at=REGIONS['MERGE'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'W1': Waypoint(at=REGIONS['W1'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'E1': Waypoint(at=REGIONS['E1'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'IAF': Waypoint(at=REGIONS['IAF'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'MERGE_REGION': QueryRegion(REGIONS['MERGE_REGION'], color='cyan', render_shape=True, render_label=True, track_temporal_state=False)
        },
            rotation=rotation,
            groups=groups,
            sampled_waypoints={
            
        },
            waypoint_fields={
            
        },
            shapes=REGIONS,
        )

