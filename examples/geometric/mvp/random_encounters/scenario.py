"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

from scipy.stats import randint
from bluesky_sandbox.sim.spawn import TowardRoute
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


# Route rules: each gives its spawn region's aircraft their routes.
def _route_ENTRY(request, rng, ctx):
    """Spawn region 'ENTRY': the route each of its aircraft flies."""
    # Straight through a point drawn in the crossing area, on to where that
    # line leaves the airspace: every track crosses the center area, and an
    # aircraft handed back its route flies on outward, not back into the
    # encounter.
    cross = ctx.shapes["crossing"].sample_point(rng)
    track, _ = kwikqdrdist(request.at.lat_deg, request.at.lon_deg, *cross)
    lat, lon = cross
    while ctx.airspace.contains(lat, lon):
        lat, lon = qdrpos(lat, lon, track, 1.0)
    return [{"waypoint": "exit", "sample": RegionBounds(PointFootprint(LatLon(lat, lon)))}]



class RandomEncountersScenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {
            'airspace': RegionBounds(DiskFootprint(LatLon(56.0, 2.0), radius_nm=60, n_vertices=72), None),
    'entry': RegionBounds(AnnularSectorFootprint(LatLon(56.0, 2.0), inner_radius_nm=40, outer_radius_nm=45, bearing_deg=0, half_angle_deg=180), None),
    'crossing': RegionBounds(DiskFootprint(LatLon(56.0, 2.0), radius_nm=8, n_vertices=48), None)
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
            spawn=SpawnConfig(regions=[
    SpawnRegion(
        shape=REGIONS['entry'],
        n_aircraft=randint(low=2, high=7, loc=0),
        params={'alt_ft': 20000, 'spd_kts': (260, 300), 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'A321': 1, 'B738': 1, 'B739': 1, 'E190': 1}),
        assign_route=_route_ENTRY,
        name='ENTRY',
        conflict_free_spawn=True,
    ),
]),
            queryables={
            'exit': Waypoint(lat=56.0, lon=2.0, reach_radius_nm=2, color='cyan', render_shape=False, render_label=False, track_temporal_state=False)
        },
            rotation=rotation,
            groups=groups,
            sampled_waypoints={
            
        },
            waypoint_fields={
            
        },
            shapes=REGIONS,
        )

