"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

from scipy.stats import betabinom
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


class PointMergeScenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {
            'ADOPI': RegionBounds(PointFootprint(LatLon(60.323611, 9.383333)), None),
    'NIVDU': RegionBounds(PointFootprint(LatLon(60.272222, 9.863889)), None),
    'BELGU': RegionBounds(PointFootprint(LatLon(61.2125, 10.597222)), None),
    'VANUK': RegionBounds(PointFootprint(LatLon(60.888889, 10.473611)), None),
    'RIPAM': RegionBounds(PointFootprint(LatLon(59.327778, 9.75)), None),
    'ROXIM': RegionBounds(PointFootprint(LatLon(59.617417, 10.214361)), None),
    'VALPU': RegionBounds(PointFootprint(LatLon(59.973611, 10.802417)), None),
    'GM400': RegionBounds(PointFootprint(LatLon(60.281025, 10.278583)), None),
    'GM402': RegionBounds(PointFootprint(LatLon(60.130306, 10.252833)), None),
    'GM404': RegionBounds(PointFootprint(LatLon(59.814444, 10.256944)), None),
    'GM405': RegionBounds(PointFootprint(LatLon(60.027056, 10.1785)), None),
    'GM406': RegionBounds(PointFootprint(LatLon(59.91725, 10.180472)), None),
    'GM407': RegionBounds(PointFootprint(LatLon(60.024306, 10.211389)), None),
    'GM408': RegionBounds(PointFootprint(LatLon(59.920528, 10.213278)), None),
    'GM409': RegionBounds(PointFootprint(LatLon(59.822778, 10.285444)), None),
    'GM410': RegionBounds(PointFootprint(LatLon(59.743583, 10.41975)), None),
    'GM455': RegionBounds(PointFootprint(LatLon(59.748833, 10.358917)), None),
    'GM456': RegionBounds(PointFootprint(LatLon(60.435667, 10.30525)), None),
    'OGRAS': RegionBounds(PointFootprint(LatLon(59.993703, 10.964169)), None),
    'XIKLA': RegionBounds(PointFootprint(LatLon(60.044731, 10.993269)), None),
    'RW01L': RegionBounds(PointFootprint(LatLon(60.185, 11.073744)), None),
    'ADOPI_ENTRY': RegionBounds(AnnularSectorFootprint(LatLon(60.323611, 9.383333), inner_radius_nm=2.5, outer_radius_nm=3.5, bearing_deg=271, half_angle_deg=30), None),
    'BELGU_ENTRY': RegionBounds(AnnularSectorFootprint(LatLon(61.2125, 10.597222), inner_radius_nm=2.5, outer_radius_nm=3.5, bearing_deg=346, half_angle_deg=30), None),
    'RIPAM_ENTRY': RegionBounds(AnnularSectorFootprint(LatLon(59.327778, 9.75), inner_radius_nm=2.5, outer_radius_nm=3.5, bearing_deg=231, half_angle_deg=30), None)
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
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[
    SpawnRegion(
        shape=REGIONS['ADOPI_ENTRY'],
        n_aircraft=betabinom(n=4, a=2, b=1),
        params={'alt_ft': (12000, 16000), 'spd_kts': 250, 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'A321': 1, 'B738': 1, 'B739': 1, 'E190': 1}),
        spawn_time=(0, 1800),
        route='ADOPI3L',
        name='ADOPI',
        conflict_free_spawn=True,
        spawn_gap_s=120.0,
    ),
    SpawnRegion(
        shape=REGIONS['BELGU_ENTRY'],
        n_aircraft=betabinom(n=4, a=2, b=1),
        params={'alt_ft': (14000, 18000), 'spd_kts': 250, 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'A321': 1, 'B738': 1, 'B739': 1, 'E190': 1}),
        spawn_time=(0, 1800),
        route='BELGU3L',
        name='BELGU',
        conflict_free_spawn=True,
        spawn_gap_s=120.0,
    ),
    SpawnRegion(
        shape=REGIONS['RIPAM_ENTRY'],
        n_aircraft=betabinom(n=4, a=2, b=1),
        params={'alt_ft': (12000, 16000), 'spd_kts': 250, 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'A321': 1, 'B738': 1, 'B739': 1, 'E190': 1}),
        spawn_time=(0, 1800),
        route='RIPAM3L',
        name='RIPAM',
        conflict_free_spawn=True,
        spawn_gap_s=120.0,
    ),
], routes={'ADOPI3L': [{'waypoint': 'ADOPI', 'speed_kts': 250, 'speed_above_kts': 0}, {'waypoint': 'NIVDU', 'speed_kts': 250, 'speed_above_kts': 0}, {'waypoint': 'GM402', 'speed_kts': 220, 'speed_above_kts': 0, 'alt_ft': 10000}, {'waypoint': 'GM405', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM406', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM404', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM455', 'speed_kts': 220, 'speed_above_kts': 0, 'alt_ft': 10000}, {'waypoint': 'VALPU', 'alt_ft': 5000, 'speed_kts': 200, 'alt_below_ft': 0, 'speed_above_kts': 0}, {'route': 'APP01L'}], 'BELGU3L': [{'waypoint': 'BELGU', 'speed_kts': 250, 'speed_above_kts': 0}, {'waypoint': 'VANUK', 'speed_kts': 250, 'speed_above_kts': 0}, {'waypoint': 'GM456', 'alt_ft': 12000, 'speed_kts': 250, 'alt_above_ft': 0, 'speed_above_kts': 0}, {'waypoint': 'GM400', 'speed_kts': 250, 'speed_above_kts': 0, 'alt_ft': 9000}, {'waypoint': 'GM402', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM405', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM406', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM404', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM455', 'speed_kts': 220, 'speed_above_kts': 0, 'alt_ft': 9000}, {'waypoint': 'VALPU', 'alt_ft': 5000, 'speed_kts': 200, 'alt_below_ft': 0, 'speed_above_kts': 0}, {'route': 'APP01L'}], 'RIPAM3L': [{'waypoint': 'RIPAM', 'speed_kts': 250, 'speed_above_kts': 0}, {'waypoint': 'ROXIM', 'alt_ft': 12000, 'speed_kts': 250, 'alt_below_ft': 0, 'speed_above_kts': 0}, {'waypoint': 'GM410', 'speed_kts': 220, 'speed_above_kts': 0, 'alt_ft': 11000}, {'waypoint': 'GM409', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM408', 'speed_kts': 220, 'speed_above_kts': 0}, {'waypoint': 'GM407', 'speed_kts': 220, 'speed_above_kts': 0, 'alt_ft': 11000}, {'waypoint': 'VALPU', 'alt_ft': 5000, 'speed_kts': 200, 'alt_below_ft': 0, 'speed_above_kts': 0}, {'route': 'APP01L'}], 'APP01L': [{'waypoint': 'OGRAS', 'alt_ft': 4000, 'alt_below_ft': 0, 'speed_kts': 180, 'speed_above_kts': 0}, {'waypoint': 'XIKLA', 'alt_ft': 3500, 'alt_below_ft': 0, 'speed_kts': 160, 'speed_above_kts': 0}, {'waypoint': 'RW01L', 'alt_ft': 707, 'speed_kts': 140, 'speed_above_kts': 0, 'flyover': True}]}),
            queryables={
            'ADOPI': Waypoint(at=REGIONS['ADOPI'], color='cyan', track_temporal_state=False),
    'NIVDU': Waypoint(at=REGIONS['NIVDU'], color='cyan', track_temporal_state=False),
    'BELGU': Waypoint(at=REGIONS['BELGU'], color='cyan', track_temporal_state=False),
    'VANUK': Waypoint(at=REGIONS['VANUK'], color='cyan', track_temporal_state=False),
    'RIPAM': Waypoint(at=REGIONS['RIPAM'], color='cyan', track_temporal_state=False),
    'ROXIM': Waypoint(at=REGIONS['ROXIM'], color='cyan', track_temporal_state=False),
    'VALPU': Waypoint(at=REGIONS['VALPU'], color='white', track_temporal_state=True),
    'GM400': Waypoint(at=REGIONS['GM400'], color='cyan', track_temporal_state=False),
    'GM402': Waypoint(at=REGIONS['GM402'], color='cyan', track_temporal_state=False),
    'GM404': Waypoint(at=REGIONS['GM404'], color='cyan', track_temporal_state=False),
    'GM405': Waypoint(at=REGIONS['GM405'], color='cyan', track_temporal_state=False),
    'GM406': Waypoint(at=REGIONS['GM406'], color='cyan', track_temporal_state=False),
    'GM407': Waypoint(at=REGIONS['GM407'], color='cyan', track_temporal_state=False),
    'GM408': Waypoint(at=REGIONS['GM408'], color='cyan', track_temporal_state=False),
    'GM409': Waypoint(at=REGIONS['GM409'], color='cyan', track_temporal_state=False),
    'GM410': Waypoint(at=REGIONS['GM410'], color='cyan', track_temporal_state=False),
    'GM455': Waypoint(at=REGIONS['GM455'], color='cyan', track_temporal_state=False),
    'GM456': Waypoint(at=REGIONS['GM456'], color='cyan', track_temporal_state=False),
    'OGRAS': Waypoint(at=REGIONS['OGRAS'], color='cyan', track_temporal_state=False),
    'XIKLA': Waypoint(at=REGIONS['XIKLA'], color='cyan', track_temporal_state=False),
    'RW01L': Waypoint(at=REGIONS['RW01L'], color='white', track_temporal_state=False)
        },
            rotation=rotation,
            groups=groups,
            sampled_waypoints={
            
        },
            waypoint_fields={
            
        },
            shapes=REGIONS,
        )

