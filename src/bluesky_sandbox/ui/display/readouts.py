"""Helpers for building GUI readout annotations."""

from __future__ import annotations

from collections.abc import Mapping

from bluesky_sandbox.interface.task import (
    WaypointReadoutItem,
    WaypointReadoutKey,
    WaypointReadoutNamespace,
    WaypointReadoutTarget,
)
from bluesky_sandbox.sim.fixes import RouteFix
from bluesky_sandbox.sim.queryables import Waypoint


def waypoint_readouts(
    waypoint: Waypoint,
    target: WaypointReadoutTarget,
    *,
    fix: RouteFix | None = None,
    label: str | None = None,
    target_alt_ft: float | None = None,
    radius_nm: float | None = None,
    alt_below_ft: float | None = None,
    alt_above_ft: float | None = None,
    speed_kts: float | None = None,
    speed_min_kts: float | None = None,
    speed_max_kts: float | None = None,
    speed_below_kts: float | None = None,
    speed_above_kts: float | None = None,
    include_altitude: bool = True,
    include_radius: bool = True,
    include_speed: bool = False,
    metadata: Mapping[str | WaypointReadoutKey, object] | None = None,
    constraints: Mapping[str | WaypointReadoutKey, object] | None = None,
    custom: Mapping[str | WaypointReadoutKey, object] | None = None,
) -> tuple[WaypointReadoutItem, ...]:
    """Build standard GUI readout items for a waypoint: its name, reach radius,
    and - from ``fix``, the aircraft's fix there (``context.query(name).target``)
    - the altitude, speed and windows its route crosses it with; a value passed
    in place of one wins."""
    items: list[WaypointReadoutItem] = []
    crossing = None if fix is None else fix.crossing

    def from_fix(value, attr):
        return value if value is not None or crossing is None else getattr(crossing, attr)

    target_alt_ft = from_fix(target_alt_ft, "alt_ft")
    alt_below_ft = from_fix(alt_below_ft, "alt_below_ft")
    alt_above_ft = from_fix(alt_above_ft, "alt_above_ft")
    speed_below_kts = from_fix(speed_below_kts, "speed_below_kts")
    speed_above_kts = from_fix(speed_above_kts, "speed_above_kts")
    if include_speed:
        speed_kts = from_fix(speed_kts, "speed_kts")

    def add(
        namespace: WaypointReadoutNamespace,
        key: str | WaypointReadoutKey,
        value: object,
    ) -> None:
        if value is not None:
            items.append(WaypointReadoutItem(target, namespace, key, value))

    add(WaypointReadoutNamespace.METADATA, WaypointReadoutKey.NAME, label)
    if include_altitude:
        add(
            WaypointReadoutNamespace.METADATA,
            WaypointReadoutKey.TARGET_ALT_FT,
            target_alt_ft,
        )
    if include_radius:
        add(
            WaypointReadoutNamespace.CONSTRAINTS,
            WaypointReadoutKey.RADIUS_NM,
            waypoint.reach_radius_nm if radius_nm is None else radius_nm,
        )
    add(WaypointReadoutNamespace.CONSTRAINTS, WaypointReadoutKey.ALT_BELOW_FT, alt_below_ft)
    add(WaypointReadoutNamespace.CONSTRAINTS, WaypointReadoutKey.ALT_ABOVE_FT, alt_above_ft)
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_KTS,
        speed_kts,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_MIN_KTS,
        speed_min_kts,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_MAX_KTS,
        speed_max_kts,
    )
    add(WaypointReadoutNamespace.CONSTRAINTS, WaypointReadoutKey.SPEED_BELOW_KTS, speed_below_kts)
    add(WaypointReadoutNamespace.CONSTRAINTS, WaypointReadoutKey.SPEED_ABOVE_KTS, speed_above_kts)
    for key, value in (metadata or {}).items():
        add(WaypointReadoutNamespace.METADATA, key, value)
    for key, value in (constraints or {}).items():
        add(WaypointReadoutNamespace.CONSTRAINTS, key, value)
    for key, value in (custom or {}).items():
        add(WaypointReadoutNamespace.CONSTRAINTS, key, value)
    return tuple(items)
