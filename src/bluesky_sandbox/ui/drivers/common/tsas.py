"""Shared TSAS table data model for pygame and Panda3D views."""

from __future__ import annotations

import math
from dataclasses import dataclass

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint


@dataclass(frozen=True)
class TsasRow:
    idx: int
    acid: str
    eta_s: float
    dist_nm: float
    alt_diff_ft: float
    state: str


@dataclass(frozen=True)
class TsasTable:
    name: str
    title: str
    color: str
    waypoint: Waypoint
    rows: list[TsasRow]


class TsasDataMixin:
    """Build toolkit-neutral TSAS tables from env queryables and bs.traf."""

    _MIN_CLOSING_KT = 1.0

    def tsas_tables(self, driver, max_rows: int | None = None) -> list[TsasTable]:
        env = driver._env
        if env is None:
            return []
        tables: list[TsasTable] = []
        for name, waypoint in self.tsas_waypoints(driver):
            rows = self.tsas_aircraft_rows(name, waypoint, driver)
            if max_rows is not None:
                rows = rows[:max_rows]
            tables.append(
                TsasTable(
                    name=name,
                    title=name,
                    color=waypoint.color,
                    waypoint=waypoint,
                    rows=rows,
                )
            )
        return tables

    def tsas_waypoints(self, driver) -> list[tuple[str, Waypoint]]:
        env = driver._env
        if env is None:
            return []
        return [
            (name, queryable)
            for name, queryable in env.episode_queryables.items()
            if isinstance(queryable, Waypoint) and self.tsas_show_waypoint(queryable)
        ]

    def tsas_aircraft_rows(self, name: str, waypoint: Waypoint, driver) -> list[TsasRow]:
        n = bs.traf.ntraf
        if n == 0:
            return []
        region = (
            self.tsas_resolve_region(driver, waypoint.tsas_region)
            if waypoint.tsas_region
            else None
        )
        lat = np.asarray(bs.traf.lat[:n], dtype=float)
        lon = np.asarray(bs.traf.lon[:n], dtype=float)
        alt_ft = np.asarray(bs.traf.alt[:n], dtype=float) / ft
        # A row needs the distance and bearing to each aircraft's fix there and
        # its height against that fix's crossing - the waypoint's batched query,
        # as the observations read it.
        batch = driver._env.query_batch(name, range(n))
        bearing_deg = batch.values("current.bearing_deg")
        dist_nm = np.asarray(batch.values("current.distance_nm"), dtype=float)
        gs_kts = np.asarray(bs.traf.gs[:n], dtype=float) / kts
        angle_deg = (bearing_deg - np.asarray(bs.traf.hdg[:n], dtype=float) + 540) % 360 - 180
        closing_kts = gs_kts * np.cos(np.radians(angle_deg))
        closing = closing_kts > self._MIN_CLOSING_KT
        eta_s = np.full(n, math.inf)
        eta_s[closing] = dist_nm[closing] / closing_kts[closing] * 3600.0
        alt_diff_ft = np.asarray(batch.values("current.alt_diff_ft"), dtype=float)

        rows: list[TsasRow] = []
        for idx, (eta, dist, alt_diff) in enumerate(
            zip(eta_s.tolist(), dist_nm.tolist(), alt_diff_ft.tolist(), strict=True)
        ):
            if region is not None and not region.shape.contains(
                lat[idx], lon[idx], alt_ft[idx]
            ):
                continue
            acid = bs.traf.id[idx]
            rows.append(
                TsasRow(
                    idx=idx,
                    acid=acid,
                    eta_s=eta,
                    dist_nm=dist,
                    alt_diff_ft=alt_diff,
                    state=driver._aircraft_state(acid),
                )
            )
        rows.sort(key=lambda row: (row.eta_s, row.dist_nm))
        return rows

    @staticmethod
    def tsas_show_waypoint(waypoint: Waypoint) -> bool:
        return (
            waypoint.render_shape
            if waypoint.render_tsas is None
            else waypoint.render_tsas
        )

    @staticmethod
    def tsas_eta_text(eta_s: float) -> str:
        if not math.isfinite(eta_s):
            return "--:--"
        secs = max(0, int(round(eta_s)))
        return f"{secs // 60:02d}:{secs % 60:02d}"

    @staticmethod
    def tsas_dtm_text(dist_nm: float) -> str:
        return f"{dist_nm:.0f}NM"

    @staticmethod
    def tsas_resolve_region(driver, name: str) -> QueryRegion:
        queryables = driver._env.episode_queryables
        if name not in queryables:
            raise AssertionError(
                f"Waypoint.tsas_region={name!r} not found in episode queryables "
                f"(known: {sorted(queryables)})"
            )
        region = queryables[name]
        if not isinstance(region, QueryRegion):
            raise AssertionError(
                f"Waypoint.tsas_region={name!r} must point to a QueryRegion, "
                f"got {type(region).__name__}"
            )
        return region
