"""Geometry shared across the step: pairs of aircraft and BlueSky's conflict
detection over them.

Airspace *volumes* are :mod:`bluesky_sandbox.sim.bounds`. :mod:`.conflict`
covers the live aircraft; :mod:`.clearance` asks the same of a *candidate*
spawn state that is not in ``bs.traf`` yet.
"""

from __future__ import annotations

from .clearance import inside_separation_zone, predicted_conflict
from .conflict import (
    ConflictView,
    conflict_geometry,
    detection,
    windowed_min_hsep_nm,
    windowed_min_vsep_ft,
    windowed_signed_vsep_at_entry_ft,
)

__all__ = [
    "ConflictView",
    "conflict_geometry",
    "detection",
    "inside_separation_zone",
    "predicted_conflict",
    "windowed_min_hsep_nm",
    "windowed_min_vsep_ft",
    "windowed_signed_vsep_at_entry_ft",
]
