"""Module-level setup for this generated task."""

from __future__ import annotations

from .config import CONFIG

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from bluesky_sandbox import BaseAgentInfo, BaseObs
    from .task_types import TaskAgentStepContext

import bluesky as bs
import numpy as np
from bluesky.tools.aero import kts, vcas2tas

from bluesky_sandbox.sim.geometry.conflict import detection

# Spacing over VALPU, in time behind the previous aircraft: time is what the
# runway receives. Target 110 s; minimum 75 s, about 3 NM at 140 kt on final.
SPACING_S = 110.0
MIN_SPACING_S = 75.0
# Delay cost, per minute flown.
TIME_COST_PER_MIN = 0.1
# Workload cost, per speed reduction given.
CLEARANCE_COST = 0.02
# Potential-based shaping on predicted conflicts (Ng et al., 1999): it signals
# a conflict minutes before the loss without changing the optimal policy,
# provided SHAPING_GAMMA equals PPO's discount (SB3's default, 0.99).
SHAPING_WEIGHT = 0.2
SHAPING_GAMMA = 0.99
# Distance from the threshold at which an aircraft has landed, NM.
LANDED_NM = 0.3


def slow_from_m(idx, cas_kts):
    """Distance before a fix at which the aircraft at ``idx`` starts slowing to
    cross it at ``cas_kts``, m: twice its deceleration distance, since it slows
    less while descending, plus its fly-by turn distance. BlueSky's VNAV uses
    the deceleration distance alone and crosses fixes too fast."""
    target = vcas2tas(cas_kts * kts, bs.traf.alt[idx])
    slowing = 0.5 * abs(bs.traf.tas[idx] ** 2 - target**2) / max(float(bs.traf.perf.axmax[idx]), 0.001)
    return 2.0 * slowing + float(bs.traf.actwp.turndist[idx])


def spacing_score(gap_s):
    """Score of a gap of ``gap_s`` behind the previous aircraft over VALPU: 0 at
    ``SPACING_S`` or more, falling linearly to -1 at ``MIN_SPACING_S``. The
    first aircraft, with no gap, scores 0."""
    if gap_s is None:
        return 0.0
    short = (SPACING_S - gap_s) / (SPACING_S - MIN_SPACING_S)
    return -min(max(short, 0.0), 1.0)


def potential(idx, horizon_s):
    """Shaping potential of the aircraft at ``idx``: ``-SHAPING_WEIGHT`` per
    predicted loss of separation, weighted from 0 at ``horizon_s`` ahead to 1
    when in loss now. The prediction is BlueSky's own conflict detection."""
    soon = np.clip(1.0 - detection().tlos_s[idx] / horizon_s, 0.0, 1.0)
    soon[idx] = 0.0
    return -SHAPING_WEIGHT * float(soon.sum())
