"""Modified Voltage Potential (MVP) conflict resolution as a policy, flying the
Random Encounters environment (``examples/environments/random_encounters``).
From the repository root::

    python -m examples.geometric.mvp [episodes] [--render [MODE]]

BlueSky's own resolution - ``bluesky.traffic.asas.mvp.MVP`` with its default
``PastCPA`` resume - computed from what each aircraft observes and flown with
its actions, instead of from inside the simulator:

* every pair the observation flags in conflict (``in_conf``) adds its MVP
  velocity change to the ownship's; with any conflict anywhere, each aircraft
  steers to its new track and ground speed (``hdg_deg``, ``spd_kts``), capped to
  its speed envelope, exactly as BlueSky recomputes for all aircraft then;
* a pair stays resolving until it is past its closest approach, out of the
  zone and not bouncing; when none of an aircraft's pairs is, it resumes its
  route (``autopilot_lnav``) at the speed it had selected before.

Horizontal resolution only - BlueSky's default (``RMETHH BOTH``), cooperative
(each aircraft of a pair takes its share), no priority rules and no wind.
Every aircraft must be an agent, as every aircraft resolves in BlueSky's.

It acts on the state BlueSky's would when the environment decides as often as
BlueSky detects conflicts, on the same instants: ``dt == simdt == asas_dt``,
as Random Encounters does (1 s).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vtas2cas

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import observation_layout, unflatten_action
from bluesky_sandbox.interface.fields.actions import (
    ActionMask,
    AutopilotLnav,
    HdgDeg,
    SpdKts,
)
from bluesky_sandbox.ui.drivers import WINDOW_RENDER_MODES
from examples.environments.random_encounters import RandomEncountersEnv


_OWNSHIP = (
    "trk_deg",
    "gs_kts",
    "alt_ft",
    "ap_cas_kts",
    "perf_vmin_kts",
    "perf_vmax_kts",
)
_INTRUDERS = (
    "rel_pos_along_track_nm",
    "rel_pos_cross_track_nm",
    "rel_vel_along_track_kts",
    "rel_vel_cross_track_kts",
    "conflict_tcpa_s",
    "in_conf",
)
_ACTIONS = (HdgDeg, SpdKts, AutopilotLnav)


def mvp_dv(drel: np.ndarray, vrel: np.ndarray, tcpa: float, rpz_m: float) -> np.ndarray:
    """The MVP horizontal velocity change (m/s) for one conflict pair, as
    BlueSky's ``MVP.MVP``: the intruder at ``drel`` (m) moving at ``vrel`` (m/s)
    relative to the ownship, its closest approach ``tcpa`` (s) away, pushed out
    to ``rpz_m`` (the zone times the resolution factor). Any frame whose axes
    are a rotation of east/north; the ownship subtracts it from its velocity."""
    dist = math.hypot(drel[0], drel[1])
    dcpa = drel + vrel * tcpa
    dabs = math.hypot(dcpa[0], dcpa[1])
    intrusion = rpz_m - dabs
    # Head-on: no miss direction to push along, so push sideways.
    if dabs <= 10.0:
        dabs = 10.0
        dcpa = np.array([drel[1] / dist * dabs, -drel[0] / dist * dabs])
    # Outside the zone: aim at the zone's tangent, not through its edge.
    if rpz_m < dist and dabs < dist:
        erratum = math.cos(math.asin(rpz_m / dist) - math.asin(dabs / dist))
        return (rpz_m / erratum - dabs) * dcpa / (abs(tcpa) * dabs)
    return intrusion * dcpa / (abs(tcpa) * dabs)


class MVP:
    """MVP for every agent of an environment built with ``config``.

    ::

        env = RandomEncountersEnv()
        policy = MVP(env.config, resofach=bs.settings.asas_marh)
        env.reset(seed=0)
        policy.reset()
        while env.agents:
            env.step(policy.act({a: env.raw_observation(a) for a in env.agents}))

    ``resofach`` is BlueSky's horizontal resolution factor (``RFACH``): the
    zone resolved to is the protected zone times it.
    """

    def __init__(self, config: EnvConfig, *, resofach: float = 1.01) -> None:
        if config.pz_radius_nm is None:
            raise ValueError(
                "MVP resolves to the protected zone: set EnvConfig.pz_radius_nm."
            )
        layout = observation_layout(config)
        for part, names in (("ownship", _OWNSHIP), ("intruders", _INTRUDERS)):
            have = {slot.name for slot in layout.get(part, [])}
            if missing := [name for name in names if name not in have]:
                raise ValueError(
                    f"MVP reads {part} fields {missing}; the observation has none."
                )
        self._config = config
        self._fields = list(config.action_fields)
        for kind in _ACTIONS:
            field = next((f for f in self._fields if type(f) is kind), None)
            if field is None or getattr(field, "normalizer", None) is not None:
                raise ValueError(
                    f"MVP flies {kind.__name__} in raw values: give it, unnormalized, as a clearance."
                )
        self._rpz_m = float(config.pz_radius_nm) * nm
        self.resofach = float(resofach)
        self.reset()

    def reset(self) -> None:
        """Forget the episode: pairs still resolving, speeds to restore."""
        self._resolving: set[tuple[str, str]] = set()
        self._selected_cas_kts: dict[str, float] = {}

    def act(self, observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Each agent's action for ``observations`` - each agent's raw
        observation (``env.raw_observation``). An intruder that is no longer an
        agent has left: the environment removes it before the next step."""
        own = {acid: raw["ownship"] for acid, raw in observations.items()}
        pairs = {}
        for acid, raw in observations.items():
            rows = raw["intruders"]
            for k, other in enumerate(rows["acid"]):
                if other in own:
                    pairs[acid, other] = (
                        # Cross, along: a rotation of east, north.
                        np.array(
                            [
                                rows["rel_pos_cross_track_nm"][k],
                                rows["rel_pos_along_track_nm"][k],
                            ]
                        )
                        * nm,
                        np.array(
                            [
                                rows["rel_vel_cross_track_kts"][k],
                                rows["rel_vel_along_track_kts"][k],
                            ]
                        )
                        * kts,
                        float(rows["conflict_tcpa_s"][k]),
                        bool(rows["in_conf"][k] > 0.5),
                    )
        conflicts = [pair for pair, (*_, in_conf) in pairs.items() if in_conf]
        targets = self._resolve(own, pairs, conflicts) if conflicts else {}
        active = self._past_cpa(own, pairs, conflicts)

        actions = {}
        for acid, ownship in own.items():
            given: dict[str, float] = {}
            if active.get(acid) and acid in targets:
                self._selected_cas_kts.setdefault(acid, float(ownship["ap_cas_kts"]))
                given["hdg_deg"], given["spd_kts"] = targets[acid]
            elif active.get(acid) is False:
                given["autopilot_lnav"] = 1.0
                if acid in self._selected_cas_kts:
                    given["spd_kts"] = self._selected_cas_kts.pop(acid)
            actions[acid] = self._action(given)
        return actions

    def _resolve(self, own, pairs, conflicts) -> dict[str, tuple[float, float]]:
        """Each aircraft's new track (deg) and CAS (kts) - BlueSky's
        ``MVP.resolve``, in each ownship's track frame."""
        dv = {acid: np.zeros(2) for acid in own}
        for pair in conflicts:
            drel, vrel, tcpa, _in_conf = pairs[pair]
            dv[pair[0]] -= mvp_dv(drel, vrel, tcpa, self._rpz_m * self.resofach)
        targets = {}
        for acid, ownship in own.items():
            cross, along = dv[acid][0], float(ownship["gs_kts"]) * kts + dv[acid][1]
            track = (
                float(ownship["trk_deg"]) + math.degrees(math.atan2(cross, along))
            ) % 360.0
            # Capped as BlueSky does: the envelope's speeds against ground speed.
            gs = min(
                max(math.hypot(cross, along), float(ownship["perf_vmin_kts"]) * kts),
                float(ownship["perf_vmax_kts"]) * kts,
            )
            # As a CAS by the atmosphere the autopilot converts it back with.
            cas = float(
                vtas2cas(np.array([gs]), np.array([float(ownship["alt_ft"]) * ft]))[0]
            )
            targets[acid] = (track, cas / kts)
        return targets

    def _past_cpa(self, own, pairs, conflicts) -> dict[str, bool]:
        """Which aircraft resolve on (True) and which resume (False) - BlueSky's
        ``PastCPA.resumenav``; one not named does neither."""
        self._resolving |= set(conflicts)
        active: dict[str, bool] = {}
        for pair in list(self._resolving):
            acid = pair[0]
            if acid not in own:
                self._resolving.discard(pair)
                continue
            resolving = False
            if pair in pairs:
                drel, vrel, _tcpa, _in_conf = pairs[pair]
                hdist = math.hypot(drel[0], drel[1])
                past_cpa = float(drel @ vrel) > 0.0
                # The intruder's track against the own: (0, gs) in this frame.
                relative_trk = math.degrees(
                    math.atan2(vrel[0], float(own[acid]["gs_kts"]) * kts + vrel[1])
                )
                bouncing = (
                    abs(relative_trk) < 30.0 and hdist < self._rpz_m * self.resofach
                )
                resolving = not past_cpa or hdist < self._rpz_m or bouncing
            if resolving:
                active[acid] = True
            else:
                active[acid] = active.get(acid, False)
                self._resolving.discard(pair)
        return active

    def hold(self) -> Any:
        """The action that gives nothing: every aircraft flies on as it was."""
        return self._action({})

    def _action(self, given: Mapping[str, float]) -> Any:
        """The action clearing ``given``, by action name, and holding the rest."""
        flat = []
        for field in self._fields:
            if isinstance(field, ActionMask):
                flat.append(0.0 if field.target in given else 1.0)
            else:
                flat.append(given.get(field.meta.name, 0.0))
        return unflatten_action(self._config, flat)


def main() -> None:
    """Fly Random Encounters with MVP and count the pairs losing separation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", type=int, nargs="?", default=10)
    parser.add_argument(
        "--render",
        nargs="?",
        const="pygame",
        choices=WINDOW_RENDER_MODES,
        help="watch it in a window: pygame unless named",
    )
    args = parser.parse_args()

    # Watched, the episode runs in real time: one second a step.
    env = RandomEncountersEnv(render_mode=args.render, realtime=args.render is not None)
    try:
        for seed in range(args.episodes):
            env.reset(seed=seed)
            # BlueSky's resolution factor, as its own MVP would read it.
            policy = MVP(env.config, resofach=bs.settings.asas_marh)
            losses = set()
            while env.agents:
                if args.render:
                    env.render()  # the first call opens the window
                env.step(policy.act({agent: env.raw_observation(agent) for agent in env.agents}))
                losses |= {frozenset(pair) for pair in bs.traf.cd.lospairs}
            print(f"episode {seed}: {len(env.spawn_log)} aircraft, {len(losses)} pairs lost separation")
    finally:
        env.close()


if __name__ == "__main__":
    main()
