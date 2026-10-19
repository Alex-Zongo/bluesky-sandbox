"""Compare conflict-resolution methods on the example environments. From the
repository root::

    python -m examples.geometric.compare                          # every environment and method, 10 seeds
    python -m examples.geometric.compare --env merge head_on --seeds 20
    python -m examples.geometric.compare --env merge --method mvp --seeds 1 --render
    python -m examples.geometric.compare --csv results.csv        # each episode, too

Every method flies the same seeds - the same traffic - in each environment, and
every episode is scored from the simulator, whatever the method observed:

* the pairs that lost separation, and the seconds each spent in loss of
  separation, summed over pairs;
* the smallest horizontal separation between two aircraft within the
  protected zone's height of each other;
* each aircraft's flight time, from its spawn to its removal, and how much
  longer it is than with no resolution on the same seed: what the
  resolutions' detours cost;
* the aircraft whose flight ended outside the airspace. Where leaving it is
  how a flight ends (the encounter environments) that is every aircraft;
  where flights end inside it (``merge``, at the IAF) it is those a
  resolution pushed off their procedure.

A method is a policy built for the environment - one that acts on every
agent's raw observation, as :class:`~examples.geometric.mvp.MVP` does - or
BlueSky's own resolution, run inside the simulator. ``METHODS`` lists them.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft
from bluesky.tools.geo import kwikqdrdist_matrix

from bluesky_sandbox.ui.drivers import WINDOW_RENDER_MODES
from examples.geometric.mvp import MVP

ENVIRONMENTS = Path(__file__).parents[1] / "environments"


@dataclass(frozen=True)
class Method:
    """How a method flies: ``policy`` builds one for an environment, to act on
    its agents' raw observations; ``reso_method`` is BlueSky's own resolution,
    run inside the simulator (``None``: off)."""

    policy: Callable[[Any], Any] | None = None
    reso_method: str | None = None


METHODS = {
    "none": Method(),
    "mvp": Method(policy=lambda env: MVP(env.config, resofach=bs.settings.asas_marh)),
    "bluesky-mvp": Method(reso_method="MVP"),
}


def environment_names() -> list[str]:
    """Every environment package in ``examples/environments``."""
    return sorted(p.parent.name for p in ENVIRONMENTS.glob("*/__init__.py"))


def make_env(name: str, method: Method, render: str | None):
    """Environment ``name``, with BlueSky's own resolution as ``method`` has it:
    off for every method that is not BlueSky's, so no design's own setting
    resolves on top of a policy."""
    package = importlib.import_module(f"examples.environments.{name}")
    config = importlib.import_module(f"examples.environments.{name}.config").CONFIG
    config = dataclasses.replace(config, reso_method=method.reso_method)
    # Watched, the episode runs in real time: one second a step.
    return package.Env(config=config, render_mode=render, realtime=render is not None)


def fly(env, method: Method, seed: int, render: bool, max_steps: int) -> dict[str, float]:
    """One episode of ``env`` on ``seed`` under ``method``, and its scores."""
    env.reset(seed=seed)
    policy = method.policy(env) if method.policy else None
    hpz_ft = float(env.config.pz_height_ft)
    los_pairs: set[frozenset[str]] = set()
    los_s = 0.0
    min_sep_nm = float("inf")
    finished_at: dict[str, float] = {}  # callsign -> when its flight ended
    left = 0
    steps = 0
    while not env.episode_done and steps < max_steps:
        if render:
            env.render()  # the first call opens the window
        actions = policy.act({agent: env.raw_observation(agent) for agent in env.agents}) if policy else {}
        _, _, terminations, truncations, infos = env.step(actions)
        steps += 1

        # An agent's flight ends on the step it is done; the environment
        # removes it then. Done outside the airspace: it never finished in it.
        for agent, terminated in terminations.items():
            if terminated or truncations[agent]:
                finished_at[agent] = env.sim_time
                left += not infos[agent]["in_airspace"]

        now = {frozenset(pair) for pair in bs.traf.cd.lospairs}
        los_pairs |= now
        los_s += len(now) * env.config.dt

        if bs.traf.ntraf > 1:
            _, dist_nm = kwikqdrdist_matrix(bs.traf.lat, bs.traf.lon, bs.traf.lat, bs.traf.lon)
            dist_nm = np.asarray(dist_nm)
            alt_ft = np.asarray(bs.traf.alt) / ft
            close = np.abs(alt_ft[:, None] - alt_ft[None, :]) < hpz_ft
            np.fill_diagonal(close, False)
            if close.any():
                min_sep_nm = min(min_sep_nm, float(dist_nm[close].min()))

    spawned = {record.callsign: record.time_s for record in env.spawn_log}
    flight_s = [finished_at[c] - spawned[c] for c in finished_at if c in spawned]
    return {
        "aircraft": len(spawned),
        "los_pairs": len(los_pairs),
        "los_s": los_s,
        "min_sep_nm": min_sep_nm,
        "flight_s": statistics.mean(flight_s) if flight_s else float("nan"),
        "ended_outside": left,
        "unfinished": len(spawned) - len(finished_at),  # still flying when cut at max_steps
    }


def table(rows: list[dict[str, Any]]) -> str:
    """Each environment and method over its seeds, as a Markdown table: means,
    and the worst episode's smallest separation."""
    lines = [
        "| environment | method | episodes | aircraft | pairs in LoS | LoS (pair-s) "
        "| min sep, mean / worst (nm) | flight time (s) | extra vs none (s) | ended outside airspace |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["environment"], row["method"]), []).append(row)
    for (env_name, method), group in groups.items():
        def mean(key):
            values = [r[key] for r in group if r[key] == r[key] and r[key] != float("inf")]
            return statistics.mean(values) if values else float("nan")

        extra = mean("extra_s")
        lines.append(
            f"| {env_name} | {method} | {len(group)} | {mean('aircraft'):.1f} | {mean('los_pairs'):.2f} "
            f"| {mean('los_s'):.0f} | {mean('min_sep_nm'):.2f} / {min(r['min_sep_nm'] for r in group):.2f} "
            f"| {mean('flight_s'):.0f} | {'' if extra != extra else f'{extra:+.0f}'} | {mean('ended_outside'):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Fly every chosen environment and method on the same seeds, print each
    episode and then the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", nargs="+", metavar="NAME", help="environments in examples/environments (default: all)")
    parser.add_argument("--method", nargs="+", choices=list(METHODS), default=list(METHODS), help="methods (default: all)")
    parser.add_argument("--seeds", type=int, default=10, help="episodes per environment and method: seeds 0..N-1")
    parser.add_argument("--max-steps", type=int, default=7200, help="cut an episode after this many steps")
    parser.add_argument("--csv", type=Path, help="also write each episode's scores to this file")
    parser.add_argument(
        "--render",
        nargs="?",
        const="pygame",
        choices=WINDOW_RENDER_MODES,
        help="watch it in a window: pygame unless named",
    )
    args = parser.parse_args()
    known = environment_names()
    names = args.env or known
    if unknown := [n for n in names if n not in known]:
        parser.error(f"no environment {', '.join(map(repr, unknown))}; there are: {', '.join(known)}")

    rows = []
    for env_name in names:
        for method_name in args.method:
            method = METHODS[method_name]
            env = make_env(env_name, method, args.render)
            try:
                for seed in range(args.seeds):
                    try:
                        scores = fly(env, method, seed, args.render is not None, args.max_steps)
                    except ValueError as e:  # a policy refusing an environment it cannot read
                        parser.exit(1, f"{method_name} cannot fly {env_name}: {e}\n")
                    rows.append({"environment": env_name, "method": method_name, "seed": seed, **scores})
                    print(f"{env_name} / {method_name} / seed {seed}: {scores['aircraft']} aircraft, "
                          f"{scores['los_pairs']} pairs lost separation, min sep {scores['min_sep_nm']:.2f} nm")
            finally:
                env.close()

    # The detour: each episode's flight time over no resolution's, same environment and seed.
    baseline = {(r["environment"], r["seed"]): r["flight_s"] for r in rows if r["method"] == "none"}
    for row in rows:
        base = baseline.get((row["environment"], row["seed"]))
        row["extra_s"] = row["flight_s"] - base if base is not None else float("nan")

    print()
    print(table(rows))
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} episodes to {args.csv}")


if __name__ == "__main__":
    main()
