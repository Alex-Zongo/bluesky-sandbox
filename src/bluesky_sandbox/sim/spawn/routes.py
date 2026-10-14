"""Route specs: the step grammar, validation, resolution and sampling.

A route is a list of steps - a waypoint name, a junction, or a weighted
choice between branches. This module turns that declarative spec into the
concrete waypoint sequence one aircraft flies. Independent of where the
aircraft comes from, which is :mod:`~bluesky_sandbox.sim.spawn.regions`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from bluesky_sandbox.sim.fixes import CROSSING_KEYS
from bluesky_sandbox.sim.sampling.distributions import sample_scalar

# A route is an ordered list of steps. A step is one of:
#   * a Waypoint queryable name (``str``) - fly to that place, no crossing;
#   * a waypoint with its crossing ``{"waypoint": "<name>", ...}`` - how this
#     route crosses it (:class:`~bluesky_sandbox.sim.fixes.Crossing`):
#     ``alt_ft`` / ``speed_kts`` and their windows (``alt_below_ft`` /
#     ``alt_above_ft``, ``speed_below_kts`` / ``speed_above_kts``), ``flyover``,
#     ``rta``. Each may be used alone: ``speed_kts`` with no ``alt_ft`` is
#     "250 KT or less" at a fix with no level, flown as a speed alone;
#   * a sampled waypoint target ``{"waypoint": "<name>", "sample": Bounds, ...}``
#     - resolved per spawned aircraft into a concrete BlueSky route target;
#   * any waypoint step may carry ``arrival_slack_s`` - seconds, or a
#     distribution of them - giving its fix an arrival time
#     (:mod:`bluesky_sandbox.sim.arrival`), which the aircraft's FMS flies as
#     an RTA where the step says ``rta``;
#   * a subroute reference ``{"route": "<name>"}`` expanded inline;
#   * a branch ``{"choice": [step, ...], "weights": [w, ...]}`` - one branch is
#     taken (weighted; uniform if ``weights`` omitted). Branches let a procedure
#     diverge (a SID core that splits into transitions) or merge (entry
#     transitions that share an arrival trunk): a branch whose endpoint is a
#     waypoint also used by the trunk joins it there, and consecutive duplicate
#     waypoints across that junction are collapsed.
# Resolution comes in two flavours: :func:`sample_route_path` picks one concrete
# path (sampling each choice), preserving per-step constraints for the aircraft's
# ADDWPT commands, and :func:`expand_route_paths` enumerates every distinct path
# as plain waypoint names for validation and network visualization.
RouteStep = str | dict
RouteSpec = list[RouteStep] | tuple[RouteStep, ...]


@dataclass(frozen=True)
class Route:
    """A named route with what it is drawn in: its ``steps`` (the grammar
    above) and ``color`` - a palette name or ``#rrggbb``; ``None`` takes the
    next of the default route colors (see :func:`route_colors`). Anywhere a
    route's steps are taken, a :class:`Route` stands for them."""

    steps: RouteSpec
    color: str | None = None


#: The colors routes without their own are drawn in, in turn - none a driver
#: already gives a meaning: yellow (the selected aircraft), magenta (its
#: route), green (spawn regions), nor an alert's hue.
DEFAULT_ROUTE_COLORS = ("cyan", "blue", "white", "#84cc16", "#94a3b8")


def route_steps(route):
    """A route's steps: a :class:`Route`'s, else the route itself."""
    return route.steps if isinstance(route, Route) else route


def map_route_steps(route, fn: Callable):
    """``route`` with ``fn`` applied to its steps - a :class:`Route` kept one,
    its color kept."""
    if isinstance(route, Route):
        return replace(route, steps=fn(route.steps))
    return fn(route)


def route_colors(routes: dict, extra: int = 0) -> list[str]:
    """Each named route's color, in order - its own, or the next default -
    then the colors of ``extra`` more routes with none (inline ones)."""
    out, i = [], 0
    for route in routes.values():
        own = route.color if isinstance(route, Route) else None
        if own is None:
            own, i = DEFAULT_ROUTE_COLORS[i % len(DEFAULT_ROUTE_COLORS)], i + 1
        out.append(own)
    for _ in range(extra):
        out.append(DEFAULT_ROUTE_COLORS[i % len(DEFAULT_ROUTE_COLORS)])
        i += 1
    return out
_WAYPOINT_STEP_KEYS = (
    "waypoint",
    *CROSSING_KEYS,
    "sample",
    "sample_alt_from_envelope",
    "sample_speed_from_envelope",
    "envelope_alt_floor_ft",
    "alt_step_ft",
    "alt_from_start",
    "reachable_from_spawn",
    "reachable_vs_fraction",
    "arrival_slack_s",
)


def route_step_name(step) -> str:
    """The waypoint queryable name a resolved route step refers to.

    Accepts a bare name (``str``) or a constrained ``{"waypoint": name, ...}``
    step (the form :func:`sample_route_path` produces).
    """
    if isinstance(step, str):
        return step
    if isinstance(step, dict) and isinstance(step.get("waypoint"), str):
        return step["waypoint"]
    raise ValueError(f"cannot derive a waypoint name from route step {step!r}.")


def route_step_names(steps) -> list[str]:
    """Names-only view of a resolved route (drops per-step constraints)."""
    return [route_step_name(step) for step in steps]


#: A crossing's numbers - a step may give each as a finite number, a
#: ``(low, high)`` range or a distribution, drawn once per episode by the
#: scenario (:func:`draw_crossings`) - or, where a spawn config is used without
#: one, per aircraft at spawn.
_CROSSING_NUMBERS = tuple(key for key in CROSSING_KEYS if key not in ("flyover", "rta"))


def _crossing_value(value) -> bool:
    """A finite number, a ``(low, high)`` range of them, or a distribution."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return len(value) == 2 and all(_crossing_value(v) for v in value)
    return hasattr(value, "rvs")


def draw_crossings(steps, rng: np.random.Generator):
    """``steps`` with each crossing number given as a range or distribution
    drawn - choice branches included."""

    def step_drawn(step):
        if isinstance(step, dict) and "choice" in step:
            branches = [
                draw_crossings(branch, rng) if isinstance(branch, list) else step_drawn(branch)
                for branch in step["choice"]
            ]
            return {**step, "choice": branches}
        if not (isinstance(step, dict) and "waypoint" in step):
            return step
        out = dict(step)
        for key in _CROSSING_NUMBERS:
            value = out.get(key)
            if value is not None and not isinstance(value, (int, float)):
                out[key] = float(sample_scalar(value, rng))
        return out

    return map_route_steps(steps, lambda flat: [step_drawn(step) for step in flat])


def _validate_waypoint_step(step: dict) -> str:
    """Validate a ``{"waypoint": name, ...}`` step; return its waypoint name."""
    name = step.get("waypoint")
    if not isinstance(name, str) or not name:
        raise ValueError(f"route waypoint step needs a 'waypoint' name, got {step!r}.")
    extra = set(step) - set(_WAYPOINT_STEP_KEYS)
    if extra:
        raise ValueError(
            f"route waypoint step {step!r} has unknown keys {sorted(extra)}; "
            f"allowed: {list(_WAYPOINT_STEP_KEYS)}."
        )
    for key in _CROSSING_NUMBERS:
        value = step.get(key)
        if value is not None and not _crossing_value(value):
            raise ValueError(
                f"route waypoint step {key!r} must be a finite number, a (low, high) "
                f"range or a distribution, got {value!r}."
            )
    for key in (
        "envelope_alt_floor_ft",
        "alt_step_ft",
        "reachable_vs_fraction",
    ):
        value = step.get(key)
        if value is not None and not (
            isinstance(value, (int, float)) and math.isfinite(value)
        ):
            raise ValueError(
                f"route waypoint step {key!r} must be a finite number, got {value!r}."
            )
    for key in ("alt_below_ft", "alt_above_ft", "speed_below_kts", "speed_above_kts"):
        if isinstance(step.get(key), (int, float)) and step[key] < 0:
            raise ValueError(
                f"route waypoint step {key!r} - how far its window reaches - must be "
                f">= 0, got {step[key]!r}."
            )
    for key in ("flyover", "rta"):
        if key in step and not isinstance(step[key], bool):
            raise ValueError(f"route waypoint step {key!r} must be True or False, got {step[key]!r}.")
    if bool(step.get("alt_from_start", False)) and (
        step.get("alt_ft") is not None or bool(step.get("sample_alt_from_envelope", False))
    ):
        raise ValueError(
            "route waypoint step alt_from_start takes the altitude the leg starts "
            f"at; it cannot also set alt_ft or sample it, got {step!r}."
        )
    if bool(step.get("reachable_from_spawn", False)) and not bool(
        step.get("sample_alt_from_envelope", False)
    ):
        raise ValueError(
            "route waypoint step reachable_from_spawn=True requires "
            f"sample_alt_from_envelope=True, got {step!r}."
        )
    frac = step.get("reachable_vs_fraction")
    if frac is not None:
        if not bool(step.get("reachable_from_spawn", False)):
            raise ValueError(
                "route waypoint step reachable_vs_fraction requires "
                f"reachable_from_spawn=True, got {step!r}."
            )
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError(
                "route waypoint step reachable_vs_fraction must be in (0, 1], "
                f"got {frac!r}."
            )
    return name


def _collapse_junctions(names: list[str]) -> list[str]:
    """Drop consecutive duplicate waypoints so shared junctions join cleanly."""
    out: list[str] = []
    for name in names:
        if not out or out[-1] != name:
            out.append(name)
    return out


def _collapse_steps(steps: list) -> list:
    """Collapse consecutive steps for the same waypoint (a shared junction).

    Keeps the more specific step at the junction - a constrained
    ``{"waypoint": ...}`` step wins over a bare name - so a crossing restriction
    survives a junction merge. A step that ``sample``s a position is a *distinct*
    target, not a shared junction, so two such steps are never collapsed even
    when they reference the same waypoint queryable - this is what lets a route
    chain several sampled fixes off one waypoint template (e.g. merge fix then
    exit).
    """
    out: list = []
    for step in steps:
        prev = out[-1] if out else None
        prev_sampled = isinstance(prev, dict) and prev.get("sample") is not None
        step_sampled = isinstance(step, dict) and step.get("sample") is not None
        if (
            prev is not None
            and route_step_name(prev) == route_step_name(step)
            and not prev_sampled
            and not step_sampled
        ):
            if isinstance(step, dict) and not isinstance(prev, dict):
                out[-1] = step
            continue
        out.append(step)
    return out


def _branch_steps(branch) -> list:
    """A choice branch may be a single step or an inline list of steps."""
    return list(branch) if isinstance(branch, (list, tuple)) else [branch]


def _validate_choice(step: dict) -> tuple[list, list[float] | None]:
    """Return ``(branches, weights)`` for a validated ``choice`` step."""
    branches = step["choice"]
    if not isinstance(branches, (list, tuple)) or len(branches) == 0:
        raise ValueError(f"route 'choice' must list >= 1 branch, got {branches!r}.")
    weights = step.get("weights")
    if weights is None:
        return list(branches), None
    if len(weights) != len(branches):
        raise ValueError(
            f"route 'choice' has {len(weights)} weights for {len(branches)} "
            f"branches; lengths must match."
        )
    w = [float(x) for x in weights]
    if any(x < 0 for x in w) or sum(w) <= 0.0:
        raise ValueError(
            f"route 'choice' weights must be non-negative with a positive sum, "
            f"got {weights!r}."
        )
    return list(branches), w


def _resolve_pick(steps, routes, seen, pick) -> list[str]:
    """Resolve ``steps`` to one flat path, using ``pick`` to choose branches."""
    out: list[str] = []
    for step in route_steps(steps):
        if isinstance(step, dict):
            if "choice" in step:
                branches, weights = _validate_choice(step)
                out.extend(_resolve_pick(_branch_steps(pick(branches, weights)), routes, seen, pick))
            elif "route" in step:
                name = step["route"]
                if name in seen:
                    raise ValueError(f"route composition cycle through {name!r}.")
                if name not in routes:
                    raise ValueError(
                        f"subroute {name!r} not found; available routes: {list(routes)}."
                    )
                out.extend(_resolve_pick(routes[name], routes, seen + (name,), pick))
            elif "waypoint" in step:
                _validate_waypoint_step(step)
                out.append(step)  # constrained leaf, preserved for ADDWPT
            else:
                raise ValueError(
                    f"route step dict must have a 'waypoint', 'route', or 'choice' "
                    f"key, got {step!r}."
                )
        else:
            out.append(step)
    return out


def _resolve_all(steps, routes, seen, keep_steps: bool = False) -> list[list]:
    """Enumerate every concrete path through ``steps`` (cartesian over choices):
    waypoint names, or - ``keep_steps`` - the steps themselves."""
    paths: list[list[str]] = [[]]
    for step in route_steps(steps):
        if isinstance(step, dict):
            if "choice" in step:
                branches, _ = _validate_choice(step)
                branch_paths: list[list[str]] = []
                for branch in branches:
                    branch_paths.extend(_resolve_all(_branch_steps(branch), routes, seen, keep_steps))
                paths = [p + bp for p in paths for bp in branch_paths]
            elif "route" in step:
                name = step["route"]
                if name in seen:
                    raise ValueError(f"route composition cycle through {name!r}.")
                if name not in routes:
                    raise ValueError(
                        f"subroute {name!r} not found; available routes: {list(routes)}."
                    )
                sub = _resolve_all(routes[name], routes, seen + (name,), keep_steps)
                paths = [p + sp for p in paths for sp in sub]
            elif "waypoint" in step:
                name = _validate_waypoint_step(step)
                leaf = step if keep_steps else name
                paths = [p + [leaf] for p in paths]
            else:
                raise ValueError(
                    f"route step dict must have a 'waypoint', 'route', or 'choice' "
                    f"key, got {step!r}."
                )
        else:
            paths = [p + [step] for p in paths]
    return paths


def resolve_route(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
) -> list[str]:
    """Flatten a route deterministically, taking the first branch of any choice.

    Expands ``{"route": name}`` subroutes and collapses junction duplicates;
    raises on a missing subroute or a reference cycle. Returns plain waypoint
    names (per-step constraints dropped). For sampling a random branch with
    constraints use :func:`sample_route_path`; to enumerate every branch use
    :func:`expand_route_paths`.
    """
    steps_out = _resolve_pick(steps, routes, (), lambda b, _w: b[0])
    return _collapse_junctions(route_step_names(steps_out))


def sample_route_path(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
    rng: np.random.Generator,
) -> list[RouteStep]:
    """Resolve ``steps`` to one concrete path, sampling each ``choice`` branch.

    Constrained ``{"waypoint": ...}`` steps are preserved (so the per-step
    crossing restriction reaches ADDWPT); bare names stay strings. Use
    :func:`route_step_names` for a names-only view.
    """

    def pick(branches, weights):
        if weights is None:
            return branches[int(rng.integers(len(branches)))]
        w = np.asarray(weights, dtype=float)
        return branches[int(rng.choice(len(branches), p=w / w.sum()))]

    return _collapse_steps(_resolve_pick(steps, routes, (), pick))


def waypoint_steps(spawn) -> dict[str, list[dict]]:
    """Every route step of ``spawn`` (a :class:`SpawnConfig`) that states how a
    waypoint is crossed, by waypoint name - its own route, its named routes and
    each region's, choice branches included."""
    out: dict[str, list[dict]] = {}

    def walk(steps):
        for step in route_steps(steps):
            if isinstance(step, dict) and "choice" in step:
                for branch in step["choice"]:
                    walk(branch if isinstance(branch, list) else [branch])
            elif isinstance(step, dict) and isinstance(step.get("waypoint"), str):
                out.setdefault(step["waypoint"], []).append(step)

    routes = [*(spawn.routes or {}).values(), spawn.route, *(r.route for r in spawn.regions)]
    for route in routes:
        if isinstance(route, (list, tuple, Route)):
            walk(route)
    return out


def expand_route_step_paths(steps: RouteSpec, routes: dict[str, RouteSpec]) -> list[list[RouteStep]]:
    """:func:`expand_route_paths` keeping each step - its crossing with it -
    rather than its waypoint's name."""
    seen: set[tuple[str, ...]] = set()
    out: list[list[RouteStep]] = []
    for path in _resolve_all(steps, routes, (), keep_steps=True):
        collapsed = _collapse_steps(path)
        key = tuple(route_step_name(step) for step in collapsed)
        if key not in seen:
            seen.add(key)
            out.append(collapsed)
    return out


def expand_route_paths(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
) -> list[list[str]]:
    """Enumerate every distinct concrete path through a (possibly branching) route.

    Used for validation (every branch's waypoints must exist) and network
    visualization (draw all of a procedure's transitions). Duplicate paths and
    junction duplicates are collapsed.
    """
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for path in _resolve_all(steps, routes, ()):
        collapsed = _collapse_junctions(path)
        key = tuple(collapsed)
        if key not in seen:
            seen.add(key)
            out.append(collapsed)
    return out
