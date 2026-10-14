"""Each fix's constraints as BlueSky's QtGL radar shows them: a label of its
name, then its altitude and its speed in chart style - ``5000+`` at or above
5000 FT, ``220-`` 220 KT or less, ``4500-5500`` a window with both sides,
a bare value one with neither.

The sim side builds the labels from the routes the aircraft fly - a fix's
crossing is its route's, so a fix two routes cross differently shows both,
``10000/9000`` - and sends them (:data:`TOPIC`); the QtGL client draws them
(``gui_fix_windows``), its radar being top-down: a window is text at its fix,
not a drawn band. Nothing here imports BlueSky's GUI, so the sim process can
use it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["LINE_CHARS", "LINES", "TOPIC", "fix_window_labels", "window_text"]

#: The network topic the labels go by, sim to QtGL client.
TOPIC = "FIXWINDOWS"
#: A label's lines - name, altitude, speed - and their width, characters.
LINES = 3
LINE_CHARS = 10


def window_text(value: float | None, below: float | None, above: float | None) -> str:
    """A value and its window in chart style (see the module); ``""`` for no
    value."""
    if value is None:
        return ""

    def num(v: float) -> str:
        return str(int(round(v)))

    if below is None and above is None:
        return num(value)
    if below == 0 and above is None:
        return f"{num(value)}+"
    if above == 0 and below is None:
        return f"{num(value)}-"
    if below is not None and above is not None:
        low, high = value - below, value + above
        return num(value) if low == high else f"{num(low)}-{num(high)}"
    # One side open, the other some way off the value: where the window ends.
    return f"{num(value - below)}+" if below is not None else f"{num(value + above)}-"


def fix_window_labels(queryables: dict[str, Any], routes: Iterable[Any]) -> dict[str, list]:
    """The labels of the fixes ``routes`` (resolved routes:
    :class:`~bluesky_sandbox.sim.fixes.ResolvedRoute`) cross with an altitude
    or speed, at the episode waypoint each is - those drawn and named
    (``render_shape``, ``render_label``): their ``lat``, ``lon`` and ``label``
    - each :data:`LINES` lines of :data:`LINE_CHARS`, one string - as
    :data:`TOPIC` sends them. A fix crossed differently by different routes
    shows each, joined by ``/``."""
    texts: dict[str, tuple[list[str], list[str]]] = {}
    for route in routes:
        for name, fix in zip(route.names, route.fixes, strict=True):
            c = fix.crossing
            if c.alt_ft is None and c.speed_kts is None:
                continue
            alts, speeds = texts.setdefault(name, ([], []))
            for out, text in (
                (alts, window_text(c.alt_ft, c.alt_below_ft, c.alt_above_ft)),
                (speeds, window_text(c.speed_kts, c.speed_below_kts, c.speed_above_kts)),
            ):
                if text and text not in out:
                    out.append(text)
    lat: list[float] = []
    lon: list[float] = []
    label: list[str] = []
    for name, (alts, speeds) in texts.items():
        q = queryables.get(name)
        if q is None or getattr(q, "lat", None) is None:
            continue
        if not getattr(q, "render_shape", True) or not getattr(q, "render_label", True):
            continue
        lines = (name, "/".join(alts), "/".join(speeds))
        lat.append(float(q.lat))
        lon.append(float(q.lon))
        label.append("".join(line[:LINE_CHARS].ljust(LINE_CHARS) for line in lines))
    return {"lat": lat, "lon": lon, "label": label}
