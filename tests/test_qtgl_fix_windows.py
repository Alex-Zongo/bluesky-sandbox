"""Each fix's crossing as BlueSky's QtGL radar labels it: chart style -
"5000+", "220-", a window's ends - for the fixes the aircraft's routes cross
with an altitude or speed, drawn and named; a fix two routes cross differently
shows both."""

from __future__ import annotations

from bluesky_sandbox.sim.fixes import Crossing, ResolvedRoute, RouteFix
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.ui.drivers.qtgl.fix_windows import LINE_CHARS, fix_window_labels, window_text


def test_a_window_reads_as_on_a_chart():
    assert window_text(5000, 0, None) == "5000+"  # at or above
    assert window_text(12000, None, 0) == "12000-"  # at or below
    assert window_text(220, None, 0) == "220-"  # N KT or less
    assert window_text(5000, 500, 500) == "4500-5500"  # both sides
    assert window_text(9000, 0, 0) == "9000"  # exactly at
    assert window_text(9000, None, None) == "9000"  # no window
    assert window_text(5000, 500, None) == "4500+"  # open above, 500 below
    assert window_text(None, 0, 0) == ""


def _route(*steps):
    """A resolved route of ``(name, crossing)`` steps."""
    return ResolvedRoute(
        names=tuple(name for name, _ in steps),
        fixes=tuple(RouteFix(lat=0.0, lon=0.0, crossing=crossing) for _, crossing in steps),
    )


def _lines(label):
    return [label[i:i + LINE_CHARS].strip() for i in range(0, len(label), LINE_CHARS)]


def test_only_fixes_crossed_with_a_constraint_and_named_are_labelled():
    fixes = {
        "VALPU": Waypoint(lat=59.97, lon=10.8),
        "PLAIN": Waypoint(lat=60.0, lon=10.0),  # crossed with nothing to show
        "QUIET": Waypoint(lat=60.0, lon=11.0, render_label=False),
    }
    route = _route(
        ("PLAIN", Crossing()),
        ("QUIET", Crossing(alt_ft=9000)),
        ("VALPU", Crossing(alt_ft=5000, alt_below_ft=0, speed_kts=220, speed_above_kts=0)),
    )
    labels = fix_window_labels(fixes, [route])
    assert labels["lat"] == [59.97] and labels["lon"] == [10.8]
    (label,) = labels["label"]
    assert _lines(label) == ["VALPU", "5000+", "220-"]


def test_a_fix_two_routes_cross_differently_shows_both():
    fixes = {"GM455": Waypoint(lat=59.75, lon=10.36)}
    adopi = _route(("GM455", Crossing(alt_ft=10000, speed_kts=220, speed_above_kts=0)))
    belgu = _route(("GM455", Crossing(alt_ft=9000, speed_kts=220, speed_above_kts=0)))
    (label,) = fix_window_labels(fixes, [adopi, belgu, adopi])["label"]
    assert _lines(label) == ["GM455", "10000/9000", "220-"]
