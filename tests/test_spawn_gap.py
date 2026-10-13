"""``SpawnRegion.spawn_gap_s``: the fewest seconds between consecutive aircraft
a region spawns - traffic handed over in trail, not launched seconds apart."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import uniform

from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.sim.spawn.sources import PlanContext, RegionSource


def _region(n=4, window=(0.0, 1800.0), gap=None):
    return SpawnRegion(
        RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.6)),
        n_aircraft=n,
        params={"alt_ft": (10_000.0, 12_000.0), "spd_kts": (240.0, 250.0)},
        aircraft_type="B738",
        spawn_time=window,
        spawn_gap_s=gap,
    )


def _times(config, seed):
    return sorted(r.time_s for _, r in config.iter_requests(np.random.default_rng(seed)))


def test_consecutive_spawns_are_at_least_the_gap_apart_within_the_window():
    config = SpawnConfig(regions=[_region(gap=120.0)])
    for seed in range(200):
        t = _times(config, seed)
        assert np.diff(t).min() >= 120.0 - 1e-9
        assert 0.0 <= t[0] and t[-1] <= 1800.0


def test_the_times_still_spread_over_the_whole_window():
    config = SpawnConfig(regions=[_region(gap=120.0)])
    first = [_times(config, seed)[0] for seed in range(400)]
    last = [_times(config, seed)[-1] for seed in range(400)]
    # The first can fall anywhere up to 1800 - 3 x 120 s, the last from 360 s.
    assert min(first) < 60 and max(first) > 1000
    assert min(last) < 900 and max(last) > 1750


def test_no_gap_draws_the_episode_as_before():
    without, zero = SpawnConfig(regions=[_region()]), SpawnConfig(regions=[_region(gap=0.0)])
    for seed in range(5):
        a = list(without.iter_requests(np.random.default_rng(seed)))
        b = list(zero.iter_requests(np.random.default_rng(seed)))
        assert [r for _, r in a] == [r for _, r in b]


def test_a_gap_moves_only_the_times():
    without, gapped = SpawnConfig(regions=[_region()]), SpawnConfig(regions=[_region(gap=120.0)])
    a = [r for _, r in without.iter_requests(np.random.default_rng(3))]
    b = [r for _, r in gapped.iter_requests(np.random.default_rng(3))]
    assert [(r.at, r.alt_ft, r.actype) for r in a] == [(r.at, r.alt_ft, r.actype) for r in b]
    # The earliest drawn is still the earliest.
    assert np.argsort([r.time_s for r in a]).tolist() == np.argsort([r.time_s for r in b]).tolist()


def test_a_window_too_short_for_the_gaps_is_refused():
    with pytest.raises(ValueError, match="4 aircraft 120 s apart need 360 s; the spawn_time window is 300 s"):
        _region(window=(0.0, 300.0), gap=120.0)


def test_a_gap_must_not_be_negative():
    with pytest.raises(ValueError, match="spawn_gap_s must be >= 0"):
        _region(gap=-1.0)


def test_without_a_window_each_is_pushed_back_to_the_gap():
    region = _region(window=uniform(loc=0.0, scale=100.0), gap=60.0)
    assert region.spaced_times([30.0, 10.0, 40.0]) == [70.0, 10.0, 130.0]


def test_a_region_source_spaces_its_aircraft_too():
    region = _region(gap=120.0)
    config = SpawnConfig(regions=[region])
    plan = RegionSource(region).plan(np.random.default_rng(0), PlanContext(spawn=config))
    assert np.diff(sorted(r.time_s for r in plan)).min() >= 120.0 - 1e-9
