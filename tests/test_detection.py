"""The pair geometry and conflict detection are BlueSky's own: positions on the
projection its detector measures on, and conflicts and losses of separation
from its ``detect`` on the current state - with either detector."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.geometry import conflict_geometry, detection, predicted_conflict
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=SpawnConfig(regions=[]), queryables={}, max_aircraft=0)


# Converging on (52, 4) from every side at FL100, one inside another's zone,
# one 2000 ft above the rest: conflicts, a loss of separation, and pairs clear
# by altitude.
_TRAFFIC = [
    ("AAA1", 52.0, 3.5, 90, 250, 10_000),
    ("AAA2", 52.0, 4.5, 270, 260, 10_000),
    ("AAA3", 51.6, 4.0, 0, 240, 10_000),
    ("AAA4", 52.02, 3.52, 120, 250, 10_000),
    ("AAA5", 52.3, 4.0, 180, 250, 12_000),
]


@pytest.fixture(params=["CSTATEBASED", "STATEBASED"])
def env(request):
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=1.0, obs_fields=[], action_fields=[], cd_method=request.param))
    env.reset(seed=0)
    for acid, lat, lon, hdg, cas, alt in _TRAFFIC:
        bs.traf.cre(acid, "B738", lat, lon, hdg, alt * 0.3048, cas * 0.514444)
    env.step({})
    try:
        yield env
    finally:
        env.close()


def _bluesky_detects_now():
    """BlueSky's detector, run by BlueSky itself on the state as it is now."""
    bs.traf.cd.update(bs.traf, bs.traf)
    return bs.traf.cd


def test_conflicts_and_losses_are_those_bluesky_detects(env):
    found = detection()
    cd = _bluesky_detects_now()
    index = {acid: i for i, acid in enumerate(bs.traf.id)}
    conf = {(index[a], index[b]) for a, b in cd.confpairs}
    los = {(index[a], index[b]) for a, b in cd.lospairs}
    assert conf and los
    assert set(zip(*np.nonzero(found.in_conf))) == conf
    assert set(zip(*np.nonzero(found.in_los))) == los
    for (a, b), tlos in zip(cd.confpairs, cd.tLOS):
        assert found.tlos_s[index[a], index[b]] == pytest.approx(tlos)


def test_the_geometry_is_the_detectors(env):
    geom = conflict_geometry()
    cd = _bluesky_detects_now()
    index = {acid: i for i, acid in enumerate(bs.traf.id)}
    for (a, b), qdr, dist, tcpa, dcpa in zip(cd.confpairs, cd.qdr, cd.dist, cd.tcpa, cd.dcpa):
        i, j = index[a], index[b]
        east, north = geom.rel_east_now_nm[i, j], geom.rel_north_now_nm[i, j]
        assert np.hypot(east, north) * 1852.0 == pytest.approx(dist, rel=1e-9)
        assert np.degrees(np.arctan2(east, north)) % 360.0 == pytest.approx(qdr % 360.0, abs=1e-6)
        assert geom.tcpa_s[i, j] == pytest.approx(tcpa, rel=1e-6)
        if np.isfinite(dcpa):  # the compiled detector's is NaN for an exact head-on
            assert geom.dcpa_nm[i, j] * 1852.0 == pytest.approx(dcpa, rel=1e-6, abs=1e-3)


def test_the_conflict_fields_read_it(env):
    found = detection()
    rows = np.arange(bs.traf.ntraf).reshape(-1, 1)
    assert np.array_equal(obs.InConf().get_pair_matrix(rows) == 1.0, found.in_conf)
    assert np.array_equal(obs.InLosNow().get_pair_matrix(rows) == 1.0, found.in_los)


def test_a_wider_zone_is_still_bluesky_detecting(env):
    wide = detection(rpz_nm=40.0)
    assert wide.in_conf.sum() > detection().in_conf.sum()
    assert np.array_equal(obs.InConf(rpz_nm=40.0).get_pair_matrix(np.arange(bs.traf.ntraf).reshape(-1, 1)) == 1.0, wide.in_conf)


def test_a_spawn_is_checked_by_bluesky_detection(env):
    # Head-on with AAA3, 36 NM apart and level: a conflict ahead, not yet a loss.
    assert predicted_conflict(52.2, 4.0, 10_000, 180, 250)
    # The same, 3000 ft above everything: clear.
    assert not predicted_conflict(52.2, 4.0, 15_000, 180, 250)
