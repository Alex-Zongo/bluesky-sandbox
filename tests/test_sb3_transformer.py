"""SB3's policy reads every intruder through a transformer: padded rows are
masked out, and the intruders are read in no order."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict

th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from bluesky_sandbox.ui.designer.package_files.networks import IntruderTransformer  # noqa: E402

N_INTRUDERS, N_FEATURES = 5, 3


def _space(intruders: bool = True) -> Dict:
    parts = {"ownship": Box(-1, 1, (4,))}
    if intruders:
        parts["intruders"] = Box(-1, 1, (N_INTRUDERS, N_FEATURES + 1))
    return Dict(parts)


def _obs(n_real: int, rng) -> dict[str, th.Tensor]:
    rows = np.zeros((1, N_INTRUDERS, N_FEATURES + 1), dtype=np.float32)
    rows[0, :n_real, :-1] = rng.uniform(-1, 1, (n_real, N_FEATURES))
    rows[0, :n_real, -1] = 1.0
    own = rng.uniform(-1, 1, (1, 4)).astype(np.float32)
    return {"ownship": th.as_tensor(own), "intruders": th.as_tensor(rows)}


@pytest.fixture
def extractor():
    th.manual_seed(0)
    return IntruderTransformer(_space()).eval()


def test_the_features_are_the_ownship_token(extractor):
    obs = _obs(3, np.random.default_rng(0))
    assert extractor(obs).shape == (1, extractor.features_dim)


def test_a_padded_row_is_not_read(extractor):
    obs = _obs(2, np.random.default_rng(1))
    before = extractor(obs)
    obs["intruders"][0, 2:, :-1] = 9.0  # padding, whatever it holds
    th.testing.assert_close(extractor(obs), before)


def test_the_intruders_are_read_in_no_order(extractor):
    obs = _obs(4, np.random.default_rng(2))
    before = extractor(obs)
    obs["intruders"] = obs["intruders"][:, [3, 0, 2, 1, 4]]
    th.testing.assert_close(extractor(obs), before)


def test_no_intruders_reads_the_ownship_alone(extractor):
    obs = _obs(0, np.random.default_rng(3))
    assert th.isfinite(extractor(obs)).all()
    lone = IntruderTransformer(_space(intruders=False))
    assert lone({"ownship": obs["ownship"]}).shape == (1, lone.features_dim)
