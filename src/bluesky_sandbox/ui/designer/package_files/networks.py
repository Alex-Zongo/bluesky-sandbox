"""The policy's network: how it reads the ownship and every intruder.

This task's own copy, written by the designer - change it as you like. SB3
builds the policy's features from it (``features_extractor_class`` in
``train.py``); its keyword arguments are ``policy_kwargs``'
``features_extractor_kwargs``, such as ``{"d_model": 128, "n_layers": 3}``.
"""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

__all__ = ["IntruderTransformer"]


class IntruderTransformer(BaseFeaturesExtractor):
    """Features from the ownship and every intruder, as a transformer reads
    them: each one a token, attended over by the others; the features are the
    ownship's token after ``n_layers`` layers.

    It reads the observation as ``train.py``'s ``make_env`` pads it: ``ownship`` a vector, ``intruders`` a row each, its last column the
    validity flag. Padded rows are masked out, so any number of intruders -
    none included - is read alike, and in no order: an intruder's row says
    where it is, not its place in the block. Other parts of the observation are
    not read.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)
        own = observation_space["ownship"]
        intruders = observation_space.spaces.get("intruders")
        self.own_embed = nn.Linear(int(own.shape[0]), d_model)
        # The validity flag masks a row, rather than being read.
        self.intruder_embed = (
            None if intruders is None else nn.Linear(int(intruders.shape[1]) - 1, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        tokens = self.own_embed(observations["ownship"]).unsqueeze(1)
        padded = None
        if self.intruder_embed is not None:
            rows = observations["intruders"]
            tokens = th.cat([tokens, self.intruder_embed(rows[..., :-1])], dim=1)
            # The ownship's token is never padding, so no row attends to nothing.
            own = th.zeros(rows.shape[0], 1, dtype=th.bool, device=rows.device)
            padded = th.cat([own, rows[..., -1] < 0.5], dim=1)
        return self.encoder(tokens, src_key_padding_mask=padded)[:, 0]
