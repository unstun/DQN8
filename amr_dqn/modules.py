"""Plug-in modules for CNN Q-networks.

Provides
--------
- SpatialMHA   Multi-head self-attention over spatial positions of a CNN feature map.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialMHA(nn.Module):
    """Multi-head self-attention over spatial positions of a feature map.

    Treats each spatial position (H*W) as a token with *channels* dimensions.
    Applies standard multi-head attention followed by a residual connection
    and LayerNorm.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        out, _ = self.mha(tokens, tokens, tokens)    # self-attention
        out = self.norm(tokens + out)                # residual + LN
        return out.transpose(1, 2).reshape(B, C, H, W)
