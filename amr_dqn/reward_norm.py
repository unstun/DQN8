"""Running reward normalization (Welford online mean/std).

V8→V9: normalize at sampling time, not at collection time.
Keeps clip as a safety net on the *normalized* value.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningRewardNormalizer:
    """Welford online running mean/std for reward normalization."""

    def __init__(self, clip: float = 5.0, eps: float = 1e-8):
        self.clip = float(clip)
        self.eps = float(eps)
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return max(float(np.sqrt(self._M2 / self._count)), self.eps)

    def update(self, reward: float) -> None:
        """Update running stats (call on raw reward, including demos)."""
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._M2 += delta * delta2

    def normalize(self, reward: float) -> float:
        """Normalize + clip. Call update() first."""
        if self._count < 2:
            return float(np.clip(reward, -self.clip, self.clip))
        normed = (reward - self._mean) / self.std
        return float(np.clip(normed, -self.clip, self.clip))

    def normalize_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        """Vectorized normalize + clip for a batch of rewards (no stats update)."""
        if self._count < 2:
            return rewards.clamp(-self.clip, self.clip)
        normed = (rewards - self._mean) / self.std
        return normed.clamp(-self.clip, self.clip)
