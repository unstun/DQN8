"""Uniform experience-replay buffer with DQfD demo-preservation.

Stores (obs, action, reward, next_obs, done, next_action_mask, demo_flag, n_steps)
in pre-allocated numpy arrays for cache-friendly random sampling.

DQfD safeguard: when the buffer is full and a non-demo transition would overwrite a
demo slot, the buffer scans forward to find a non-demo slot instead.  This keeps
expert demonstrations available throughout training for the margin/CE losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_action_masks: np.ndarray
    dones: np.ndarray
    demos: np.ndarray
    n_steps: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, *, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self._rng = rng

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_action_masks = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._demos = np.zeros((self.capacity,), dtype=np.float32)
        self._n_steps = np.ones((self.capacity,), dtype=np.int64)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        next_action_mask: np.ndarray | None = None,
        demo: bool = False,
        n_steps: int = 1,
    ) -> None:
        i = self._idx
        # DQfD-style safeguard: preserve demonstration transitions so they
        # remain available for supervised losses throughout long trainings.
        #
        # If the buffer is full and we'd overwrite a demo transition with a
        # non-demo transition, search for the next non-demo slot to overwrite.
        if self._size >= self.capacity and (not bool(demo)) and float(self._demos[i]) > 0.5:
            j = int(i)
            for _ in range(int(self.capacity)):
                if float(self._demos[j]) <= 0.5:
                    i = int(j)
                    break
                j = (int(j) + 1) % int(self.capacity)
        self._obs[i] = obs
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._next_obs[i] = next_obs
        if next_action_mask is None:
            self._next_action_masks[i] = True
        else:
            m = np.asarray(next_action_mask, dtype=bool).reshape(-1)
            if m.size != int(self.n_actions):
                raise ValueError("next_action_mask must have shape (n_actions,)")
            self._next_action_masks[i] = True if not bool(m.any()) else m
        self._dones[i] = 1.0 if done else 0.0
        self._demos[i] = 1.0 if demo else 0.0
        self._n_steps[i] = int(max(1, int(n_steps)))

        self._idx = (int(i) + 1) % int(self.capacity)
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        n = min(int(batch_size), self._size)
        idxs = self._rng.integers(0, self._size, size=n, dtype=np.int64)
        return Batch(
            obs=self._obs[idxs],
            actions=self._actions[idxs],
            rewards=self._rewards[idxs],
            next_obs=self._next_obs[idxs],
            next_action_masks=self._next_action_masks[idxs],
            dones=self._dones[idxs],
            demos=self._demos[idxs],
            n_steps=self._n_steps[idxs],
        )
