"""Epsilon-decay schedules for exploration.

- linear_epsilon()    Linear decay from eps_start -> eps_final over decay_episodes.
- adaptive_epsilon()  Sigmoid decay (paper Eq. 15): smooth transition with configurable steepness.
"""

from __future__ import annotations

import math


def linear_epsilon(episode: int, *, eps_start: float, eps_final: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return float(eps_final)
    t = min(max(episode, 0), decay_episodes)
    frac = 1.0 - (t / float(decay_episodes))
    return float(eps_final + (eps_start - eps_final) * frac)


def adaptive_epsilon(episode: int, *, eps_start: float, eps_final: float, eps_decay: float) -> float:
    """Adaptive epsilon schedule (paper Eq. 15).

    Eq. (15) in the paper:
        eps_k = eps_f + (eps_i - eps_f) / (1 + exp(k / eps_d))
    """
    if eps_decay <= 0:
        return float(eps_final)

    k = max(0.0, float(episode))
    denom = 1.0 + math.exp(k / float(eps_decay))
    eps = float(eps_final) + (float(eps_start) - float(eps_final)) / float(denom)
    return float(max(min(eps, float(eps_start)), float(eps_final)))
