"""Lemming Optimization Algorithm (LOA) — 论文忠实复现。

复现自：
  Chen et al. 2025, "Improved Hybrid A* Algorithm Based on Lemming
  Optimization for Path Planning of Autonomous Vehicles",
  Applied Sciences 15(14) 7734.

四种行为（按论文公式编号）：

  1. 迁徙（Migration） — Eq. 3
     Z_new = Z_best + F · BM · (R·(Z_best - Z_i) + (1-R)·(Z_i - Z_a))
     其中 BM = Brownian 随机向量，R = rand, Z_a = 种群均值

  2. 挖洞（Burrow digging） — Eq. 7
     Z_new = Z_i + F · L · (Z_best - Z_b)
     其中 L = rand · sin(t/2)，Z_b = 随机个体

  3. 觅食（Foraging） — Eq. 9
     Z_new = Z_best + F · spiral · (Z_best - Z_i)
     其中 spiral = radius · (sin(2πr) + cos(2πr))（Eq. 10），radius = ‖Z_best - Z_i‖（Eq. 11）
     注：论文原文为 spiral·Z_i，但按收敛语义应为差向量

  4. 逃避（Predator evasion） — Eq. 12
     Z_new = Z_best + F · G · Levy(Dim) · (Z_best - Z_i)
     其中 G = 2·(1 - t/T)（探索因子随迭代递减）

初始化（Algorithm 1）：
  Z_i = seed + N(0, 0.2·range)，即以 seed 为中心的正态分布。
  若无 seed 则以 (lo+hi)/2 为中心。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _brownian(dim: int, rng: np.random.Generator) -> np.ndarray:
    """标准 Brownian 随机向量（均值0，方差1）。"""
    return rng.normal(0.0, 1.0, size=dim)


def _levy_flight(dim: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    """Mantegna 算法生成 Lévy 飞行步长向量（论文 Eq. 14）。

    Levy(x) = 0.01 × u·σ / |v|^{1/β}
    """
    sigma_u = (
        math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.normal(0.0, 1.0, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    v_abs = np.maximum(np.abs(v), 1e-12)
    # 论文 Eq. 14：0.01 缩放因子防止步长过大
    return 0.01 * u * sigma_u / (v_abs ** (1.0 / beta))


# ------------------------------------------------------------------
# 主类
# ------------------------------------------------------------------

class LemmingOptimizer:
    """基于旅鼠行为的元启发式优化器（LOA）。

    Parameters
    ----------
    population_size : int
        种群规模（论文默认 20）。
    max_iterations : int
        最大迭代次数 T。
    F : float
        全局缩放因子（论文 F，默认 0.5）。
    seed : int or None
        随机数种子，保证可重复性。
    """

    def __init__(
        self,
        population_size: int = 20,
        max_iterations: int = 30,
        seed: int | None = None,
    ):
        self.population_size = max(4, int(population_size))
        self.max_iterations = max(1, int(max_iterations))
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        seed_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """最小化 fitness_fn，返回最优参数向量。

        Parameters
        ----------
        fitness_fn : callable
            接受 1-D 参数向量，返回标量 fitness（越小越好）。
            不可行解返回 ``+inf``。
        bounds : list of (lo, hi)
            各维度搜索范围。
        seed_params : ndarray or None
            可选的已知较好起点（论文 Algorithm 1 的初始化中心）。

        Returns
        -------
        best_params : ndarray
            找到的最优参数向量。
        """
        dim = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo  # 各维度范围

        # ----------------------------------------------------------------
        # 初始化（论文 Algorithm 1）：
        # pop[0] = seed（精英保留），其余以 seed 为中心正态采样
        # ----------------------------------------------------------------
        center = np.clip(seed_params, lo, hi) if seed_params is not None else (lo + hi) * 0.5
        sigma = 0.2 * span  # 标准差 = 0.2 × range（论文 Algorithm 1）
        pop = center + self.rng.normal(0.0, 1.0, size=(self.population_size, dim)) * sigma
        pop = np.clip(pop, lo, hi)
        pop[0] = center  # Algorithm 1: 第一个个体保留为 seed_params

        # 评估初始适应度
        fitness = np.array([fitness_fn(pop[i]) for i in range(self.population_size)], dtype=np.float64)

        best_idx = int(np.argmin(fitness))
        best_params = pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        T = self.max_iterations  # 最大迭代数，全局引用

        for t in range(1, T + 1):
            # 当前迭代进度比例
            t_ratio = float(t) / float(T)

            # G 因子（逃避行为 Eq. 13）：G = 2·(1 - t/T)
            G = 2.0 * (1.0 - t_ratio)

            for i in range(self.population_size):
                # 论文 Eq. 5：方向参数 F = ±1（每个体每步独立抽取）
                F = 1.0 if self.rng.random() < 0.5 else -1.0

                # 随机选择行为（四种各占 25%）
                behaviour = self.rng.random()

                Z_i = pop[i]
                Z_best = best_params

                if behaviour < 0.25:
                    # -------------------------------------------------
                    # 行为 1：迁徙（Migration）— 论文 Eq. 3
                    # Z_new = Z_best + F · BM · (R·(Z_best-Z_i) + (1-R)·(Z_i-Z_a))
                    # -------------------------------------------------
                    BM = _brownian(dim, self.rng)
                    # Eq. 6: R = 2·rand(1,Dim) - 1，向量 ∈ [-1,1]^dim
                    R = 2.0 * self.rng.random(dim) - 1.0
                    # Eq. 3: Z_a = 种群均值（论文原文）
                    Z_a = np.mean(pop, axis=0)
                    candidate = Z_best + F * BM * (R * (Z_best - Z_i) + (1.0 - R) * (Z_i - Z_a))

                elif behaviour < 0.50:
                    # -------------------------------------------------
                    # 行为 2：挖洞（Burrow digging）— 论文 Eq. 7
                    # Z_new = Z_i + F · L · (Z_best - Z_b)
                    # L = rand · sin(t/2)（Eq. 8），Z_b 为随机个体
                    # -------------------------------------------------
                    b_idx = int(self.rng.integers(0, self.population_size))
                    Z_b = pop[b_idx]
                    L = self.rng.random() * math.sin(t / 2.0)
                    candidate = Z_i + F * L * (Z_best - Z_b)

                elif behaviour < 0.75:
                    # -------------------------------------------------
                    # 行为 3：觅食（Foraging）— 论文 Eq. 9
                    # Z_new = Z_best + F · spiral · (Z_best - Z_i)
                    # spiral = radius · (sin(2πr) + cos(2πr))（Eq. 10）
                    # radius = ‖Z_best - Z_i‖（Eq. 11）
                    # 注：论文字面为 spiral·Z_i，但按收敛语义应为差向量
                    # -------------------------------------------------
                    r_angle = self.rng.random()
                    radius = float(np.linalg.norm(Z_best - Z_i))
                    spiral = radius * (math.sin(2.0 * math.pi * r_angle) + math.cos(2.0 * math.pi * r_angle))
                    candidate = Z_best + F * spiral * (Z_best - Z_i)

                else:
                    # -------------------------------------------------
                    # 行为 4：逃避（Predator evasion）— 论文 Eq. 12
                    # Z_new = Z_best + F · G · Levy(Dim) · (Z_best - Z_i)
                    # G = 2·(1 - t/T)（Eq. 13）
                    # -------------------------------------------------
                    levy = _levy_flight(dim, self.rng)
                    candidate = Z_best + F * G * levy * (Z_best - Z_i)

                # 约束到搜索范围
                candidate = np.clip(candidate, lo, hi)

                # 贪婪选择（只有更优时才替换）
                f_new = fitness_fn(candidate)
                if f_new < fitness[i]:
                    pop[i] = candidate
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_params = candidate.copy()

            # 每轮更新最优索引（用于下轮 Z_best 引用）
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_fitness:
                best_fitness = float(fitness[best_idx])
                best_params = pop[best_idx].copy()

        return best_params
