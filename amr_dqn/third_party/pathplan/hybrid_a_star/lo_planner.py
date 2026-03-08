"""LO-Hybrid A* planner — 内层改进 Hybrid A*。

忠实复现论文：
  Chen et al. 2025, "Improved Hybrid A* Algorithm Based on Lemming
  Optimization for Path Planning of Autonomous Vehicles",
  Applied Sciences 15(14) 7734.

内层两项改进：
1. **八分距离启发式 H*(n)**（论文 Eq. 20）
   h = (√2-1)·min(|dx|,|dy|) + max(|dx|,|dy|)
   取 max(octile, 父节点启发式) 保单调性。

2. **航向变化惩罚 P_θ·|Δθ|**（论文 Eq. 22）
   在 _evaluate_primitive 中对 g-cost 加上 P_θ·|Δheading|。

删除内容（相对旧版本）：
- 障碍距离惩罚因子 _obstacle_penalty_factor
- 多曲率 RS 展开 _try_analytic_expansion
- obstacle_field 依赖与 _dist_field 初始化
- safety_margin / obstacle_penalty_weight / multi_curvature_rs / curvature_steps 参数
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..primitives import MotionPrimitive
from ..robot import AckermannParams, AckermannState
from .planner import HybridAStarPlanner, Node


def _octile_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """八分距离（Octile Distance），论文 Eq. 20 的欧式格近似。

    h = (√2-1)·min(|dx|,|dy|) + max(|dx|,|dy|)
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    return (math.sqrt(2.0) - 1.0) * min(dx, dy) + max(dx, dy)


class LOHybridAStarPlanner(HybridAStarPlanner):
    """内层改进 Hybrid A* —— 八分启发式 + 航向变化惩罚。

    继承自 :class:`HybridAStarPlanner`，新增一个构造参数：

    * ``heading_change_penalty`` (P_θ)：每弧度航向变化的额外 g-cost，
      对应论文 Eq. 22 的 P_θ·|Δθ|。
    """

    def __init__(
        self,
        grid_map,
        footprint,
        params: AckermannParams,
        *,
        heading_change_penalty: float = 0.01,
        **kwargs,
    ):
        # 父节点的 steering_penalty 设为 0，避免与航向惩罚重复计算
        kwargs.setdefault("steering_penalty", 0.0)
        super().__init__(grid_map, footprint, params, **kwargs)
        self.heading_change_penalty = max(0.0, float(heading_change_penalty))

    # ------------------------------------------------------------------
    # 1. 八分距离启发式（论文 Eq. 20）
    # ------------------------------------------------------------------

    def _heuristic(self, state, goal, dist_map, goal_center, goal_offset) -> float:
        """取 max(八分距离, 父节点启发式)，保证单调性（论文 Eq. 20）。"""
        # 父节点启发式（通常是 Dijkstra 预处理 + RS 解析展开距离）
        h_parent = super()._heuristic(state, goal, dist_map, goal_center, goal_offset)
        # 八分距离近似
        h_oct = _octile_distance(state.x, state.y, goal.x, goal.y)
        # 取较大值保效率（不降低可采纳性：二者均是下界）
        return max(h_oct, h_parent)

    # ------------------------------------------------------------------
    # 2. 航向变化惩罚（论文 Eq. 22）
    # ------------------------------------------------------------------

    def _evaluate_primitive(
        self, current: Node, prim: MotionPrimitive, goal: AckermannState
    ) -> Optional[Tuple[AckermannState, float]]:
        result = super()._evaluate_primitive(current, prim, goal)
        if result is None:
            return None
        nxt, g_new = result

        # 计算实际航向差 |Δθ|，归一化到 [0, π]
        dtheta = abs(nxt.theta - current.state.theta)
        # 折叠到 [0, π]（角度差最大为 π）
        if dtheta > math.pi:
            dtheta = 2.0 * math.pi - dtheta

        # 论文 Eq. 22：g_new += P_θ · |Δθ|
        g_new += self.heading_change_penalty * dtheta
        return nxt, g_new
