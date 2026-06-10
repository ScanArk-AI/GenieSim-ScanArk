#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_planner.py
带自适应安全代价场的 A* 路径规划 + String Pulling 路径平滑（封装模块）

算法要点：
1. 各向异性安全代价场
   - expanded_traversability 作为硬边界
   - 距离变换梯度方向一致性区分"直墙"与"拐角"
   - 三段式代价：危险区 → 梯度区（自适应指数衰减）→ 安全区
2. A* 搜索（8方向，欧氏距离启发）
3. 路径平滑：String Pulling（视线剪枝，代价感知）

典型用法：
    from managers.path_planner import ShortestPathPlanner

    planner = ShortestPathPlanner(traversable)
    result  = planner.plan(start, goal)
    # result["raw"]   原始 A* 路径
    # result["path"]  String Pulling 后的路径
"""

import os
import cv2
import heapq
import numpy as np
from typing import List, Optional, Tuple

# ─── 默认参数 ──────────────────────────────────────────────────────────────────
DANGER_RADIUS        = 8.0
DANGER_COST          = 300.0
WALL_WEIGHT_STRAIGHT = 15.0
WALL_WEIGHT_CORNER   = 40.0
INFLUENCE_STRAIGHT   = 12.0
INFLUENCE_CORNER     = 38.0
CORNER_DETECT_SIGMA  = 10.0
PULL_MAX_COST        = WALL_WEIGHT_CORNER   # 40.0


# ─── 节点 ──────────────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("x", "y", "g", "h", "f", "px", "py")

    def __init__(self, x: int, y: int, g: float, h: float,
                 px: int = -1, py: int = -1):
        self.x, self.y = x, y
        self.g = g
        self.h = h
        self.f = g + h
        self.px, self.py = px, py

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f


# ─── 安全代价场（各向异性） ────────────────────────────────────────────────────
def build_safety_cost(traversable: np.ndarray,
                      danger_radius:        float = DANGER_RADIUS,
                      danger_cost:          float = DANGER_COST,
                      wall_weight_straight: float = WALL_WEIGHT_STRAIGHT,
                      wall_weight_corner:   float = WALL_WEIGHT_CORNER,
                      influence_straight:   float = INFLUENCE_STRAIGHT,
                      influence_corner:     float = INFLUENCE_CORNER,
                      corner_sigma:         float = CORNER_DETECT_SIGMA,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    各向异性三段式安全代价场。

    核心思路：区分"直墙"和"拐角"，分别赋予不同的影响范围和权重。

    拐角检测（gradient direction uniformity）：
      计算距离场的梯度方向，在 corner_sigma 邻域内做圆形统计平均。
      若邻域内所有格子的梯度方向高度一致（resultant length ≈ 1）→ 直墙；
      若梯度方向分散（resultant length ≈ 0）→ 拐角或开放空间交界处。

      corner_score = 1 - resultant_length  ∈ [0, 1]
        ≈ 0 → 直墙（短影响范围，不吸引路径绕向侧方开阔区域）
        ≈ 1 → 拐角（长影响范围 + 高权重，路径主动绕离危险角落）

    三段代价（自适应影响范围）：
      危险区  d < danger_radius         → danger_cost（极高，近似阻断）
      梯度区  danger_radius ≤ d < inf   → adaptive_weight * exp(-3*(d-dr)/decay)
      安全区  d ≥ inf                   → 0

    Returns:
        (cost, dist, corner_score)
    """
    # 1. 距离变换
    dist = cv2.distanceTransform(
        (traversable > 0).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    ).astype(np.float32)

    # 2. 拐角检测：距离场梯度方向的局部一致性
    #    用 Scharr 算子（精度优于 Sobel）计算梯度
    gx = cv2.Scharr(dist, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(dist, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    gx_n = gx / mag   # 单位梯度向量 x 分量
    gy_n = gy / mag   # 单位梯度向量 y 分量

    #  在 corner_sigma 邻域内做高斯平均（近似圆形统计均值）
    k = max(int(6 * corner_sigma) | 1, 3)
    cos_mean = cv2.GaussianBlur(gx_n, (k, k), corner_sigma)
    sin_mean = cv2.GaussianBlur(gy_n, (k, k), corner_sigma)
    #  resultant length ∈ [0,1]：1=方向完全一致(直墙), 0=方向完全分散(拐角)
    uniformity = np.sqrt(cos_mean ** 2 + sin_mean ** 2).clip(0.0, 1.0)
    corner_score = (1.0 - uniformity).astype(np.float32)

    # 3. 自适应影响范围与权重（逐格插值）
    adaptive_inf    = influence_straight + corner_score * (influence_corner - influence_straight)
    adaptive_weight = wall_weight_straight + corner_score * (wall_weight_corner - wall_weight_straight)

    # 4. 三段式代价（危险区 / 梯度区 / 安全区）
    cost = np.zeros_like(dist, dtype=np.float32)

    # 危险区：极高固定代价（各处相同）
    danger_mask = dist < danger_radius
    cost[danger_mask] = danger_cost

    # 梯度区：自适应指数衰减
    #   在 d=danger_radius 处代价 = adaptive_weight，在 d=adaptive_inf 处趋近 0
    grad_mask   = (~danger_mask) & (dist < adaptive_inf)
    d_shifted   = (dist - danger_radius).clip(0.0)
    decay_range = (adaptive_inf - danger_radius).clip(1.0)
    cost[grad_mask] = (
        adaptive_weight * np.exp(-3.0 * d_shifted / decay_range)
    )[grad_mask]

    return cost, dist, corner_score


# ─── 视线检测（Bresenham + 角落保护 + 可选代价上限） ──────────────────────────
def line_of_sight(x0: int, y0: int, x1: int, y1: int,
                  traversable: np.ndarray,
                  cost_map: np.ndarray = None,
                  max_cost: float = None) -> bool:
    """
    Bresenham 直线视线检测，含对角穿角保护和可选代价上限。

    参数：
        traversable : 可通行掩码（0=不可通行）
        cost_map    : 安全代价场（可选）；若提供，则路径上所有格子的代价必须 ≤ max_cost
        max_cost    : 代价上限（可选）；路径途经格子代价超过此值时视为"不可通"

    同时传入 cost_map 和 max_cost 时，同时检查障碍物和代价，
    确保 string pulling 生成的捷径不会穿越高代价的危险区或梯度区。
    """
    H, W = traversable.shape
    check_cost = (cost_map is not None) and (max_cost is not None)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if x < 0 or x >= W or y < 0 or y >= H or traversable[y, x] == 0:
            return False
        # 代价检查：途经格子的代价不得超过阈值
        if check_cost and cost_map[y, x] > max_cost:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        step_x = e2 > -dy
        step_y = e2 <  dx
        if step_x and step_y:
            if 0 <= x + sx < W and traversable[y,      x + sx] == 0:
                return False
            if 0 <= y + sy < H and traversable[y + sy, x     ] == 0:
                return False
        if step_x:
            err -= dy
            x   += sx
        if step_y:
            err += dx
            y   += sy


# ─── A* 搜索 ──────────────────────────────────────────────────────────────────
def astar(start: Tuple[int, int],
          goal:  Tuple[int, int],
          traversable: np.ndarray,
          safety_cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    带安全代价场的 A* 路径搜索。

    - 外圈（expanded_traversability == 0）不可进入（硬约束）
    - 可通行区域内的安全代价场作为软约束，引导路径远离障碍物
    - 移动代价 = 基础欧氏距离（直行 1.0，斜行 1.414）+ 目标格安全代价

    Returns:
        路径点列表 [(x, y), ...]，从 start 到 goal；若无路径返回 []。
    """
    H, W = traversable.shape
    sx, sy = start
    gx, gy = goal

    # 边界及可通行检查
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        print("起点或终点超出地图范围")
        return []
    if traversable[sy, sx] == 0 or traversable[gy, gx] == 0:
        print("起点或终点位于不可通行区域")
        return []

    # 8 方向：(dx, dy, base_move_cost)
    DIRS = [
        (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        ( 0, -1, 1.0  ),               ( 0, 1, 1.0  ),
        ( 1, -1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
    ]

    def heuristic(x: int, y: int) -> float:
        return float(np.hypot(gx - x, gy - y))

    start_node = Node(sx, sy, 0.0, heuristic(sx, sy))
    open_set: list = [start_node]
    closed_set: set = set()
    all_nodes: dict = {(sx, sy): start_node}

    while open_set:
        cur = heapq.heappop(open_set)
        key = (cur.x, cur.y)
        if key in closed_set:
            continue
        closed_set.add(key)

        # 到达终点，回溯路径
        if cur.x == gx and cur.y == gy:
            path = []
            node = cur
            while node.px != -1:
                path.append((node.x, node.y))
                node = all_nodes[(node.px, node.py)]
            path.append((sx, sy))
            path.reverse()
            return path

        for dx, dy, base_cost in DIRS:
            nx, ny = cur.x + dx, cur.y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if traversable[ny, nx] == 0:
                continue
            if (nx, ny) in closed_set:
                continue

            # 移动代价 = 基础距离 + 安全惩罚
            new_g = cur.g + base_cost + float(safety_cost[ny, nx])
            new_h = heuristic(nx, ny)
            new_f = new_g + new_h

            if (nx, ny) not in all_nodes or new_f < all_nodes[(nx, ny)].f:
                node = Node(nx, ny, new_g, new_h, cur.x, cur.y)
                all_nodes[(nx, ny)] = node
                heapq.heappush(open_set, node)

    return []


# ─── 路径平滑 1：String Pulling（视线剪枝，代价感知） ────────────────────────
def string_pulling(path: List[Tuple[int, int]],
                   traversable: np.ndarray,
                   safety_cost: np.ndarray = None,
                   max_cost: float = PULL_MAX_COST) -> List[Tuple[int, int]]:
    """
    贪心视线剪枝（String Pulling），带代价感知。

    从当前锚点出发，找到最远的视线可达点，直接跳过中间所有节点。
    消除 8 方向栅格 A* 在近似直线方向上产生的密集锯齿转弯。

    代价感知改进：
      除障碍物硬边界外，还检查直线途经格子的代价是否超过 max_cost。
      若捷径会穿越高代价区域（如危险区 cost=300，或代价较高的拐角梯度区），
      则拒绝该捷径，保留原始 A* 路径绕行的结果。
      这避免了"string pulling 把 A* 精心绕开的高代价区域重新穿回去"的问题。

    参数：
        safety_cost : 安全代价场（None 表示仅做障碍物检查，与原行为相同）
        max_cost    : 捷径途经格子的代价上限（默认 PULL_MAX_COST = WALL_WEIGHT_CORNER）
    """
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        far = anchor + 1
        for j in range(len(path) - 1, anchor + 1, -1):
            if line_of_sight(path[anchor][0], path[anchor][1],
                             path[j][0],      path[j][1],
                             traversable,
                             cost_map=safety_cost,
                             max_cost=max_cost):
                far = j
                break
        result.append(path[far])
        anchor = far
    return result


class ShortestPathPlanner:
    """
    带自适应安全代价场的 A* 路径规划器。

    将安全代价场构建、A* 搜索与路径平滑封装为统一接口，
    实例化后可重复调用 plan() 为不同起终点规划路径。

    典型用法::

        planner = ShortestPathPlanner(traversable)
        result  = planner.plan(start, goal)
        # result["raw"]      原始 A* 路径
        # result["result"]   String Pulling 后的路径

    参数均与模块级常量对应，可在实例化时覆盖：

        planner = ShortestPathPlanner(
            traversable,
            danger_radius=6.0,
        )
    """

    def __init__(self,
                 traversable: np.ndarray,
                 *,
                 danger_radius:        float = DANGER_RADIUS,
                 danger_cost:          float = DANGER_COST,
                 wall_weight_straight: float = WALL_WEIGHT_STRAIGHT,
                 wall_weight_corner:   float = WALL_WEIGHT_CORNER,
                 influence_straight:   float = INFLUENCE_STRAIGHT,
                 influence_corner:     float = INFLUENCE_CORNER,
                 corner_sigma:         float = CORNER_DETECT_SIGMA,
                 pull_max_cost:        float = None):
        self.traversable         = traversable
        self.danger_radius        = danger_radius
        self.danger_cost          = danger_cost
        self.wall_weight_straight = wall_weight_straight
        self.wall_weight_corner   = wall_weight_corner
        self.influence_straight   = influence_straight
        self.influence_corner     = influence_corner
        self.corner_sigma         = corner_sigma
        # 默认与 wall_weight_corner 保持一致
        self.pull_max_cost = pull_max_cost if pull_max_cost is not None else wall_weight_corner

        # 预计算安全代价场（耗时较长，只算一次）
        self.safety_cost, self.dist_field, self.corner_score = build_safety_cost(
            traversable,
            danger_radius        = self.danger_radius,
            danger_cost          = self.danger_cost,
            wall_weight_straight = self.wall_weight_straight,
            wall_weight_corner   = self.wall_weight_corner,
            influence_straight   = self.influence_straight,
            influence_corner     = self.influence_corner,
            corner_sigma         = self.corner_sigma,
        )

    # ── 核心步骤（直接委托给模块级函数，避免重复代码） ──────────────────────────

    def astar(self, start: Tuple[int, int],
              goal:  Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* 搜索，返回原始栅格路径。"""
        return astar(start, goal, self.traversable, self.safety_cost)

    def string_pulling(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """String Pulling 视线剪枝，消除冗余中间节点。"""
        return string_pulling(path, self.traversable,
                              self.safety_cost, self.pull_max_cost)

    # ── 一键规划接口 ─────────────────────────────────────────────────────────────

    def plan(self, start: Tuple[int, int],
             goal:  Tuple[int, int]) -> dict:
        """
        完整规划流水线：A* → String Pulling → 移动平均平滑。

        Returns:
            dict with keys:
                "raw"      : 原始 A* 路径点列表
                "result"   : String Pulling 后的路径点列表
            若无路径，三个列表均为空。
        """
        raw = self.astar(start, goal)
        if not raw:
            return {"raw": [], "result": []}
        pulled   = self.string_pulling(raw)
        return {"raw": raw, "result": pulled}

    @staticmethod
    def path_length(pts: List[Tuple[int, int]],
                    resolution: float = 1.0) -> float:
        """计算路径总长度（默认单位：格子数；乘以 resolution 得米）。"""
        if len(pts) < 2:
            return 0.0
        arr = np.array(pts, dtype=np.float64)
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))) * resolution

    def visualize_path(self,
                       plan_result: dict,
                       out_path: str,
                       bg_image: Optional[np.ndarray] = None,
                       resolution: float = 1.0) -> None:
        """
        将 plan() 结果绘制到地图上并保存为图像。

        Args:
            plan_result : plan() 的返回值 {"raw": ..., "result": ...}
            out_path    : 输出图像路径（.png）
            bg_image    : 可选 BGR 背景图（H×W×3，如彩色点云投影）；
                          为 None 时用可通行掩码生成灰度背景
            resolution  : 地图分辨率（米/格），仅用于图例中显示路径长度
        """
        raw    = plan_result.get("raw",    [])
        result = plan_result.get("result", [])

        # ── 背景 ──────────────────────────────────────────────────────────────
        if bg_image is not None:
            base = bg_image.copy()
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        else:
            base = np.zeros((*self.traversable.shape, 3), dtype=np.uint8)
            base[self.traversable > 0] = [55, 55, 55]

        # ── 叠加安全代价热图（半透明） ─────────────────────────────────────────
        trav_mask = self.traversable > 0
        if self.safety_cost.max() > 0:
            cost_clip = self.safety_cost.clip(0, WALL_WEIGHT_CORNER * 1.5)
            cost_norm = (cost_clip / cost_clip.max() * 255).astype(np.uint8)
            cost_heat = cv2.applyColorMap(cost_norm, cv2.COLORMAP_HOT)
            base = base.astype(np.float32)
            base[trav_mask] = (0.55 * base[trav_mask]
                               + 0.45 * cost_heat[trav_mask].astype(np.float32))
            base = base.astype(np.uint8)
        base[self.traversable == 0] = [0, 0, 0]

        # ── 路径绘制辅助 ───────────────────────────────────────────────────────
        def draw_line(pts, color, thickness):
            for i in range(len(pts) - 1):
                cv2.line(base, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

        # 原始 A* 路径（细，橙红）
        if raw:
            draw_line(raw, (40, 90, 210), 1)

        # String Pulling 结果路径（粗，青绿 + 中间节点标记）
        if result:
            draw_line(result, (30, 220, 80), 2)
            for pt in result[1:-1]:
                cv2.circle(base, pt, 5, (0, 240, 180), -1)
                cv2.circle(base, pt, 5, (255, 255, 255), 1)

        # 起点（绿圆）/ 终点（蓝圆）
        ref = raw or result
        if ref:
            start, goal = ref[0], ref[-1]
            cv2.circle(base, start, 9, (0, 255,   0), -1)
            cv2.circle(base, start, 9, (0, 140,   0),  2)
            cv2.circle(base, goal,  9, (30,  30, 255), -1)
            cv2.circle(base, goal,  9, (0,    0, 160),  2)

        # ── 图例 ───────────────────────────────────────────────────────────────
        len_raw = self.path_length(raw,    resolution)
        len_res = self.path_length(result, resolution)
        legend = [
            (( 40,  90, 210), f"Raw A*           {len(raw)} pts  {len_raw:.1f} m"),
            (( 30, 220,  80), f"String Pulling   {len(result)} pts  {len_res:.1f} m"),
            ((  0, 255,   0), "Start"),
            (( 30,  30, 255), "Goal"),
        ]
        lx, ly = 10, 24
        for color, text in legend:
            cv2.rectangle(base, (lx, ly - 12), (lx + 18, ly + 4), color, -1)
            cv2.putText(base, text, (lx + 24, ly + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            ly += 22

        # ── 保存 ───────────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, base)
        print(f"[PathPlanner] 路径可视化已保存: {out_path}")
