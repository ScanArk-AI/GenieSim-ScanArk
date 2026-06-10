#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan2Occ3D - 点云路径规划与相机位姿生成工具 (Python版)
功能:
1. 读取PLY格式点云文件
2. 生成障碍物地图和可通行性地图
3. A*路径规划算法
4. 交互式可视化界面
5. 生成相机位姿并保存为JSON
"""

import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入log管理器
from managers.log_manager import logger, setup_file_logger

# 导入地图管理类
from managers.occupancy_map_manager import OccupancyMapManager

# 导入路径规划器
from managers.path_planner import ShortestPathPlanner

# 导入渲染管理类
try:
    from managers.render_manager import RenderManager
    RENDER_AVAILABLE = True
except ImportError as e:
    logger.info(f"警告: 渲染管理器不可用: {e}")
    RENDER_AVAILABLE = False

# 设置环境变量以避免 Jupyter comm 错误
import os
os.environ['JUPYTER_PLATFORM_DIRS'] = '1'

try:
    import open3d as o3d
except ImportError:
    logger.info("错误: 需要安装 open3d 库")
    logger.info("请运行: pip install open3d")
    sys.exit(1)
except Exception as e:
    logger.info(f"警告: open3d 导入时出现问题: {e}")
    logger.info("尝试仅导入必需模块...")
    try:
        # 仅导入必需的 io 模块
        import open3d.io as o3d_io
        import open3d.geometry as o3d_geometry
        import open3d.utility as o3d_utility
        # 创建一个模拟的 o3d 对象
        class O3DWrapper:
            io = o3d_io
            geometry = o3d_geometry
            utility = o3d_utility
        o3d = O3DWrapper()
    except Exception as e2:
        logger.info(f"错误: 无法导入 open3d: {e2}")
        sys.exit(1)



class Scan2Occ3D:
    """点云路径规划主类"""
    
    def __init__(self, args=None):
        # 地图管理器
        self.map_manager = None
        
        # 全局变量用于交互式可视化
        self.display_image = None
        self.expanded_traversability = None
        self.original_traversability = None
        self.obstacle_map = None  # 添加障碍物地图
        self.point_cloud_coverage = None  # 点云覆盖掩码
        self.grid_width = 0
        self.grid_height = 0
        self.grid_resolution = 0.0
        self.min_pt = None
        self.max_pt = None
        
        # 路径规划相关
        self.current_path = []
        self.camera_poses_grid = []
        self.camera_poses_directions = []
        self.path_found = False
        self._path_planner = None  # ShortestPathPlanner 懒加载实例
        self._trajectory_paths: dict = {}  # {trajectory_id: path} 路径缓存，地图变更时清空
        self.show_trajectories: bool = True  # 'h' 键切换：是否显示历史轨迹
        self.trajectory_display_count: int = 3  # 最多显示最近 N 条轨迹
        
        # 交互状态
        self.start_point = None
        self.goal_point = None
        self.start_yaw = 0.0  # 起点朝向（弧度）
        self.goal_yaw = 0.0  # 终点朝向（弧度）
        self.waypoints = []  # 途径点列表
        self.start_set = False
        self.goal_set = False
        self.waypoint_mode = False  # 途径点添加模式
        
        # 鼠标拖拽状态
        self.dragging_start = False
        self.dragging_goal = False
        self.drag_start_pos = None
        
        # 键盘状态（用于组合键）
        self.key_s_pressed = False  # 's' 键按下状态
        self.key_g_pressed = False  # 'g' 键按下状态
        
        # log相关
        self.args = args
        self.output_dir = os.path.join(self.args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f'Output directory: {self.output_dir}')
        
        # 渲染和标注相关
        self.enable_render = getattr(args, 'render', False)
        self.enable_annotate = getattr(args, 'annotate', False)
        self.ply_path = getattr(args, 'ply_path', None)
        self.camera_type = getattr(args, 'camera_type', 'single')  # 'single' 或 'pano'
        self.camera_height = getattr(args, 'camera_height', 1.2)   # 相机距地高度（米）
        self.current_episode_id = 0
        self.annotate_episodes = []
        self.last_render_dir = None
        self.instruction_input_mode = False
        self.current_instruction = ""
        self.last_saved_mode = "discrete"
        
        # 加载3dgs渲染器
        self.render_manager = None
        self._init_render_manager() 
        
        # 视频播放控制
        self.rgb_cap = None
        self.depth_cap = None
        self.video_playing = False
        self.video_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.video_fps = 10
        
        # GUI 状态
        self.text_input_active = False
        self.text_input_buffer = ""
        self.text_cursor_visible = True
        self.text_cursor_timer = 0
        self.integrated_window_created = False  # 跟踪集成界面窗口是否已创建
        self.request_video_switch = False  # 请求切换视频的标志
        
        # 终端输入状态
        self.terminal_input_active = False
        self.terminal_input_thread = None
        self.terminal_input_lock = threading.Lock()
        
        # 终端命令监听
        self.terminal_command_thread = None
        self.terminal_command_enabled = False
        self.should_exit = False
        
        # 批量标注模式
        self.batch_annotation_mode = False  # 是否在批量标注模式
        self.batch_selection_mode = False  # 是否在批量选点模式
        self.trajectories = []  # 存储批量选点生成的轨迹
        self.current_trajectory_idx = 0  # 当前正在标注的轨迹索引
        self.trajectory_id_counter = 0  # trajectory_id 计数器
        
        # 批量渲染状态
        self.batch_render_thread = None  # 批量渲染线程
        self.render_status = {}  # {trajectory_id: {"status": "pending/rendering/completed/error", "render_dir": "...", "video_paths": {...}}}
        self.render_lock = threading.Lock()  # 渲染状态锁
        self.batch_render_active = False  # 批量渲染是否激活
        
        # 路径噪声采样配置
        self.sample_num_paths = getattr(args, 'num_samples', 5)  # 每组起终点采样路径数量
        self.sample_noise_level = getattr(args, 'noise_level', 0.3)  # 噪声强度 [0, 1]
        self.sample_noise_type = getattr(args, 'noise_type', 'waypoint')  # 噪声类型: 'waypoint' 或 'grid'
        self._sampling_active = False   # 采样线程是否正在运行（防止重复触发 / stdin 竞争）
        self._sampling_thread: Optional[threading.Thread] = None
        self._needs_display_update = False  # 后台线程请求主线程刷新 cv2 显示
        self._batch_render_requested = False  # 终端 'b' 命令请求主线程启动批量渲染
        self._stdin_lock = threading.Lock()  # 确保同一时刻只有一个线程读取 stdin
        
        # 加载已有的标注（如果存在）
        if self.enable_annotate:
            self.load_existing_annotations()
            self.load_trajectories()
            self.check_existing_renders()  # 检查已存在的渲染文件
            
        
    def load_existing_annotations(self):
        """加载已有的标注数据"""
        annotate_json_path = os.path.join(self.output_dir, "annotate_episodes.json")
        if os.path.exists(annotate_json_path):
            try:
                with open(annotate_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.annotate_episodes = data.get("episodes", [])
                self.current_episode_id = len(self.annotate_episodes)
                logger.info(f"已加载 {self.current_episode_id} 个已有标注")
            except Exception as e:
                logger.error(f"加载标注数据失败: {e}")
                self.annotate_episodes = []
                self.current_episode_id = 0
    
    def load_trajectories(self):
        """加载批量选点生成的轨迹数据"""
        trajectories_json_path = os.path.join(self.output_dir, "trajectories.json")
        if os.path.exists(trajectories_json_path):
            try:
                with open(trajectories_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.trajectories = data.get("trajectories", [])
                self.trajectory_id_counter = data.get("next_trajectory_id", len(self.trajectories))
                logger.info(f"已加载 {len(self.trajectories)} 条轨迹数据")
            except Exception as e:
                logger.info(f"加载轨迹数据失败: {e}")
                self.trajectories = []
                self.trajectory_id_counter = 0
    
    def check_existing_renders(self):
        """检查已存在的渲染文件并更新状态"""
        for trajectory in self.trajectories:
            trajectory_id = trajectory['trajectory_id']
            
            # 检查是否有保存的render_dir
            if 'render_dir' in trajectory:
                render_dir = trajectory['render_dir']
                if self.args.camera_type == 'pano':
                    rgb_video = os.path.join(render_dir, "rgb_pano_video.mp4")
                    depth_video = os.path.join(render_dir, "depth_pano_video.mp4")
                else:
                    rgb_video = os.path.join(render_dir, "rgb_video.mp4")
                    depth_video = os.path.join(render_dir, "depth_video.mp4")
                
                if os.path.exists(rgb_video):
                    with self.render_lock:
                        self.render_status[trajectory_id] = {
                            "status": "completed",
                            "render_dir": render_dir,
                            "video_paths": {
                                "rgb": rgb_video,
                                "depth": depth_video if os.path.exists(depth_video) else None
                            }
                        }
                    logger.info(f"  Trajectory {trajectory_id}: 已存在渲染文件")
                else:
                    with self.render_lock:
                        self.render_status[trajectory_id] = {"status": "pending"}
            else:
                with self.render_lock:
                    self.render_status[trajectory_id] = {"status": "pending"}
    
    def save_trajectories(self):
        """保存轨迹数据到JSON文件"""
        trajectories_json_path = os.path.join(self.output_dir, "trajectories.json")
        try:
            with open(trajectories_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "num_trajectories": len(self.trajectories),
                    "next_trajectory_id": self.trajectory_id_counter,
                    "trajectories": self.trajectories
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 已保存 {len(self.trajectories)} 条轨迹到: {trajectories_json_path}")
        except Exception as e:
            logger.info(f"❌ 保存轨迹数据失败: {e}")
    
    def save_current_path_as_trajectory(self):
        """将当前路径保存为一条轨迹（不包含instruction）"""
        if not self.start_set or not self.goal_set:
            logger.info("❌ 错误: 起点或终点未设置")
            return False
        
        # if not self.path_found or not self.current_path:
        #     logger.info("❌ 错误: 请先完成路径规划")
        #     return False
        
        # 转换起点和终点为世界坐标
        start_world = self.grid_to_world(self.start_point[0], self.start_point[1], 0.0)
        goal_world = self.grid_to_world(self.goal_point[0], self.goal_point[1], 0.0)
        
        # 构建轨迹数据
        trajectory_data = {
            "trajectory_id": self.trajectory_id_counter,
            "start_position": start_world.tolist(),
            "start_yaw": float(self.start_yaw),
            "goal_position": goal_world.tolist(),
            "goal_yaw": float(self.goal_yaw),
            "scene_bounds": {
                "min": self.min_pt.tolist(),
                "max": self.max_pt.tolist()
            },
            "grid_resolution": float(self.grid_resolution),
            "annotated": False,  # 标记是否已标注
        }
        
        # 添加途径点（用户手动设置的 waypoints）
        if self.waypoints:
            waypoints_world = []
            for wp in self.waypoints:
                wp_world = self.grid_to_world(wp[0], wp[1], 0.0)
                waypoints_world.append(wp_world.tolist())
            trajectory_data["waypoints"] = waypoints_world

        # 从当前规划路径中提取路径 waypoints
        if self.path_found and self.current_path:
            trajectory_data["path_waypoints"] = self._extract_path_waypoints(self.current_path)
        
        self.trajectories.append(trajectory_data)
        trajectory_id = trajectory_data['trajectory_id']
        self.trajectory_id_counter += 1
        
        # 初始化新轨迹的渲染状态
        with self.render_lock:
            self.render_status[trajectory_id] = {"status": "pending"}
        
        logger.info(f"✓ 已添加轨迹 trajectory_{trajectory_data['trajectory_id']:04d}")
        return True
    
    def sample_paths_with_noise(
        self,
        start_point: Tuple[int, int],
        goal_point: Tuple[int, int],
        waypoints: List[Tuple[int, int]],
        start_yaw: float,
        goal_yaw: float,
        num_samples: int,
        noise_level: float = 0.3,
        noise_type: str = 'waypoint',
    ) -> List[dict]:
        """在起点-中间点-终点之间按路径噪声采样出多条路径。

        噪声策略（快速版）
        ------------------
        核心思路：先规划一次原始最短路 base_path，然后对 base_path 上的
        关键节点（起点附近、终点附近、中间锚点）施加随机偏移，再做一次
        单段 A* 规划，避免重复多段规划带来的性能开销。

        waypoint (默认)
            对 base_path 上均匀采样的若干锚点施加随机偏移，
            同时对起点/终点也施加小幅扰动，然后一次性规划整段路径。

        grid
            与 waypoint 相同策略，但锚点数量更多、偏移幅度更大。

        Args:
            start_point: 起点网格坐标 (gx, gy)
            goal_point:  终点网格坐标 (gx, gy)
            waypoints:   途径点列表（网格坐标）
            start_yaw:   起点朝向（弧度）
            goal_yaw:    终点朝向（弧度）
            num_samples: 采样路径数量（含原始最短路）
            noise_level: 噪声强度 [0, 1]，0 = 无噪声，1 = 最大扰动
            noise_type:  噪声类型 'waypoint' 或 'grid'

        Returns:
            list of dict，每个 dict 包含：
              - 'path': List[Tuple[int,int]]  完整网格路径
              - 'waypoints_grid': 实际使用的途径点（含扰动点）
              - 'start_yaw', 'goal_yaw'
              - 'sample_idx': int
        """
        planner = self._get_path_planner()
        results = []

        def _plan_full_path(seq: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """按顺序规划多段路径并拼接。"""
            full: List[Tuple[int, int]] = []
            for i in range(len(seq) - 1):
                seg = planner.plan(seq[i], seq[i + 1])["result"]
                if not seg:
                    return []
                full = full + (seg if not full else seg[1:])
            return full

        def _is_traversable(gx: int, gy: int) -> bool:
            """检查网格点是否在可通行区域内。"""
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return False
            return self.expanded_traversability[gy, gx] != 0

        def _clamp_to_traversable(gx: int, gy: int, max_search: int = 15) -> Optional[Tuple[int, int]]:
            """将网格点移动到最近的可通行位置（BFS，限制搜索范围以加速）。"""
            if _is_traversable(gx, gy):
                return (gx, gy)
            from collections import deque
            visited = set()
            q = deque([(gx, gy)])
            visited.add((gx, gy))
            limit = max_search * max_search
            count = 0
            while q and count < limit:
                count += 1
                cx, cy = q.popleft()
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        if _is_traversable(nx, ny):
                            return (nx, ny)
                        q.append((nx, ny))
            return None

        def _perturb_point(pt: Tuple[int, int], max_offset: int, rng) -> Optional[Tuple[int, int]]:
            """对一个点施加随机偏移并 clamp 到可通行区域。"""
            ox = int(rng.integers(-max_offset, max_offset + 1))
            oy = int(rng.integers(-max_offset, max_offset + 1))
            return _clamp_to_traversable(pt[0] + ox, pt[1] + oy)

        # ── 0. 先规划原始最短路（sample_idx=0）────────────────────────────
        base_seq = [start_point] + waypoints + [goal_point]
        base_path = _plan_full_path(base_seq)
        if not base_path:
            logger.info("❌ 无法规划原始最短路，采样中止")
            return []

        results.append({
            'path': base_path,
            'waypoints_grid': list(waypoints),
            'start_yaw': start_yaw,
            'goal_yaw': goal_yaw,
            'sample_idx': 0,
        })

        if num_samples <= 1:
            return results

        # ── 1. 预计算扰动参数（只做一次）────────────────────────────────
        total_len = len(base_path)
        # 起点/终点最大偏移：路径总长的 noise_level * 10%，最少 2 格
        endpoint_offset = max(2, int(noise_level * total_len * 0.10))
        # 中间锚点最大偏移：路径总长的 noise_level * 25%，最少 3 格
        mid_offset = max(3, int(noise_level * total_len * 0.25))
        # 锚点数量：waypoint 模式 2~4 个，grid 模式 3~6 个
        if noise_type == 'grid':
            n_anchors = max(3, int(noise_level * 6) + 2)
        else:
            n_anchors = max(2, int(noise_level * 4) + 1)

        # 在 base_path 上均匀取锚点索引（排除首尾）
        anchor_indices = np.linspace(0, total_len - 1, n_anchors + 2, dtype=int)[1:-1]

        rng = np.random.default_rng()
        attempts = 0
        max_attempts = num_samples * 8

        while len(results) < num_samples and attempts < max_attempts:
            attempts += 1

            # ── 扰动起点（小幅偏移）──────────────────────────────────────
            noisy_start = _perturb_point(start_point, endpoint_offset, rng)
            if noisy_start is None:
                noisy_start = start_point

            # ── 扰动终点（小幅偏移）──────────────────────────────────────
            noisy_goal = _perturb_point(goal_point, endpoint_offset, rng)
            if noisy_goal is None:
                noisy_goal = goal_point

            # ── 扰动中间锚点（较大偏移）──────────────────────────────────
            noisy_seq = [noisy_start]
            noisy_mid_pts = []
            for idx in anchor_indices:
                pt = base_path[idx]
                clamped = _perturb_point(pt, mid_offset, rng)
                if clamped is not None:
                    noisy_seq.append(clamped)
                    noisy_mid_pts.append(clamped)
            noisy_seq.append(noisy_goal)

            # ── 一次性规划带噪声的完整路径 ───────────────────────────────
            noisy_path = _plan_full_path(noisy_seq)
            if not noisy_path or len(noisy_path) < 2:
                continue

            # ── 去重：路径长度差异 + 中间点位置差异 ──────────────────────
            is_duplicate = False
            mid_i = len(noisy_path) // 2
            pt_a = noisy_path[mid_i]
            for prev in results:
                prev_path = prev['path']
                len_diff = abs(len(noisy_path) - len(prev_path)) / max(len(prev_path), 1)
                if len_diff < 0.05:
                    mid_j = len(prev_path) // 2
                    pt_b = prev_path[mid_j]
                    if np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]) < 5:
                        is_duplicate = True
                        break
            if is_duplicate:
                continue

            results.append({
                'path': noisy_path,
                'waypoints_grid': noisy_mid_pts,
                'start_yaw': start_yaw,
                'goal_yaw': goal_yaw,
                'sample_idx': len(results),
            })

        logger.info(f"✓ 路径采样完成: 成功 {len(results)}/{num_samples} 条（尝试 {attempts} 次）")
        return results

    def sample_and_save_trajectories(
        self,
        num_samples: Optional[int] = None,
        noise_level: Optional[float] = None,
        noise_type: Optional[str] = None,
    ) -> int:
        """基于当前起点、途径点、终点，采样多条带噪声路径并批量保存为轨迹。

        Args:
            num_samples: 采样数量，None 则使用 self.sample_num_paths
            noise_level: 噪声强度，None 则使用 self.sample_noise_level
            noise_type:  噪声类型，None 则使用 self.sample_noise_type

        Returns:
            实际保存的轨迹数量
        """
        if not self.start_set or not self.goal_set:
            logger.info("❌ 错误: 起点或终点未设置，无法采样")
            return 0

        num_samples  = num_samples  if num_samples  is not None else self.sample_num_paths
        noise_level  = noise_level  if noise_level  is not None else self.sample_noise_level
        noise_type   = noise_type   if noise_type   is not None else self.sample_noise_type

        logger.info(f"\n{'='*60}")
        logger.info(f"路径噪声采样")
        logger.info(f"  起点: {self.start_point}  终点: {self.goal_point}")
        logger.info(f"  途径点数量: {len(self.waypoints)}")
        logger.info(f"  采样数量: {num_samples}  噪声强度: {noise_level:.2f}  噪声类型: {noise_type}")
        logger.info(f"{'='*60}")

        sampled = self.sample_paths_with_noise(
            start_point=self.start_point,
            goal_point=self.goal_point,
            waypoints=list(self.waypoints),
            start_yaw=self.start_yaw,
            goal_yaw=self.goal_yaw,
            num_samples=num_samples,
            noise_level=noise_level,
            noise_type=noise_type,
        )

        if not sampled:
            logger.info("❌ 采样失败，未生成任何路径")
            return 0

        saved_count = 0
        for sample in sampled:
            path = sample['path']
            noisy_waypoints = sample['waypoints_grid']

            # 转换起点、终点为世界坐标
            start_world = self.grid_to_world(self.start_point[0], self.start_point[1], 0.0)
            goal_world  = self.grid_to_world(self.goal_point[0],  self.goal_point[1],  0.0)

            # 构建轨迹数据
            trajectory_data: dict = {
                "trajectory_id":  self.trajectory_id_counter,
                "sample_idx":     sample['sample_idx'],
                "start_position": start_world.tolist(),
                "start_yaw":      float(self.start_yaw),
                "goal_position":  goal_world.tolist(),
                "goal_yaw":       float(self.goal_yaw),
                "scene_bounds": {
                    "min": self.min_pt.tolist(),
                    "max": self.max_pt.tolist()
                },
                "grid_resolution": float(self.grid_resolution),
                "annotated":       False,
                "noise_level":     float(noise_level),
                "noise_type":      noise_type,
            }

            # 保存噪声途径点（世界坐标）
            if noisy_waypoints:
                wps_world = []
                for wp in noisy_waypoints:
                    wp_world = self.grid_to_world(wp[0], wp[1], 0.0)
                    wps_world.append(wp_world.tolist())
                trajectory_data["waypoints"] = wps_world

            # 从完整路径中提取路径 waypoints
            trajectory_data["path_waypoints"] = self._extract_path_waypoints(path)

            self.trajectories.append(trajectory_data)
            trajectory_id = trajectory_data['trajectory_id']
            self.trajectory_id_counter += 1

            # 初始化渲染状态
            with self.render_lock:
                self.render_status[trajectory_id] = {"status": "pending"}

            saved_count += 1
            logger.info(
                f"  ✓ 保存 trajectory_{trajectory_id:04d}  "
                f"(sample_idx={sample['sample_idx']}, path_len={len(path)}, "
                f"path_waypoints={len(trajectory_data['path_waypoints'])})"
            )

        # 持久化
        self.save_trajectories()

        logger.info(f"\n✓ 共保存 {saved_count} 条采样轨迹，当前总计 {len(self.trajectories)} 条")
        logger.info("提示: 按 'B' 键（在地图窗口中）或终端输入 'b' 开始批量渲染和标注")

        # 通知主线程刷新显示（cv2.imshow 必须在主线程调用，不能在后台线程中直接调用）
        self._needs_display_update = True
        return saved_count

    # ──────────────────────────────────────────────────────────────────────────
    # 随机起点/终点/waypoint 采样功能
    # ──────────────────────────────────────────────────────────────────────────

    def _get_random_traversable_points(self, n: int, min_dist_m: float = 3.0,
                                        exclude_grid: Optional[Tuple[int, int]] = None,
                                        exclude_radius_m: float = 1.0) -> List[Tuple[int, int]]:
        """在可通行区域中随机采样 n 个网格点。

        Args:
            n: 需要采样的点数量
            min_dist_m: 采样点之间的最小距离（米）
            exclude_grid: 要排除的网格点（例如起点/终点），在其附近不采样
            exclude_radius_m: 排除半径（米）

        Returns:
            List of (gx, gy) grid coordinates
        """
        if self.expanded_traversability is None:
            return []

        min_dist_cells = max(1, int(min_dist_m / self.grid_resolution))
        exclude_radius_cells = max(1, int(exclude_radius_m / self.grid_resolution))

        # 获取所有可通行像素坐标
        ys, xs = np.where(self.expanded_traversability != 0)
        if len(xs) == 0:
            return []

        all_pts = list(zip(xs.tolist(), ys.tolist()))
        rng = np.random.default_rng()
        rng.shuffle(all_pts)

        selected: List[Tuple[int, int]] = []
        for pt in all_pts:
            gx, gy = pt
            # 排除 exclude_grid 附近
            if exclude_grid is not None:
                dist_ex = np.hypot(gx - exclude_grid[0], gy - exclude_grid[1])
                if dist_ex < exclude_radius_cells:
                    continue
            # 与已选点保持最小距离
            too_close = False
            for sel in selected:
                if np.hypot(gx - sel[0], gy - sel[1]) < min_dist_cells:
                    too_close = True
                    break
            if not too_close:
                selected.append((gx, gy))
            if len(selected) >= n:
                break

        return selected

    def sample_random_goals_from_start(self, num_goals: int = 5, min_dist_m: float = 3.0) -> int:
        """固定当前起点，随机采样多个终点，每个终点生成一条轨迹并保存。

        快捷键 'F' 触发。

        Args:
            num_goals: 随机终点数量
            min_dist_m: 终点与起点的最小距离（米）

        Returns:
            实际保存的轨迹数量
        """
        if not self.start_set:
            logger.info("❌ 请先设置起点（按 S 键后点击地图）")
            return 0

        logger.info(f"\n{'='*60}")
        logger.info(f"随机终点采样（固定起点）")
        logger.info(f"  起点: {self.start_point}")
        logger.info(f"  目标终点数量: {num_goals}  最小距离: {min_dist_m:.1f}m")
        logger.info(f"{'='*60}")

        goal_candidates = self._get_random_traversable_points(
            n=num_goals,
            min_dist_m=min_dist_m,
            exclude_grid=self.start_point,
            exclude_radius_m=min_dist_m,
        )

        if not goal_candidates:
            logger.info("❌ 未能在可通行区域中采样到有效终点")
            return 0

        planner = self._get_path_planner()
        saved_count = 0
        start_world = self.grid_to_world(self.start_point[0], self.start_point[1], 0.0)
        rng = np.random.default_rng()

        for goal_pt in goal_candidates:
            # 规划路径，验证可达性
            result = planner.plan(self.start_point, goal_pt)
            path = result.get("result", [])
            if not path or len(path) < 2:
                logger.info(f"  ⚠️ 终点 {goal_pt} 不可达，跳过")
                continue

            goal_world = self.grid_to_world(goal_pt[0], goal_pt[1], 0.0)
            goal_yaw = float(rng.uniform(-np.pi, np.pi))

            # 从路径中均匀提取 waypoints（路径中间节点）
            path_waypoints_world = self._extract_path_waypoints(path)

            trajectory_data: dict = {
                "trajectory_id":  self.trajectory_id_counter,
                "sample_idx":     saved_count,
                "sample_mode":    "random_goals_from_start",
                "start_position": start_world.tolist(),
                "start_yaw":      float(self.start_yaw),
                "goal_position":  goal_world.tolist(),
                "goal_yaw":       goal_yaw,
                "waypoints":      path_waypoints_world,
                "scene_bounds": {
                    "min": self.min_pt.tolist(),
                    "max": self.max_pt.tolist()
                },
                "grid_resolution": float(self.grid_resolution),
                "annotated":       False,
            }

            self.trajectories.append(trajectory_data)
            trajectory_id = trajectory_data['trajectory_id']
            self.trajectory_id_counter += 1

            with self.render_lock:
                self.render_status[trajectory_id] = {"status": "pending"}

            saved_count += 1
            logger.info(
                f"  ✓ trajectory_{trajectory_id:04d}  goal={goal_pt}  "
                f"path_len={len(path)}  waypoints={len(path_waypoints_world)}"
            )

        if saved_count > 0:
            self.save_trajectories()
            logger.info(f"\n✓ 共保存 {saved_count} 条轨迹（固定起点随机终点），当前总计 {len(self.trajectories)} 条")
            self._needs_display_update = True
        else:
            logger.info("❌ 未能保存任何轨迹")

        return saved_count

    def sample_random_starts_from_goal(self, num_starts: int = 5, min_dist_m: float = 3.0) -> int:
        """固定当前终点，随机采样多个起点，每个起点生成一条轨迹并保存。

        快捷键 'V' 触发。

        Args:
            num_starts: 随机起点数量
            min_dist_m: 起点与终点的最小距离（米）

        Returns:
            实际保存的轨迹数量
        """
        if not self.goal_set:
            logger.info("❌ 请先设置终点（按 G 键后点击地图）")
            return 0

        logger.info(f"\n{'='*60}")
        logger.info(f"随机起点采样（固定终点）")
        logger.info(f"  终点: {self.goal_point}")
        logger.info(f"  目标起点数量: {num_starts}  最小距离: {min_dist_m:.1f}m")
        logger.info(f"{'='*60}")

        start_candidates = self._get_random_traversable_points(
            n=num_starts,
            min_dist_m=min_dist_m,
            exclude_grid=self.goal_point,
            exclude_radius_m=min_dist_m,
        )

        if not start_candidates:
            logger.info("❌ 未能在可通行区域中采样到有效起点")
            return 0

        planner = self._get_path_planner()
        saved_count = 0
        goal_world = self.grid_to_world(self.goal_point[0], self.goal_point[1], 0.0)
        rng = np.random.default_rng()

        for start_pt in start_candidates:
            result = planner.plan(start_pt, self.goal_point)
            path = result.get("result", [])
            if not path or len(path) < 2:
                logger.info(f"  ⚠️ 起点 {start_pt} 不可达，跳过")
                continue

            start_world = self.grid_to_world(start_pt[0], start_pt[1], 0.0)
            start_yaw = float(rng.uniform(-np.pi, np.pi))

            path_waypoints_world = self._extract_path_waypoints(path)

            trajectory_data: dict = {
                "trajectory_id":  self.trajectory_id_counter,
                "sample_idx":     saved_count,
                "sample_mode":    "random_starts_from_goal",
                "start_position": start_world.tolist(),
                "start_yaw":      start_yaw,
                "goal_position":  goal_world.tolist(),
                "goal_yaw":       float(self.goal_yaw),
                "waypoints":      path_waypoints_world,
                "scene_bounds": {
                    "min": self.min_pt.tolist(),
                    "max": self.max_pt.tolist()
                },
                "grid_resolution": float(self.grid_resolution),
                "annotated":       False,
            }

            self.trajectories.append(trajectory_data)
            trajectory_id = trajectory_data['trajectory_id']
            self.trajectory_id_counter += 1

            with self.render_lock:
                self.render_status[trajectory_id] = {"status": "pending"}

            saved_count += 1
            logger.info(
                f"  ✓ trajectory_{trajectory_id:04d}  start={start_pt}  "
                f"path_len={len(path)}  waypoints={len(path_waypoints_world)}"
            )

        if saved_count > 0:
            self.save_trajectories()
            logger.info(f"\n✓ 共保存 {saved_count} 条轨迹（固定终点随机起点），当前总计 {len(self.trajectories)} 条")
            self._needs_display_update = True
        else:
            logger.info("❌ 未能保存任何轨迹")

        return saved_count

    def sample_waypoint_loop(self, num_waypoints: int = 4, loop_radius_m: float = 5.0,
                              return_radius_m: float = 2.0) -> int:
        """固定当前起点，随机采样 waypoints，路径最终返回起点附近，保存为一条轨迹。

        快捷键 'W' 触发。

        Args:
            num_waypoints: 随机 waypoint 数量
            loop_radius_m: waypoint 采样半径（米），在起点周围此范围内采样
            return_radius_m: 返回终点距起点的最大距离（米）

        Returns:
            实际保存的轨迹数量（0 或 1）
        """
        if not self.start_set:
            logger.info("❌ 请先设置起点（按 S 键后点击地图）")
            return 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Waypoint 环形路径采样（固定起点，返回起点附近）")
        logger.info(f"  起点: {self.start_point}")
        logger.info(f"  waypoint 数量: {num_waypoints}  采样半径: {loop_radius_m:.1f}m")
        logger.info(f"  返回半径: {return_radius_m:.1f}m")
        logger.info(f"{'='*60}")

        loop_radius_cells = max(1, int(loop_radius_m / self.grid_resolution))
        return_radius_cells = max(1, int(return_radius_m / self.grid_resolution))

        # 在起点周围 loop_radius_cells 范围内采样 waypoints
        if self.expanded_traversability is None:
            logger.info("❌ 可通行性地图未初始化")
            return 0

        sx, sy = self.start_point
        ys, xs = np.where(self.expanded_traversability != 0)
        rng = np.random.default_rng()

        # 筛选在环形区域内的可通行点（距起点 1m ~ loop_radius_m）
        min_inner_cells = max(1, int(1.0 / self.grid_resolution))
        mask = (
            (np.abs(xs - sx) <= loop_radius_cells) &
            (np.abs(ys - sy) <= loop_radius_cells)
        )
        dist_cells = np.hypot(xs[mask] - sx, ys[mask] - sy)
        valid_mask = (dist_cells >= min_inner_cells) & (dist_cells <= loop_radius_cells)
        cand_xs = xs[mask][valid_mask]
        cand_ys = ys[mask][valid_mask]

        if len(cand_xs) == 0:
            logger.info("❌ 起点附近没有足够的可通行区域用于 waypoint 采样")
            return 0

        # 随机打乱并选取 waypoints（保持相互间距）
        indices = rng.permutation(len(cand_xs))
        min_wp_dist_cells = max(2, int(1.5 / self.grid_resolution))
        selected_wps: List[Tuple[int, int]] = []
        for idx in indices:
            pt = (int(cand_xs[idx]), int(cand_ys[idx]))
            too_close = any(
                np.hypot(pt[0] - w[0], pt[1] - w[1]) < min_wp_dist_cells
                for w in selected_wps
            )
            if not too_close:
                selected_wps.append(pt)
            if len(selected_wps) >= num_waypoints:
                break

        if not selected_wps:
            logger.info("❌ 未能采样到有效 waypoints")
            return 0

        # 在起点附近采样一个"返回终点"
        return_candidates = [
            (int(cand_xs[i]), int(cand_ys[i]))
            for i in range(len(cand_xs))
            if np.hypot(cand_xs[i] - sx, cand_ys[i] - sy) <= return_radius_cells
            and (int(cand_xs[i]), int(cand_ys[i])) != self.start_point
        ]
        if not return_candidates:
            # fallback：直接用起点作为终点
            return_pt = self.start_point
        else:
            return_pt = return_candidates[rng.integers(0, len(return_candidates))]

        # 规划完整路径：start -> wp1 -> wp2 -> ... -> return_pt
        planner = self._get_path_planner()
        sequence = [self.start_point] + selected_wps + [return_pt]
        full_path: List[Tuple[int, int]] = []
        for i in range(len(sequence) - 1):
            seg = planner.plan(sequence[i], sequence[i + 1]).get("result", [])
            if not seg:
                logger.info(f"  ⚠️ 无法规划从 {sequence[i]} 到 {sequence[i+1]} 的路径，中止")
                return 0
            full_path = full_path + (seg if not full_path else seg[1:])

        if len(full_path) < 2:
            logger.info("❌ 路径过短，中止")
            return 0

        start_world = self.grid_to_world(sx, sy, 0.0)
        return_world = self.grid_to_world(return_pt[0], return_pt[1], 0.0)

        # waypoints 世界坐标
        wps_world = [
            self.grid_to_world(wp[0], wp[1], 0.0).tolist()
            for wp in selected_wps
        ]

        # 从完整路径中提取路径 waypoints
        path_waypoints_world = self._extract_path_waypoints(full_path)

        trajectory_data: dict = {
            "trajectory_id":  self.trajectory_id_counter,
            "sample_idx":     0,
            "sample_mode":    "waypoint_loop",
            "start_position": start_world.tolist(),
            "start_yaw":      float(self.start_yaw),
            "goal_position":  return_world.tolist(),
            "goal_yaw":       float(self.start_yaw),  # 返回时朝向与出发时相同
            "waypoints":      wps_world,
            "path_waypoints": path_waypoints_world,
            "loop_radius_m":  loop_radius_m,
            "return_radius_m": return_radius_m,
            "scene_bounds": {
                "min": self.min_pt.tolist(),
                "max": self.max_pt.tolist()
            },
            "grid_resolution": float(self.grid_resolution),
            "annotated":       False,
        }

        self.trajectories.append(trajectory_data)
        trajectory_id = trajectory_data['trajectory_id']
        self.trajectory_id_counter += 1

        with self.render_lock:
            self.render_status[trajectory_id] = {"status": "pending"}

        self.save_trajectories()
        logger.info(
            f"\n✓ 已保存 waypoint 环形轨迹 trajectory_{trajectory_id:04d}  "
            f"waypoints={len(selected_wps)}  path_len={len(full_path)}  "
            f"当前总计 {len(self.trajectories)} 条"
        )
        self._needs_display_update = True
        return 1

    def _extract_path_waypoints(self, path: List[Tuple[int, int]],
                                 num_waypoints: int = 5) -> List[list]:
        """从完整路径中均匀提取中间 waypoints（世界坐标），不含起点和终点。

        Args:
            path: 网格路径
            num_waypoints: 提取的 waypoint 数量

        Returns:
            List of [x, y, z] world coordinates
        """
        if len(path) <= 2:
            return []
        # 均匀采样中间节点（排除首尾）
        inner = path[1:-1]
        if not inner:
            return []
        n = min(num_waypoints, len(inner))
        indices = np.linspace(0, len(inner) - 1, n, dtype=int)
        result = []
        seen = set()
        for idx in indices:
            pt = inner[idx]
            if pt not in seen:
                seen.add(pt)
                result.append(self.grid_to_world(pt[0], pt[1], 0.0).tolist())
        return result

    def show_random_sample_config_dialog(self, mode: str) -> Optional[dict]:
        """在终端中交互式配置随机采样参数。

        Args:
            mode: 'goals' | 'starts' | 'waypoint_loop'

        Returns:
            dict with relevant keys, or None if cancelled
        """
        logger.info("\n" + "="*60)
        if mode == 'goals':
            logger.info("【随机终点采样配置】（固定起点）")
        elif mode == 'starts':
            logger.info("【随机起点采样配置】（固定终点）")
        else:
            logger.info("【Waypoint 环形路径采样配置】")
        logger.info("="*60)
        logger.info("直接按 Enter 使用默认值")

        with self._stdin_lock:
            try:
                if mode in ('goals', 'starts'):
                    label = "终点" if mode == 'goals' else "起点"
                    raw = input(f"  随机{label}数量 [5]: ").strip()
                    num_pts = int(raw) if raw else 5
                    num_pts = max(1, min(num_pts, 50))

                    raw = input(f"  与固定点的最小距离（米）[3.0]: ").strip()
                    min_dist = float(raw) if raw else 3.0
                    min_dist = max(0.5, min_dist)

                    confirm = input(f"  确认? (y/n) [y]: ").strip().lower()
                    if confirm == 'n':
                        logger.info("❌ 已取消")
                        return None
                    return {'num_pts': num_pts, 'min_dist': min_dist}

                else:  # waypoint_loop
                    raw = input(f"  Waypoint 数量 [4]: ").strip()
                    num_wps = int(raw) if raw else 4
                    num_wps = max(1, min(num_wps, 20))

                    raw = input(f"  采样半径（米）[5.0]: ").strip()
                    loop_radius = float(raw) if raw else 5.0
                    loop_radius = max(1.0, loop_radius)

                    raw = input(f"  返回终点距起点最大距离（米）[2.0]: ").strip()
                    return_radius = float(raw) if raw else 2.0
                    return_radius = max(0.5, return_radius)

                    confirm = input(f"  确认? (y/n) [y]: ").strip().lower()
                    if confirm == 'n':
                        logger.info("❌ 已取消")
                        return None
                    return {
                        'num_waypoints': num_wps,
                        'loop_radius_m': loop_radius,
                        'return_radius_m': return_radius,
                    }

            except (ValueError, EOFError) as e:
                logger.info(f"❌ 输入无效: {e}，已取消")
                return None

    def show_sample_config_dialog(self) -> Optional[dict]:
        """在终端中交互式配置采样参数。

        Returns:
            dict with keys: num_samples, noise_level, noise_type
            or None if cancelled
        """
        logger.info("\n" + "="*60)
        logger.info("【路径噪声采样配置】")
        logger.info("="*60)
        logger.info(f"当前配置:")
        logger.info(f"  采样数量 (num_samples): {self.sample_num_paths}")
        logger.info(f"  噪声强度 (noise_level): {self.sample_noise_level:.2f}  [0.0~1.0]")
        logger.info(f"  噪声类型 (noise_type):  {self.sample_noise_type}  [waypoint/grid]")
        logger.info("-"*60)
        logger.info("直接按 Enter 使用当前值，输入新值后按 Enter 修改")

        # 使用 stdin_lock 确保同一时刻只有一个线程读取 stdin（防止与终端监听线程竞争）
        with self._stdin_lock:
            try:
                # 采样数量
                raw = input(f"  采样数量 [{self.sample_num_paths}]: ").strip()
                num_samples = int(raw) if raw else self.sample_num_paths
                num_samples = max(1, min(num_samples, 50))

                # 噪声强度
                raw = input(f"  噪声强度 [{self.sample_noise_level:.2f}]: ").strip()
                noise_level = float(raw) if raw else self.sample_noise_level
                noise_level = float(np.clip(noise_level, 0.0, 1.0))

                # 噪声类型
                raw = input(f"  噪声类型 [{self.sample_noise_type}] (waypoint/grid): ").strip().lower()
                noise_type = raw if raw in ('waypoint', 'grid') else self.sample_noise_type

                logger.info(f"\n确认配置: num_samples={num_samples}, "
                            f"noise_level={noise_level:.2f}, noise_type={noise_type}")
                confirm = input("确认? (y/n) [y]: ").strip().lower()
                if confirm == 'n':
                    logger.info("❌ 已取消")
                    return None

                # 更新实例配置
                self.sample_num_paths  = num_samples
                self.sample_noise_level = noise_level
                self.sample_noise_type  = noise_type

                return {
                    'num_samples': num_samples,
                    'noise_level': noise_level,
                    'noise_type':  noise_type,
                }

            except (ValueError, EOFError) as e:
                logger.info(f"❌ 输入无效: {e}，已取消")
                return None

    def start_batch_rendering(self, max_workers=3):
        """启动批量渲染所有trajectories
        
        Args:
            max_workers: 最大并行渲染线程数，默认为3
        """
        if self.batch_render_active:
            logger.info("⚠️ 批量渲染已在运行中")
            return
        
        # 统计需要渲染的轨迹
        pending_trajectories = []
        for trajectory in self.trajectories:
            trajectory_id = trajectory['trajectory_id']
            
            with self.render_lock:
                # 如果轨迹还没有渲染状态，初始化为pending
                if trajectory_id not in self.render_status:
                    self.render_status[trajectory_id] = {"status": "pending"}
                status = self.render_status.get(trajectory_id, {}).get("status", "pending")
            if status == "pending":
                pending_trajectories.append(trajectory)
        
        if not pending_trajectories:
            logger.info("✓ 所有轨迹已渲染完成")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始批量渲染 {len(pending_trajectories)} 条轨迹 (并行线程数: {max_workers})")
        logger.info(f"{'='*60}")
        
        self.batch_render_active = True
        # 使用线程来运行线程池，避免阻塞主线程
        self.batch_render_thread = threading.Thread(
            target=self._batch_render_coordinator,
            args=(pending_trajectories, max_workers),
            daemon=True
        )
        self.batch_render_thread.start()
    
    def _batch_render_coordinator(self, trajectories_to_render, max_workers):
        """批量渲染协调器，使用线程池并行渲染多个轨迹"""
        total = len(trajectories_to_render)
        completed = 0
        failed = 0
        
        # 使用线程池并行渲染
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有渲染任务
            future_to_trajectory = {
                executor.submit(self._render_single_trajectory, trajectory): trajectory
                for trajectory in trajectories_to_render
            }
            
            # 等待任务完成
            for future in as_completed(future_to_trajectory):
                trajectory = future_to_trajectory[future]
                trajectory_id = trajectory['trajectory_id']
                
                try:
                    success = future.result()
                    if success:
                        completed += 1
                        logger.info(f"✓ [{completed+failed}/{total}] Trajectory {trajectory_id} 渲染完成")
                    else:
                        failed += 1
                        logger.info(f"❌ [{completed+failed}/{total}] Trajectory {trajectory_id} 渲染失败")
                except Exception as e:
                    failed += 1
                    logger.info(f"❌ [{completed+failed}/{total}] Trajectory {trajectory_id} 渲染异常: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"批量渲染完成: 成功 {completed}/{total}, 失败 {failed}/{total}")
        logger.info(f"{'='*60}")
        self.batch_render_active = False
    
    def _render_single_trajectory(self, trajectory):
        """渲染单个轨迹（线程安全）
        
        Args:
            trajectory: 轨迹数据字典
            
        Returns:
            bool: 渲染成功返回True，失败返回False
        """
        trajectory_id = trajectory['trajectory_id']
        
        try:
            logger.info(f"\n开始渲染 Trajectory {trajectory_id}...")
            
            # 更新状态为rendering
            with self.render_lock:
                self.render_status[trajectory_id] = {"status": "rendering"}
            
            # 恢复轨迹的起点、终点和途径点（使用局部变量）
            start_world = np.array(trajectory["start_position"])
            goal_world = np.array(trajectory["goal_position"])
            
            # 转换回网格坐标
            start_grid_x = int((start_world[0] - self.min_pt[0]) / self.grid_resolution)
            start_grid_y = int((self.max_pt[1] - start_world[1]) / self.grid_resolution)
            goal_grid_x = int((goal_world[0] - self.min_pt[0]) / self.grid_resolution)
            goal_grid_y = int((self.max_pt[1] - goal_world[1]) / self.grid_resolution)
            
            start_point = (start_grid_x, start_grid_y)
            goal_point = (goal_grid_x, goal_grid_y)
            start_yaw = trajectory["start_yaw"]
            goal_yaw = trajectory["goal_yaw"]
            
            # 恢复途径点
            waypoints = []
            if "waypoints" in trajectory:
                for wp_world in trajectory["waypoints"]:
                    wp_world = np.array(wp_world)
                    wp_grid_x = int((wp_world[0] - self.min_pt[0]) / self.grid_resolution)
                    wp_grid_y = int((self.max_pt[1] - wp_world[1]) / self.grid_resolution)
                    waypoints.append((wp_grid_x, wp_grid_y))
            
            # 运行A*规划（使用局部变量）
            waypoint_sequence = [start_point] + waypoints + [goal_point]
            full_path = []
            
            for i in range(len(waypoint_sequence) - 1):
                start = waypoint_sequence[i]
                goal = waypoint_sequence[i + 1]
                
                segment_path = self._get_path_planner().plan(
                    (start[0], start[1]), (goal[0], goal[1]))["result"]
                
                if not segment_path:
                    raise Exception(f"无法找到从 {start} 到 {goal} 的路径")
                
                if full_path:
                    full_path.extend(segment_path[1:])
                else:
                    full_path.extend(segment_path)
            
            # 生成相机位姿（线程安全，不修改实例变量）
            rotations, translations, camera_poses_grid, camera_poses_directions, actions = \
                self.generate_camera_poses(full_path, mode="discrete", 
                                          start_yaw=start_yaw, goal_yaw=goal_yaw)
            
            if rotations is None:
                raise Exception("生成相机位姿失败")
            
            # 持久化相机 JSON（勿删除，供标注与 render_status 引用；与 main_sample 字段一致）
            traj_camera_json = os.path.join(
                self.output_dir, f"camera_poses_traj_{trajectory_id:04d}.json")
            self.save_camera_poses_json(
                rotations, translations, len(rotations),
                actions=actions, mode="discrete",
                output_path=traj_camera_json,
                full_path=full_path,
                start_yaw=start_yaw,
                goal_yaw=goal_yaw,
            )
            
            # 读取相机位姿
            with open(traj_camera_json, 'r') as f:
                camera_data = json.load(f)
            
            camera_poses = camera_data.get('cameras', [])
            if not camera_poses:
                raise Exception("JSON文件中没有相机位姿")
            
            # 创建渲染输出目录（使用trajectory_id确保唯一性）
            render_dir = os.path.join(self.output_dir, f"render_trajectory_{trajectory_id:04d}")
            os.makedirs(render_dir, exist_ok=True)
            
            # 创建线程专属的RenderManager实例（避免共享资源冲突）
            from managers.render_manager import RenderManager
            render_manager = RenderManager(
                ply_path=self.ply_path,
                sh_degree=3,
                background_color=[0, 0, 0]
            )
            
            # 根据相机类型选择渲染方法
            if self.camera_type == 'pano':
                # 全景视角渲染
                logger.info(f"[Trajectory {trajectory_id}] 使用全景视角渲染...")
                success = render_manager.render_panorama_trajectory(
                    camera_poses=camera_poses,
                    output_dir=render_dir,
                    create_video=True,
                    video_fps=10,
                    pano_width=2048,
                    pano_height=1024,
                    save_cube_faces=False,
                    use_multithreading=True,
                    max_workers=6,
                    apply_diffuser=False,
                    diffuser_strength=0.3,
                    enable_depth=self.args.enable_depth
                )
            else:
                # 单视角渲染
                logger.info(f"[Trajectory {trajectory_id}] 使用单视角渲染...")
                success = render_manager.render_from_json(
                    camera_json_path=traj_camera_json,
                    output_dir=render_dir,
                    create_video=True,
                    video_fps=10,
                    enable_depth=self.args.enable_depth
                )
            
            if not success:
                raise Exception("渲染失败")
            
            # 更新轨迹数据
            trajectory['render_dir'] = render_dir
            with self.render_lock:
                self.save_trajectories()
            
            # 更新渲染状态
            if self.args.camera_type == 'pano':
                rgb_video = os.path.join(render_dir, "rgb_pano_video.mp4")
                depth_video = os.path.join(render_dir, "depth_pano_video.mp4")
            else:
                rgb_video = os.path.join(render_dir, "rgb_video.mp4")
                depth_video = os.path.join(render_dir, "depth_video.mp4")
            
            with self.render_lock:
                self.render_status[trajectory_id] = {
                    "status": "completed",
                    "render_dir": render_dir,
                    "camera_json": traj_camera_json,
                    "video_paths": {
                        "rgb": rgb_video,
                        "depth": depth_video if os.path.exists(depth_video) else None
                    }
                }
            
            return True
            
        except Exception as e:
            logger.info(f"❌ Trajectory {trajectory_id} 渲染失败: {e}")
            import traceback
            traceback.print_exc()
            
            with self.render_lock:
                self.render_status[trajectory_id] = {
                    "status": "error",
                    "error": str(e)
                }
            return False
    
    def start_batch_annotation(self):
        """开始批量标注模式"""
        if not self.trajectories:
            logger.info("❌ 没有可标注的轨迹")
            return
        
        # 主循环：处理视频切换
        while self.batch_annotation_mode:
            if self.current_trajectory_idx >= len(self.trajectories):
                logger.info("✓ 所有轨迹已标注完成")
                self.batch_annotation_mode = False
                return
            
            # 重置切换标志
            self.request_video_switch = False
            
            # 加载当前轨迹
            trajectory = self.trajectories[self.current_trajectory_idx]
            trajectory_id = trajectory['trajectory_id']
            
            # 如果当前轨迹已标注，自动跳到下一个未标注的
            # if trajectory.get("annotated", False):
            #     logger.info(f"⚠️ 轨迹 {trajectory_id} 已标注，自动跳到下一个...")
            #     self.current_trajectory_idx += 1
            #     continue  # 继续下一次循环
            
            logger.info(f"\n{'='*60}")
            logger.info(f"加载轨迹 {self.current_trajectory_idx + 1}/{len(self.trajectories)}")
            logger.info(f"Trajectory ID: {trajectory_id}")
            logger.info(f"{'='*60}")
            
            # 检查渲染状态
            with self.render_lock:
                # 如果轨迹还没有渲染状态，初始化为pending
                if trajectory_id not in self.render_status:
                    self.render_status[trajectory_id] = {"status": "pending"}
                render_info = self.render_status.get(trajectory_id, {"status": "pending"})
                status = render_info.get("status", "pending")
            
            if status == "completed":
                # 视频已渲染完成，直接加载
                logger.info(f"✓ 检测到已渲染的视频，直接加载...")
                render_dir = render_info["render_dir"]
                rgb_video = render_info["video_paths"]["rgb"]
                depth_video = render_info["video_paths"].get("depth")
                
                # 恢复轨迹参数用于显示地图
                self._restore_trajectory_params(trajectory)
                
                # 重新规划路径（用于地图可视化）
                self._replan_trajectory_path(trajectory)
                
                # 直接加载已存在的视频
                self.last_render_dir = render_dir
                self.last_saved_mode = "discrete"
                self.load_and_display_video(rgb_video, depth_video, for_batch=True)
                
                # 视频界面退出后，检查是否需要切换视频
                if not self.request_video_switch:
                    # 如果不需要切换，说明用户退出了批量模式
                    break
                # 否则继续循环，加载下一个视频
                
            elif status == "rendering":
                # 正在渲染中，等待完成
                logger.info(f"⚠️ 当前 Trajectory {trajectory_id} 视频还在生成中，请稍候...")
                self._wait_for_rendering(trajectory_id, trajectory)
                
            elif status == "error":
                # 渲染失败，直接跳到下一条
                error_msg = render_info.get("error", "未知错误")
                logger.info(f"❌ Trajectory {trajectory_id} 渲染失败: {error_msg}")
                self.current_trajectory_idx += 1
                continue  # 继续下一次循环
                
            else:  # pending
                # 尚未开始渲染，等待批量渲染线程处理
                logger.info(f"⚠️ Trajectory {trajectory_id} 还未开始渲染...")
                logger.info("等待批量渲染线程处理...")
                # 等待渲染开始或完成
                self._wait_for_rendering(trajectory_id, trajectory)
    
    def _restore_trajectory_params(self, trajectory):
        """恢复轨迹参数"""
        start_world = np.array(trajectory["start_position"])
        goal_world = np.array(trajectory["goal_position"])
        
        # 转换回网格坐标
        start_grid_x = int((start_world[0] - self.min_pt[0]) / self.grid_resolution)
        start_grid_y = int((self.max_pt[1] - start_world[1]) / self.grid_resolution)
        goal_grid_x = int((goal_world[0] - self.min_pt[0]) / self.grid_resolution)
        goal_grid_y = int((self.max_pt[1] - goal_world[1]) / self.grid_resolution)
        
        self.start_point = (start_grid_x, start_grid_y)
        self.goal_point = (goal_grid_x, goal_grid_y)
        self.start_yaw = trajectory["start_yaw"]
        self.goal_yaw = trajectory["goal_yaw"]
        self.start_set = True
        self.goal_set = True
        
        # 恢复途径点
        if "waypoints" in trajectory:
            self.waypoints = []
            for wp_world in trajectory["waypoints"]:
                wp_world = np.array(wp_world)
                wp_grid_x = int((wp_world[0] - self.min_pt[0]) / self.grid_resolution)
                wp_grid_y = int((self.max_pt[1] - wp_world[1]) / self.grid_resolution)
                self.waypoints.append((wp_grid_x, wp_grid_y))
        else:
            self.waypoints = []
    
    def _replan_trajectory_path(self, trajectory):
        """重新规划轨迹路径（仅用于可视化）"""
        waypoint_sequence = [self.start_point] + self.waypoints + [self.goal_point]
        full_path = []
        
        for i in range(len(waypoint_sequence) - 1):
            start = waypoint_sequence[i]
            goal = waypoint_sequence[i + 1]
            
            segment_path = self._get_path_planner().plan(
                (start[0], start[1]), (goal[0], goal[1]))["result"]
            
            if not segment_path:
                logger.info(f"警告: 无法重新规划从 {start} 到 {goal} 的路径")
                return
            
            if full_path:
                full_path.extend(segment_path[1:])
            else:
                full_path.extend(segment_path)
        
        self.current_path = full_path
        self.path_found = True
    
    def _wait_for_rendering(self, trajectory_id, trajectory):
        """等待渲染完成"""
        import time
        max_wait_time = 600  # 最多等待10分钟
        wait_interval = 2  # 每2秒检查一次
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            
            with self.render_lock:
                render_info = self.render_status.get(trajectory_id, {})
                status = render_info.get("status", "pending")
            
            if status == "completed":
                logger.info(f"✓ Trajectory {trajectory_id} 渲染完成!")
                # 重新调用start_batch_annotation加载视频
                self.start_batch_annotation()
                return
            elif status == "error":
                error_msg = render_info.get("error", "未知错误")
                logger.info(f"❌ Trajectory {trajectory_id} 渲染失败: {error_msg}")
                return
            
            if elapsed_time % 10 == 0:  # 每10秒提示一次
                logger.info(f"  等待中... ({elapsed_time}s)")
        
        logger.info(f"⚠️ 等待超时，Trajectory {trajectory_id} 渲染可能失败")
    
    def next_trajectory(self):
        """切换到下一条轨迹"""
        if not self.batch_annotation_mode:
            logger.info("❌ 不在批量标注模式")
            return
        
        # 标记当前轨迹已跳过
        self.current_trajectory_idx += 1
        
        if self.current_trajectory_idx >= len(self.trajectories):
            logger.info("✓ 已到达最后一条轨迹")
            self.current_trajectory_idx = len(self.trajectories) - 1
            return
        
        # 设置标志，请求切换视频（让主界面循环处理）
        self.request_video_switch = True
    
    def previous_trajectory(self):
        """切换到上一条轨迹"""
        if not self.batch_annotation_mode:
            logger.info("❌ 不在批量标注模式")
            return
        
        if self.current_trajectory_idx <= 0:
            logger.info("⚠️ 已是第一条轨迹")
            return
        
        self.current_trajectory_idx -= 1
        
        # 设置标志，请求切换视频（让主界面循环处理）
        self.request_video_switch = True
    
    def load_point_cloud(self, ply_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        加载PLY点云文件
        返回: (points, colors) - colors可能为None
        """
        logger.info(f"读取点云文件: {ply_path}")
        
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
        
        # 使用Open3D读取点云
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if pcd.is_empty():
            raise ValueError("点云文件为空!")
        
        # 获取点云坐标
        points = np.asarray(pcd.points)
        
        # 尝试获取颜色信息
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)  # 范围 [0, 1]
            colors = (colors * 255).astype(np.uint8)  # 转换为 [0, 255]
            logger.info(f"点云包含RGB颜色信息")
        
        logger.info(f"成功读取点云,共 {len(points)} 个点")
        if colors is not None:
            logger.info(f"  (包含RGB颜色信息)")
        else:
            logger.info(f"  (无颜色信息)")
        
        return points, colors
    
    def compute_bounds(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算点云边界"""
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        
        dx, dy, dz = max_pt - min_pt
        
        logger.info("\n=== 点云边界 ===")
        logger.info(f"X: [{min_pt[0]:.3f}, {max_pt[0]:.3f}] (范围: {dx:.3f} m)")
        logger.info(f"Y: [{min_pt[1]:.3f}, {max_pt[1]:.3f}] (范围: {dy:.3f} m)")
        logger.info(f"Z: [{min_pt[2]:.3f}, {max_pt[2]:.3f}] (范围: {dz:.3f} m)")
        
        return min_pt, max_pt
    
    def compute_ground_level(self, points: np.ndarray) -> Tuple[float, float, float]:
        """
        计算地面高度和障碍物高度范围
        返回: (ground_z, z_lower_bound, z_upper_bound)
        """
        z_values = points[:, 2]
        z_values_sorted = np.sort(z_values)
        
        # 使用10分位数作为地面高度
        ground_idx = int(len(z_values_sorted) * 0.10)
        ground_z = z_values_sorted[ground_idx]
        
        # 定义障碍物高度范围 (0.3m到1.5m以上地面)
        z_lower_bound = ground_z + 0.2
        z_upper_bound = ground_z + 1.5
        
        logger.info("\n=== 障碍物高度范围 ===")
        logger.info(f"地面高度 (10分位数): {ground_z:.3f} m")
        logger.info(f"障碍物Z范围: [{z_lower_bound:.3f}, {z_upper_bound:.3f}] m")
        logger.info(f"  (地面以上 0.2m - 1.5m)")
        
        return ground_z, z_lower_bound, z_upper_bound
    
    def create_occupancy_grid(self, points: np.ndarray, colors: Optional[np.ndarray],
                             min_pt: np.ndarray, max_pt: np.ndarray,
                             z_lower: float, z_upper: float,
                             resolution: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建占用网格
        返回: (occupancy_grid, color_projection, binary_projection)
        """
        dx = max_pt[0] - min_pt[0]
        dy = max_pt[1] - min_pt[1]
        
        # 计算网格尺寸
        grid_width = int(np.ceil(dx / resolution)) + 1
        grid_height = int(np.ceil(dy / resolution)) + 1
        
        logger.info(f"\n=== 网格配置 ===")
        logger.info(f"网格分辨率: {resolution} m/像素")
        logger.info(f"网格尺寸: {grid_width} x {grid_height} 像素")
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_resolution = resolution
        self.min_pt = min_pt
        self.max_pt = max_pt
        
        # 创建网格
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        color_projection = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        binary_projection = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # 投影点云到网格
        obstacle_points = 0
        total_points_in_range = 0
        
        for i, pt in enumerate(points):
            # 转换世界坐标到网格索引
            grid_x = int((pt[0] - min_pt[0]) / resolution)
            grid_y = int((max_pt[1] - pt[1]) / resolution)  # Y轴翻转
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                # 标记为占用
                binary_projection[grid_y, grid_x] = 255
                
                # 设置颜色
                if colors is not None:
                    color_projection[grid_y, grid_x] = colors[i][::-1]  # RGB -> BGR
                else:
                    # 基于高度的颜色映射
                    z_range = max_pt[2] - min_pt[2]
                    if z_range > 0:
                        normalized_z = (pt[2] - min_pt[2]) / z_range
                        if normalized_z < 0.5:
                            b = 255
                            g = int(normalized_z * 2 * 255)
                            r = 0
                        else:
                            b = int((1.0 - normalized_z) * 2 * 255)
                            g = 255
                            r = int((normalized_z - 0.5) * 2 * 255)
                        color_projection[grid_y, grid_x] = [b, g, r]
                    else:
                        color_projection[grid_y, grid_x] = [128, 128, 128]
                
                # 检查是否在障碍物高度范围内
                if z_lower <= pt[2] <= z_upper:
                    occupancy_grid[grid_y, grid_x] = 255
                    obstacle_points += 1
                
                total_points_in_range += 1
        
        logger.info(f"\n=== 投影统计 ===")
        logger.info(f"障碍物高度范围内的点数: {obstacle_points}")
        logger.info(f"总投影点数: {total_points_in_range}")
        
        obstacle_pixels = np.count_nonzero(occupancy_grid)
        total_pixels = grid_width * grid_height
        obstacle_ratio = obstacle_pixels / total_pixels * 100.0
        logger.info(f"障碍物像素: {obstacle_pixels} ({obstacle_ratio:.2f}%)")
        
        return occupancy_grid, color_projection, binary_projection
    
    def filter_occupancy_grid(self, occupancy_grid: np.ndarray) -> np.ndarray:
        """对占用网格进行过滤处理"""
        logger.info("\n=== 应用轻度滤波 ===")
        
        # 1. 中值滤波
        filtered = cv2.medianBlur(occupancy_grid, 3)
        
        # 2. 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 3. 移除微小孤立障碍物
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8)
        
        total_pixels = occupancy_grid.shape[0] * occupancy_grid.shape[1]
        min_area = int(total_pixels * 0.0001)  # 0.01%
        
        cleaned = np.zeros_like(occupancy_grid)
        valid_obstacles = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask = (labels == i).astype(np.uint8)
                cleaned[mask > 0] = closed[mask > 0]
                valid_obstacles += 1
        
        logger.info(f"移除了 {num_labels - 1 - valid_obstacles} 个微小孤立障碍物")
        logger.info(f"剩余障碍物: {valid_obstacles}")
        
        return cleaned
    
    def expand_obstacles(self, obstacle_map: np.ndarray, 
                        robot_radius: float = 0.3) -> np.ndarray:
        """扩展障碍物以考虑机器人半径"""
        logger.info("\n=== 路径规划 ===")
        
        expansion_radius = int(np.ceil(robot_radius / self.grid_resolution))
        logger.info(f"机器人半径: {robot_radius} m")
        logger.info(f"扩展半径: {expansion_radius} 像素")
        
        # 如果扩展半径为0或负数，直接返回原障碍物地图
        if expansion_radius <= 0:
            logger.info("扩展半径为0，跳过障碍物扩展")
            return obstacle_map.copy()
        
        # 膨胀操作 - 使用椭圆形核
        kernel_size = 2 * expansion_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(obstacle_map, kernel)
        
        # 统计扩展效果
        original_obstacle_pixels = np.count_nonzero(obstacle_map)
        expanded_obstacle_pixels = np.count_nonzero(expanded)
        additional_pixels = expanded_obstacle_pixels - original_obstacle_pixels
        logger.info(f"原始障碍物像素: {original_obstacle_pixels}")
        logger.info(f"扩展后障碍物像素: {expanded_obstacle_pixels} (+{additional_pixels})")
        
        return expanded
    
    def _get_path_planner(self) -> ShortestPathPlanner:
        """返回路径规划器实例，首次调用时自动构建安全代价场。"""
        if self._path_planner is None:
            logger.info("[PathPlanner] 初始化安全代价场...")
            self._path_planner = ShortestPathPlanner(self.expanded_traversability)
            logger.info("[PathPlanner] 初始化完成")
        return self._path_planner
    
    def create_base_visualization_image(self, use_color_map: bool = False) -> np.ndarray:
        """
        创建基础可视化图像
        
        Args:
            use_color_map: 如果为True，使用彩色点云地图；如果为False，使用黑白地图
        
        白色=可通行区域, 黑色=障碍物, 灰色=未知区域（点云外）
        """
        if use_color_map and self.display_image is not None:
            # 使用彩色点云地图作为背景
            base_image = self.display_image.copy()
            
            # 在未知区域（点云外）显示为深灰色
            if self.point_cloud_coverage is not None:
                unknown_mask = (self.point_cloud_coverage == 0)
                base_image[unknown_mask] = [64, 64, 64]
            
            # 半透明覆盖障碍物区域（红色标记）
            if self.obstacle_map is not None:
                obstacle_mask = (self.obstacle_map == 255)
                # 使用半透明红色覆盖障碍物
                base_image[obstacle_mask] = cv2.addWeighted(
                    base_image[obstacle_mask], 0.5,
                    np.full_like(base_image[obstacle_mask], [0, 0, 180]), 0.5, 0
                )
        else:
            # 创建灰色背景（未知区域 - 点云外）
            base_image = np.full((self.grid_height, self.grid_width, 3), 128, dtype=np.uint8)
            
            # 只在点云覆盖范围内设置颜色
            if self.point_cloud_coverage is not None:
                # 点云覆盖范围内：先设为白色（可通行）
                base_image[self.point_cloud_coverage == 255] = [255, 255, 255]
                
                # 黑色：障碍物区域（会覆盖白色）
                base_image[self.obstacle_map == 255] = [0, 0, 0]
            else:
                # 兼容旧逻辑
                base_image[self.expanded_traversability == 255] = [255, 255, 255]
                base_image[self.obstacle_map == 255] = [0, 0, 0]
        
        return base_image
    
    def run_astar_interactive(self, mode: str = "discrete", render_only: bool = False, for_batch_annotation: bool = False):
        """运行交互式A*搜索（支持多个途径点）
        
        Args:
            mode: "continuous" 或 "discrete"
            render_only: 是否只进行渲染（用于批量标注模式）
            for_batch_annotation: 是否为批量标注模式
        """
        if not self.start_set or not self.goal_set:
            logger.info("错误: 必须先设置起点和终点!")
            return
        
        logger.info("开始A*路径搜索...")
        logger.info(f"  起点: ({self.start_point[0]}, {self.start_point[1]})")
        if self.waypoints:
            logger.info(f"  途径点数量: {len(self.waypoints)}")
            for i, wp in enumerate(self.waypoints):
                logger.info(f"    途径点{i+1}: ({wp[0]}, {wp[1]})")
        logger.info(f"  终点: ({self.goal_point[0]}, {self.goal_point[1]})")
        logger.info(f"  模式: {mode}")
        
        self.path_found = False
        self.current_path = []
        
        # 构建完整路径序列：起点 -> 途径点1 -> 途径点2 -> ... -> 终点
        waypoint_sequence = [self.start_point] + self.waypoints + [self.goal_point]
        
        # 逐段执行A*搜索
        full_path = []
        for i in range(len(waypoint_sequence) - 1):
            start = waypoint_sequence[i]
            goal = waypoint_sequence[i + 1]
            
            logger.info(f"\n搜索路径段 {i+1}/{len(waypoint_sequence)-1}: ({start[0]}, {start[1]}) -> ({goal[0]}, {goal[1]})")
            
            segment_path = self._get_path_planner().plan(
                (start[0], start[1]), (goal[0], goal[1]))["result"]
            
            if not segment_path:
                logger.info(f"错误: 无法找到从 ({start[0]}, {start[1]}) 到 ({goal[0]}, {goal[1]}) 的路径!")
                self.draw_no_path()
                return
            
            # 合并路径（避免重复点）
            if full_path:
                full_path.extend(segment_path[1:])  # 跳过第一个点（与上一段的最后一个点重复）
            else:
                full_path.extend(segment_path)
            
            logger.info(f"  找到路径段，长度: {len(segment_path)} 个节点")
        
        if full_path:
            self.current_path = full_path
            self.path_found = True
            
            logger.info(f"\n完整路径找到! 总长度: {len(full_path)} 个节点")
            
            # 根据模式生成相机位姿
            if mode == "discrete":
                rotations, translations, camera_poses_grid, camera_poses_directions, actions = \
                    self.generate_camera_poses(full_path, mode="discrete", 
                                              start_yaw=self.start_yaw, goal_yaw=self.goal_yaw)
            else:
                rotations, translations, camera_poses_grid, camera_poses_directions, actions = \
                    self.generate_camera_poses(full_path, num_poses=100, mode="continuous")
            
            # 更新实例变量用于可视化
            if rotations is not None:
                self.camera_poses_grid = camera_poses_grid
                self.camera_poses_directions = camera_poses_directions
                
                # 保存相机位姿到JSON（字段与 main_sample.run_single_path 的 result 一致）
                self.save_camera_poses_json(
                    rotations, translations, len(rotations),
                    actions=actions, mode=mode,
                    full_path=full_path,
                    start_yaw=self.start_yaw,
                    goal_yaw=self.goal_yaw,
                )
            
            # 绘制最终路径
            self.draw_final_path()
            
            # 如果启用渲染，则渲染视频
            if self.enable_render or self.enable_annotate:
                self.render_and_display_video(mode)
        else:
            logger.info("\n未找到路径!")
            self.draw_no_path()
    
    def draw_final_path(self):
        """绘制最终路径和相机位姿"""
        # 使用彩色点云地图创建基础图像
        final_image = self.create_base_visualization_image(use_color_map=False)
        
        # 叠加所有已保存的历史轨迹（置于当前路径之下）
        self.draw_all_trajectories(final_image)
        
        # 绘制路径 - 使用鲜艳的红色以便清晰可见
        for i in range(len(self.current_path) - 1):
            pt1 = self.current_path[i]
            pt2 = self.current_path[i + 1]
            cv2.line(final_image, pt1, pt2, (0, 0, 255), 3)  # 红色路径，加粗
        
        # 绘制相机位姿
        if self.camera_poses_grid:
            axis_length = 5.0
            for i, cam_pos in enumerate(self.camera_poses_grid):
                # 绘制相机位置 - 使用橙色
                cv2.circle(final_image, cam_pos, 3, (0, 165, 255), -1)  # 橙色
                
                # 绘制坐标轴
                if i < len(self.camera_poses_directions):
                    z_dir = self.camera_poses_directions[i]
                    z_end = (int(cam_pos[0] + z_dir[0] * axis_length),
                            int(cam_pos[1] + z_dir[1] * axis_length))
                    cv2.arrowedLine(final_image, cam_pos, z_end, 
                                  (255, 0, 0), 2, tipLength=0.3)  # 蓝色Z轴
                    
                    # X轴(右方向)
                    x_dir_x = -z_dir[1]
                    x_dir_y = z_dir[0]
                    x_end = (int(cam_pos[0] + x_dir_x * axis_length),
                            int(cam_pos[1] + x_dir_y * axis_length))
                    cv2.arrowedLine(final_image, cam_pos, x_end,
                                  (0, 255, 0), 2, tipLength=0.3)  # 绿色X轴
        
        # 绘制起点、终点、途径点及朝向
        self.draw_points_and_orientations(final_image)
        
        cv2.imshow("A* Path Planning", final_image)
    
    def draw_no_path(self):
        """绘制未找到路径的提示"""
        # 使用彩色点云地图创建基础图像
        vis_image = self.create_base_visualization_image(use_color_map=False)
        
        # 绘制起点、终点、途径点及朝向
        self.draw_points_and_orientations(vis_image)
        
        # 添加提示文字
        cv2.putText(vis_image, "No path found!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("A* Path Planning", vis_image)
    
    def grid_to_world(self, grid_x: int, grid_y: int, z_height: float) -> np.ndarray:
        """将网格坐标转换为世界坐标"""
        world_x = self.min_pt[0] + grid_x * self.grid_resolution
        world_y = self.max_pt[1] - grid_y * self.grid_resolution  # Y轴翻转
        return np.array([world_x, world_y, z_height])
    
    def simplify_path(self, path: List[Tuple[int, int]], angle_threshold: float = 10.0, 
                     min_distance: float = 0.5) -> List[Tuple[int, int]]:
        """简化路径，只保留关键节点
        
        Args:
            path: 原始路径
            angle_threshold: 角度阈值（度），超过此阈值才保留节点
            min_distance: 最小距离阈值（米），相邻关键节点的最小距离
        
        Returns:
            简化后的路径
        """
        if len(path) <= 2:
            return path
        
        logger.info(f"\n=== 路径简化 ===")
        logger.info(f"原始路径节点数: {len(path)}")
        logger.info(f"角度阈值: {angle_threshold}°")
        logger.info(f"最小距离阈值: {min_distance}m")
        
        # 转换为世界坐标（2D）
        world_points = []
        for p in path:
            world_x = self.min_pt[0] + p[0] * self.grid_resolution
            world_y = self.max_pt[1] - p[1] * self.grid_resolution
            world_points.append([world_x, world_y])
        world_points = np.array(world_points)
        
        # 简化算法：基于角度和距离
        simplified_indices = [0]  # 总是保留起点
        
        for i in range(1, len(path) - 1):
            # 获取前一个关键点
            prev_idx = simplified_indices[-1]
            
            # 计算距离
            dist = np.linalg.norm(world_points[i] - world_points[prev_idx])
            
            # 如果距离太近，跳过（除非是最后一个点）
            if dist < min_distance:
                continue
            
            # 计算角度变化
            # 向量1: 从上一个关键点到当前点
            v1 = world_points[i] - world_points[prev_idx]
            # 向量2: 从当前点到下一个点
            v2 = world_points[i + 1] - world_points[i]
            
            # 归一化
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                continue
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # 计算夹角（使用点积）
            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
            
            # 如果角度变化超过阈值，保留该点
            if angle > angle_threshold:
                simplified_indices.append(i)
        
        # 总是保留终点
        simplified_indices.append(len(path) - 1)
        
        # 构建简化后的路径
        simplified_path = [path[i] for i in simplified_indices]
        
        logger.info(f"简化后路径节点数: {len(simplified_path)}")
        logger.info(f"压缩率: {(1 - len(simplified_path)/len(path)) * 100:.1f}%")
        
        return simplified_path
    
    def generate_discrete_actions(self, path: List[Tuple[int, int]], start_yaw: float = 0.0, goal_yaw: float = 0.0):
        """生成VLN风格的离散动作序列（Pure Pursuit路径跟踪算法）

        动作定义:
        1: 向前走0.25米
        2: 向左转15度
        3: 向右转15度

        使用Pure Pursuit（纯追踪）算法：将当前位置投影到完整路径上，
        沿路径前瞻固定距离得到目标点，通过转向消除横向偏差后再前进，
        实现对原始最短路的精确跟踪。

        Args:
            path: A*算出的路径（网格坐标）
            start_yaw: 起点朝向（弧度）
            goal_yaw: 终点朝向（弧度）

        Returns:
            tuple: (rotations, translations, camera_poses_grid, camera_poses_directions, actions)
        """
        logger.info("\n=== 生成离散动作序列（Pure Pursuit） ===")

        FORWARD_DIST    = 0.25             # 单步前进距离（米）
        TURN_ANGLE      = np.radians(15)   # 单次转向角（弧度）
        ANGLE_TOLERANCE = np.radians(15)  # 允许前进的最大角度偏差（半个转向步长）
        LOOKAHEAD_DIST  = 0.5              # Pure Pursuit前瞻距离（米）
        # GOAL_THRESHOLD 必须 > FORWARD_DIST，否则机器人会越过终点后永远无法触发退出条件
        GOAL_THRESHOLD  = 0.4   # 0.30m：前进一步内必然停止
        MAX_CONSEC_TURNS = 24              # 连续转向上限（相当于最多转一整圈），超出即退出
        MAX_ITERATIONS  = 5000            # 路径节点 * FORWARD_DIST 决定合理步数，5000 足够长

        # Use the configured final rendered camera height above the ground.
        camera_height = self.camera_height

        # 使用完整路径（不做简化），保留所有曲线细节以减小横向误差
        world_path = np.array(
            [[self.grid_to_world(p[0], p[1], camera_height)[0],
              self.grid_to_world(p[0], p[1], camera_height)[1]]
             for p in path],
            dtype=float
        )

        def _normalize_angle(a: float) -> float:
            while a > np.pi:
                a -= 2.0 * np.pi
            while a < -np.pi:
                a += 2.0 * np.pi
            return a

        # 预计算每段路径的弧长，用于限制 seg_idx 搜索窗口（解决回溯路径问题）
        seg_arc_lens = np.array([
            float(np.linalg.norm(world_path[i + 1] - world_path[i]))
            for i in range(len(world_path) - 1)
        ], dtype=float)
        # 每段的弧长起始累计值（seg i 从 seg_arc_cum[i] 开始）
        seg_arc_cum = np.concatenate([[0.0], np.cumsum(seg_arc_lens)])

        def _find_nearest_segment(pos: np.ndarray, wpath: np.ndarray, from_seg: int,
                                   traveled_arc: float) -> int:
            """从 from_seg 起，在弧长前向窗口内找到距 pos 最近的路径段索引。

            关键改动：限制搜索范围为 [from_seg, from_seg + WINDOW_SEGS]，
            防止回溯路径（起终点相同）时 seg_idx 被吸回到路径前半段，
            导致 lookahead 目标永远指向已走过的方向而原地打转。

            搜索窗口用弧长限制：当前弧长进度 ± ARC_WINDOW 范围内的段。
            """
            ARC_WINDOW = LOOKAHEAD_DIST * 4  # 在前瞻距离 4 倍弧长窗口内搜索
            arc_lo = traveled_arc - FORWARD_DIST  # 允许小幅回退（对齐误差补偿）
            arc_hi = traveled_arc + ARC_WINDOW

            min_dist = float('inf')
            best_seg = from_seg
            for i in range(from_seg, len(wpath) - 1):
                # 超出弧长上界：停止搜索
                if seg_arc_cum[i] > arc_hi:
                    break
                # 未到弧长下界：跳过
                if seg_arc_cum[i + 1] < arc_lo:
                    continue
                p1, p2 = wpath[i], wpath[i + 1]
                seg = p2 - p1
                seg_len_sq = float(np.dot(seg, seg))
                if seg_len_sq < 1e-12:
                    continue
                t = float(np.clip(np.dot(pos - p1, seg) / seg_len_sq, 0.0, 1.0))
                dist = float(np.linalg.norm(pos - (p1 + t * seg)))
                if dist < min_dist:
                    min_dist = dist
                    best_seg = i
            return best_seg

        def _get_lookahead_point(pos: np.ndarray, wpath: np.ndarray,
                                 seg_idx: int, lookahead: float) -> np.ndarray:
            """Pure Pursuit前瞻：将 pos 投影到 seg_idx 段上，
            然后沿路径向前累积 lookahead 距离，返回前瞻目标点。
            横向偏差通过投影自动修正，无需额外PID积分项。
            始终从 seg_idx 向路径末尾方向前进，不会倒退。
            """
            p1, p2 = wpath[seg_idx], wpath[seg_idx + 1]
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            t = float(np.clip(np.dot(pos - p1, seg) / max(seg_len_sq, 1e-12), 0.0, 1.0))
            proj = p1 + t * seg  # 当前位置在路径上的投影点

            remaining = lookahead
            for i in range(seg_idx, len(wpath) - 1):
                start = proj if i == seg_idx else wpath[i]
                end   = wpath[i + 1]
                seg_vec = end - start
                seg_len = float(np.linalg.norm(seg_vec))
                if seg_len < 1e-6:
                    continue
                if remaining <= seg_len:
                    return start + remaining * seg_vec / seg_len
                remaining -= seg_len

            return wpath[-1].copy()

        current_pos = world_path[0].copy()
        current_yaw = float(start_yaw)

        actions: List[int] = []
        positions = [current_pos.copy()]
        yaws = [current_yaw]

        # ── 初始转向：对齐到路径第一段的方向 ─────────────────────────────
        if len(world_path) > 1:
            first_seg = world_path[1] - world_path[0]
            if np.linalg.norm(first_seg) > 1e-6:
                target_yaw  = float(np.arctan2(first_seg[1], first_seg[0]))
                angle_diff  = _normalize_angle(target_yaw - current_yaw)
                initial_turns = 0
                while abs(angle_diff) > np.radians(7.5) and initial_turns < 24:
                    if angle_diff > 0:
                        actions.append(2)
                        current_yaw = _normalize_angle(current_yaw + TURN_ANGLE)
                    else:
                        actions.append(3)
                        current_yaw = _normalize_angle(current_yaw - TURN_ANGLE)
                    angle_diff = _normalize_angle(target_yaw - current_yaw)
                    positions.append(current_pos.copy())
                    yaws.append(current_yaw)
                    initial_turns += 1
                if initial_turns > 0:
                    logger.info(f"初始转向: {initial_turns} 次，"
                                f"转向角度: {np.degrees(_normalize_angle(target_yaw - start_yaw)):.1f}°")

        # ── Pure Pursuit主循环 ────────────────────────────────────────────
        path_total_len = float(seg_arc_cum[-1])  # 路径总弧长（世界坐标）
        # 至少走完路径总弧长的 60% 后才启用终点距离检测
        # 防止起终点很近（含 waypoint 回溯）时出发即退出
        MIN_PROGRESS_RATIO = 0.6
        traveled_arc = 0.0  # 累计前进弧长（只由 FORWARD 动作累加）

        seg_idx      = 0
        iteration    = 0
        consec_turns = 0   # 连续转向计数，用于检测震荡
        stop_reason  = "unknown"
        last_near_end_by_seg = False
        last_near_end_by_dist = False
        last_progress_sufficient = False
        while iteration < MAX_ITERATIONS:
            iteration += 1

            # 已走过足够弧长 + 实际进入目标半径 → 退出
            # 不能只依赖 seg_idx 接近末段，否则在稀疏/长路径中会提前停止，
            # 导致模拟轨迹终点离用户选择的目标点很远。
            near_end_by_seg = seg_idx >= len(world_path) - 2
            near_end_by_dist = (
                float(np.linalg.norm(world_path[-1] - current_pos)) < GOAL_THRESHOLD
            )
            progress_sufficient = traveled_arc >= path_total_len * MIN_PROGRESS_RATIO
            last_near_end_by_seg = near_end_by_seg
            last_near_end_by_dist = near_end_by_dist
            last_progress_sufficient = progress_sufficient
            if progress_sufficient and near_end_by_dist:
                stop_reason = "near_end_by_dist"
                break

            # 更新路径进度：在弧长窗口内搜索最近段，防止回溯路径时 seg_idx 倒退
            seg_idx = _find_nearest_segment(current_pos, world_path, seg_idx, traveled_arc)

            # 计算前瞻目标点（隐式修正横向偏差，始终朝路径前进方向）
            lookahead_pt = _get_lookahead_point(current_pos, world_path, seg_idx, LOOKAHEAD_DIST)

            to_target = lookahead_pt - current_pos
            dist = float(np.linalg.norm(to_target))
            if dist < 1e-6:
                seg_idx = min(seg_idx + 1, len(world_path) - 2)
                continue

            target_yaw = float(np.arctan2(to_target[1], to_target[0]))
            angle_diff = _normalize_angle(target_yaw - current_yaw)

            if abs(angle_diff) > ANGLE_TOLERANCE and consec_turns < MAX_CONSEC_TURNS:
                # 方向未对齐：转向
                if angle_diff > 0:
                    actions.append(2)  # 左转
                    current_yaw = _normalize_angle(current_yaw + TURN_ANGLE)
                else:
                    actions.append(3)  # 右转
                    current_yaw = _normalize_angle(current_yaw - TURN_ANGLE)
                consec_turns += 1
            else:
                if consec_turns >= MAX_CONSEC_TURNS:
                    if stop_reason == "unknown":
                        stop_reason = "max_consecutive_turns"
                    # 连续转满一圈仍未对齐 → 机器人陷入震荡，提前退出
                    logger.info(f"警告: 连续转向 {consec_turns} 次仍未对齐，提前退出主循环")
                    break
                # 方向对齐：向前走
                next_pos = current_pos + FORWARD_DIST * np.array(
                    [np.cos(current_yaw), np.sin(current_yaw)]
                )
                actions.append(1)
                current_pos = next_pos
                traveled_arc += FORWARD_DIST
                consec_turns = 0  # 成功前进，重置计数器

            positions.append(current_pos.copy())
            yaws.append(current_yaw)

        if iteration >= MAX_ITERATIONS:
            if stop_reason == "unknown":
                stop_reason = "max_iterations"
            logger.info(f"警告: 达到最大迭代次数 {MAX_ITERATIONS}")

        # ── 终点朝向对齐 ──────────────────────────────────────────────────
        final_turns = 0
        angle_diff  = _normalize_angle(goal_yaw - current_yaw)
        while abs(angle_diff) > np.radians(7.5) and final_turns < 24:
            if angle_diff > 0:
                actions.append(2)
                current_yaw = _normalize_angle(current_yaw + TURN_ANGLE)
            else:
                actions.append(3)
                current_yaw = _normalize_angle(current_yaw - TURN_ANGLE)
            angle_diff = _normalize_angle(goal_yaw - current_yaw)
            positions.append(current_pos.copy())
            yaws.append(current_yaw)
            final_turns += 1

        if final_turns > 0:
            logger.info(f"终点转向: {final_turns} 次，"
                        f"转向角度: {np.degrees(_normalize_angle(goal_yaw - yaws[-final_turns - 1])):.1f}°")

        logger.info(f"生成了 {len(actions)} 个离散动作")
        logger.info(f"动作统计:")
        logger.info(f"  前进(1): {actions.count(1)} 次")
        logger.info(f"  左转(2): {actions.count(2)} 次")
        logger.info(f"  右转(3): {actions.count(3)} 次")

        # ── 生成相机位姿 ──────────────────────────────────────────────────
        rotations            = []
        translations         = []
        camera_poses_grid    = []
        camera_poses_directions = []

        for pos, yaw in zip(positions, yaws):
            world_pos = np.array([pos[0], pos[1], camera_height])

            dir_x = np.cos(yaw)
            dir_y = np.sin(yaw)

            R = np.zeros((3, 3))
            R[:, 0] = [dir_y, -dir_x, 0.0]   # X轴（右）
            R[:, 1] = [0.0,   0.0,   -1.0]   # Y轴（下）
            R[:, 2] = [dir_x,  dir_y,  0.0]  # Z轴（前）

            rotations.append(R)
            translations.append(world_pos)

            grid_x = int((pos[0] - self.min_pt[0]) / self.grid_resolution)
            grid_y = int((self.max_pt[1] - pos[1]) / self.grid_resolution)
            camera_poses_grid.append((grid_x, grid_y))
            camera_poses_directions.append([dir_x, dir_y])

        final_xy = np.array([translations[-1][0], translations[-1][1]], dtype=float)
        goal_xy = np.array(world_path[-1], dtype=float)
        path_start_xy = np.array(world_path[0], dtype=float)
        final_distance_to_goal = float(np.linalg.norm(final_xy - goal_xy))
        start_distance_to_goal = float(np.linalg.norm(path_start_xy - goal_xy))
        self._last_discrete_debug = {
            "stop_reason": stop_reason,
            "iterations": int(iteration),
            "num_actions": int(len(actions)),
            "num_forward": int(actions.count(1)),
            "num_turn_left": int(actions.count(2)),
            "num_turn_right": int(actions.count(3)),
            "path_num_points": int(len(world_path)),
            "path_total_len": float(path_total_len),
            "traveled_arc": float(traveled_arc),
            "seg_idx": int(seg_idx),
            "near_end_by_seg": bool(last_near_end_by_seg),
            "near_end_by_dist": bool(last_near_end_by_dist),
            "progress_sufficient": bool(last_progress_sufficient),
            "goal_threshold": float(GOAL_THRESHOLD),
            "min_progress_ratio": float(MIN_PROGRESS_RATIO),
            "start_xy": path_start_xy.tolist(),
            "goal_xy": goal_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "start_distance_to_goal": start_distance_to_goal,
            "final_distance_to_goal": final_distance_to_goal,
            "last_positions_xy": [np.array(p, dtype=float).tolist() for p in positions[-10:]],
            "last_distances_to_goal": [float(np.linalg.norm(np.array(p, dtype=float) - goal_xy)) for p in positions[-10:]],
        }
        logger.info(f"第一个位姿 T = {translations[0]}")
        logger.info(f"最后一个位姿 T = {translations[-1]}")
        logger.info(
            "离散动作诊断: stop_reason=%s, final_distance_to_goal=%.3f, seg_idx=%s/%s",
            stop_reason,
            final_distance_to_goal,
            seg_idx,
            len(world_path) - 1,
        )

        return rotations, translations, camera_poses_grid, camera_poses_directions, actions
    
    def generate_camera_poses(self, path: List[Tuple[int, int]], num_poses: int = 100, mode: str = "discrete", 
                             start_yaw: float = 0.0, goal_yaw: float = 0.0):
        """沿路径生成相机位姿
        
        Args:
            path: A*算出的路径
            num_poses: 相机位姿数量（仅continuous模式使用）
            mode: "continuous" 或 "discrete"
                - continuous: 按固定频率采样相机位姿
                - discrete: VLN风格的离散动作序列
            start_yaw: 起点朝向（弧度，仅discrete模式使用）
            goal_yaw: 终点朝向（弧度，仅discrete模式使用）
            
        Returns:
            tuple: (rotations, translations, camera_poses_grid, camera_poses_directions, actions)
                   actions仅在discrete模式下有值，continuous模式为None
        """
        if len(path) < 2:
            logger.info("路径太短,无法生成相机位姿!")
            return None, None, None, None, None
        
        logger.info(f"\n=== 生成相机位姿 ===")
        logger.info(f"模式: {mode}")
        logger.info(f"路径长度: {len(path)} 个节点")
        if mode == "continuous":
            logger.info(f"相机位姿数量: {num_poses}")
        
        if mode == "discrete":
            # VLN风格的离散动作模式
            return self.generate_discrete_actions(path, start_yaw, goal_yaw)
        
        # continuous模式: 计算路径总长度
        segment_lengths = []
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = (path[i+1][0] - path[i][0]) * self.grid_resolution
            dy = (path[i+1][1] - path[i][1]) * self.grid_resolution
            seg_len = np.sqrt(dx**2 + dy**2)
            segment_lengths.append(seg_len)
            total_length += seg_len
        
        logger.info(f"路径总长度: {total_length:.3f} m")
        
        camera_height = self.camera_height  # Final camera height above ground, controlled by --camera_height
        
        # 沿路径采样位置
        camera_world_positions_raw = []
        camera_grid_positions_raw = []
        
        step_length = total_length / (num_poses - 1)
        current_distance = 0.0
        path_segment = 0
        segment_accumulated = 0.0
        
        for i in range(num_poses):
            # 找到当前所在路径段
            while (path_segment < len(segment_lengths) and 
                   current_distance > segment_accumulated + segment_lengths[path_segment]):
                segment_accumulated += segment_lengths[path_segment]
                path_segment += 1
            
            if path_segment >= len(path) - 1:
                path_segment = len(path) - 2
            
            # 在当前段内插值
            t = 0.0
            if segment_lengths[path_segment] > 1e-6:
                t = (current_distance - segment_accumulated) / segment_lengths[path_segment]
            t = np.clip(t, 0.0, 1.0)
            
            # 插值网格坐标
            grid_x = path[path_segment][0] + t * (path[path_segment + 1][0] - path[path_segment][0])
            grid_y = path[path_segment][1] + t * (path[path_segment + 1][1] - path[path_segment][1])
            
            camera_grid_positions_raw.append((int(grid_x), int(grid_y)))
            
            # 转换为世界坐标
            world_pos = self.grid_to_world(int(grid_x), int(grid_y), camera_height)
            camera_world_positions_raw.append(world_pos)
            
            current_distance += step_length
        
        # 平滑轨迹
        camera_world_positions = []
        camera_grid_positions = []
        smooth_window = 3
        half_window = smooth_window // 2
        
        for i in range(num_poses):
            # 世界坐标平滑
            start_idx = max(0, i - half_window)
            end_idx = min(num_poses, i + half_window + 1)
            smoothed_pos = np.mean(camera_world_positions_raw[start_idx:end_idx], axis=0)
            camera_world_positions.append(smoothed_pos)
            
            # 网格坐标平滑
            grid_positions = camera_grid_positions_raw[start_idx:end_idx]
            avg_x = int(np.mean([p[0] for p in grid_positions]))
            avg_y = int(np.mean([p[1] for p in grid_positions]))
            camera_grid_positions.append((avg_x, avg_y))
        
        # 计算旋转矩阵和方向
        rotations = []
        translations = []
        camera_poses_grid = []
        camera_poses_directions = []
        
        for i in range(num_poses):
            # 计算切线方向(世界坐标)
            if i < num_poses - 1:
                dx = camera_world_positions[i+1][0] - camera_world_positions[i][0]
                dy = camera_world_positions[i+1][1] - camera_world_positions[i][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 1e-6:
                    dir_x, dir_y = dx / dist, dy / dist
                else:
                    dir_x, dir_y = 1.0, 0.0
                
                # 网格方向(用于可视化)
                grid_dx = camera_grid_positions[i+1][0] - camera_grid_positions[i][0]
                grid_dy = camera_grid_positions[i+1][1] - camera_grid_positions[i][1]
                grid_dist = np.sqrt(grid_dx**2 + grid_dy**2)
                if grid_dist > 1e-6:
                    grid_dir_x = grid_dx / grid_dist
                    grid_dir_y = grid_dy / grid_dist
                else:
                    grid_dir_x, grid_dir_y = 1.0, 0.0
            else:
                # 最后一个位姿使用后向切线
                dx = camera_world_positions[i][0] - camera_world_positions[i-1][0]
                dy = camera_world_positions[i][1] - camera_world_positions[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 1e-6:
                    dir_x, dir_y = dx / dist, dy / dist
                else:
                    dir_x, dir_y = 1.0, 0.0
                
                grid_dx = camera_grid_positions[i][0] - camera_grid_positions[i-1][0]
                grid_dy = camera_grid_positions[i][1] - camera_grid_positions[i-1][1]
                grid_dist = np.sqrt(grid_dx**2 + grid_dy**2)
                if grid_dist > 1e-6:
                    grid_dir_x = grid_dx / grid_dist
                    grid_dir_y = grid_dy / grid_dist
                else:
                    grid_dir_x, grid_dir_y = 1.0, 0.0
            
            # 存储方向用于可视化
            camera_poses_directions.append([grid_dir_x, grid_dir_y])
            camera_poses_grid.append(camera_grid_positions[i])
            
            # 构建旋转矩阵 (相机到世界)
            # 世界坐标系: Z上, X前, Y左
            # 相机坐标系: Z前, X右, Y下
            R = np.zeros((3, 3))
            R[:, 0] = [dir_y, -dir_x, 0.0]   # X轴(右)
            R[:, 1] = [0.0, 0.0, -1.0]       # Y轴(下)
            R[:, 2] = [dir_x, dir_y, 0.0]    # Z轴(前)
            
            rotations.append(R)
            translations.append(camera_world_positions[i])
        
        logger.info(f"第一个相机位姿 T = {translations[0]}")
        logger.info(f"最后一个相机位姿 T = {translations[-1]}")
        
        # 返回结果（continuous模式不包含动作序列）
        return rotations, translations, camera_poses_grid, camera_poses_directions, None
    
    def _init_render_manager(self):
        """初始化渲染管理器（延迟加载）"""
        if self.render_manager is not None:
            return True
        
        if not RENDER_AVAILABLE:
            logger.info("错误: 渲染功能不可用")
            return False
        
        if self.ply_path is None or not os.path.exists(self.ply_path):
            logger.info("错误: 未找到点云文件，无法初始化渲染器")
            return False
        
        try:
            logger.info(f"\n=== 初始化渲染器（仅首次加载）===")
            self.render_manager = RenderManager(
                ply_path=self.ply_path,
                sh_degree=3,
                background_color=[0, 0, 0]
            )
            return True
        except Exception as e:
            logger.info(f"渲染器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def render_and_display_video(self, mode: str, for_batch: bool = False):
        """渲染路径并显示视频（使用预加载的模型）
        
        Args:
            mode: "continuous" 或 "discrete"
            for_batch: 是否为批量标注模式
        """
        
        # 确定相机位姿JSON文件路径
        if mode == "discrete":
            camera_json = os.path.join(self.output_dir, "camera_poses_discrete.json")
        else:
            camera_json = os.path.join(self.output_dir, "camera_poses.json")
        
        if not os.path.exists(camera_json):
            # 创建相机位姿文件
            with open(camera_json, 'w') as f:
                json.dump({
                    "mode": mode,
                    "num_cameras": 0,
                    "cameras": [],
                }, f, indent=2, ensure_ascii=False)
        
        if not os.path.exists(camera_json):
            logger.info(f"错误: 相机位姿文件不存在: {camera_json}")
            return
        
        # 创建渲染输出目录
        render_output_dir = os.path.join(self.output_dir, f"render_episode_{self.current_episode_id:04d}")
        os.makedirs(render_output_dir, exist_ok=True)
        self.last_render_dir = render_output_dir
        
        logger.info(f"\n=== 开始渲染 Episode {self.current_episode_id} ===")
        logger.info(f"相机类型: {self.camera_type}")
        logger.info(f"相机位姿: {camera_json}")
        logger.info(f"输出目录: {render_output_dir}")
        
        try:
            if self.camera_type == 'pano':
                # 全景视角渲染
                logger.info("使用全景视角渲染...")
                
                # 读取相机位姿
                with open(camera_json, 'r') as f:
                    camera_data = json.load(f)
                
                camera_poses = camera_data.get('cameras', [])
                if not camera_poses:
                    logger.info("错误: JSON文件中没有相机位姿")
                    return
                
                # 渲染全景轨迹
                success = self.render_manager.render_panorama_trajectory(
                    camera_poses=camera_poses,
                    output_dir=render_output_dir,
                    create_video=True,
                    video_fps=10,
                    pano_width=2048,
                    pano_height=1024,
                    save_cube_faces=False,  # 保存立方体6个面用于调试
                    use_multithreading=True,
                    max_workers=6,
                    enable_depth=self.args.enable_depth
                )
            else:
                # 单视角渲染
                logger.info("使用单视角渲染...")
                success = self.render_manager.render_from_json(
                    camera_json_path=camera_json,
                    output_dir=render_output_dir,
                    create_video=True,
                    video_fps=10,
                    enable_depth=self.args.enable_depth
                )
            
            if not success:
                logger.info("渲染失败")
                return
            
            logger.info("✓ 渲染完成!")
            
            # 显示视频
            self.display_rendered_video(render_output_dir, for_batch=for_batch)
            
        except Exception as e:
            logger.info(f"渲染过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    def display_rendered_video(self, render_dir: str, for_batch: bool = False):
        """在集成界面中显示渲染的视频
        
        Args:
            render_dir: 渲染输出目录
            for_batch: 是否为批量标注模式
        """
        # 根据相机类型选择视频文件
        if self.camera_type == 'pano':
            rgb_video_path = os.path.join(render_dir, "rgb_pano_video.mp4")
            depth_video_path = os.path.join(render_dir, "depth_pano_video.mp4")
        else:
            rgb_video_path = os.path.join(render_dir, "rgb_video.mp4")
            depth_video_path = os.path.join(render_dir, "depth_video.mp4")
        
        if not os.path.exists(rgb_video_path):
            logger.info(f"RGB视频不存在: {rgb_video_path}")
            return
        
        logger.info(f"\n=== 显示渲染视频 ===")
        logger.info(f"相机类型: {self.camera_type}")
        logger.info("操作说明:")
        logger.info("  空格键: 播放/暂停")
        logger.info("  拖动进度条: 跳转到指定帧")
        logger.info("\n指令输入方式:")
        logger.info("  方式1 - GUI输入: 点击输入框，在界面中输入，按Enter保存")
        logger.info("  方式2 - 终端输入: 按'i'键，在终端中输入，按Enter保存")
        logger.info("  ESC: 取消输入")
        logger.info("\n  按 'q' 或 ESC: 退出")
        
        # 先关闭旧视频资源（如果存在）
        self.close_video()
        
        # 打开视频
        self.rgb_cap = cv2.VideoCapture(rgb_video_path)
        self.depth_cap = cv2.VideoCapture(depth_video_path) if os.path.exists(depth_video_path) else None
        
        if not self.rgb_cap.isOpened():
            logger.info("无法打开RGB视频")
            return
        
        # 获取视频信息
        self.total_frames = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0:
            self.video_fps = 10
        
        self.current_frame = 0
        self.video_playing = True
        self.video_paused = False
        
        # 创建集成界面
        self.show_integrated_interface()
        
        # 清理
        self.close_video()
        
        # 如果是批量模式，更新渲染状态
        if for_batch and self.batch_annotation_mode:
            trajectory = self.trajectories[self.current_trajectory_idx]
            trajectory_id = trajectory['trajectory_id']
            trajectory['render_dir'] = render_dir
            self.save_trajectories()
            
            with self.render_lock:
                self.render_status[trajectory_id] = {
                    "status": "completed",
                    "render_dir": render_dir,
                    "video_paths": {
                        "rgb": rgb_video_path,
                        "depth": depth_video_path if os.path.exists(depth_video_path) else None
                    }
                }
    
    def load_and_display_video(self, rgb_video_path: str, depth_video_path: str = None, for_batch=False):
        """加载并显示已渲染的视频
        
        Args:
            rgb_video_path: RGB视频路径
            depth_video_path: Depth视频路径（可选）
        """
        if not os.path.exists(rgb_video_path):
            logger.info(f"RGB视频不存在: {rgb_video_path}")
            return
        
        logger.info(f"\n=== 加载已渲染视频 ===")
        logger.info(f"RGB视频: {rgb_video_path}")
        if depth_video_path and os.path.exists(depth_video_path):
            logger.info(f"Depth视频: {depth_video_path}")
        
        # 先关闭旧视频资源（如果存在）
        self.close_video()
        
        # 打开视频
        self.rgb_cap = cv2.VideoCapture(rgb_video_path)
        self.depth_cap = cv2.VideoCapture(depth_video_path) if (depth_video_path and os.path.exists(depth_video_path)) else None
        
        if not self.rgb_cap.isOpened():
            logger.info("无法打开RGB视频")
            return
        
        # 获取视频信息
        self.total_frames = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0:
            self.video_fps = 10
        
        self.current_frame = 0
        self.video_playing = True
        self.video_paused = False
        
        # 创建集成界面
        self.show_integrated_interface(for_batch=for_batch)
        
        # 清理
        self.close_video()
    
    def show_integrated_interface(self, for_batch=False):
        """显示集成的界面（地图+视频+标注）"""
        window_name = "Navigation Annotation Interface"
        
        # 只在窗口未创建时创建窗口（避免批量模式下重复创建）
        if not self.integrated_window_created:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1600, 900)
            # 设置鼠标回调
            cv2.setMouseCallback(window_name, self.integrated_mouse_callback)
            # 启动终端命令监听线程
            self.start_terminal_command_listener()
            self.integrated_window_created = True
        
        while True:
            # 检查是否需要退出
            if self.should_exit:
                logger.info("✓ 收到退出信号，正在关闭界面...")
                break
            
            # 检查是否需要切换视频（批量模式）
            if self.request_video_switch and for_batch:
                logger.info("✓ 检测到视频切换请求，退出当前界面...")
                break
            
            # 创建集成界面
            interface = self.create_integrated_interface()
            
            cv2.imshow(window_name, interface)
            
            # 控制帧率
            delay = int(1000 / self.video_fps) if not self.video_paused else 100
            key = cv2.waitKey(delay) & 0xFF
            
            # 处理键盘输入
            if self.text_input_active:
                # 文本输入模式
                if key == 27:  # ESC - 取消输入
                    self.text_input_active = False
                    self.text_input_buffer = ""
                elif key == 13 or key == 10:  # Enter - 保存
                    if self.text_input_buffer.strip():
                        self.current_instruction = self.text_input_buffer.strip()
                        self.save_annotation()
                        logger.info("✓ 指令已通过GUI输入并保存")
                        self.text_input_buffer = ""
                    else:
                        logger.info("❌ 输入为空，取消保存")
                    self.text_input_active = False
                elif key == 8:  # Backspace
                    if len(self.text_input_buffer) > 0:
                        self.text_input_buffer = self.text_input_buffer[:-1]
                elif key != 255 and 32 <= key <= 126:  # 可打印字符
                    self.text_input_buffer += chr(key)
            else:
                # 普通控制模式
                if key == ord('q') or key == 27:  # q 或 ESC
                    break
                elif key == ord(' '):  # 空格 - 播放/暂停
                    self.video_paused = not self.video_paused
                elif key == ord('r'):  # r - 重新播放
                    self.current_frame = 0
                    self.rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.depth_cap:
                        self.depth_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                elif key == ord('i') and self.enable_annotate:  # i - 终端输入模式
                    if self.terminal_input_active:
                        logger.info("⚠️ 终端输入模式已经在运行中，请先完成当前输入")
                    # elif self.path_found and self.last_render_dir is not None:
                    #     logger.info("\n" + "="*60)
                    #     logger.info("已切换到终端输入模式")
                    #     logger.info("="*60)
                    #     self.input_instruction(use_terminal=True)
                    else:
                        logger.info("请先完成路径规划和渲染!")
                elif key == ord('g') and self.enable_annotate:  # g - GUI输入模式
                    if self.terminal_input_active:
                        logger.info("⚠️ 终端输入模式正在运行中，请先完成终端输入")
                    elif self.path_found and self.last_render_dir is not None:
                        logger.info("\n已切换到GUI输入模式，请点击下方输入框输入指令")
                        self.text_input_active = True
                        self.text_input_buffer = self.current_instruction
                    else:
                        logger.info("请先完成路径规划和渲染!")
            
            # 更新视频帧
            if not self.video_paused and self.video_playing:
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0
                    self.rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.depth_cap:
                        self.depth_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 更新光标闪烁
            self.text_cursor_timer += 1
            if self.text_cursor_timer > 5:
                self.text_cursor_timer = 0
                self.text_cursor_visible = not self.text_cursor_visible

        if not for_batch:
            cv2.destroyWindow(window_name)
    
    def create_integrated_interface(self) -> np.ndarray:
        """创建集成界面布局
        
        布局:
        ┌─────────────┬─────────────┐
        │             │   RGB Video │
        │     Map     ├─────────────┤
        │             │ Depth Video │
        ├─────────────┴─────────────┤
        │   Video Progress Bar      │
        ├───────────────────────────┤
        │   Instruction Input       │
        └───────────────────────────┘
        """
        # 尺寸定义
        map_width, map_height = 800, 600
        video_width = 800
        rgb_video_height = 290
        depth_video_height = map_height - rgb_video_height  # 310
        progress_height = 50
        input_height = 160
        
        total_width = map_width + video_width
        total_height = map_height + progress_height + input_height
        
        # 创建空白画布
        canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240
        
        # 1. 左侧：地图视图
        map_view = self.create_map_view(map_width, map_height)
        canvas[0:map_height, 0:map_width] = map_view
        
        # 2. 右上：RGB 视频
        rgb_view = self.create_video_view("RGB", video_width, rgb_video_height, is_rgb=True)
        canvas[0:rgb_video_height, map_width:total_width] = rgb_view
        
        # 3. 右中：Depth 视频
        depth_view = self.create_video_view("Depth", video_width, depth_video_height, is_rgb=False)
        canvas[rgb_video_height:map_height, map_width:total_width] = depth_view
        
        # 4. 底部：进度条
        progress_view = self.create_progress_bar(total_width, progress_height)
        canvas[map_height:map_height+progress_height, 0:total_width] = progress_view
        
        # 5. 最底部：指令输入区域
        input_view = self.create_input_area(total_width, input_height)
        canvas[map_height+progress_height:total_height, 0:total_width] = input_view
        
        # 存储区域坐标（用于鼠标交互）
        self.ui_regions = {
            'map': (0, 0, map_width, map_height),
            'rgb_video': (map_width, 0, total_width, rgb_video_height),
            'depth_video': (map_width, rgb_video_height, total_width, map_height),
            'progress_bar': (0, map_height, total_width, map_height + progress_height),
            'input_area': (0, map_height + progress_height, total_width, total_height),
        }
        
        return canvas
    
    def create_map_view(self, width: int, height: int) -> np.ndarray:
        """创建地图视图"""
        # 获取当前地图
        map_img = self.create_base_visualization_image(use_color_map=False)
        
        # 绘制路径和相机位姿
        if self.path_found and self.current_path:
            for i in range(len(self.current_path) - 1):
                pt1 = self.current_path[i]
                pt2 = self.current_path[i + 1]
                cv2.line(map_img, pt1, pt2, (0, 0, 255), 2)
        
        # 绘制起点、终点
        self.draw_points_and_orientations(map_img)
        
        # 调整大小
        map_resized = cv2.resize(map_img, (width, height))
        
        # 添加标题
        cv2.putText(map_resized, "Navigation Map", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return map_resized
    
    def create_video_view(self, title: str, width: int, height: int, is_rgb: bool) -> np.ndarray:
        """创建视频视图"""
        # 创建空白帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 读取当前帧
        cap = self.rgb_cap if is_rgb else self.depth_cap
        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, video_frame = cap.read()
            if ret:
                # 调整大小以适应视图
                video_frame_resized = cv2.resize(video_frame, (width, height))
                frame = video_frame_resized
        
        # 添加标题和帧信息
        cv2.putText(frame, f"{title} Video", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 播放状态
        status = "Paused" if self.video_paused else "Playing"
        cv2.putText(frame, status, (width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not self.video_paused else (0, 165, 255), 2)
        
        return frame
    
    def close_video(self):
        """关闭并释放视频资源"""
        if self.rgb_cap is not None:
            self.rgb_cap.release()
            self.rgb_cap = None
        
        if self.depth_cap is not None:
            self.depth_cap.release()
            self.depth_cap = None
        
        # 重置视频状态
        self.video_playing = False
        self.video_paused = False
        self.current_frame = 0
        self.total_frames = 0
    
    def create_progress_bar(self, width: int, height: int) -> np.ndarray:
        """创建视频进度条"""
        bar = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # 绘制进度条背景
        bar_y = height // 2
        bar_height = 20
        bar_x_start = 50
        bar_x_end = width - 50
        bar_width = bar_x_end - bar_x_start
        
        # 背景条
        cv2.rectangle(bar, (bar_x_start, bar_y - bar_height//2),
                     (bar_x_end, bar_y + bar_height//2), (150, 150, 150), -1)
        
        # 进度条
        if self.total_frames > 0:
            progress = self.current_frame / self.total_frames
            progress_x = bar_x_start + int(bar_width * progress)
            cv2.rectangle(bar, (bar_x_start, bar_y - bar_height//2),
                         (progress_x, bar_y + bar_height//2), (0, 120, 255), -1)
            
            # 拖动手柄
            cv2.circle(bar, (progress_x, bar_y), 12, (0, 80, 200), -1)
            cv2.circle(bar, (progress_x, bar_y), 12, (255, 255, 255), 2)
        
        # 时间标签
        current_time = self.current_frame / self.video_fps if self.video_fps > 0 else 0
        total_time = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        time_text = f"{current_time:.1f}s / {total_time:.1f}s"
        cv2.putText(bar, time_text, (width // 2 - 80, bar_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        
        # 控制提示
        cv2.putText(bar, "Space: Play/Pause | R: Restart | Drag: Seek",
                   (bar_x_start, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        return bar
    
    def create_input_area(self, width: int, height: int) -> np.ndarray:
        """创建指令输入区域"""
        area = np.ones((height, width, 3), dtype=np.uint8) * 250
        
        # 标题
        if self.batch_annotation_mode and self.current_trajectory_idx < len(self.trajectories):
            trajectory = self.trajectories[self.current_trajectory_idx]
            title = f"Navigation Instruction (Trajectory ID: {trajectory['trajectory_id']}  Progress: {self.current_trajectory_idx+1}/{len(self.trajectories)})"
            cv2.putText(area, title, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        else:
            cv2.putText(area, "Navigation Instruction:", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        # 输入框
        input_x, input_y = 20, 50
        input_w, input_h = width - 40, 60
        
        # 输入框背景
        box_color = (255, 255, 255) if self.text_input_active else (240, 240, 240)
        cv2.rectangle(area, (input_x, input_y), (input_x + input_w, input_y + input_h),
                     box_color, -1)
        
        # 输入框边框
        if self.terminal_input_active:
            border_color = (0, 200, 0)  # 绿色 - 终端输入模式
        elif self.text_input_active:
            border_color = (0, 120, 255)  # 蓝色 - GUI输入模式
        else:
            border_color = (180, 180, 180)  # 灰色 - 未激活
        cv2.rectangle(area, (input_x, input_y), (input_x + input_w, input_y + input_h),
                     border_color, 2)
        
        # 显示文本
        if self.terminal_input_active:
            display_text = ">>> Waiting for terminal input... (Check your terminal)"
            text_color = (0, 150, 0)
        else:
            display_text = self.text_input_buffer if self.text_input_active else self.current_instruction
            if not display_text and not self.text_input_active:
                display_text = "Click here or press 'G' to enter instruction in GUI..."
                text_color = (150, 150, 150)
            else:
                text_color = (0, 0, 0)
        
        # 添加光标
        if self.text_input_active and self.text_cursor_visible:
            display_text += "|"
        
        # 文本换行处理
        max_chars_per_line = 100
        if len(display_text) > max_chars_per_line:
            line1 = display_text[:max_chars_per_line]
            line2 = display_text[max_chars_per_line:]
            cv2.putText(area, line1, (input_x + 10, input_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(area, line2, (input_x + 10, input_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        else:
            cv2.putText(area, display_text, (input_x + 10, input_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # 按钮区域
        button_y = input_y + input_h + 15
        
        # Save 按钮
        save_btn_x, save_btn_w = width - 280, 120
        save_btn_color = (0, 180, 0) if self.text_input_buffer.strip() else (150, 150, 150)
        cv2.rectangle(area, (save_btn_x, button_y), (save_btn_x + save_btn_w, button_y + 35),
                     save_btn_color, -1)
        cv2.putText(area, "Save (Enter)", (save_btn_x + 10, button_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Clear 按钮
        clear_btn_x, clear_btn_w = width - 150, 120
        cv2.rectangle(area, (clear_btn_x, button_y), (clear_btn_x + clear_btn_w, button_y + 35),
                     (200, 100, 100), -1)
        cv2.putText(area, "Clear (ESC)", (clear_btn_x + 10, button_y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 提示信息
        if self.enable_annotate:
            if self.terminal_input_active:
                hint = "Terminal Input Mode ACTIVE - Please input instruction in terminal (GUI continues running)"
                hint_color = (0, 200, 0)  # 绿色
            elif self.text_input_active:
                hint = "GUI Input Mode: Type instruction, press Enter to save, ESC to cancel"
                hint_color = (0, 120, 255)
            else:
                hint = "Input Options: [Click box / 'G' key] = GUI input  |  ['I' key] = Terminal input  |  Q: Exit"
                hint_color = (100, 100, 100)
            cv2.putText(area, hint, (20, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, hint_color, 1)
        else:
            hint = "Annotation mode not enabled | Q: Exit"
            cv2.putText(area, hint, (20, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        # 保存区域坐标
        self.ui_buttons = {
            'input_box': (input_x, input_y, input_x + input_w, input_y + input_h),
            'save_btn': (save_btn_x, button_y, save_btn_x + save_btn_w, button_y + 35),
            'clear_btn': (clear_btn_x, button_y, clear_btn_x + clear_btn_w, button_y + 35),
        }
        
        return area
    
    def integrated_mouse_callback(self, event, x, y, flags, param):
        """集成界面的鼠标回调"""
        if not hasattr(self, 'ui_regions'):
            return
        
        # 检查点击位置
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击输入框
            if hasattr(self, 'ui_buttons'):
                input_box = self.ui_buttons.get('input_box')
                if input_box:
                    x1, y1, x2, y2 = input_box
                    # 调整坐标到实际画布位置
                    progress_h = 50
                    y1_adj = y1 + self.ui_regions['input_area'][1]
                    y2_adj = y2 + self.ui_regions['input_area'][1]
                    x1_adj = x1
                    x2_adj = x2
                    
                    if x1_adj <= x <= x2_adj and y1_adj <= y <= y2_adj:
                        if self.terminal_input_active:
                            logger.info("⚠️ 终端输入模式正在运行中，请先完成终端输入")
                        else:
                            self.text_input_active = True
                            self.text_input_buffer = self.current_instruction
                        return
                
                # 检查是否点击 Save 按钮
                save_btn = self.ui_buttons.get('save_btn')
                if save_btn and self.text_input_buffer.strip():
                    x1, y1, x2, y2 = save_btn
                    y1_adj = y1 + self.ui_regions['input_area'][1]
                    y2_adj = y2 + self.ui_regions['input_area'][1]
                    if x1 <= x <= x2 and y1_adj <= y <= y2_adj:
                        self.current_instruction = self.text_input_buffer.strip()
                        self.save_annotation()
                        logger.info("✓ 指令已通过GUI输入并保存")
                        self.text_input_buffer = ""
                        self.text_input_active = False
                        return
                
                # 检查是否点击 Clear 按钮
                clear_btn = self.ui_buttons.get('clear_btn')
                if clear_btn:
                    x1, y1, x2, y2 = clear_btn
                    y1_adj = y1 + self.ui_regions['input_area'][1]
                    y2_adj = y2 + self.ui_regions['input_area'][1]
                    if x1 <= x <= x2 and y1_adj <= y <= y2_adj:
                        self.text_input_buffer = ""
                        self.text_input_active = False
                        logger.info("✓ 输入已清除")
                        return
            
            # 检查是否点击进度条
            progress_region = self.ui_regions.get('progress_bar')
            if progress_region:
                x1, y1, x2, y2 = progress_region
                if y1 <= y <= y2:
                    # 计算点击位置对应的帧
                    bar_x_start = 50
                    bar_x_end = x2 - x1 - 50
                    if bar_x_start <= x <= bar_x_end:
                        progress = (x - bar_x_start) / (bar_x_end - bar_x_start)
                        new_frame = int(progress * self.total_frames)
                        self.current_frame = max(0, min(new_frame, self.total_frames - 1))
                        
                        # 更新视频位置
                        self.rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                        if self.depth_cap:
                            self.depth_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # 拖动进度条
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            progress_region = self.ui_regions.get('progress_bar')
            if progress_region:
                x1, y1, x2, y2 = progress_region
                if y1 <= y <= y2:
                    bar_x_start = 50
                    bar_x_end = x2 - x1 - 50
                    if bar_x_start <= x <= bar_x_end:
                        progress = (x - bar_x_start) / (bar_x_end - bar_x_start)
                        new_frame = int(progress * self.total_frames)
                        self.current_frame = max(0, min(new_frame, self.total_frames - 1))
                        
                        # 更新视频位置
                        self.rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                        if self.depth_cap:
                            self.depth_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    
    def input_instruction(self, use_terminal: bool = True):
        """输入指令并保存标注
        
        Args:
            use_terminal: 是否使用终端输入（否则使用GUI输入）
        """
        if use_terminal:
            # 检查是否已经有终端输入线程在运行
            if self.terminal_input_active:
                logger.info("⚠️ 终端输入模式已经在运行中，请先完成当前输入")
                return
            
            # 启动终端输入线程
            self.terminal_input_active = True
            self.terminal_input_thread = threading.Thread(target=self._terminal_input_thread, daemon=True)
            self.terminal_input_thread.start()
            logger.info("✓ 终端输入线程已启动，GUI继续运行")
        else:
            # GUI 输入模式 - 激活文本框
            logger.info("✓ GUI输入模式已激活，请在界面中输入指令")
            self.text_input_active = True
            self.text_input_buffer = self.current_instruction
    
    def input_instruction_batch(self):
        """批量标注模式：输入指令并保存（包含trajectory_id）。
        
        同一条轨迹可多次调用，每次保存一条新的 episode，episode_id 自动递增。
        不自动跳转轨迹，需用 'n'/'p' 手动切换。
        """
        if not self.batch_annotation_mode:
            logger.info("❌ 不在批量标注模式")
            return
        
        if self.current_trajectory_idx >= len(self.trajectories):
            logger.info("❌ 没有可标注的轨迹")
            return
        
        trajectory = self.trajectories[self.current_trajectory_idx]
        trajectory_id = trajectory['trajectory_id']

        # 统计当前轨迹已有的标注数量（用于提示）
        existing_count = sum(
            1 for ep in self.annotate_episodes
            if ep.get("trajectory_id") == trajectory_id
        )
        
        try:
            logger.info("\n" + "="*60)
            logger.info("【批量标注 - 指令输入】")
            logger.info("="*60)
            logger.info(f"Trajectory ID: {trajectory_id}")
            logger.info(f"进度: {self.current_trajectory_idx + 1}/{len(self.trajectories)}")
            if existing_count > 0:
                logger.info(f"当前轨迹已有 {existing_count} 条指令，继续添加第 {existing_count + 1} 条")
            logger.info("请输入导航指令（英文），然后按Enter保存：")
            logger.info("（留空并按Enter可取消，输入 'n'/'p' 切换轨迹）")
            logger.info("-"*60)
            instruction = input(">>> ").strip()
            
            if not instruction:
                logger.info("⚠️ 未输入指令，取消保存（轨迹未切换）")
                return
            
            # 保存带有trajectory_id的标注（episode_id 由 save_annotation_with_trajectory 自动递增）
            self.save_annotation_with_trajectory(instruction, trajectory_id)
            
            # 标记轨迹为已标注
            trajectory['annotated'] = True
            self.save_trajectories()
            
            new_count = existing_count + 1
            logger.info("✓ 指令已保存")
            logger.info(f"   Episode ID: episode_{self.current_episode_id - 1:04d}")
            logger.info(f"   Trajectory ID: {trajectory_id}（已累计 {new_count} 条指令）")
            logger.info("提示：继续输入 'i' 可为同一轨迹添加更多指令，输入 'n'/'p' 切换轨迹")
            
        except Exception as e:
            logger.info(f"❌ 输入出错: {e}")
    
    def _terminal_input_thread(self):
        """在独立线程中处理终端输入，不阻塞GUI"""
        try:
            logger.info("\n" + "="*60)
            logger.info("【终端输入模式】")
            logger.info("="*60)
            logger.info("请输入导航指令（英文），然后按Enter保存：")
            logger.info("（留空并按Enter可取消输入）")
            logger.info("提示：GUI视频将继续播放，不受影响")
            logger.info("-"*60)
            instruction = input(">>> ").strip()
            
            with self.terminal_input_lock:
                if not instruction:
                    logger.info("❌ 未输入指令，取消保存")
                else:
                    self.current_instruction = instruction
                    # 保存标注
                    self.save_annotation()
                    logger.info("✓ 指令已通过终端输入并保存")
        except Exception as e:
            logger.info(f"❌ 终端输入出错: {e}")
        finally:
            self.terminal_input_active = False
            logger.info("-"*60)
    
    def _terminal_command_listener(self):
        """持续监听终端命令，不阻塞GUI"""
        logger.info("\n" + "="*60)
        logger.info("【终端命令监听已启动】")
        logger.info("="*60)
        logger.info("可用命令:")
        logger.info("  'i' + Enter: 为当前轨迹添加一条指令并保存（可重复，不切换轨迹）")
        logger.info("  'n' + Enter: 下一条轨迹（批量标注模式）")
        logger.info("  'p' + Enter: 上一条轨迹（批量标注模式）")
        logger.info("  'sample' + Enter: 对当前起点/途径点/终点进行噪声采样，批量生成多条路径")
        logger.info("  'b' + Enter: 采样完成后启动批量渲染和标注（等效于在地图窗口按 'B' 键）")
        logger.info("  'q' + Enter: 退出程序")
        logger.info("  'h' + Enter: 显示帮助")
        logger.info("提示：GUI和终端可以同时操作，互不影响")
        logger.info("-"*60 + "\n")
        
        while not self.should_exit:
            try:
                # 使用 stdin_lock：防止与 P 键后台线程（show_sample_config_dialog）争抢 stdin
                with self._stdin_lock:
                    cmd = input("Terminal> ").strip().lower()

                if cmd == 'q':
                    logger.info("✓ 收到退出命令，正在关闭程序...")
                    self.should_exit = True
                    break
                elif cmd == 'i':
                    if self.enable_annotate:
                        if self.batch_annotation_mode:
                            # 批量标注模式下的输入
                            logger.info("✓ 触发指令输入（批量标注模式）")
                            print("触发指令输入（批量标注模式）")
                            self.input_instruction_batch()
                        elif self.path_found and self.last_render_dir is not None:
                            logger.info("✓ 触发指令输入")
                            print("触发指令输入")
                            self.input_instruction(use_terminal=True)
                        else:
                            logger.info("❌ 请先完成路径规划和渲染!")
                    else:
                        logger.info("❌ 标注功能未启用，请使用 --annotate 参数")
                elif cmd == 'n':
                    if self.batch_annotation_mode:
                        logger.info("✓ 切换到下一条轨迹")
                        print("切换到下一条轨迹")
                        self.next_trajectory()
                    else:
                        logger.info("❌ 不在批量标注模式")
                elif cmd == 'p':
                    if self.batch_annotation_mode:
                        logger.info("✓ 切换到上一条轨迹")
                        print('切换到上一条轨迹')
                        self.previous_trajectory()
                    else:
                        logger.info("❌ 不在批量标注模式")
                elif cmd == 'sample':
                    if self.enable_annotate:
                        if self._sampling_active:
                            logger.info("⚠️ 采样正在进行中，请等待完成后再触发")
                        elif self.start_set and self.goal_set:
                            # 终端命令监听线程本身就持有 _stdin_lock，可以直接调用 show_sample_config_dialog
                            logger.info("✓ 触发路径噪声采样")
                            cfg = self.show_sample_config_dialog()
                            if cfg is not None:
                                # 采样计算在当前线程内串行执行（终端命令监听线程），不再嵌套新线程
                                self._sampling_active = True
                                try:
                                    self.sample_and_save_trajectories(
                                        num_samples=cfg['num_samples'],
                                        noise_level=cfg['noise_level'],
                                        noise_type=cfg['noise_type'],
                                    )
                                finally:
                                    self._sampling_active = False
                                    logger.info("✓ 采样完成！请点击地图窗口并按 'B'，或在终端输入 'b' 开始批量渲染")
                        else:
                            logger.info("❌ 请先在 GUI 中设置起点和终点！")
                    else:
                        logger.info("❌ 标注功能未启用，请使用 --annotate 参数")
                elif cmd == 'b':
                    # 'b' 命令：在终端触发批量渲染（等效于在地图窗口中按 'B' 键）
                    if self.enable_annotate:
                        if self._sampling_active:
                            logger.info("⚠️ 采样仍在进行中，请等待完成后再输入 'b'")
                        else:
                            unannotated = [t for t in self.trajectories if not t.get("annotated", False)]
                            if unannotated:
                                logger.info(f"✓ 收到批量渲染请求，共 {len(unannotated)} 条未标注轨迹")
                                self._batch_render_requested = True
                            else:
                                logger.info("❌ 没有未标注的轨迹")
                    else:
                        logger.info("❌ 标注功能未启用，请使用 --annotate 参数")
                elif cmd == 'h' or cmd == 'help':
                    logger.info("\n" + "-"*60)
                    if self.batch_annotation_mode:
                        logger.info("批量标注模式命令:")
                        logger.info("  'i' + Enter: 为当前轨迹添加一条指令并保存（可重复，不切换轨迹）")
                        logger.info("  'n' + Enter: 跳转到下一条轨迹")
                        logger.info("  'p' + Enter: 回到上一条轨迹")
                        logger.info("  'q' + Enter: 退出程序")
                    else:
                        logger.info("可用命令:")
                        logger.info("  'i' + Enter: 触发指令输入（需要先完成路径规划）")
                        logger.info("  'n' + Enter: 下一条轨迹（需要在批量标注模式）")
                        logger.info("  'p' + Enter: 上一条轨迹（需要在批量标注模式）")
                        logger.info("  'sample' + Enter: 噪声采样（需要先在GUI设置起点/终点）")
                        logger.info("  'b' + Enter: 启动批量渲染和标注（采样完成后使用）")
                        logger.info("  'q' + Enter: 退出程序")
                    logger.info("  'h' + Enter: 显示此帮助信息")
                    logger.info("-"*60 + "\n")
                elif cmd == '':
                    # 空命令，忽略
                    continue
                else:
                    logger.info(f"❌ 未知命令: '{cmd}'，输入 'h' 查看帮助")
            except EOFError:
                # 终端关闭
                break
            except Exception as e:
                logger.info(f"❌ 命令处理出错: {e}")
        
        logger.info("✓ 终端命令监听已停止")
    
    def start_terminal_command_listener(self):
        """启动终端命令监听线程"""
        if not self.terminal_command_enabled:
            self.terminal_command_enabled = True
            self.terminal_command_thread = threading.Thread(
                target=self._terminal_command_listener, 
                daemon=True
            )
            self.terminal_command_thread.start()
    
    def save_annotation(self):
        """保存当前episode的标注到JSON文件"""
        if not self.start_set or not self.goal_set:
            logger.info("错误: 起点或终点未设置")
            return
        
        # 读取当前保存的相机位姿JSON
        mode = self.last_saved_mode if hasattr(self, 'last_saved_mode') else "discrete"

        # 批量模式：优先使用当前轨迹的专属 temp 文件
        if self.batch_annotation_mode and self.current_trajectory_idx < len(self.trajectories):
            cur_traj = self.trajectories[self.current_trajectory_idx]
            tid = cur_traj["trajectory_id"]
            traj_json = os.path.join(
                self.output_dir, f"camera_poses_traj_{tid:04d}.json")
            with self.render_lock:
                render_info = self.render_status.get(tid, {})
            camera_json = render_info.get("camera_json", "")
            if not camera_json or not os.path.exists(camera_json):
                camera_json = traj_json if os.path.exists(traj_json) else ""
        else:
            camera_json = ""

        # 非批量 / fallback：按模式使用通用文件
        if not camera_json:
            camera_json = os.path.join(
                self.output_dir,
                "camera_poses_discrete.json" if mode == "discrete" else "camera_poses.json")
            
            if not os.path.exists(camera_json):
                # 创建相机位姿文件
                with open(camera_json, 'w') as f:
                    json.dump({
                        "mode": mode,
                        "num_cameras": 0,
                        "cameras": [],
                    }, f, indent=2, ensure_ascii=False)
        
        if not os.path.exists(camera_json):
            logger.info(f"错误: 相机位姿文件不存在: {camera_json}")
            return
        
        # 读取相机位姿数据
        with open(camera_json, 'r') as f:
            camera_data = json.load(f)
        
        # 转换起点和终点为世界坐标
        start_world = self.grid_to_world(self.start_point[0], self.start_point[1], 0.0)
        goal_world = self.grid_to_world(self.goal_point[0], self.goal_point[1], 0.0)
        
        # 构建episode数据
        episode_data = {
            "episode_id": f"episode_{self.current_episode_id:04d}",
            "instruction": self.current_instruction,
            "start_position": start_world.tolist(),
            "start_yaw": float(self.start_yaw),
            "goal_position": goal_world.tolist(),
            "goal_yaw": float(self.goal_yaw),
            "mode": mode,
            "num_cameras": camera_data["num_cameras"],
            "cameras": camera_data.get("cameras", []),
            "render_dir": self.last_render_dir,
            "scene_bounds": {
                "min": self.min_pt.tolist(),
                "max": self.max_pt.tolist()
            },
            "grid_resolution": float(self.grid_resolution),
        }
        
        # 添加动作序列（如果是discrete模式）
        if mode == "discrete" and "actions" in camera_data:
            episode_data["actions"] = camera_data["actions"]
            episode_data["num_actions"] = camera_data["num_actions"]
        
        # 添加途径点（如果有）
        if self.waypoints:
            waypoints_world = []
            for wp in self.waypoints:
                wp_world = self.grid_to_world(wp[0], wp[1], 0.0)
                waypoints_world.append(wp_world.tolist())
            episode_data["waypoints"] = waypoints_world
        
        self.annotate_episodes.append(episode_data)
        
        # 保存到JSON文件
        annotate_json_path = os.path.join(self.output_dir, "annotate_episodes.json")
        with open(annotate_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "num_episodes": len(self.annotate_episodes),
                "episodes": self.annotate_episodes
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ 标注已保存到: {annotate_json_path}")
        logger.info(f"  Episode ID: {episode_data['episode_id']}")
        logger.info(f"  指令: {self.current_instruction}")
        
        # 更新episode ID
        self.current_episode_id += 1
        
        # 清空当前指令
        self.current_instruction = ""
    
    def save_annotation_with_trajectory(self, instruction: str, trajectory_id: int):
        """保存带有trajectory_id的标注（批量标注模式）
        
        Args:
            instruction: 导航指令
            trajectory_id: 轨迹ID
        """
        if not self.start_set or not self.goal_set:
            logger.info("错误: 起点或终点未设置")
            return
        
        # 读取当前保存的相机位姿JSON
        # 批量模式下优先使用轨迹专属 temp 文件，fallback 到通用文件
        mode = "discrete"
        traj_camera_json = os.path.join(
            self.output_dir, f"camera_poses_traj_{trajectory_id:04d}.json")
        generic_camera_json = os.path.join(self.output_dir, "camera_poses_discrete.json")

        # 从 render_status 中取已记录的路径（最可靠）
        with self.render_lock:
            render_info = self.render_status.get(trajectory_id, {})
        camera_json = render_info.get("camera_json", "")

        # 按优先级确定实际可用的文件
        if not camera_json or not os.path.exists(camera_json):
            if os.path.exists(traj_camera_json):
                camera_json = traj_camera_json
            elif os.path.exists(generic_camera_json):
                camera_json = generic_camera_json
            else:
                logger.info(
                    f"❌ 相机位姿文件不存在，已查找路径：\n"
                    f"   {traj_camera_json}\n"
                    f"   {generic_camera_json}\n"
                    f"请确认批量渲染已完成。")
                return


        
        # 读取相机位姿数据
        with open(camera_json, 'r') as f:
            camera_data = json.load(f)
        
        # 转换起点和终点为世界坐标
        start_world = self.grid_to_world(self.start_point[0], self.start_point[1], 0.0)
        goal_world = self.grid_to_world(self.goal_point[0], self.goal_point[1], 0.0)
        
        # 构建episode数据
        episode_data = {
            "episode_id": f"episode_{self.current_episode_id:04d}",
            "trajectory_id": trajectory_id,  # 添加trajectory_id
            "instruction": instruction,
            "start_position": start_world.tolist(),
            "start_yaw": float(self.start_yaw),
            "goal_position": goal_world.tolist(),
            "goal_yaw": float(self.goal_yaw),
            "mode": mode,
            "num_cameras": camera_data["num_cameras"],
            "cameras": camera_data.get("cameras", []),
            "render_dir": self.last_render_dir,
            "scene_bounds": {
                "min": self.min_pt.tolist(),
                "max": self.max_pt.tolist()
            },
            "grid_resolution": float(self.grid_resolution),
        }
        
        # 添加动作序列（如果是discrete模式）
        if "actions" in camera_data:
            episode_data["actions"] = camera_data["actions"]
            episode_data["num_actions"] = camera_data["num_actions"]
        
        # 添加途径点（如果有）
        if self.waypoints:
            waypoints_world = []
            for wp in self.waypoints:
                wp_world = self.grid_to_world(wp[0], wp[1], 0.0)
                waypoints_world.append(wp_world.tolist())
            episode_data["waypoints"] = waypoints_world
        
        self.annotate_episodes.append(episode_data)
        
        # 保存到JSON文件
        annotate_json_path = os.path.join(self.output_dir, "annotate_episodes.json")
        with open(annotate_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "num_episodes": len(self.annotate_episodes),
                "episodes": self.annotate_episodes
            }, f, indent=2, ensure_ascii=False)
        
        # print
        print(f"已保存新的标注: episode_id: {episode_data['episode_id']}, trajectory_id: {trajectory_id}, instruction: {instruction}")
        
        # 更新episode ID
        self.current_episode_id += 1
    
    def append_episode_to_json(self, episode: Dict, output_path: str):
        """增量追加单个 episode（与 main_sample.append_episode_to_json 行为一致）。
        若 episode 中无 episode_id，则按当前列表长度自动生成 episode_XXXX。
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                episodes = data.get("episodes", [])
            except Exception as e:
                logger.info(f"警告: 读取已有文件失败: {e}，将新建")
                episodes = []
        else:
            episodes = []
        ep = dict(episode)
        if "episode_id" not in ep:
            ep["episode_id"] = f"episode_{len(episodes):04d}"
        episodes.append(ep)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {"num_episodes": len(episodes), "episodes": episodes},
                f, indent=2, ensure_ascii=False,
            )
    
    def save_camera_poses_json(self, rotations: List[np.ndarray], 
                               translations: List[np.ndarray], num_poses: int,
                               actions: Optional[List[int]] = None,
                               mode: str = "continuous",
                               output_path: Optional[str] = None,
                               full_path: Optional[List[Tuple[int, int]]] = None,
                               start_yaw: float = 0.0,
                               goal_yaw: float = 0.0):
        """保存相机位姿到 JSON（与 main_sample.run_single_path 返回的 result 字段对齐）。
        
        默认 discrete 写入 camera_poses_discrete.json 时，同时增量写入
        camera_poses_discrete_episodes.json（避免中断丢数据；渲染仍读单次主文件）。
        """
        if output_path is None:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 根据模式选择文件名
            if mode == "discrete":
                json_path = output_dir / "camera_poses_discrete.json"
            else:
                json_path = output_dir / "camera_poses.json"
        else:
            json_path = Path(output_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Course default: align rendered frames with GenieSim G2 head_front_Camera.
        width = int(getattr(self, "camera_width", 640))
        height = int(getattr(self, "camera_height_px", 400))
        fx = float(getattr(self, "camera_fx", 317.25))
        fy = float(getattr(self, "camera_fy", 314.72))
        cx = float(getattr(self, "camera_cx", width / 2.0))
        cy = float(getattr(self, "camera_cy", height / 2.0))
        
        camera_data: Dict = {
            "mode": mode,
            "num_cameras": num_poses,
            "cameras": []
        }
        
        if mode == "discrete" and actions is not None:
            camera_data["num_actions"] = len(actions)
            camera_data["actions"] = actions
            camera_data["action_definitions"] = {
                "1": "forward 0.25m",
                "2": "turn_left 15deg",
                "3": "turn_right 15deg"
            }
        
        for i in range(num_poses):
            camera = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "R": rotations[i].tolist(),
                "T": translations[i].tolist()
            }
            camera_data["cameras"].append(camera)
        
        if full_path is not None and len(full_path) >= 2:
            sx, sy = full_path[0]
            gx, gy = full_path[-1]
            camera_data["start_grid"] = [int(sx), int(sy)]
            camera_data["goal_grid"] = [int(gx), int(gy)]
            camera_data["start_yaw"] = float(start_yaw)
            camera_data["goal_yaw"] = float(goal_yaw)
            camera_data["start_world"] = self.grid_to_world(sx, sy, 0.0).tolist()
            camera_data["goal_world"] = self.grid_to_world(gx, gy, 0.0).tolist()
            camera_data["path_length"] = float(
                ShortestPathPlanner.path_length(full_path, self.grid_resolution))
            camera_data["num_waypoints"] = len(full_path)

        if mode == "discrete":
            debug_info = getattr(self, "_last_discrete_debug", None)
            if debug_info is not None:
                camera_data["debug"] = debug_info

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(camera_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n相机位姿已保存到: {json_path}")
        logger.info(f"生成了 {len(rotations)} 个相机位姿")
        if mode == "discrete" and actions is not None:
            logger.info(f"动作序列已保存 ({len(actions)} 个动作)")
        
        self.last_saved_mode = mode
        
        # 交互式默认 discrete 主文件：按 main_sample 批量逻辑增量备份
        if (
            mode == "discrete"
            and full_path is not None
            and len(full_path) >= 2
            and output_path is None
        ):
            log_path = os.path.join(self.output_dir, "camera_poses_discrete_episodes.json")
            try:
                self.append_episode_to_json(dict(camera_data), log_path)
            except Exception as e:
                logger.info(f"警告: 增量写入 camera_poses_discrete_episodes.json 失败: {e}")
        
        return str(json_path)
    
    # ── 调色板：最多支持 20 条轨迹，超出则循环 ──────────────────────────────────
    _TRAJ_PALETTE = [
        (  0, 200, 255), (255, 140,   0), ( 80, 255,  80), (255,  80, 200),
        (255, 220,   0), ( 80, 160, 255), (255,  80,  80), (  0, 220, 180),
        (200, 100, 255), (180, 255,   0), (  0, 180, 255), (255, 180,  60),
        (100, 255, 180), (255,  60, 120), ( 60, 200, 255), (220, 180, 255),
        (255, 255, 100), ( 80, 255, 220), (200,  60, 255), (255, 200,  80),
    ]

    def _get_trajectory_path(self, traj: dict) -> List[Tuple[int, int]]:
        """获取轨迹的网格路径（带缓存，地图变更时缓存被清空）。"""
        tid = traj["trajectory_id"]
        if tid in self._trajectory_paths:
            return self._trajectory_paths[tid]

        # 世界坐标 → 网格坐标
        def w2g(wx, wy):
            gx = int((wx - self.min_pt[0]) / self.grid_resolution)
            gy = int((self.max_pt[1] - wy) / self.grid_resolution)
            return (gx, gy)

        sp = traj["start_position"]
        gp = traj["goal_position"]
        start_g = w2g(sp[0], sp[1])
        goal_g  = w2g(gp[0], gp[1])

        waypoints_g = []
        for wp in traj.get("waypoints", []):
            waypoints_g.append(w2g(wp[0], wp[1]))

        planner = self._get_path_planner()
        sequence = [start_g] + waypoints_g + [goal_g]
        full_path: List[Tuple[int, int]] = []
        for i in range(len(sequence) - 1):
            seg = planner.plan(sequence[i], sequence[i + 1])["result"]
            if not seg:
                full_path = []
                break
            full_path = full_path + (seg if not full_path else seg[1:])

        self._trajectory_paths[tid] = full_path
        return full_path

    def draw_all_trajectories(self, image: np.ndarray):
        """将已保存的轨迹叠加绘制到 image 上。

        - 受 self.show_trajectories 开关控制（'h' 键切换）
        - 只显示最近 self.trajectory_display_count 条
        - 每条轨迹用调色板中的不同颜色区分，并标注编号及起止点
        """
        if not self.trajectories or self.expanded_traversability is None:
            return

        n_total = len(self.trajectories)

        # 隐藏模式：仅在右上角显示一行提示，不绘制路径
        if not self.show_trajectories:
            cv2.putText(image,
                        f"Trajectories HIDDEN (H to show) [{n_total} saved]",
                        (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
            return

        # 只取最近 trajectory_display_count 条
        visible = self.trajectories[-self.trajectory_display_count:]
        offset  = n_total - len(visible)  # 全局编号偏移

        arrow_len = 25
        for local_idx, traj in enumerate(visible):
            idx   = local_idx + offset          # 全局编号（1-based显示用 idx+1）
            color = self._TRAJ_PALETTE[idx % len(self._TRAJ_PALETTE)]

            # ── 计算网格坐标 ─────────────────────────────────────────────────
            def w2g(wx, wy):
                gx = int((wx - self.min_pt[0]) / self.grid_resolution)
                gy = int((self.max_pt[1] - wy) / self.grid_resolution)
                return (gx, gy)

            sp = traj["start_position"]
            gp = traj["goal_position"]
            sg = w2g(sp[0], sp[1])
            gg = w2g(gp[0], gp[1])

            # ── 绘制路径（懒计算，带缓存） ────────────────────────────────────
            try:
                path = self._get_trajectory_path(traj)
                if path and len(path) > 1:
                    pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], False, color, 2, cv2.LINE_AA)
            except Exception:
                pass

            label = str(idx + 1)

            # ── 起点：实心圆 + 朝向箭头 + 编号 ──────────────────────────────
            cv2.circle(image, sg, 8, color, -1)
            cv2.circle(image, sg, 8, (0, 0, 0), 1)
            yaw_s = traj.get("start_yaw", 0.0)
            ex = int(sg[0] + arrow_len * np.cos(yaw_s))
            ey = int(sg[1] - arrow_len * np.sin(yaw_s))
            cv2.arrowedLine(image, sg, (ex, ey), color, 2, tipLength=0.35)
            cv2.putText(image, label, (sg[0] + 9, sg[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # ── 终点：空心圆（双圆）+ 朝向箭头 + 编号 ────────────────────────
            cv2.circle(image, gg, 8, color, 2)
            cv2.circle(image, gg, 4, color, -1)
            yaw_g = traj.get("goal_yaw", 0.0)
            ex = int(gg[0] + arrow_len * np.cos(yaw_g))
            ey = int(gg[1] - arrow_len * np.sin(yaw_g))
            cv2.arrowedLine(image, gg, (ex, ey), color, 2, tipLength=0.35)
            cv2.putText(image, label, (gg[0] + 9, gg[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # ── 图例（只列出 visible 条，顶部注明总数） ──────────────────────────
        nv = len(visible)
        legend_x = 10
        legend_y = image.shape[0] - 10 - nv * 20 - 20  # 多留一行给 header
        legend_y = max(legend_y, 10)
        cv2.rectangle(image,
                      (legend_x - 4, legend_y - 16),
                      (legend_x + 190, legend_y + nv * 20 + 6),
                      (30, 30, 30), -1)
        header = (f"Showing {nv}/{n_total} trajs  [H=hide]"
                  if n_total <= self.trajectory_display_count
                  else f"Showing last {nv}/{n_total}  [H=hide]")
        cv2.putText(image, header, (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        for local_idx, traj in enumerate(visible):
            idx   = local_idx + offset
            color = self._TRAJ_PALETTE[idx % len(self._TRAJ_PALETTE)]
            tid   = traj["trajectory_id"]
            y     = legend_y + 16 + local_idx * 20
            annotated = traj.get("annotated", False)
            mark = "v" if annotated else "-"
            cv2.rectangle(image, (legend_x, y - 10), (legend_x + 14, y + 4), color, -1)
            cv2.putText(image, f"[{mark}] #{idx+1:02d}  id={tid:04d}",
                        (legend_x + 18, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def draw_points_and_orientations(self, image: np.ndarray):
        """绘制起点、终点、途径点及其朝向"""
        arrow_length = 30
        
        # 绘制起点及朝向
        if self.start_set:
            cv2.circle(image, self.start_point, 8, (0, 255, 0), -1)  # 绿色起点
            cv2.circle(image, self.start_point, 8, (0, 0, 0), 2)  # 黑色边框
            
            # 绘制朝向箭头
            # 注意：yaw是世界坐标系，绘制时Y需要翻转（世界Y向上，屏幕Y向下）
            end_x = int(self.start_point[0] + arrow_length * np.cos(self.start_yaw))
            end_y = int(self.start_point[1] - arrow_length * np.sin(self.start_yaw))  # 翻转Y
            cv2.arrowedLine(image, self.start_point, (end_x, end_y),
                           (0, 200, 0), 3, tipLength=0.3)
        
        # 绘制途径点
        for i, waypoint in enumerate(self.waypoints):
            cv2.circle(image, waypoint, 7, (0, 255, 255), -1)
            cv2.circle(image, waypoint, 7, (0, 0, 0), 2)
            cv2.putText(image, str(i+1), (waypoint[0]-5, waypoint[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 绘制终点及朝向
        if self.goal_set:
            cv2.circle(image, self.goal_point, 8, (0, 0, 255), -1)  # 红色终点
            cv2.circle(image, self.goal_point, 8, (0, 0, 0), 2)  # 黑色边框
            
            # 绘制朝向箭头
            # 注意：yaw是世界坐标系，绘制时Y需要翻转（世界Y向上，屏幕Y向下）
            end_x = int(self.goal_point[0] + arrow_length * np.cos(self.goal_yaw))
            end_y = int(self.goal_point[1] - arrow_length * np.sin(self.goal_yaw))  # 翻转Y
            cv2.arrowedLine(image, self.goal_point, (end_x, end_y),
                           (0, 0, 200), 3, tipLength=0.3)
    
    def update_display_with_orientation(self):
        """更新显示，包含所有点和朝向"""
        vis_image = self.create_base_visualization_image(use_color_map=False)
        # 先叠加所有已保存的历史轨迹
        self.draw_all_trajectories(vis_image)
        self.draw_points_and_orientations(vis_image)
        
        # 添加提示信息
        if self.dragging_start:
            cv2.putText(vis_image, "Dragging: Set start orientation...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.dragging_goal:
            cv2.putText(vis_image, "Dragging: Set goal orientation...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.key_s_pressed:
            cv2.putText(vis_image, "'S' Key Active: Click & drag to set START point (Green)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.key_g_pressed:
            cv2.putText(vis_image, "'G' Key Active: Click & drag to set GOAL point (Red)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.waypoint_mode:
            cv2.putText(vis_image, "WAYPOINT MODE: Click to add waypoint (Yellow)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.start_set and self.goal_set:
            cv2.putText(vis_image, "Ready! Press SPACE/'D' to search",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis_image, "Press 'S' for Start, 'G' for Goal",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("A* Path Planning", vis_image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        # 检查边界
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return
        
        # 处理鼠标移动事件（用于拖拽）
        if event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_start and self.drag_start_pos is not None:
                # 计算从起点到当前鼠标位置的方向
                dx = x - self.drag_start_pos[0]
                dy = y - self.drag_start_pos[1]
                if abs(dx) > 3 or abs(dy) > 3:  # 最小拖拽距离
                    # 注意：屏幕Y轴向下，世界Y轴向上，需要翻转dy
                    self.start_yaw = np.arctan2(-dy, dx)
                    # 实时更新显示
                    self.update_display_with_orientation()
            elif self.dragging_goal and self.drag_start_pos is not None:
                # 计算从终点到当前鼠标位置的方向
                dx = x - self.drag_start_pos[0]
                dy = y - self.drag_start_pos[1]
                if abs(dx) > 3 or abs(dy) > 3:  # 最小拖拽距离
                    # 注意：屏幕Y轴向下，世界Y轴向上，需要翻转dy
                    self.goal_yaw = np.arctan2(-dy, dx)
                    # 实时更新显示
                    self.update_display_with_orientation()
            return
        
        # 鼠标释放事件
        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging_start:
                self.dragging_start = False
                if self.drag_start_pos is not None:
                    dx = x - self.drag_start_pos[0]
                    dy = y - self.drag_start_pos[1]
                    if abs(dx) > 3 or abs(dy) > 3:
                        # 注意：屏幕Y轴向下，世界Y轴向上，需要翻转dy
                        self.start_yaw = np.arctan2(-dy, dx)
                        logger.info(f"起点朝向设置为: {np.degrees(self.start_yaw):.1f}°")
                self.drag_start_pos = None
                # 重置键盘状态
                self.key_s_pressed = False
                self.update_display_with_orientation()
            elif self.dragging_goal:
                self.dragging_goal = False
                if self.drag_start_pos is not None:
                    dx = x - self.drag_start_pos[0]
                    dy = y - self.drag_start_pos[1]
                    if abs(dx) > 3 or abs(dy) > 3:
                        # 注意：屏幕Y轴向下，世界Y轴向上，需要翻转dy
                        self.goal_yaw = np.arctan2(-dy, dx)
                        logger.info(f"终点朝向设置为: {np.degrees(self.goal_yaw):.1f}°")
                self.drag_start_pos = None
                # 重置键盘状态
                self.key_g_pressed = False
                self.update_display_with_orientation()
            return
        
        # 检查是否在原始可通行区域（只对按下事件检查）
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
            if self.original_traversability[y, x] == 0:
                logger.info("无效点击: 不在可通行区域!")
                return
            
            # 警告如果在扩展障碍区域
            if self.expanded_traversability[y, x] == 0:
                logger.info("警告: 点击在扩展障碍区域(0.3m缓冲), 该点可能无法被A*搜索到达")
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 如果在途径点模式，左键添加途径点
            if self.waypoint_mode:
                self.waypoints.append((x, y))
                self.path_found = False
                self.current_path = []
                logger.info(f"途径点{len(self.waypoints)}设置在: ({x}, {y})")
                self.waypoint_mode = False  # 退出途径点模式
                
                # 更新显示
                self.update_display_with_orientation()
                vis_image = self.create_base_visualization_image(use_color_map=False)
                self.draw_points_and_orientations(vis_image)
                cv2.putText(vis_image, f"Waypoint {len(self.waypoints)} added. Press 'c' for more or 'r' to reset",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("A* Path Planning", vis_image)
            # 按住 's' 键 + 左键：设置起点
            elif self.key_s_pressed:
                self.start_point = (x, y)
                self.start_set = True
                self.start_yaw = 0.0  # 默认朝向
                self.path_found = False
                self.current_path = []
                self.dragging_start = True
                self.drag_start_pos = (x, y)
                logger.info(f"起点设置在: ({x}, {y}) - 拖拽鼠标设置朝向")
            # 按住 'g' 键 + 左键：设置终点
            elif self.key_g_pressed:
                self.goal_point = (x, y)
                self.goal_set = True
                self.goal_yaw = 0.0  # 默认朝向
                self.path_found = False
                self.current_path = []
                self.dragging_goal = True
                self.drag_start_pos = (x, y)
                logger.info(f"终点设置在: ({x}, {y}) - 拖拽鼠标设置朝向")
                
                if not self.start_set:
                    logger.info("警告: 起点未设置! 请先按住 's' 键并点击设置起点")
            else:
                # 提示用户需要配合键盘
                logger.info("提示: 按住 's' 键 + 左键拖拽设置起点，按住 'g' 键 + 左键拖拽设置终点")
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键添加途径点
            self.waypoints.append((x, y))
            self.path_found = False
            self.current_path = []
            logger.info(f"途径点{len(self.waypoints)}设置在: ({x}, {y})")
            
            # 更新显示
            self.update_display_with_orientation()
    
    def run_interactive(self):
        """运行交互式可视化界面"""
        logger.info("\n=== 交互式A*路径规划 ===")
        logger.info("打开交互窗口...")
        logger.info("显示: 彩色点云top-down地图")
        logger.info("\n【设置起点和终点】")
        logger.info("  按住 's' 键 + 左键拖拽: 设置起点和朝向 (绿色+箭头)")
        logger.info("  按住 'g' 键 + 左键拖拽: 设置终点和朝向 (红色+箭头)")
        logger.info("  'c'键 + 左键点击: 添加途径点 (黄色)")
        logger.info("\n【路径规划】")
        logger.info("  空格键: 开始A*搜索 (continuous模式)")
        logger.info("  'd'键: 开始A*搜索 (discrete模式)")
        logger.info("  'r'键: 清除所有途径点")
        logger.info("  'e'键: 进入地图编辑模式")
        logger.info("  'x'键: 进入地图裁剪模式")
        if self.enable_annotate:
            logger.info("\n【批量标注功能】")
            logger.info("  'A'键: 保存当前路径为轨迹（可多次使用）")
            logger.info("  'B'键: 开始批量标注模式（渲染并依次标注所有轨迹）")
            logger.info("  批量标注模式中:")
            logger.info("    终端输入 'i': 为当前轨迹添加指令")
            logger.info("    终端输入 'n': 跳到下一条轨迹")
            logger.info("    终端输入 'p': 回到上一条轨迹")
            logger.info("\n【路径噪声采样功能】")
            logger.info("  'P'键: 对当前起点/途径点/终点进行噪声采样，批量生成多条路径")
            logger.info("    - 在终端中配置采样数量、噪声强度和噪声类型")
            logger.info("    - 采样结果自动保存为轨迹，可用 'B' 键批量渲染标注")
            logger.info("\n【随机起点/终点/Waypoint 采样功能】")
            logger.info("  'F'键: 固定起点，随机采样多个终点，每个终点生成一条轨迹")
            logger.info("    - 需先设置起点（'S'键+拖拽）")
            logger.info("    - 在终端中配置终点数量和最小距离")
            logger.info("  'V'键: 固定终点，随机采样多个起点，每个起点生成一条轨迹")
            logger.info("    - 需先设置终点（'G'键+拖拽）")
            logger.info("    - 在终端中配置起点数量和最小距离")
            logger.info("  'W'键: 固定起点，随机采样 Waypoints，规划环形路径并返回起点附近")
            logger.info("    - 需先设置起点（'S'键+拖拽）")
            logger.info("    - 在终端中配置 waypoint 数量、采样半径和返回半径")
            logger.info("    - 保存的轨迹包含 waypoints 和 path_waypoints 字段")
        if self.enable_annotate or self.enable_render:
            logger.info("\n【渲染完成后的集成界面操作】")
            logger.info("  空格键: 播放/暂停视频")
            logger.info("  鼠标拖动进度条: 调整视频位置")
            logger.info("\n【指令输入方式（二选一）】")
            logger.info("  方式1 - GUI输入: 点击输入框 或 按'g'键，在界面中输入，按Enter保存")
            logger.info("  方式2 - 终端输入: 按'i'键 或 在终端输入'i'+Enter，在终端中输入指令")
            logger.info("  ESC: 取消输入")
        logger.info("\n【退出程序】")
        logger.info("  方式1: 在GUI中按 'q' 或 ESC")
        logger.info("  方式2: 在终端中输入 'q' + Enter")
        
        # 创建窗口
        cv2.namedWindow("A* Path Planning", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("A* Path Planning", 1200, 800)
        cv2.setMouseCallback("A* Path Planning", self.mouse_callback)
        
        # 提前启动终端命令监听线程（在 B 键之前就可以响应 'b'/'sample' 等命令）
        self.start_terminal_command_listener()
        
        # 初始显示 - 使用彩色点云地图
        initial_display = self.create_base_visualization_image(use_color_map=False)
        
        cv2.putText(initial_display, "=== Navigation Path Planning Interface ===",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(initial_display, "1. Hold 'S' + Left-click & drag: Start point + orientation (Green)",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(initial_display, "2. Hold 'G' + Left-click & drag: Goal point + orientation (Red)",
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(initial_display, "3. Press 'C' then left-click: Add Waypoint (Yellow)",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(initial_display, "Commands:",
                   (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        help_text = "SPACE: Continuous | D: Discrete | R: Clear | E: Edit | X: Crop"
        cv2.putText(initial_display, help_text,
                   (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.enable_render or self.enable_annotate:
            cv2.putText(initial_display, "After planning: Integrated view with video & annotation",
                       (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        cv2.putText(initial_display, "Press Q or ESC to exit",
                   (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("A* Path Planning", initial_display)
        
        # 事件循环
        while True:
            # 检查是否需要退出
            if self.should_exit:
                logger.info("✓ 收到退出信号，正在关闭界面...")
                break

            # ── 后台线程请求主线程刷新 cv2 显示 ──────────────────────────
            if self._needs_display_update:
                self._needs_display_update = False
                self.update_display_with_orientation()

            # ── 终端 'b' 命令请求启动批量渲染（与 B 键等效）──────────────
            if self._batch_render_requested:
                self._batch_render_requested = False
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样仍在进行中，请等待采样完成后再启动批量渲染")
                    else:
                        unannotated = [t for t in self.trajectories if not t.get("annotated", False)]
                        if unannotated:
                            logger.info(f"\n{'='*60}")
                            logger.info("进入批量标注模式（由终端 'b' 命令触发）")
                            logger.info(f"{'='*60}")
                            logger.info(f"共有 {len(unannotated)} 条未标注的轨迹")
                            self.batch_annotation_mode = True
                            self.current_trajectory_idx = 0
                            for i, t in enumerate(self.trajectories):
                                if not t.get("annotated", False):
                                    self.current_trajectory_idx = i
                                    break
                            logger.info("正在启动后台批量渲染...")
                            self.start_batch_rendering()
                            self.start_batch_annotation()
                        else:
                            logger.info("❌ 没有未标注的轨迹")

            key = cv2.waitKey(30) & 0xFF
            
            # 检测 's' 和 'g' 键状态
            key_state_changed = False
            if key == ord('s') or key == ord('S'):
                self.key_s_pressed = True
                self.key_g_pressed = False
                key_state_changed = True
                logger.info("按住 'S' 键，现在点击并拖拽设置起点...")
            elif key == ord('g') or key == ord('G'):
                self.key_g_pressed = True
                self.key_s_pressed = False
                key_state_changed = True
                logger.info("按住 'G' 键，现在点击并拖拽设置终点...")
            elif key != 255:  # 任何其他按键都重置状态
                # 但不重置正在拖拽的状态
                if not self.dragging_start and not self.dragging_goal:
                    if self.key_s_pressed or self.key_g_pressed:
                        key_state_changed = True
                    self.key_s_pressed = False
                    self.key_g_pressed = False
            
            # 如果键盘状态改变，更新显示
            if key_state_changed:
                self.update_display_with_orientation()
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('a') or key == ord('A'):  # 'A'键 - 保存当前路径为trajectory
                if self.enable_annotate:
                    if self.save_current_path_as_trajectory():
                        self.save_trajectories()
                        # 清空当前路径，准备下一条
                        self.start_set = False
                        self.goal_set = False
                        self.path_found = False
                        self.current_path = []
                        self.waypoints = []
                        logger.info(f"✓ 已保存轨迹，共 {len(self.trajectories)} 条。可以继续设置下一条路径。")
                        logger.info(f"   提示：完成所有路径后，按 'B' 键开始批量标注")
                        self.update_display_with_orientation()
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
            elif key == ord('p') or key == ord('P'):  # 'P'键 - 路径噪声采样
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样正在进行中，请等待完成后再触发")
                    elif self.start_set and self.goal_set:
                        # 将【配置对话框 + 采样计算】全部放入后台线程：
                        #   - 主线程继续运行 cv2.waitKey()，窗口保持响应、不丢失焦点
                        #   - 后台线程独占 stdin（此时终端监听线程尚未启动，无竞争）
                        self._sampling_active = True
                        logger.info("✓ 请在终端中配置采样参数...")
                        def _config_and_sample():
                            try:
                                cfg = self.show_sample_config_dialog()
                                if cfg is not None:
                                    self.sample_and_save_trajectories(
                                        num_samples=cfg['num_samples'],
                                        noise_level=cfg['noise_level'],
                                        noise_type=cfg['noise_type'],
                                    )
                                else:
                                    logger.info("❌ 已取消采样")
                            except Exception as e:
                                logger.info(f"❌ 采样出错: {e}")
                            finally:
                                self._sampling_active = False
                                logger.info("✓ 采样完成！请点击地图窗口并按 'B' 键，或在终端输入 'b' 开始批量渲染")
                        self._sampling_thread = threading.Thread(target=_config_and_sample, daemon=True)
                        self._sampling_thread.start()
                    else:
                        logger.info("❌ 请先设置起点和终点！")
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
            elif key == ord('b') or key == ord('B'):  # 'B'键 - 开始批量标注模式
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样仍在进行中，请等待采样完成后再按 'B' 键")
                        continue
                    unannotated = [t for t in self.trajectories if not t.get("annotated", False)]
                    if unannotated:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"进入批量标注模式")
                        logger.info(f"{'='*60}")
                        logger.info(f"共有 {len(unannotated)} 条未标注的轨迹")
                        logger.info(f"提示：在终端输入 'i' 添加指令（可重复为同一轨迹添加多条），'n' 下一条，'p' 上一条，'q' 退出")
                        self.batch_annotation_mode = True
                        self.current_trajectory_idx = 0
                        # 找到第一个未标注的轨迹
                        for i, t in enumerate(self.trajectories):
                            if not t.get("annotated", False):
                                self.current_trajectory_idx = i
                                break
                        
                        # 启动批量渲染（后台线程）
                        logger.info("正在启动后台批量渲染...")
                        self.start_batch_rendering()
                        
                        # 开始第一条轨迹的标注
                        self.start_batch_annotation()
                    else:
                        logger.info("❌ 没有未标注的轨迹")
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
            elif key == ord(' '):  # 空格键 - continuous模式
                if self.start_set and self.goal_set:
                    logger.info("开始A*搜索 (Continuous模式)...")
                    self.run_astar_interactive(mode="continuous")
                else:
                    logger.info("请先设置起点和终点!")
            elif key == ord('d') or key == ord('D'):  # 'd'键 - discrete模式
                if self.start_set and self.goal_set:
                    logger.info("开始A*搜索 (Discrete模式)...")
                    self.run_astar_interactive(mode="discrete")
                else:
                    logger.info("请先设置起点和终点!")
            elif key == ord('c') or key == ord('C'):  # 'c'键 - 进入途径点添加模式
                self.waypoint_mode = True
                logger.info("途径点添加模式: 请点击鼠标左键设置途径点位置")
                
                # 更新显示
                vis_image = self.create_base_visualization_image(use_color_map=False)
                if self.start_set:
                    cv2.circle(vis_image, self.start_point, 8, (0, 255, 0), -1)
                    cv2.circle(vis_image, self.start_point, 8, (0, 0, 0), 2)
                
                # 绘制所有途径点
                for i, waypoint in enumerate(self.waypoints):
                    cv2.circle(vis_image, waypoint, 7, (0, 255, 255), -1)
                    cv2.circle(vis_image, waypoint, 7, (0, 0, 0), 2)
                    cv2.putText(vis_image, str(i+1), (waypoint[0]-5, waypoint[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                if self.goal_set:
                    cv2.circle(vis_image, self.goal_point, 8, (0, 0, 255), -1)
                    cv2.circle(vis_image, self.goal_point, 8, (0, 0, 0), 2)
                
                cv2.putText(vis_image, "WAYPOINT MODE: Click to add waypoint (Yellow)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("A* Path Planning", vis_image)
            elif key == ord('r') or key == ord('R'):  # 'r'键 - 重置途径点
                if self.waypoints:
                    logger.info(f"清除 {len(self.waypoints)} 个途径点")
                    self.waypoints = []
                    self.path_found = False
                    self.current_path = []
                    
                    # 更新显示
                    vis_image = self.create_base_visualization_image(use_color_map=False)
                    self.draw_points_and_orientations(vis_image)
                    cv2.putText(vis_image, "Waypoints cleared",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("A* Path Planning", vis_image)
                else:
                    logger.info("没有途径点需要清除")
            elif key == ord('e') or key == ord('E'):  # 'e'键 - 进入地图编辑模式
                if self.map_manager is not None:
                    logger.info("\n进入地图编辑模式...")
                    cv2.destroyWindow("A* Path Planning")
                    self.map_manager.start_edit_mode()
                    
                    # 编辑后更新数据
                    map_data = self.map_manager.get_map_data()
                    self.obstacle_map = map_data["obstacle_map"]
                    self.expanded_traversability = map_data["expanded_traversability"]
                    self.original_traversability = map_data["traversability_mask"]
                    self._path_planner = None  # 地图已变，重置规划器
                    self._trajectory_paths = {}
                    
                    # 重新创建窗口
                    cv2.namedWindow("A* Path Planning", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("A* Path Planning", 1200, 800)
                    cv2.setMouseCallback("A* Path Planning", self.mouse_callback)
                    
                    # 清除路径（地图已改变）
                    self.path_found = False
                    self.current_path = []
                    
                    # 更新显示
                    initial_display = self.create_base_visualization_image(use_color_map=False)
                    cv2.putText(initial_display, "Map edited. Please set new path.",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("A* Path Planning", initial_display)
            
            elif key == ord('x') or key == ord('X'):  # 'x'键 - 进入地图裁剪模式
                if self.map_manager is not None:
                    logger.info("\n进入地图裁剪模式...")
                    cv2.destroyWindow("A* Path Planning")
                    crop_success = self.map_manager.interactive_crop()
                    
                    if crop_success:
                        # 裁剪后更新数据
                        map_data = self.map_manager.get_map_data()
                        self.obstacle_map = map_data["obstacle_map"]
                        self.expanded_traversability = map_data["expanded_traversability"]
                        self.original_traversability = map_data["traversability_mask"]
                        self.point_cloud_coverage = map_data["point_cloud_coverage"]
                        self.display_image = map_data["color_projection"]
                        self.grid_width = map_data["grid_width"]
                        self.grid_height = map_data["grid_height"]
                        self.grid_resolution = map_data["grid_resolution"]
                        self._path_planner = None  # 地图已变，重置规划器
                        self._trajectory_paths = {}
                        self.min_pt = map_data["min_pt"]
                        self.max_pt = map_data["max_pt"]
                        
                        logger.info("✓ 地图裁剪完成，数据已更新")
                        
                        # 清除路径（地图已改变）
                        self.path_found = False
                        self.current_path = []
                        self.start_point_world = None
                        self.goal_point_world = None
                    else:
                        logger.info("地图裁剪已取消")
                    
                    # 重新创建窗口
                    cv2.namedWindow("A* Path Planning", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("A* Path Planning", 1200, 800)
                    cv2.setMouseCallback("A* Path Planning", self.mouse_callback)
                    
                    # 更新显示
                    initial_display = self.create_base_visualization_image(use_color_map=False)
                    if crop_success:
                        cv2.putText(initial_display, "Map cropped. Please set new path.",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(initial_display, "Crop cancelled. Ready for path planning.",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("A* Path Planning", initial_display)
                else:
                    logger.info("错误: 地图管理器未初始化")
            elif key == ord('i') or key == ord('I'):  # 'i'键 - 输入指令并保存标注（终端模式）
                if self.enable_annotate:
                    if self.path_found and self.last_render_dir is not None:
                        self.input_instruction(use_terminal=False)
                    else:
                        logger.info("请先完成路径规划和渲染!")
                else:
                    logger.info("标注功能未启用，请使用 --annotate 参数")
            elif key == ord('h') or key == ord('H'):  # 'h'键 - 切换历史轨迹显示/隐藏
                self.show_trajectories = not self.show_trajectories
                state = "显示" if self.show_trajectories else "隐藏"
                logger.info(f"历史轨迹已{state}（最近 {self.trajectory_display_count} 条）")
                self.update_display_with_orientation()
            elif key == ord('f') or key == ord('F'):  # 'F'键 - 固定起点，随机采样多个终点
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样正在进行中，请等待完成后再触发")
                    elif self.start_set:
                        self._sampling_active = True
                        logger.info("✓ 请在终端中配置随机终点采样参数...")
                        def _sample_random_goals():
                            try:
                                cfg = self.show_random_sample_config_dialog('goals')
                                if cfg is not None:
                                    self.sample_random_goals_from_start(
                                        num_goals=cfg['num_pts'],
                                        min_dist_m=cfg['min_dist'],
                                    )
                                else:
                                    logger.info("❌ 已取消")
                            except Exception as e:
                                logger.info(f"❌ 采样出错: {e}")
                            finally:
                                self._sampling_active = False
                        self._sampling_thread = threading.Thread(target=_sample_random_goals, daemon=True)
                        self._sampling_thread.start()
                    else:
                        logger.info("❌ 请先设置起点（按 S 键后点击地图）！")
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
            elif key == ord('v') or key == ord('V'):  # 'V'键 - 固定终点，随机采样多个起点
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样正在进行中，请等待完成后再触发")
                    elif self.goal_set:
                        self._sampling_active = True
                        logger.info("✓ 请在终端中配置随机起点采样参数...")
                        def _sample_random_starts():
                            try:
                                cfg = self.show_random_sample_config_dialog('starts')
                                if cfg is not None:
                                    self.sample_random_starts_from_goal(
                                        num_starts=cfg['num_pts'],
                                        min_dist_m=cfg['min_dist'],
                                    )
                                else:
                                    logger.info("❌ 已取消")
                            except Exception as e:
                                logger.info(f"❌ 采样出错: {e}")
                            finally:
                                self._sampling_active = False
                        self._sampling_thread = threading.Thread(target=_sample_random_starts, daemon=True)
                        self._sampling_thread.start()
                    else:
                        logger.info("❌ 请先设置终点（按 G 键后点击地图）！")
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
            elif key == ord('w') or key == ord('W'):  # 'W'键 - 固定起点，随机采样waypoints并返回起点附近
                if self.enable_annotate:
                    if self._sampling_active:
                        logger.info("⚠️ 采样正在进行中，请等待完成后再触发")
                    elif self.start_set:
                        self._sampling_active = True
                        logger.info("✓ 请在终端中配置 Waypoint 环形路径采样参数...")
                        def _sample_waypoint_loop():
                            try:
                                cfg = self.show_random_sample_config_dialog('waypoint_loop')
                                if cfg is not None:
                                    self.sample_waypoint_loop(
                                        num_waypoints=cfg['num_waypoints'],
                                        loop_radius_m=cfg['loop_radius_m'],
                                        return_radius_m=cfg['return_radius_m'],
                                    )
                                else:
                                    logger.info("❌ 已取消")
                            except Exception as e:
                                logger.info(f"❌ 采样出错: {e}")
                            finally:
                                self._sampling_active = False
                        self._sampling_thread = threading.Thread(target=_sample_waypoint_loop, daemon=True)
                        self._sampling_thread.start()
                    else:
                        logger.info("❌ 请先设置起点（按 S 键后点击地图）！")
                else:
                    logger.info("❌ 需要启用标注模式（--annotate）")
        
        cv2.destroyAllWindows()
        logger.info("交互窗口已关闭")
        
        # 清理渲染管理器资源
        if self.render_manager is not None:
            self.render_manager.cleanup()
            self.render_manager = None
    
    def save_results(self, color_projection: np.ndarray, 
                    obstacle_map: np.ndarray, traversability_mask: np.ndarray,
                    expanded_traversability: np.ndarray, path_image: np.ndarray):
        """保存所有结果到output目录"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 应用轻微模糊到颜色投影
        color_projection_smooth = cv2.GaussianBlur(color_projection, (3, 3), 0.5)
        
        # 保存各种图像
        cv2.imwrite(str(output_dir / "color_projection.png"), color_projection_smooth)
        logger.info(f"\n彩色点云投影已保存到: {output_dir / 'color_projection.png'}")
        
        cv2.imwrite(str(output_dir / "obstacle_map.png"), obstacle_map)
        logger.info(f"\n障碍物地图已保存到: {output_dir / 'obstacle_map.png'}")
        logger.info("  (白色=障碍物, 黑色=可通行)")
        
        cv2.imwrite(str(output_dir / "traversability_mask.png"), traversability_mask)
        logger.info(f"可通行性掩码已保存到: {output_dir / 'traversability_mask.png'}")
        logger.info("  (白色=可通行, 黑色=障碍物)")
        
        cv2.imwrite(str(output_dir / "expanded_traversability_mask.png"), 
                   expanded_traversability)
        logger.info(f"\n扩展可通行性掩码已保存到: {output_dir / 'expanded_traversability_mask.png'}")
        logger.info("  (白色=扩展后可通行, 黑色=障碍物)")
        
        cv2.imwrite(str(output_dir / "path_visualization.png"), path_image)
        logger.info(f"路径可视化已保存到: {output_dir / 'path_visualization.png'}")
        logger.info("  (蓝色线=规划路径, 绿色圆=起点, 红色圆=终点)")
    
    def run(self, ply_path: str = None, scene_name: str = None, load_existing: bool = False,
            edit_map: bool = False, crop_map: bool = False, resolution: float = 0.02):
        """
        主运行函数
        
        Args:
            ply_path: 点云文件路径
            scene_name: 场景名称（用于保存/加载地图）
            load_existing: 是否加载已有地图
            edit_map: 是否进入地图编辑模式
            crop_map: 是否进入地图裁剪模式
        """
        # 保存ply_path用于渲染
        if ply_path is not None:
            self.ply_path = ply_path
        
        # 从PLY路径提取场景名称
        if scene_name is None and ply_path is not None:
            scene_name = Path(ply_path).stem
        elif scene_name is None:
            scene_name = "default"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"场景名称: {scene_name}")
        logger.info(f"{'='*60}")
        
        # 创建地图管理器
        maps_dir = self.args.map_dir
        self.map_manager = OccupancyMapManager(scene_name=scene_name, output_dir=maps_dir)
        
        # 尝试加载已有地图
        map_loaded = False
        if load_existing or self.map_manager.map_exists():
            if self.map_manager.map_exists():
                logger.info(f"\n发现已有地图文件...")
                if load_existing:
                    logger.info("正在加载已有地图...")
                    map_loaded = self.map_manager.load_map()
                else:
                    user_input = input("是否加载已有地图？(y/n): ").strip().lower()
                    if user_input == 'y':
                        map_loaded = self.map_manager.load_map()
        
        # 如果没有加载到地图，从点云创建
        if not map_loaded:
            if ply_path is None:
                raise ValueError("必须提供点云文件路径或已有的地图文件")
            
            logger.info("\n正在从点云创建地图...")
            
            # 加载点云
            points, colors = self.map_manager.load_point_cloud(ply_path)
            
            # 创建占用网格
            self.map_manager.create_occupancy_grid(points, colors, resolution=resolution)
            
            # 自动保存地图
            self.map_manager.save_map()
            logger.info(f"✓ 地图已自动保存到场景 '{scene_name}' 下")
        
        # 获取地图数据
        map_data = self.map_manager.get_map_data()
        
        # 设置成员变量（用于路径规划）
        self.obstacle_map = map_data["obstacle_map"]
        self.expanded_traversability = map_data["expanded_traversability"]
        self.original_traversability = map_data["traversability_mask"]
        self.point_cloud_coverage = map_data["point_cloud_coverage"]
        self.display_image = map_data["color_projection"]
        self.grid_width = map_data["grid_width"]
        self.grid_height = map_data["grid_height"]
        self.grid_resolution = map_data["grid_resolution"]
        self._path_planner = None  # 地图已更新，重置规划器
        self._trajectory_paths = {}
        self.min_pt = map_data["min_pt"]
        self.max_pt = map_data["max_pt"]
        
        # 如果需要编辑地图
        if edit_map:
            logger.info("\n进入地图编辑模式...")
            self.map_manager.start_edit_mode()
            
            # 编辑后更新数据
            map_data = self.map_manager.get_map_data()
            self.obstacle_map = map_data["obstacle_map"]
            self.expanded_traversability = map_data["expanded_traversability"]
            self.original_traversability = map_data["traversability_mask"]
            self._path_planner = None  # 地图已变，重置规划器
            self._trajectory_paths = {}
        
        # 如果需要裁剪地图
        if crop_map:
            logger.info("\n进入地图裁剪模式...")
            crop_success = self.map_manager.interactive_crop()
            
            if crop_success:
                # 裁剪后更新数据
                map_data = self.map_manager.get_map_data()
                self.obstacle_map = map_data["obstacle_map"]
                self.expanded_traversability = map_data["expanded_traversability"]
                self.original_traversability = map_data["traversability_mask"]
                self.point_cloud_coverage = map_data["point_cloud_coverage"]
                self.display_image = map_data["color_projection"]
                self.grid_width = map_data["grid_width"]
                self.grid_height = map_data["grid_height"]
                self.grid_resolution = map_data["grid_resolution"]
                self._path_planner = None  # 地图已变，重置规划器
                self._trajectory_paths = {}
                self.min_pt = map_data["min_pt"]
                self.max_pt = map_data["max_pt"]
                logger.info("✓ 地图裁剪完成，数据已更新")
            else:
                logger.info("地图裁剪已取消")
        
        # 显示最终统计
        total_pixels = self.grid_width * self.grid_height
        coverage_pixels = np.count_nonzero(self.point_cloud_coverage)
        obstacle_pixels = np.count_nonzero(self.obstacle_map)
        traversable_pixels = np.count_nonzero(self.original_traversability)
        expanded_traversable_pixels = np.count_nonzero(self.expanded_traversability)
        
        logger.info(f"\n=== 最终统计 ===")
        logger.info(f"总网格单元: {total_pixels}")
        logger.info(f"点云覆盖单元: {coverage_pixels} ({coverage_pixels/total_pixels*100:.2f}%)")
        logger.info(f"障碍物单元: {obstacle_pixels} ({obstacle_pixels/coverage_pixels*100:.2f}% of coverage)")
        logger.info(f"原始可通行单元: {traversable_pixels} ({traversable_pixels/coverage_pixels*100:.2f}% of coverage)")
        logger.info(f"扩展后可通行单元: {expanded_traversable_pixels} ({expanded_traversable_pixels/coverage_pixels*100:.2f}% of coverage)")
        
        # 运行交互式路径规划界面
        self.run_interactive()
        
        logger.info("\n程序完成!")

def process_args(args):
    args.scan_dir = os.path.join(args.base_dir, 'scans')
    args.base_log_dir = os.path.join(args.base_dir, 'logs')
    args.map_dir = os.path.join(args.base_dir, 'scans', 'maps')
    args.output_dir = os.path.join(args.base_log_dir, args.exp_name)
    
    # 配置日志文件输出（按时间命名）
    log_file = setup_file_logger(args.output_dir)
    print(f'Log file: {log_file}')
    
    return args

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scan2Occ3D - 点云路径规划与相机位姿生成工具"
    )
    parser.add_argument('--base_dir', type=str, default='data/',
                       help='基础目录')
    parser.add_argument('--output_dir', type=str, default='data/logs',
                       help='输出目录')
    parser.add_argument('--ply_path', type=str, default=None,
                       help='点云文件路径（PLY格式）')
    parser.add_argument('--scene_name', type=str, default=None,
                       help='场景名称（用于保存/加载地图，默认从PLY文件名提取）')
    parser.add_argument('--load_map', action='store_true',
                       help='加载已有的地图文件（如果存在）')
    parser.add_argument('--edit_map', action='store_true',
                       help='启动后立即进入地图编辑模式')
    parser.add_argument('--exp_name', type=str, default='default',
                       help='实验名称（用于保存log文件，默认default）')
    parser.add_argument('--render', action='store_true',
                       help='在路径规划完成后渲染RGB和深度视频')
    parser.add_argument('--annotate', action='store_true',
                       help='启用标注模式（自动启用渲染，允许输入导航指令）')
    parser.add_argument('--camera_type', type=str, default='single', choices=['single', 'pano'],
                       help='相机类型: single(单视角) 或 pano(全景360度), 默认single')
    parser.add_argument('--enable-depth', action='store_true', default=False,
                       help='启用深度图渲染 (默认False)')
    parser.add_argument('--camera_height', type=float, default=1.2,
                       help='相机距地面高度（米），默认1.2m（模拟机器人搭载相机高度）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='路径噪声采样数量（含原始最短路），默认5')
    parser.add_argument('--noise_level', type=float, default=0.3,
                       help='路径噪声强度 [0.0~1.0]，0=无噪声，1=最大扰动，默认0.3')
    parser.add_argument('--noise_type', type=str, default='waypoint',
                       choices=['waypoint', 'grid'],
                       help='路径噪声类型: waypoint(在路径中间插入扰动点) 或 grid(对路径节点施加偏移)，默认waypoint')
    args = parser.parse_args()
    
    # 如果启用标注，自动启用渲染
    if args.annotate:
        args.render = True
    
    args = process_args(args)
    
    # 如果没有指定PLY路径，使用默认值
    if args.ply_path is None:
        # 默认点云文件路径
        default_paths = [
            'data/scans/93Sydney_3DGSResults/93SydneyLibby.ply',
            "data/scans/demo_data_625/1224_625_3dgs_epoch_30k.ply"
        ]
        
        # 尝试找到存在的文件
        ply_path = None
        for path in default_paths:
            if os.path.exists(path):
                ply_path = path
                break
        
        # 如果没有找到，并且不是加载模式，则报错
        if ply_path is None and not args.load_map:
            logger.info("错误: 未找到点云文件，请使用 --ply_path 指定点云文件路径")
            logger.info("或使用 --load_map 加载已有地图")
            logger.info("\n使用示例:")
            logger.info("  python main.py --ply_path data/scans/xxx.ply")
            logger.info("  python main.py --load_map --scene_name my_scene")
            return
    else:
        ply_path = args.ply_path
        
        # 检查文件是否存在
        if not os.path.exists(ply_path) and not args.load_map:
            logger.info(f"错误: 点云文件不存在: {ply_path}")
            return

    # 创建并运行
    app = Scan2Occ3D(args=args)
    app.run(ply_path=ply_path, 
           scene_name=args.scene_name, 
           load_existing=args.load_map,
           edit_map=args.edit_map)


if __name__ == "__main__":
    main()

