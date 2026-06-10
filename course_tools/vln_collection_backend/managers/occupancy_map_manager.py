#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OccupancyMapManager - 占用网格地图管理类
功能:
1. 从点云生成占用网格地图
2. 保存和加载地图
3. 地图编辑功能（涂抹不可达区域）
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import pickle

# 导入log管理器
from managers.log_manager import logger

try:
    import open3d as o3d
except ImportError:
    logger.info("错误: 需要安装 open3d 库")
    import sys
    sys.exit(1)


class OccupancyMapManager:
    """占用网格地图管理类"""
    
    def __init__(self, scene_name: str = "default", output_dir: str = "data/scans/maps"):
        """
        初始化地图管理器
        
        Args:
            scene_name: 场景名称（用于保存/加载地图）
            output_dir: 地图输出目录
        """
        self.scene_name = scene_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 地图数据
        self.obstacle_map = None
        self.traversability_mask = None
        self.expanded_traversability = None
        self.point_cloud_coverage = None
        self.color_projection = None
        self.binary_projection = None
        
        # 网格参数
        self.grid_width = 0
        self.grid_height = 0
        self.grid_resolution = 0.0
        self.min_pt = None
        self.max_pt = None
        
        # 地面高度信息
        self.ground_z = 0.0
        
        # 编辑模式
        self.editing_mode = False
        self.brush_size = 10
        self.edit_as_obstacle = True  # True=涂抹障碍物, False=涂抹可通行
    
    def get_map_path(self) -> Path:
        """获取当前场景的地图保存路径"""
        return self.output_dir / f"{self.scene_name}_map.pkl"
    
    def map_exists(self) -> bool:
        """检查地图文件是否存在"""
        return self.get_map_path().exists()
    
    def save_map(self, scene_name: str = None):
        """保存地图到文件"""
        if scene_name:
            self.scene_name = scene_name
            logger.info(f"场景名称已更新为: {self.scene_name}")
        else:
            logger.info(f"场景名称未更新")
            
        map_data = {
            "obstacle_map": self.obstacle_map,
            "traversability_mask": self.traversability_mask,
            "expanded_traversability": self.expanded_traversability,
            "point_cloud_coverage": self.point_cloud_coverage,
            "color_projection": self.color_projection,
            "binary_projection": self.binary_projection,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "grid_resolution": self.grid_resolution,
            "min_pt": self.min_pt,
            "max_pt": self.max_pt,
            "ground_z": self.ground_z,
        }
        
        map_path = self.get_map_path()
        with open(map_path, 'wb') as f:
            pickle.dump(map_data, f)
        
        logger.info(f"\n✓ 地图已保存到: {map_path}")
        
        # 同时保存可视化图像
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(vis_dir / f"{self.scene_name}_obstacle_map.png"), self.obstacle_map)
        cv2.imwrite(str(vis_dir / f"{self.scene_name}_traversability.png"), self.traversability_mask)
        cv2.imwrite(str(vis_dir / f"{self.scene_name}_color_projection.png"), self.color_projection)
        
        logger.info(f"✓ 可视化图像已保存到: {vis_dir}")
        return True
    
    def load_map(self) -> bool:
        """
        加载地图文件
        
        Returns:
            True if successful, False otherwise
        """
        map_path = self.get_map_path()
        
        if not map_path.exists():
            logger.info(f"地图文件不存在: {map_path}")
            return False
        
        try:
            with open(map_path, 'rb') as f:
                map_data = pickle.load(f)
            
            self.obstacle_map = map_data["obstacle_map"]
            self.traversability_mask = map_data["traversability_mask"]
            self.expanded_traversability = map_data["expanded_traversability"]
            self.point_cloud_coverage = map_data["point_cloud_coverage"]
            self.color_projection = map_data["color_projection"]
            self.binary_projection = map_data["binary_projection"]
            self.grid_width = map_data["grid_width"]
            self.grid_height = map_data["grid_height"]
            self.grid_resolution = map_data["grid_resolution"]
            self.min_pt = map_data["min_pt"]
            self.max_pt = map_data["max_pt"]
            self.ground_z = map_data["ground_z"]
            
            logger.info(f"\n✓ 地图已从文件加载: {map_path}")
            logger.info(f"  网格尺寸: {self.grid_width} x {self.grid_height}")
            logger.info(f"  分辨率: {self.grid_resolution} m/像素")
            
            return True
        except Exception as e:
            logger.info(f"加载地图失败: {e}")
            return False
    
    def _visualize_point_cloud_o3d(
        self, pcd: o3d.geometry.PointCloud, title: str = "Point cloud"
    ) -> None:
        """用 Open3D 打开点云窗口（阻塞至窗口关闭）；无显示环境时会记录失败信息。"""
        if pcd is None or pcd.is_empty():
            logger.info("点云为空，跳过可视化")
            return
        try:
            logger.info(f"Open3D 可视化（关闭窗口后继续）: {title}")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=title[:256],
                width=1280,
                height=720,
            )
        except Exception as e:
            logger.info(f"点云可视化失败: {e}")

    def load_point_cloud(
        self,
        ply_path: str,
        remove_outliers: bool = True,
        save_filtered_ply: bool = False,
        save_filtered_ply_path: Optional[str] = None,
        load_filter_map: bool = False,
        vis: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        加载PLY点云文件
        
        Args:
            ply_path: 点云文件路径（用于读取原始点云；若启用 load_filter_map 且存在缓存，则仅用于解析
                ``{stem}_filtered.ply`` 的文件名）
            remove_outliers: 是否移除离群点（默认True；若已加载缓存则忽略）
            save_filtered_ply: 为 True 时，将离群点过滤后的点云保存到
                ``{output_dir}/{原文件名}_filtered.ply``（仅当 remove_outliers=True 时生效）
            save_filtered_ply_path: 若指定，将过滤后的点云保存到该路径（覆盖 save_filtered_ply 的默认路径；
                仅当 remove_outliers=True 时生效）；若 load_filter_map=True，也作为优先查找的缓存路径
            load_filter_map: 为 True 时，若存在 ``save_filtered_ply_path`` 或
                ``{output_dir}/{ply_path 的文件名}_filtered.ply``，则直接加载该文件并跳过离群点过滤
            vis: 为 True 时，在返回前用 Open3D 打开当前点云（阻塞）
        
        返回: (points, colors) - colors可能为None
        """
        filtered_cache_path: Optional[str] = None
        if load_filter_map:
            filtered_cache_path = save_filtered_ply_path or str(
                self.output_dir / f"{Path(ply_path).stem}_filtered.ply"
            )

        if filtered_cache_path and os.path.isfile(filtered_cache_path):
            logger.info(
                f"\n读取已缓存的过滤点云（跳过离群点过滤）: {filtered_cache_path}"
            )
            pcd = o3d.io.read_point_cloud(filtered_cache_path)
            if pcd.is_empty():
                raise ValueError("过滤点云文件为空!")
            logger.info(f"✓ 共 {len(pcd.points)} 个点")
            points = np.asarray(pcd.points)
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                colors = (colors * 255).astype(np.uint8)
                logger.info("点云包含RGB颜色信息")
            if vis:
                self._visualize_point_cloud_o3d(pcd, str(filtered_cache_path))
            return points, colors

        logger.info(f"\n读取点云文件: {ply_path}")

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
        
        # 使用Open3D读取点云
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if pcd.is_empty():
            raise ValueError("点云文件为空!")
        
        original_count = len(pcd.points)
        logger.info(f"✓ 成功读取点云，共 {original_count} 个点")
        
        # 移除离群点
        if remove_outliers:
            logger.info("正在移除离群点...")
            
            # 第一步：使用统计离群点移除方法
            # nb_neighbors: 分析每个点的邻居数量（增加到30，考虑更多邻居）
            # std_ratio: 标准差倍数阈值（降低到1.5，更激进地过滤）
            logger.info("  步骤1: 统计方法过滤...")
            pcd_filtered, _ = pcd.remove_statistical_outlier(
                nb_neighbors=30,
                std_ratio=1.5
            )
            
            step1_removed = original_count - len(pcd_filtered.points)
            logger.info(f"    移除 {step1_removed} 个点")
            
            # 第二步：使用半径离群点移除方法（移除周围邻居点太少的点）
            logger.info("  步骤2: 半径方法过滤...")
            pcd_filtered, _ = pcd_filtered.remove_radius_outlier(
                nb_points=10,  # 在半径内至少需要10个邻居点
                radius=0.1     # 搜索半径0.1米
            )
            
            step2_removed = original_count - step1_removed - len(pcd_filtered.points)
            logger.info(f"    移除 {step2_removed} 个点")
            
            # 第三步：使用聚类方法移除远离主要区域的孤立点群
            # logger.info("  步骤3: 聚类方法过滤远离主要区域的点...")
            # labels = np.array(pcd_filtered.cluster_dbscan(
            #     eps=0.15,      # 聚类半径0.15米
            #     min_points=50,  # 最小点数50
            #     print_progress=False
            # ))
            
            # if len(labels) > 0 and labels.max() >= 0:
            #     # 统计每个聚类的点数
            #     unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                
            #     if len(unique_labels) > 0:
            #         # 找到最大的聚类（主要区域）
            #         largest_cluster_label = unique_labels[np.argmax(counts)]
            #         largest_cluster_size = counts[np.argmax(counts)]
                    
            #         # 保留最大聚类和较大的聚类（至少是最大聚类的10%）
            #         size_threshold = max(largest_cluster_size * 0.1, 100)
            #         keep_labels = unique_labels[counts >= size_threshold]
                    
            #         # 创建掩码，保留主要聚类的点
            #         keep_mask = np.isin(labels, keep_labels)
            #         pcd_filtered = pcd_filtered.select_by_index(np.where(keep_mask)[0])
                    
            #         step3_removed = original_count - step1_removed - step2_removed - len(pcd_filtered.points)
            #         num_clusters_kept = len(keep_labels)
            #         num_clusters_removed = len(unique_labels) - num_clusters_kept
                    
            #         logger.info(f"    发现 {len(unique_labels)} 个聚类")
            #         logger.info(f"    保留 {num_clusters_kept} 个主要聚类，移除 {num_clusters_removed} 个孤立点群")
            #         logger.info(f"    移除 {step3_removed} 个点")
            #     else:
            #         step3_removed = 0
            #         logger.info(f"    未发现有效聚类，跳过")
            # else:
            #     step3_removed = 0
            #     logger.info(f"    未发现有效聚类，跳过")
            
            # 总计
            total_removed = original_count - len(pcd_filtered.points)
            removed_percent = (total_removed / original_count) * 100
            
            logger.info(f"✓ 总共移除 {total_removed} 个离群点 ({removed_percent:.2f}%)")
            logger.info(f"  剩余点数: {len(pcd_filtered.points)}")
            
            pcd = pcd_filtered

        out_ply = save_filtered_ply_path
        if out_ply is None and save_filtered_ply:
            out_ply = str(self.output_dir / f"{Path(ply_path).stem}_filtered.ply")
        if out_ply is not None:
            if remove_outliers:
                Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
                if o3d.io.write_point_cloud(out_ply, pcd):
                    logger.info(f"✓ 离群点过滤后的点云已保存: {out_ply}")
                else:
                    logger.info(f"✗ 保存过滤点云失败: {out_ply}")
            else:
                logger.info("跳过保存过滤点云（remove_outliers=False）")

        if vis:
            self._visualize_point_cloud_o3d(pcd, ply_path)
        
        # 获取点云坐标
        points = np.asarray(pcd.points)
        
        # 尝试获取颜色信息
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)  # 范围 [0, 1]
            colors = (colors * 255).astype(np.uint8)  # 转换为 [0, 255]
            logger.info(f"点云包含RGB颜色信息")
        
        return points, colors
    
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
        
        # 定义障碍物高度范围 (0.2m到1.5m以上地面)
        z_lower_bound = ground_z + 0.2
        z_upper_bound = ground_z + 1.5
        
        logger.info(f"\n=== 障碍物高度范围 ===")
        logger.info(f"地面高度 (10分位数): {ground_z:.3f} m")
        logger.info(f"障碍物Z范围: [{z_lower_bound:.3f}, {z_upper_bound:.3f}] m")
        
        self.ground_z = ground_z
        
        return ground_z, z_lower_bound, z_upper_bound
    
    def create_occupancy_grid(self, points: np.ndarray, colors: Optional[np.ndarray],
                             resolution: float = 0.02, use_complex_filter=True) -> bool:
        """
        从点云创建占用网格
        
        Args:
            points: 点云坐标
            colors: 点云颜色（可选）
            resolution: 网格分辨率（米/像素）
            use_complex_filter: 是否开启复杂的过滤过程
        
        Returns:
            True if successful
            
        变量	        255表示	                   0表示
        obstacle_map	障碍物	                   非障碍
        traversability_mask	可通行	              不可通行
        expanded_traversability	安全可通行      	不可通行
        """
        logger.info(f"\n=== 创建占用网格 ===")
        
        # 计算边界
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        
        dx, dy, dz = max_pt - min_pt
        logger.info(f"点云边界:")
        logger.info(f"  X: [{min_pt[0]:.3f}, {max_pt[0]:.3f}] (范围: {dx:.3f} m)")
        logger.info(f"  Y: [{min_pt[1]:.3f}, {max_pt[1]:.3f}] (范围: {dy:.3f} m)")
        logger.info(f"  Z: [{min_pt[2]:.3f}, {max_pt[2]:.3f}] (范围: {dz:.3f} m)")
        
        # 计算地面高度
        ground_z, z_lower, z_upper = self.compute_ground_level(points)

        # # 将 points 中 z > 0 的点保存到 new_points # for 417 !!!
        # new_points = points[points[:, 2] > -0.5]
        # ground_z, z_lower, z_upper = self.compute_ground_level(new_points)
        
        # 计算网格尺寸
        grid_width = int(np.ceil(dx / resolution)) + 1
        grid_height = int(np.ceil(dy / resolution)) + 1
        
        logger.info(f"\n网格配置:")
        logger.info(f"  分辨率: {resolution} m/像素")
        logger.info(f"  尺寸: {grid_width} x {grid_height} 像素")
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_resolution = resolution
        self.min_pt = min_pt
        self.max_pt = max_pt
        
        # 创建网格
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        color_projection = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        binary_projection = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # 向量化投影点云到网格
        grid_xs = ((points[:, 0] - min_pt[0]) / resolution).astype(int)
        grid_ys = ((max_pt[1] - points[:, 1]) / resolution).astype(int)  # Y轴翻转

        # 有效点掩码（在网格范围内）
        valid_mask = (
            (grid_xs >= 0) & (grid_xs < grid_width) &
            (grid_ys >= 0) & (grid_ys < grid_height)
        )
        valid_gx = grid_xs[valid_mask]
        valid_gy = grid_ys[valid_mask]

        # binary_projection
        binary_projection[valid_gy, valid_gx] = 255

        # 颜色投影
        if colors is not None:
            color_projection[valid_gy, valid_gx] = colors[valid_mask][:, ::-1]  # RGB -> BGR
        else:
            z_range = max_pt[2] - min_pt[2]
            if z_range > 0:
                normalized_z = (points[valid_mask, 2] - min_pt[2]) / z_range
                b_vals = np.where(normalized_z < 0.5, 255.0, (1.0 - normalized_z) * 2 * 255)
                g_vals = np.where(normalized_z < 0.5, normalized_z * 2 * 255, 255.0)
                r_vals = np.where(normalized_z < 0.5, 0.0, (normalized_z - 0.5) * 2 * 255)
                color_projection[valid_gy, valid_gx] = np.clip(
                    np.stack([b_vals, g_vals, r_vals], axis=1), 0, 255
                ).astype(np.uint8)

        # occupancy_grid：只处理在障碍物高度范围内的点
        z_mask = (points[:, 2] >= z_lower) & (points[:, 2] <= z_upper)
        occ_mask = valid_mask & z_mask
        occupancy_grid[grid_ys[occ_mask], grid_xs[occ_mask]] = 255
        
        # 过滤占用网格
        if use_complex_filter:
            obstacle_map = self.filter_occupancy_grid(occupancy_grid)
        else:
            obstacle_map = occupancy_grid
            
        # 生成可通行性掩码
        point_cloud_coverage = binary_projection.copy()
        
        # 对点云覆盖区域进行形态学闭运算，填充地板上的小洞（减少核大小保留边缘）
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 从7x7降到5x5
        point_cloud_coverage = cv2.morphologyEx(point_cloud_coverage, cv2.MORPH_CLOSE, 
                                                kernel_fill, iterations=1)  # 从2次降到1次
        
        # 生成初始可通行性掩码
        traversability_mask = cv2.bitwise_and(
            point_cloud_coverage,
            cv2.bitwise_not(obstacle_map)
        )
        
        # 对可通行区域进行轻度形态学开运算，去除小的孤立可通行点（但保留边缘）
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 从5x5降到3x3
        traversability_mask = cv2.morphologyEx(traversability_mask, cv2.MORPH_OPEN, 
                                            kernel_clean, iterations=1)
        
        # 再次应用闭运算，填充可通行区域中的小洞
        traversability_mask = cv2.morphologyEx(traversability_mask, cv2.MORPH_CLOSE, 
                                            kernel_clean, iterations=1)  # 从2次降到1次
        
        # 扩展障碍物
        expanded_obstacles = self.expand_obstacles(obstacle_map, robot_radius=0.1)
        expanded_traversability = cv2.bitwise_and(
            point_cloud_coverage,
            cv2.bitwise_not(expanded_obstacles)
        )
        
        # 对扩展后的可通行区域也进行轻度清理
        expanded_traversability = cv2.morphologyEx(expanded_traversability, cv2.MORPH_CLOSE,
                                                kernel_clean, iterations=1)  # 只用闭运算填洞，不用开运算
        # else:
        #     # 扫描比较好的情况，直接用occupancy_grid
        #     obstacle_map = occupancy_grid

        #     traversability_mask = cv2.bitwise_not(obstacle_map)

        #     expanded_obstacles = self.expand_obstacles(obstacle_map, robot_radius=0.15)

        #     expanded_traversability = cv2.bitwise_and(
        #         traversability_mask,
        #         cv2.bitwise_not(expanded_obstacles)
        #     )

        #     traversability_mask = cv2.bitwise_and(
        #         binary_projection,
        #         traversability_mask
        #     )

        #     expanded_traversability = cv2.bitwise_and(
        #         binary_projection,
        #         expanded_traversability
        #     )
                    
            
        # 保存结果
        self.obstacle_map = obstacle_map
        self.traversability_mask = traversability_mask
        self.expanded_traversability = expanded_traversability
        self.point_cloud_coverage = point_cloud_coverage
        self.color_projection = cv2.GaussianBlur(color_projection, (3, 3), 0.5)
        self.binary_projection = binary_projection
        
        # 统计
        coverage_pixels = np.count_nonzero(point_cloud_coverage)
        obstacle_pixels = np.count_nonzero(obstacle_map)
        traversable_pixels = np.count_nonzero(traversability_mask)
        
        logger.info(f"\n=== 地图统计 ===")
        logger.info(f"点云覆盖像素: {coverage_pixels}")
        logger.info(f"障碍物像素: {obstacle_pixels}")
        logger.info(f"可通行像素: {traversable_pixels}")
        
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # RANSAC 地面提取版本
    # ─────────────────────────────────────────────────────────────────────────
    def create_occupancy_grid_ransac(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        resolution: float = 0.06,
        voxel_size: float = 0.05,
        plane_distance_threshold: float = 0.02,
        plane_ransac_n: int = 3,
        plane_num_iterations: int = 1000,
        normal_up_threshold: float = 0.8,
        obstacle_height_min: float = 0.1,
        obstacle_height_max: float = 2.5,
        max_planes: int = 5,
        robot_radius: float = 0.3,
    ) -> bool:
        """
        使用 Open3D RANSAC 平面分割提取地面，并生成 2D 占用栅格地图。

        算法流程：
          1. 体素下采样（提升 RANSAC 速度与稳定性）
          2. 迭代 RANSAC：最多提取 max_planes 个水平平面
             – 法向量 Z 分量 >= normal_up_threshold 时判定为水平地面
             – 非水平平面视为障碍墙体，终止迭代
          3. 确定 ground_z（所有地面内点 Z 中位数）
          4. 将地面内点投影到 2D 网格 → point_cloud_coverage
          5. 将高于地面 [obstacle_height_min, obstacle_height_max] 的点投影为障碍物
          6. 形态学后处理 → traversability_mask / expanded_traversability

        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)，uint8 BGR；可为 None
            resolution: 栅格分辨率（米/像素）
            voxel_size: 体素下采样尺寸（米）；0 表示不下采样
            plane_distance_threshold: RANSAC 内点距离阈值（米）
            plane_ransac_n: RANSAC 每次随机采样的最少点数
            plane_num_iterations: RANSAC 迭代次数
            normal_up_threshold: 法向量 Z 分量阈值，超过此值视为水平面
            obstacle_height_min: 地面以上视为障碍物的最小高度（米）
            obstacle_height_max: 地面以上视为障碍物的最大高度（米）
            max_planes: 最多迭代提取地面平面的次数
            robot_radius: 机器人半径（米），用于扩展障碍物

        Returns:
            True if successful
        """
        logger.info("\n=== 创建占用网格 [RANSAC 地面提取] ===")

        # ── 1. 构建 Open3D 点云并体素下采样 ────────────────────────
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd_full.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        if voxel_size > 0:
            pcd_ds = pcd_full.voxel_down_sample(voxel_size=voxel_size)
            logger.info(
                f"体素下采样: {len(pcd_full.points)} → {len(pcd_ds.points)} 点"
                f"（voxel_size={voxel_size} m）"
            )
        else:
            pcd_ds = pcd_full

        # ── 2. 迭代 RANSAC 提取水平地面平面 ────────────────────────
        pts_ds = np.asarray(pcd_ds.points)
        remaining_indices = np.arange(len(pts_ds))   # 当前还未归类的点索引
        ground_indices_ds: list[int] = []            # 地面内点索引（下采样坐标系）
        planes_found = 0

        for iteration in range(max_planes):
            if len(remaining_indices) < plane_ransac_n:
                break

            pcd_remain = pcd_ds.select_by_index(remaining_indices.tolist())
            try:
                plane_model, inlier_local = pcd_remain.segment_plane(
                    distance_threshold=plane_distance_threshold,
                    ransac_n=plane_ransac_n,
                    num_iterations=plane_num_iterations,
                )
            except Exception as e:
                logger.info(f"  RANSAC 第 {iteration + 1} 次迭代失败: {e}")
                break

            a, b, c, d = plane_model
            # 法向量取绝对值比较（平面方程 ax+by+cz+d=0，法向量可能朝下）
            normal_z_abs = abs(c) / (np.sqrt(a**2 + b**2 + c**2) + 1e-9)
            inlier_count = len(inlier_local)

            logger.info(
                f"  平面 {iteration + 1}: 法向量 Z={normal_z_abs:.3f}，"
                f"内点数={inlier_count}，"
                f"方程=[{a:.3f},{b:.3f},{c:.3f},{d:.3f}]"
            )

            if normal_z_abs >= normal_up_threshold:
                # 水平地面平面
                global_inlier_idx = remaining_indices[inlier_local].tolist()
                ground_indices_ds.extend(global_inlier_idx)
                remaining_indices = np.delete(remaining_indices, inlier_local)
                planes_found += 1
                logger.info(f"    → 判定为地面，累计地面点: {len(ground_indices_ds)}")
            else:
                # 非水平面（墙体等），停止继续向下找地面
                logger.info(f"    → 法向量非水平，停止迭代")
                break

        if planes_found == 0:
            logger.info("警告: 未找到任何水平地面平面，回退到高度分位法")
            z_vals = pts_ds[:, 2]
            ground_z_fallback = np.percentile(z_vals, 10)
            ground_mask = z_vals < (ground_z_fallback + plane_distance_threshold * 2)
            ground_indices_ds = np.where(ground_mask)[0].tolist()
            logger.info(f"  回退地面点数: {len(ground_indices_ds)}")

        # ── 3. 确定 ground_z ────────────────────────────────────────
        ground_pts_ds = pts_ds[ground_indices_ds]
        ground_z = float(np.median(ground_pts_ds[:, 2]))
        self.ground_z = ground_z
        logger.info(f"地面高度 ground_z = {ground_z:.4f} m")

        # ── 4. 计算全图边界（用原始全精度点云保证精度）─────────────
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        dx, dy = max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]
        logger.info(
            f"点云边界: X=[{min_pt[0]:.2f},{max_pt[0]:.2f}]  "
            f"Y=[{min_pt[1]:.2f},{max_pt[1]:.2f}]  "
            f"Z=[{min_pt[2]:.2f},{max_pt[2]:.2f}]"
        )

        grid_width = int(np.ceil(dx / resolution)) + 1
        grid_height = int(np.ceil(dy / resolution)) + 1
        logger.info(f"网格尺寸: {grid_width} × {grid_height}，分辨率={resolution} m/px")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_resolution = resolution
        self.min_pt = min_pt
        self.max_pt = max_pt

        # 辅助函数：世界坐标 → 像素坐标
        def _to_grid(x: float, y: float):
            gx = int((x - min_pt[0]) / resolution)
            gy = int((max_pt[1] - y) / resolution)  # Y 轴翻转
            return gx, gy

        # ── 5. 投影地面点到 2D 网格（point_cloud_coverage）─────────
        binary_projection = np.zeros((grid_height, grid_width), dtype=np.uint8)
        color_projection  = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for idx in ground_indices_ds:
            pt = pts_ds[idx]
            gx, gy = _to_grid(pt[0], pt[1])
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                binary_projection[gy, gx] = 255

        # 颜色投影（使用原始全精度点云，投影所有点）
        z_range = max_pt[2] - min_pt[2] + 1e-9
        for i, pt in enumerate(points):
            gx, gy = _to_grid(pt[0], pt[1])
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                if colors is not None:
                    color_projection[gy, gx] = colors[i][::-1]  # RGB→BGR
                else:
                    nz = (pt[2] - min_pt[2]) / z_range
                    if nz < 0.5:
                        color_projection[gy, gx] = [255, int(nz * 2 * 255), 0]
                    else:
                        color_projection[gy, gx] = [int((1 - nz) * 2 * 255), 255, int((nz - 0.5) * 2 * 255)]

        # ── 6. 投影障碍物点（高于地面 [obstacle_height_min, max] 的点）─
        z_low  = ground_z + obstacle_height_min
        z_high = ground_z + obstacle_height_max
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for pt in points:
            if z_low <= pt[2] <= z_high:
                gx, gy = _to_grid(pt[0], pt[1])
                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                    occupancy_grid[gy, gx] = 255

        logger.info(
            f"初始障碍物像素: {np.count_nonzero(occupancy_grid)}  "
            f"（Z 范围 [{z_low:.2f}, {z_high:.2f}] m）"
        )

        # ── 7. 形态学后处理 ─────────────────────────────────────────
        # 地面覆盖：闭运算填小洞
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        point_cloud_coverage = cv2.morphologyEx(binary_projection, cv2.MORPH_CLOSE, k5, iterations=1)

        # 障碍物：中值滤波 + 开运算去噪
        obs_filtered = cv2.medianBlur(occupancy_grid, 3)
        obstacle_map = cv2.morphologyEx(obs_filtered, cv2.MORPH_OPEN, k3, iterations=1)
        obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_CLOSE, k3, iterations=1)

        # 可通行性：地面覆盖 & 非障碍
        traversability_mask = cv2.bitwise_and(
            point_cloud_coverage, cv2.bitwise_not(obstacle_map)
        )
        traversability_mask = cv2.morphologyEx(traversability_mask, cv2.MORPH_OPEN,  k3, iterations=1)
        traversability_mask = cv2.morphologyEx(traversability_mask, cv2.MORPH_CLOSE, k3, iterations=1)

        # 扩展障碍物（考虑机器人半径）
        expanded_obstacles    = self.expand_obstacles(obstacle_map, robot_radius=robot_radius)
        expanded_traversability = cv2.bitwise_and(
            point_cloud_coverage, cv2.bitwise_not(expanded_obstacles)
        )
        expanded_traversability = cv2.morphologyEx(
            expanded_traversability, cv2.MORPH_CLOSE, k3, iterations=1
        )

        # ── 8. 保存结果 ─────────────────────────────────────────────
        self.obstacle_map           = obstacle_map
        self.traversability_mask    = traversability_mask
        self.expanded_traversability = expanded_traversability
        self.point_cloud_coverage   = point_cloud_coverage
        self.color_projection       = cv2.GaussianBlur(color_projection, (3, 3), 0.5)
        self.binary_projection      = binary_projection

        logger.info(f"\n=== 地图统计 ===")
        logger.info(f"地面覆盖像素    : {np.count_nonzero(point_cloud_coverage)}")
        logger.info(f"障碍物像素      : {np.count_nonzero(obstacle_map)}")
        logger.info(f"可通行像素      : {np.count_nonzero(traversability_mask)}")
        logger.info(f"安全可通行像素  : {np.count_nonzero(expanded_traversability)}")

        return True

    def filter_occupancy_grid(self, occupancy_grid: np.ndarray) -> np.ndarray:
        """对占用网格进行过滤处理，去除地板噪声"""
        # 1. 中值滤波去除椒盐噪声
        filtered = cv2.medianBlur(occupancy_grid, 5)
        
        # 2. 形态学开运算：先腐蚀后膨胀，去除小的噪声点（减少迭代次数保留边缘）
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 降到1次
        
        # 3. 形态学闭运算：填充障碍物中的小洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # 4. 移除微小孤立障碍物（连通域分析）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8)
        
        total_pixels = occupancy_grid.shape[0] * occupancy_grid.shape[1]
        # 适中的最小面积阈值，平衡噪声去除和边缘保留
        min_area = int(total_pixels * 0.0003)  # 降到0.0003
        
        cleaned = np.zeros_like(occupancy_grid)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask = (labels == i).astype(np.uint8)
                cleaned[mask > 0] = closed[mask > 0]
        
        logger.info(f"  过滤前障碍物像素: {np.count_nonzero(occupancy_grid)}")
        logger.info(f"  过滤后障碍物像素: {np.count_nonzero(cleaned)}")
        
        return cleaned
    
    def expand_obstacles(self, obstacle_map: np.ndarray, 
                        robot_radius: float = 0.15) -> np.ndarray:
        """扩展障碍物以考虑机器人半径"""
        expansion_radius = int(np.ceil(robot_radius / self.grid_resolution))
        
        if expansion_radius <= 0:
            return obstacle_map.copy()
        
        kernel_size = 2 * expansion_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(obstacle_map, kernel)
        
        return expanded
    
    def start_edit_mode(self, window_name: str = "Map Editor"):
        """
        启动地图编辑模式
        
        Args:
            window_name: OpenCV窗口名称
        """
        if self.obstacle_map is None:
            logger.info("错误: 没有可编辑的地图！请先加载或创建地图。")
            return
        
        self.editing_mode = True
        logger.info("\n=== 地图编辑模式 ===")
        logger.info("鼠标左键拖动: 涂抹区域")
        logger.info("'o'键: 切换为障碍物模式（涂抹不可达区域）")
        logger.info("'t'键: 切换为可通行模式（擦除障碍物）")
        logger.info("'+'键: 增大画笔大小")
        logger.info("'-'键: 减小画笔大小")
        logger.info("'s'键: 保存地图")
        logger.info("'q'或ESC: 退出编辑模式")
        
        # 创建编辑窗口
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # 设置鼠标回调
        drawing = [False]  # 使用列表来在闭包中共享状态
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing[0] = True
                self._draw_on_map(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing[0]:
                    self._draw_on_map(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing[0] = False
                # 更新扩展可通行性
                self._update_traversability()
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # 编辑循环
        while self.editing_mode:
            # 创建显示图像
            display = self._create_edit_display()
            
            # 添加提示信息
            mode_text = "障碍物模式" if self.edit_as_obstacle else "可通行模式"
            color = (0, 0, 255) if self.edit_as_obstacle else (0, 255, 0)
            cv2.putText(display, f"模式: {mode_text} | 画笔大小: {self.brush_size}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "'o':障碍物 | 't':可通行 | '+/-':画笔 | 's':保存 | 'q':退出",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            # 处理键盘输入
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.editing_mode = False
            elif key == ord('o'):
                self.edit_as_obstacle = True
                logger.info("切换到障碍物模式")
            elif key == ord('t'):
                self.edit_as_obstacle = False
                logger.info("切换到可通行模式")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(50, self.brush_size + 2)
                logger.info(f"画笔大小: {self.brush_size}")
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(2, self.brush_size - 2)
                logger.info(f"画笔大小: {self.brush_size}")
            elif key == ord('s'):
                self.save_map()
                logger.info("地图已保存！")
        
        cv2.destroyWindow(window_name)
        logger.info("退出编辑模式")
    
    def _draw_on_map(self, x: int, y: int):
        """在地图上绘制"""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return
        
        # 只在点云覆盖范围内编辑
        if self.point_cloud_coverage[y, x] == 0:
            return
        
        # 使用圆形画笔
        if self.edit_as_obstacle:
            # 涂抹障碍物
            cv2.circle(self.obstacle_map, (x, y), self.brush_size, 255, -1)
        else:
            # 擦除障碍物（涂抹为可通行）
            cv2.circle(self.obstacle_map, (x, y), self.brush_size, 0, -1)
    
    def _update_traversability(self, trav_mask: np.ndarray = None, expanded_trav_mask: np.ndarray = None):
        """更新可通行性掩码"""
        # 重新计算可通行性掩码
        if trav_mask is None:
            self.traversability_mask = cv2.bitwise_and(
                self.point_cloud_coverage,
                cv2.bitwise_not(self.obstacle_map)
            )
        else:
            self.traversability_mask = trav_mask
        
        # 重新计算扩展可通行性
        if expanded_trav_mask is None:
            expanded_obstacles = self.expand_obstacles(self.obstacle_map, robot_radius=0.1)
            self.expanded_traversability = cv2.bitwise_and(
                self.point_cloud_coverage,
                cv2.bitwise_not(expanded_obstacles)
            )
        else:
            self.expanded_traversability = expanded_trav_mask
    
    def _create_edit_display(self) -> np.ndarray:
        """创建编辑显示图像"""
        # 使用彩色投影作为背景
        display = self.color_projection.copy()
        
        # 显示未知区域（点云外）为深灰色
        unknown_mask = (self.point_cloud_coverage == 0)
        display[unknown_mask] = [64, 64, 64]
        
        # 半透明覆盖障碍物区域（红色）
        obstacle_mask = (self.obstacle_map == 255)
        display[obstacle_mask] = cv2.addWeighted(
            display[obstacle_mask], 0.5,
            np.full_like(display[obstacle_mask], [0, 0, 180]), 0.5, 0
        )
        
        return display
    
    def interactive_crop(self, window_name: str = "Map Crop Tool"):
        """
        交互式裁剪地图
        
        用户可以用鼠标框选要保留的区域，然后裁剪地图。
        
        Args:
            window_name: OpenCV窗口名称
        """
        if self.obstacle_map is None:
            logger.info("错误: 没有可裁剪的地图！请先加载或创建地图。")
            return False
        
        logger.info("\n=== 交互式地图裁剪工具 ===")
        logger.info("操作说明:")
        logger.info("  1. 鼠标左键拖动: 选择要保留的矩形区域")
        logger.info("  2. 'r'键: 重置选择")
        logger.info("  3. 'c'键: 确认裁剪")
        logger.info("  4. 's'键: 裁剪并保存")
        logger.info("  5. 'q'或ESC: 取消并退出")
        
        # 创建窗口
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # 状态变量
        crop_state = {
            'selecting': False,
            'start_point': None,
            'end_point': None,
            'confirmed': False,
            'cancelled': False
        }
        
        def mouse_callback(event, x, y, flags, param):
            # 获取显示图像的尺寸，以便计算缩放比例
            h, w = param['display_shape'][:2]
            orig_h, orig_w = param['original_shape'][:2]
            
            # 计算缩放比例
            scale_x = orig_w / w
            scale_y = orig_h / h
            
            # 将显示坐标转换为原始图像坐标
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            
            # 限制在图像范围内
            orig_x = max(0, min(orig_w - 1, orig_x))
            orig_y = max(0, min(orig_h - 1, orig_y))
            
            if event == cv2.EVENT_LBUTTONDOWN:
                crop_state['selecting'] = True
                crop_state['start_point'] = (orig_x, orig_y)
                crop_state['end_point'] = (orig_x, orig_y)
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if crop_state['selecting']:
                    crop_state['end_point'] = (orig_x, orig_y)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                crop_state['selecting'] = False
                crop_state['end_point'] = (orig_x, orig_y)
        
        # 设置鼠标回调
        callback_param = {
            'display_shape': None,  # 将在循环中更新
            'original_shape': self.color_projection.shape
        }
        cv2.setMouseCallback(window_name, mouse_callback, callback_param)
        
        # 主循环
        while not crop_state['confirmed'] and not crop_state['cancelled']:
            # 创建显示图像
            display = self.color_projection.copy()
            
            # 显示未知区域为深灰色
            unknown_mask = (self.point_cloud_coverage == 0)
            display[unknown_mask] = [64, 64, 64]
            
            # 半透明显示障碍物（红色）
            obstacle_mask = (self.obstacle_map == 255)
            display[obstacle_mask] = cv2.addWeighted(
                display[obstacle_mask], 0.7,
                np.full_like(display[obstacle_mask], [0, 0, 200]), 0.3, 0
            )
            
            # 绘制选择框
            if crop_state['start_point'] is not None and crop_state['end_point'] is not None:
                x1, y1 = crop_state['start_point']
                x2, y2 = crop_state['end_point']
                
                # 确保坐标顺序正确
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                # 绘制矩形框
                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 绘制半透明遮罩（保留区域外）
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), 
                            (100, 100, 100), -1)
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), 
                            (0, 0, 0), -1)
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
                
                # 重新绘制绿色边框（在遮罩之上）
                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # 显示选择区域的尺寸信息
                width_pixels = x_max - x_min
                height_pixels = y_max - y_min
                width_meters = width_pixels * self.grid_resolution
                height_meters = height_pixels * self.grid_resolution
                
                info_text = f"选择区域: {width_pixels}x{height_pixels} 像素 ({width_meters:.2f}x{height_meters:.2f} 米)"
                cv2.putText(display, info_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 添加提示信息
            cv2.putText(display, "拖动鼠标选择要保留的区域", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "'r':重置 | 'c':确认裁剪 | 's':裁剪并保存 | 'q':取消", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 更新回调参数中的显示尺寸
            callback_param['display_shape'] = display.shape
            
            cv2.imshow(window_name, display)
            
            # 处理键盘输入
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                crop_state['cancelled'] = True
                logger.info("取消裁剪")
                
            elif key == ord('r'):  # 重置
                crop_state['start_point'] = None
                crop_state['end_point'] = None
                logger.info("已重置选择")
                
            elif key == ord('c'):  # 确认裁剪
                if crop_state['start_point'] is not None and crop_state['end_point'] is not None:
                    crop_state['confirmed'] = True
                else:
                    logger.info("请先选择裁剪区域！")
                    
            elif key == ord('s'):  # 裁剪并保存
                if crop_state['start_point'] is not None and crop_state['end_point'] is not None:
                    crop_state['confirmed'] = True
                    # 标记需要保存
                    crop_state['save_after_crop'] = True
                else:
                    logger.info("请先选择裁剪区域！")
        
        cv2.destroyWindow(window_name)
        
        # 如果确认裁剪
        if crop_state['confirmed']:
            x1, y1 = crop_state['start_point']
            x2, y2 = crop_state['end_point']
            
            # 确保坐标顺序正确
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # 执行裁剪
            success = self._crop_map(x_min, y_min, x_max, y_max)
            
            if success:
                logger.info(f"✓ 地图裁剪成功！")
                logger.info(f"  新尺寸: {self.grid_width} x {self.grid_height} 像素")
                logger.info(f"  物理尺寸: {self.grid_width * self.grid_resolution:.2f} x {self.grid_height * self.grid_resolution:.2f} 米")
                
                # 如果需要保存
                if crop_state.get('save_after_crop', False):
                    self.save_map()
                    logger.info("✓ 裁剪后的地图已保存！")
                    
                return True
            else:
                logger.info("✗ 地图裁剪失败！")
                return False
        
        return False
    
    def _crop_map(self, x_min: int, y_min: int, x_max: int, y_max: int) -> bool:
        """
        执行地图裁剪
        
        Args:
            x_min, y_min: 左上角坐标（像素）
            x_max, y_max: 右下角坐标（像素）
            
        Returns:
            True if successful
        """
        try:
            # 验证坐标
            if x_min < 0 or y_min < 0 or x_max > self.grid_width or y_max > self.grid_height:
                logger.info(f"错误: 裁剪坐标超出地图范围！")
                return False
            
            if x_max <= x_min or y_max <= y_min:
                logger.info(f"错误: 无效的裁剪区域！")
                return False
            
            # 裁剪所有地图数据
            self.obstacle_map = self.obstacle_map[y_min:y_max, x_min:x_max].copy()
            self.traversability_mask = self.traversability_mask[y_min:y_max, x_min:x_max].copy()
            self.expanded_traversability = self.expanded_traversability[y_min:y_max, x_min:x_max].copy()
            self.point_cloud_coverage = self.point_cloud_coverage[y_min:y_max, x_min:x_max].copy()
            self.color_projection = self.color_projection[y_min:y_max, x_min:x_max].copy()
            self.binary_projection = self.binary_projection[y_min:y_max, x_min:x_max].copy()
            
            # 更新世界坐标边界
            # 保存旧的边界用于日志
            old_min_pt = self.min_pt.copy()
            old_max_pt = self.max_pt.copy()
            
            # 计算新的世界坐标
            self.min_pt[0] = old_min_pt[0] + x_min * self.grid_resolution
            self.min_pt[1] = old_max_pt[1] - y_max * self.grid_resolution  # Y轴是翻转的
            
            self.max_pt[0] = old_min_pt[0] + x_max * self.grid_resolution
            self.max_pt[1] = old_max_pt[1] - y_min * self.grid_resolution  # Y轴是翻转的
            
            # 更新网格尺寸
            self.grid_height, self.grid_width = self.obstacle_map.shape
            
            logger.info(f"\n=== 裁剪详情 ===")
            logger.info(f"像素坐标: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
            logger.info(f"世界坐标 X: [{self.min_pt[0]:.3f}, {self.max_pt[0]:.3f}] 米")
            logger.info(f"世界坐标 Y: [{self.min_pt[1]:.3f}, {self.max_pt[1]:.3f}] 米")
            logger.info(f"世界坐标 Z: [{self.min_pt[2]:.3f}, {self.max_pt[2]:.3f}] 米")
            
            return True
            
        except Exception as e:
            logger.info(f"裁剪地图时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_map_data(self) -> Dict:
        """获取地图数据（用于其他模块使用）"""
        return {
            "obstacle_map": self.obstacle_map,
            "traversability_mask": self.traversability_mask,
            "expanded_traversability": self.expanded_traversability,
            "point_cloud_coverage": self.point_cloud_coverage,
            "color_projection": self.color_projection,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "grid_resolution": self.grid_resolution,
            "min_pt": self.min_pt,
            "max_pt": self.max_pt,
            "ground_z": self.ground_z,
        }