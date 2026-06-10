#!/usr/bin/env python3
#
# 从 3D Gaussian Splatting PLY 文件中提取地面点云，并生成俯视图二值化掩码
# 用于机器人路径规划，标识可通行的地面区域

import numpy as np
import argparse
from plyfile import PlyData
from PIL import Image
import os


def load_ply_points(ply_path):
    """
    从PLY文件加载点云坐标
    
    Returns:
        xyz: (N, 3) numpy数组，点云坐标
    """
    print(f"Loading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    x = np.asarray(plydata.elements[0]["x"])
    y = np.asarray(plydata.elements[0]["y"])
    z = np.asarray(plydata.elements[0]["z"])
    
    xyz = np.stack([x, y, z], axis=1)
    
    print(f"Loaded {len(xyz)} points")
    print(f"Point cloud bounds:")
    print(f"  X: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"  Y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    print(f"  Z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
    
    return xyz


def segment_ground_ransac(xyz, distance_threshold=0.1, max_iterations=1000):
    """
    使用RANSAC算法分割地面点云
    
    Args:
        xyz: (N, 3) 点云坐标
        distance_threshold: 点到平面的距离阈值
        max_iterations: RANSAC最大迭代次数
    
    Returns:
        ground_mask: (N,) 布尔数组，True表示地面点
        plane_params: (4,) 平面参数 [a, b, c, d]，满足 ax + by + cz + d = 0
    """
    print(f"\nSegmenting ground using RANSAC...")
    print(f"  Distance threshold: {distance_threshold}")
    print(f"  Max iterations: {max_iterations}")
    
    best_plane = None
    best_inliers = None
    best_inlier_count = 0
    
    # 使用高度最低的点作为初始估计
    z_min_idx = np.argmin(xyz[:, 2])
    z_min = xyz[z_min_idx, 2]
    
    # 随机采样点进行RANSAC
    n_points = len(xyz)
    
    for iteration in range(max_iterations):
        # 随机选择3个点
        sample_indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = xyz[sample_indices]
        
        # 计算平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # 检查法向量是否有效（不为零）
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        
        normal = normal / norm
        
        # 平面方程: ax + by + cz + d = 0
        # 其中 (a, b, c) = normal, d = -normal · p1
        d = -np.dot(normal, p1)
        plane_params = np.array([normal[0], normal[1], normal[2], d])
        
        # 计算所有点到平面的距离
        distances = np.abs(np.dot(xyz, normal) + d)
        
        # 找到内点（距离平面小于阈值的点）
        inliers = distances < distance_threshold
        
        # 检查平面是否大致水平（法向量Z分量应该接近1或-1）
        normal_z_abs = abs(normal[2])
        if normal_z_abs < 0.7:  # 平面倾斜超过45度，不太可能是地面
            continue
        
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane_params
            best_inliers = inliers
    
    if best_plane is None:
        print("Warning: RANSAC failed to find a ground plane, using height-based method")
        return segment_ground_height(xyz), None
    
    print(f"  Found ground plane with {best_inlier_count} inliers ({best_inlier_count/n_points*100:.1f}% of points)")
    print(f"  Plane normal: [{best_plane[0]:.3f}, {best_plane[1]:.3f}, {best_plane[2]:.3f}]")
    
    return best_inliers, best_plane


def segment_ground_bbox_adaptive(xyz, min_coverage=0.05, vertical_axis=2, start_height=0.1, step=0.1, max_height=2.0):
    """
    自适应bounding box地面分割：自动递增阈值直到达到目标覆盖率
    
    Args:
        xyz: (N, 3) 点云坐标
        min_coverage: 目标最小覆盖率（0-1），默认0.05（5%）
        vertical_axis: 垂直轴索引（0=X, 1=Y, 2=Z），默认2（Z轴）
        start_height: 起始高度阈值（米），默认0.1m
        step: 每次递增的步长（米），默认0.1m
        max_height: 最大高度阈值（米），默认2.0m
    
    Returns:
        ground_mask: (N,) 布尔数组，True表示地面点
        final_height: 最终使用的高度阈值
    """
    axis_names = ['X', 'Y', 'Z']
    axis_name = axis_names[vertical_axis]
    
    print(f"\nSegmenting ground using adaptive bounding box method...")
    print(f"  Target coverage: {min_coverage*100:.1f}%")
    print(f"  Using {axis_name}-axis as vertical direction")
    print(f"  Starting from {start_height}m, step {step}m, max {max_height}m")
    
    # 计算bounding box（使用指定的垂直轴）
    vertical_values = xyz[:, vertical_axis]
    v_min = vertical_values.min()
    v_max = vertical_values.max()
    v_range = v_max - v_min
    v_mean = vertical_values.mean()
    v_median = np.median(vertical_values)
    
    print(f"  Point cloud {axis_name} range: [{v_min:.2f}, {v_max:.2f}] (range: {v_range:.2f}m)")
    print(f"  {axis_name} mean: {v_mean:.2f}, {axis_name} median: {v_median:.2f}")
    
    # 从start_height开始，逐步增加阈值
    current_height = start_height
    best_mask = None
    best_coverage = 0.0
    best_height = start_height
    
    while current_height <= max_height:
        ground_threshold = v_min + current_height
        ground_mask = vertical_values <= ground_threshold
        coverage = np.sum(ground_mask) / len(xyz)
        
        print(f"  Trying height {current_height:.2f}m: {np.sum(ground_mask)} points ({coverage*100:.2f}% coverage)")
        
        if coverage >= min_coverage:
            print(f"  [OK] Target coverage reached at {current_height:.2f}m!")
            return ground_mask, current_height
        
        # 记录最好的结果
        if coverage > best_coverage:
            best_coverage = coverage
            best_mask = ground_mask
            best_height = current_height
        
        current_height += step
    
    # 如果达到最大高度仍未满足要求，返回最好的结果
    if best_coverage < min_coverage:
        print(f"  Warning: Maximum height {max_height}m reached, best coverage: {best_coverage*100:.2f}%")
        print(f"  Using best result at {best_height:.2f}m")
    
    return best_mask, best_height


def segment_ground_bbox(xyz, ground_height=0.2, vertical_axis=2):
    """
    基于bounding box分割地面点：最下方指定高度内的点都算作地面
    
    Args:
        xyz: (N, 3) 点云坐标
        ground_height: 地面高度阈值（米），默认0.2米（20cm）
        vertical_axis: 垂直轴索引（0=X, 1=Y, 2=Z），默认2（Z轴）
    
    Returns:
        ground_mask: (N,) 布尔数组，True表示地面点
    """
    axis_names = ['X', 'Y', 'Z']
    axis_name = axis_names[vertical_axis]
    
    print(f"\nSegmenting ground using bounding box method...")
    print(f"  Ground height threshold: {ground_height}m ({ground_height*100:.0f}cm)")
    print(f"  Using {axis_name}-axis as vertical direction")
    
    # 计算bounding box（使用指定的垂直轴）
    vertical_values = xyz[:, vertical_axis]
    v_min = vertical_values.min()
    v_max = vertical_values.max()
    v_range = v_max - v_min
    v_mean = vertical_values.mean()
    v_median = np.median(vertical_values)
    
    print(f"  Point cloud {axis_name} range: [{v_min:.2f}, {v_max:.2f}] (range: {v_range:.2f}m)")
    print(f"  {axis_name} mean: {v_mean:.2f}, {axis_name} median: {v_median:.2f}")
    
    # 最下方ground_height米内的点都算作地面
    ground_threshold = v_min + ground_height
    ground_mask = vertical_values <= ground_threshold
    
    # 统计不同高度范围内的点数
    num_points_bottom_20cm = np.sum(vertical_values <= (v_min + 0.2))
    num_points_bottom_50cm = np.sum(vertical_values <= (v_min + 0.5))
    num_points_bottom_1m = np.sum(vertical_values <= (v_min + 1.0))
    
    print(f"  Ground threshold ({axis_name}): {ground_threshold:.2f}")
    print(f"  Points in bottom 20cm: {num_points_bottom_20cm}")
    print(f"  Points in bottom 50cm: {num_points_bottom_50cm}")
    print(f"  Points in bottom 1m: {num_points_bottom_1m}")
    print(f"  Found {np.sum(ground_mask)} ground points ({np.sum(ground_mask)/len(xyz)*100:.1f}% of points)")
    
    # 如果找到的地面点太少，自动检测垂直轴
    if np.sum(ground_mask) < len(xyz) * 0.01:  # 少于1%的点
        print(f"  Warning: Very few ground points found with {axis_name}-axis.")
        # 尝试自动检测垂直轴（范围最小的轴通常是垂直方向）
        x_range = xyz[:, 0].max() - xyz[:, 0].min()
        y_range = xyz[:, 1].max() - xyz[:, 1].min()
        z_range = xyz[:, 2].max() - xyz[:, 2].min()
        ranges = [x_range, y_range, z_range]
        suggested_axis = np.argmin(ranges)
        if suggested_axis != vertical_axis:
            print(f"  Suggestion: Try using {axis_names[suggested_axis]}-axis (smallest range: {ranges[suggested_axis]:.2f}m)")
            print(f"    Use --vertical_axis {suggested_axis} to switch")
    
    return ground_mask


def segment_ground_height(xyz, height_threshold_percentile=5):
    """
    基于高度阈值分割地面点（备用方法）
    
    Args:
        xyz: (N, 3) 点云坐标
        height_threshold_percentile: 高度阈值百分位数（低于此高度的点被认为是地面）
    
    Returns:
        ground_mask: (N,) 布尔数组，True表示地面点
    """
    print(f"\nSegmenting ground using height threshold (percentile {height_threshold_percentile})...")
    
    z_values = xyz[:, 2]
    height_threshold = np.percentile(z_values, height_threshold_percentile)
    
    # 添加一个小的容差
    tolerance = (z_values.max() - z_values.min()) * 0.05
    ground_mask = z_values <= (height_threshold + tolerance)
    
    print(f"  Height threshold: {height_threshold:.2f}")
    print(f"  Found {np.sum(ground_mask)} ground points ({np.sum(ground_mask)/len(xyz)*100:.1f}% of points)")
    
    return ground_mask


def detect_obstacles_voxel(xyz, voxel_size=0.1, min_points=5, vertical_axis=2):
    """
    使用voxel方法检测障碍物
    
    Args:
        xyz: (N, 3) 点云坐标
        voxel_size: voxel大小（米），默认0.25m
        min_points: 被认为是障碍物的最小点数，默认10
        vertical_axis: 垂直轴索引（0=X, 1=Y, 2=Z）
    
    Returns:
        obstacle_voxels: 字典，key是voxel坐标(x_idx, y_idx, z_idx)，value是True表示障碍物
        voxel_bounds: voxel网格的边界信息
    """
    print(f"\nDetecting obstacles using voxel method...")
    print(f"  Voxel size: {voxel_size}m")
    print(f"  Min points per voxel for obstacle: {min_points}")
    
    # 计算点云边界
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    
    # 计算voxel网格大小
    x_voxels = int(np.ceil((x_max - x_min) / voxel_size)) + 1
    y_voxels = int(np.ceil((y_max - y_min) / voxel_size)) + 1
    z_voxels = int(np.ceil((z_max - z_min) / voxel_size)) + 1
    
    print(f"  Voxel grid size: {x_voxels} x {y_voxels} x {z_voxels}")
    
    # 将点云坐标转换为voxel索引
    x_indices = ((xyz[:, 0] - x_min) / voxel_size).astype(int)
    y_indices = ((xyz[:, 1] - y_min) / voxel_size).astype(int)
    z_indices = ((xyz[:, 2] - z_min) / voxel_size).astype(int)
    
    # 确保索引在有效范围内
    x_indices = np.clip(x_indices, 0, x_voxels - 1)
    y_indices = np.clip(y_indices, 0, y_voxels - 1)
    z_indices = np.clip(z_indices, 0, z_voxels - 1)
    
    # 统计每个voxel内的点数
    from collections import defaultdict
    voxel_counts = defaultdict(int)
    
    for i in range(len(xyz)):
        voxel_key = (x_indices[i], y_indices[i], z_indices[i])
        voxel_counts[voxel_key] += 1
    
    # 标记障碍物voxel（点数超过阈值）
    obstacle_voxels = {}
    total_voxels = len(voxel_counts)
    obstacle_count = 0
    
    for voxel_key, count in voxel_counts.items():
        if count >= min_points:
            obstacle_voxels[voxel_key] = True
            obstacle_count += 1
    
    print(f"  Total voxels with points: {total_voxels}")
    print(f"  Obstacle voxels: {obstacle_count} ({obstacle_count/total_voxels*100:.1f}%)")
    
    voxel_bounds = {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "z_min": float(z_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "z_max": float(z_max),
        "voxel_size": float(voxel_size),
        "x_voxels": int(x_voxels),
        "y_voxels": int(y_voxels),
        "z_voxels": int(z_voxels)
    }
    
    return obstacle_voxels, voxel_bounds


def filter_ground_mask_by_obstacles(mask, bounds, obstacle_voxels, voxel_bounds, ground_z_max, resolution=512, vertical_axis=2):
    """
    根据Z轴正上方的障碍物过滤地面掩码
    
    Args:
        mask: (H, W) 地面掩码
        bounds: 掩码的边界信息
        obstacle_voxels: 障碍物voxel字典
        voxel_bounds: voxel网格边界信息
        ground_z_max: 地面点的最大Z值
        resolution: 掩码分辨率
        vertical_axis: 垂直轴索引
    
    Returns:
        filtered_mask: 过滤后的掩码
    """
    print(f"\nFiltering ground mask by obstacles above...")
    
    # 获取水平轴
    if vertical_axis == 0:  # X是垂直轴
        h1_axis, h2_axis = 1, 2  # Y, Z
        h1_name, h2_name = "y", "z"
    elif vertical_axis == 1:  # Y是垂直轴
        h1_axis, h2_axis = 0, 2  # X, Z
        h1_name, h2_name = "x", "z"
    else:  # Z是垂直轴
        h1_axis, h2_axis = 0, 1  # X, Y
        h1_name, h2_name = "x", "y"
    
    # 获取掩码的水平轴边界
    h1_min_key = f"{h1_name}_min"
    h2_min_key = f"{h2_name}_min"
    h1_min = bounds[h1_min_key]
    h2_min = bounds[h2_min_key]
    scale = bounds["scale"]
    
    voxel_size = voxel_bounds["voxel_size"]
    voxel_h1_min = voxel_bounds[f"{h1_name}_min"]
    voxel_h2_min = voxel_bounds[f"{h2_name}_min"]
    v_axis_name = ["x", "y", "z"][vertical_axis]
    voxel_v_min = voxel_bounds[f"{v_axis_name}_min"]
    voxel_v_voxels = voxel_bounds[f"{v_axis_name}_voxels"]
    
    filtered_mask = mask.copy()
    removed_pixels = 0
    total_ground_pixels = np.sum(mask > 0)
    
    if total_ground_pixels == 0:
        print("  No ground pixels to filter")
        return filtered_mask
    
    # 遍历掩码中的每个像素
    h, w = mask.shape
    
    # 批量处理以提高效率（使用numpy向量化）
    ground_pixel_coords = np.argwhere(mask > 0)
    
    print(f"  Checking {len(ground_pixel_coords)} ground pixels for obstacles above...")
    
    for y, x in ground_pixel_coords:
        # 将像素坐标转换为世界坐标（水平平面）
        h1_world = h1_min + x / scale
        h2_world = h2_min + y / scale
        
        # 转换为voxel索引（水平方向）
        h1_voxel = int((h1_world - voxel_h1_min) / voxel_size)
        h2_voxel = int((h2_world - voxel_h2_min) / voxel_size)
        
        # 确保voxel索引在有效范围内
        h1_voxel = max(0, min(h1_voxel, voxel_bounds[f"{h1_name}_voxels"] - 1))
        h2_voxel = max(0, min(h2_voxel, voxel_bounds[f"{h2_name}_voxels"] - 1))
        
        # 检查从地面垂直值到点云最大垂直值之间的所有voxel
        ground_v_voxel = int((ground_z_max - voxel_v_min) / voxel_size)
        ground_v_voxel = max(0, min(ground_v_voxel, voxel_v_voxels - 1))
        v_max_voxel = voxel_v_voxels - 1
        
        # 检查垂直轴正上方是否有障碍物
        # "上方"意味着垂直轴值更大（对于Z轴，更大的Z值在上方）
        has_obstacle_above = False
        for v_voxel in range(ground_v_voxel + 1, v_max_voxel + 1):
            # 根据垂直轴构建voxel key
            # 注意：voxel key的格式是(x_idx, y_idx, z_idx)，无论vertical_axis是什么
            # 我们需要根据vertical_axis来正确映射
            if vertical_axis == 2:  # Z是垂直轴，voxel key是(x, y, z)
                voxel_key = (h1_voxel, h2_voxel, v_voxel)
            elif vertical_axis == 1:  # Y是垂直轴，voxel key是(x, y, z)，但y是垂直的
                voxel_key = (h1_voxel, v_voxel, h2_voxel)
            else:  # X是垂直轴，voxel key是(x, y, z)，但x是垂直的
                voxel_key = (v_voxel, h1_voxel, h2_voxel)
            
            if voxel_key in obstacle_voxels:
                has_obstacle_above = True
                break
        
        # 如果上方有障碍物，从掩码中移除
        if has_obstacle_above:
            filtered_mask[y, x] = 0
            removed_pixels += 1
    
    print(f"  Removed {removed_pixels} pixels with obstacles above ({removed_pixels/total_ground_pixels*100:.1f}% of ground pixels)")
    print(f"  Remaining ground pixels: {np.sum(filtered_mask>0)} ({np.sum(filtered_mask>0)/(h*w)*100:.1f}% of mask)")
    
    return filtered_mask


def create_top_view_mask(xyz, ground_mask, resolution=512, padding=0.1, vertical_axis=2, 
                         filter_obstacles=False, obstacle_voxel_size=0.1, obstacle_min_points=5, 
                         obstacle_voxels=None, voxel_bounds=None, ground_z_max=None):
    """
    创建俯视图二值化掩码
    
    Args:
        xyz: (N, 3) 点云坐标
        ground_mask: (N,) 布尔数组，标识地面点
        resolution: 输出图像分辨率（像素）
        padding: 边界填充比例（相对于点云范围）
        vertical_axis: 垂直轴索引（0=X, 1=Y, 2=Z）
    
    Returns:
        mask: (resolution, resolution) 二值化掩码，1表示可通行地面，0表示非地面
        bounds: 点云边界信息
    """
    axis_names = ['X', 'Y', 'Z']
    print(f"\nCreating top-view mask (resolution: {resolution}x{resolution})...")
    print(f"  Projecting onto plane perpendicular to {axis_names[vertical_axis]}-axis")
    
    # 根据垂直轴选择投影平面
    if vertical_axis == 0:  # X是垂直轴，投影到YZ平面
        horizontal_axes = [1, 2]  # Y, Z
        axis_labels = ['Y', 'Z']
    elif vertical_axis == 1:  # Y是垂直轴，投影到XZ平面
        horizontal_axes = [0, 2]  # X, Z
        axis_labels = ['X', 'Z']
    else:  # Z是垂直轴，投影到XY平面（默认）
        horizontal_axes = [0, 1]  # X, Y
        axis_labels = ['X', 'Y']
    
    # 获取点云边界（在水平平面上）
    h1_min, h1_max = xyz[:, horizontal_axes[0]].min(), xyz[:, horizontal_axes[0]].max()
    h2_min, h2_max = xyz[:, horizontal_axes[1]].min(), xyz[:, horizontal_axes[1]].max()
    
    h1_range = h1_max - h1_min
    h2_range = h2_max - h2_min
    
    # 添加填充
    h1_padding = h1_range * padding
    h2_padding = h2_range * padding
    
    h1_min -= h1_padding
    h1_max += h1_padding
    h2_min -= h2_padding
    h2_max += h2_padding
    
    # 计算缩放因子（保持宽高比）
    h1_range_padded = h1_max - h1_min
    h2_range_padded = h2_max - h2_min
    
    scale_h1 = resolution / h1_range_padded
    scale_h2 = resolution / h2_range_padded
    scale = min(scale_h1, scale_h2)  # 使用较小的缩放因子以保持宽高比
    
    # 创建掩码
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    
    # 提取地面点
    ground_points = xyz[ground_mask]
    
    if len(ground_points) == 0:
        print("Warning: No ground points found!")
        bounds = {
            f"{axis_labels[0].lower()}_min": h1_min,
            f"{axis_labels[0].lower()}_max": h1_max,
            f"{axis_labels[1].lower()}_min": h2_min,
            f"{axis_labels[1].lower()}_max": h2_max,
            "scale": scale,
            "resolution": resolution,
            "vertical_axis": vertical_axis
        }
        return mask, bounds
    
    # 将地面点投影到图像坐标（使用水平轴）
    h1_img = ((ground_points[:, horizontal_axes[0]] - h1_min) * scale).astype(int)
    h2_img = ((ground_points[:, horizontal_axes[1]] - h2_min) * scale).astype(int)
    
    # 确保坐标在有效范围内
    valid = (h1_img >= 0) & (h1_img < resolution) & (h2_img >= 0) & (h2_img < resolution)
    h1_img = h1_img[valid]
    h2_img = h2_img[valid]
    
    # 在掩码上标记地面点（白色=255=可通行）
    # 注意：图像坐标是(y, x)，所以h2对应行，h1对应列
    mask[h2_img, h1_img] = 255
    
    # 可选：进行形态学操作以填充小洞和去除噪声
    try:
        import cv2
        # 闭运算：先膨胀后腐蚀，填充小洞
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 开运算：先腐蚀后膨胀，去除小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        print("  Applied morphological operations to clean mask")
    except ImportError:
        print("  OpenCV not available, skipping morphological operations")
    
    print(f"  Mask coverage: {np.sum(mask > 0) / (resolution * resolution) * 100:.1f}%")
    
    # 如果启用障碍物过滤，检查垂直轴正上方的障碍物
    if filter_obstacles and obstacle_voxels is not None and voxel_bounds is not None and ground_z_max is not None:
        mask = filter_ground_mask_by_obstacles(
            mask, 
            {
                f"{axis_labels[0].lower()}_min": h1_min,
                f"{axis_labels[0].lower()}_max": h1_max,
                f"{axis_labels[1].lower()}_min": h2_min,
                f"{axis_labels[1].lower()}_max": h2_max,
                "scale": scale,
                "resolution": resolution,
                "vertical_axis": vertical_axis,
                "horizontal_axes": axis_labels
            },
            obstacle_voxels, 
            voxel_bounds, 
            ground_z_max, 
            resolution, 
            vertical_axis
        )
    
    bounds = {
        f"{axis_labels[0].lower()}_min": h1_min,
        f"{axis_labels[0].lower()}_max": h1_max,
        f"{axis_labels[1].lower()}_min": h2_min,
        f"{axis_labels[1].lower()}_max": h2_max,
        "scale": scale,
        "resolution": resolution,
        "vertical_axis": vertical_axis,
        "horizontal_axes": axis_labels
    }
    
    return mask, bounds


def main():
    parser = argparse.ArgumentParser(description="从PLY点云提取地面掩码")
    parser.add_argument("--ply", "-p", required=True, help="PLY文件路径")
    parser.add_argument("--output", "-o", default="./ground_mask.png", help="输出掩码PNG文件路径")
    parser.add_argument("--resolution", "-r", type=int, default=512, help="输出图像分辨率（默认512）")
    parser.add_argument("--method", "-m", choices=["bbox", "bbox_adaptive", "ransac", "height"], default="bbox_adaptive", 
                       help="地面分割方法：bbox（固定阈值）、bbox_adaptive（自适应阈值，直到达到目标覆盖率）、ransac（RANSAC平面拟合）或height（高度阈值）")
    parser.add_argument("--ground_height", type=float, default=0.2, 
                       help="地面高度阈值（米），最下方此高度内的点算作地面（默认0.2，即20cm）。仅在bbox方法中使用")
    parser.add_argument("--min_coverage", type=float, default=0.05,
                       help="目标最小覆盖率（0-1），默认0.05（5%）。仅在bbox_adaptive方法中使用")
    parser.add_argument("--start_height", type=float, default=0.1,
                       help="自适应方法的起始高度阈值（米），默认0.1m")
    parser.add_argument("--step_height", type=float, default=0.1,
                       help="自适应方法的步长（米），默认0.1m")
    parser.add_argument("--max_height", type=float, default=2.0,
                       help="自适应方法的最大高度阈值（米），默认2.0m")
    parser.add_argument("--filter_obstacles", action="store_true",
                       help="启用障碍物过滤：移除Z轴正上方有障碍物的地面区域")
    parser.add_argument("--obstacle_voxel_size", type=float, default=0.1,
                       help="障碍物检测的voxel大小（米），默认0.1m")
    parser.add_argument("--obstacle_min_points", type=int, default=5,
                       help="被认为是障碍物的voxel最小点数，默认5")
    parser.add_argument("--vertical_axis", type=int, choices=[0, 1, 2], default=2,
                       help="垂直轴索引（0=X, 1=Y, 2=Z），默认2（Z轴）")
    parser.add_argument("--distance_threshold", type=float, default=0.1, 
                       help="RANSAC距离阈值（默认0.1）")
    parser.add_argument("--height_percentile", type=float, default=5, 
                       help="高度阈值百分位数（默认5，即最低5%的点）")
    parser.add_argument("--padding", type=float, default=0.1, 
                       help="边界填充比例（默认0.1，即10%）")
    parser.add_argument("--save_bounds", action="store_true", 
                       help="保存边界信息到JSON文件")
    
    args = parser.parse_args()
    
    # 加载点云
    xyz = load_ply_points(args.ply)
    
    # 移除天花板：过滤掉垂直轴高度超过（最大值 - 2m）的点
    print(f"\nRemoving ceiling (points above 2m from top)...")
    vertical_values = xyz[:, args.vertical_axis]
    v_max = vertical_values.max()
    v_min = vertical_values.min()
    ceiling_threshold = v_max - 2.0  # 移除最高点以下2m以上的点
    
    before_count = len(xyz)
    xyz = xyz[vertical_values <= ceiling_threshold]
    after_count = len(xyz)
    
    print(f"  Vertical axis range: [{v_min:.2f}, {v_max:.2f}]")
    print(f"  Ceiling threshold: {ceiling_threshold:.2f} (removing points above this)")
    removed_count = before_count - after_count
    removed_percent = removed_count / before_count * 100
    print(f"  Removed {removed_count} points ({removed_percent:.1f}%)")
    print(f"  Remaining points: {after_count}")
    
    # 分割地面点
    if args.method == "bbox_adaptive":
        ground_mask, final_height = segment_ground_bbox_adaptive(
            xyz, 
            min_coverage=args.min_coverage,
            vertical_axis=args.vertical_axis,
            start_height=args.start_height,
            step=args.step_height,
            max_height=args.max_height
        )
        plane_params = None
        print(f"\nFinal ground height threshold: {final_height:.2f}m")
    elif args.method == "bbox":
        ground_mask = segment_ground_bbox(xyz, ground_height=args.ground_height, vertical_axis=args.vertical_axis)
        plane_params = None
        final_height = args.ground_height
    elif args.method == "ransac":
        ground_mask, plane_params = segment_ground_ransac(
            xyz, 
            distance_threshold=args.distance_threshold
        )
    else:
        ground_mask = segment_ground_height(xyz, height_threshold_percentile=args.height_percentile)
        plane_params = None
    
    # 检测障碍物（如果启用）
    obstacle_voxels = None
    voxel_bounds = None
    ground_z_max = None
    
    if args.filter_obstacles:
        # 检测障碍物voxel
        obstacle_voxels, voxel_bounds = detect_obstacles_voxel(
            xyz,
            voxel_size=args.obstacle_voxel_size,
            min_points=args.obstacle_min_points,
            vertical_axis=args.vertical_axis
        )
        
        # 计算地面点的最大Z值
        ground_points = xyz[ground_mask]
        if len(ground_points) > 0:
            ground_z_max = ground_points[:, args.vertical_axis].max()
            print(f"  Ground points Z max: {ground_z_max:.2f}")
    
    # 创建俯视图掩码
    mask, bounds = create_top_view_mask(
        xyz, 
        ground_mask, 
        resolution=args.resolution,
        padding=args.padding,
        vertical_axis=args.vertical_axis,
        filter_obstacles=args.filter_obstacles,
        obstacle_voxel_size=args.obstacle_voxel_size,
        obstacle_min_points=args.obstacle_min_points,
        obstacle_voxels=obstacle_voxels,
        voxel_bounds=voxel_bounds,
        ground_z_max=ground_z_max
    )
    
    # 保存掩码
    mask_image = Image.fromarray(mask, mode='L')
    mask_image.save(args.output)
    print(f"\nSaved ground mask to: {args.output}")
    
    # 保存边界信息（如果请求）
    if args.save_bounds:
        import json
        bounds_file = args.output.replace('.png', '_bounds.json')
        
        # 构建bounds数据（使用动态键名）
        bounds_data = {
            "scale": float(bounds["scale"]),
            "resolution": bounds["resolution"],
            "pixel_to_meter": float(1.0 / bounds["scale"]),  # 每个像素对应的米数
            "vertical_axis": bounds.get("vertical_axis", args.vertical_axis),
            "horizontal_axes": bounds.get("horizontal_axes", ["X", "Y"])
        }
        
        # 添加水平轴的边界信息
        axis_labels = bounds.get("horizontal_axes", ["X", "Y"])
        for i, axis_label in enumerate(axis_labels):
            bounds_data[f"{axis_label.lower()}_min"] = float(bounds[f"{axis_label.lower()}_min"])
            bounds_data[f"{axis_label.lower()}_max"] = float(bounds[f"{axis_label.lower()}_max"])
        
        if plane_params is not None:
            bounds_data["plane_params"] = [float(x) for x in plane_params.tolist()]
        
        # 添加最终使用的高度阈值（如果是自适应方法）
        if args.method == "bbox_adaptive":
            bounds_data["final_ground_height"] = float(final_height)
            bounds_data["target_coverage"] = args.min_coverage
        
        with open(bounds_file, 'w') as f:
            json.dump(bounds_data, f, indent=2)
        print(f"Saved bounds info to: {bounds_file}")
    
    # 打印使用说明
    print(f"\n使用说明:")
    print(f"  - 白色区域（255）：可通行的地面区域")
    print(f"  - 黑色区域（0）：非地面/障碍物区域")
    print(f"  - 图像分辨率：{args.resolution}x{args.resolution} 像素")
    print(f"  - 每个像素对应：{1.0/bounds['scale']:.3f} 米")


if __name__ == "__main__":
    main()

