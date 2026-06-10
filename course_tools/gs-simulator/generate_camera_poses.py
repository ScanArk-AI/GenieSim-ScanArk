#!/usr/bin/env python3
#
# 生成多个相机位姿的脚本
# 从 PLY 文件读取点云范围，生成在点云内部的相机位姿

import numpy as np
import json
import argparse
from plyfile import PlyData
from utils.graphics_utils import focal2fov


def get_point_cloud_bounds(ply_path, return_xyz=False):
    """
    从 PLY 文件获取点云的边界框
    
    Returns:
        min_bounds: [min_x, min_y, min_z]
        max_bounds: [max_x, max_y, max_z]
        center: [center_x, center_y, center_z]
        xyz: (可选) 所有点的坐标数组
    """
    print(f"Reading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    x = np.asarray(plydata.elements[0]["x"])
    y = np.asarray(plydata.elements[0]["y"])
    z = np.asarray(plydata.elements[0]["z"])
    
    xyz = np.stack([x, y, z], axis=1)
    
    min_bounds = xyz.min(axis=0)
    max_bounds = xyz.max(axis=0)
    center = xyz.mean(axis=0)
    
    print(f"Point cloud bounds:")
    print(f"  X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}]")
    print(f"  Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}]")
    print(f"  Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}]")
    print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"  Total points: {len(xyz)}")
    
    if return_xyz:
        return min_bounds, max_bounds, center, xyz
    return min_bounds, max_bounds, center


def count_gaussians_in_frustum(cam_pos, look_at, fovx, fovy, znear, zfar, gaussian_positions):
    """
    粗略估计相机视锥内的高斯球数量
    
    Args:
        cam_pos: 相机位置
        look_at: 相机看向的点
        fovx, fovy: 水平和垂直视场角
        znear, zfar: 近远平面
        gaussian_positions: 所有高斯球的位置 (N, 3)
    
    Returns:
        视锥内的高斯球数量
    """
    # 计算相机方向
    direction = look_at - cam_pos
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # 计算所有高斯球相对于相机的位置
    gaussian_relative = gaussian_positions - cam_pos
    
    # 计算到相机的距离
    distances = np.linalg.norm(gaussian_relative, axis=1)
    
    # 距离过滤
    mask = (distances >= znear) & (distances <= zfar)
    
    if not np.any(mask):
        return 0
    
    # 计算方向（归一化）
    gaussian_directions = gaussian_relative[mask] / (distances[mask, np.newaxis] + 1e-8)
    
    # 计算与相机朝向的夹角
    cos_angles = np.dot(gaussian_directions, direction)
    
    # 计算视锥的半角（取较大的FOV）
    half_fov = max(fovx, fovy) / 2
    cos_half_fov = np.cos(half_fov)
    
    # 在视锥内的高斯球（角度小于FOV）
    in_frustum = cos_angles > cos_half_fov
    
    return np.sum(in_frustum)


def generate_camera_poses(num_cameras, min_bounds, max_bounds, center, 
                         width=1920, height=1080, fx=1200.0, fy=1200.0,
                         gaussian_positions=None, min_gaussians_in_view=1000):
    """
    生成多个相机位姿，确保相机视野内包含大量高斯球
    
    策略：将相机放在点云外部，朝向点云内部，确保能看到点云内容
    
    Args:
        num_cameras: 要生成的相机数量
        min_bounds: 点云最小边界
        max_bounds: 点云最大边界
        center: 点云中心
        width, height: 图像分辨率
        fx, fy: 焦距
    
    Returns:
        List of camera pose dictionaries
    """
    cameras = []
    
    # 计算点云尺寸
    size = max_bounds - min_bounds
    max_size = max(size)
    diagonal = np.linalg.norm(size)
    
    # 相机应该放在点云外部，距离点云边缘一定距离
    # 距离点云边缘的距离（基于点云对角线长度）
    camera_distance_factor = 0.5  # 相机距离点云边缘的距离因子
    camera_distance = diagonal * camera_distance_factor
    
    # 扩展的边界（相机可以放置的范围）
    extended_min = min_bounds - size * camera_distance_factor
    extended_max = max_bounds + size * camera_distance_factor
    
    # 生成相机位置
    np.random.seed(42)  # 可重复的结果
    
    print(f"Generating {num_cameras} camera poses...")
    print(f"Camera placement range (extended):")
    print(f"  X: [{extended_min[0]:.2f}, {extended_max[0]:.2f}]")
    print(f"  Y: [{extended_min[1]:.2f}, {extended_max[1]:.2f}]")
    print(f"  Z: [{extended_min[2]:.2f}, {extended_max[2]:.2f}]")
    
    # 计算FOV（提前计算，用于验证）
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    znear = 0.01
    zfar = diagonal * 3  # 远平面设置为点云对角线的3倍
    
    generated = 0
    attempts = 0
    max_attempts = num_cameras * 20  # 增加尝试次数，确保找到好的视角
    
    while generated < num_cameras and attempts < max_attempts:
        attempts += 1
        
        # 在扩展边界内随机生成相机位置
        cam_pos = np.random.uniform(extended_min, extended_max)
        
        # 计算相机到点云中心的距离
        dist_to_center = np.linalg.norm(cam_pos - center)
        
        # 确保相机在点云外部（距离中心足够远）
        min_dist = diagonal * 0.4  # 最小距离（稍微增加，确保能看到更多点云）
        max_dist = diagonal * 1.2  # 最大距离
        
        if dist_to_center < min_dist or dist_to_center > max_dist:
            continue
        
        # 计算相机朝向点云内部的某个点（不是中心，而是中心附近的随机点）
        # 这样可以增加视角多样性
        look_at_offset = size * 0.2 * (np.random.rand(3) - 0.5)  # 在中心附近随机偏移
        look_at = center + look_at_offset
        
        # 确保look_at在点云内部
        look_at = np.clip(look_at, min_bounds, max_bounds)
        
        # 计算朝向look_at的方向
        direction = look_at - cam_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 构建相机坐标系
        # 上向量（假设Y向上，但可以有一些随机倾斜）
        up_base = np.array([0, 1, 0])
        # 添加一些随机倾斜（±15度）
        tilt = np.random.uniform(-0.15, 0.15)
        up = up_base + np.array([0, 0, tilt])
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # 右向量
        right = np.cross(direction, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # 如果方向几乎平行于上向量，使用另一个上向量
            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # 重新计算上向量（确保正交）
        up = np.cross(right, direction)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # 构建 C2W（相机到世界）旋转矩阵
        # R 的每一列是相机坐标系在世界坐标系中的基向量
        R = np.column_stack([right, up, -direction])  # 注意：direction取负因为相机看向-Z方向
        
        # T 是 C2W 平移向量（相机在世界坐标系中的位置）
        T = cam_pos
        
        # 如果提供了高斯球位置，验证视野内的高斯球数量
        if gaussian_positions is not None:
            gaussian_count = count_gaussians_in_frustum(
                cam_pos, look_at, fovx, fovy, znear, zfar, gaussian_positions
            )
            if gaussian_count < min_gaussians_in_view:
                continue  # 视野内高斯球太少，跳过这个相机
        
        camera_pose = {
            "width": int(width),
            "height": int(height),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(width / 2),
            "cy": float(height / 2),
            "R": R.tolist(),
            "T": T.tolist(),
            "position": cam_pos.tolist(),
            "look_at": look_at.tolist(),
            "distance_to_center": float(dist_to_center)
        }
        
        cameras.append(camera_pose)
        generated += 1
        
        if generated % 100 == 0:
            print(f"  Generated {generated}/{num_cameras} cameras... (attempts: {attempts})")
    
    if generated < num_cameras:
        print(f"Warning: Only generated {generated} cameras out of {num_cameras} requested")
    
    return cameras


def main():
    parser = argparse.ArgumentParser(description="生成相机位姿JSON文件")
    parser.add_argument("--ply", "-p", required=True, help="PLY 文件路径")
    parser.add_argument("--output", "-o", default="camera_poses_100.json", help="输出JSON文件路径")
    parser.add_argument("--num", "-n", type=int, default=100, help="要生成的相机数量")
    parser.add_argument("--width", type=int, default=1920, help="图像宽度")
    parser.add_argument("--height", type=int, default=1080, help="图像高度")
    parser.add_argument("--fx", type=float, default=1200.0, help="焦距x")
    parser.add_argument("--fy", type=float, default=1200.0, help="焦距y")
    
    parser.add_argument("--verify", action="store_true", help="验证每个相机视野内的高斯球数量（较慢）")
    parser.add_argument("--min_gaussians", type=int, default=1000, help="每个相机视野内最少的高斯球数量（仅在使用--verify时有效）")
    
    args = parser.parse_args()
    
    # 获取点云边界
    if args.verify:
        min_bounds, max_bounds, center, xyz = get_point_cloud_bounds(args.ply, return_xyz=True)
        gaussian_positions = xyz
    else:
        min_bounds, max_bounds, center = get_point_cloud_bounds(args.ply)
        gaussian_positions = None
    
    # 生成相机位姿
    print(f"\nGenerating {args.num} camera poses...")
    if args.verify:
        print(f"Verification enabled: ensuring at least {args.min_gaussians} Gaussians in each view")
    cameras = generate_camera_poses(
        args.num, min_bounds, max_bounds, center,
        args.width, args.height, args.fx, args.fy,
        gaussian_positions=gaussian_positions,
        min_gaussians_in_view=args.min_gaussians if args.verify else 0
    )
    
    # 保存为JSON
    output_data = {
        "num_cameras": args.num,
        "point_cloud_bounds": {
            "min": min_bounds.tolist(),
            "max": max_bounds.tolist(),
            "center": center.tolist()
        },
        "cameras": cameras
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {args.num} camera poses to {args.output}")
    print(f"Camera positions range:")
    positions = np.array([cam["position"] for cam in cameras])
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")


if __name__ == "__main__":
    main()

