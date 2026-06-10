#!/usr/bin/env python3
#
# 批量渲染 3D Gaussian Splatting 场景
# 从包含多个相机位姿的JSON文件读取并批量渲染RGB和深度图像

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, will use matplotlib for jet colormap")


def bilateral_filter_depth(depth_np, d=9, sigma_color=75, sigma_space=75):
    """
    对深度图进行双边滤波
    
    Args:
        depth_np: numpy 数组，形状为 (H, W) 的深度图
        d: 滤波时每个像素邻域的直径
        sigma_color: 颜色空间的标准差，控制颜色相似度
        sigma_space: 坐标空间的标准差，控制空间相似度
    
    Returns:
        滤波后的深度图
    """
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available, skipping bilateral filter")
        return depth_np
    
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    
    if depth_max - depth_min < 1e-6:
        return depth_np
    
    # 归一化到0-1范围，转换为float32格式（OpenCV双边滤波支持float32）
    depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min)).astype(np.float32)
    
    # 应用双边滤波（sigma_color需要根据归一化后的范围调整）
    # 将sigma_color从0-255范围转换为0-1范围
    sigma_color_normalized = sigma_color / 255.0
    depth_filtered = cv2.bilateralFilter(depth_normalized, d, sigma_color_normalized, sigma_space)
    
    # 转换回原始范围
    depth_filtered = depth_filtered * (depth_max - depth_min) + depth_min
    
    return depth_filtered


def depth_to_jet_colormap(depth_np):
    """
    将深度图转换为 jet 风格的伪彩色图像
    
    Args:
        depth_np: numpy 数组，形状为 (H, W) 的深度图
    
    Returns:
        RGB 图像，形状为 (H, W, 3)，值范围 0-255
    """
    # 归一化深度值到 0-255
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    
    if depth_max - depth_min < 1e-6:
        # 如果深度值几乎相同，返回全零图像
        depth_normalized = np.zeros_like(depth_np)
    else:
        depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    
    # 使用 OpenCV 的 jet colormap（如果可用）
    if CV2_AVAILABLE:
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        # OpenCV 使用 BGR，转换为 RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    else:
        # 使用 matplotlib 的 jet colormap
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            colormap = cm.get_cmap('jet')
            depth_colored = (colormap(depth_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            # 简单的线性插值实现 jet colormap
            depth_colored = simple_jet_colormap(depth_normalized)
    
    return depth_colored


def simple_jet_colormap(gray):
    """
    简单的 jet colormap 实现（如果 OpenCV 和 matplotlib 都不可用）
    """
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Jet colormap: 蓝色 -> 青色 -> 绿色 -> 黄色 -> 红色
    for i in range(h):
        for j in range(w):
            g = gray[i, j]
            if g < 0.25:
                rgb[i, j, 0] = 0
                rgb[i, j, 1] = int(4 * g * 255)
                rgb[i, j, 2] = 255
            elif g < 0.5:
                rgb[i, j, 0] = 0
                rgb[i, j, 1] = 255
                rgb[i, j, 2] = int((1 + 4 * (0.25 - g)) * 255)
            elif g < 0.75:
                rgb[i, j, 0] = int(4 * (g - 0.5) * 255)
                rgb[i, j, 1] = 255
                rgb[i, j, 2] = 0
            else:
                rgb[i, j, 0] = 255
                rgb[i, j, 1] = int((1 + 4 * (0.75 - g)) * 255)
                rgb[i, j, 2] = 0
    
    return rgb


def load_camera_poses_from_json(json_path):
    """
    从JSON文件加载多个相机位姿
    
    JSON格式：
    {
        "num_cameras": 100,
        "cameras": [
            {
                "width": 1920,
                "height": 1080,
                "fx": 1200.0,
                "fy": 1200.0,
                "cx": 960.0,
                "cy": 540.0,
                "R": [[...], [...], [...]],  // C2W（相机到世界）旋转矩阵
                "T": [...]  // C2W（相机到世界）平移向量（相机在世界坐标系中的位置）
            },
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "cameras" in data:
        # 新格式：包含多个相机的JSON
        cameras = data["cameras"]
        print(f"Loaded {len(cameras)} camera poses from {json_path}")
        return cameras
    else:
        # 旧格式：单个相机或相机列表
        if isinstance(data, list):
            return data
        else:
            return [data]


def create_camera_from_pose(cam_data, znear=0.01, zfar=100.0):
    """
    从相机数据字典创建 Camera 对象
    """
    from utils.graphics_utils import focal2fov
    
    width = cam_data['width']
    height = cam_data['height']
    
    # 计算FOV
    fx = cam_data.get('fx', width / 2)
    fy = cam_data.get('fy', height / 2)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    
    # 旋转矩阵和平移向量
    # R 是 C2W（相机到世界）旋转矩阵
    # T 是 C2W（相机到世界）平移向量（相机在世界坐标系中的位置）
    R = np.array(cam_data['R'], dtype=np.float32)
    T_c2w = np.array(cam_data['T'], dtype=np.float32)
    
    # 将 C2W 平移转换为 W2C 平移
    # C2W = [R | T_c2w], W2C = inv(C2W) = [R^T | -R^T @ T_c2w]
    T_w2c = -R.transpose() @ T_c2w
    
    # 计算变换矩阵（getWorld2View2 期望 W2C 格式的 T）
    world_view_transform = torch.tensor(getWorld2View2(R, T_w2c), dtype=torch.float32).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    # 创建 MiniCam
    cam = MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform
    )
    
    return cam


def render_batch(ply_path, camera_poses_json, output_dir, background_color=[0, 0, 0], sh_degree=3,
                 bilateral_filter=True, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75):
    """
    批量渲染 3D Gaussian Splatting 场景
    
    Args:
        ply_path: PLY 文件路径
        camera_poses_json: 包含多个相机位姿的JSON文件路径
        output_dir: 输出目录
        background_color: 背景颜色 [R, G, B]，范围 0-1
        sh_degree: 球谐函数度数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    depth_filtered_dir = os.path.join(output_dir, "depth_filtered")
    rgbd_dir = os.path.join(output_dir, "RGBD")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_filtered_dir, exist_ok=True)
    os.makedirs(rgbd_dir, exist_ok=True)
    
    # 加载 Gaussian 模型
    print(f"Loading Gaussian model from {ply_path}...")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # 加载相机位姿
    camera_poses = load_camera_poses_from_json(camera_poses_json)
    print(f"Rendering {len(camera_poses)} views...")
    
    # 设置背景颜色
    bg_color = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    
    # 创建管道参数
    from arguments import PipelineParams
    import argparse
    parser = argparse.ArgumentParser()
    pipeline_params = PipelineParams(parser)
    pipeline = pipeline_params.extract(argparse.Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False
    ))
    
    # 批量渲染
    from tqdm import tqdm
    for idx, cam_data in enumerate(tqdm(camera_poses, desc="Rendering")):
        try:
            # 创建相机
            cam = create_camera_from_pose(cam_data)
            
            # 渲染
            with torch.no_grad():
                rendering = render(cam, gaussians, pipeline, bg_color, separate_sh=False)
            
            # 保存 RGB 图像
            rgb_image = rendering["render"].clamp(0, 1)
            rgb_np = (rgb_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_pil = Image.fromarray(rgb_np)
            rgb_path = os.path.join(rgb_dir, f"{idx:05d}.png")
            rgb_pil.save(rgb_path)
            
            # 获取深度图像并转换为 jet 伪彩色
            depth_image = rendering["depth"]
            depth_np = depth_image.cpu().numpy()
            
            # 如果深度图是 3D 的，取第一个通道或压缩
            if depth_np.ndim == 3:
                depth_np = depth_np[0] if depth_np.shape[0] == 1 else depth_np.squeeze()
            
            # 确保深度图是2D的
            if depth_np.ndim != 2:
                if depth_np.ndim == 1:
                    # 如果是1D，需要reshape（这种情况不应该发生）
                    print(f"Warning: depth image is 1D for camera {idx}, skipping")
                    continue
                else:
                    depth_np = depth_np.squeeze()
                    if depth_np.ndim != 2:
                        print(f"Warning: cannot process depth image shape {depth_np.shape} for camera {idx}, skipping")
                        continue
            
            # 检查深度图是否有效
            if np.all(np.isnan(depth_np)) or np.all(depth_np == 0) or depth_np.size == 0:
                print(f"Warning: invalid depth image for camera {idx}")
                depth_colored_original = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
                depth_colored_filtered = depth_colored_original.copy()
                depth_np_original = depth_np.copy()
                depth_np_filtered = depth_np.copy()
            else:
                # 保存滤波前的深度图（jet伪彩色）
                depth_colored_original = depth_to_jet_colormap(depth_np)
                
                # 保存滤波前的原始深度数据
                depth_np_original = depth_np.copy()
                
                # 对深度图进行双边滤波（如果启用）
                if bilateral_filter:
                    depth_np_filtered = bilateral_filter_depth(
                        depth_np, 
                        d=bilateral_d, 
                        sigma_color=bilateral_sigma_color, 
                        sigma_space=bilateral_sigma_space
                    )
                    # 保存滤波后的深度图（jet伪彩色）
                    depth_colored_filtered = depth_to_jet_colormap(depth_np_filtered)
                else:
                    # 如果未启用滤波，滤波后的图与原始图相同
                    depth_np_filtered = depth_np.copy()
                    depth_colored_filtered = depth_colored_original.copy()
            
            # 保存滤波前的伪彩色深度图
            depth_pil_original = Image.fromarray(depth_colored_original)
            depth_path_original = os.path.join(depth_dir, f"{idx:05d}.png")
            depth_pil_original.save(depth_path_original)
            
            # 保存滤波前的原始深度数据为 numpy 格式
            depth_np_path_original = os.path.join(depth_dir, f"{idx:05d}.npy")
            np.save(depth_np_path_original, depth_np_original)
            
            # 保存滤波后的伪彩色深度图
            depth_pil_filtered = Image.fromarray(depth_colored_filtered)
            depth_path_filtered = os.path.join(depth_filtered_dir, f"{idx:05d}.png")
            depth_pil_filtered.save(depth_path_filtered)
            
            # 保存滤波后的原始深度数据为 numpy 格式
            depth_np_path_filtered = os.path.join(depth_filtered_dir, f"{idx:05d}.npy")
            np.save(depth_np_path_filtered, depth_np_filtered)
            
            # 水平拼接 RGB 和滤波后的伪彩色深度图
            # 确保深度图和RGB图的高度一致（如果不同，调整深度图高度）
            if rgb_np.shape[0] != depth_colored_filtered.shape[0]:
                print(f"Warning: RGB height {rgb_np.shape[0]} != depth height {depth_colored_filtered.shape[0]}, resizing depth")
                depth_pil_temp = Image.fromarray(depth_colored_filtered)
                depth_pil_temp = depth_pil_temp.resize((depth_colored_filtered.shape[1], rgb_np.shape[0]), Image.Resampling.LANCZOS)
                depth_colored_filtered = np.array(depth_pil_temp)
            
            rgbd_image = np.concatenate([rgb_np, depth_colored_filtered], axis=1)
            rgbd_pil = Image.fromarray(rgbd_image)
            rgbd_path = os.path.join(rgbd_dir, f"{idx:05d}.png")
            rgbd_pil.save(rgbd_path)
            
            print(f"  Saved RGB: {rgb_path}")
            print(f"  Saved Depth (original): {depth_path_original}")
            print(f"  Saved Depth (filtered): {depth_path_filtered}")
            print(f"  Saved RGBD: {rgbd_path}")
            
        except Exception as e:
            print(f"\nError rendering camera {idx}: {e}")
            continue
    
    print(f"\nRendering complete! Output saved to {output_dir}")
    print(f"  RGB images: {rgb_dir}")
    print(f"  Depth images (original): {depth_dir}")
    print(f"  Depth images (filtered): {depth_filtered_dir}")
    print(f"  RGBD images: {rgbd_dir}")


def main():
    parser = argparse.ArgumentParser(description="批量渲染 3D Gaussian Splatting 场景")
    parser.add_argument("--ply", "-p", required=True, help="PLY 格式的 3DGS 模型文件路径")
    parser.add_argument("--cameras", "-c", required=True, help="包含多个相机位姿的JSON文件路径")
    parser.add_argument("--output", "-o", default="./output_batch", help="输出目录")
    parser.add_argument("--background", "-bg", nargs=3, type=float, default=[0, 0, 0], 
                       help="背景颜色 RGB (0-1)，默认 [0, 0, 0] (黑色)")
    parser.add_argument("--sh_degree", type=int, default=3, help="球谐函数度数，默认 3")
    parser.add_argument("--no_bilateral", action="store_true", help="禁用双边滤波")
    parser.add_argument("--bilateral_d", type=int, default=9, help="双边滤波直径，默认 9")
    parser.add_argument("--bilateral_sigma_color", type=float, default=75, help="双边滤波颜色空间标准差，默认 75")
    parser.add_argument("--bilateral_sigma_space", type=float, default=75, help="双边滤波坐标空间标准差，默认 75")
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.ply):
        raise FileNotFoundError(f"PLY file not found: {args.ply}")
    if not os.path.exists(args.cameras):
        raise FileNotFoundError(f"Camera poses file not found: {args.cameras}")
    
    # 执行批量渲染
    render_batch(
        ply_path=args.ply,
        camera_poses_json=args.cameras,
        output_dir=args.output,
        background_color=args.background,
        sh_degree=args.sh_degree,
        bilateral_filter=not args.no_bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space
    )


if __name__ == "__main__":
    main()

