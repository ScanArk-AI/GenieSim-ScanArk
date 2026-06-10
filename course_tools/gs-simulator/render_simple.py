#!/usr/bin/env python3
#
# 简化的 3D Gaussian Splatting 渲染脚本
# 功能：读取 PLY 格式的 3DGS 模型和相机位姿，输出 RGB 和深度图像（深度图使用 jet 伪彩色）

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from plyfile import PlyData
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera, MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, will use matplotlib for jet colormap")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, will use matplotlib for jet colormap")


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


def load_camera_pose_from_json(json_path):
    """
    从 JSON 文件加载相机位姿
    
    JSON 格式示例：
    {
        "width": 1920,
        "height": 1080,
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 960.0,
        "cy": 540.0,
        "R": [[1,0,0], [0,1,0], [0,0,1]],  // C2W（相机到世界）旋转矩阵
        "T": [0, 0, 5]  // C2W（相机到世界）平移向量（相机在世界坐标系中的位置）
    }
    """
    with open(json_path, 'r') as f:
        data_loaded = json.load(f)
    
    data = data_loaded['cameras']
    width = data['width']
    height = data['height']
    
    # 从内参计算 FOV
    fx = data.get('fx', width / 2)
    fy = data.get('fy', height / 2)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    
    # 旋转矩阵和平移向量
    R = np.array(data['R'], dtype=np.float32)
    T = np.array(data['T'], dtype=np.float32)
    
    return width, height, fovx, fovy, R, T


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


def create_camera_from_pose(width, height, fovx, fovy, R, T, znear=0.01, zfar=100.0):
    """
    从相机参数创建 Camera 对象
    
    Args:
        R: C2W（相机到世界）旋转矩阵
        T: C2W（相机到世界）平移向量（相机在世界坐标系中的位置）
    """
    # 确保 R 和 T 是 numpy 数组
    R = np.array(R, dtype=np.float32)
    T = np.array(T, dtype=np.float32)
    
    # 将 C2W 平移转换为 W2C 平移
    # C2W = [R | T_c2w], W2C = inv(C2W) = [R^T | -R^T @ T_c2w]
    T_w2c = -R.transpose() @ T
    
    # 计算变换矩阵（getWorld2View2 期望 W2C 格式的 T）
    world_view_transform = torch.tensor(getWorld2View2(R, T_w2c), dtype=torch.float32).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    # 创建 MiniCam（简化版相机）
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


def images_to_video(image_dir, output_video_path, fps=30, codec='mp4v'):
    """
    将图像序列转换为视频
    
    Args:
        image_dir: 图像目录路径
        output_video_path: 输出视频路径
        fps: 帧率
        codec: 视频编码（'mp4v' 或 'avc1' for H.264）
    """
    if not CV2_AVAILABLE:
        print(f"Warning: OpenCV not available, cannot create video from {image_dir}")
        return False
    
    # 获取所有图像文件并排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if not image_files:
        print(f"Warning: No PNG images found in {image_dir}")
        return False
    
    # 读取第一张图像以获取尺寸
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Warning: Cannot read image {first_image_path}")
        return False
    
    height, width, _ = first_image.shape
    
    # 创建视频写入器
    # 使用 mp4v 编码（更通用）或 avc1/H.264
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Warning: Cannot open video writer for {output_video_path}")
        return False
    
    # 写入所有帧
    print(f"Creating video from {len(image_files)} images...")
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")
    return True


def render_gaussian_scene(ply_path, camera_poses, output_dir, background_color=[0, 0, 0], sh_degree=3,
                         bilateral_filter=True, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                         create_video=False, video_fps=30):
    """
    渲染 3D Gaussian Splatting 场景
    
    Args:
        ply_path: PLY 文件路径
        camera_poses: 相机位姿列表，每个元素是 (width, height, fovx, fovy, R, T) 或相机 JSON 文件路径
        output_dir: 输出目录
        background_color: 背景颜色 [R, G, B]，范围 0-1
        sh_degree: 球谐函数度数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    depth_filtered_dir = os.path.join(output_dir, "depth_filtered")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_filtered_dir, exist_ok=True)
    
    # 加载 Gaussian 模型
    print(f"Loading Gaussian model from {ply_path}...")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
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
    
    # 渲染每个相机视角
    for idx, pose in enumerate(camera_poses):
        print(f"Rendering camera {idx+1}/{len(camera_poses)}...")
        
        # 解析相机位姿
        if isinstance(pose, str):
            # 如果是文件路径，加载 JSON
            width, height, fovx, fovy, R, T = load_camera_pose_from_json(pose)
        elif isinstance(pose, dict):
            # 如果是字典
            width = pose['width']
            height = pose['height']
            
            # 尝试直接获取 fovx 和 fovy，如果没有则从 fx, fy 计算
            if 'fovx' in pose and 'fovy' in pose:
                fovx = pose['fovx']
                fovy = pose['fovy']
            elif 'fx' in pose and 'fy' in pose:
                fx = pose['fx']
                fy = pose['fy']
                fovx = focal2fov(fx, width)
                fovy = focal2fov(fy, height)
            else:
                # 默认使用宽度/高度的一半作为焦距
                fx = width / 2
                fy = height / 2
                fovx = focal2fov(fx, width)
                fovy = focal2fov(fy, height)
            
            # 转换为 numpy 数组
            R = np.array(pose['R'], dtype=np.float32)
            T = np.array(pose['T'], dtype=np.float32)
        else:
            # 如果是元组
            width, height, fovx, fovy, R, T = pose
        
        # 创建相机
        cam = create_camera_from_pose(width, height, fovx, fovy, R, T)
        
        # 渲染
        with torch.no_grad():
            rendering = render(cam, gaussians, pipeline, bg_color, separate_sh=False)
        
        # 保存 RGB 图像
        rgb_image = rendering["render"].clamp(0, 1)
        rgb_np = (np.array(rgb_image.permute(1, 2, 0).cpu()) * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_np)
        rgb_path = os.path.join(rgb_dir, f"{idx:05d}.png")
        rgb_pil.save(rgb_path)
        
        # 获取深度图像
        depth_image = rendering["depth"]
        depth_np = np.array(depth_image.cpu())
        
        # 如果深度图是 3D 的，取第一个通道或压缩
        if depth_np.ndim == 3:
            depth_np = depth_np[0] if depth_np.shape[0] == 1 else depth_np.squeeze()
        
        # 确保深度图是2D的
        if depth_np.ndim != 2:
            depth_np = depth_np.squeeze()
            if depth_np.ndim != 2:
                print(f"Warning: cannot process depth image shape {depth_np.shape} for camera {idx}")
                continue
        
        # 检查深度图是否有效
        if np.all(np.isnan(depth_np)) or np.all(depth_np == 0) or depth_np.size == 0:
            print(f"Warning: invalid depth image for camera {idx}")
            depth_colored_original = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
            depth_colored_filtered = depth_colored_original.copy()
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
        
        print(f"  Saved RGB: {rgb_path}")
        print(f"  Saved Depth (original): {depth_path_original}")
        print(f"  Saved Depth (filtered): {depth_path_filtered}")
    
    print(f"\nRendering complete! Output saved to {output_dir}")
    print(f"  RGB images: {rgb_dir}")
    print(f"  Depth images (original): {depth_dir}")
    print(f"  Depth images (filtered): {depth_filtered_dir}")
    
    # 创建视频（如果启用）
    if create_video:
        print("\n" + "="*50)
        print("Creating videos...")
        print("="*50)
        
        # RGB 视频
        rgb_video_path = os.path.join(output_dir, "rgb_video.mp4")
        if images_to_video(rgb_dir, rgb_video_path, fps=video_fps, codec='mp4v'):
            print(f"✓ RGB video: {rgb_video_path}")
        
        # 深度视频（原始）
        depth_video_path = os.path.join(output_dir, "depth_video.mp4")
        if images_to_video(depth_dir, depth_video_path, fps=video_fps, codec='mp4v'):
            print(f"✓ Depth video (original): {depth_video_path}")
        
        # 深度视频（滤波后）
        depth_filtered_video_path = os.path.join(output_dir, "depth_filtered_video.mp4")
        if images_to_video(depth_filtered_dir, depth_filtered_video_path, fps=video_fps, codec='mp4v'):
            print(f"✓ Depth video (filtered): {depth_filtered_video_path}")
        
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="渲染 3D Gaussian Splatting 场景")
    parser.add_argument("--ply", "-p", required=True, help="PLY 格式的 3DGS 模型文件路径")
    parser.add_argument("--cameras", "-c", required=True, help="相机位姿 JSON 文件路径（单个文件）或包含多个 JSON 文件的目录")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    parser.add_argument("--background", "-bg", nargs=3, type=float, default=[0, 0, 0], 
                       help="背景颜色 RGB (0-1)，默认 [0, 0, 0] (黑色)")
    parser.add_argument("--sh_degree", type=int, default=3, help="球谐函数度数，默认 3")
    parser.add_argument("--no_bilateral", action="store_true", help="禁用双边滤波")
    parser.add_argument("--bilateral_d", type=int, default=9, help="双边滤波直径，默认 9")
    parser.add_argument("--bilateral_sigma_color", type=float, default=75, help="双边滤波颜色空间标准差，默认 75")
    parser.add_argument("--bilateral_sigma_space", type=float, default=75, help="双边滤波坐标空间标准差，默认 75")
    parser.add_argument("--video", action="store_true", help="将渲染的图像序列保存为视频文件（MP4 格式）")
    parser.add_argument("--video_fps", type=int, default=30, help="视频帧率，默认 30")
    parser.add_argument("--episode_range", nargs=2, type=int, default=None, 
                       help="渲染指定范围的 episodes（起始索引和结束索引，包含），例如 --episode_range 0 10")
    
    args = parser.parse_args()
    
    # 检查 PLY 文件
    if not os.path.exists(args.ply):
        raise FileNotFoundError(f"PLY file not found: {args.ply}")
    
    # 加载相机位姿
    episodes_data = []
    if os.path.isfile(args.cameras):
        # 单个 JSON 文件
        print(f"Loading cameras from {args.cameras}")
        try:
            with open(args.cameras, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError as e:
            raise ValueError(
                f"无法读取文件 {args.cameras}，文件不是有效的文本文件。\n"
                f"错误信息: {str(e)}\n"
                f"请确保提供的是 JSON 文件而不是图片或其他二进制文件。"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"文件 {args.cameras} 不是有效的 JSON 格式。\n"
                f"错误信息: {str(e)}"
            )
        
        # 检查是否是多 episodes 格式
        if 'episodes' in data and 'num_episodes' in data:
            # 多 episodes 格式
            num_episodes = data['num_episodes']
            episodes = data['episodes']
            print(f"Detected multi-episode format: {num_episodes} episodes")
            
            # 应用 episode 范围过滤
            if args.episode_range:
                start_idx, end_idx = args.episode_range
                episodes = episodes[start_idx:end_idx+1]
                print(f"Rendering episodes {start_idx} to {end_idx} ({len(episodes)} episodes)")
            else:
                print(f"Rendering all {len(episodes)} episodes")
            
            for episode in episodes:
                episode_id = episode.get('episode_id', f"episode_{len(episodes_data):04d}")
                camera_poses = episode['cameras']
                episodes_data.append({
                    'episode_id': episode_id,
                    'cameras': camera_poses,
                    'metadata': {k: v for k, v in episode.items() if k not in ['cameras', 'episode_id']}
                })
                print(f"  {episode_id}: {len(camera_poses)} cameras")
        
        # 单 episode 格式（兼容旧格式）
        elif 'cameras' in data:
            camera_poses = data['cameras']
            print(f"Loaded {len(camera_poses)} cameras from JSON file (single episode)")
            episodes_data.append({
                'episode_id': 'single_episode',
                'cameras': camera_poses,
                'metadata': {k: v for k, v in data.items() if k != 'cameras'}
            })
        else:
            raise ValueError(f"No 'cameras' or 'episodes' field found in {args.cameras}")
    
    elif os.path.isdir(args.cameras):
        # 目录，查找所有 JSON 文件（兼容旧格式）
        json_files = sorted(Path(args.cameras).glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {args.cameras}")
        camera_poses = [str(f) for f in json_files]
        print(f"Found {len(camera_poses)} camera pose files (legacy format)")
        episodes_data.append({
            'episode_id': 'single_episode',
            'cameras': camera_poses,
            'metadata': {}
        })
    else:
        raise ValueError(f"Invalid cameras path: {args.cameras}")
    
    # 渲染所有 episodes
    print("\n" + "="*70)
    print(f"Starting batch rendering: {len(episodes_data)} episode(s)")
    print("="*70)
    
    for ep_idx, episode_data in enumerate(episodes_data):
        episode_id = episode_data['episode_id']
        camera_poses = episode_data['cameras']
        metadata = episode_data['metadata']
        
        print(f"\n{'='*70}")
        print(f"Rendering Episode {ep_idx+1}/{len(episodes_data)}: {episode_id}")
        print(f"{'='*70}")
        print(f"  Number of cameras: {len(camera_poses)}")
        if metadata:
            for key, value in metadata.items():
                if key not in ['width', 'height', 'fx', 'fy', 'cx', 'cy']:  # 跳过详细的相机参数
                    print(f"  {key}: {value}")
        
        # 为每个 episode 创建单独的输出目录
        if len(episodes_data) > 1:
            episode_output_dir = os.path.join(args.output, episode_id)
        else:
            episode_output_dir = args.output
        
        # 执行渲染
        render_gaussian_scene(
            ply_path=args.ply,
            camera_poses=camera_poses,
            output_dir=episode_output_dir,
            background_color=args.background,
            sh_degree=args.sh_degree,
            bilateral_filter=not args.no_bilateral,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
            create_video=args.video,
            video_fps=args.video_fps
        )
        
        print(f"\n✓ Episode {episode_id} completed!")
    
    print("\n" + "="*70)
    print(f"All episodes rendered successfully!")
    print(f"Output directory: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()

