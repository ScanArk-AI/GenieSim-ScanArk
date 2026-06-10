#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RenderManager - 3D Gaussian Splatting 渲染管理器
功能：
1. 预加载 Gaussian 模型，避免重复加载
2. 高效渲染多个相机视角
3. 生成 RGB 和深度视频
"""
from __future__ import annotations

import torch
import numpy as np
import json
import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入log管理器
from managers.log_manager import logger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import py360convert
    PY360CONVERT_AVAILABLE = True
except ImportError:
    PY360CONVERT_AVAILABLE = False
    logger.info("警告: py360convert 不可用，全景拼接将使用内置方法")

# 添加 course_tools 内置 gs-simulator 到路径
gs_simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../gs-simulator"))
if os.path.exists(gs_simulator_path):
    sys.path.insert(0, gs_simulator_path)

from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from arguments import PipelineParams
GAUSSIAN_AVAILABLE = True

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.info("警告: OpenCV 不可用，将无法创建视频")

try:
    from diffusers import (
        DiffusionPipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.info("警告: Diffusers 不可用，将无法使用图像质量优化功能")


class RenderManager:
    """3D Gaussian Splatting 渲染管理器"""
    
    def __init__(self, ply_path: str, sh_degree: int = 3, background_color: List[float] = None,
                 enable_diffuser: bool = False, diffuser_model: str = None,
                 diffuser_lora: str = None,
                 diffuser_prompt: str = "high quality, detailed, sharp, professional photography",
                 diffuser_steps: int = 8, diffuser_cfg: float = 1.0):
        """
        初始化渲染管理器并预加载模型
        
        Args:
            ply_path: PLY 文件路径
            sh_degree: 球谐函数度数
            background_color: 背景颜色 [R, G, B]，范围 0-1
            enable_diffuser: 是否启用 diffuser 图像质量优化
            diffuser_model: Diffuser 模型路径（如 "Qwen/Qwen-Image" 或 "Qwen/Qwen-2509"）
            diffuser_lora: Diffuser LoRA 权重路径（可选）
            diffuser_prompt: 用于图像优化的提示词
            diffuser_steps: Diffuser 推理步数（默认8）
            diffuser_cfg: Diffuser CFG scale（默认1.0）
        """
        if not GAUSSIAN_AVAILABLE:
            raise ImportError("Gaussian Splatting 模块不可用，请检查依赖")
        
        self.ply_path = ply_path
        self.sh_degree = sh_degree
        self.background_color = background_color or [0, 0, 0]
        
        # 预加载模型
        logger.info(f"[RenderManager] 正在加载 Gaussian 模型: {ply_path}")
        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.load_ply(ply_path)
        logger.info(f"[RenderManager] ✓ 模型加载完成，共 {self.gaussians.get_xyz.shape[0]} 个 Gaussians")
        
        # 设置背景颜色
        self.bg_color = torch.tensor(self.background_color, dtype=torch.float32, device="cuda")
        
        # 创建管道参数
        import argparse
        parser = argparse.ArgumentParser()
        pipeline_params = PipelineParams(parser)
        self.pipeline = pipeline_params.extract(argparse.Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False
        ))
        
        # 创建渲染锁（用于多线程安全）
        self._render_lock = threading.Lock()
        
        # 初始化 Diffuser（可选）
        self.enable_diffuser = enable_diffuser
        self.diffuser_pipe = None
        self.diffuser_prompt = diffuser_prompt
        self.diffuser_steps = diffuser_steps
        self.diffuser_cfg = diffuser_cfg
        
        if enable_diffuser:
            if not DIFFUSERS_AVAILABLE:
                logger.info("[RenderManager] 警告: Diffusers 不可用，将禁用图像优化")
                self.enable_diffuser = False
            elif diffuser_model is None:
                logger.info("[RenderManager] 警告: 未指定 diffuser_model，将禁用图像优化")
                self.enable_diffuser = False
            else:
                logger.info(f"[RenderManager] 正在加载 Diffuser 模型: {diffuser_model}")
                try:
                    # 检测模型类型
                    if "2509" in diffuser_model or "2511" in diffuser_model:
                        pipe_cls = QwenImageEditPlusPipeline
                    else:
                        pipe_cls = QwenImageEditPipeline
                    
                    # 加载模型
                    if torch.cuda.is_available():
                        torch_dtype = torch.bfloat16
                    else:
                        torch_dtype = torch.float32
                    
                    self.diffuser_pipe = pipe_cls.from_pretrained(
                        diffuser_model, 
                        torch_dtype=torch_dtype
                    )
                    
                    # 如果提供了 LoRA 权重，加载它
                    if diffuser_lora and os.path.exists(diffuser_lora):
                        logger.info(f"[RenderManager] 正在加载 LoRA 权重: {diffuser_lora}")
                        try:
                            self.diffuser_pipe.load_lora_weights(diffuser_lora)
                            logger.info(f"[RenderManager] ✓ LoRA 权重加载完成")
                        except Exception as e:
                            logger.info(f"[RenderManager] 警告: LoRA 权重加载失败: {e}")
                    elif diffuser_lora:
                        logger.info(f"[RenderManager] 警告: LoRA 文件不存在: {diffuser_lora}")
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.diffuser_pipe = self.diffuser_pipe.to(device)
                    
                    logger.info(f"[RenderManager] ✓ Diffuser 模型加载完成")
                except Exception as e:
                    logger.info(f"[RenderManager] 警告: Diffuser 模型加载失败: {e}")
                    self.enable_diffuser = False
        
        logger.info("[RenderManager] ✓ 渲染器初始化完成")
    
    def render_single_view(self, camera_params: Dict, camera_config: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染单个视角
        
        Args:
            camera_params: 相机参数字典，包含 width, height, fx, fy, R, T 等
        
        Returns:
            (rgb_image, depth_image): RGB 图像和深度图，都是 numpy 数组
        """
        # 解析相机参数
        if camera_config is not None:
            width = camera_config['width']
            height = camera_config['height']
            fx = camera_config['fx']
            fy = camera_config['fy']
        else:
            width = camera_params['width']
            height = camera_params['height']
            fx = camera_params.get('fx', width / 2)
            fy = camera_params.get('fy', height / 2)
            
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)
        
        # 获取旋转和平移
        R = np.array(camera_params['R'], dtype=np.float32)
        T = np.array(camera_params['T'], dtype=np.float32)
        
        # 创建相机
        cam = self._create_camera(width, height, fovx, fovy, R, T)
        
        # 渲染
        with torch.no_grad():
            rendering = render(cam, self.gaussians, self.pipeline, self.bg_color, separate_sh=False)
        
        # 提取 RGB 图像
        rgb_image = rendering["render"].clamp(0, 1)
        rgb_np = (np.array(rgb_image.permute(1, 2, 0).cpu()) * 255).astype(np.uint8)
        
        # 提取深度图像
        depth_image = rendering["depth"]
        depth_np = np.array(depth_image.cpu())
        
        # 处理深度图维度
        if depth_np.ndim == 3:
            depth_np = depth_np[0] if depth_np.shape[0] == 1 else depth_np.squeeze()
        
        return rgb_np, depth_np
    
    def apply_diffuser_enhancement(self, rgb_image: np.ndarray, 
                                   custom_prompt: str = None,
                                   strength: float = 0.3) -> np.ndarray:
        """
        使用 Diffuser 优化渲染的图像质量
        
        Args:
            rgb_image: 输入的 RGB 图像 (numpy array, uint8)
            custom_prompt: 自定义提示词（如果为None，使用默认提示词）
            strength: 优化强度（0-1，越大改变越明显，建议0.2-0.4）
        
        Returns:
            优化后的 RGB 图像 (numpy array, uint8)
        """
        if not self.enable_diffuser or self.diffuser_pipe is None:
            return rgb_image
        
        try:
            # 将 numpy 转换为 PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # 准备提示词
            prompt = custom_prompt if custom_prompt is not None else self.diffuser_prompt
            
            # 准备输入参数
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_args = {
                "prompt": prompt,
                "image": pil_image,
                "generator": torch.Generator(device=device).manual_seed(42),
                "num_inference_steps": self.diffuser_steps,
                "true_cfg_scale": self.diffuser_cfg,
                "negative_prompt": "blurry, low quality, distorted, artifacts",
            }
            
            # 如果支持 strength 参数（某些版本的 diffuser 支持）
            # strength 控制保留原图的程度，越小保留越多
            if hasattr(self.diffuser_pipe, 'strength'):
                input_args["strength"] = strength
            
            # 运行优化
            with torch.no_grad():
                result = self.diffuser_pipe(**input_args)
            
            # 转换回 numpy
            enhanced_image = np.array(result.images[0])
            
            return enhanced_image
            
        except Exception as e:
            logger.info(f"[RenderManager] 警告: Diffuser 优化失败: {e}")
            return rgb_image
    
    def render_trajectory(self, camera_poses: List[Dict], output_dir: str,
                         create_video: bool = True, video_fps: int = 10,
                         apply_bilateral_filter: bool = True,
                         apply_diffuser: bool = False,
                         diffuser_strength: float = 0.3,
                         enable_depth: bool = False,
                         camera_config: Dict = None,
                         preview_callback = None) -> bool:
        """
        渲染完整的相机轨迹
        
        Args:
            camera_poses: 相机位姿列表
            output_dir: 输出目录
            create_video: 是否创建视频
            video_fps: 视频帧率
            apply_bilateral_filter: 是否对深度图应用双边滤波
            apply_diffuser: 是否应用 Diffuser 优化图像质量（需要在初始化时启用）
            diffuser_strength: Diffuser 优化强度（0-1，建议0.2-0.4）
            enable_depth: 是否启用深度图
            camera_config: 相机配置
            preview_callback: 可选逐帧预览回调，参数为 (frame_index, rgb, depth_colored)
        Returns:
            成功返回 True，失败返回 False
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_dir = os.path.join(output_dir, "depth")
        depth_filtered_dir = os.path.join(output_dir, "depth_filtered")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(depth_filtered_dir, exist_ok=True)
        
        logger.info(f"[RenderManager] 开始渲染 {len(camera_poses)} 个视角...")
        
        # 渲染每个视角
        camera_iterator = enumerate(camera_poses)
        if TQDM_AVAILABLE:
            camera_iterator = tqdm(
                camera_iterator, 
                total=len(camera_poses),
                desc="Rendering",
                unit="frame",
                ncols=100
            )
        
        for idx, camera_params in camera_iterator:
            # 渲染当前视角
            rgb_np, depth_np = self.render_single_view(camera_params, camera_config)
            
            # 应用 Diffuser 优化（可选）
            if apply_diffuser and self.enable_diffuser:
                rgb_np = self.apply_diffuser_enhancement(rgb_np, strength=diffuser_strength)
            
            # 保存 RGB 图像
            rgb_pil = Image.fromarray(rgb_np)
            rgb_path = os.path.join(rgb_dir, f"{idx:05d}.png")
            rgb_pil.save(rgb_path)
            preview_depth = None

            # 处理深度图
            if enable_depth:
                if not np.all(np.isnan(depth_np)) and not np.all(depth_np == 0):
                    # 保存原始深度数据
                    depth_np_path = os.path.join(depth_dir, f"{idx:05d}.npy")
                    np.save(depth_np_path, depth_np)
                    
                    # 保存伪彩色深度图
                    depth_colored = self._depth_to_jet(depth_np)
                    preview_depth = depth_colored
                    depth_pil = Image.fromarray(depth_colored)
                    depth_path = os.path.join(depth_dir, f"{idx:05d}.png")
                    depth_pil.save(depth_path)
                    
                    # 双边滤波
                    if apply_bilateral_filter and CV2_AVAILABLE:
                        depth_filtered = self._bilateral_filter_depth(depth_np)
                    else:
                        depth_filtered = depth_np.copy()
                    
                    # 保存滤波后的深度
                    depth_filtered_np_path = os.path.join(depth_filtered_dir, f"{idx:05d}.npy")
                    np.save(depth_filtered_np_path, depth_filtered)
                    
                    depth_filtered_colored = self._depth_to_jet(depth_filtered)
                    depth_filtered_pil = Image.fromarray(depth_filtered_colored)
                    depth_filtered_path = os.path.join(depth_filtered_dir, f"{idx:05d}.png")
                    depth_filtered_pil.save(depth_filtered_path)
                else:
                    # 无效深度，保存空图
                    empty_depth = np.zeros((camera_params['height'], camera_params['width'], 3), dtype=np.uint8)
                    preview_depth = empty_depth
                    Image.fromarray(empty_depth).save(os.path.join(depth_dir, f"{idx:05d}.png"))
                    Image.fromarray(empty_depth).save(os.path.join(depth_filtered_dir, f"{idx:05d}.png"))

            if preview_callback is not None:
                preview_callback(idx, rgb_np, preview_depth)

        logger.info(f"[RenderManager] ✓ 渲染完成!")
        
        # 创建视频
        if create_video and CV2_AVAILABLE:
            logger.info(f"[RenderManager] 正在创建视频...")
            self._create_videos(output_dir, video_fps, enable_depth=enable_depth)
            logger.info(f"[RenderManager] ✓ 视频创建完成!")
        
        return True
    
    def render_from_json(self, camera_json_path: str, output_dir: str,
                        create_video: bool = True, video_fps: int = 10,
                        enable_depth: bool = False) -> bool:
        """
        从 JSON 文件读取相机位姿并渲染
        
        Args:
            camera_json_path: 相机位姿 JSON 文件路径
            output_dir: 输出目录
            create_video: 是否创建视频
            video_fps: 视频帧率
            enable_depth: 是否启用深度图
        
        Returns:
            成功返回 True，失败返回 False
        """
        # 读取相机位姿
        with open(camera_json_path, 'r') as f:
            data = json.load(f)
        
        camera_poses = data.get('cameras', [])
        if not camera_poses:
            logger.info(f"[RenderManager] 错误: JSON 文件中没有相机位姿")
            return False
        
        return self.render_trajectory(camera_poses, output_dir, create_video, video_fps, enable_depth=enable_depth)
    
    def _create_camera(self, width: int, height: int, fovx: float, fovy: float,
                      R: np.ndarray, T: np.ndarray, znear: float = 0.01, 
                      zfar: float = 100.0) -> MiniCam:
        """创建相机对象"""
        # 将 C2W 转换为 W2C
        T_w2c = -R.transpose() @ T
        
        # 计算变换矩阵
        world_view_transform = torch.tensor(
            getWorld2View2(R, T_w2c), dtype=torch.float32
        ).transpose(0, 1).cuda()
        
        projection_matrix = getProjectionMatrix(
            znear, zfar, fovx, fovy
        ).transpose(0, 1).cuda()
        
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
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
    
    def _depth_to_jet(self, depth_np: np.ndarray) -> np.ndarray:
        """将深度图转换为 jet 伪彩色"""
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        
        if depth_max - depth_min < 1e-6:
            return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
        
        depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        if CV2_AVAILABLE:
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        else:
            # 简单的伪彩色映射
            depth_colored = np.stack([depth_normalized] * 3, axis=-1)
        
        return depth_colored
    
    def _bilateral_filter_depth(self, depth_np: np.ndarray, d: int = 9,
                                sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """对深度图应用双边滤波"""
        if not CV2_AVAILABLE:
            return depth_np
        
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        
        if depth_max - depth_min < 1e-6:
            return depth_np
        
        # 归一化
        depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min)).astype(np.float32)
        
        # 双边滤波
        sigma_color_normalized = sigma_color / 255.0
        depth_filtered = cv2.bilateralFilter(depth_normalized, d, sigma_color_normalized, sigma_space)
        
        # 恢复原始范围
        depth_filtered = depth_filtered * (depth_max - depth_min) + depth_min
        
        return depth_filtered
    
    def _create_videos(self, output_dir: str, fps: int = 10, enable_depth: bool = False):
        """创建视频文件"""
        if not CV2_AVAILABLE:
            return
        
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_dir = os.path.join(output_dir, "depth")
        depth_filtered_dir = os.path.join(output_dir, "depth_filtered")
        
        # RGB 视频
        rgb_video_path = os.path.join(output_dir, "rgb_video.mp4")
        self._images_to_video(rgb_dir, rgb_video_path, fps)
        
        # Depth 视频
        if enable_depth:
            depth_video_path = os.path.join(output_dir, "depth_video.mp4")
            self._images_to_video(depth_dir, depth_video_path, fps)
            
            # Depth filtered 视频
            depth_filtered_video_path = os.path.join(output_dir, "depth_filtered_video.mp4")
            self._images_to_video(depth_filtered_dir, depth_filtered_video_path, fps)
    
    def _images_to_video(self, image_dir: str, output_video_path: str, fps: int = 10):
        """将图像序列转换为视频（使用FFmpeg编码H.264格式，兼容VSCode播放器）"""
        if not CV2_AVAILABLE:
            return
        
        # 获取所有图像
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            return
        
        # 读取第一张图像获取尺寸
        first_image_path = os.path.join(image_dir, image_files[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            return
        
        height, width, _ = first_image.shape
        
        # 确保尺寸是偶数（H.264要求）
        if width % 2 != 0:
            width += 1
        if height % 2 != 0:
            height += 1
        
        # 查找ffmpeg可执行文件
        ffmpeg_path = None
        possible_paths = [
            'ffmpeg',
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, 
                                      timeout=5)
                if result.returncode == 0:
                    ffmpeg_path = path
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if ffmpeg_path is None:
            logger.warning("未找到FFmpeg，将使用cv2.VideoWriter（可能不兼容VSCode播放器）")
            # 降级到cv2.VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                return
            
            image_iterator = image_files
            if TQDM_AVAILABLE:
                video_name = os.path.basename(output_video_path)
                image_iterator = tqdm(
                    image_files,
                    desc=f"Creating {video_name}",
                    unit="frame",
                    ncols=100,
                    leave=False
                )
            
            for image_file in image_iterator:
                image_path = os.path.join(image_dir, image_file)
                frame = cv2.imread(image_path)
                if frame is not None:
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height))
                    video_writer.write(frame)
            
            video_writer.release()
            return
        
        # 检测FFmpeg支持的编码器
        has_libx264 = False
        try:
            result = subprocess.run([ffmpeg_path, '-codecs'], 
                                  capture_output=True, 
                                  timeout=5,
                                  text=True)
            if 'libx264' in result.stdout:
                has_libx264 = True
        except:
            pass
        
        # 使用FFmpeg命令（根据支持情况选择编码器）
        if has_libx264:
            # 优先使用H.264编码（高质量，VSCode兼容）
            command = [
                ffmpeg_path,
                "-y",  # 覆盖已存在文件
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",  # 从管道读取输入
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "23",
                "-movflags", "+faststart",
                str(output_video_path),
            ]
        else:
            # 降级使用mpeg4编码器（不需要-crf参数）
            logger.info(f"[RenderManager] FFmpeg不支持libx264，使用mpeg4编码器")
            command = [
                ffmpeg_path,
                "-y",  # 覆盖已存在文件
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",  # 从管道读取输入
                "-c:v", "mpeg4",
                "-pix_fmt", "yuv420p",
                "-q:v", "5",  # 质量参数（1-31，越小质量越好）
                "-movflags", "+faststart",
                str(output_video_path),
            ]
        
        try:
            # 开启ffmpeg进程
            process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            
            # 准备进度条
            image_iterator = image_files
            if TQDM_AVAILABLE:
                video_name = os.path.basename(output_video_path)
                image_iterator = tqdm(
                    image_files,
                    desc=f"Creating {video_name}",
                    unit="frame",
                    ncols=100,
                    leave=False
                )
            
            # 写入每一帧
            frame_written = 0
            for image_file in image_iterator:
                # 检查进程是否还在运行
                if process.poll() is not None:
                    logger.error(f"FFmpeg进程提前退出！已写入帧数: {frame_written}")
                    if process.stderr:
                        stderr = process.stderr.read()
                        logger.error(f"FFmpeg stderr: {stderr.decode()}")
                    return
                
                # 读取图像
                image_path = os.path.join(image_dir, image_file)
                frame = cv2.imread(image_path)
                
                if frame is not None:
                    # 调整尺寸以匹配目标尺寸
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height))
                    
                    # 将帧写入ffmpeg的stdin
                    try:
                        process.stdin.write(frame.tobytes())
                        frame_written += 1
                    except BrokenPipeError:
                        logger.error(f"Broken pipe at frame {image_file}, 已写入帧数: {frame_written}")
                        if process.stderr:
                            stderr = process.stderr.read()
                            logger.error(f"FFmpeg stderr: {stderr.decode()}")
                        return
            
            # 关闭管道并等待结束
            try:
                process.stdin.close()
            except:
                pass  # stdin可能已经关闭
            
            # 等待进程结束
            process.wait()
            if process.stderr:
                stderr = process.stderr.read()
            else:
                stderr = b""
            
            if process.returncode == 0:
                logger.info(f"✓ H.264视频已保存 (FFmpeg): {output_video_path}")
            else:
                logger.error(f"FFmpeg返回码: {process.returncode}")
                if stderr:
                    logger.error(f"FFmpeg stderr: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"生成视频时发生异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试读取FFmpeg的错误信息
            try:
                if 'process' in locals() and process and process.poll() is None:
                    process.kill()
                if 'process' in locals() and process and process.stderr:
                    stderr = process.stderr.read()
                    logger.error(f"FFmpeg stderr: {stderr.decode()}")
            except:
                pass
    
    def render_panorama_single_view(self, camera_params: Dict, 
                                    pano_width: int = 2048, 
                                    pano_height: int = 1024,
                                    return_cube_faces: bool = False,
                                    use_multithreading: bool = False,
                                    max_workers: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染单个位置的全景视角（360度）
        
        Args:
            camera_params: 相机位置参数（使用R作为基准朝向，T作为位置）
            pano_width: 全景图宽度（对应360度）
            pano_height: 全景图高度（对应180度）
            return_cube_faces: 是否返回立方体6个面的图像
            use_multithreading: 是否使用多线程渲染（实验性，推荐用于加速）
            max_workers: 最大线程数（推荐2-3个，避免GPU资源竞争）
        
        Returns:
            如果 return_cube_faces=False:
                (panoramic_rgb, panoramic_depth): 全景RGB和深度图
            如果 return_cube_faces=True:
                (panoramic_rgb, panoramic_depth, face_images, face_depths): 全景图和立方体面
        """
        # 获取全景中心位置和基准朝向
        T = np.array(camera_params['T'], dtype=np.float32)
        R_base = np.array(camera_params['R'], dtype=np.float32)
        
        # 定义6个立方体面的参数（前、后、左、右、上、下）
        # 每个面使用90度FOV
        cube_size = 512  # 立方体面的分辨率
        fov = np.pi / 2  # 90度FOV
        
        # 定义相对于基准朝向的旋转
        # 水平旋转：绕世界Z轴（竖直轴）- 使用左乘
        # 垂直旋转：绕相机X轴（右轴）- 使用右乘
        
        # 创建旋转矩阵的辅助函数
        def rot_world_z(angle):
            """绕世界Z轴旋转（水平转动）"""
            c, s = np.cos(angle), np.sin(angle)
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        
        def rot_camera_x(angle):
            """绕相机X轴旋转（上下俯仰）- 在相机坐标系中"""
            c, s = np.cos(angle), np.sin(angle)
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=np.float32)
        
        # 渲染6个面
        face_images = {}
        face_depths = {}
        
        # front: 使用基准朝向
        R_front = R_base
        
        # back: 绕世界Z轴旋转180度（水平转身）
        R_back = rot_world_z(np.pi) @ R_base
        
        # left: 绕世界Z轴旋转90度（水平左转）
        R_left = rot_world_z(np.pi / 2) @ R_base
        
        # right: 绕世界Z轴旋转-90度（水平右转）
        R_right = rot_world_z(-np.pi / 2) @ R_base
        
        # up: 绕相机X轴旋转-90度（抬头看）
        R_up = R_base @ rot_camera_x(np.pi / 2)
        
        # down: 绕相机X轴旋转90度（低头看）
        R_down = R_base @ rot_camera_x(-np.pi / 2)
        
        # 6个面的旋转矩阵字典
        rotations = {
            'front': R_front,
            'back': R_back,
            'left': R_left,
            'right': R_right,
            'up': R_up,
            'down': R_down,
        }
        
        if use_multithreading:
            # 多线程渲染（实验性）
            def render_face(face_name, R):
                """渲染单个面的线程函数"""
                face_cam_params = {
                    'width': cube_size,
                    'height': cube_size,
                    'fx': cube_size / 2,
                    'fy': cube_size / 2,
                    'R': R,
                    'T': T
                }
                
                # 使用锁保护 GPU 渲染操作
                with self._render_lock:
                    rgb, depth = self.render_single_view(face_cam_params)
                
                return face_name, rgb, depth
            
            # 使用线程池
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(render_face, face_name, R): face_name
                    for face_name, R in rotations.items()
                }
                
                # 收集结果
                for future in as_completed(futures):
                    face_name, rgb, depth = future.result()
                    face_images[face_name] = rgb
                    face_depths[face_name] = depth
        else:
            # 原始串行渲染
            for face_name, R in rotations.items():
                
                # 创建该面的相机参数
                face_cam_params = {
                    'width': cube_size,
                    'height': cube_size,
                    'fx': cube_size / 2,
                    'fy': cube_size / 2,
                    'R': R,
                    'T': T
                }
                
                # 渲染该面
                rgb, depth = self.render_single_view(face_cam_params)
                face_images[face_name] = rgb
                face_depths[face_name] = depth
        
        # 将6个面拼接成全景图（等距柱状投影）
        if PY360CONVERT_AVAILABLE:
            # 使用 py360convert 进行拼接
            # 构建立方体贴图字典 (RGB)
            cube_faces_rgb = {
                'F': face_images['front'],
                'R': face_images['right'],
                'B': face_images['back'],
                'L': face_images['left'],
                'U': face_images['up'],
                'D': face_images['down']
            }
            panoramic_rgb = py360convert.c2e(cube_faces_rgb, h=pano_height, w=pano_width, cube_format='dict')
            
            # 构建立方体贴图字典 (Depth)
            # 需要将深度图转换为3通道以供 py360convert 处理，然后提取单通道
            cube_faces_depth = {}
            for key, face_name in [('F', 'front'), ('R', 'right'), ('B', 'back'), 
                                   ('L', 'left'), ('U', 'up'), ('D', 'down')]:
                depth = face_depths[face_name]
                # 将单通道深度图复制为3通道
                if depth.ndim == 2:
                    depth_3ch = np.stack([depth, depth, depth], axis=-1).astype(np.float32)
                else:
                    depth_3ch = depth.astype(np.float32)
                cube_faces_depth[key] = depth_3ch
            
            panoramic_depth_3ch = py360convert.c2e(cube_faces_depth, h=pano_height, w=pano_width, cube_format='dict')
            # 提取单通道
            panoramic_depth = panoramic_depth_3ch[:, :, 0] if panoramic_depth_3ch.ndim == 3 else panoramic_depth_3ch
            
            # 确保数据类型正确
            if panoramic_rgb.dtype != np.uint8:
                # py360convert 可能返回 float 类型，需要转换
                if panoramic_rgb.max() <= 1.0:
                    panoramic_rgb = (panoramic_rgb * 255).astype(np.uint8)
                else:
                    panoramic_rgb = panoramic_rgb.astype(np.uint8)
        else:
            # 使用内置方法进行拼接
            panoramic_rgb = self._cubemap_to_equirectangular(face_images, pano_width, pano_height)
            panoramic_depth = self._cubemap_to_equirectangular_depth(face_depths, pano_width, pano_height)
        
        if return_cube_faces:
            return panoramic_rgb, panoramic_depth, face_images, face_depths
        else:
            return panoramic_rgb, panoramic_depth
    
    def _cubemap_to_equirectangular(self, cube_faces: Dict[str, np.ndarray],
                                   width: int, height: int) -> np.ndarray:
        """
        将立方体贴图转换为等距柱状投影全景图（RGB）
        
        Args:
            cube_faces: 6个面的图像字典
            width: 输出全景图宽度
            height: 输出全景图高度
        
        Returns:
            全景RGB图像
        """
        panorama = np.zeros((height, width, 3), dtype=np.uint8)
        
        for j in range(height):
            for i in range(width):
                # 等距柱状投影坐标 -> 球面坐标
                theta = (i / width) * 2 * np.pi  # 经度 [0, 2π]
                phi = (j / height) * np.pi  # 纬度 [0, π]
                
                # 球面坐标 -> 3D方向向量
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                
                # 确定方向向量对应的立方体面和UV坐标
                face_name, u, v = self._direction_to_cubemap_uv(x, y, z)
                
                # 从立方体面采样
                face_img = cube_faces[face_name]
                face_h, face_w = face_img.shape[:2]
                
                # UV坐标 [-1, 1] -> 像素坐标
                px = int((u + 1) / 2 * face_w)
                py = int((v + 1) / 2 * face_h)
                px = np.clip(px, 0, face_w - 1)
                py = np.clip(py, 0, face_h - 1)
                
                panorama[j, i] = face_img[py, px]
        
        return panorama
    
    def _cubemap_to_equirectangular_depth(self, cube_faces: Dict[str, np.ndarray],
                                         width: int, height: int) -> np.ndarray:
        """
        将立方体贴图转换为等距柱状投影全景深度图
        
        Args:
            cube_faces: 6个面的深度图字典
            width: 输出全景图宽度
            height: 输出全景图高度
        
        Returns:
            全景深度图
        """
        panorama = np.zeros((height, width), dtype=np.float32)
        
        for j in range(height):
            for i in range(width):
                # 等距柱状投影坐标 -> 球面坐标
                theta = (i / width) * 2 * np.pi  # 经度 [0, 2π]
                phi = (j / height) * np.pi  # 纬度 [0, π]
                
                # 球面坐标 -> 3D方向向量
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                
                # 确定方向向量对应的立方体面和UV坐标
                face_name, u, v = self._direction_to_cubemap_uv(x, y, z)
                
                # 从立方体面采样
                face_depth = cube_faces[face_name]
                face_h, face_w = face_depth.shape[:2]
                
                # UV坐标 [-1, 1] -> 像素坐标
                px = int((u + 1) / 2 * face_w)
                py = int((v + 1) / 2 * face_h)
                px = np.clip(px, 0, face_w - 1)
                py = np.clip(py, 0, face_h - 1)
                
                panorama[j, i] = face_depth[py, px]
        
        return panorama
    
    def _direction_to_cubemap_uv(self, x: float, y: float, z: float) -> Tuple[str, float, float]:
        """
        将3D方向向量转换为立方体面名称和UV坐标
        
        Args:
            x, y, z: 归一化的方向向量
        
        Returns:
            (face_name, u, v): 面名称和UV坐标 ([-1, 1]范围)
        """
        abs_x = abs(x)
        abs_y = abs(y)
        abs_z = abs(z)
        
        # 找到最大的分量，确定是哪个面
        if abs_x >= abs_y and abs_x >= abs_z:
            # X轴主导
            if x > 0:
                # +X面（前）
                face_name = 'front'
                u = -y / abs_x
                v = -z / abs_x
            else:
                # -X面（后）
                face_name = 'back'
                u = y / abs_x
                v = -z / abs_x
        elif abs_y >= abs_x and abs_y >= abs_z:
            # Y轴主导
            if y > 0:
                # +Y面（左）
                face_name = 'left'
                u = -x / abs_y
                v = -z / abs_y
            else:
                # -Y面（右）
                face_name = 'right'
                u = x / abs_y
                v = -z / abs_y
        else:
            # Z轴主导
            if z > 0:
                # +Z面（上）
                face_name = 'up'
                u = -y / abs_z
                v = x / abs_z
            else:
                # -Z面（下）
                face_name = 'down'
                u = -y / abs_z
                v = -x / abs_z
        
        return face_name, u, v
    
    def render_panorama_trajectory(self, camera_poses: List[Dict], output_dir: str,
                                  create_video: bool = True, video_fps: int = 10,
                                  pano_width: int = 2048, pano_height: int = 1024,
                                  save_cube_faces: bool = True,
                                  use_multithreading: bool = False,
                                  max_workers: int = 3,
                                  apply_diffuser: bool = False,
                                  diffuser_strength: float = 0.3,
                                  enable_depth: bool = False) -> bool:
        """
        渲染全景相机轨迹
        
        Args:
            camera_poses: 相机位姿列表
            output_dir: 输出目录
            create_video: 是否创建视频
            video_fps: 视频帧率
            pano_width: 全景图宽度
            pano_height: 全景图高度
            save_cube_faces: 是否保存立方体6个面（用于调试）
            use_multithreading: 是否使用多线程加速立方体面渲染（实验性）
            max_workers: 最大线程数（推荐2-3个）
            apply_diffuser: 是否应用 Diffuser 优化图像质量（需要在初始化时启用）
            diffuser_strength: Diffuser 优化强度（0-1，建议0.2-0.4）
            enable_depth: 是否启用深度图
        
        Returns:
            成功返回 True，失败返回 False
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        rgb_dir = os.path.join(output_dir, "rgb_pano")
        depth_dir = os.path.join(output_dir, "depth_pano")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # 如果需要保存立方体面
        if save_cube_faces:
            cube_dir = os.path.join(output_dir, "cube_faces")
            os.makedirs(cube_dir, exist_ok=True)
        
        logger.info(f"[RenderManager] 开始渲染全景视角 {len(camera_poses)} 个位置...")
        logger.info(f"  全景图分辨率: {pano_width} x {pano_height}")
        logger.info(f"  保存立方体面: {save_cube_faces}")
        logger.info(f"  多线程加速: {'启用 (' + str(max_workers) + ' 线程)' if use_multithreading else '禁用'}")
        
        # 渲染每个位置
        camera_iterator = enumerate(camera_poses)
        if TQDM_AVAILABLE:
            camera_iterator = tqdm(
                camera_iterator,
                total=len(camera_poses),
                desc="Rendering Panorama",
                unit="frame",
                ncols=100
            )
        
        for idx, camera_params in camera_iterator:
            # 渲染当前位置的全景
            if save_cube_faces:
                pano_rgb, pano_depth, face_images, face_depths = self.render_panorama_single_view(
                    camera_params, pano_width, pano_height, return_cube_faces=True,
                    use_multithreading=use_multithreading, max_workers=max_workers
                )
                
                # 保存立方体6个面
                frame_cube_dir = os.path.join(cube_dir, f"frame_{idx:05d}")
                os.makedirs(frame_cube_dir, exist_ok=True)
                
                for face_name in ['front', 'back', 'left', 'right', 'up', 'down']:
                    # 保存RGB面
                    face_rgb = face_images[face_name]
                    face_rgb_pil = Image.fromarray(face_rgb)
                    face_rgb_path = os.path.join(frame_cube_dir, f"{face_name}_rgb.png")
                    face_rgb_pil.save(face_rgb_path)
                    
                    # 保存深度面
                    face_depth = face_depths[face_name]
                    if not np.all(np.isnan(face_depth)) and not np.all(face_depth == 0):
                        # 保存原始深度
                        face_depth_npy_path = os.path.join(frame_cube_dir, f"{face_name}_depth.npy")
                        np.save(face_depth_npy_path, face_depth)
                        
                        # 保存伪彩色深度
                        face_depth_colored = self._depth_to_jet(face_depth)
                        face_depth_pil = Image.fromarray(face_depth_colored)
                        face_depth_path = os.path.join(frame_cube_dir, f"{face_name}_depth.png")
                        face_depth_pil.save(face_depth_path)
            else:
                pano_rgb, pano_depth = self.render_panorama_single_view(
                    camera_params, pano_width, pano_height,
                    use_multithreading=use_multithreading, max_workers=max_workers
                )
            
            # 应用 Diffuser 优化（可选）
            if apply_diffuser and self.enable_diffuser:
                pano_rgb = self.apply_diffuser_enhancement(pano_rgb, strength=diffuser_strength)
            
            # 保存RGB全景图
            rgb_pil = Image.fromarray(pano_rgb)
            rgb_path = os.path.join(rgb_dir, f"{idx:05d}.png")
            rgb_pil.save(rgb_path)
            
            # 保存深度全景图
            if enable_depth:
                if not np.all(np.isnan(pano_depth)) and not np.all(pano_depth == 0):
                    # 保存原始深度数据
                    depth_np_path = os.path.join(depth_dir, f"{idx:05d}.npy")
                    np.save(depth_np_path, pano_depth)
                    
                    # 保存伪彩色深度图
                    depth_colored = self._depth_to_jet(pano_depth)
                    depth_pil = Image.fromarray(depth_colored)
                    depth_path = os.path.join(depth_dir, f"{idx:05d}.png")
                    depth_pil.save(depth_path)
                else:
                    # 无效深度
                    empty_depth = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
                    Image.fromarray(empty_depth).save(os.path.join(depth_dir, f"{idx:05d}.png"))
        
        logger.info(f"[RenderManager] ✓ 全景渲染完成!")
        
        # 创建视频
        if create_video and CV2_AVAILABLE:
            logger.info(f"[RenderManager] 正在创建全景视频...")
            
            # RGB视频
            rgb_video_path = os.path.join(output_dir, "rgb_pano_video.mp4")
            self._images_to_video(rgb_dir, rgb_video_path, video_fps)
            
            # Depth视频（仅在启用深度图时创建）
            if enable_depth:
                depth_video_path = os.path.join(output_dir, "depth_pano_video.mp4")
                self._images_to_video(depth_dir, depth_video_path, video_fps)
            
            logger.info(f"[RenderManager] ✓ 全景视频创建完成!")
        
        return True
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'gaussians'):
            del self.gaussians
        if hasattr(self, 'diffuser_pipe') and self.diffuser_pipe is not None:
            del self.diffuser_pipe
        torch.cuda.empty_cache()
        logger.info("[RenderManager] ✓ 资源已清理")


# 便捷函数
def quick_render(ply_path: str, camera_json_path: str, output_dir: str,
                video_fps: int = 10, sh_degree: int = 3,
                enable_diffuser: bool = False, diffuser_model: str = None,
                diffuser_lora: str = None, enable_depth: bool = False) -> bool:
    """
    快速渲染函数
    
    Args:
        ply_path: PLY 文件路径
        camera_json_path: 相机位姿 JSON 路径
        output_dir: 输出目录
        video_fps: 视频帧率
        sh_degree: 球谐函数度数
        enable_diffuser: 是否启用 Diffuser 优化
        diffuser_model: Diffuser 模型路径
        diffuser_lora: Diffuser LoRA 权重路径
        enable_depth: 是否启用深度图
    
    Returns:
        成功返回 True
    """
    try:
        manager = RenderManager(
            ply_path, 
            sh_degree=sh_degree,
            enable_diffuser=enable_diffuser,
            diffuser_model=diffuser_model,
            diffuser_lora=diffuser_lora
        )
        success = manager.render_from_json(camera_json_path, output_dir, 
                                          create_video=True, video_fps=video_fps, enable_depth=enable_depth)
        manager.cleanup()
        return success
    except Exception as e:
        logger.info(f"[RenderManager] 渲染失败: {e}")
        import traceback
        traceback.print_exc()
        return False


"""
使用 Diffuser 优化的示例：

# 示例 1: 基本使用，启用 Diffuser 优化（使用 LoRA 加速）
manager = RenderManager(
    ply_path="path/to/model.ply",
    sh_degree=3,
    enable_diffuser=True,
    diffuser_model="Qwen/Qwen-Image",  # 或 "Qwen/Qwen-2509" 使用更高级的模型
    diffuser_lora="data/qwen-image-lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
    diffuser_prompt="high quality, detailed, sharp, professional photography",
    diffuser_steps=4,  # 使用 LoRA 时可以减少步数
    diffuser_cfg=1.0
)

# 渲染时应用 Diffuser 优化
manager.render_trajectory(
    camera_poses=poses,
    output_dir="output",
    apply_diffuser=True,  # 启用优化
    diffuser_strength=0.3  # 优化强度，0.2-0.4 为推荐值
)

manager.cleanup()

# 示例 2: 全景渲染使用 Diffuser + Lightning LoRA
manager = RenderManager(
    ply_path="path/to/model.ply",
    enable_diffuser=True,
    diffuser_model="Qwen/Qwen-2509",  # 使用 EditPlus 版本
    diffuser_lora="data/qwen-image-lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
    diffuser_steps=4
)

manager.render_panorama_trajectory(
    camera_poses=poses,
    output_dir="output_pano",
    apply_diffuser=True,
    diffuser_strength=0.25,
    use_multithreading=True  # 可以同时使用多线程加速
)

manager.cleanup()

# 示例 3: 快速渲染函数（带 LoRA）
quick_render(
    ply_path="path/to/model.ply",
    camera_json_path="path/to/cameras.json",
    output_dir="output",
    enable_diffuser=True,
    diffuser_model="Qwen/Qwen-Image",
    diffuser_lora="data/qwen-image-lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
)

注意事项：
1. diffuser_strength 推荐值 0.2-0.4，过大会改变原图过多
2. diffuser_steps 推荐值：
   - 不使用 LoRA：50 步（高质量）或 8 步（快速）
   - 使用 Lightning LoRA：4-8 步（快速且高质量）
3. 使用 Diffuser 会显著增加渲染时间：
   - 不使用 LoRA + 50步：每帧约 5-10 秒
   - 使用 Lightning LoRA + 4步：每帧约 1-2 秒
4. 需要额外的 GPU 显存（约 2-4GB）
5. 可用的模型：
   - "Qwen/Qwen-Image": 基础版本
   - "Qwen/Qwen-2509" 或 "Qwen/Qwen-2511": EditPlus 版本，质量更高
6. Lightning LoRA 推荐：
   - Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors
   - 大幅提速，推荐使用 4 步推理
"""

