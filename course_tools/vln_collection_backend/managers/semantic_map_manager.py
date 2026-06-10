#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemanticMapManager - 语义地图管理类
功能:
1. 在占用网格地图基础上生成语义地图
2. 集成语义分割/检测模型（GroundingDINO / YOLO-World 等），识别场景中的物体
3. 将检测到的物体信息（类别、3D坐标、置信度）存储并索引
4. 支持按物体类别查询2D像素坐标与3D世界坐标
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any, Union
import pickle
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy

try:
    import open3d as o3d
except ImportError:
    import sys
    print("错误: 需要安装 open3d 库")
    sys.exit(1)
    
# 导入log管理器
from managers.log_manager import logger
# 导入占用地图管理器作为基类
from managers.occupancy_map_manager import OccupancyMapManager


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class SemanticObject:
    """
    语义物体信息载体

    Attributes:
        object_id       : 场景内唯一 ID
        category        : 物体类别名称（如 "chair", "table"）
        confidence      : 检测置信度 [0, 1]
        center_world    : 世界坐标系中心点 [x, y, z]（米）
        center_pixel    : 语义地图中的像素坐标 (col, row)
        bbox_pixel      : 像素边界框 (x_min, y_min, x_max, y_max)
        mask_pixel      : 二值分割掩码（与地图同尺寸），可为 None
        source_image_id : 来源图像 ID / 帧号
        additional_info : 其他扩展信息字典
    """
    object_id: int
    category: str
    confidence: float
    center_world: np.ndarray                     # shape (3,)
    center_pixel: Tuple[int, int]                # (col, row)
    bbox_pixel: Tuple[int, int, int, int]        # (x_min, y_min, x_max, y_max)
    mask_pixel: Optional[np.ndarray] = None
    source_image_id: Optional[Any] = None
    additional_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """序列化为普通字典（可 JSON 导出）"""
        return {
            "object_id":       self.object_id,
            "category":        self.category,
            "confidence":      float(self.confidence),
            "center_world":    self.center_world.tolist()
                               if isinstance(self.center_world, np.ndarray)
                               else list(self.center_world),
            "center_pixel":    list(self.center_pixel),
            "bbox_pixel":      list(self.bbox_pixel),
            "source_image_id": self.source_image_id,
            "additional_info": self.additional_info,
        }

    def __repr__(self) -> str:
        wx, wy, wz = self.center_world
        return (f"SemanticObject(id={self.object_id}, category='{self.category}', "
                f"conf={self.confidence:.2f}, world=({wx:.2f},{wy:.2f},{wz:.2f}))")


# ============================================================
# 语义检测模型后端（策略模式）
# ============================================================

class SemanticModelBackend(ABC):
    """语义检测后端抽象基类"""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        categories: Optional[List[str]] = None,
        text_prompt: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        对单张 BGR 图像进行语义检测
        
        Returns:
            检测结果列表，每项格式：
            {
                "category": str,
                "confidence": float,
                "bbox": [x_min, y_min, x_max, y_max],  # 像素坐标
                "mask": np.ndarray or None,             # 与 image 同尺寸的二值掩码
            }
        """
        raise NotImplementedError


class GroundingDINOBackend(SemanticModelBackend):
    """
    Grounding DINO 检测后端（开放词汇、文本驱动）
    依赖: pip install groundingdino-py
    """

    def __init__(self, model_config_path: str = None, model_checkpoint_path: str = None,
                 device: str = "cuda"):
        try:
            from groundingdino.util.inference import load_model, predict
            from groundingdino.util import box_ops
        except ImportError:
            raise ImportError(
                "未找到 groundingdino，请安装: pip install groundingdino-py"
            )
        import torch
        self.device = device
        logger.info("正在加载 Grounding DINO 模型...")
        self.model = load_model(model_config_path, model_checkpoint_path)
        self.model.to(device)
        self.predict_fn = predict
        self.box_ops = box_ops
        logger.info("✓ Grounding DINO 加载完成")

    def detect(
        self,
        image: np.ndarray,
        categories: Optional[List[str]] = None,
        text_prompt: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ) -> List[Dict]:
        import torch
        from groundingdino.util.inference import load_image_tensor

        if text_prompt is None and categories is not None:
            text_prompt = " . ".join(categories) + " ."
        elif text_prompt is None:
            raise ValueError("categories 或 text_prompt 至少提供一个")

        # BGR -> RGB -> tensor
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image_tensor = load_image_tensor(rgb)

        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            device=self.device,
        )

        results = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
            results.append({
                "category":   phrase.strip(),
                "confidence": float(logit),
                "bbox":       [x_min, y_min, x_max, y_max],
                "mask":       None,
            })
        return results


class YOLOWorldBackend(SemanticModelBackend):
    """
    YOLO-World 检测后端（轻量高速、开放词汇）
    依赖: pip install ultralytics>=8.1.0
    """

    def __init__(self, model_path: str = "yolov8x-worldv2.pt", device: str = "cuda"):
        try:
            from ultralytics import YOLOWorld as _YOLOWorld
        except ImportError:
            raise ImportError(
                "未找到 ultralytics，请安装: pip install ultralytics>=8.1.0"
            )
        logger.info(f"正在加载 YOLO-World 模型: {model_path}")
        self.model = _YOLOWorld(model_path)
        self.device = device
        logger.info("✓ YOLO-World 加载完成")

    def set_categories(self, categories: List[str]):
        self.model.set_classes(categories)

    def detect(
        self,
        image: np.ndarray,
        categories: Optional[List[str]] = None,
        text_prompt: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ) -> List[Dict]:
        if categories:
            self.model.set_classes(categories)
        results = self.model.predict(image, conf=confidence_threshold,
                                     device=self.device, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cat = (categories[cls_id]
                       if categories and cls_id < len(categories)
                       else r.names.get(cls_id, str(cls_id)))
                detections.append({
                    "category":   cat,
                    "confidence": conf,
                    "bbox":       [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "mask":       None,
                })
        return detections


class SAMEnhancedBackend(SemanticModelBackend):
    """
    在任意检测后端之上叠加 SAM 分割掩码（装饰器模式）
    依赖: pip install segment-anything
    """

    def __init__(self, base_backend: SemanticModelBackend,
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth",
                 model_type: str = "vit_h",
                 device: str = "cuda"):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "未找到 segment_anything，请安装: pip install segment-anything"
            )
        self.base = base_backend
        logger.info(f"正在加载 SAM 模型: {sam_checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        logger.info("✓ SAM 加载完成")

    def detect(
        self,
        image: np.ndarray,
        categories: Optional[List[str]] = None,
        text_prompt: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ) -> List[Dict]:
        detections = self.base.detect(
            image, categories, text_prompt, confidence_threshold
        )
        if not detections:
            return detections

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            box = np.array([[x_min, y_min, x_max, y_max]])
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            det["mask"] = masks[0].astype(np.uint8) * 255  # H x W
        return detections


# ============================================================
# 语义地图管理器
# ============================================================

class SemanticMapManager(OccupancyMapManager):
    """
    语义地图管理器

    在占用网格地图的基础上，叠加语义物体信息：
      - 通过语义检测模型处理 RGB 图像，获得检测框 / 掩码
      - 利用深度图或点云将检测结果反投影到3D世界坐标
      - 将物体投影到 2D 语义地图像素坐标
      - 以字典形式索引，支持按类别高效查询

    用法示例::

        mgr = SemanticMapManager(scene_name="room_1")
        mgr.load_map()                        # 加载已有占用地图
        mgr.initialize_model("yolo_world")    # 加载语义模型

        # 从图像帧添加语义信息
        mgr.add_objects_from_image(
            image=bgr_image,
            depth_map=depth_m,
            camera_intrinsics=K,
            camera_pose_w2c=T_wc,
            categories=["chair", "table", "sofa"],
        )

        # 查询所有椅子
        chairs = mgr.get_objects_by_category("chair")
        for obj in chairs:
            print(obj)

        # 查询距离 (x, y) 最近的桌子
        nearest = mgr.get_nearest_object("table", query_pos=(1.0, 2.0))
    """

    def __init__(self, scene_name: str = "default",
                 output_dir: str = "data/scans/maps"):
        super().__init__(scene_name=scene_name, output_dir=output_dir)

        # 语义数据
        self._object_id_counter: int = 0
        # category -> List[SemanticObject]
        self._objects_by_category: Dict[str, List[SemanticObject]] = {}
        # object_id -> SemanticObject
        self._objects_by_id: Dict[int, SemanticObject] = {}

        # 语义可视化叠加层（与 color_projection 同尺寸）
        self.semantic_overlay: Optional[np.ndarray] = None

        # 语义模型后端
        self._model_backend: Optional[SemanticModelBackend] = None

        # 调色板（类别名 -> BGR 颜色）
        self._category_colors: Dict[str, Tuple[int, int, int]] = {}
        self._color_palette = [
            (0, 128, 255), (0, 200, 80),  (200, 50,  50),
            (255, 180, 0),  (150, 0, 200), (0,  200, 200),
            (255, 100, 200),(100, 255, 100),(200, 100, 0),
            (50,  50,  220),
        ]

    # ----------------------------------------------------------
    # 模型初始化
    # ----------------------------------------------------------

    def initialize_model(
        self,
        backend_type: str = "yolo_world",
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """
        初始化语义检测模型后端
        
        Args:
            backend_type: 后端类型，可选 "yolo_world" | "grounding_dino" | "none"
            device      : 推理设备 "cuda" / "cpu"
            **kwargs    : 传递给后端构造函数的额外参数
                - yolo_world: model_path (str)
                - grounding_dino: model_config_path (str), model_checkpoint_path (str)
                - sam_wrap: sam_checkpoint (str), model_type (str)，
                            需同时提供 base_backend 键
        """
        backend_type = backend_type.lower()

        if backend_type == "yolo_world":
            model_path = kwargs.get("model_path", "yolov8x-worldv2.pt")
            self._model_backend = YOLOWorldBackend(model_path=model_path, device=device)

        elif backend_type == "grounding_dino":
            self._model_backend = GroundingDINOBackend(
                model_config_path=kwargs["model_config_path"],
                model_checkpoint_path=kwargs["model_checkpoint_path"],
                device=device,
            )

        elif backend_type == "none":
            self._model_backend = None
            logger.info("语义模型后端已设置为 None（可手动添加物体）")

        else:
            raise ValueError(f"不支持的后端类型: {backend_type}")

        # 可选：用 SAM 进行分割增强
        if kwargs.get("use_sam", False) and self._model_backend is not None:
            self._model_backend = SAMEnhancedBackend(
                base_backend=self._model_backend,
                sam_checkpoint=kwargs.get("sam_checkpoint", "sam_vit_h_4b8939.pth"),
                model_type=kwargs.get("sam_model_type", "vit_h"),
                device=device,
            )

        logger.info(f"✓ 语义模型后端初始化完成: {backend_type}")

    # ----------------------------------------------------------
    # 添加语义物体 - 从图像帧
    # ----------------------------------------------------------

    def add_objects_from_image(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        camera_intrinsics: np.ndarray,
        camera_pose_c2w: np.ndarray,
        categories: Optional[List[str]] = None,
        text_prompt: Optional[str] = None,
        confidence_threshold: float = 0.35,
        source_image_id: Any = None,
        merge_iou_threshold: float = 0.5,
    ) -> List[SemanticObject]:
        """
        从一帧 RGB-D 图像中检测语义物体并添加到地图

        Args:
            image               : BGR 图像 (H, W, 3)
            depth_map           : 深度图 (H, W)，单位米；若为 None 则使用地图高度近似
            camera_intrinsics   : 相机内参矩阵 K (3×3)
            camera_pose_c2w     : 相机 → 世界变换矩阵 (4×4)
            categories          : 待检测类别列表
            text_prompt         : 文本提示（Grounding DINO 用）
            confidence_threshold: 置信度阈值
            source_image_id     : 帧号 / 图像 ID，用于溯源
            merge_iou_threshold : 若新检测与已有物体 IoU 超过此阈值则融合

        Returns:
            本次新增/更新的 SemanticObject 列表
        """
        if self._model_backend is None:
            raise RuntimeError("语义模型后端未初始化，请先调用 initialize_model()")
        if self.grid_resolution == 0.0 or self.min_pt is None:
            raise RuntimeError("地图尚未加载，请先调用 load_map() 或 create_occupancy_grid()")

        raw_detections = self._model_backend.detect(
            image, categories, text_prompt, confidence_threshold
        )

        added_objects = []
        for det in raw_detections:
            cat = det["category"]
            conf = det["confidence"]
            bbox = det["bbox"]         # [x_min, y_min, x_max, y_max]
            mask = det.get("mask")     # H×W or None

            # 计算图像像素中心
            img_cx = (bbox[0] + bbox[2]) // 2
            img_cy = (bbox[1] + bbox[3]) // 2

            # 反投影到世界坐标
            world_pos = self._backproject_to_world(
                img_cx, img_cy, depth_map, camera_intrinsics, camera_pose_c2w
            )
            if world_pos is None:
                continue

            # 世界坐标 → 地图像素坐标
            map_pixel = self._world_to_map_pixel(world_pos)
            if map_pixel is None:
                logger.info(f"  物体 '{cat}' 的世界坐标 {world_pos} 超出地图范围，跳过")
                continue

            # 计算物体在地图上的像素边界框
            map_bbox = self._image_bbox_to_map_bbox(
                bbox, depth_map, camera_intrinsics, camera_pose_c2w
            )

            # 可选：将掩码投影到地图坐标系
            map_mask = None
            if mask is not None and self.obstacle_map is not None:
                map_mask = self._project_mask_to_map(
                    mask, depth_map, camera_intrinsics, camera_pose_c2w
                )

            # 尝试与已有同类物体合并
            existing = self._find_mergeable_object(
                cat, map_pixel, map_bbox, merge_iou_threshold
            )

            if existing is not None:
                # 更新已有物体（加权平均坐标）
                alpha = conf / (conf + existing.confidence)
                existing.center_world = (
                    (1 - alpha) * existing.center_world + alpha * world_pos
                )
                existing.center_pixel = map_pixel
                existing.confidence = max(existing.confidence, conf)
                if map_mask is not None:
                    existing.mask_pixel = (
                        cv2.bitwise_or(existing.mask_pixel, map_mask)
                        if existing.mask_pixel is not None
                        else map_mask
                    )
                added_objects.append(existing)
            else:
                # 新建物体
                obj = self._create_object(
                    category=cat,
                    confidence=conf,
                    center_world=world_pos,
                    center_pixel=map_pixel,
                    bbox_pixel=map_bbox,
                    mask_pixel=map_mask,
                    source_image_id=source_image_id,
                )
                added_objects.append(obj)

        if added_objects:
            self._rebuild_semantic_overlay()
            logger.info(f"✓ 本次新增/更新 {len(added_objects)} 个语义物体")

        return added_objects

    # ----------------------------------------------------------
    # 添加语义物体 - 从带标签的点云
    # ----------------------------------------------------------

    def add_objects_from_labeled_pointcloud(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        label_names: Dict[int, str],
        confidences: Optional[np.ndarray] = None,
        min_points_per_object: int = 50,
    ) -> List[SemanticObject]:
        """
        从已经语义标注的点云中提取物体信息

        Args:
            points              : 点云坐标 (N, 3)
            labels              : 每点的类别 ID (N,)，-1 表示背景/忽略
            label_names         : {label_id: category_name}
            confidences         : 每点置信度 (N,)，可为 None
            min_points_per_object: 最少点数阈值，过小的聚类忽略

        Returns:
            新增的 SemanticObject 列表
        """
        if self.grid_resolution == 0.0 or self.min_pt is None:
            raise RuntimeError("地图尚未加载，请先调用 load_map() 或 create_occupancy_grid()")

        added_objects = []
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            if label_id < 0:
                continue  # 背景
            cat = label_names.get(int(label_id), f"class_{label_id}")
            mask = labels == label_id
            pts = points[mask]

            if len(pts) < min_points_per_object:
                continue

            conf = float(np.mean(confidences[mask])) if confidences is not None else 1.0

            # 聚类（一个类别可能有多个实例）
            instances = self._cluster_points(pts)
            for inst_pts in instances:
                center_world = inst_pts.mean(axis=0)
                map_pixel = self._world_to_map_pixel(center_world)
                if map_pixel is None:
                    continue
                map_bbox = self._compute_map_bbox_from_points(inst_pts)

                obj = self._create_object(
                    category=cat,
                    confidence=conf,
                    center_world=center_world,
                    center_pixel=map_pixel,
                    bbox_pixel=map_bbox,
                )
                added_objects.append(obj)

        if added_objects:
            self._rebuild_semantic_overlay()
            logger.info(f"✓ 从点云中提取 {len(added_objects)} 个语义物体")

        return added_objects

    # ----------------------------------------------------------
    # 手动添加物体
    # ----------------------------------------------------------

    def add_object_manual(
        self,
        category: str,
        world_position: Union[np.ndarray, Tuple, List],
        confidence: float = 1.0,
        additional_info: Optional[Dict] = None,
    ) -> SemanticObject:
        """
        手动添加一个语义物体（已知世界坐标）
        
        Args:
            category      : 类别名称
            world_position: 世界坐标 [x, y, z]
            confidence    : 置信度
            additional_info: 附加信息
        
        Returns:
            创建的 SemanticObject
        """
        world_pos = np.asarray(world_position, dtype=float)
        map_pixel = self._world_to_map_pixel(world_pos)
        if map_pixel is None:
            raise ValueError(f"世界坐标 {world_pos} 超出当前地图范围")

        obj = self._create_object(
            category=category,
            confidence=confidence,
            center_world=world_pos,
            center_pixel=map_pixel,
            bbox_pixel=(map_pixel[0], map_pixel[1], map_pixel[0], map_pixel[1]),
            additional_info=additional_info or {},
        )
        self._rebuild_semantic_overlay()
        logger.info(f"✓ 手动添加物体: {obj}")
        return obj

    # ----------------------------------------------------------
    # 查询接口
    # ----------------------------------------------------------

    def get_objects_by_category(
        self,
        category: str,
        min_confidence: float = 0.0,
        fuzzy: bool = False,
    ) -> List[SemanticObject]:
        """
        按类别名称查询所有物体
        
        Args:
            category      : 精确类别名称
            min_confidence: 置信度下限过滤
            fuzzy         : 是否启用模糊匹配（包含 category 子串的类别）

        Returns:
            符合条件的 SemanticObject 列表
        """
        if fuzzy:
            results = []
            for cat, objs in self._objects_by_category.items():
                if category.lower() in cat.lower():
                    results.extend(o for o in objs if o.confidence >= min_confidence)
            return results
        else:
            objs = self._objects_by_category.get(category, [])
            return [o for o in objs if o.confidence >= min_confidence]

    def get_object_by_id(self, object_id: int) -> Optional[SemanticObject]:
        """按 ID 查询单个物体"""
        return self._objects_by_id.get(object_id)

    def get_all_objects(self, min_confidence: float = 0.0) -> List[SemanticObject]:
        """获取所有语义物体"""
        return [o for o in self._objects_by_id.values()
                if o.confidence >= min_confidence]

    def get_all_categories(self) -> List[str]:
        """获取所有已知类别名称"""
        return list(self._objects_by_category.keys())

    def get_object_count(self, category: Optional[str] = None) -> int:
        """获取物体数量（可指定类别）"""
        if category is None:
            return len(self._objects_by_id)
        return len(self._objects_by_category.get(category, []))

    def get_nearest_object(
        self,
        category: str,
        query_pos: Union[np.ndarray, Tuple, List],
        use_3d: bool = False,
        min_confidence: float = 0.0,
        fuzzy: bool = False,
    ) -> Optional[SemanticObject]:
        """
        查询距给定位置最近的指定类别物体

        Args:
            category   : 类别名称
            query_pos  : 查询坐标，2D (x, y) 或 3D (x, y, z)
            use_3d     : True 则用3D欧氏距离；False 则仅用 XY 平面距离
            min_confidence: 置信度下限
            fuzzy      : 是否模糊匹配类别

        Returns:
            最近的 SemanticObject，若无则返回 None
        """
        candidates = self.get_objects_by_category(category, min_confidence, fuzzy)
        if not candidates:
            return None

        query = np.asarray(query_pos, dtype=float)
        best, best_dist = None, float("inf")
        for obj in candidates:
            pos = obj.center_world
            if use_3d:
                q = query[:3] if len(query) >= 3 else np.append(query, 0)
                dist = float(np.linalg.norm(pos - q))
            else:
                dist = float(np.linalg.norm(pos[:2] - query[:2]))
            if dist < best_dist:
                best_dist = dist
                best = obj
        return best

    def get_objects_within_radius(
        self,
        query_pos: Union[np.ndarray, Tuple, List],
        radius: float,
        category: Optional[str] = None,
        use_3d: bool = False,
    ) -> List[SemanticObject]:
        """
        查询给定范围内的所有物体

        Args:
            query_pos: 查询坐标 (x, y) 或 (x, y, z)
            radius   : 搜索半径（米）
            category : 可选类别过滤
            use_3d   : 是否使用3D距离

        Returns:
            距离在 radius 以内的 SemanticObject 列表，按距离排序
        """
        if category is not None:
            candidates = self.get_objects_by_category(category, fuzzy=True)
        else:
            candidates = self.get_all_objects()

        query = np.asarray(query_pos, dtype=float)
        results = []
        for obj in candidates:
            pos = obj.center_world
            if use_3d:
                q = query[:3] if len(query) >= 3 else np.append(query, 0)
                dist = float(np.linalg.norm(pos - q))
            else:
                dist = float(np.linalg.norm(pos[:2] - query[:2]))
            if dist <= radius:
                results.append((dist, obj))

        results.sort(key=lambda t: t[0])
        return [obj for _, obj in results]

    def get_object_world_coords(
        self,
        category: str,
        min_confidence: float = 0.0,
        fuzzy: bool = False,
    ) -> np.ndarray:
        """
        获取某类物体的所有世界坐标（N×3 数组）

        Returns:
            (N, 3) ndarray，每行为一个物体的 [x, y, z]
        """
        objs = self.get_objects_by_category(category, min_confidence, fuzzy)
        if not objs:
            return np.empty((0, 3), dtype=float)
        return np.stack([o.center_world for o in objs], axis=0)

    def export_objects_to_json(self, save_path: Optional[str] = None) -> str:
        """
        将所有语义物体导出为 JSON 字符串（或保存到文件）

        Args:
            save_path: 若指定则写入文件

        Returns:
            JSON 字符串
        """
        data = {cat: [o.to_dict() for o in objs]
                for cat, objs in self._objects_by_category.items()}
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if save_path:
            Path(save_path).write_text(json_str, encoding="utf-8")
            logger.info(f"✓ 语义物体已导出到: {save_path}")
        return json_str

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------

    def visualize_semantic_map(
        self,
        show_labels: bool = True,
        show_bboxes: bool = True,
        alpha_overlay: float = 0.6,
    ) -> np.ndarray:
        """
        生成语义地图可视化图像（在彩色投影地图上叠加物体标注）

        Args:
            show_labels   : 是否显示类别文字标签
            show_bboxes   : 是否显示边界框
            alpha_overlay : 半透明叠加透明度

        Returns:
            BGR 可视化图像 (H, W, 3)
        """
        if self.color_projection is None:
            raise RuntimeError("地图未加载")

        vis = self.color_projection.copy()

        # 未知区域置灰
        if self.point_cloud_coverage is not None:
            vis[self.point_cloud_coverage == 0] = [50, 50, 50]

        # 半透明障碍物（红色）
        if self.obstacle_map is not None:
            obs_mask = self.obstacle_map == 255
            vis[obs_mask] = cv2.addWeighted(
                vis[obs_mask], 0.4,
                np.full_like(vis[obs_mask], [0, 0, 180]), 0.6, 0,
            )

        # 叠加每个物体的掩码和标注
        overlay = vis.copy()
        for obj in self._objects_by_id.values():
            color = self._get_category_color(obj.category)

            # 绘制分割掩码
            if obj.mask_pixel is not None:
                mask_bool = obj.mask_pixel > 0
                overlay[mask_bool] = color

            # 绘制边界框
            if show_bboxes:
                x1, y1, x2, y2 = obj.bbox_pixel
                h_vis, w_vis = vis.shape[:2]
                x1 = max(0, min(x1, w_vis - 1))
                x2 = max(0, min(x2, w_vis - 1))
                y1 = max(0, min(y1, h_vis - 1))
                y2 = max(0, min(y2, h_vis - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # 绘制中心点
            cx, cy = obj.center_pixel
            cv2.circle(overlay, (cx, cy), 5, color, -1)
            cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), 1)

            # 绘制标签
            if show_labels:
                label = f"{obj.category}({obj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                tx = max(0, cx - tw // 2)
                ty = max(th + 2, cy - 8)
                cv2.rectangle(overlay, (tx - 1, ty - th - 2),
                              (tx + tw + 1, ty + 2), (0, 0, 0), -1)
                cv2.putText(overlay, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)

        vis = cv2.addWeighted(vis, 1.0 - alpha_overlay, overlay, alpha_overlay, 0)
        return vis

    def save_semantic_visualization(self, filename: Optional[str] = None) -> str:
        """保存语义地图可视化图像"""
        vis = self.visualize_semantic_map()
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        fname = filename or f"{self.scene_name}_semantic_map.png"
        out_path = str(vis_dir / fname)
        cv2.imwrite(out_path, vis)
        logger.info(f"✓ 语义地图可视化已保存: {out_path}")
        return out_path

    # ----------------------------------------------------------
    # 保存 / 加载（扩展父类）
    # ----------------------------------------------------------

    def save_map(self, scene_name: str = None) -> bool:
        """保存占用地图 + 语义物体数据"""
        success = super().save_map(scene_name)
        if not success:
            return False

        # 额外保存语义数据
        semantic_data = {
            "objects_by_id":       self._objects_by_id,
            "objects_by_category": self._objects_by_category,
            "object_id_counter":   self._object_id_counter,
            "category_colors":     self._category_colors,
        }
        sem_path = self.output_dir / f"{self.scene_name}_semantic.pkl"
        with open(sem_path, "wb") as f:
            pickle.dump(semantic_data, f)
        logger.info(f"✓ 语义数据已保存到: {sem_path}")

        # 保存语义可视化
        if self.obstacle_map is not None:
            self.save_semantic_visualization()

        # 导出 JSON
        json_path = str(self.output_dir / f"{self.scene_name}_objects.json")
        self.export_objects_to_json(json_path)
        return True

    def load_map(self) -> bool:
        """加载占用地图 + 语义物体数据"""
        if not super().load_map():
            return False

        sem_path = self.output_dir / f"{self.scene_name}_semantic.pkl"
        if sem_path.exists():
            try:
                with open(sem_path, "rb") as f:
                    semantic_data = pickle.load(f)
                self._objects_by_id       = semantic_data["objects_by_id"]
                self._objects_by_category = semantic_data["objects_by_category"]
                self._object_id_counter   = semantic_data["object_id_counter"]
                self._category_colors     = semantic_data.get("category_colors", {})
                logger.info(f"✓ 语义数据加载完成，共 {len(self._objects_by_id)} 个物体")
                self._rebuild_semantic_overlay()
            except Exception as e:
                logger.info(f"警告: 语义数据加载失败（{e}），将以空语义地图继续")
        else:
            logger.info("未找到语义数据文件，将以空语义地图继续")
        return True

    # ----------------------------------------------------------
    # 坐标转换工具
    # ----------------------------------------------------------

    def world_to_map_pixel(
        self, world_pos: Union[np.ndarray, Tuple, List]
    ) -> Optional[Tuple[int, int]]:
        """世界坐标 → 地图像素坐标（公开接口）"""
        return self._world_to_map_pixel(np.asarray(world_pos, dtype=float))

    def map_pixel_to_world(
        self, pixel: Union[Tuple[int, int], List[int]]
    ) -> np.ndarray:
        """地图像素坐标 → 世界坐标（XY 平面，Z 取地面高度）"""
        col, row = pixel
        x = self.min_pt[0] + col * self.grid_resolution
        y = self.max_pt[1] - row * self.grid_resolution  # Y 轴翻转
        z = self.ground_z
        return np.array([x, y, z], dtype=float)

    # ----------------------------------------------------------
    # 私有辅助方法
    # ----------------------------------------------------------

    def _world_to_map_pixel(
        self, world_pos: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """世界坐标 → 地图像素坐标（内部实现）"""
        if self.min_pt is None or self.max_pt is None:
            return None
        col = int((world_pos[0] - self.min_pt[0]) / self.grid_resolution)
        row = int((self.max_pt[1] - world_pos[1]) / self.grid_resolution)
        if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
            return (col, row)
        return None

    def _backproject_to_world(
        self,
        img_col: int,
        img_row: int,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        T_c2w: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        将图像像素反投影到世界坐标
        
        Args:
            img_col, img_row: 图像像素坐标
            depth_map       : 深度图（米），可为 None（则使用地面高度回退）
            K               : 3×3 相机内参矩阵
            T_c2w           : 4×4 相机坐标系 → 世界坐标系变换矩阵

        Returns:
            世界坐标 [x, y, z] 或 None
        """
        if depth_map is not None:
            h, w = depth_map.shape[:2]
            r = max(0, min(img_row, h - 1))
            c = max(0, min(img_col, w - 1))
            depth = float(depth_map[r, c])
            if depth <= 0 or not np.isfinite(depth):
                # 回退：在 bbox 邻域取中位数
                r0, r1 = max(0, r - 5), min(h, r + 6)
                c0, c1 = max(0, c - 5), min(w, c + 6)
                patch = depth_map[r0:r1, c0:c1]
                valid = patch[patch > 0]
                if len(valid) == 0:
                    depth = None
                else:
                    depth = float(np.median(valid))
        else:
            depth = None

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        if depth is not None and np.isfinite(depth):
            # 标准反投影
            x_c = (img_col - cx) / fx * depth
            y_c = (img_row - cy) / fy * depth
            z_c = depth
        else:
            # 无深度：射线与地面平面 z=ground_z 的交点
            ray_cam = np.array([(img_col - cx) / fx,
                                 (img_row - cy) / fy,
                                 1.0], dtype=float)
            R = T_c2w[:3, :3]
            t = T_c2w[:3, 3]
            ray_world = R @ ray_cam
            camera_origin = t
            target_z = self.ground_z
            if abs(ray_world[2]) < 1e-6:
                return None
            lam = (target_z - camera_origin[2]) / ray_world[2]
            if lam <= 0:
                return None
            world_pos = camera_origin + lam * ray_world
            return world_pos.astype(float)

        point_cam = np.array([x_c, y_c, z_c, 1.0], dtype=float)
        point_world = T_c2w @ point_cam
        return point_world[:3].astype(float)

    def _image_bbox_to_map_bbox(
        self,
        bbox: List[int],
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        T_c2w: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """将图像边界框四角反投影后取地图 BBox"""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        pixel_coords = []
        for cx, cy in corners:
            wp = self._backproject_to_world(cx, cy, depth_map, K, T_c2w)
            if wp is not None:
                mp = self._world_to_map_pixel(wp)
                if mp is not None:
                    pixel_coords.append(mp)

        if not pixel_coords:
            # 无法反投影，返回中心点退化框
            cx_map = (x1 + x2) // 2
            cy_map = (y1 + y2) // 2
            wp = self._backproject_to_world(cx_map, cy_map, depth_map, K, T_c2w)
            mp = self._world_to_map_pixel(wp) if wp is not None else (0, 0)
            return (mp[0], mp[1], mp[0], mp[1])

        cols = [p[0] for p in pixel_coords]
        rows = [p[1] for p in pixel_coords]
        return (min(cols), min(rows), max(cols), max(rows))

    def _project_mask_to_map(
        self,
        mask: np.ndarray,
        depth_map: Optional[np.ndarray],
        K: np.ndarray,
        T_c2w: np.ndarray,
    ) -> Optional[np.ndarray]:
        """将图像分割掩码投影到地图坐标系（稀疏采样）"""
        if self.obstacle_map is None:
            return None

        map_mask = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.uint8
        )
        ys, xs = np.where(mask > 0)
        # 稀疏采样（步长 4 提速）
        step = max(1, len(ys) // 2000)
        for y, x in zip(ys[::step], xs[::step]):
            wp = self._backproject_to_world(int(x), int(y), depth_map, K, T_c2w)
            if wp is None:
                continue
            mp = self._world_to_map_pixel(wp)
            if mp is not None:
                map_mask[mp[1], mp[0]] = 255

        # 膨胀填充稀疏点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        map_mask = cv2.dilate(map_mask, kernel, iterations=1)
        return map_mask

    def _cluster_points(
        self, points: np.ndarray, eps: float = 0.5, min_samples: int = 10
    ) -> List[np.ndarray]:
        """对一类物体的点云进行 DBSCAN 聚类，返回各实例点集列表"""
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                points[:, :2]  # 只用 XY
            )
            instances = []
            for lbl in set(labels):
                if lbl < 0:
                    continue
                instances.append(points[labels == lbl])
            return instances if instances else [points]
        except ImportError:
            return [points]  # 无 sklearn 则整体当一个实例

    def _compute_map_bbox_from_points(
        self, points: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """从3D点集计算地图 BBox"""
        map_pts = []
        for pt in points[::max(1, len(points) // 500)]:
            mp = self._world_to_map_pixel(pt)
            if mp is not None:
                map_pts.append(mp)
        if not map_pts:
            return (0, 0, 0, 0)
        cols = [p[0] for p in map_pts]
        rows = [p[1] for p in map_pts]
        return (min(cols), min(rows), max(cols), max(rows))

    def _find_mergeable_object(
        self,
        category: str,
        map_pixel: Tuple[int, int],
        map_bbox: Tuple[int, int, int, int],
        iou_threshold: float,
    ) -> Optional[SemanticObject]:
        """查找可以合并的同类已有物体"""
        candidates = self._objects_by_category.get(category, [])
        for obj in candidates:
            iou = self._bbox_iou(obj.bbox_pixel, map_bbox)
            if iou >= iou_threshold:
                return obj
        return None

    @staticmethod
    def _bbox_iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        """计算两个像素 BBox 的 IoU"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _create_object(
        self,
        category: str,
        confidence: float,
        center_world: np.ndarray,
        center_pixel: Tuple[int, int],
        bbox_pixel: Tuple[int, int, int, int],
        mask_pixel: Optional[np.ndarray] = None,
        source_image_id: Any = None,
        additional_info: Optional[Dict] = None,
    ) -> SemanticObject:
        """创建并注册一个新 SemanticObject"""
        obj_id = self._object_id_counter
        self._object_id_counter += 1

        obj = SemanticObject(
            object_id=obj_id,
            category=category,
            confidence=confidence,
            center_world=center_world.copy() if isinstance(center_world, np.ndarray)
                         else np.asarray(center_world, dtype=float),
            center_pixel=center_pixel,
            bbox_pixel=bbox_pixel,
            mask_pixel=mask_pixel,
            source_image_id=source_image_id,
            additional_info=additional_info or {},
        )

        self._objects_by_id[obj_id] = obj
        self._objects_by_category.setdefault(category, []).append(obj)
        return obj

    def _get_category_color(
        self, category: str
    ) -> Tuple[int, int, int]:
        """为每个类别分配固定 BGR 颜色"""
        if category not in self._category_colors:
            idx = len(self._category_colors) % len(self._color_palette)
            self._category_colors[category] = self._color_palette[idx]
        return self._category_colors[category]

    def _rebuild_semantic_overlay(self) -> None:
        """重建语义叠加层缓存"""
        if self.obstacle_map is None:
            return
        self.semantic_overlay = np.zeros(
            (self.grid_height, self.grid_width, 3), dtype=np.uint8
        )
        for obj in self._objects_by_id.values():
            color = self._get_category_color(obj.category)
            if obj.mask_pixel is not None:
                self.semantic_overlay[obj.mask_pixel > 0] = color
            cx, cy = obj.center_pixel
            if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
                cv2.circle(self.semantic_overlay, (cx, cy), 5, color, -1)

    def print_summary(self) -> None:
        """打印语义地图摘要"""
        logger.info("\n=== 语义地图摘要 ===")
        logger.info(f"场景: {self.scene_name}")
        logger.info(f"物体总数: {len(self._objects_by_id)}")
        logger.info(f"类别数: {len(self._objects_by_category)}")
        for cat, objs in sorted(self._objects_by_category.items(),
                                key=lambda kv: len(kv[1]), reverse=True):
            avg_conf = np.mean([o.confidence for o in objs])
            logger.info(f"  {cat:20s}: {len(objs):3d} 个  "
                        f"(平均置信度 {avg_conf:.2f})")


# ============================================================
# 自动探索系统（附加到 SemanticMapManager）
# ============================================================

def _build_c2w_rotation(yaw_rad: float, pitch_rad: float = 0.0) -> np.ndarray:
    """
    构建相机到世界的旋转矩阵（3×3）

    坐标约定（与 RenderManager 一致）：
      - 世界坐标系：Z 轴朝上，XY 为水平面
      - 相机坐标系：X 轴朝右，Y 轴朝下，Z 轴朝前（进入场景）
      - R_c2w 的每列分别为相机 X/Y/Z 轴在世界坐标系中的方向

    Args:
        yaw_rad  : 水平偏航角（弧度），0 = 朝 +X，逆时针为正
        pitch_rad: 俯仰角（弧度），正值 = 向上仰，负值 = 向下俯

    Returns:
        3×3 numpy float32 数组
    """
    # 水平基础旋转（纯偏航，无俯仰）
    # 列 = 相机轴在世界中的方向：
    #   col0 = X_cam (right)   = [sinθ, -cosθ, 0]
    #   col1 = Y_cam (down)    = [0,     0,    -1]
    #   col2 = Z_cam (forward) = [cosθ,  sinθ,  0]
    cy, sy = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    R_yaw = np.column_stack([
        np.array([ sy, -cy, 0.0], dtype=np.float32),   # X_cam (right)
        np.array([0.0, 0.0, -1.0], dtype=np.float32),  # Y_cam (down)
        np.array([ cy,  sy, 0.0], dtype=np.float32),   # Z_cam (forward)
    ])  # shape (3, 3)

    if pitch_rad == 0.0:
        return R_yaw.astype(np.float32)

    # --- 俯仰修正：在相机坐标系内绕 X 轴旋转 ---
    # rot_camera_x(pitch_rad): Y' = cos*Y - sin*Z, Z' = sin*Y + cos*Z
    # (正 pitch = 向上仰视；与 render_manager 中 R_up=R@rot(π/2) 一致)
    cp, sp = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    rot_cx = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  cp,  -sp ],
        [0.0,  sp,   cp ],
    ], dtype=np.float32)

    # R_c2w_pitched = R_yaw @ rot_cx  (先 yaw，再在相机系内 pitch)
    return (R_yaw @ rot_cx).astype(np.float32)


# ---- 将以下方法注入 SemanticMapManager ----

def _sem_sample_viewpoints(
    self: "SemanticMapManager",
    viewpoint_spacing_m: float = 1.5,
    camera_height_offset_m: float = 1.5,
    safe_margin_pixels: int = 20,
) -> List[np.ndarray]:
    """
    在可通行区域内以规则网格采样探索视点
        
        Args:
        viewpoint_spacing_m    : 视点间距（米）
        camera_height_offset_m : 相机高于地面的高度（米）
        safe_margin_pixels     : 额外侵蚀像素，使视点远离障碍物边缘

    Returns:
        List of [x, y, z] world-coordinate arrays
    """
    if self.expanded_traversability is None:
        raise RuntimeError("请先加载占用地图")

    # 在 expanded_traversability 基础上再腐蚀，留出安全边距
    mask = (self.expanded_traversability > 0).astype(np.uint8)
    if safe_margin_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * safe_margin_pixels + 1, 2 * safe_margin_pixels + 1)
        )
        mask = cv2.erode(mask, kernel, iterations=1)

    # 网格间距（像素）
    spacing_px = max(1, int(viewpoint_spacing_m / self.grid_resolution))

    camera_z = self.ground_z + camera_height_offset_m
    viewpoints = []

    # 以 spacing_px 步长遍历地图像素
    rows_range = range(spacing_px // 2, self.grid_height, spacing_px)
    cols_range = range(spacing_px // 2, self.grid_width, spacing_px)

    for row in rows_range:
        for col in cols_range:
            if mask[row, col] > 0:
                # 像素 → 世界坐标（XY）
                wx = self.min_pt[0] + col * self.grid_resolution
                wy = self.max_pt[1] - row * self.grid_resolution  # Y 轴翻转
                viewpoints.append(np.array([wx, wy, camera_z], dtype=float))

    logger.info(f"[探索] 采样视点数: {len(viewpoints)}  (间距={viewpoint_spacing_m:.1f}m, "
                f"网格步长={spacing_px}px)")
    return viewpoints


def _sem_order_viewpoints_greedy(
    self: "SemanticMapManager",
    viewpoints: List[np.ndarray],
    start_pos: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    贪心最近邻排序视点，减少总移动距离（近似 TSP）

    Args:
        viewpoints: 待排序视点列表
        start_pos : 起始世界坐标（默认使用地图中心处最近视点）

    Returns:
        排序后的视点列表
    """
    if not viewpoints:
        return []

    pts = np.array([v[:2] for v in viewpoints], dtype=float)   # N×2 XY

    if start_pos is None:
        # 默认从地图可通行区域的质心开始
        if self.expanded_traversability is not None:
            ys, xs = np.where(self.expanded_traversability > 0)
            cen_col = int(xs.mean())
            cen_row = int(ys.mean())
            cx = self.min_pt[0] + cen_col * self.grid_resolution
            cy = self.max_pt[1] - cen_row * self.grid_resolution
            start_xy = np.array([cx, cy])
        else:
            start_xy = pts[0]
    else:
        start_xy = np.asarray(start_pos[:2], dtype=float)

    remaining = list(range(len(pts)))
    ordered_idx = []
    cur = start_xy

    while remaining:
        dists = np.linalg.norm(pts[remaining] - cur, axis=1)
        nearest_i = int(np.argmin(dists))
        idx = remaining.pop(nearest_i)
        ordered_idx.append(idx)
        cur = pts[idx]

    return [viewpoints[i] for i in ordered_idx]


def _sem_build_camera_params(
    self: "SemanticMapManager",
    world_pos: np.ndarray,
    yaw_rad: float,
    pitch_rad: float = -0.15,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 90.0,
) -> Dict:
    """
    构建与 RenderManager.render_single_view() 兼容的 camera_params 字典

    RenderManager 约定：
      - R (3×3): 相机到世界旋转矩阵（列 = 相机轴在世界中的方向）
      - T (3,) : 相机在世界坐标系中的位置
      内部做 T_w2c = -R^T @ T，再传给 getWorld2View2(R, T_w2c)

    Args:
        world_pos : 相机世界坐标 [x, y, z]
        yaw_rad   : 水平偏航角（0=朝+X，逆时针为正）
        pitch_rad : 俯仰角（负值=向下俯）
        width,height: 图像分辨率
        fov_deg   : 水平视场角

    Returns:
        camera_params 字典
    """
    fx = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy = fx  # 假设方形像素

    R_c2w = _build_c2w_rotation(yaw_rad, pitch_rad)

    return {
        "width":  width,
        "height": height,
        "fx":     fx,
        "fy":     fy,
        "R":      R_c2w,
        "T":      np.asarray(world_pos, dtype=np.float32),
    }


def _sem_compute_c2w_matrix(
    self: "SemanticMapManager",
    world_pos: np.ndarray,
    yaw_rad: float,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """
    构建 4×4 相机到世界变换矩阵（用于 _backproject_to_world）

    Returns:
        (4, 4) float64 ndarray
    """
    R = _build_c2w_rotation(yaw_rad, pitch_rad).astype(float)
    T = np.asarray(world_pos, dtype=float)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = R
    mat[:3,  3] = T
    return mat


def _sem_build_intrinsics(
    width: int,
    height: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """返回 3×3 相机内参矩阵 K"""
    fx = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy = fx
    K = np.array([
        [fx,  0.0, width  / 2.0],
        [0.0, fy,  height / 2.0],
        [0.0, 0.0, 1.0         ],
    ], dtype=float)
    return K


def _sem_compute_coverage_contribution(
    self: "SemanticMapManager",
    pixel: Tuple[int, int],
    max_range_px: int,
    traversability: np.ndarray,
) -> np.ndarray:
    """
    计算一个视点能看到（简化为圆形范围内可达）的地图像素区域掩码

    Args:
        pixel       : 视点在地图上的像素坐标 (col, row)
        max_range_px: 最大可视范围（像素）
        traversability: 可通行掩码（用于视线遮挡近似）

    Returns:
        二值可见掩码 (H, W) uint8
    """
    H, W = traversability.shape
    col, row = pixel
    # 简单圆形掩码（快速近似）
    Y, X = np.ogrid[:H, :W]
    dist_sq = (X - col) ** 2 + (Y - row) ** 2
    circle = (dist_sq <= max_range_px ** 2).astype(np.uint8)
    visible = cv2.bitwise_and(circle, (traversability > 0).astype(np.uint8))
    return visible


def _sem_visualize_exploration_progress(
    self: "SemanticMapManager",
    coverage_map: np.ndarray,
    current_pos_pixel: Optional[Tuple[int, int]],
    visited_pixels: List[Tuple[int, int]],
    save_path: str,
) -> None:
    """保存探索进度可视化图"""
    base = self.color_projection.copy() if self.color_projection is not None \
           else np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

    # 未知区域置灰
    if self.point_cloud_coverage is not None:
        base[self.point_cloud_coverage == 0] = [50, 50, 50]

    # 障碍物红色半透明
    if self.obstacle_map is not None:
        om = self.obstacle_map == 255
        base[om] = cv2.addWeighted(base[om], 0.4,
                                   np.full_like(base[om], [0, 0, 180]), 0.6, 0)

    # 覆盖区域浅绿色半透明
    cov_mask = coverage_map > 0
    base[cov_mask] = cv2.addWeighted(base[cov_mask], 0.5,
                                     np.full_like(base[cov_mask], [0, 180, 0]), 0.5, 0)

    # 已访问视点（蓝色小圆）
    for (vc, vr) in visited_pixels:
        cv2.circle(base, (vc, vr), 4, (200, 100, 0), -1)

    # 当前视点（黄色大圆）
    if current_pos_pixel is not None:
        cv2.circle(base, current_pos_pixel, 8, (0, 220, 220), -1)
        cv2.circle(base, current_pos_pixel, 9, (255, 255, 255), 1)

    # 语义物体标注
    for obj in self._objects_by_id.values():
        color = self._get_category_color(obj.category)
        cv2.circle(base, obj.center_pixel, 5, color, -1)

    cv2.imwrite(save_path, base)


def auto_explore_and_annotate(
    self: "SemanticMapManager",
    render_manager,
    categories: Optional[List[str]] = None,
    text_prompt: Optional[str] = None,
    viewpoint_spacing_m: float = 1.5,
    camera_height_offset_m: float = 1.5,
    yaw_directions: int = 4,
    pitch_rad: float = -0.15,
    fov_deg: float = 90.0,
    image_width: int = 640,
    image_height: int = 480,
    confidence_threshold: float = 0.30,
    max_view_range_m: float = 8.0,
    coverage_target: float = 0.90,
    save_interval: int = 10,
    progress_vis_dir: Optional[str] = None,
    safe_margin_pixels: int = 20,
) -> Dict:
    """
    自动探索占用地图，渲染多视角图像并生成语义地图

    算法流程：
      1. 在 expanded_traversability 上以规则网格采样视点
      2. 贪心最近邻排序视点，减少总行进距离
      3. 逐视点循环：
         a. 在该点渲染 `yaw_directions` 个水平方向的 RGB-D 图像
         b. 对每张图像运行语义检测模型
         c. 将检测结果反投影到3D世界坐标并写入语义地图
      4. 实时跟踪覆盖率，达到 `coverage_target` 时提前结束
      5. 每 `save_interval` 个视点自动保存中间结果
        
        Args:
        render_manager         : 已初始化的 RenderManager 实例
        categories             : 待检测类别列表（与 text_prompt 二选一）
        text_prompt            : 文本提示（Grounding DINO 风格）
        viewpoint_spacing_m    : 视点网格间距（米）
        camera_height_offset_m : 相机高于地面高度（米）
        yaw_directions         : 每个视点拍摄的水平方向数（4/6/8）
        pitch_rad              : 俯仰角（负值=向下俯视，推荐 -0.15 rad ≈ -8.6°）
        fov_deg                : 水平视场角（度）
        image_width, image_height : 渲染图像分辨率
        confidence_threshold   : 语义检测置信度阈值
        max_view_range_m       : 单视点最大可视范围（米，用于覆盖率统计）
        coverage_target        : 目标覆盖率 [0,1]（可通行区域被观测的比例）
        save_interval          : 每隔多少视点保存一次中间状态
        progress_vis_dir       : 若指定，保存探索进度可视化图到该目录
        safe_margin_pixels     : 视点采样时距障碍物的安全边距（像素）
            
        Returns:
        统计字典：{
            "total_viewpoints"  : 实际访问的视点数,
            "total_images"      : 渲染图像总数,
            "total_objects"     : 检测到的物体总数,
            "coverage_rate"     : 最终覆盖率,
            "categories_found"  : 发现的物体类别列表,
        }
    """
    if self._model_backend is None:
        raise RuntimeError("语义模型未初始化，请先调用 initialize_model()")
    if self.expanded_traversability is None:
        raise RuntimeError("占用地图未加载，请先调用 load_map() 或 create_occupancy_grid()")

    # ── 准备工作 ────────────────────────────────────────────────
    K = _sem_build_intrinsics(image_width, image_height, fov_deg)

    if progress_vis_dir:
        Path(progress_vis_dir).mkdir(parents=True, exist_ok=True)

    # 可通行区域总像素（用于计算覆盖率）
    traversable_mask = (self.expanded_traversability > 0).astype(np.uint8)
    total_traversable_px = int(np.count_nonzero(traversable_mask))
    if total_traversable_px == 0:
        raise RuntimeError("可通行区域为空，无法探索")

    # 覆盖地图（0=未覆盖，255=已覆盖）
    coverage_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
    max_range_px = max(1, int(max_view_range_m / self.grid_resolution))

    # 偏航方向序列（均匀分布 0 ~ 2π）
    yaw_angles = [2 * np.pi * k / yaw_directions for k in range(yaw_directions)]

    # ── 1. 采样并排序视点 ───────────────────────────────────────
    logger.info("\n=== [自动探索] 开始 ===")
    logger.info(f"检测类别: {categories or text_prompt}")

    viewpoints = _sem_sample_viewpoints(
        self,
        viewpoint_spacing_m=viewpoint_spacing_m,
        camera_height_offset_m=camera_height_offset_m,
        safe_margin_pixels=safe_margin_pixels,
    )
    if not viewpoints:
        logger.info("[探索] 警告：未能采样到任何视点，请检查地图或调整参数")
        return {}

    viewpoints = _sem_order_viewpoints_greedy(self, viewpoints)

    # ── 2. 逐视点探索 ──────────────────────────────────────────
    total_images = 0
    visited_pixels: List[Tuple[int, int]] = []
    stats_objects_added = 0

    try:
        from tqdm import tqdm as _tqdm
        vp_iter = _tqdm(enumerate(viewpoints), total=len(viewpoints),
                        desc="探索视点", unit="vp")
    except ImportError:
        vp_iter = enumerate(viewpoints)

    for vp_idx, world_pos in vp_iter:
        # 当前视点对应的地图像素
        mp = self._world_to_map_pixel(world_pos)
        if mp is None:
            continue
        visited_pixels.append(mp)

        # 更新覆盖地图
        contrib = _sem_compute_coverage_contribution(
            self, mp, max_range_px, traversable_mask
        )
        coverage_map = cv2.bitwise_or(coverage_map, contrib)

        # 逐方向渲染 + 检测
        for yaw in yaw_angles:
            cam_params = _sem_build_camera_params(
                self, world_pos, yaw,
                pitch_rad=pitch_rad,
                width=image_width,
                height=image_height,
                fov_deg=fov_deg,
            )

            # 渲染
            try:
                rgb_np, depth_np = render_manager.render_single_view(cam_params)
            except Exception as e:
                logger.info(f"  [渲染失败] vp={vp_idx}, yaw={np.degrees(yaw):.0f}°: {e}")
                continue

            total_images += 1

            # 构建 c2w 矩阵（用于反投影）
            T_c2w = _sem_compute_c2w_matrix(self, world_pos, yaw, pitch_rad)

            # 语义检测 + 写入地图
            try:
                new_objs = self.add_objects_from_image(
                    image=cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
                    depth_map=depth_np,
                    camera_intrinsics=K,
                    camera_pose_c2w=T_c2w,
                    categories=categories,
                    text_prompt=text_prompt,
                    confidence_threshold=confidence_threshold,
                    source_image_id=f"vp{vp_idx}_yaw{np.degrees(yaw):.0f}",
                )
                stats_objects_added += len(new_objs)
            except Exception as e:
                logger.info(f"  [检测失败] vp={vp_idx}, yaw={np.degrees(yaw):.0f}°: {e}")
                continue

        # ── 计算当前覆盖率 ──
        covered_px = int(np.count_nonzero(coverage_map))
        coverage_rate = covered_px / total_traversable_px

        # 可视化进度
        if progress_vis_dir and (vp_idx % max(1, save_interval // 2) == 0):
            vis_path = str(Path(progress_vis_dir)
                           / f"explore_{vp_idx:04d}.png")
            _sem_visualize_exploration_progress(
                self, coverage_map, mp, visited_pixels, vis_path
            )

        # 定期保存
        if (vp_idx + 1) % save_interval == 0:
            logger.info(f"\n[探索] 已访问 {vp_idx+1}/{len(viewpoints)} 视点  "
                        f"覆盖率={coverage_rate*100:.1f}%  "
                        f"累计检测物体={stats_objects_added}")
            self.save_map()

        # 提前结束：覆盖率已达目标
        if coverage_rate >= coverage_target:
            logger.info(f"\n[探索] 覆盖率已达 {coverage_rate*100:.1f}% ≥ "
                        f"{coverage_target*100:.1f}%，提前结束")
            break

    # ── 3. 保存最终结果 ─────────────────────────────────────────
    self.save_map()
    if progress_vis_dir:
        final_vis = str(Path(progress_vis_dir) / "explore_final.png")
        _sem_visualize_exploration_progress(
            self, coverage_map, None, visited_pixels, final_vis
        )

    final_coverage = int(np.count_nonzero(coverage_map)) / total_traversable_px
    result = {
        "total_viewpoints":  len(visited_pixels),
        "total_images":      total_images,
        "total_objects":     len(self._objects_by_id),
        "coverage_rate":     final_coverage,
        "categories_found":  self.get_all_categories(),
    }

    logger.info("\n=== [自动探索] 完成 ===")
    logger.info(f"  访问视点:   {result['total_viewpoints']}")
    logger.info(f"  渲染图像:   {result['total_images']}")
    logger.info(f"  检测物体:   {result['total_objects']}")
    logger.info(f"  地图覆盖率: {result['coverage_rate']*100:.1f}%")
    logger.info(f"  发现类别:   {result['categories_found']}")

    return result


# ── 将独立函数绑定为 SemanticMapManager 的方法 ─────────────────
SemanticMapManager.sample_viewpoints         = _sem_sample_viewpoints
SemanticMapManager.order_viewpoints_greedy   = _sem_order_viewpoints_greedy
SemanticMapManager.build_camera_params       = _sem_build_camera_params
SemanticMapManager.compute_c2w_matrix        = _sem_compute_c2w_matrix
SemanticMapManager.auto_explore_and_annotate = auto_explore_and_annotate
