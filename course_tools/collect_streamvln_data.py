#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Course tool: collect StreamVLN samples for one selected 3DGS target.

This script wraps the course_tools local collection backend's existing mapping,
planning, sampling, and rendering logic into a simplified workflow for students:

1. Load/create a 2D traversability map from a 3DGS PLY.
2. Select a target stopping point on the map.
3. Sample many random starts to that fixed goal.
4. Render RGB frames and export StreamVLN annotations.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
COURSE_TOOLS_ROOT = Path(__file__).resolve().parent
COLLECTION_BACKEND_ROOT = COURSE_TOOLS_ROOT / "vln_collection_backend"
GS_SIMULATOR_ROOT = COURSE_TOOLS_ROOT / "gs-simulator"
sys.path.insert(0, str(COLLECTION_BACKEND_ROOT))
sys.path.insert(0, str(GS_SIMULATOR_ROOT))

INFINITY_IMPORT_ERROR: Optional[BaseException] = None
try:
    from main_interactive_startEndSample import Scan2Occ3D  # noqa: E402
    from managers.log_manager import setup_file_logger  # noqa: E402
except BaseException as exc:  # Keep --help usable even when optional collection deps are missing.
    INFINITY_IMPORT_ERROR = exc
    Scan2Occ3D = object  # type: ignore[assignment]

    def setup_file_logger(output_dir: str):  # type: ignore[no-redef]
        return None

VALID_ACTIONS = {-1, 0, 1, 2, 3}


class CourseStreamVLNCollector(Scan2Occ3D):
    """Thin course-focused wrapper around the local Scan2Occ3D backend."""

    def __init__(self, args: SimpleNamespace):
        self._skip_initial_render_manager = bool(getattr(args, "skip_render", False))
        super().__init__(args)
        self.target_confirmed = False
        self.window_name = "StreamVLN Course Target Selector"
        self._display_scale = 1.0
        self._display_offset = (0, 0)
        self.preview_enabled = bool(getattr(args, "preview", True))
        self.camera_width = int(getattr(args, "camera_width", 640))
        self.camera_height_px = int(getattr(args, "camera_height_px", 400))
        self.camera_fx = float(getattr(args, "camera_fx", 317.25))
        self.camera_fy = float(getattr(args, "camera_fy", 314.72))
        self.camera_cx = float(getattr(args, "camera_cx", self.camera_width / 2.0))
        self.camera_cy = float(getattr(args, "camera_cy", self.camera_height_px / 2.0))
        self._preview_window_name = "StreamVLN Data Collection Preview"
        self._preview_map_cache: Dict[int, np.ndarray] = {}
        self.start_region_points: List[Tuple[int, int]] = []
        self.start_region_mask: Optional[np.ndarray] = None
        self.start_region_window_name = "StreamVLN Start Region Selector"

    def _init_render_manager(self):
        if getattr(self, "_skip_initial_render_manager", False):
            self.render_manager = None
            return
        return super()._init_render_manager()

    def initialize_map(self, ply_path: Optional[str], scene_name: str, load_existing: bool, resolution: float) -> None:
        """Load or create map data without entering the original full interactive UI."""
        if ply_path is not None:
            self.ply_path = ply_path

        from managers.occupancy_map_manager import OccupancyMapManager

        self.map_manager = OccupancyMapManager(scene_name=scene_name, output_dir=self.args.map_dir)

        map_loaded = False
        if load_existing and self.map_manager.map_exists():
            print("[Course] Loading existing map cache...")
            map_loaded = self.map_manager.load_map()

        if not map_loaded:
            if ply_path is None:
                raise ValueError("Must provide --ply_path unless --load_map finds an existing map")
            print("[Course] Creating traversability map from PLY...")
            points, colors = self.map_manager.load_point_cloud(ply_path)
            self.map_manager.create_occupancy_grid(points, colors, resolution=resolution)
            self.map_manager.save_map()
            print(f"[Course] Map cached for scene '{scene_name}'")

        self._load_map_data_into_state()
        self._print_map_stats()

    def _load_map_data_into_state(self) -> None:
        map_data = self.map_manager.get_map_data()
        self.obstacle_map = map_data["obstacle_map"]
        self.expanded_traversability = map_data["expanded_traversability"]
        self.original_traversability = map_data["traversability_mask"]
        self.point_cloud_coverage = map_data["point_cloud_coverage"]
        self.display_image = map_data["color_projection"]
        self.grid_width = map_data["grid_width"]
        self.grid_height = map_data["grid_height"]
        self.grid_resolution = map_data["grid_resolution"]
        self.min_pt = map_data["min_pt"]
        self.max_pt = map_data["max_pt"]
        self._path_planner = None
        self._trajectory_paths = {}

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        grid_x = int(round((world_x - float(self.min_pt[0])) / self.grid_resolution))
        grid_y = int(round((float(self.max_pt[1]) - world_y) / self.grid_resolution))
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        return grid_x, grid_y

    def _print_map_stats(self) -> None:
        total_pixels = self.grid_width * self.grid_height
        coverage_pixels = int(np.count_nonzero(self.point_cloud_coverage))
        obstacle_pixels = int(np.count_nonzero(self.obstacle_map))
        traversable_pixels = int(np.count_nonzero(self.original_traversability))
        expanded_traversable_pixels = int(np.count_nonzero(self.expanded_traversability))
        print("\n[Course] Map statistics")
        print(f"  grid: {self.grid_width} x {self.grid_height}")
        print(f"  resolution: {self.grid_resolution:.4f} m/cell")
        print(f"  coverage cells: {coverage_pixels}/{total_pixels}")
        print(f"  obstacle cells: {obstacle_pixels}")
        print(f"  traversable cells: {traversable_pixels}")
        print(f"  safe traversable cells: {expanded_traversable_pixels}\n")

    def select_start_region_interactively(self) -> None:
        """Let students draw a polygon that constrains random start sampling."""
        if self.display_image is None:
            raise RuntimeError("Map display image is not initialized")

        self.start_region_points = []
        self.start_region_mask = None

        cv2.namedWindow(self.start_region_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.start_region_window_name, 1100, 850)
        cv2.setMouseCallback(self.start_region_window_name, self._start_region_mouse_callback)

        print("[Course] Start region selection")
        print("  Left click: add polygon vertex")
        print("  Right click or Backspace: remove last vertex")
        print("  Enter/Space: confirm polygon")
        print("  A: use all safe traversable area")
        print("  R: reset polygon")
        print("  Q/Esc: quit\n")

        confirmed = False
        while True:
            canvas = self._draw_start_region_canvas()
            cv2.imshow(self.start_region_window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):
                if len(self.start_region_points) >= 3:
                    self._build_start_region_mask()
                    confirmed = True
                    break
                print("[Course] Please choose at least 3 vertices, or press A to use all traversable area.")
            elif key in (ord("a"), ord("A")):
                self.start_region_points = []
                self.start_region_mask = None
                confirmed = True
                print("[Course] Start region: all safe traversable cells")
                break
            elif key in (8, 127):
                if self.start_region_points:
                    self.start_region_points.pop()
            elif key in (ord("r"), ord("R")):
                self.start_region_points = []
                self.start_region_mask = None
            elif key in (ord("q"), ord("Q"), 27):
                break

        cv2.destroyWindow(self.start_region_window_name)
        if not confirmed:
            raise RuntimeError("Start region selection cancelled")

        if self.start_region_mask is not None:
            allowed_cells = int(np.count_nonzero(self.start_region_mask & self.expanded_traversability))
            print(f"[Course] Start region confirmed: {allowed_cells} safe traversable cells inside polygon\n")

    def _start_region_mouse_callback(self, event, x, y, flags, param) -> None:
        grid_point = self._screen_to_grid(x, y)
        if grid_point is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_region_points.append(grid_point)
        elif event == cv2.EVENT_RBUTTONDOWN and self.start_region_points:
            self.start_region_points.pop()

    def _draw_start_region_canvas(self) -> np.ndarray:
        canvas = self._create_selectable_map_canvas()
        points = [self._grid_to_display(point) for point in self.start_region_points]

        if len(points) >= 2:
            for idx in range(len(points) - 1):
                cv2.line(canvas, points[idx], points[idx + 1], (255, 180, 0), 2)
        if len(points) >= 3:
            overlay = canvas.copy()
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 180, 255))
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [polygon], True, (0, 220, 255), 2)

        for idx, point in enumerate(points):
            cv2.circle(canvas, point, 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(idx + 1), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        instructions = [
            "Draw allowed random-start region",
            "Left click: add vertex | Right click/Backspace: undo",
            "Enter/Space: confirm | A: all traversable | R: reset | Q/Esc: quit",
        ]
        for row, text in enumerate(instructions):
            cv2.putText(canvas, text, (20, 30 + row * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return canvas

    def _build_start_region_mask(self) -> None:
        mask = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        polygon = np.array(self.start_region_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        self.start_region_mask = mask.astype(bool)

    def _create_selectable_map_canvas(self) -> np.ndarray:
        canvas = self._create_base_map_image()
        if self._display_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self._display_scale, fy=self._display_scale, interpolation=cv2.INTER_NEAREST)
        if self._display_offset != (0, 0):
            offset_x, offset_y = self._display_offset
            padded = np.zeros((canvas.shape[0] + offset_y * 2, canvas.shape[1] + offset_x * 2, 3), dtype=np.uint8)
            padded[offset_y:offset_y + canvas.shape[0], offset_x:offset_x + canvas.shape[1]] = canvas
            canvas = padded
        return canvas

    def _screen_to_grid(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        offset_x, offset_y = self._display_offset
        grid_x = int(round((x - offset_x) / self._display_scale))
        grid_y = int(round((y - offset_y) / self._display_scale))
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return (grid_x, grid_y)
        return None

    def _grid_to_display(self, point: Tuple[int, int]) -> Tuple[int, int]:
        offset_x, offset_y = self._display_offset
        return (
            int(round(point[0] * self._display_scale + offset_x)),
            int(round(point[1] * self._display_scale + offset_y)),
        )

    def _create_base_map_image(self) -> np.ndarray:
        if self.display_image is not None:
            base = self.display_image.copy()
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            elif base.shape[2] == 4:
                base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
        else:
            base = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        if base.shape[:2] != (self.grid_height, self.grid_width):
            base = cv2.resize(base, (self.grid_width, self.grid_height), interpolation=cv2.INTER_NEAREST)

        safe = self.expanded_traversability.astype(bool)
        overlay = base.copy()
        overlay[safe] = (80, 140, 80)
        base = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
        return base

    def select_goal_interactively(self) -> Tuple[Tuple[int, int], float]:
        """Simplified OpenCV target selector.

        Left-click to place the target stopping point. Drag from the point to set
        final yaw. Press Enter/Space to confirm, R to reset, Q/Esc to quit.
        """
        if self.display_image is None:
            raise RuntimeError("Map display image is not initialized")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1100, 850)
        cv2.setMouseCallback(self.window_name, self._goal_mouse_callback)

        print("[Course] Target selection")
        print("  Left click: choose target stopping point")
        print("  Drag from target: set final robot heading")
        print("  Enter/Space: confirm")
        print("  R: reset target")
        print("  Q/Esc: quit")

        while True:
            frame = self._render_goal_selection_view()
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10, 32):  # Enter or Space
                if self.goal_set:
                    self.target_confirmed = True
                    break
                print("[Course] Please select a target point first.")
            elif key in (ord("r"), ord("R")):
                self.goal_point = None
                self.goal_set = False
                self.goal_yaw = 0.0
                self.dragging_goal = False
            elif key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt("Target selection cancelled")

        cv2.destroyWindow(self.window_name)
        assert self.goal_point is not None
        return self.goal_point, float(self.goal_yaw)

    def _goal_mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        grid_point = self._screen_to_grid(x, y)
        if grid_point is None:
            return
        grid_x, grid_y = grid_point

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self._is_safe_traversable(grid_x, grid_y):
                print(f"[Course] ({grid_x}, {grid_y}) is not safely traversable; choose another point.")
                return
            self.goal_point = (grid_x, grid_y)
            self.goal_set = True
            self.dragging_goal = True
            self.goal_yaw = 0.0
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_goal and self.goal_point is not None:
            dx = grid_x - self.goal_point[0]
            dy = grid_y - self.goal_point[1]
            if abs(dx) + abs(dy) > 1:
                self.goal_yaw = self._grid_delta_to_world_yaw(dx, dy)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_goal and self.goal_point is not None:
                dx = grid_x - self.goal_point[0]
                dy = grid_y - self.goal_point[1]
                if abs(dx) + abs(dy) > 1:
                    self.goal_yaw = self._grid_delta_to_world_yaw(dx, dy)
            self.dragging_goal = False

    def _is_safe_traversable(self, grid_x: int, grid_y: int) -> bool:
        if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_width or grid_y >= self.grid_height:
            return False
        return bool(self.expanded_traversability[grid_y, grid_x] > 0)

    def _grid_delta_to_world_yaw(self, dx: int, dy: int) -> float:
        # Grid y grows downward, world y grows upward.
        return float(math.atan2(-dy, dx))

    def _render_goal_selection_view(self) -> np.ndarray:
        base = self.create_base_visualization_image(use_color_map=True)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        frame = base.copy()

        if self.goal_set and self.goal_point is not None:
            gx, gy = self.goal_point
            cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1)
            arrow_len = 35
            end_x = int(gx + math.cos(self.goal_yaw) * arrow_len)
            end_y = int(gy - math.sin(self.goal_yaw) * arrow_len)
            cv2.arrowedLine(frame, (gx, gy), (end_x, end_y), (0, 0, 255), 3, tipLength=0.35)
            world = self.grid_to_world(gx, gy, 0.0)
            status = f"target grid=({gx},{gy}) world=({world[0]:.2f},{world[1]:.2f}) yaw={math.degrees(self.goal_yaw):.1f}deg"
        else:
            status = "Click a target stopping point near the object. Drag to set heading."

        help_lines = [
            "StreamVLN Course Tool - Target Selection",
            status,
            "Enter/Space: confirm | R: reset | Q/Esc: quit",
        ]
        y = 25
        for line in help_lines:
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
            y += 27

        return frame

    def _screen_to_grid(self, x: int, y: int) -> Tuple[Optional[int], Optional[int]]:
        # OpenCV displays the image at native coordinates in this script; WINDOW_NORMAL
        # scaling is handled by OpenCV but mouse callbacks still use image coordinates
        # for imshow content in typical HighGUI backends. Clamp defensively.
        grid_x = int(round(x))
        grid_y = int(round(y))
        if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_width or grid_y >= self.grid_height:
            return None, None
        return grid_x, grid_y

    def sample_course_trajectories(
        self,
        target_name: str,
        instruction: str,
        num_episodes: int,
        min_distance: float,
        max_attempt_rounds: int,
        min_action_steps: int,
        max_action_steps: int,
    ) -> List[Dict[str, Any]]:
        """Sample trajectories until enough quality samples are available."""
        if not self.goal_set:
            raise RuntimeError("Goal must be selected before sampling")

        collected_start = len(self.trajectories)
        rounds = 0
        while self._count_quality_pending(collected_start, min_action_steps, max_action_steps) < num_episodes:
            rounds += 1
            if rounds > max_attempt_rounds:
                break

            remaining = num_episodes - self._count_quality_pending(collected_start, min_action_steps, max_action_steps)
            request_n = max(remaining * 2, remaining)
            before = len(self.trajectories)
            self.sample_random_starts_from_goal(num_starts=request_n, min_dist_m=min_distance)
            after = len(self.trajectories)
            if after == before:
                print("[Course] Sampling produced no new trajectories; stopping early.")
                break

            for trajectory in self.trajectories[before:after]:
                trajectory["target_name"] = target_name
                trajectory["instruction"] = instruction
                trajectory["course_stage"] = "stage1_target_random_starts"

        selected = []
        for trajectory in self.trajectories[collected_start:]:
            if self._trajectory_passes_action_filter(trajectory, min_action_steps, max_action_steps):
                selected.append(trajectory)
            if len(selected) >= num_episodes:
                break

        if len(selected) < num_episodes:
            print(f"[Course] Warning: requested {num_episodes}, got {len(selected)} quality trajectories.")

        for new_id, trajectory in enumerate(selected):
            trajectory["trajectory_id"] = new_id

        self.trajectories = selected
        self.trajectory_id_counter = len(selected)
        self.save_trajectories()
        return selected

    def _count_quality_pending(self, start_index: int, min_action_steps: int, max_action_steps: int) -> int:
        return sum(
            1 for trajectory in self.trajectories[start_index:]
            if self._trajectory_passes_action_filter(trajectory, min_action_steps, max_action_steps)
        )

    def _trajectory_passes_action_filter(self, trajectory: Dict[str, Any], min_action_steps: int, max_action_steps: int) -> bool:
        try:
            episode = self._build_episode_from_trajectory(
                trajectory,
                instruction=trajectory.get("instruction", ""),
                render=False,
                write_outputs=False,
            )
        except Exception:
            return False
        action_count = len(episode.get("actions", []))
        return min_action_steps <= action_count <= max_action_steps

    def _get_random_traversable_points(
        self,
        count: Optional[int] = None,
        n: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[int, int]]:
        """Sample start candidates, constrained by the optional student polygon."""
        requested = count if count is not None else n
        if requested is None:
            requested = kwargs.get("num_points")
        if requested is None:
            raise TypeError("_get_random_traversable_points requires count, n, or num_points")

        if self.start_region_mask is None:
            return super()._get_random_traversable_points(n=requested, **kwargs)

        valid_mask = self.expanded_traversability.astype(bool) & self.start_region_mask
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            raise RuntimeError("Selected start region contains no safe traversable cells")

        min_dist_m = float(kwargs.get("min_dist_m", 0.0))
        exclude_grid = kwargs.get("exclude_grid")
        exclude_radius_m = float(kwargs.get("exclude_radius_m", 0.0))
        if self.grid_resolution <= 0:
            raise ValueError("Map grid_resolution is not initialized")

        min_dist_px = max(0.0, min_dist_m / self.grid_resolution)
        exclude_radius_px = max(0.0, exclude_radius_m / self.grid_resolution)

        candidates = [(int(x), int(y)) for x, y in zip(xs, ys)]
        np.random.shuffle(candidates)

        selected: List[Tuple[int, int]] = []
        for point in candidates:
            if exclude_grid is not None:
                distance_to_excluded = np.hypot(point[0] - exclude_grid[0], point[1] - exclude_grid[1])
                if distance_to_excluded < exclude_radius_px:
                    continue

            if min_dist_px > 0 and any(
                np.hypot(point[0] - existing[0], point[1] - existing[1]) < min_dist_px
                for existing in selected
            ):
                continue

            selected.append(point)
            if len(selected) >= requested:
                break

        if not selected:
            raise RuntimeError(
                "Selected start region has traversable cells, but none satisfy the distance constraints"
            )

        print(f"[Course] Sampled {len(selected)} start candidates from selected polygon region")
        return selected

    def render_and_export_dataset(self, trajectories: List[Dict[str, Any]], instruction: str, skip_render: bool) -> None:
        self._remove_unselected_trajectory_outputs({int(t["trajectory_id"]) for t in trajectories})
        episodes = []
        annotations = []

        for index, trajectory in enumerate(trajectories):
            print(f"[Course] Processing trajectory {index + 1}/{len(trajectories)} (id={trajectory['trajectory_id']})")
            episode = self._build_episode_from_trajectory(trajectory, instruction=instruction, render=not skip_render)
            episode["episode_id"] = f"episode_{index:04d}"
            episodes.append(episode)
            annotations.append(self._episode_to_streamvln_annotation(episode, index))

        annotate_path = Path(self.output_dir) / "annotate_episodes.json"
        with annotate_path.open("w", encoding="utf-8") as f:
            json.dump({"num_episodes": len(episodes), "episodes": episodes}, f, indent=2, ensure_ascii=False)

        annotations_path = Path(self.output_dir) / "annotations.json"
        with annotations_path.open("w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"[Course] Wrote {annotate_path}")
        print(f"[Course] Wrote {annotations_path}")

    def _build_episode_from_trajectory(
        self,
        trajectory: Dict[str, Any],
        instruction: str,
        render: bool,
        write_outputs: bool = True,
    ) -> Dict[str, Any]:
        trajectory_id = trajectory["trajectory_id"]
        full_path = self._plan_full_path_for_trajectory(trajectory)
        start_yaw = float(trajectory["start_yaw"])
        goal_yaw = float(trajectory["goal_yaw"])

        rotations, translations, _, _, actions = self.generate_camera_poses(
            full_path,
            mode="discrete",
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
        )
        if rotations is None:
            raise RuntimeError(f"Failed to generate camera poses for trajectory {trajectory_id}")

        camera_json_path = Path(self.output_dir) / f"camera_poses_traj_{trajectory_id:04d}.json"
        if write_outputs:
            self.save_camera_poses_json(
                rotations,
                translations,
                len(rotations),
                actions=actions,
                mode="discrete",
                output_path=str(camera_json_path),
                full_path=full_path,
                start_yaw=start_yaw,
                goal_yaw=goal_yaw,
            )
            with camera_json_path.open("r", encoding="utf-8") as f:
                camera_data = json.load(f)
        else:
            camera_data = self._camera_pose_data_from_arrays(
                rotations,
                translations,
                actions,
                full_path,
                start_yaw,
                goal_yaw,
            )

        render_dir = Path(self.output_dir) / f"render_trajectory_{trajectory_id:04d}"
        if render and write_outputs:
            self._render_camera_data(camera_data, render_dir)
        elif write_outputs:
            rgb_dir = render_dir / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)

        start_world = np.array(trajectory["start_position"])
        goal_world = np.array(trajectory["goal_position"])
        return {
            "episode_id": f"episode_{trajectory_id:04d}",
            "trajectory_id": trajectory_id,
            "instruction": instruction,
            "target_name": trajectory.get("target_name", ""),
            "start_position": start_world.tolist(),
            "start_yaw": start_yaw,
            "goal_position": goal_world.tolist(),
            "goal_yaw": goal_yaw,
            "mode": "discrete",
            "num_cameras": camera_data["num_cameras"],
            "cameras": camera_data.get("cameras", []),
            "render_dir": str(render_dir),
            "scene_bounds": {"min": self.min_pt.tolist(), "max": self.max_pt.tolist()},
            "grid_resolution": float(self.grid_resolution),
            "actions": camera_data.get("actions", []),
            "num_actions": camera_data.get("num_actions", 0),
        }

    def _camera_pose_data_from_arrays(
        self,
        rotations: List[np.ndarray],
        translations: List[np.ndarray],
        actions: List[int],
        full_path: List[Tuple[int, int]],
        start_yaw: float,
        goal_yaw: float,
    ) -> Dict[str, Any]:
        width = int(getattr(self, "camera_width", 640))
        height = int(getattr(self, "camera_height_px", 400))
        fx = float(getattr(self, "camera_fx", 317.25))
        fy = float(getattr(self, "camera_fy", 314.72))
        cx = float(getattr(self, "camera_cx", width / 2.0))
        cy = float(getattr(self, "camera_cy", height / 2.0))
        camera_data: Dict[str, Any] = {
            "mode": "discrete",
            "num_cameras": len(rotations),
            "num_actions": len(actions),
            "actions": actions,
            "action_definitions": {
                "1": "forward 0.25m",
                "2": "turn_left 15deg",
                "3": "turn_right 15deg",
            },
            "cameras": [],
        }
        for rotation, translation in zip(rotations, translations):
            camera_data["cameras"].append({
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "R": rotation.tolist(),
                "T": translation.tolist(),
            })
        if len(full_path) >= 2:
            start_x, start_y = full_path[0]
            goal_x, goal_y = full_path[-1]
            camera_data["start_grid"] = [int(start_x), int(start_y)]
            camera_data["goal_grid"] = [int(goal_x), int(goal_y)]
            camera_data["start_yaw"] = float(start_yaw)
            camera_data["goal_yaw"] = float(goal_yaw)
            camera_data["start_world"] = self.grid_to_world(start_x, start_y, 0.0).tolist()
            camera_data["goal_world"] = self.grid_to_world(goal_x, goal_y, 0.0).tolist()
            camera_data["path_length"] = float(len(full_path) * self.grid_resolution)
            camera_data["num_waypoints"] = len(full_path)
        return camera_data

    def _remove_unselected_trajectory_outputs(self, selected_ids: set[int]) -> None:
        output_dir = Path(self.output_dir)
        for camera_json_path in output_dir.glob("camera_poses_traj_*.json"):
            try:
                trajectory_id = int(camera_json_path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if trajectory_id not in selected_ids:
                camera_json_path.unlink()

        for render_dir in output_dir.glob("render_trajectory_*"):
            if not render_dir.is_dir():
                continue
            try:
                trajectory_id = int(render_dir.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if trajectory_id not in selected_ids:
                shutil.rmtree(render_dir)

    def _plan_full_path_for_trajectory(self, trajectory: Dict[str, Any]) -> List[Tuple[int, int]]:
        start_world = np.array(trajectory["start_position"])
        goal_world = np.array(trajectory["goal_position"])
        start_point = self._world_to_grid(start_world)
        goal_point = self._world_to_grid(goal_world)
        waypoints = [self._world_to_grid(np.array(wp)) for wp in trajectory.get("waypoints", [])]

        sequence = [start_point] + waypoints + [goal_point]
        full_path: List[Tuple[int, int]] = []
        planner = self._get_path_planner()
        for start, goal in zip(sequence[:-1], sequence[1:]):
            segment_path = planner.plan(start, goal)["result"]
            if not segment_path:
                raise RuntimeError(f"No path from {start} to {goal}")
            full_path.extend(segment_path[1:] if full_path else segment_path)
        return full_path

    def _world_to_grid(self, world: np.ndarray) -> Tuple[int, int]:
        grid_x = int((world[0] - self.min_pt[0]) / self.grid_resolution)
        grid_y = int((self.max_pt[1] - world[1]) / self.grid_resolution)
        return grid_x, grid_y

    def _render_camera_data(self, camera_data: Dict[str, Any], render_dir: Path) -> None:
        from managers.render_manager import RenderManager

        preview_callback = self._make_preview_callback(camera_data) if self.preview_enabled else None
        render_manager = RenderManager(
            ply_path=self.ply_path,
            sh_degree=3,
            background_color=[0, 0, 0],
        )
        success = render_manager.render_trajectory(
            camera_poses=camera_data.get("cameras", []),
            output_dir=str(render_dir),
            create_video=False,
            enable_depth=self.preview_enabled,
            camera_config=camera_data.get("camera_config"),
            preview_callback=preview_callback,
        )
        if not success:
            raise RuntimeError(f"Rendering failed for {render_dir}")

    def _make_preview_callback(self, camera_data: Dict[str, Any]):
        try:
            map_image = self._build_trajectory_map_preview(camera_data)
        except Exception as exc:
            print(f"[preview] Failed to prepare trajectory preview: {exc}")
            return None

        visited_points: List[Tuple[int, int]] = []

        def _preview(frame_index: int, rgb: np.ndarray, depth_color: Optional[np.ndarray]) -> bool:
            try:
                current_map = map_image.copy()
                cameras = camera_data.get("cameras", [])
                camera = cameras[frame_index] if frame_index < len(cameras) else {}
                position = camera.get("T") or camera.get("position")
                if position is not None:
                    point = self._world_to_grid(np.array(position, dtype=float))
                    if not visited_points or visited_points[-1] != point:
                        visited_points.append(point)

                if len(visited_points) >= 2:
                    cv2.polylines(
                        current_map,
                        [np.array(visited_points, dtype=np.int32)],
                        False,
                        (0, 220, 0),
                        4,
                    )
                for trail_point in visited_points:
                    cv2.circle(current_map, trail_point, 2, (0, 180, 0), -1)

                if visited_points:
                    current_point = visited_points[-1]
                    cv2.circle(current_map, current_point, 7, (0, 255, 255), -1)
                    cv2.circle(current_map, current_point, 10, (0, 0, 0), 2)
                    self._draw_current_pose_arrow(current_map, current_point, camera, visited_points)

                self._draw_preview_endpoints(current_map, camera_data)

                cv2.putText(
                    current_map,
                    f"Frame {frame_index + 1}/{len(camera_data.get('cameras', []))}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    current_map,
                    f"Frame {frame_index + 1}/{len(camera_data.get('cameras', []))}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                preview = self._compose_preview_window(current_map, rgb, depth_color)
                cv2.imshow("StreamVLN Data Collection Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.preview_enabled = False
                    cv2.destroyWindow("StreamVLN Data Collection Preview")
                    return False
                return True
            except Exception as exc:
                print(f"[preview] Disabled after display error: {exc}")
                self.preview_enabled = False
                return False

        return _preview

    def _build_trajectory_map_preview(self, camera_data: Dict[str, Any]) -> np.ndarray:
        base = self.create_base_visualization_image(use_color_map=False)
        path = camera_data.get("path") or []
        if len(path) >= 2:
            points = [(int(p[0]), int(p[1])) for p in path]
            cv2.polylines(base, [np.array(points, dtype=np.int32)], False, (0, 160, 0), 2)
        self._draw_preview_endpoints(base, camera_data)
        return base

    def _draw_preview_endpoints(self, image: np.ndarray, camera_data: Dict[str, Any]) -> None:
        cameras = camera_data.get("cameras", [])
        if not cameras:
            return
        start_position = cameras[0].get("T") or cameras[0].get("position")
        goal_position = cameras[-1].get("T") or cameras[-1].get("position")
        if start_position is not None:
            start_point = self._world_to_grid(np.array(start_position, dtype=float))
            cv2.circle(image, start_point, 10, (0, 255, 0), -1)
            cv2.circle(image, start_point, 13, (0, 0, 0), 2)
        if goal_position is not None:
            goal_point = self._world_to_grid(np.array(goal_position, dtype=float))
            cv2.circle(image, goal_point, 10, (0, 0, 255), -1)
            cv2.circle(image, goal_point, 13, (0, 0, 0), 2)

    def _draw_current_pose_arrow(
        self,
        image: np.ndarray,
        current_point: Tuple[int, int],
        camera: Dict[str, Any],
        visited_points: List[Tuple[int, int]],
    ) -> None:
        direction = self._camera_forward_grid_direction(camera)
        if direction is None and len(visited_points) >= 2:
            prev = np.array(visited_points[-2], dtype=float)
            curr = np.array(current_point, dtype=float)
            delta = curr - prev
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                direction = delta / norm
        if direction is None:
            return

        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return
        direction = direction / norm
        start = np.array(current_point, dtype=float)
        end = start + direction * 28.0
        start_i = tuple(np.round(start).astype(int))
        end_i = tuple(np.round(end).astype(int))
        cv2.arrowedLine(image, start_i, end_i, (0, 0, 0), 6, tipLength=0.35)
        cv2.arrowedLine(image, start_i, end_i, (0, 255, 255), 3, tipLength=0.35)

    def _camera_forward_grid_direction(self, camera: Dict[str, Any]) -> Optional[np.ndarray]:
        rotation = camera.get("R")
        if rotation is None:
            return None
        try:
            r = np.array(rotation, dtype=float)
            if r.shape != (3, 3):
                return None
            forward_world = r[:, 2]
            grid_dx = forward_world[0] / self.grid_resolution
            grid_dy = -forward_world[1] / self.grid_resolution
            direction = np.array([grid_dx, grid_dy], dtype=float)
            norm = np.linalg.norm(direction)
            if norm <= 1e-6:
                return None
            return direction / norm
        except Exception:
            return None

    def _compose_preview_window(self, map_image: np.ndarray, rgb: np.ndarray, depth_color: Optional[np.ndarray]) -> np.ndarray:
        map_panel = self._fit_preview_panel(map_image, 640, 800)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 and rgb.shape[2] >= 3 else rgb
        rgb_panel = self._fit_preview_panel(rgb_bgr, 640, 400)
        if depth_color is None:
            depth_color = np.zeros_like(rgb_panel)
            cv2.putText(depth_color, "Depth unavailable", (32, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        depth_panel = self._fit_preview_panel(depth_color, 640, 400)

        cv2.putText(map_panel, "2D Trajectory", (16, 760), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(map_panel, "2D Trajectory", (16, 760), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(rgb_panel, "RGB", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(rgb_panel, "RGB", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(depth_panel, "Depth", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(depth_panel, "Depth", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        right_panel = np.vstack([rgb_panel, depth_panel])
        return np.hstack([map_panel, right_panel])

    @staticmethod
    def _fit_preview_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image_height, image_width = image.shape[:2]
        scale = min(width / image_width, height / image_height)
        resized = cv2.resize(image, (max(1, int(image_width * scale)), max(1, int(image_height * scale))))
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        offset_y = (height - resized.shape[0]) // 2
        offset_x = (width - resized.shape[1]) // 2
        panel[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
        return panel

    def _episode_to_streamvln_annotation(self, episode: Dict[str, Any], index: int) -> Dict[str, Any]:
        actions = list(episode.get("actions", []))
        if actions and actions[-1] == 0:
            actions = actions[:-1]
        actions = [-1] + actions
        render_folder = Path(episode["render_dir"]).name
        return {
            "id": index,
            "video": render_folder,
            "instructions": [episode["instruction"]],
            "actions": actions,
        }


def build_scan_args(args: argparse.Namespace) -> SimpleNamespace:
    base_dir = str(COURSE_TOOLS_ROOT / "data")
    output_dir = str(Path(args.output_dir).resolve())
    map_dir = str(Path(args.map_dir).resolve()) if args.map_dir else str(Path(base_dir) / "scans" / "maps")
    return SimpleNamespace(
        base_dir=base_dir,
        output_dir=output_dir,
        ply_path=args.ply_path,
        scene_name=args.scene_name,
        load_map=args.load_map,
        edit_map=False,
        render=not args.skip_render,
        annotate=True,
        camera_type="single",
        enable_depth=bool(not args.skip_render and not args.no_preview),
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        camera_height_px=args.camera_height_px,
        camera_fx=args.camera_fx,
        camera_fy=args.camera_fy,
        camera_cx=args.camera_cx,
        camera_cy=args.camera_cy,
        skip_render=args.skip_render,
        preview=bool(not args.no_preview),
        exp_name=Path(output_dir).name,
        map_dir=map_dir,
    )


def prompt_if_missing(value: Optional[str], prompt: str) -> str:
    if value:
        return value.strip()
    while True:
        entered = input(prompt).strip()
        if entered:
            return entered



def write_target_json(
    output_dir: Path,
    target_name: str,
    instruction: str,
    collector: CourseStreamVLNCollector,
    num_episodes: int,
) -> None:
    assert collector.goal_point is not None
    goal_world = collector.grid_to_world(collector.goal_point[0], collector.goal_point[1], 0.0)
    start_region_world = []
    for point in collector.start_region_points:
        world = collector.grid_to_world(point[0], point[1], 0.0)
        start_region_world.append(world.tolist())

    target = {
        "target_name": target_name,
        "instruction": instruction,
        "goal_grid": list(collector.goal_point),
        "goal_position": goal_world.tolist(),
        "goal_yaw": float(collector.goal_yaw),
        "start_region_grid": [list(point) for point in collector.start_region_points],
        "start_region_world": start_region_world,
        "num_episodes": num_episodes,
        "created_at": int(time.time()),
    }
    with (output_dir / "target.json").open("w", encoding="utf-8") as f:
        json.dump(target, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect StreamVLN samples for one selected 3DGS target")
    parser.add_argument("--ply_path", type=str, default=None, help="3DGS PLY path")
    parser.add_argument("--scene_name", type=str, required=True, help="Scene name for map cache")
    parser.add_argument("--output_dir", type=str, required=True, help="Dataset output directory")
    parser.add_argument("--map_dir", type=str, default=None, help="Map cache directory")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of trajectories to collect (default: 50)")
    parser.add_argument("--min_distance", type=float, default=3.0, help="Minimum start-goal distance in meters")
    parser.add_argument("--resolution", type=float, default=0.02, help="Occupancy grid resolution")
    parser.add_argument("--camera_height", type=float, default=1.45, help="Final rendered camera height above ground; default approximates G2 head camera")
    parser.add_argument("--camera_width", type=int, default=640, help="Rendered image width; default matches G2 head camera render product")
    parser.add_argument("--camera_height_px", type=int, default=400, help="Rendered image height; default matches G2 head camera render product")
    parser.add_argument("--camera_fx", type=float, default=317.25, help="Rendered camera fx; default matches G2 head_front_Camera")
    parser.add_argument("--camera_fy", type=float, default=314.72, help="Rendered camera fy; default matches G2 head_front_Camera")
    parser.add_argument("--camera_cx", type=float, default=320.0, help="Rendered camera cx; default is image center for 640 width")
    parser.add_argument("--camera_cy", type=float, default=200.0, help="Rendered camera cy; default is image center for 400 height")
    parser.add_argument("--target_name", type=str, default=None, help="Target object name")
    parser.add_argument("--instruction", type=str, default=None, help="Natural language instruction")
    parser.add_argument("--load_map", action="store_true", help="Load existing map cache")
    parser.add_argument("--skip_render", action="store_true", help="Skip RGB rendering; creates empty rgb dirs for dry-run")
    parser.add_argument("--no_preview", action="store_true", help="Disable OpenCV preview during rendering")
    parser.add_argument("--max_attempt_rounds", type=int, default=5, help="Maximum extra sampling rounds")
    parser.add_argument("--min_action_steps", type=int, default=4, help="Minimum action count before StreamVLN sentinel")
    parser.add_argument("--max_action_steps", type=int, default=120, help="Maximum action count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if INFINITY_IMPORT_ERROR is not None:
        raise SystemExit(
            "Failed to import the local course collection backend. Please install the "
            "collection dependencies first, especially open3d and 3DGS rendering dependencies. Original error: "
            f"{INFINITY_IMPORT_ERROR!r}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logger(str(output_dir))

    scan_args = build_scan_args(args)
    collector = CourseStreamVLNCollector(scan_args)
    collector.initialize_map(args.ply_path, args.scene_name, args.load_map, args.resolution)

    collector.select_start_region_interactively()
    collector.select_goal_interactively()

    target_name = prompt_if_missing(args.target_name, "Target object name: ")

    if args.instruction:
        instruction = args.instruction.strip()
    else:
        default_instruction = f"Go to the {target_name} and stop in front of it."
        entered = input(f"Instruction [{default_instruction}]: ").strip()
        instruction = entered or default_instruction

    write_target_json(output_dir, target_name, instruction, collector, args.num_episodes)

    trajectories = collector.sample_course_trajectories(
        target_name=target_name,
        instruction=instruction,
        num_episodes=args.num_episodes,
        min_distance=args.min_distance,
        max_attempt_rounds=args.max_attempt_rounds,
        min_action_steps=args.min_action_steps,
        max_action_steps=args.max_action_steps,
    )
    collector.render_and_export_dataset(trajectories, instruction=instruction, skip_render=args.skip_render)
    print("\n[Course] Done.")
    print(f"[Course] Dataset root: {output_dir}")
    print(f"[Course] Train with StreamVLN --video_folder {output_dir}")


if __name__ == "__main__":
    main()
