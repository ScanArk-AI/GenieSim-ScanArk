#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate a StreamVLN navigation dataset directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

VALID_ACTIONS = {-1, 0, 1, 2, 3}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_annotations(data_root: Path) -> List[Dict[str, Any]]:
    annotation_path = data_root / "annotations.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("annotations.json must be a JSON list")
    return data


def check_item(data_root: Path, item: Dict[str, Any], index: int) -> List[str]:
    errors: List[str] = []

    for key in ("video", "instructions", "actions"):
        if key not in item:
            errors.append(f"missing field: {key}")

    video = item.get("video")
    if not isinstance(video, str) or not video:
        errors.append("video must be a non-empty relative path string")
        video_dir = None
    else:
        if Path(video).is_absolute():
            errors.append("video must be relative, not absolute")
        video_dir = data_root / video
        if not video_dir.exists():
            errors.append(f"video dir does not exist: {video_dir}")

    rgb_files: List[Path] = []
    if video_dir is not None and video_dir.exists():
        rgb_dir = video_dir / "rgb"
        if not rgb_dir.exists():
            errors.append(f"rgb dir does not exist: {rgb_dir}")
        elif not rgb_dir.is_dir():
            errors.append(f"rgb path is not a directory: {rgb_dir}")
        else:
            rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not rgb_files:
                errors.append(f"rgb dir contains no image frames: {rgb_dir}")

    instructions = item.get("instructions")
    if isinstance(instructions, str):
        if not instructions.strip():
            errors.append("instructions string is empty")
    elif isinstance(instructions, list):
        if not instructions:
            errors.append("instructions list is empty")
        elif not all(isinstance(x, str) and x.strip() for x in instructions):
            errors.append("instructions list must contain non-empty strings")
    else:
        errors.append("instructions must be a string or list of strings")

    actions = item.get("actions")
    if not isinstance(actions, list):
        errors.append("actions must be a list")
    else:
        if len(actions) < 4:
            errors.append(f"actions length must be >= 4, got {len(actions)}")
        if actions and actions[0] != -1:
            errors.append("actions must start with -1 sentinel")
        invalid = [a for a in actions if not isinstance(a, int) or a not in VALID_ACTIONS]
        if invalid:
            errors.append(f"actions contain invalid values: {sorted(set(invalid))}")
        effective_action_count = max(0, len(actions) - 1)
        if rgb_files and len(rgb_files) < effective_action_count:
            errors.append(
                f"not enough rgb frames: {len(rgb_files)} frames for {effective_action_count} effective actions"
            )

    return [f"item {index}: {err}" for err in errors]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check StreamVLN dataset directory")
    parser.add_argument("--data_root", required=True, help="Dataset root containing annotations.json")
    parser.add_argument("--max_errors", type=int, default=50, help="Maximum errors to print")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    annotations = load_annotations(data_root)

    all_errors: List[str] = []
    action_lengths = []
    frame_counts = []
    instruction_counter = Counter()

    for idx, item in enumerate(annotations):
        errors = check_item(data_root, item, idx)
        all_errors.extend(errors)

        actions = item.get("actions")
        if isinstance(actions, list):
            action_lengths.append(len(actions))
        video = item.get("video")
        if isinstance(video, str):
            rgb_dir = data_root / video / "rgb"
            if rgb_dir.is_dir():
                frame_counts.append(len([p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]))
        instructions = item.get("instructions")
        if isinstance(instructions, list) and instructions:
            instruction_counter[instructions[0]] += 1
        elif isinstance(instructions, str):
            instruction_counter[instructions] += 1

    print(f"Dataset root: {data_root}")
    print(f"Samples: {len(annotations)}")
    if action_lengths:
        print(f"Actions length: min={min(action_lengths)} max={max(action_lengths)} avg={sum(action_lengths)/len(action_lengths):.1f}")
    if frame_counts:
        print(f"RGB frames: min={min(frame_counts)} max={max(frame_counts)} avg={sum(frame_counts)/len(frame_counts):.1f}")
    if instruction_counter:
        print(f"Unique instructions: {len(instruction_counter)}")
        for instruction, count in instruction_counter.most_common(3):
            print(f"  {count}x {instruction}")

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} issue(s) found")
        for err in all_errors[: args.max_errors]:
            print(f"  - {err}")
        if len(all_errors) > args.max_errors:
            print(f"  ... {len(all_errors) - args.max_errors} more")
        raise SystemExit(1)

    print("\nOK: dataset looks compatible with StreamVLN navigation training.")


if __name__ == "__main__":
    main()
