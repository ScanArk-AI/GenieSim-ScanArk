#!/usr/bin/env python3
"""
Set robot base pose from an episode JSON file.

Usage:
    python scripts/set_robot_pose_from_episode.py \
        --episode_json /amax/gennisim/episode_val_seen.json \
        --episode_id episode_0004_0

or:
    python scripts/set_robot_pose_from_episode.py \
        --episode_json /amax/gennisim/episode_val_seen.json \
        --episode_index 0
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/amax/gennisim/scripts")
from vln_env import VLNEnv


def yaw_to_wxyz(yaw_rad: float):
    half = yaw_rad / 2.0
    qw = math.cos(half)
    qx = 0.0
    qy = 0.0
    qz = math.sin(half)
    return qw, qx, qy, qz


def main():
    parser = argparse.ArgumentParser(description="Set robot pose from episode JSON start pose")
    parser.add_argument("--episode_json", required=True, help="Path to episode JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_id", help="Episode ID to use")
    group.add_argument("--episode_index", type=int, help="Episode index to use (0-based)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12347)
    args = parser.parse_args()

    with open(args.episode_json, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.episode_json}")

    if args.episode_id is not None:
        episode = None
        for ep in episodes:
            if ep.get("episode_id") == args.episode_id:
                episode = ep
                break
        if episode is None:
            raise RuntimeError(f"Episode ID not found: {args.episode_id}")
    else:
        if args.episode_index < 0 or args.episode_index >= len(episodes):
            raise RuntimeError(f"Episode index out of range: {args.episode_index}")
        episode = episodes[args.episode_index]

    start_world = episode["start_world"]
    start_yaw = episode["start_yaw"]
    x, y, _ = start_world
    qw, qx, qy, qz = yaw_to_wxyz(start_yaw)

    env = VLNEnv(host=args.host, port=args.port)
    try:
        # Use the robot's current standing height from the loaded scene.
        _, robot_info = env._send_cmd("robot_pose")
        z = float(robot_info["z"])

        print(f"Episode: {episode.get('episode_id', '<unknown>')}")
        print(f"Start world (xy from episode, z from scene): ({x:.4f}, {y:.4f}, {z:.4f})")
        print(f"Start yaw (rad): {start_yaw:.6f}")
        print(f"Quaternion (wxyz): ({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})")

        info = env.set_robot_pose(x, y, z, qw, qx, qy, qz)
        if "error" in info:
            raise RuntimeError(info["error"])
        print("Robot pose set successfully:")
        print(json.dumps(info, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
