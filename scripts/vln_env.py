#!/usr/bin/env python3
"""
VLN Environment Client - Gym-style interface for VLN model inference.

Usage:
    from vln_env import VLNEnv

    env = VLNEnv()
    obs, info = env.reset()           # obs: RGB numpy array (H, W, 3)
    obs, info = env.step("forward")   # actions: forward, backward, turn_left, turn_right, stop
    obs, info = env.observe()         # get current observation without moving

Actions (Habitat-compatible):
    "forward"    - move forward 0.25m
    "backward"   - move backward 0.25m
    "turn_left"  - turn left 15 degrees
    "turn_right" - turn right 15 degrees
    "stop"       - stop (episode ends)

Run VLN app first:
    SIM_REPO_ROOT=/geniesim/main CUDA_VISIBLE_DEVICES=0 omni_python scripts/vln_app.py --config source/geniesim/config/my_scene_vln.yaml
"""

import socket
import struct
import json
import numpy as np
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None


class VLNEnv:
    """Gym-style VLN environment client that communicates with Isaac Sim."""

    ACTIONS = ["forward", "backward", "turn_left", "turn_right", "stop"]

    def __init__(self, host="127.0.0.1", port=12347):
        self.host = host
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        """Connect to the VLN API server."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"[VLNEnv] Connected to {self.host}:{self.port}")

    def _send_cmd(self, cmd):
        """Send command and receive response (state dict + image)."""
        self.sock.sendall((cmd + "\n").encode("utf-8"))

        # Read 4 bytes: header length
        header_len = struct.unpack(">I", self._recv_exact(4))[0]
        # Read header JSON
        header_bytes = self._recv_exact(header_len)
        info = json.loads(header_bytes.decode("utf-8"))
        # Read 4 bytes: image length
        img_len = struct.unpack(">I", self._recv_exact(4))[0]
        # Read image JPEG bytes
        img_bytes = self._recv_exact(img_len) if img_len > 0 else None

        # Decode JPEG to numpy array
        obs = None
        if img_bytes and Image is not None:
            pil_img = Image.open(BytesIO(img_bytes))
            obs = np.array(pil_img)  # (H, W, 3) uint8 RGB
        elif img_bytes:
            # Fallback: return raw JPEG bytes if PIL not available
            obs = img_bytes

        return obs, info

    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            buf += chunk
        return buf

    def reset(self):
        """
        Reset robot to initial position and return first observation.

        Returns:
            obs: numpy array (H, W, 3) uint8 RGB image
            info: dict with keys {x, y, z, yaw}
        """
        return self._send_cmd("reset")

    def step(self, action):
        """
        Execute one discrete action and return new observation.

        Args:
            action: str, one of "forward", "backward", "turn_left", "turn_right", "stop"

        Returns:
            obs: numpy array (H, W, 3) uint8 RGB image
            info: dict with keys {x, y, z, yaw, action}
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.ACTIONS}")
        return self._send_cmd(f"step:{action}")

    def observe(self):
        """
        Get current observation without taking any action.

        Returns:
            obs: numpy array (H, W, 3) uint8 RGB image
            info: dict with keys {x, y, z, yaw}
        """
        return self._send_cmd("observe")

    def list_objects(self):
        """List top-level prims under /World."""
        _, info = self._send_cmd("list_objects")
        return info

    def list_collision(self):
        """List all prims with collision API."""
        _, info = self._send_cmd("list_collision")
        return info

    def list_cameras(self):
        """List all camera prims currently present in the stage."""
        _, info = self._send_cmd("list_cameras")
        return info

    def material_info(self, prim_prefix):
        """List material bindings for meshes under a prim prefix."""
        _, info = self._send_cmd(f"material_info:{prim_prefix}")
        return info

    def list_subtree(self, prim_prefix):
        """List all prims under a prim prefix with their USD type."""
        _, info = self._send_cmd(f"list_subtree:{prim_prefix}")
        return info

    def query_pose(self, prim_path):
        """Query world pose of any prim. Returns dict with position and quaternion."""
        _, info = self._send_cmd(f"query_pose:{prim_path}")
        return info

    def camera_debug(self, prim_path):
        """Inspect a camera prim's world pose, parent pose, relative rotation and camera axes."""
        _, info = self._send_cmd(f"camera_debug:{prim_path}")
        return info

    def set_robot_pose(self, x, y, z, qw, qx, qy, qz):
        """Teleport robot base to a specific world pose (quaternion in wxyz order)."""
        _, info = self._send_cmd(f"set_robot_pose:{x},{y},{z},{qw},{qx},{qy},{qz}")
        return info

    def spawn_robot(self):
        """Trigger the demo robot to spawn and free-fall into the scene."""
        _, info = self._send_cmd("spawn_robot")
        return info

    def robot_pose(self):
        """Get the robot base pose without capturing an image."""
        _, info = self._send_cmd("robot_pose")
        return info

    def spawn_object(self, usd_path, x, y, z, label="spawned_obj"):
        """
        Spawn a USD object at (x, y, z).

        Args:
            usd_path: relative to assets dir, e.g. "objects/genie/cola/Aligned.usd"
            x, y, z: world position
            label: object name (used as prim path /World/Objects/<label>)

        Returns:
            info: dict with status, prim path, position
        """
        _, info = self._send_cmd(f"spawn_object:{usd_path},{x},{y},{z},{label}")
        return info

    def open_gripper(self, arm="right"):
        """Open gripper (left or right)."""
        _, info = self._send_cmd(f"open_gripper:{arm}")
        return info

    def close_gripper(self, arm="right"):
        """Close gripper (left or right)."""
        _, info = self._send_cmd(f"close_gripper:{arm}")
        return info

    def get_ee_pose(self):
        """Get current end-effector poses for both arms."""
        _, info = self._send_cmd("get_ee_pose")
        return info

    def move_arm(self, x, y, z, qw=0.0, qx=0.7071, qy=0.0, qz=0.7071, arm="right"):
        """Move arm end-effector to target pose using IK."""
        _, info = self._send_cmd(f"move_arm:{x},{y},{z},{qw},{qx},{qy},{qz},{arm}")
        return info

    def pick(self, object_prim_path, arm="right"):
        """Pick up an object (approach → close gripper → lift)."""
        _, info = self._send_cmd(f"pick:{object_prim_path},{arm}")
        return info

    def place(self, x, y, z, arm="right"):
        """Place object at target position (lower → open gripper → retract)."""
        _, info = self._send_cmd(f"place:{x},{y},{z},{arm}")
        return info

    def close(self):
        """Close connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

    def __del__(self):
        self.close()


# ─── Example / Test ───
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VLN environment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12347)
    parser.add_argument("--save-images", action="store_true", help="Save images to disk")
    args = parser.parse_args()

    env = VLNEnv(host=args.host, port=args.port)

    # Reset
    print("\n--- Reset ---")
    obs, info = env.reset()
    print(f"State: {info}")
    print(f"Image shape: {obs.shape if obs is not None else None}")
    if args.save_images and obs is not None and Image is not None:
        Image.fromarray(obs).save("/tmp/vln_reset.jpg")
        print("Saved /tmp/vln_reset.jpg")

    # Take some actions
    actions = ["forward", "forward", "turn_left", "forward", "turn_right", "forward"]
    for i, action in enumerate(actions):
        print(f"\n--- Step {i+1}: {action} ---")
        obs, info = env.step(action)
        print(f"State: {info}")
        print(f"Image shape: {obs.shape if obs is not None else None}")
        if args.save_images and obs is not None and Image is not None:
            Image.fromarray(obs).save(f"/tmp/vln_step_{i+1}.jpg")
            print(f"Saved /tmp/vln_step_{i+1}.jpg")

    # Observe
    print("\n--- Observe ---")
    obs, info = env.observe()
    print(f"State: {info}")

    env.close()
    print("\nDone!")
