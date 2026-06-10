#!/usr/bin/env python3
"""
Interactive scene tool - explore prims, query poses, spawn objects, and control robot.

Usage:
    python scripts/scene_tool.py

Commands:
    list                          - list top-level scene prims
    collision                     - list all collision prims
    list_cameras                  - list all camera prims currently in the stage
    material_info <prim_prefix>   - list mesh/subset/material bindings under a prim prefix
    list_subtree <prim_prefix>    - list all prims under a prefix with their USD type
    pose <prim_path>              - query world pose of a prim
    camera_debug <prim_path>      - inspect a camera's world/parent pose and axes
    robot                         - show current robot pose
    ee                            - show end-effector poses (both arms)
    spawn_robot                   - spawn the demo robot and let it free-fall into the scene
    spawn <x> <y> <z> [label]     - spawn cola bottle at (x,y,z)
    spawn_custom <usd> <x> <y> <z> [label] - spawn custom USD object
    look                          - capture and save current camera view
    open_left / open_right        - open left/right gripper
    close_left / close_right      - close left/right gripper
    pick <object_prim> [arm]      - pick up object (default: right arm)
    place <x> <y> <z> [arm]       - place object at position
    help                          - show this help
    quit                          - exit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vln_env import VLNEnv

DEFAULT_BOTTLE_USD = "objects/genie/cola/Aligned.usd"


def main():
    env = VLNEnv()
    print("\n=== Scene Tool ===")
    print("Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            cmd = input("scene> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action == "quit":
            break

        elif action == "help":
            print(__doc__)

        elif action == "list":
            info = env.list_objects()
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                for p in info.get("prims", []):
                    print(f"  {p}")
                print(f"\n  Total: {len(info.get('prims', []))} prims")

        elif action == "collision":
            info = env.list_collision()
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                for c in info.get("colliders", []):
                    print(f"  {c['type']:20s} {c['path']}")
                print(f"\n  Total: {info.get('count', 0)} collision prims")

        elif action == "list_cameras":
            info = env.list_cameras()
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                for p in info.get("cameras", []):
                    print(f"  {p}")
                print(f"\n  Total: {len(info.get('cameras', []))} camera prims")

        elif action == "material_info" and len(parts) >= 2:
            prim_prefix = parts[1]
            info = env.material_info(prim_prefix)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                mats = info.get("materials", [])
                for item in mats:
                    print(f"  {item['mesh']}\n    -> {item['material']}")
                subsets = info.get("subset_bindings", [])
                if subsets:
                    print("\n  Subset bindings:")
                    for item in subsets:
                        print(f"    {item['subset']}\n      -> {item['material']}")
                looks = info.get("looks_prims", [])
                material_prims = info.get("material_prims", [])
                if looks:
                    print("\n  Looks prims:")
                    for p in looks:
                        print(f"    {p}")
                if material_prims:
                    print("\n  Material prims:")
                    for p in material_prims:
                        print(f"    {p}")
                print(f"\n  Total: {len(mats)} mesh bindings | {len(subsets)} subset bindings | {len(looks)} looks prims | {len(material_prims)} material prims")

        elif action == "list_subtree" and len(parts) >= 2:
            prim_prefix = parts[1]
            info = env.list_subtree(prim_prefix)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                for item in info.get("prims", []):
                    print(f"  {item['type']:20s} {item['path']}")
                print(f"\n  Total: {len(info.get('prims', []))} prims")

        elif action == "pose" and len(parts) >= 2:
            prim_path = parts[1]
            info = env.query_pose(prim_path)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                pos = info["position"]
                quat = info["quaternion"]
                print(f"  Position:   ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                print(f"  Quaternion: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")

        elif action == "camera_debug" and len(parts) >= 2:
            prim_path = parts[1]
            info = env.camera_debug(prim_path)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"  Camera:  {info['camera']}")
                print(f"  Parent:  {info['parent']}")
                cp = info['camera_position']
                cq = info['camera_quaternion']
                pp = info['parent_position']
                pq = info['parent_quaternion']
                rq = info['relative_quaternion']
                fw = info['forward_world']
                up = info['up_world']
                rg = info['right_world']
                print(f"  Cam pos: ({cp[0]:.4f}, {cp[1]:.4f}, {cp[2]:.4f})")
                print(f"  Cam q:   ({cq[0]:.4f}, {cq[1]:.4f}, {cq[2]:.4f}, {cq[3]:.4f})")
                print(f"  Par pos: ({pp[0]:.4f}, {pp[1]:.4f}, {pp[2]:.4f})")
                print(f"  Par q:   ({pq[0]:.4f}, {pq[1]:.4f}, {pq[2]:.4f}, {pq[3]:.4f})")
                print(f"  Rel q:   ({rq[0]:.4f}, {rq[1]:.4f}, {rq[2]:.4f}, {rq[3]:.4f})")
                print(f"  Forward: ({fw[0]:.4f}, {fw[1]:.4f}, {fw[2]:.4f})")
                print(f"  Up:      ({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f})")
                print(f"  Right:   ({rg[0]:.4f}, {rg[1]:.4f}, {rg[2]:.4f})")

        elif action == "robot":
            info = env.robot_pose()
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Robot: x={info['x']:.4f}, y={info['y']:.4f}, z={info['z']:.4f}, yaw={info['yaw']:.1f}")

        elif action == "spawn_robot":
            print("Spawning demo robot and triggering free-fall...")
            info = env.spawn_robot()
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Result: {info}")

        elif action == "spawn" and len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            label = parts[4] if len(parts) > 4 else "cola_bottle"
            print(f"Spawning cola at ({x}, {y}, {z})...")
            info = env.spawn_object(DEFAULT_BOTTLE_USD, x, y, z, label)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"  OK: {info.get('prim', '')} at {info.get('position', [])}")

        elif action == "spawn_custom" and len(parts) >= 5:
            usd = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            label = parts[5] if len(parts) > 5 else "custom_obj"
            print(f"Spawning {usd} at ({x}, {y}, {z})...")
            info = env.spawn_object(usd, x, y, z, label)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"  OK: {info.get('prim', '')} at {info.get('position', [])}")

        elif action == "look":
            obs, info = env.observe()
            if obs is not None:
                from PIL import Image
                Image.fromarray(obs).save("scene_view.jpg")
                print(f"  Saved scene_view.jpg (robot at x={info['x']:.2f}, y={info['y']:.2f})")
            else:
                print("  No image captured")

        elif action in ("open_left", "open_right"):
            arm = action.split("_")[1]
            info = env.open_gripper(arm)
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  ✓ Opened {arm} gripper")

        elif action in ("close_left", "close_right"):
            arm = action.split("_")[1]
            info = env.close_gripper(arm)
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  ✓ Closed {arm} gripper")

        elif action == "ee":
            info = env.get_ee_pose()
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                for arm_name in ["right", "left"]:
                    d = info.get(arm_name, {})
                    if "position" in d:
                        p = d["position"]
                        o = d["orientation"]
                        print(f"  {arm_name:5s} EE: pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  quat=({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}, {o[3]:.4f})")
                    elif "note" in d:
                        print(f"  {arm_name:5s} EE: {d['note']}")
                    elif "error" in d:
                        print(f"  {arm_name:5s} EE: Error - {d['error']}")

        elif action == "pick" and len(parts) >= 2:
            obj_prim = parts[1]
            arm = parts[2] if len(parts) > 2 else "right"
            print(f"  Picking {obj_prim} with {arm} arm...")
            info = env.pick(obj_prim, arm)
            if "error" in info:
                print(f"  Error: {info['error']}")
            elif info.get("status") == "ok":
                print(f"  ✓ Pick complete!")
            else:
                print(f"  Result: {info}")

        elif action == "place" and len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            arm = parts[4] if len(parts) > 4 else "right"
            print(f"  Placing at ({x}, {y}, {z}) with {arm} arm...")
            info = env.place(x, y, z, arm)
            if "error" in info:
                print(f"  Error: {info['error']}")
            elif info.get("status") == "ok":
                print(f"  ✓ Place complete!")
            else:
                print(f"  Result: {info}")

        elif action == "test_ik":
            print("  Running IK diagnostic tests...")
            _, info = env._send_cmd("test_ik")
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                for item in info.get("results", []):
                    name = item[0]
                    ok = item[1]
                    extra = f" (joints={item[2]})" if len(item) > 2 else ""
                    status = "✓ OK" if ok else "✗ FAIL"
                    print(f"  {status}  {name}{extra}")

        else:
            print(f"Unknown command: {cmd}. Type 'help'.")

    env.close()
    print("Bye!")


if __name__ == "__main__":
    main()
