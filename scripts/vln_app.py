#!/usr/bin/env python3
# VLN Navigation App for GenieSim
# Usage: SIM_REPO_ROOT=/geniesim/main CUDA_VISIBLE_DEVICES=0 omni_python scripts/vln_app.py --config source/geniesim/config/my_scene_vln.yaml

import os, sys, math, time, threading, socket, struct, json
from pathlib import Path
from io import BytesIO

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "source"))

import geniesim.utils.system_utils as system_utils
from geniesim.config.params import *

system_utils.check_and_fix_env()

ps = ParameterServer()
for f in fields(Config):
    ps.declare_parameter(f.name, None)
ps.set_parameters_from_yaml(system_utils.config_path() + "/config.yaml")
ps.override_from_cli()
cfg = load_dataclass(Config, ps)

from geniesim.app.workflow import AppLauncher
app_launcher = AppLauncher(cfg.app)
simulation_app = app_launcher.app

import numpy as np
from scipy.spatial.transform import Rotation

from isaacsim.core.utils import extensions
extensions.enable_extension("isaacsim.ros2.bridge")

start = time.time()
while True:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        break
    except ModuleNotFoundError:
        if time.time() - start > 10:
            raise RuntimeError("rclpy not available")
        time.sleep(0.1)

rclpy.init()

from isaacsim.core.api import World
from geniesim.app.controllers import APICore
from geniesim.app.workflow.ui_builder import UIBuilder
from geniesim.app.utils.robot import RobotCfg
from geniesim.utils.name_utils import (
    G2_DUAL_ARM_JOINT_NAMES,
    G2_WAIST_JOINT_NAMES,
    G2_HEAD_JOINT_NAMES,
    OMNIPICKER_AJ_NAMES,
)
from geniesim.benchmark.config.robot_init_states import G2_DEFAULT_STATES


# ─── VLN Config ───
FORWARD_STEP = 0.25
TURN_STEP = 15.0


# ─── Global References ───
_api_core = None  # set in main(), used by TCP API handler


# ─── Global State ───
class State:
    x = 0.0
    y = 0.0
    z = 0.0
    yaw = 0.0
    pending_action = None
    initialized = False
    scene_loaded = False
    lock_joint_indices = None
    lock_joint_positions = None
    # Synchronization for VLN API
    action_done = threading.Event()
    capture_requested = threading.Event()
    capture_done = threading.Event()
    last_image = None         # numpy RGBA array from annotator
    rgb_annotator = None      # rep annotator for head camera

state = State()


# ─── UDP Listener (replaces ROS2 for keyboard input) ───
def start_udp_listener():
    """Listen for actions via UDP on port 12345."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 12346))
    sock.settimeout(0.1)
    print("[VLN] UDP listener started on 0.0.0.0:12346")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            action = data.decode().strip().lower()
            state.pending_action = action
            print(f"[VLN] Received: {action}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[VLN] UDP error: {e}")


class VLNRosNode(Node):
    def __init__(self):
        super().__init__("vln_controller")
        self.sub = self.create_subscription(String, "/vln_action", self.cb, 10)
        self.get_logger().info("VLN controller listening on /vln_action")

    def cb(self, msg):
        state.pending_action = msg.data.strip().lower()
        self.get_logger().info(f"Received: {state.pending_action}")


# ─── TCP API Server for VLN model inference ───
VLN_API_PORT = 12347

def _get_state_dict():
    return {
        "x": round(state.x, 4),
        "y": round(state.y, 4),
        "z": round(state.z, 4),
        "yaw": round(state.yaw, 2),
    }


def _capture_and_encode():
    """Request image capture from main loop, wait, encode as JPEG."""
    if state.rgb_annotator is None:
        return None
    state.capture_done.clear()
    state.capture_requested.set()
    if not state.capture_done.wait(timeout=3.0):
        print("[API] Timeout waiting for image capture")
        return None
    img = state.last_image
    if img is None:
        return None
    # Encode as JPEG
    from PIL import Image
    pil_img = Image.fromarray(img[:, :, :3])  # RGBA -> RGB
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _send_response(conn, state_dict, jpeg_bytes):
    """Send response: 4B header_len + header_json + 4B img_len + jpeg_bytes."""
    header = json.dumps(state_dict).encode("utf-8")
    img_data = jpeg_bytes or b""
    conn.sendall(struct.pack(">I", len(header)))
    conn.sendall(header)
    conn.sendall(struct.pack(">I", len(img_data)))
    if img_data:
        conn.sendall(img_data)


def _handle_client(conn, addr):
    """Handle one VLN API client connection."""
    print(f"[API] Client connected: {addr}")
    buf = b""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf-8").strip()
                if not cmd:
                    continue

                if cmd == "observe":
                    jpeg = _capture_and_encode()
                    _send_response(conn, _get_state_dict(), jpeg)

                elif cmd == "robot_pose":
                    _send_response(conn, _get_state_dict(), None)

                elif cmd.startswith("set_robot_pose:"):
                    # set_robot_pose:x,y,z,qw,qx,qy,qz
                    try:
                        parts = cmd.split(":", 1)[1].strip().split(",")
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        qw, qx, qy, qz = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        pos = np.array([x, y, z], dtype=float)
                        quat = np.array([qw, qx, qy, qz], dtype=float)

                        # Teleport robot base through existing render-thread-safe API
                        _api_core.update_robot_base(pos, quat)
                        time.sleep(0.2)

                        # Keep VLN state tracker consistent with the teleported pose
                        state.x, state.y, state.z = x, y, z
                        r = Rotation.from_quat([qx, qy, qz, qw])  # scipy expects xyzw
                        state.yaw = float(r.as_euler('xyz', degrees=True)[2])
                        state.initialized = True
                        state.pending_action = None

                        info = {
                            "status": "ok",
                            "position": [round(x, 4), round(y, 4), round(z, 4)],
                            "quaternion": [round(qw, 4), round(qx, 4), round(qy, 4), round(qz, 4)],
                            "yaw": round(state.yaw, 2),
                        }
                        print(f"[API] Set robot pose to ({x:.3f}, {y:.3f}, {z:.3f}), yaw={state.yaw:.1f}")
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("query_pose:"):
                    prim_path = cmd.split(":", 1)[1].strip()
                    try:
                        pos, quat = _api_core.get_obj_world_pose(prim_path)
                        info = {"prim": prim_path,
                                "position": [round(float(v), 4) for v in pos],
                                "quaternion": [round(float(v), 4) for v in quat]}
                    except Exception as e:
                        info = {"error": str(e), "prim": prim_path}
                    _send_response(conn, info, None)

                elif cmd.startswith("camera_debug:"):
                    # camera_debug:/camera/prim/path
                    prim_path = cmd.split(":", 1)[1].strip()
                    try:
                        from scipy.spatial.transform import Rotation as R
                        stage = _api_core.ui_builder.my_world.stage
                        cam_prim = stage.GetPrimAtPath(prim_path)
                        if not cam_prim.IsValid():
                            raise RuntimeError(f"Camera prim not found: {prim_path}")
                        parent_prim = cam_prim.GetParent()
                        parent_path = str(parent_prim.GetPath()) if parent_prim and parent_prim.IsValid() else None

                        cam_pos, cam_quat = _api_core.get_obj_world_pose(prim_path)
                        if parent_path:
                            parent_pos, parent_quat = _api_core.get_obj_world_pose(parent_path)
                        else:
                            parent_pos, parent_quat = (0,0,0), (1,0,0,0)

                        # convert wxyz -> xyzw for scipy
                        cam_r = R.from_quat([float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3]), float(cam_quat[0])])
                        parent_r = R.from_quat([float(parent_quat[1]), float(parent_quat[2]), float(parent_quat[3]), float(parent_quat[0])])
                        rel_r = parent_r.inv() * cam_r
                        rel_xyzw = rel_r.as_quat()
                        rel_wxyz = [float(rel_xyzw[3]), float(rel_xyzw[0]), float(rel_xyzw[1]), float(rel_xyzw[2])]

                        # USD/Isaac camera convention: forward=-Z, up=+Y, right=+X in local camera frame
                        forward = cam_r.apply([0.0, 0.0, -1.0]).tolist()
                        up = cam_r.apply([0.0, 1.0, 0.0]).tolist()
                        right = cam_r.apply([1.0, 0.0, 0.0]).tolist()

                        info = {
                            "camera": prim_path,
                            "parent": parent_path,
                            "camera_position": [round(float(v), 4) for v in cam_pos],
                            "camera_quaternion": [round(float(v), 4) for v in cam_quat],
                            "parent_position": [round(float(v), 4) for v in parent_pos],
                            "parent_quaternion": [round(float(v), 4) for v in parent_quat],
                            "relative_quaternion": [round(float(v), 4) for v in rel_wxyz],
                            "forward_world": [round(float(v), 4) for v in forward],
                            "up_world": [round(float(v), 4) for v in up],
                            "right_world": [round(float(v), 4) for v in right],
                        }
                    except Exception as e:
                        info = {"error": str(e), "camera": prim_path}
                    _send_response(conn, info, None)

                elif cmd == "list_objects":
                    try:
                        from pxr import Usd, UsdGeom
                        stage = _api_core.ui_builder.my_world.stage
                        prims = []
                        for prim in stage.Traverse():
                            path = str(prim.GetPath())
                            if path.startswith("/World") and prim.IsA(UsdGeom.Xform):
                                depth = path.count("/")
                                if depth <= 5:
                                    prims.append(path)
                        info = {"prims": prims[:200]}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd == "list_cameras":
                    try:
                        from pxr import UsdGeom
                        stage = _api_core.ui_builder.my_world.stage
                        cameras = []
                        for prim in stage.Traverse():
                            if prim.IsA(UsdGeom.Camera):
                                cameras.append(str(prim.GetPath()))
                        info = {"cameras": cameras}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("material_info:"):
                    # material_info:/some/prim/prefix
                    prim_prefix = cmd.split(":", 1)[1].strip()
                    try:
                        from pxr import UsdGeom, UsdShade
                        stage = _api_core.ui_builder.my_world.stage
                        items = []
                        materials = []
                        looks = []
                        subsets = []
                        for prim in stage.Traverse():
                            path = str(prim.GetPath())
                            if path.startswith(prim_prefix) and prim.IsA(UsdGeom.Mesh):
                                binding_api = UsdShade.MaterialBindingAPI(prim)
                                material, _ = binding_api.ComputeBoundMaterial()
                                items.append({
                                    "mesh": path,
                                    "material": str(material.GetPath()) if material and material.GetPath() else None,
                                })
                            if path.startswith(prim_prefix) and prim.IsA(UsdGeom.Subset):
                                binding_api = UsdShade.MaterialBindingAPI(prim)
                                material, _ = binding_api.ComputeBoundMaterial()
                                subsets.append({
                                    "subset": path,
                                    "material": str(material.GetPath()) if material and material.GetPath() else None,
                                })
                            if path.startswith(prim_prefix) and prim.IsA(UsdShade.Material):
                                materials.append(path)
                            if path.startswith(prim_prefix) and "/Looks" in path:
                                looks.append(path)
                        info = {"materials": items, "material_prims": materials, "looks_prims": looks, "subset_bindings": subsets}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("list_subtree:"):
                    # list_subtree:/some/prim/prefix
                    prim_prefix = cmd.split(":", 1)[1].strip()
                    try:
                        from pxr import Usd
                        stage = _api_core.ui_builder.my_world.stage
                        items = []
                        root = stage.GetPrimAtPath(prim_prefix)
                        if not root.IsValid():
                            raise RuntimeError(f"Prim not found: {prim_prefix}")
                        for prim in Usd.PrimRange(root):
                            items.append({"path": str(prim.GetPath()), "type": prim.GetTypeName()})
                        info = {"prims": items}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("spawn_object:"):
                    # "spawn_object:usd_path,x,y,z[,label]"
                    try:
                        parts = cmd.split(":", 1)[1].strip().split(",")
                        usd_path = parts[0].strip()
                        ox, oy, oz = float(parts[1]), float(parts[2]), float(parts[3])
                        label = parts[4].strip() if len(parts) > 4 else "spawned_obj"
                        prim_path = f"/World/Objects/{label}"
                        # Rotate 90° around X-axis to convert Y-up model to Z-up
                        # quaternion (w,x,y,z) for 90° around X = (0.7071, 0.7071, 0, 0)
                        rot = [0.7071068, 0.7071068, 0.0, 0.0]
                        _api_core.add_usd_obj(
                            usd_path=usd_path,
                            prim_path=prim_path,
                            label_name=label,
                            position=[ox, oy, oz],
                            rotation=rot,
                            scale=[1.0, 1.0, 1.0],
                            object_color=None,
                            object_material="general",
                            object_mass=0.3,
                            add_particle=False,
                            particle_position=None,
                            particle_scale=None,
                            particle_color=None,
                            object_com=None,
                            model_type="convexDecomposition",
                            static_friction=0.5,
                            dynamic_friction=0.5,
                        )
                        time.sleep(1.0)
                        info = {"status": "ok", "prim": prim_path,
                                "position": [ox, oy, oz]}
                        print(f"[API] Spawned {label} at ({ox}, {oy}, {oz})")
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd == "list_collision":
                    # List ALL prims with collision API
                    try:
                        from pxr import UsdPhysics
                        stage = _api_core.ui_builder.my_world.stage
                        colliders = []
                        for prim in stage.Traverse():
                            if prim.HasAPI(UsdPhysics.CollisionAPI):
                                colliders.append({"path": str(prim.GetPath()),
                                                  "type": prim.GetTypeName()})
                        info = {"colliders": colliders, "count": len(colliders)}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("open_gripper:"):
                    # open_gripper:left or open_gripper:right
                    arm = cmd.split(":")[1].strip().lower()
                    try:
                        joint_idx = _api_core.get_robot_joint_indices()
                        if arm == "right":
                            joints = ["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]
                        elif arm == "left":
                            joints = ["idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"]
                        else:
                            raise ValueError(f"Invalid arm: {arm}")
                        _api_core.set_joint_positions(
                            [2.0, 2.0],  # opened positions
                            joint_indices=[joint_idx[n] for n in joints],
                            is_trajectory=False,
                        )
                        info = {"status": "ok", "gripper": arm, "action": "open"}
                        print(f"[API] Opened {arm} gripper")
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("close_gripper:"):
                    # close_gripper:left or close_gripper:right
                    arm = cmd.split(":")[1].strip().lower()
                    try:
                        joint_idx = _api_core.get_robot_joint_indices()
                        if arm == "right":
                            joints = ["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]
                        elif arm == "left":
                            joints = ["idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"]
                        else:
                            raise ValueError(f"Invalid arm: {arm}")
                        _api_core.set_joint_positions(
                            [0.0, 0.0],  # closed positions
                            joint_indices=[joint_idx[n] for n in joints],
                            is_trajectory=False,
                        )
                        info = {"status": "ok", "gripper": arm, "action": "close"}
                        print(f"[API] Closed {arm} gripper")
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd == "get_ee_pose":
                    # Get end-effector poses by querying prim world pose directly
                    # Also ensure IK solver is initialized for later use
                    try:
                        ub = _api_core.ui_builder
                        # Init IK solver if needed (for pick/move_arm commands later)
                        ks = getattr(ub, 'kinematics_solver', 'NOT_SET')
                        if ks is None or ks == 'NOT_SET':
                            print(f"[API] Initializing IK solver...")
                            from geniesim.app.controllers.kinematics_solver import Kinematics_Solver
                            arm_type = getattr(ub, 'arm_type', 'right')
                            ee_name = "gripper_r_center_link" if arm_type == "right" else "gripper_l_center_link"
                            try:
                                ub.kinematics_solver = Kinematics_Solver(
                                    robot_description_path="/G2_omnipicker/config_right_arm.yaml",
                                    urdf_path="/G2_omnipicker/G2_omnipicker.urdf",
                                    end_effector_name=ee_name,
                                    articulation=ub.articulation,
                                )
                                print(f"[API] IK solver initialized! arm_type={arm_type}")
                            except Exception as e2:
                                print(f"[API] IK solver init failed: {e2}")

                        # Get EE poses via prim world pose (reliable)
                        info = {}
                        for arm, prim_path in [("right", "/genie/gripper_r_center_link"),
                                               ("left", "/genie/gripper_l_center_link")]:
                            pos, quat = _api_core.get_obj_world_pose(prim_path)
                            info[arm] = {
                                "position": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
                                "orientation": [round(float(quat[0]), 4), round(float(quat[1]), 4), round(float(quat[2]), 4), round(float(quat[3]), 4)],
                            }
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("move_arm:"):
                    # move_arm:x,y,z,qw,qx,qy,qz,arm
                    # Move arm end-effector to target pose using IK
                    try:
                        parts = cmd.split(":")[1].strip().split(",")
                        tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
                        qw, qx, qy, qz = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        arm = parts[7].strip() if len(parts) > 7 else "right"
                        is_right = (arm == "right")

                        target_pos = np.array([tx, ty, tz])
                        target_rot = np.array([qw, qx, qy, qz])

                        ub = _api_core.ui_builder
                        # Sync articulation pose with actual prim pose
                        actual_pos, actual_quat = _api_core.get_obj_world_pose("/genie")
                        ub.articulation.set_world_pose(
                            position=np.array([float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])]),
                            orientation=np.array([float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3])])
                        )

                        success, actions = ub._get_ik_status(target_pos, target_rot, is_right)
                        if success:
                            # Apply joint positions
                            joint_positions = actions.joint_positions
                            joint_indices = [i for i in range(len(joint_positions)) if joint_positions[i] is not None]
                            positions = [joint_positions[i] for i in joint_indices]
                            _api_core.set_joint_positions(
                                positions,
                                joint_indices=joint_indices,
                                is_trajectory=False,
                            )
                            info = {"status": "ok", "arm": arm, "target": [tx, ty, tz]}
                            print(f"[API] Moved {arm} arm to ({tx:.3f}, {ty:.3f}, {tz:.3f})")
                        else:
                            info = {"status": "ik_failed", "arm": arm, "target": [tx, ty, tz]}
                            print(f"[API] IK failed for {arm} arm at ({tx:.3f}, {ty:.3f}, {tz:.3f})")
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("pick:"):
                    # pick:object_prim_path,arm
                    # Full pick sequence: approach -> close gripper -> lift
                    try:
                        parts = cmd.split(":")[1].strip().split(",")
                        obj_prim = parts[0].strip()
                        arm = parts[1].strip() if len(parts) > 1 else "right"
                        is_right = (arm == "right")

                        # Ensure IK solver is initialized
                        ub = _api_core.ui_builder
                        if getattr(ub, 'kinematics_solver', None) is None:
                            from geniesim.app.controllers.kinematics_solver import Kinematics_Solver
                            arm_type = getattr(ub, 'arm_type', 'right')
                            ee_name = "gripper_r_center_link" if arm_type == "right" else "gripper_l_center_link"
                            ub.kinematics_solver = Kinematics_Solver(
                                robot_description_path="/G2_omnipicker/config.yaml",
                                urdf_path="/G2_omnipicker/G2_omnipicker.urdf",
                                end_effector_name=ee_name,
                                articulation=ub.articulation,
                            )
                            print(f"[API] IK solver initialized for pick")

                        # Update IK solver with current robot base pose (from actual prim, not articulation cache)
                        actual_pos, actual_quat = _api_core.get_obj_world_pose("/genie")
                        robot_base_pos = np.array([float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])])
                        robot_base_quat = np.array([float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3])])
                        print(f"[API] Robot base pose: pos={robot_base_pos}, quat={robot_base_quat}")

                        # Set robot base pose directly on kinematics solver
                        ks = ub.kinematics_solver
                        ks._kinematics_solver.set_robot_base_pose(robot_base_pos, robot_base_quat)

                        # Helper: solve IK on render loop using Lula solver directly
                        _ik_result = {"success": False, "actions": None, "done": threading.Event()}

                        def _solve_ik(target_pos, target_rot):
                            _ik_result["done"].clear()
                            _ik_result["success"] = False
                            _ik_result["actions"] = None

                            def _do_ik():
                                try:
                                    # Get actual robot prim pose
                                    actual_pos, actual_quat = _api_core.get_obj_world_pose("/genie")
                                    base_pos = np.array([float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])])
                                    base_quat = np.array([float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3])])

                                    # Set base pose directly on Lula solver (don't touch articulation - causes deadlock)
                                    ks = ub.kinematics_solver
                                    ks._kinematics_solver.set_robot_base_pose(base_pos, base_quat)
                                    ret_a, ret_b = ks._articulation_kinematics_solver.compute_inverse_kinematics(
                                        target_pos, target_rot
                                    )
                                    if isinstance(ret_a, bool):
                                        _ik_result["success"] = ret_a
                                        _ik_result["actions"] = ret_b
                                    else:
                                        _ik_result["success"] = ret_b
                                        _ik_result["actions"] = ret_a

                                    if _ik_result["success"]:
                                        print(f"[API] IK success!")
                                except Exception as e:
                                    print(f"[API] IK error: {e}")
                                    import traceback; traceback.print_exc()
                                    _ik_result["success"] = False
                                finally:
                                    _ik_result["done"].set()

                            _api_core.run_on_render_loop(_do_ik)
                            _ik_result["done"].wait(timeout=10.0)
                            return _ik_result["success"], _ik_result["actions"]

                        # Get object world pose
                        obj_pos, obj_quat = _api_core.get_obj_world_pose(obj_prim)
                        obj_x, obj_y, obj_z = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
                        print(f"[API] Pick target: {obj_prim} at ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})")

                        joint_idx = _api_core.get_robot_joint_indices()

                        # Compute distance from end-effector to object (not robot base)
                        ee_prim = "/genie/gripper_r_center_link" if arm == "right" else "/genie/gripper_l_center_link"
                        ee_pos, _ = _api_core.get_obj_world_pose(ee_prim)
                        ee_x, ee_y, ee_z = float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
                        dist = np.sqrt((obj_x - ee_x)**2 + (obj_y - ee_y)**2 + (obj_z - ee_z)**2)
                        dist_xy = np.sqrt((obj_x - ee_x)**2 + (obj_y - ee_y)**2)
                        print(f"[API] EE position: ({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f})")
                        print(f"[API] Distance EE -> cola: {dist:.3f}m (XY: {dist_xy:.3f}m)")

                        # If object is low or far, lean the torso forward to extend reach
                        if dist > 0.5 or obj_z < 0.0:
                            print("[API] Leaning torso forward to extend reach...")
                            waist_joints = ["idx05_body_joint5", "idx04_body_joint4", "idx03_body_joint3", "idx02_body_joint2", "idx01_body_joint1"]
                            waist_indices = [joint_idx[n] for n in waist_joints]
                            # Lean forward: increase hip bend
                            _api_core.set_joint_positions(
                                [0.0, 0.0, 0.0, -0.8, 0.8],  # more forward lean than default [-0.35, 0.35]
                                joint_indices=waist_indices,
                                is_trajectory=False,
                            )
                            time.sleep(1.0)
                            # Update robot base pose for IK after leaning
                            actual_pos, actual_quat = _api_core.get_obj_world_pose("/genie")
                            robot_base_pos = np.array([float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2])])
                            robot_base_quat = np.array([float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3])])
                            ks._kinematics_solver.set_robot_base_pose(robot_base_pos, robot_base_quat)

                        # Gripper joints
                        if arm == "right":
                            gripper_joints = ["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]
                        else:
                            gripper_joints = ["idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"]
                        gripper_indices = [joint_idx[n] for n in gripper_joints]

                        # Try multiple grasp orientations, including current EE orientation
                        # Get current EE orientation as best starting point
                        ee_pos_curr, ee_quat_curr = _api_core.get_obj_world_pose(ee_prim)
                        curr_quat = np.array([float(ee_quat_curr[0]), float(ee_quat_curr[1]),
                                              float(ee_quat_curr[2]), float(ee_quat_curr[3])])
                        print(f"[API] Current EE orientation: {curr_quat}")

                        grasp_orientations = [
                            curr_quat,                                   # current EE orientation
                            np.array([0.0, 0.7071, 0.0, 0.7071]),       # gripper facing -X down
                            np.array([0.5, 0.5, -0.5, 0.5]),            # angled approach
                            np.array([0.5, 0.5, 0.5, 0.5]),             # angled approach 2
                            np.array([0.0, 1.0, 0.0, 0.0]),             # gripper facing -Z
                            np.array([0.7071, 0.0, 0.7071, 0.0]),       # gripper facing -Y
                            np.array([0.0, 0.0, 0.7071, 0.7071]),       # another orientation
                            np.array([0.7071, 0.7071, 0.0, 0.0]),       # 90 deg around X
                            np.array([0.0, 0.0, 1.0, 0.0]),             # 180 deg around Y
                            np.array([0.5, -0.5, 0.5, 0.5]),            # angled 3
                            np.array([0.5, -0.5, -0.5, 0.5]),           # angled 4
                            np.array([0.3827, 0.9239, 0.0, 0.0]),       # 45 deg tilt
                        ]

                        # Step 1: Open gripper
                        print("[API] Pick step 1: Open gripper")
                        _api_core.set_joint_positions([2.0, 2.0], joint_indices=gripper_indices, is_trajectory=False)
                        time.sleep(0.5)

                        # Step 2: Move to pre-grasp (above object)
                        pre_grasp_pos = np.array([obj_x, obj_y, obj_z + 0.15])
                        print(f"[API] Pick step 2: Pre-grasp at ({obj_x:.3f}, {obj_y:.3f}, {obj_z + 0.15:.3f})")
                        grasp_rot = None
                        for i, rot in enumerate(grasp_orientations):
                            success, actions = _solve_ik(pre_grasp_pos, rot)
                            print(f"[API]   Orientation {i}: success={success}")
                            if success:
                                grasp_rot = rot
                                break
                        if grasp_rot is None:
                            raise RuntimeError(f"IK failed for pre-grasp. EE->obj dist={dist:.2f}m (XY:{dist_xy:.2f}m). Move robot closer!")
                        # Apply IK result - handle both ArticulationAction and numpy array
                        def _apply_ik(act):
                            if hasattr(act, 'joint_positions'):
                                jp = act.joint_positions
                                ji = [i for i in range(len(jp)) if jp[i] is not None]
                                _api_core.set_joint_positions([jp[i] for i in ji], joint_indices=ji, is_trajectory=False)
                            else:
                                arm_joint_names = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
                                arm_indices = [joint_idx[n] for n in arm_joint_names]
                                _api_core.set_joint_positions(list(act), joint_indices=arm_indices, is_trajectory=False)
                        _apply_ik(actions)
                        time.sleep(1.0)

                        # Step 3: Move to grasp position (at object)
                        grasp_pos = np.array([obj_x, obj_y, obj_z + 0.02])
                        print(f"[API] Pick step 3: Grasp at z={obj_z + 0.02:.3f}")
                        success, actions = _solve_ik(grasp_pos, grasp_rot)
                        if success:
                            _apply_ik(actions)
                        time.sleep(1.0)

                        # Step 4: Close gripper
                        print("[API] Pick step 4: Close gripper")
                        _api_core.set_joint_positions([0.0, 0.0], joint_indices=gripper_indices, is_trajectory=False)
                        time.sleep(1.0)

                        # Step 5: Lift object
                        lift_pos = np.array([obj_x, obj_y, obj_z + 0.25])
                        print(f"[API] Pick step 5: Lift to z={obj_z + 0.25:.3f}")
                        success, actions = _solve_ik(lift_pos, grasp_rot)
                        if success:
                            _apply_ik(actions)
                        time.sleep(0.5)

                        info = {"status": "ok", "action": "pick", "object": obj_prim,
                                "position": [obj_x, obj_y, obj_z]}
                        print(f"[API] Pick complete for {obj_prim}")

                    except Exception as e:
                        import traceback; traceback.print_exc()
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd.startswith("place:"):
                    # place:x,y,z,arm
                    # Full place sequence: move to position -> open gripper -> retract
                    try:
                        parts = cmd.split(":")[1].strip().split(",")
                        tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
                        arm = parts[3].strip() if len(parts) > 3 else "right"
                        is_right = (arm == "right")

                        ub = _api_core.ui_builder
                        joint_idx = _api_core.get_robot_joint_indices()

                        if arm == "right":
                            gripper_joints = ["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]
                        else:
                            gripper_joints = ["idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"]
                        gripper_indices = [joint_idx[n] for n in gripper_joints]

                        place_rot = np.array([0.0, 0.7071, 0.0, 0.7071])

                        # Step 1: Move above target
                        pre_place_pos = np.array([tx, ty, tz + 0.15])
                        print(f"[API] Place step 1: Above target at z={tz + 0.15:.3f}")
                        success, actions = ub._get_ik_status(pre_place_pos, place_rot, is_right)
                        if success:
                            jp = actions.joint_positions
                            ji = [i for i in range(len(jp)) if jp[i] is not None]
                            _api_core.set_joint_positions([jp[i] for i in ji], joint_indices=ji, is_trajectory=False)
                        time.sleep(1.0)

                        # Step 2: Lower to target
                        place_pos = np.array([tx, ty, tz + 0.02])
                        print(f"[API] Place step 2: Lower to z={tz + 0.02:.3f}")
                        success, actions = ub._get_ik_status(place_pos, place_rot, is_right)
                        if success:
                            jp = actions.joint_positions
                            ji = [i for i in range(len(jp)) if jp[i] is not None]
                            _api_core.set_joint_positions([jp[i] for i in ji], joint_indices=ji, is_trajectory=False)
                        time.sleep(0.5)

                        # Step 3: Open gripper
                        print("[API] Place step 3: Open gripper")
                        _api_core.set_joint_positions([2.0, 2.0], joint_indices=gripper_indices, is_trajectory=False)
                        time.sleep(0.5)

                        # Step 4: Retract upward
                        retract_pos = np.array([tx, ty, tz + 0.25])
                        print(f"[API] Place step 4: Retract to z={tz + 0.25:.3f}")
                        success, actions = ub._get_ik_status(retract_pos, place_rot, is_right)
                        if success:
                            jp = actions.joint_positions
                            ji = [i for i in range(len(jp)) if jp[i] is not None]
                            _api_core.set_joint_positions([jp[i] for i in ji], joint_indices=ji, is_trajectory=False)
                        time.sleep(0.5)

                        info = {"status": "ok", "action": "place", "position": [tx, ty, tz]}
                        print(f"[API] Place complete at ({tx:.3f}, {ty:.3f}, {tz:.3f})")

                    except Exception as e:
                        import traceback; traceback.print_exc()
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd == "test_ik":
                    # Diagnostic: test if IK solver works at all
                    # Try to solve IK for a position near the current EE
                    try:
                        ub = _api_core.ui_builder
                        ks = getattr(ub, 'kinematics_solver', None)

                        # Init IK if needed
                        if ks is None:
                            from geniesim.app.controllers.kinematics_solver import Kinematics_Solver
                            arm_type = getattr(ub, 'arm_type', 'right')
                            ee_name = "gripper_r_center_link" if arm_type == "right" else "gripper_l_center_link"
                            ub.kinematics_solver = Kinematics_Solver(
                                robot_description_path="/G2_omnipicker/config.yaml",
                                urdf_path="/G2_omnipicker/G2_omnipicker.urdf",
                                end_effector_name=ee_name,
                                articulation=ub.articulation,
                            )
                            ks = ub.kinematics_solver

                        _test = {"done": threading.Event(), "results": []}

                        def _run_test():
                            try:
                                # Print articulation state
                                art_pos, art_quat = ub.articulation.get_world_pose()
                                print(f"[TEST] Articulation base: pos={art_pos}, quat={art_quat}")

                                # Print actual prim pos
                                actual_pos, actual_quat = _api_core.get_obj_world_pose("/genie")
                                print(f"[TEST] Actual prim base: pos={actual_pos}, quat={actual_quat}")

                                # Get current EE pose via FK
                                ee_pos, ee_rot = ub._get_ee_pose(is_right=True)
                                print(f"[TEST] Current EE (FK): pos={ee_pos}, rot={ee_rot}")

                                # Get EE from prim
                                ee_prim_pos, ee_prim_quat = _api_core.get_obj_world_pose("/genie/gripper_r_center_link")
                                print(f"[TEST] Current EE (prim): pos={ee_prim_pos}, quat={ee_prim_quat}")

                                # Test 1: IK to current EE position (should always succeed)
                                print(f"\n[TEST] === Test 1: IK to current EE pos ===")
                                ret_a, ret_b = ks._articulation_kinematics_solver.compute_inverse_kinematics(
                                    np.array(ee_pos).flatten(), np.array(ee_rot).flatten()
                                )
                                ik_ok = ret_a if isinstance(ret_a, bool) else ret_b
                                ik_act = ret_b if isinstance(ret_a, bool) else ret_a
                                jp = ik_act.joint_positions if hasattr(ik_act, 'joint_positions') else None
                                n_joints = len([v for v in jp if v is not None]) if jp is not None else 0
                                print(f"[TEST] Result: ok={ik_ok}, n_joints={n_joints}")
                                if jp is not None:
                                    print(f"[TEST] Joint positions: {[round(float(v),4) if v is not None else None for v in jp]}")
                                _test["results"].append(("current_ee", bool(ik_ok), n_joints))

                                # Test 2: IK to slightly offset position
                                print(f"\n[TEST] === Test 2: IK to EE + 5cm forward ===")
                                offset_pos = np.array(ee_pos).flatten() + np.array([0.05, 0, 0])
                                r_a, r_b = ks._articulation_kinematics_solver.compute_inverse_kinematics(
                                    offset_pos, np.array(ee_rot).flatten()
                                )
                                ok2 = r_a if isinstance(r_a, bool) else r_b
                                print(f"[TEST] Result: ok={ok2}")
                                _test["results"].append(("ee+5cm", bool(ok2)))

                                # Test 3: IK to 20cm below EE
                                print(f"\n[TEST] === Test 3: IK to EE - 20cm Z ===")
                                down_pos = np.array(ee_pos).flatten() + np.array([0, 0, -0.2])
                                r_a, r_b = ks._articulation_kinematics_solver.compute_inverse_kinematics(
                                    down_pos, np.array(ee_rot).flatten()
                                )
                                ok3 = r_a if isinstance(r_a, bool) else r_b
                                print(f"[TEST] Result: ok={ok3}")
                                _test["results"].append(("ee-20cm_z", bool(ok3)))

                                # Test 4: IK with different orientations
                                print(f"\n[TEST] === Test 4: Different orientations at EE pos ===")
                                test_rots = [
                                    ("current", np.array(ee_rot).flatten()),
                                    ("down", np.array([0.0, 0.7071, 0.0, 0.7071])),
                                    ("ident", np.array([1.0, 0.0, 0.0, 0.0])),
                                ]
                                for name, rot in test_rots:
                                    r_a, r_b = ks._articulation_kinematics_solver.compute_inverse_kinematics(
                                        np.array(ee_pos).flatten(), rot
                                    )
                                    s = r_a if isinstance(r_a, bool) else r_b
                                    print(f"[TEST]   {name}: ok={s}")
                                    _test["results"].append((f"rot_{name}", bool(s)))

                            except Exception as e:
                                print(f"[TEST] Error: {e}")
                                import traceback; traceback.print_exc()
                            finally:
                                _test["done"].set()

                        _api_core.run_on_render_loop(_run_test)
                        _test["done"].wait(timeout=10.0)

                        info = {"results": _test["results"]}
                    except Exception as e:
                        info = {"error": str(e)}
                    _send_response(conn, info, None)

                elif cmd == "reset":
                    # Re-read robot's current actual position (don't teleport)
                    state.initialized = False  # force re-read from physics
                    time.sleep(0.5)  # wait for physics to update
                    jpeg = _capture_and_encode()
                    _send_response(conn, _get_state_dict(), jpeg)

                elif cmd.startswith("step:"):
                    action = cmd[5:].strip().lower()
                    state.action_done.clear()
                    state.pending_action = action
                    # Wait for action to be processed
                    if not state.action_done.wait(timeout=2.0):
                        print(f"[API] Timeout waiting for action: {action}")
                    # Wait one more render cycle for updated image
                    time.sleep(0.15)
                    jpeg = _capture_and_encode()
                    info = _get_state_dict()
                    info["action"] = action
                    _send_response(conn, info, jpeg)

                else:
                    print(f"[API] Unknown command: {cmd}")
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()
        print(f"[API] Client disconnected: {addr}")


def start_api_server():
    """TCP server for VLN model inference."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", VLN_API_PORT))
    srv.listen(1)
    print(f"[API] VLN API server listening on 0.0.0.0:{VLN_API_PORT}")
    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=_handle_client, args=(conn, addr), daemon=True)
        t.start()


class VLNRosNode(Node):
    def __init__(self):
        super().__init__("vln_controller")
        self.sub = self.create_subscription(String, "/vln_action", self.cb, 10)
        self.get_logger().info("VLN controller listening on /vln_action")

    def cb(self, msg):
        state.pending_action = msg.data.strip().lower()
        self.get_logger().info(f"Received: {state.pending_action}")


# ─── Collision detection using Isaac Sim native PhysX scene queries ───
# We use raycast_closest at multiple heights and angles to detect obstacles.
#
# KEY DESIGN: Rays originate from the robot's OUTER SHELL, not the center.
# The robot is approximated as a cylinder with radius ROBOT_OUTER_RADIUS.
# Rays start at the edge of this cylinder and extend outward, so the
# reported hit distance = actual clearance from robot body to obstacle.
#
# G2 robot actual dimensions (computed via FK with VLN joint angles):
#   Body waist joints: [0.0, 0.0, 0.0, -0.35, 0.35]
#   Arm joints: G2_DEFAULT_STATES["init_arm"]
#   Bounding box: X=[-0.15, 0.64], Y=[-0.47, 0.47], Z=[0.14, 0.95]
#   Max XY from center: 0.79m (at arm_l/r_link6/7 = elbow/wrist tips)
#   Body only (no arms): ~0.25m
#
# The arms extend to ~0.79m but are mostly in the upper half (z > 0.7m).
# We use different radii per height layer to avoid over-blocking:
#   - Low  (0.30m): body only = 0.25m
#   - Mid  (0.50m): body + upper arm root = 0.35m
#   - High (0.70m): body + full arm reach = 0.55m (arms at ~0.63-0.79m from center)

ROBOT_PRIM_PREFIX = None    # set from robot_cfg at startup, used to filter self-collisions
MIN_CLEARANCE = 0.05            # stop when obstacle is within 5cm of robot shell

# (height_above_feet, shell_radius_at_this_height)
ROBOT_RAY_LAYERS = [
    (0.30, 0.25),   # legs/base only
    (0.50, 0.35),   # torso + upper arm roots
    (0.70, 0.55),   # full arm reach
]


def _is_robot_prim(prim_path: str) -> bool:
    """Return True if the prim belongs to the robot itself (should be ignored)."""
    return prim_path.startswith(ROBOT_PRIM_PREFIX)


def multi_ray_check(position, direction, move_distance):
    """
    Cast multiple rays from the robot's OUTER SHELL in the move direction.
    Uses per-height-layer shell radius matching actual robot geometry.

    Returns (is_clear, min_clearance, hit_info).
    """
    from omni.physx import get_physx_scene_query_interface
    physx_query = get_physx_scene_query_interface()

    dx, dy = float(direction[0]), float(direction[1])
    ray_length = float(move_distance) + MIN_CLEARANCE

    # Perpendicular direction for side rays
    perp_x, perp_y = -dy, dx

    min_clearance = ray_length
    hit_found = False
    hit_info = "clear"

    for (h_above_feet, shell_radius) in ROBOT_RAY_LAYERS:
        ray_z = float(position[2]) + h_above_feet

        # Rays from the outer shell at this height:
        #   - Front-center
        #   - Front-left 45deg
        #   - Front-right 45deg
        shell_offsets = [
            (shell_radius, 0.0),                                     # front center
            (shell_radius * 0.707, shell_radius * 0.707),            # front-left 45deg
            (shell_radius * 0.707, -shell_radius * 0.707),           # front-right 45deg
        ]

        for (fwd, lat) in shell_offsets:
            ox = float(position[0]) + dx * fwd + perp_x * lat
            oy = float(position[1]) + dy * fwd + perp_y * lat
            origin = (ox, oy, ray_z)
            dir_carb = (dx, dy, 0.0)

            result = physx_query.raycast_closest(origin, dir_carb, ray_length)

            if result is None:
                continue

            try:
                hit = bool(result["hit"])
            except (KeyError, TypeError):
                continue

            if not hit:
                continue

            try:
                hit_dist = float(result["distance"])
                hit_path = str(result.get("rigidBody", "unknown"))
            except (KeyError, TypeError):
                hit_dist = 0.0
                hit_path = "unknown"

            if _is_robot_prim(hit_path):
                continue

            hit_found = True
            if hit_dist < min_clearance:
                min_clearance = hit_dist
                hit_info = f"z={ray_z:.2f}m r={shell_radius:.2f}m prim={hit_path} clearance={hit_dist:.2f}m"
            print(f"[VLN] Ray hit: z={ray_z:.2f}m r={shell_radius:.2f}m -> {hit_path} at {hit_dist:.2f}m")

    return (not hit_found), min_clearance, hit_info


def collision_check(current_pos, direction, distance):
    """
    Check if the robot can move 'distance' in 'direction' without collision.
    Returns (is_clear, info_string).
    """
    is_clear, clearance, info = multi_ray_check(current_pos, direction, distance)
    if not is_clear and clearance < distance + MIN_CLEARANCE:
        return False, f"blocked: {info}"
    return True, "clear"


def apply_action(api_core):
    if not state.scene_loaded or "robot" not in api_core.usd_objects:
        return

    if not state.initialized:
        try:
            pos, quat = api_core.usd_objects["robot"].get_world_pose()
            state.x, state.y, state.z = float(pos[0]), float(pos[1]), float(pos[2])
            r = Rotation.from_quat([float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
            state.yaw = r.as_euler('xyz', degrees=True)[2]
            state.initialized = True
            print(f"[VLN] Init pose: x={state.x:.2f}, y={state.y:.2f}, z={state.z:.2f}, yaw={state.yaw:.1f}")
        except Exception as e:
            print(f"[VLN] Init failed: {e}")
            return

    if state.pending_action is None:
        return

    action = state.pending_action
    state.pending_action = None

    new_x, new_y = state.x, state.y

    if action == "forward":
        yaw_rad = math.radians(state.yaw)
        dx = math.cos(yaw_rad)
        dy = math.sin(yaw_rad)
        current_pos = np.array([state.x, state.y, state.z])
        direction = np.array([dx, dy, 0.0])
        clear, info = collision_check(current_pos, direction, FORWARD_STEP)
        if clear:
            new_x = state.x + FORWARD_STEP * dx
            new_y = state.y + FORWARD_STEP * dy
        else:
            print(f"[VLN] Blocked (forward): {info}")
            return
    elif action == "backward":
        yaw_rad = math.radians(state.yaw)
        dx = -math.cos(yaw_rad)
        dy = -math.sin(yaw_rad)
        current_pos = np.array([state.x, state.y, state.z])
        direction = np.array([dx, dy, 0.0])
        clear, info = collision_check(current_pos, direction, FORWARD_STEP)
        if clear:
            new_x = state.x + FORWARD_STEP * dx
            new_y = state.y + FORWARD_STEP * dy
        else:
            print(f"[VLN] Blocked (backward): {info}")
            return
    elif action == "turn_left":
        state.yaw += TURN_STEP
    elif action == "turn_right":
        state.yaw -= TURN_STEP
    elif action == "stop":
        print("[VLN] Stop")
        return
    else:
        return

    # Apply position update
    state.x, state.y = new_x, new_y

    r = Rotation.from_euler('z', state.yaw, degrees=True)
    q = r.as_quat()
    pos = np.array([state.x, state.y, state.z])
    quat = np.array([q[3], q[0], q[1], q[2]])

    try:
        api_core.usd_objects["robot"].set_world_pose(pos, quat)
        print(f"[VLN] {action}: x={state.x:.2f}, y={state.y:.2f}, yaw={state.yaw:.1f}")
    except Exception as e:
        print(f"[VLN] Move error: {e}")
    finally:
        state.action_done.set()


def main():
    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / cfg.app.physics_step,
        rendering_dt=1.0 / cfg.app.rendering_step,
    )
    ui_builder = UIBuilder(world=world)
    api_core = APICore(ui_builder=ui_builder, config=cfg)
    global _api_core
    _api_core = api_core

    # Load task config
    task_config = system_utils.load_json(
        os.path.join(
            system_utils.benchmark_conf_path(),
            "eval_tasks",
            cfg.benchmark.task_name + ".json",
        )
    )

    # Set the robot prim prefix dynamically from the selected G2 robot config so
    # self-collision filtering follows the configured robot prim path.
    global ROBOT_PRIM_PREFIX
    try:
        robot_cfg_path = os.path.join(str(system_utils.app_root_path()), "robot_cfg", task_config["robot"]["robot_cfg"])
        robot_cfg_obj = RobotCfg(robot_cfg_path)
        ROBOT_PRIM_PREFIX = robot_cfg_obj.robot_prim_path
        print(f"[VLN] ROBOT_PRIM_PREFIX set to {ROBOT_PRIM_PREFIX}")
    except Exception as e:
        print(f"[VLN] WARNING: failed to resolve robot prim prefix, fallback={ROBOT_PRIM_PREFIX}, err={e}")

    sub_usd_path = ""
    if cfg.benchmark.sub_task_name:
        sub_usd_path = os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task",
            cfg.benchmark.sub_task_name,
            "0",
            "scene.usda",
        )

    robot_init_pose = task_config["robot"]["robot_init_pose"]
    first_workspace = list(robot_init_pose.values())[0]
    init_position = first_workspace["position"]
    init_rotation = first_workspace["quaternion"]

    # Create VLN node in main thread
    vln_node = VLNRosNode()

    # Start UDP listener thread for keyboard input
    udp_thread = threading.Thread(target=start_udp_listener, daemon=True)
    udp_thread.start()

    # Start TCP API server for VLN model
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    # Load scene in background thread
    def load_scene():
        try:
            print("[VLN] Loading scene and robot...")
            api_core.init_robot_cfg(
                task_config["robot"]["robot_cfg"],
                task_config["scene"]["scene_usd"],
                init_position,
                init_rotation,
                sub_usd_path,
            )
            print("[VLN] Scene and robot loaded successfully!")

            def _set_viewport_camera_light():
                """Prefer viewport camera/head light for GUI inspection.

                This changes viewport display settings only; it does not add lights to the USD stage
                and does not affect the robot/camera prims used by the VLN API.
                """
                try:
                    import omni.kit.commands
                    import carb.settings

                    # This is the same mode used by the viewport Lighting menu.
                    # Internally it also sets /rtx/useViewLightingMode=True.
                    omni.kit.commands.execute(
                        "SetLightingMenuModeCommand",
                        lighting_mode="camera",
                        usd_context_name="",
                    )

                    # Keep the renderer-side setting explicit as a fallback for UI timing.
                    carb.settings.get_settings().set("/rtx/useViewLightingMode", True)
                    print("[VLN] Set viewport Lighting mode to Camera Light")
                except Exception as e:
                    print(f"[VLN] WARNING: failed to set viewport Camera Light mode: {e}")

            api_core.run_on_render_loop(_set_viewport_camera_light)

            # Load collision mesh (invisible, physics only) - MUST load BEFORE setting scene_loaded
            collision_usd = str(system_utils.assets_path()) + "/background/my_scene/sample.usda"
            if os.path.exists(collision_usd):
                def _load_collision():
                    from isaacsim.core.utils.stage import add_reference_to_stage
                    from pxr import UsdGeom

                    add_reference_to_stage(collision_usd, "/World/CollisionMesh")

                    # Keep the collision mesh available for PhysX but hidden in the viewport by default.
                    stage = api_core.ui_builder.my_world.stage
                    collision_prim = stage.GetPrimAtPath("/World/CollisionMesh")
                    if collision_prim and collision_prim.IsValid():
                        UsdGeom.Imageable(collision_prim).MakeInvisible()

                    print("[VLN] Collision mesh loaded and hidden by default!")
                api_core.run_on_render_loop(_load_collision)

                # Wait a few physics steps for PhysX to register the new colliders
                print("[VLN] Waiting for PhysX to register collision mesh...")
                time.sleep(0.5)
            else:
                print(f"[VLN] WARNING: Collision mesh not found: {collision_usd}")
                print(f"[VLN] Robot will have NO wall collision!")

            state.scene_loaded = True
            print("[VLN] Collision ready, navigation enabled.")

            robot_cfg_name = task_config["robot"]["robot_cfg"]
            if robot_cfg_name != "G2_omnipicker.json":
                raise RuntimeError(f"VLN app is configured for G2_omnipicker.json only, got {robot_cfg_name}")

            # Set G2 initial joint positions (arms in front, ready pose)
            print("[VLN] Setting initial joint positions...")
            joint_idx = api_core.get_robot_joint_indices()

            # Body (waist) joints - slight knee bend to lower height to ~1650mm
            # Order: [idx05, idx04, idx03, idx02, idx01]
            # joint1=0.35 bends hip forward, joint2=-0.35 compensates to keep torso upright
            api_core.set_joint_positions(
                [0.0, 0.0, 0.0, -0.35, 0.35],
                joint_indices=[joint_idx[n] for n in G2_WAIST_JOINT_NAMES],
                is_trajectory=False,
            )
            # Head joints - slightly upward to compensate body lean
            # [yaw, roll, pitch]: negative pitch = look up
            api_core.set_joint_positions(
                [0.0, 0.0, -0.15],
                joint_indices=[joint_idx[n] for n in G2_HEAD_JOINT_NAMES],
                is_trajectory=False,
            )
            # Arm joints (left + right, 14 DOF)
            api_core.set_joint_positions(
                G2_DEFAULT_STATES["init_arm"],
                joint_indices=[joint_idx[n] for n in G2_DUAL_ARM_JOINT_NAMES],
                is_trajectory=False,
            )
            # Gripper joints
            api_core.set_joint_positions(
                G2_DEFAULT_STATES["init_hand"],
                joint_indices=[joint_idx[n] for n in OMNIPICKER_AJ_NAMES],
                is_trajectory=False,
            )
            print("[VLN] Initial joint positions set!")

            # Set up image capture annotator for VLN API
            print("[VLN] Setting up image capture...")
            def _setup_annotator():
                import omni.replicator.core as rep
                camera_path = "/genie/head_link3/head_front_Camera"
                rp = rep.create.render_product(camera_path, (640, 400))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])
                state.rgb_annotator = annotator
                print(f"[VLN] Image capture annotator ready (640x400) on {camera_path}")
            api_core.run_on_render_loop(_setup_annotator)


        except Exception as e:
            print(f"[VLN] ERROR loading scene: {e}")
            import traceback
            traceback.print_exc()

    load_thread = threading.Thread(target=load_scene, daemon=True)
    load_thread.start()

    _count = [0]
    _t = [time.time()]

    def callback_physics(step_size):
        _count[0] += 1
        now = time.time()
        if now - _t[0] >= 5.0:
            print(f"[Physics] {_count[0] / (now - _t[0]):.1f} Hz | scene_loaded={state.scene_loaded} | initialized={state.initialized}")
            _count[0] = 0
            _t[0] = now

        api_core.physics_step()

        if state.scene_loaded:
            try:
                api_core.on_ros_tick(step_size)
            except Exception:
                pass
            try:
                rclpy.spin_once(vln_node, timeout_sec=0)
            except Exception:
                pass
            apply_action(api_core)

    ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)

    print("\n" + "=" * 50)
    print("[VLN] Starting... wait for scene to load.")
    print("[VLN] Keyboard: W=forward A=left S=back D=right")
    print("[VLN] Step: 0.25m forward, 15 deg turn")
    print("=" * 50 + "\n")

    while simulation_app.is_running():
        ui_builder.my_world.step(render=True)
        api_core.render_step()
        # Handle image capture request from API server
        if state.capture_requested.is_set() and state.rgb_annotator is not None:
            try:
                state.last_image = state.rgb_annotator.get_data()
            except Exception as e:
                print(f"[VLN] Capture error: {e}")
                state.last_image = None
            state.capture_requested.clear()
            state.capture_done.set()
        if api_core.exit:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
