#!/usr/bin/env python3
"""
StreamVLN + GenieSim Adapter
Connects the StreamVLN model to Isaac Sim via vln_env.py.

Usage:
    1. Start Isaac Sim:  (in Docker container)
       SIM_REPO_ROOT=/geniesim/main CUDA_VISIBLE_DEVICES=0 omni_python scripts/vln_app.py --config source/geniesim/config/my_scene_vln.yaml

    2. Run this script:  (on host, in streamvln conda env)
       conda activate streamvln
       cd /path/to/genie_sim/StreamVLN
       python run_geniesim.py --instruction "Walk to the coffee table and stop."

    The model runs on cuda:0 by default.
"""

import argparse
import sys
import os
import time
import json
from collections import Counter
import numpy as np
import torch
import transformers
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "streamvln"))

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))  # for vln_env
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "streamvln_real_world",
)
DEFAULT_ADAPTER_PATH = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "course_classroom_target_from_streamvln_real_world",
)
DEFAULT_VISION_TOWER_PATH = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "siglip-so400m-patch14-384",
)

from PIL import Image
from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM
from vln_env import VLNEnv

# StreamVLN action mapping -> VLN env action strings
ACTION_MAP = {
    0: "stop",
    1: "forward",
    2: "turn_left",
    3: "turn_right",
}

DEFAULT_INSTRUCTION = "Go to the sofa and stop in front of it."


def _load_non_lora_trainables(model, adapter_path):
    """Load non-LoRA trainables such as mm_projector saved by StreamVLN training."""
    if not adapter_path:
        return

    non_lora_path = os.path.join(adapter_path, "non_lora_trainables.bin")
    if not os.path.isfile(non_lora_path):
        return

    print(f"[StreamVLN] Loading non-LoRA trainables: {non_lora_path}")
    state_dict = torch.load(non_lora_path, map_location="cpu")
    cleaned = {}
    for key, value in state_dict.items():
        candidates = [key]
        for prefix in ("base_model.model.", "base_model."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])
        for candidate in candidates:
            cleaned[candidate] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(unexpected)
    print(f"[StreamVLN] Loaded non-LoRA tensors: {loaded}; unexpected: {len(unexpected)}")
    if unexpected:
        print(f"[StreamVLN] Unexpected non-LoRA keys example: {unexpected[:3]}")


def load_model(
    model_path,
    device="cuda:0",
    adapter_path=None,
    vision_tower_path=DEFAULT_VISION_TOWER_PATH,
    attn_implementation="flash_attention_2",
):
    """Load StreamVLN base model, optional LoRA adapter, and tokenizer."""
    print(f"[StreamVLN] Loading base model from {model_path} on {device}...")
    if adapter_path:
        print(f"[StreamVLN] Loading LoRA adapter from {adapter_path}")
    t0 = time.time()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=4096, padding_side="right"
    )

    config = transformers.AutoConfig.from_pretrained(model_path)
    if vision_tower_path:
        vision_tower_path = os.path.abspath(vision_tower_path)
        if not os.path.isdir(vision_tower_path):
            raise FileNotFoundError(f"Vision tower path not found: {vision_tower_path}")
        config.mm_vision_tower = vision_tower_path
        config.vision_tower = vision_tower_path
        print(f"[StreamVLN] Using local vision tower: {vision_tower_path}")

    model = StreamVLNForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )

    _load_non_lora_trainables(model, adapter_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("[StreamVLN] LoRA adapter loaded")

    model.model.num_history = 8
    model.reset(1)
    model.requires_grad_(False)
    model.to(device)
    model.eval()

    print(f"[StreamVLN] Model loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def create_evaluator(model, tokenizer, args):
    """Create VLNEvaluator with sensor config."""
    vln_sensor_config = {
        "rgb_height": 1.25,
        "camera_intrinsic": np.array([
            [192., 0., 191.42857143, 0.],
            [0., 192., 191.42857143, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]),
    }
    return VLNEvaluator(vln_sensor_config, model=model, tokenizer=tokenizer, args=args)


def run_episode(env, evaluator, instruction, max_steps=100, save_images=False, output_dir="runs", return_trace=False):
    """
    Run one VLN episode: model receives images from Isaac Sim and outputs actions.

    Args:
        env: VLNEnv instance connected to Isaac Sim
        evaluator: VLNEvaluator with loaded model
        instruction: natural language navigation instruction
        max_steps: maximum steps before forced stop
        save_images: whether to save observation images
        output_dir: directory for saved images
    """
    if save_images:
        # Find next task number to avoid overwriting previous runs
        os.makedirs(output_dir, exist_ok=True)
        existing = [d for d in os.listdir(output_dir) if d.startswith("task") and os.path.isdir(os.path.join(output_dir, d))]
        task_num = len(existing) + 1
        task_dir = os.path.join(output_dir, f"task{task_num:02d}")
        os.makedirs(task_dir, exist_ok=True)
        # Save instruction
        with open(os.path.join(task_dir, "instruction.txt"), "w") as f:
            f.write(instruction + "\n")
    else:
        task_dir = output_dir

    # Reset
    print(f"\n{'='*60}")
    print(f"[Episode] Instruction: {instruction}")
    print(f"{'='*60}")

    evaluator.reset_memory()
    obs, info = env.observe()  # stay at current position, don't teleport to origin
    print(f"[Episode] Initial state: {info}")
    trace = []
    if isinstance(info, dict):
        trace.append({
            "step": 0,
            "action": "observe",
            "x": info.get("x"),
            "y": info.get("y"),
            "yaw": info.get("yaw"),
        })

    if save_images and obs is not None:
        Image.fromarray(obs).save(f"{task_dir}/step_000.jpg")

    step_count = 0
    action_history = []
    start_time = time.time()

    while step_count < max_steps:
        if obs is None:
            print("[Episode] ERROR: No observation received!")
            break

        # Convert RGB image to BGR (StreamVLN expects BGR from OpenCV convention)
        rgb_image = obs  # (H, W, 3) RGB from our env
        bgr_image = rgb_image[:, :, ::-1]  # RGB -> BGR

        # Run model: every 4th step does actual inference, others reuse last prediction
        run_model = (evaluator.step_id % 4 == 0)
        return_action, gen_time, llm_output = evaluator.step(
            0, bgr_image, instruction, run_model=run_model
        )
        evaluator.step_id += 1

        if return_action is not None:
            action_seq = return_action

        # Get current action from the sequence
        action_idx_in_seq = (evaluator.step_id - 1) % 4
        if action_idx_in_seq < len(action_seq):
            action_id = action_seq[action_idx_in_seq]
        else:
            action_id = action_seq[-1] if len(action_seq) > 0 else 0

        action_str = ACTION_MAP.get(action_id, "stop")
        step_count += 1
        action_history.append(action_str)

        if run_model and llm_output:
            print(f"[Model] LLM output: {llm_output[-100:]}")  # last 100 chars
        print(f"[Step {step_count:3d}] Action: {action_str:10s} | gen_time={gen_time:.2f}s | pos=({info.get('x',0):.2f}, {info.get('y',0):.2f}) yaw={info.get('yaw',0):.1f}")

        if action_str == "stop":
            print(f"\n[Episode] STOP reached at step {step_count}")
            break

        # Execute action in Isaac Sim
        obs, info = env.step(action_str)
        if isinstance(info, dict):
            trace.append({
                "step": step_count,
                "action": action_str,
                "x": info.get("x"),
                "y": info.get("y"),
                "yaw": info.get("yaw"),
            })

        if save_images and obs is not None:
            Image.fromarray(obs).save(f"{task_dir}/step_{step_count:03d}.jpg")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[Episode] Finished in {step_count} steps, {elapsed:.1f}s")
    print(f"[Episode] Final state: {info}")
    print(f"[Episode] Actions: {' -> '.join(action_history)}")
    print(f"{'='*60}\n")

    if return_trace:
        return action_history, info, trace
    return action_history, info


def _load_target_eval(target_json_path):
    """Load Stage 1 target point for evaluation."""
    with open(target_json_path, "r") as f:
        data = json.load(f)

    goal = data.get("goal_position") or data.get("goal")
    if goal is None or len(goal) < 2:
        raise ValueError(f"target_json missing goal_position: {target_json_path}")

    return {
        "target_json": target_json_path,
        "target_name": data.get("target_name", ""),
        "instruction": data.get("instruction", ""),
        "goal_position": [float(goal[0]), float(goal[1]), float(goal[2]) if len(goal) > 2 else 0.0],
        "goal_yaw": data.get("goal_yaw"),
    }


def _compute_eval_metrics(target, trace, actions, final_info, success_radius):
    """Compute navigation metrics from an episode trace."""
    goal_xy = np.array(target["goal_position"][:2], dtype=float)
    points = []
    for item in trace:
        x, y = item.get("x"), item.get("y")
        if x is None or y is None:
            continue
        points.append(np.array([float(x), float(y)], dtype=float))

    if points:
        final_xy = points[-1]
        dists = [float(np.linalg.norm(p - goal_xy)) for p in points]
        path_length = float(sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))))
        min_distance = float(min(dists))
        final_distance = float(dists[-1])
    else:
        final_xy = np.array([float("nan"), float("nan")])
        path_length = 0.0
        min_distance = float("inf")
        final_distance = float("inf")

    stopped = len(actions) > 0 and actions[-1] == "stop"
    success = bool(stopped and final_distance <= success_radius)
    oracle_success = bool(min_distance <= success_radius)

    return {
        "target_name": target.get("target_name", ""),
        "target_json": target.get("target_json", ""),
        "goal_position": target["goal_position"],
        "success_radius": float(success_radius),
        "success": success,
        "oracle_success": oracle_success,
        "stopped": stopped,
        "num_steps": int(len(actions)),
        "final_position": [float(final_xy[0]), float(final_xy[1])],
        "final_yaw": final_info.get("yaw") if isinstance(final_info, dict) else None,
        "final_distance_to_goal": final_distance,
        "min_distance_to_goal": min_distance,
        "path_length": path_length,
        "action_counts": dict(Counter(actions)),
        "actions": list(actions),
        "trace": trace,
    }


def evaluate_episode(target_json_path, trace, actions, final_info, success_radius, instruction):
    target = _load_target_eval(target_json_path)
    metrics = _compute_eval_metrics(
        target=target,
        trace=trace,
        actions=actions,
        final_info=final_info,
        success_radius=success_radius,
    )
    metrics["instruction"] = instruction
    metrics["target_json"] = target_json_path
    return metrics


def _print_eval_metrics(metrics):
    print("\n" + "=" * 60)
    print("[Eval] Navigation metrics")
    print("=" * 60)
    print(f"[Eval] target: {metrics['target_name']}")
    print(f"[Eval] success: {metrics['success']}")
    print(f"[Eval] oracle_success: {metrics['oracle_success']}")
    print(f"[Eval] stopped: {metrics['stopped']}")
    print(f"[Eval] final_distance_to_goal: {metrics['final_distance_to_goal']:.3f} m")
    print(f"[Eval] min_distance_to_goal: {metrics['min_distance_to_goal']:.3f} m")
    print(f"[Eval] path_length: {metrics['path_length']:.3f} m")
    print(f"[Eval] num_steps: {metrics['num_steps']}")
    print(f"[Eval] action_counts: {metrics['action_counts']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run StreamVLN on GenieSim")
    parser.add_argument("--model_path", type=str,
                        default=DEFAULT_MODEL_PATH,
                        help="Path to base StreamVLN model checkpoint")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional LoRA adapter checkpoint from course fine-tuning")
    parser.add_argument("--vision_tower_path", type=str, default=DEFAULT_VISION_TOWER_PATH,
                        help="Path to local SigLIP vision tower checkpoint")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation for model loading")
    parser.add_argument("--instruction", type=str,
                        default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU for model inference")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="VLN API server host")
    parser.add_argument("--port", type=int, default=12347,
                        help="VLN API server port")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="runs/geniesim")
    parser.add_argument("--target_json", type=str, default=None,
                        help="Optional Stage 1 target.json for success evaluation")
    parser.add_argument("--success_radius", type=float, default=1.0,
                        help="Distance threshold in meters for success when target_json is provided")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="Optional path to save evaluation metrics JSON")
    # StreamVLN model params
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        vision_tower_path=args.vision_tower_path,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
    )

    # Create evaluator
    evaluator = create_evaluator(model, tokenizer, args)

    # Warm up model with a dummy forward pass
    print("[StreamVLN] Warming up model...")
    evaluator.step(0, np.zeros((400, 640, 3), dtype=np.uint8), "move forward", run_model=True)
    evaluator.reset_memory()
    print("[StreamVLN] Model ready!")

    # Connect to Isaac Sim
    print(f"[StreamVLN] Connecting to Isaac Sim at {args.host}:{args.port}...")
    env = VLNEnv(host=args.host, port=args.port)

    # Run episode
    episode_result = run_episode(
        env, evaluator, args.instruction,
        max_steps=args.max_steps,
        save_images=args.save_images,
        output_dir=args.output_dir,
        return_trace=bool(args.target_json),
    )

    if args.target_json:
        action_history, final_info, trace = episode_result
        metrics = evaluate_episode(
            args.target_json,
            trace,
            action_history,
            final_info,
            success_radius=args.success_radius,
            instruction=args.instruction,
        )
        _print_eval_metrics(metrics)

        eval_output = args.eval_output
        if eval_output is None:
            eval_output = os.path.join(args.output_dir, "eval_result.json")
        os.makedirs(os.path.dirname(os.path.abspath(eval_output)), exist_ok=True)
        with open(eval_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Eval] Saved metrics to {eval_output}")
    else:
        action_history, final_info = episode_result

    env.close()


if __name__ == "__main__":
    main()
