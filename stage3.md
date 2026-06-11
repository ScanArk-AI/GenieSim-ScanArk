# Stage 3: GenieSim 实时推理

本阶段目标：在 GenieSim 中加载 Stage 2 微调后的 StreamVLN LoRA 模型，输入自然语言指令，观察机器人是否能在仿真环境中完成目标导航。

本阶段使用：

```text
基础模型：/root/genie_sim/StreamVLN/checkpoints/streamvln_real_world
LoRA adapter：/root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world
视觉塔：/root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384
推理脚本：/root/genie_sim/StreamVLN/run_geniesim.py
```

## 1. 前置条件

确认 Stage 2 训练输出存在：

```bash
ls -lh /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world
```

通常应包含：

```text
adapter_config.json
adapter_model.bin 或 adapter_model.safetensors
non_lora_trainables.bin
training_args.bin
```

确认基础模型和视觉塔存在：

```bash
ls /root/genie_sim/StreamVLN/checkpoints/streamvln_real_world/config.json
ls /root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384/config.json
```

## 2. 启动 GenieSim VLN 仿真端

开第一个终端，在项目根目录执行：

```bash
cd /root/genie_sim

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib
export SIM_REPO_ROOT=/root/genie_sim

/isaac-sim/python.sh scripts/vln_app.py \
  --config source/geniesim/config/my_scene_vln.yaml \
  --/renderer/multiGpu/enabled=False \
  --/physics/cudaDevice=0
```

等待日志出现：

```text
[VLN] Scene and robot loaded successfully!
[API] VLN API server listening on 0.0.0.0:12347
```

如果机器人材质发黑，在 Isaac Sim 右上角 Lighting 选择 `Camera Light`。当前代码会尝试默认设置为 Camera Light。

## 3. 运行实时推理

开第二个终端：

```bash
cd /root/genie_sim/StreamVLN
conda activate streamvln

python run_geniesim.py \
  --model_path /root/genie_sim/StreamVLN/checkpoints/streamvln_real_world \
  --adapter_path /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world \
  --vision_tower_path /root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384 \
  --attn_implementation flash_attention_2 \
  --device cuda:0 \
  --instruction "Go to the target object and stop in front of it." \
  --target_json /root/genie_sim/data/course/classroom_target/target.json \
  --success_radius 1.0 \
  --eval_output /root/genie_sim/StreamVLN/runs/eval_result.json \
  --max_steps 100 \
  --save_images \
  --output_dir /root/genie_sim/StreamVLN/runs
```

把 `--instruction` 换成 自己设定的自然语言导航指令，推荐用与采集数据相同的指令。

如果希望使用 [run_geniesim.py](StreamVLN/run_geniesim.py) 里的默认 instruction，可以省略 `--instruction`。

评测参数说明：

- `--target_json`：Stage 1 生成的目标点文件，通常是 `data/course/classroom_target/target.json`。
- `--success_radius`：机器人最终停止点距离目标点多少米以内算成功，默认建议 `1.0`。
- `--eval_output`：保存评测结果 JSON。

## 4. 观察结果与评测指标

运行过程中终端会打印：

```text
[Episode] Instruction: ...
[Step   1] Action: forward ...
[Step   2] Action: turn_left ...
...
```

如果加了 `--save_images`，图像会保存到：

```text
/root/genie_sim/StreamVLN/runs/taskXX/
```

其中包括：

```text
instruction.txt
step_000.jpg
step_001.jpg
...
```

可以通过保存的图片检查模型每一步看到的第一视角画面。

如果传入 `--target_json`，运行结束后终端还会打印：

```text
[Eval] success: True/False
[Eval] stopped: True/False
[Eval] final_distance_to_goal: ... m
[Eval] min_distance_to_goal: ... m
[Eval] path_length: ... m
[Eval] num_steps: ...
```

同时会保存评测结果到：

```text
/root/genie_sim/StreamVLN/runs/eval_result.json
```

评测指标含义：

- `success`：机器人输出 `stop`，且最终停止点在目标点 `success_radius` 范围内。
- `oracle_success`：机器人运动过程中曾经进入过目标点 `success_radius` 范围内。
- `final_distance_to_goal`：最终停止位置到目标点的距离。
- `min_distance_to_goal`：整个过程中离目标最近的距离。
- `path_length`：机器人实际移动路径长度。
- `num_steps`：执行动作步数。
- `action_counts`：不同动作出现次数。

## 5. 建议测试方式

建议每次测试记录：

```text
instruction
是否到达目标
是否正确 stop
用了多少步
失败时停在哪里
失败时保存的图像
```

如果模型表现不稳定，可以回到 Stage 1/2 调整：

- 增加训练数据数量
- 改善目标点和起点区域选择
- 改善导航指令
- 增加训练 epoch
- 检查训练 loss 和 grad_norm 是否正常

## 6. 常见问题

### 连接不上 12347

确认 GenieSim 终端中有：

```text
[API] VLN API server listening on 0.0.0.0:12347
```

### LoRA adapter 找不到

确认：

```bash
ls /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world/adapter_config.json
ls /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world/non_lora_trainables.bin
```

### 模型一开始就 stop

可能原因：

- instruction 和训练数据不一致
- 起点离目标过近
- 训练数据太少
- 模型没有学到有效动作策略

可以先用 `--save_images` 保存过程图像，检查模型看到的画面。

### 显存不足

关闭其它 GPU 进程，尤其是多余的训练进程。也可以尝试：

```bash
--attn_implementation sdpa
```

但课程默认推荐 `flash_attention_2`。
