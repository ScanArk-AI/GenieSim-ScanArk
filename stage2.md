# Stage 2: 用采集数据微调 StreamVLN 基础权重

本阶段目标：使用 Stage 1 采集好的目标导航数据，对 StreamVLN real-world 基础权重做小样本 LoRA fine-tune，得到可在 GenieSim 中实时推理的 adapter 权重。

当前推荐基础权重：

```text
/root/genie_sim/StreamVLN/checkpoints/streamvln_real_world
```

本地 SigLIP 视觉塔：

```text
/root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384
```

示例训练数据：

```text
/root/genie_sim/data/course/classroom_target
```

## 1. 前置检查

激活环境：

```bash
conda activate streamvln
cd /root/genie_sim
```

检查 Stage 1 数据：

```bash
python course_tools/check_streamvln_dataset.py \
  --data_root data/course/classroom_target
```

看到下面输出说明数据格式基本正确：

```text
OK: dataset looks compatible with StreamVLN navigation training.
```

## 2. 确认基础权重配置

`streamvln_real_world/config.json` 中的视觉塔必须指向本地 SigLIP：

```text
/root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384
```

可以用下面命令检查：

```bash
python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path('/root/genie_sim/StreamVLN/checkpoints/streamvln_real_world/config.json').read_text())
print('mm_vision_tower:', cfg.get('mm_vision_tower'))
print('vision_tower:', cfg.get('vision_tower'))
PY
```

如果仍然是：

```text
google/siglip-so400m-patch14-384
```

需要改成本地路径，否则离线环境会访问 HuggingFace。

## 3. 检查 flash-attn

课程环境推荐使用 FlashAttention2，4090 24GB 上更省显存。

验证：

```bash
python - <<'PY'
import flash_attn
print('flash_attn:', flash_attn.__version__)
PY
```

如果没有安装，可以安装已经准备好的 wheel：

```bash
python -m pip install /root/genie_sim/3rdparty/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

如果 `transformers` 报 `huggingface-hub>=0.23.2,<1.0`，说明下载模型时升级了 `huggingface_hub`，需要降回兼容版本：

```bash
python -m pip install "huggingface_hub==0.25.2"
```

## 4. 训练命令

进入 StreamVLN 目录：

```bash
cd /root/genie_sim/StreamVLN
conda activate streamvln
```

设置环境变量：

```bash
export PYTHONPATH=/root/genie_sim/StreamVLN:/root/genie_sim/StreamVLN/streamvln:$PYTHONPATH
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

清理旧输出：

```bash
rm -rf /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world
```

运行 4090 24GB 推荐 LoRA fine-tune：

```bash
python streamvln/streamvln_train.py \
  --model_name_or_path /root/genie_sim/StreamVLN/checkpoints/streamvln_real_world \
  --version qwen_1_5 \
  --video_folder /root/genie_sim/data/course/classroom_target \
  --group_by_task False \
  --num_history 4 \
  --num_future_steps 4 \
  --num_frames 16 \
  --data_augmentation True \
  --attn_implementation flash_attention_2 \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --mm_tunable_parts="mm_mlp_adapter,mm_lora_layer" \
  --vision_tower /root/genie_sim/StreamVLN/checkpoints/siglip-so400m-patch14-384 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  --bf16 True \
  --run_name course_classroom_target_from_streamvln_real_world \
  --output_dir /root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --mm_vision_tower_lr 2e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --dataloader_drop_last False \
  --report_to none
```

## 5. 如何判断训练正常

训练日志中重点看：

```text
loss
grad_norm
```

正常现象：

```text
loss: 有波动或下降
grad_norm: 非 0
```

例如一次正常日志：

```text
{'loss': 0.4068, 'grad_norm': 1.0128, ...}
{'loss': 0.3809, 'grad_norm': 0.8209, ...}
{'loss': 0.1810, 'grad_norm': 1.0436, ...}
```


## 6. 训练输出

训练完成后输出目录为：

```text
/root/gpufree-data/checkpoints/course_classroom_target_from_streamvln_real_world
```

LoRA 训练输出通常包含：

```text
adapter_config.json
adapter_model.bin 或 adapter_model.safetensors
non_lora_trainables.bin
training_args.bin
```

其中：

- `adapter_model.*`：LoRA adapter。
- `non_lora_trainables.bin`：训练过的非 LoRA 权重，主要是 projector。

Stage 3 推理时需要同时加载：

```text
基础模型：streamvln_real_world
LoRA adapter：course_classroom_target_from_streamvln_real_world
本地视觉塔：siglip-so400m-patch14-384
```

## 7. 显存不足时如何调整

如果 4090 24GB 出现 OOM，优先降这些参数：

```bash
--num_history 2
--num_frames 8
--model_max_length 4096
--dataloader_num_workers 0
```

## 8. 可选开发建议：尝试不同训练策略

本阶段的 baseline 命令已经验证可以正常训练。鼓励学员在跑通 baseline 后，尝试不同的训练超参数，并记录对 Stage 3 推理效果的影响。

建议优先尝试下面这些参数。

### 8.1 LoRA 参数

当前 baseline：

```bash
--lora_r 8
--lora_alpha 16
--lora_dropout 0.05
```

可尝试：

```bash
--lora_r 4
--lora_r 16
--lora_alpha 8
--lora_alpha 32
--lora_dropout 0.0
--lora_dropout 0.1
```

一般来说：

- `lora_r` 越大，可训练容量越强，但更容易过拟合，也更占显存。
- `lora_dropout` 可以缓解小数据过拟合，但过大可能学不动。

### 8.2 学习率和 epoch

当前 baseline：

```bash
--learning_rate 1e-5
--num_train_epochs 1
```

可尝试：

```bash
--learning_rate 5e-6
--learning_rate 2e-5
--num_train_epochs 2
--num_train_epochs 3
```

小数据集上不要只看训练 loss。训练 loss 很低不一定代表仿真效果更好，可能只是过拟合。最终还是要看 Stage 3 的机器人是否真的能到目标前停止。

### 8.3 历史帧和视频帧数

当前 baseline：

```bash
--num_history 4
--num_frames 16
--model_max_length 8192
```

显存允许时可尝试：

```bash
--num_history 8
--num_frames 32
--model_max_length 16384
```

显存不足时可降为：

```bash
--num_history 2
--num_frames 8
--model_max_length 4096
```

历史帧更多，模型能看到更长的导航上下文；但显存和训练时间都会增加。

### 8.4 可训练模块

当前 baseline：

```bash
--mm_tunable_parts="mm_mlp_adapter,mm_lora_layer"
```

这是课程推荐配置，比较稳定。进阶学员可以尝试不同组合，但要注意显存和训练稳定性。例如只训练 LoRA：

```bash
--mm_tunable_parts="mm_lora_layer"
```

或只训练 projector：

```bash
--mm_tunable_parts="mm_mlp_adapter"
```

通常不建议在 4090 24GB 上全量训练视觉塔和语言模型，因为很容易 OOM。

### 8.5 实验记录建议

每次实验建议记录：

```text
数据量
指令是否人工细化
lora_r / lora_alpha / lora_dropout
learning_rate
epoch
num_history / num_frames
训练 loss
Stage 3 是否成功到达目标
失败原因分析
```

这样最终不仅能比较谁跑得更好，也能解释为什么某些训练策略更有效。

## 9. 课程建议

流程验证：

```text
数据量：5-10 条
epoch：1
```

正式小样本实验：

```text
数据量：约 50 条
epoch：1-3
```

建议先从 1 epoch 开始。如果 Stage 3 仿真表现不稳定，再增加到 2-3 epoch。
