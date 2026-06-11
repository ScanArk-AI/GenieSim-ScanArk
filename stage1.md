# Stage 1: 采集 StreamVLN 训练数据

本阶段目标：给定一个 3DGS `.ply` 场景，手动在 2D 地图上框选随机起点区域，并点击目标物体前方的停靠点，生成 StreamVLN fine-tune 使用的数据集。

教师可以在课堂上口头指定所有学员选择同一个目标物体和同一个起点区域，但工具本身仍保留交互式选择流程，方便学员理解数据是如何产生的。

## 1. 环境检查

```bash
cd /root/genie_sim
conda activate streamvln

python - <<'PY'
import open3d
import cv2
import plyfile
import py360convert
import diff_gaussian_rasterization
import simple_knn
import torch

print("open3d OK")
print("cv2 OK")
print("plyfile OK")
print("py360convert OK")
print("diff_gaussian_rasterization OK")
print("simple_knn OK")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

## 2. 采集数据

正式采集示例：

```bash
cd /root/genie_sim
conda activate streamvln

python course_tools/collect_streamvln_data.py \
  --ply_path scenario/sample.ply \
  --scene_name classroom \
  --output_dir data/course/classroom_target \
  --num_episodes 50
```

运行后会依次出现：

1. 起点区域选择窗口
2. 目标点选择窗口
3. 轨迹渲染预览窗口

## 3. 起点区域选择

窗口名：

```text
StreamVLN Start Region Selector
```

操作：

- 鼠标左键：添加多边形顶点
- 鼠标右键或 `Backspace`：撤销上一个点
- `A`：使用全部安全可通行区域
- `R`：重画区域
- `Enter` 或 `Space`：确认区域
- `Q` 或 `Esc`：退出


## 4. 目标点选择

窗口名：

```text
StreamVLN Course Target Selector
```

操作：

- 鼠标左键点击目标物体前方的可通行停靠点
- 按住并拖动鼠标可以设置机器人最终朝向
- `Enter` 或 `Space`：确认目标
- `R`：重新选择
- `Q` 或 `Esc`：退出

确认后，终端会要求输入：

- `target_name`
- `instruction`

如果只输入目标名称，不输入指令，工具会使用默认模板：

```text
Go to the {target_name} and stop in front of it.
```

## 5. 采集过程预览

渲染每条轨迹时，工具默认会弹出：

```text
StreamVLN Data Collection Preview
```

窗口内容：

- 左侧：2D 地图、起点、终点、当前轨迹和朝向箭头
- 右上：当前 RGB 渲染帧
- 右下：当前深度图

如果没有图形界面，或想更快生成数据，可以加：

```bash
--no_preview
```

## 6. 小规模测试

正式采 50 条前，可以先采 2-5 条确认流程：

```bash
python course_tools/collect_streamvln_data.py \
  --ply_path scenario/sample.ply \
  --scene_name classroom \
  --output_dir /tmp/classroom_test \
  --num_episodes 2
```

## 7. 输出结构

生成完成后，输出目录类似：

```text
data/course/classroom_target/
  target.json
  trajectories.json
  annotate_episodes.json
  annotations.json
  camera_poses_traj_0000.json
  render_trajectory_0000/
    rgb/*.png
    depth/*.npy
    depth_color/*.png
```

其中：

- `target.json`：目标名称、指令、目标点和起点区域
- `trajectories.json`：中间轨迹数据
- `annotate_episodes.json`：带指令和动作的 episode 文件
- `annotations.json`：Stage 2 训练直接使用的标注文件
- `render_trajectory_xxxx/rgb/`：StreamVLN 训练使用的 RGB 帧

## 8. 检查数据集

```bash
python course_tools/check_streamvln_dataset.py \
  --data_root data/course/classroom_target
```

看到下面输出说明格式可用于训练：

```text
OK: dataset looks compatible with StreamVLN navigation training.
```

## 9. 可选开发建议

本阶段的 baseline 已经可以生成可训练数据。为了更好的效果，鼓励学员优化下面两个方向。

### 9.1 优化路径规划算法

当前轨迹规划主要在下面文件中实现：

```text
course_tools/vln_collection_backend/managers/path_planner.py
```

采集脚本 [course_tools/collect_streamvln_data.py](course_tools/collect_streamvln_data.py) 会调用后端采样与规划逻辑，最终使用这里的路径规划结果生成动作序列和渲染轨迹。

学员可以尝试优化：

- 障碍物避让策略，减少路径贴墙或穿过狭窄区域。
- 机器人本体膨胀半径，让路径更符合真实机器人尺寸。
- A* cost function，例如给靠近障碍物的区域更高代价。
- 路径平滑算法，让轨迹更自然、更少抖动。
- 转弯和 waypoint 选择，让动作序列更稳定。
- 过滤不合理路径，例如过长、转向过多、绕路明显的轨迹。

优化目标不是只让路径能到达终点，而是生成更适合训练 StreamVLN 的轨迹：

```text
路径连续、视觉清晰、转弯自然、终点朝向合理、动作序列稳定。
```

### 9.2 人工细粒度标注自然语言指令

默认情况下，如果学员只输入目标名称，工具会生成统一模板：

```text
Go to the {target_name} and stop in front of it.
```

这个模板可以跑通训练，但表达比较粗。学员可以在采集完成后，查看每一条轨迹的第一视角 RGB 图像，并为不同轨迹人工编写更细粒度的导航指令。

每条轨迹的 RGB 图像在：

```text
data/course/classroom_target/render_trajectory_xxxx/rgb/
```

训练标注文件在：

```text
data/course/classroom_target/annotations.json
```

学员可以打开 `annotations.json`，把每条样本中的：

```json
"instructions": ["Go to the target and stop in front of it."]
```

改成更具体的描述，例如：

```json
"instructions": ["Move forward along the counter, pass the chairs on your right, then stop in front of the Sprite bottle on the counter."]
```

更好的指令通常包含：

- 目标物体名称和外观。
- 目标所在位置，例如柜台上、桌子旁、门口附近。
- 路径中的关键地标，例如椅子、柜台、货架、门、墙面。
- 转向信息，例如经过某物后左转或右转。
- 停止条件，例如停在目标正前方。

注意：指令必须和实际 RGB 轨迹一致。如果指令描述了轨迹中没有出现的地标，或者方向与轨迹不一致，反而会降低训练效果。

## 10. 进入 Stage 2

Stage 1 完成后，使用：

```text
data/course/classroom_target
```

作为 Stage 2 的训练数据目录。
