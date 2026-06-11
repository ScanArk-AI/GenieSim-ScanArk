# 拉取最新代码
cd /root/genie_sim
git pull
git fetch origin
git reset --hard origin/main

# GenieSim VLN 启动与键盘控制说明

本文档适用于当前服务器环境：打开终端后已经在 Docker 容器内部，提示符类似：

```bash
root@gpufree-container:~#
```

---

## 0. 当前环境约定

- 项目目录：`/root/genie_sim`
- Isaac Sim Python：`/isaac-sim/python.sh`
- VLN 启动脚本：`/root/genie_sim/scripts/vln_app.py`
- 键盘控制脚本：`/root/genie_sim/scripts/keyboard_control.py`
- VLN 配置文件：`/root/genie_sim/source/geniesim/config/my_scene_vln.yaml`

当前配置会加载：

- 场景：`source/geniesim/assets/background/my_scene/xxx.usdz`
- 机器人：`G2_omnipicker.json`
- 初始位置：`[-5.0, -2.0, 0.0]`

---

## 1. 启动前检查

在容器终端中执行：

```bash
cd /root/genie_sim
ls /isaac-sim/python.sh
nvidia-smi
echo $DISPLAY
```

说明：

- `ls /isaac-sim/python.sh` 能找到文件，说明 Isaac Sim Python 路径正确。
- `nvidia-smi` 正常输出 GPU 信息，说明容器能访问 GPU。
- `DISPLAY` 如果为空，Isaac Sim 可能可以启动，但通常不会弹出 GUI 窗口；需要服务器/容器已配置图形显示或远程桌面。

---

## 2. 终端 1：启动 Isaac Sim 场景和机器人

在第一个容器终端执行：

```bash
cd /root/genie_sim

export SIM_REPO_ROOT=/root/genie_sim
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib

/isaac-sim/python.sh scripts/vln_app.py \
  --config source/geniesim/config/my_scene_vln.yaml \
  --/renderer/multiGpu/enabled=False \
  --/physics/cudaDevice=0
```

启动后需要等待 Isaac Sim 加载场景、机器人、碰撞网格和 VLN 控制接口。

---

## 3. 启动成功标志

终端中看到类似日志，说明场景和机器人已经加载成功：

```text
[VLN] Loading scene and robot...
[VLN] Scene and robot loaded successfully!
[VLN] Collision ready, navigation enabled.
[VLN] UDP listener started on 0.0.0.0:12346
[API] VLN API server listening on 0.0.0.0:12347
```

其中：

- `12346`：键盘控制 UDP 端口。
- `12347`：VLN / StreamVLN TCP API 端口。

只要 `vln_app.py` 还在运行，后续键盘控制脚本就可以向它发送动作。

---

## 4. 终端 2：启动键盘控制

再打开一个新的服务器终端，进入同一个容器环境后执行：

```bash
cd /root/genie_sim
python3 scripts/keyboard_control.py
```

启动后会看到：

```text
=== VLN Keyboard Controller (Habitat-style) ===
  W - Forward  (0.25m)
  S - Backward (0.25m)
  A - Turn Left  (15 deg)
  D - Turn Right (15 deg)
  Space - Stop
  Ctrl+C - Exit
================================================
Sending actions to 127.0.0.1:12346 via UDP
```

---

## 5. 键盘控制说明

键盘控制脚本每按一次键，会发送一个离散动作给 `vln_app.py`：

| 按键 | 动作 | 效果 |
| --- | --- | --- |
| `W` | `forward` | 前进 `0.25m` |
| `S` | `backward` | 后退 `0.25m` |
| `A` | `turn_left` | 左转 `15°` |
| `D` | `turn_right` | 右转 `15°` |
| `Space` | `stop` | 停止 |
| `Ctrl+C` | - | 退出键盘控制 |

`vln_app.py` 收到动作后，会在终端 1 中输出类似：

```text
[VLN] Received: forward
[VLN] forward: x=-4.75, y=-2.00, yaw=0.0
```

如果前方有碰撞物体，可能会输出：

```text
[VLN] Blocked (forward): blocked: ...
```

这表示碰撞检测阻止了机器人继续移动。

---
