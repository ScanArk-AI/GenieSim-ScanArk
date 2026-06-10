# 3D Gaussian Splatting 渲染工具

本仓库提供了两个用于渲染 3D Gaussian Splatting (3DGS) 模型的脚本，可以从训练好的 PLY 模型文件生成 RGB 图像和深度图像。

## 运行环境配置

### 系统要求

- **操作系统**：Ubuntu 18.04 / 20.04 / 22.04（推荐）
- **GPU**：NVIDIA GPU（支持 CUDA，建议显存 >= 4GB）
- **CUDA**：CUDA 11.0 或更高版本
- **Python**：Python 3.8 或更高版本

### Ubuntu 环境配置步骤

#### 1. 安装 NVIDIA 驱动和 CUDA

首先检查 GPU 和驱动状态：

```bash
# 检查 GPU 是否被识别
nvidia-smi

# 如果没有输出，需要安装 NVIDIA 驱动
# Ubuntu 20.04/22.04 推荐使用自动安装
sudo ubuntu-drivers autoinstall
# 或手动安装特定版本
sudo apt install nvidia-driver-535  # 根据您的GPU选择合适版本

# 重启系统
sudo reboot
```

安装 CUDA Toolkit（如果尚未安装）：

```bash
# 对于 Ubuntu 20.04/22.04，推荐使用 CUDA 11.8 或 12.0
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 或使用包管理器安装（Ubuntu 22.04）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

配置环境变量（添加到 `~/.bashrc` 或 `~/.zshrc`）：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 使配置生效
source ~/.bashrc  # 或 source ~/.zshrc
```

#### 2. 安装 Python 和 pip

```bash
# 检查 Python 版本（需要 3.8+）
python3 --version

# 如果没有安装，使用 apt 安装
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 升级 pip
python3 -m pip install --upgrade pip
```

#### 3. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv gs_env

# 激活虚拟环境
source gs_env/bin/activate

# 升级 pip 和 setuptools
pip install --upgrade pip setuptools wheel
```

#### 4. 安装 PyTorch（支持 CUDA）

根据您的 CUDA 版本安装对应的 PyTorch：

```bash
# 对于 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 PyTorch 和 CUDA 是否正常工作
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 5. 安装 Python 依赖

```bash
# 安装基础依赖
pip install numpy pillow plyfile tqdm

# 安装 OpenCV（推荐，用于深度图可视化）
pip install opencv-python

# 或者使用 matplotlib（OpenCV 的替代方案）
pip install matplotlib
```

#### 6. 编译 CUDA 扩展模块

本项目需要编译三个 CUDA 扩展模块：

**a) diff-gaussian-rasterization**

```bash
cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../..
```

**b) simple-knn**

```bash
cd submodules/simple-knn
python setup.py install
cd ../..
```

**c) fused-ssim（可选，用于训练时计算 SSIM）**

```bash
cd submodules/fused-ssim
python setup.py install
cd ../..
```

**如果编译遇到问题，请检查：**

- 确保已安装 CUDA Toolkit 和对应的开发工具
- 确保环境变量 `CUDA_HOME` 或 `CUDA_PATH` 正确设置：
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- 确保安装了编译工具：
  ```bash
  sudo apt install build-essential
  ```

#### 7. 验证安装

运行以下命令验证所有依赖是否正确安装：

```bash
python3 -c "
import torch
import numpy as np
from PIL import Image
import cv2
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print('✓ diff-gaussian-rasterization installed')
except:
    print('✗ diff-gaussian-rasterization not installed')

try:
    import simple_knn
    print('✓ simple-knn installed')
except:
    print('✗ simple-knn not installed')

print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
```

### 快速安装脚本（Ubuntu）

您也可以使用以下脚本快速配置环境：

```bash
#!/bin/bash
# 快速安装脚本（请根据实际情况修改）

# 激活虚拟环境（如果使用）
# source gs_env/bin/activate

# 安装 PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Python 依赖
pip install numpy pillow plyfile tqdm opencv-python

# 编译 CUDA 扩展
cd submodules/diff-gaussian-rasterization && python setup.py install && cd ../..
cd submodules/simple-knn && python setup.py install && cd ../..
cd submodules/fused-ssim && python setup.py install && cd ../..

echo "安装完成！"
```

### 常见问题排查

1. **CUDA out of memory**：减少渲染分辨率或使用更大的 GPU
2. **编译错误**：确保 CUDA 版本与 PyTorch 版本兼容，检查 `CUDA_HOME` 环境变量
3. **模块导入错误**：确保在项目根目录运行脚本，所有 CUDA 扩展已正确编译
4. **OpenCV 不可用**：脚本会回退到 matplotlib，但推荐安装 OpenCV 以获得更好的性能

## 脚本功能说明

### render_simple.py

简化的 3DGS 渲染脚本，适用于单次或少量相机视角的渲染。

**主要功能：**
- 读取 PLY 格式的 3DGS 模型文件
- 支持从单个 JSON 文件或包含多个 JSON 文件的目录读取相机位姿
- 渲染每个相机视角的 RGB 图像和深度图像
- 深度图使用 jet 伪彩色可视化
- 支持双边滤波对深度图进行平滑处理
- 输出原始深度图和滤波后的深度图（PNG 可视化 + NPY 原始数据）

**输出目录结构：**
```
output/
├── rgb/              # RGB 图像（PNG格式）
├── depth/            # 原始深度图（PNG伪彩色 + NPY原始数据）
└── depth_filtered/   # 滤波后的深度图（PNG伪彩色 + NPY原始数据）
```

### render_batch.py

批量渲染脚本，适用于大量相机视角的高效批量渲染。

**主要功能：**
- 从单个 JSON 文件读取多个相机位姿
- 批量渲染所有相机视角
- 自动生成 RGBD 拼接图像（RGB 和深度图水平拼接）
- 支持深度图双边滤波
- 使用进度条显示渲染进度

**输出目录结构：**
```
output_batch/
├── rgb/              # RGB 图像（PNG格式）
├── depth/            # 原始深度图（PNG伪彩色 + NPY原始数据）
├── depth_filtered/   # 滤波后的深度图（PNG伪彩色 + NPY原始数据）
└── RGBD/             # RGB和深度图拼接图像（PNG格式）
```

**与 render_simple.py 的区别：**
- 需要从单个 JSON 文件读取多个相机位姿（JSON 中包含 `cameras` 数组）
- 额外输出 RGBD 拼接图像
- 使用 tqdm 显示批量渲染进度
- 更适合批量处理大量视角

## 相机位姿 JSON 格式

### render_simple.py 使用的格式

单个相机位姿 JSON 文件（或目录中的每个 JSON 文件）格式：

```json
{
    "width": 1920,
    "height": 1080,
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 960.0,
    "cy": 540.0,
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "T": [0, 0, 5]
}
```

**参数说明：**
- `width`, `height`: 图像分辨率
- `fx`, `fy`: 相机焦距（像素单位）
- `cx`, `cy`: 主点坐标（像素单位）
- `R`: C2W（相机到世界）旋转矩阵，3x3 数组
- `T`: C2W（相机到世界）平移向量（相机在世界坐标系中的位置）

### render_batch.py 使用的格式

包含多个相机位姿的 JSON 文件格式：

```json
{
    "num_cameras": 100,
    "cameras": [
        {
            "width": 1920,
            "height": 1080,
            "fx": 1200.0,
            "fy": 1200.0,
            "cx": 960.0,
            "cy": 540.0,
            "R": [[...], [...], [...]],
            "T": [...]
        },
        ...
    ]
}
```

## 使用方法

### render_simple.py

**基本用法（单个相机位姿文件）：**

```bash
python render_simple.py --ply path/to/model.ply --cameras path/to/camera.json --output ./output
```

**批量渲染（包含多个 JSON 文件的目录）：**

```bash
python render_simple.py --ply path/to/model.ply --cameras path/to/camera_dir --output ./output
```

**完整参数示例：**

```bash
python render_simple.py \
    --ply path/to/model.ply \
    --cameras path/to/camera.json \
    --output ./output \
    --background 0 0 0 \
    --sh_degree 3 \
    --bilateral_d 9 \
    --bilateral_sigma_color 75 \
    --bilateral_sigma_space 75
```

**参数说明：**
- `--ply`, `-p`: （必需）PLY 格式的 3DGS 模型文件路径
- `--cameras`, `-c`: （必需）相机位姿 JSON 文件路径或包含多个 JSON 文件的目录
- `--output`, `-o`: 输出目录，默认为 `./output`
- `--background`, `-bg`: 背景颜色 RGB (0-1)，默认 `[0, 0, 0]`（黑色）
- `--sh_degree`: 球谐函数度数，默认 `3`
- `--no_bilateral`: 禁用双边滤波（默认启用）
- `--bilateral_d`: 双边滤波直径，默认 `9`
- `--bilateral_sigma_color`: 双边滤波颜色空间标准差，默认 `75`
- `--bilateral_sigma_space`: 双边滤波坐标空间标准差，默认 `75`

### render_batch.py

**基本用法：**

```bash
python render_batch.py --ply path/to/model.ply --cameras path/to/cameras.json --output ./output_batch
```

**完整参数示例：**

```bash
python render_batch.py \
    --ply path/to/model.ply \
    --cameras path/to/cameras.json \
    --output ./output_batch \
    --background 0 0 0 \
    --sh_degree 3 \
    --bilateral_d 9 \
    --bilateral_sigma_color 75 \
    --bilateral_sigma_space 75
```

**参数说明：**
- `--ply`, `-p`: （必需）PLY 格式的 3DGS 模型文件路径
- `--cameras`, `-c`: （必需）包含多个相机位姿的 JSON 文件路径
- `--output`, `-o`: 输出目录，默认为 `./output_batch`
- `--background`, `-bg`: 背景颜色 RGB (0-1)，默认 `[0, 0, 0]`（黑色）
- `--sh_degree`: 球谐函数度数，默认 `3`
- `--no_bilateral`: 禁用双边滤波（默认启用）
- `--bilateral_d`: 双边滤波直径，默认 `9`
- `--bilateral_sigma_color`: 双边滤波颜色空间标准差，默认 `75`
- `--bilateral_sigma_space`: 双边滤波坐标空间标准差，默认 `75`

## 输出文件说明

### RGB 图像
- 格式：PNG
- 命名：`00000.png`, `00001.png`, ...（按相机索引顺序）
- 内容：3DGS 模型在该相机视角下的渲染结果

### 深度图

**PNG 格式（伪彩色）：**
- 使用 jet colormap 将深度值转换为颜色
- 蓝色表示近距离，红色表示远距离
- 便于直观查看深度分布

**NPY 格式（原始数据）：**
- NumPy 数组格式，保存原始深度值
- 可用于后续数值计算和处理
- 使用 `np.load()` 加载

### RGBD 拼接图像（仅 render_batch.py）
- 格式：PNG
- 内容：RGB 图像和滤波后的深度图水平拼接
- 方便同时查看 RGB 和深度信息

## 依赖要求

**核心依赖：**
- PyTorch（支持 CUDA，版本 >= 1.12.0）
- NumPy
- PIL (Pillow)
- plyfile
- tqdm（用于 render_batch.py 的进度条）

**可选依赖：**
- OpenCV（推荐，用于深度图 jet colormap 和双边滤波）
- matplotlib（OpenCV 的替代方案，用于 jet colormap）

**CUDA 扩展模块（需要编译）：**
- diff-gaussian-rasterization
- simple-knn
- fused-ssim（可选）

详细的安装步骤请参考前面的"运行环境配置"章节。

## 注意事项

1. **GPU 要求**：脚本需要在支持 CUDA 的 GPU 上运行
2. **相机坐标系**：确保相机位姿 JSON 中的 `R` 和 `T` 使用 C2W（相机到世界）格式
3. **深度图处理**：如果不需要深度图滤波，可以使用 `--no_bilateral` 参数禁用
4. **文件格式**：确保 PLY 文件是标准的 3DGS 模型格式

## 示例

假设您有一个训练好的 3DGS 模型 `model.ply` 和相机位姿文件 `cameras.json`：

```bash
# 使用 render_simple.py 渲染
python render_simple.py --ply model.ply --cameras cameras.json --output ./output

# 或使用 render_batch.py 批量渲染
python render_batch.py --ply model.ply --cameras cameras.json --output ./output_batch
```

渲染完成后，您可以在输出目录中找到所有生成的图像文件。
