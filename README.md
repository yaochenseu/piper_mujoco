# Piper\_Mujoco\_Sim
---

## Overview / 项目简介


本仓库为“基于多核 AMP 通信的人机交互机械臂系统”这个作品的核心代码：

---

## Repository Layout / 目录结构

```
Piper_Mujoco_Sim/
├── main.py                          # UR5 sim pipeline (env.render() dict interface)
├── main_yoloWorld_sam.py            # UR5 MuJoCo sim + GroundingDINO/SAM + GraspNet
├── main_piper_yoloWorld_sam.py      # Piper MuJoCo sim + GroundingDINO/SAM + GraspNet
├── main_real_piper_yoloWorld_sam.py # Real Piper + D435i + GraspNet
├── LLM_task_plan.py                 # LLM task planner (Qwen / DeepSeek)
├── cv_process.py                    # GroundingDINO detection + SAM segmentation
│
├── utils/                           # Shared utilities
│   ├── path_setup.py                # sys.path registration for graspnet-baseline
│   ├── data_loader.py               # Unified point-cloud data loading
│   ├── graspnet_runner.py           # GraspNet model loader
│   └── config.py                    # YAML config loader
│
├── configs/                         # Per-scenario YAML configs
│   ├── sim_piper.yaml               # Piper sim camera + GraspNet params
│   ├── sim_ur5.yaml                 # UR5 sim camera + GraspNet params
│   ├── real_piper.yaml              # D435i intrinsics + data dirs
│   └── cv.yaml                      # GroundingDINO / SAM weight paths
│
├── manipulator_grasp/               # Robot envs, controllers, motion planning
│   ├── assets/                      # MJCF/URDF models (Piper, UR5e, Robotiq, fruits)
│   ├── arm/                         # Controllers, motion planners, trajectory interpolation
│   ├── env/                         # MuJoCo Gym-style envs
│   ├── real/                        # Real-robot wrappers (piper_grasp_real)
│   └── utils/                       # MuJoCo / Robotics Toolbox helpers
│
├── graspnet-baseline/               # Git submodule: GraspNet CVPR 2020
├── weights/                         # Model weights (NOT in git — see setup below)
├── docs/screenshots/                # Demo images
├── mjmodel.mjb                      # Pre-compiled MuJoCo binary scene
├── requirements.txt
├── .env.example                     # Template for API keys / path overrides
└── LICENSE
```

---

## Environment Setup / 环境部署

### 1. System Requirements / 系统依赖

| 项目 | 要求 |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| GPU | CUDA-capable (CUDA 11.x recommended) |
| Python | 3.10 |
| MuJoCo | ≥ 2.3.4 |

### 2. Clone with Submodules / 克隆仓库（含子模块）

```bash
git clone --recursive https://github.com/<your-repo>/Piper_Mujoco_Sim.git
cd Piper_Mujoco_Sim
```

If you forgot `--recursive` / 如果忘记加 `--recursive`：

```bash
git submodule update --init --recursive
```

### 3. Create Conda Environment / 创建 Conda 环境

```bash
conda create -n piper_mujoco python=3.10 -y
conda activate piper_mujoco
pip install -r requirements.txt
```

> **Important / 重要**：`numpy==1.23.4` is pinned because `graspnet-baseline` requires this exact version. Installing a newer numpy will break GraspNet's CUDA ops.  
> `numpy==1.23.4` 被锁定版本，因为 GraspNet 子模块的 CUDA 扩展依赖该版本，请勿升级。

### 4. Build GraspNet CUDA Extensions / 编译 GraspNet CUDA 扩展

```bash
# PointNet++ grouping op
cd graspnet-baseline/pointnet2
python setup.py install
cd ../..

# KNN op
cd graspnet-baseline/knn
python setup.py install
cd ../..

# GraspNet Python API
pip install -e graspnet-baseline/graspnetAPI
```

### 5. Download Model Weights / 下载模型权重

`main_piperx_vision.py` needs GroundingDINO (checkpoint, config and BERT),
SAM-B and the GraspNet checkpoint. YOLO-World and CLIP are only for the legacy
YOLO-World pipeline. The weights are not stored in Git; place them under
`weights/`.

`main_piperx_vision.py` 需要 GroundingDINO、SAM-B 和 GraspNet；YOLO-World
与 CLIP 仅供旧 YOLO-World 流程使用。权重不进入 Git，请放到 `weights/`。

| 文件 / File | 版本与官方来源 / Version and source | SHA256 |
|---|---|---|
| `logs/log_rs/checkpoint-rs.tar` | [GraspNet official RealSense checkpoint](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view) (no release tag) | `60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868` |
| `groundingdino_swint_ogc.pth` | [GroundingDINO `v0.1.0-alpha`](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| `groundingdino_swint_ogc.py` | [GroundingDINO config at `856dde2`](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/config/GroundingDINO_SwinT_OGC.py), patched to use local BERT | `9707627e89288382c452f963238caa26841488a8bb300654e1dee7c232c1291d` |
| `sam_b.pt` | [Ultralytics SAM-B `v8.2.0`](https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt) | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `bert-base-uncased/` | [Hugging Face `google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594), revision `86b5e09` | `model.safetensors`: `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` |
| `yolov8s-world.pt` (legacy) | [Ultralytics YOLO-World small `v8.2.0`](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt) | `095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf` |
| `clip/ViT-B-32.pt` (legacy) | OpenAI CLIP `ViT-B/32`, source commit [`d05afc4`](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6) | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

Expected layout / 权重目录结构：

```
weights/
├── logs/log_rs/checkpoint-rs.tar
├── groundingdino_swint_ogc.pth
├── groundingdino_swint_ogc.py
├── sam_b.pt
├── bert-base-uncased/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── yolov8s-world.pt              # legacy
└── clip/ViT-B-32.pt              # legacy
```

Download the required files and verify them:

```bash
mkdir -p weights/logs/log_rs weights/bert-base-uncased weights/clip

# GroundingDINO checkpoint and config
curl -L --fail --retry 3 -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl -L --fail --retry 3 -o weights/groundingdino_swint_ogc.py \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/config/GroundingDINO_SwinT_OGC.py
sed -i 's|text_encoder_type = "bert-base-uncased"|text_encoder_type = "weights/bert-base-uncased"|' \
  weights/groundingdino_swint_ogc.py

# BERT base uncased at a fixed Hugging Face revision
BERT_REV=86b5e0934494bd15c9632b12f734a8a67f723594
for file in config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt; do
  curl -L --fail --retry 3 -o "weights/bert-base-uncased/$file" \
    "https://huggingface.co/google-bert/bert-base-uncased/resolve/$BERT_REV/$file"
done

# SAM-B
curl -L --fail --retry 3 -o weights/sam_b.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt

# GraspNet official RealSense checkpoint
python -m pip install gdown
gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O weights/logs/log_rs/checkpoint-rs.tar

# Legacy YOLO-World pipeline (optional)
curl -L --fail --retry 3 -o weights/yolov8s-world.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt
curl -L --fail --retry 3 -o weights/clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

```bash
sha256sum -c <<'EOF'
60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868  weights/logs/log_rs/checkpoint-rs.tar
3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  weights/groundingdino_swint_ogc.pth
9707627e89288382c452f963238caa26841488a8bb300654e1dee7c232c1291d  weights/groundingdino_swint_ogc.py
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912  weights/sam_b.pt
68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3  weights/bert-base-uncased/model.safetensors
EOF
```

Only if the legacy YOLO-World weights were downloaded:

```bash
sha256sum -c <<'EOF'
095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf  weights/yolov8s-world.pt
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af  weights/clip/ViT-B-32.pt
EOF
```

### 6. Configure Environment Variables / 配置环境变量

```bash
cp .env.example .env
# 用编辑器打开 .env，填入 API key
```

`.env` 内容示例：

```bash
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx   # Qwen (DashScope)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx    # DeepSeek (optional)
# WEIGHTS_DIR=/path/to/weights             # 可选：覆盖 weights/ 路径
# DATA_DIR=/path/to/data                   # 可选：覆盖数据保存路径
```

> **Security Note / 安全提示**：`.env` 已在 `.gitignore` 中，不会被提交。若你的 git 历史中曾有明文 API key，请立即前往 [DashScope 控制台](https://dashscope.console.aliyun.com/) 和 [DeepSeek Platform](https://platform.deepseek.com/) 吊销并轮换密钥。

### 7. Real Robot Extra Setup / 真机额外配置

```bash
# Intel RealSense SDK Python bindings
pip install pyrealsense2

# Agilex Piper SDK (install from official source)
# https://github.com/agilexrobotics/piper_sdk
pip install piper_sdk
```

Configure the CAN interface for Piper according to the [piper_sdk documentation](https://github.com/agilexrobotics/piper_sdk).  
按照 piper_sdk 文档配置 CAN 接口后方可控制真实机械臂。

---

## Quick Start / 快速开始

All scripts are run from the **repository root** / 所有脚本均在**仓库根目录**下运行。

### Piper X MuJoCo Simulation / Piper X 仿真

```bash
conda activate piper_mujoco
python main_piperx.py --hold
```

使用官方 Piper X + AGX 平行夹爪模型，完成瓶子的接近、抓取、搬运和放置；增加 `--object cube` 可切回原红色方块。
基础演示只依赖 NumPy 和 MuJoCo，使用物体已知坐标，不需要视觉权重。
无窗口验收可运行 `python main_piperx.py --headless`。

已安装视觉依赖和权重时，可直接运行视觉引导抓取：

```bash
python main_piperx_vision.py --hold
# 切回红方块：python main_piperx_vision.py --object cube --hold
# 抓取前停留 10 秒查看位姿：增加 --grasp-preview-seconds 10
# 无窗口：增加 --headless --fast，去掉 --hold
```

视觉入口默认抓取蓝色瓶子，检测提示词随物体选项自动切换。相机直接安装在夹爪横向支架前侧；窗口右上角显示实时腕部 RGB 和深度画面，增加 `--no-camera-preview` 可隐藏。程序使用拍摄时的外参转换抓取位姿，预检查整条轨迹、用当前触觉确认夹持，放置后重新拍摄确认；失败时在尝试次数内恢复并重新感知。

语义指令入口支持左、右、前、后、中间五种放置方向；其中 `right` 已完成完整抓放验收，其余方向目前使用固定世界坐标，尚需逐项验证。方向和落点的详细限制见 [Piper X 仿真说明](docs/piperx_sim.md#语义指令与放置方向)。

最终选中的抓取位姿会显示在三维场景和腕部 RGB 图中：红/绿/蓝为 TCP 的 X/Y/Z 轴，青色为夹爪轮廓，黄色为接近方向。窗口左下角显示世界坐标下的 XYZ、RPY 和开口；默认抓取前停留 2 秒，`--grasp-preview-seconds 0` 可取消停留。
模型生成方式、控制约定和视觉接入状态见 [Piper X 仿真说明](docs/piperx_sim.md)。
本轮接入过程、实现方法和验收结果汇总见 [PiperX 仿真视觉抓取工作总结](docs/piperx_project_summary.md)。

### UR5 MuJoCo Simulation / UR5 仿真

```bash
conda activate piper_mujoco
python main_yoloWorld_sam.py
```

Opens MuJoCo viewer → prompts for object name → detects, segments, plans grasp → executes.  
打开 MuJoCo viewer → 输入目标物体名称 → 检测分割 → 规划并执行抓取。

### Piper MuJoCo Simulation / Piper 仿真

```bash
python main_piper_yoloWorld_sam.py
```

Same pipeline with Piper arm kinematics (Pinocchio IK).  
相同流程，使用 Piper 机械臂运动学（Pinocchio IK）。

### Real Piper + RealSense / 真实 Piper + D435i

```bash
python main_real_piper_yoloWorld_sam.py
```

Connects to D435i, streams frames, runs full pipeline on live images.  
连接 D435i，实时采集图像，运行完整抓取流水线。

### LLM-Driven Task Planning / LLM 任务规划

```bash
python LLM_task_plan.py
```

Requires `DASHSCOPE_API_KEY` in `.env`. Enter commands like "抓取苹果放到右边".  
需要 `.env` 中的 `DASHSCOPE_API_KEY`。输入指令如"抓取苹果放到右边"。

---

## Script Reference / 脚本说明

### `cv_process.py`

**功能**：目标检测（GroundingDINO text-prompt）+ 实例分割（SAM）。  
**主入口**：`segment_image(image, output_mask='mask1.png') -> np.ndarray`  
**配置**：`configs/cv.yaml`（权重路径，可通过 `WEIGHTS_DIR` 环境变量覆盖）  
**硬件**：强烈建议使用 GPU (CUDA)

### `main_yoloWorld_sam.py`

**功能**：UR5 MuJoCo 仿真完整抓取循环。  
**流程**：渲染 → GroundingDINO+SAM → GraspNet → 轨迹规划 → MuJoCo step  
**配置**：`configs/sim_ur5.yaml`

### `main_piper_yoloWorld_sam.py`

**功能**：Piper MuJoCo 仿真完整抓取。  
**流程**：与 UR5 相同，但使用 Pinocchio IK 计算 Piper 关节角度。  
**配置**：`configs/sim_piper.yaml`

### `main_real_piper_yoloWorld_sam.py`

**功能**：真实 Piper 机械臂 + Intel RealSense D435i 部署。  
**流程**：D435i 采集 → 分割 → GraspNet → Piper SDK 执行  
**配置**：`configs/real_piper.yaml`（D435i 内参、数据目录）  
**硬件**：Agilex Piper 机械臂 + Intel RealSense D435i + CAN 接口

### `LLM_task_plan.py`

**功能**：基于 Qwen / DeepSeek 的 LLM 任务规划器。  
**接口**：终端交互循环。`TerminalInterface` 发送指令至 LLM，`SubTaskExecutor` 将 LLM 输出映射为机器人原语（`back_zero`、`gripper_control`、`get_frame`、`Find_obj_Segment`、`Grasp_Pose_Inference`、`move_to_grasp`、`move_to_place` 等）。  
**配置**：`.env`（API keys）+ `configs/real_piper.yaml`（数据目录）

### `main.py`

**功能**：UR5 原始抓取流水线（基于 `env.render()` 返回 dict 的旧接口）。  
**备注**：保留供参考，建议使用 `main_yoloWorld_sam.py`。

---

## Configuration Reference / 配置说明

### `configs/sim_piper.yaml` / `configs/sim_ur5.yaml`

```yaml
camera:
  width: 640
  height: 640
  fovy: 0.785   # pi/4，MuJoCo 相机垂直视场角（弧度）
  scale: 1.0    # 深度因子（仿真直接用米）

graspnet:
  checkpoint: logs/log_rs/checkpoint-rs.tar
  num_points: 3000       # 点云采样数量
  collision_thresh: 0.05 # 碰撞检测阈值
  voxel_size: 0.01
  depth_trunc: 2.0       # 深度截断距离（超过此值的点被丢弃，单位：米）
```

### `configs/real_piper.yaml`

```yaml
camera:
  width: 640
  height: 480
  fx: 386.11     # D435i 深度相机焦距
  fy: 386.11
  ppx: 321.10    # 主点 x
  ppy: 243.83    # 主点 y
  scale: 1000.0  # mm → m 转换因子

graspnet:
  checkpoint: weights/logs/log_rs/checkpoint-rs.tar
  num_points: 10000
  collision_thresh: 0.01
  voxel_size: 0.01

image_dirs:          # 数据保存路径（相对于 DATA_DIR）
  rgb: image/rgb
  depth: image/depth
  depth_map: image/depth_map
  mask: image/mask
```

### `configs/cv.yaml`

```yaml
weights_dir: weights   # 可通过 WEIGHTS_DIR 环境变量覆盖
models:
  groundingdino_config: weights/groundingdino_swint_ogc.py
  groundingdino_checkpoint: weights/groundingdino_swint_ogc.pth
  sam_checkpoint: weights/sam_b.pt
detection:
  box_threshold: 0.25
  text_threshold: 0.25
```

---

## Troubleshooting / 常见问题

### MuJoCo NaN/Inf warnings

仿真初始化时可能出现 Inf/NaN 接触力，可尝试：
- 在场景 XML (`manipulator_grasp/assets/agilex_piper/scene.xml`) 中增大 `timestep`。
- 减小被抓物体质量。
- 在抓取前先跑若干 warm-up `env.step()` 步。

### numpy version conflict / numpy 版本冲突

```bash
pip install "numpy==1.23.4"
cd graspnet-baseline/pointnet2 && python setup.py install
cd ../knn && python setup.py install
```

### `DASHSCOPE_API_KEY is not set`

```bash
cp .env.example .env   # 然后填入密钥
```

### GroundingDINO / SAM weight not found

```bash
export WEIGHTS_DIR=/your/weights/dir
python main_piper_yoloWorld_sam.py
```

### pointnet2 / knn compile error

确认 PyTorch 的 CUDA 版本与本机 CUDA toolkit 匹配：

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

版本不匹配时需先重装对应版本的 PyTorch。

### RealSense D435i not detected

```bash
rs-enumerate-devices   # 检查设备是否被系统识别
```

---

## Acknowledgements / 致谢

- [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline) — CVPR 2020 grasp pose detection
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — Open-set object detection with language
- [Ultralytics SAM](https://github.com/ultralytics/ultralytics) — Segment Anything integration
- [Agilex Piper](https://github.com/agilexrobotics/piper_sdk) — Desktop robotic arm SDK
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — UR5e / Robotiq MJCF models
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) — Rigid body dynamics and IK library

---

## License / 许可证

See [LICENSE](LICENSE).
