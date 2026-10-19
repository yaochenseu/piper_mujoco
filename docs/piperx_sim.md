# Piper X 仿真

本入口从 `external/agx_arm_urdf` 中的官方 Piper X + AGX 平行夹爪模型生成 MuJoCo 模型，默认对场景中的 YCB 物体执行接近、闭爪、提起、搬运、释放和撤离。`main_piperx.py --target` 接受任意场景 body 名，默认 `ycb_banana`。

当前默认演示使用仿真物体的已知中心计算抓取位姿，不调用目标识别或 GraspNet。物体具有自由关节，由两侧手指接触和摩擦夹持；没有物体绑定约束，也没有在抓取过程中直接修改物体位置。

## 运行

```bash
conda activate piper_mujoco
python main_piperx.py --hold
```

`--hold` 使窗口在完成动作后保持打开，不加则执行完退出。关闭窗口或按 Ctrl+C 可以停止。

无窗口验收：

```bash
python main_piperx.py --headless
python -m unittest discover -s tests -p 'test_piperx_sim.py' -v
```

程序检查双指接触、提起高度和最终放置误差。失败时抛出异常并返回非零退出状态。目标放置 XY 为世界坐标 `(0.42, 0.16)`，最终高度由物体形状及桌面接触决定，具体结果以运行报告为准。

只查看模型和场景：

```bash
python -m mujoco.viewer --mjcf=manipulator_grasp/assets/agilex_piper_x/scene.xml
```

`mujoco.viewer` 只加载模型，不会运行抓取控制器和重力前馈。请用 `main_piperx.py` 查看完整动作。

### 视觉抓取

视觉入口为 `main_piperx_vision.py`，默认使用末端随动的 `wrist` RGB-D 相机；控制逻辑分离在 `utils/piperx_vision_control.py`：

```bash
python main_piperx_vision.py --target "banana ." --hold      # 任意 DINO 提示词
python main_piperx_vision.py --headless --fast
python main_piperx_vision.py --target ycb_tennis_ball --obj-x 0.30 --obj-y 0.13
python main_piperx_vision.py --instruction "把草莓放到右边" --hold
python main_piperx_vision.py --test-release-first            # 故障恢复演练
python main_piperx_vision.py --no-camera-preview --hold     # 隐藏相机画面
```

场景清单定义在 `configs/piperx_scene.yaml`，包含五个 YCB 物体（banana、strawberry、pear、tennis_ball、tomato_soup_can）的 body 名、源模型、质量、初始 XY 和偏航角。构建期参数由 `scripts/prepare_ycb_objects.py` 读取该清单并生成 `manipulator_grasp/assets/agilex_piper_x/ycb_objects.xml`；运行时的抓取目标由 `--target`/`--text` 的 DINO 提示词和分割点云匹配决定，不依赖构建清单。

当前入口仍执行一次指定目标的抓放，尚未实现多个物体逐个抓取。五个物体均为独立自由刚体，受重力、碰撞和摩擦影响，腕部相机初始画面可见；相邻物体距离约 0.1 m，点云质心匹配阈值为 0.06 m。

GraspNet 从 SAM 目标点云提供候选评分及夹爪开合方向，随后进行去重、点云碰撞筛选和完整路径检查。碰撞代理的接触平面采用 AGX TCP 平面（插入深度为零）；原始候选保存在 `grasps_raw.npy`。抓取策略不读取目标的注册身份或仿真真值，只使用分割点云、触觉接触和 TCP 位姿。

- 视觉定位采用 DINO 文本检测 → SAM 分割 → RGB-D 点云 → GraspNet 候选。运动前预检查接近、下降、提起、搬运、下放、撤离、返回观察位七段轨迹，连续插值姿态，逐个采样检查逆解、关节跳变和机械臂与自身/桌面的碰撞。
- 抓取候选优先使用 GraspNet；仅当六自由度候选都没有可达、无碰撞路径时，才用 SAM 目标点云的稳健包围盒中心生成竖直顶抓候选作为兜底。该兜底不读取物体真值，也不为单个物体手调候选。
- 抓取使用连续姿态到达目标；夹持是否成立由当前触觉（接触）确认，而不是只看规划值。
- 放置后返回观察位，再采集 RGB-D 确认目标在放置区域。允许重试时，失败会触发恢复、重新拍摄和重新生成候选；恢复失败会停止并记录原因，不继续执行旧计划。
- 物体真值仅用于独立验收，不参与抓取决策。
- 默认最多 3 次新感知尝试（`--max-attempts` 取 1..5）；每次感知最多对 12 个通过几何筛选的候选做预检查。
- 诊断输出位于 `docs/vision_runs/<timestamp>/`：`cycle_00/pickup/`、`cycle_00/placement/`、`cycle_00/grasps.json` 和 `result.json`。

#### 语义指令与放置方向

视觉入口支持通过 `--instruction` 接收自然语言抓放指令。语义规划器先输出目标英文短语、目标空间选择（`leftmost`、`rightmost`、`closest`、`farthest` 或 `none`）、动作和放置方向；其中目标短语再交给 GroundingDINO，方向只用于选择下面的固定世界坐标落点，不会让相机重新判断桌面空位。

| 方向 | 中文示例 | 固定落点 `(x, y)`，米 | 当前状态 |
| --- | --- | --- | --- |
| `left` | 左边 | `(0.36, 0.28)` | 已解析；未完成当前场景的独立抓放验收 |
| `right` | 右边 | `(0.22, -0.34)` | 已解析并完成抓放验收 |
| `front` | 前面 | `(0.40, 0.24)` | 已解析；固定落点尚未验证 |
| `back` | 后面 | `(0.30, 0.24)` | 已解析；固定落点尚未验证 |
| `center` | 中间、中央 | `(0.42, 0.16)` | 已解析；固定落点尚未单独验收 |

因此，当前接口声明支持五个方向，但实际可靠支持仍以 `right` 为主。`front` 与 `back` 目前只相差 0.10 m，x 坐标不同而 y 坐标相同，不能据此认为它们已经在场景中形成符合直觉的“前/后”落点。若要支持任意方向，应根据腕部 RGB-D 检测桌面可用区域、避开物体和机械臂可达范围后生成落点，并为每个方向补充完整抓放验收，而不是继续扩充固定坐标表。

中文指令中的目标侧别和目的地方向分开解析。例如“把左边的香蕉放到右边”会得到目标短语“香蕉”、目标选择 `leftmost` 和目的地 `right`；VLM 显式返回合法方向时以 VLM 结果为准，规则后端或 VLM 未给出方向时取指令中最后出现的方向词，避免把“左边的”误当成放置方向。指令仍无方向时默认为 `center`。

语义层是开放词汇的：`--instruction` 由 VLM 自由生成 GroundingDINO 提示词，不要求目标出现在固定名单中。首次分割完成后，`main_piperx_vision.py` 用分割点云的 XY 质心与场景中所有 free joint body 的 `xipos` 做最近邻匹配（阈值 0.06 m），匹配结果写入 `target_body.json` 并作为仿真验收目标；最近距离超过阈值时报错并列出全部候选 body。`--target` 现在是自由的 DINO 提示词（例如 `teacup`），不再是注册表键；只有与 `--obj-x/--obj-y/--obj-yaw` 一起使用时才要求它是场景 body 名（如 `ycb_banana`），用于感知前摆位。场景物体清单位于 `configs/piperx_scene.yaml`。

`--semantic-backend` 默认 `openai`，必须配置 `PIPERX_VLM_BASE_URL` 和 `PIPERX_VLM_MODEL`，否则启动时给出明确报错。显式传 `rules` 使用开放词汇规则解析，或传 `auto` 在 VLM 失败时回退到规则；不提供 `--instruction` 时可用 `--text` 直接给出 DINO 提示词。

可以手工指定落点，命令行参数优先于方向表：

```bash
python main_piperx_vision.py --instruction "把香蕉放到右边" --hold
python main_piperx_vision.py --instruction "把草莓放到左边" --semantic-backend rules --hold
python main_piperx_vision.py --text "tennis ball ." --target ycb_tennis_ball \
    --obj-x 0.30 --obj-y 0.13 --obj-yaw 0.3 --hold
```

使用 `openai` 后端时，需要设置 `PIPERX_VLM_BASE_URL`、`PIPERX_VLM_MODEL` 和 `PIPERX_VLM_API_KEY`，模型需支持图像输入；`--drop-x` 和 `--drop-y` 可以分别覆盖单个坐标。`semantic_plan.json` 记录解析出的目标、空间选择、方向和 DINO 提示词，`target_body.json` 记录点云质心、每个候选 body 的距离和最终匹配结果，最终落点写入运行结果 `result.json` 并打印到日志，便于区分语义解析结果与固定坐标表。

### 腕部 RGB-D 相机

相机通过短安装板直接固定在两根夹指上方的横向支架前侧，位于横梁中间，不使用伸向腕部上方的长支杆。它属于 `gripper_base` 的子体 `wrist_camera_mount`，随腕部六自由度运动。相机壳体中心相对夹爪基座为 `(-0.055, 0, 0.048)` 米；壳体宽度与横梁平行，安装倾角为 0°，两侧夹指仍可开合。镜头沿夹爪基座 +Z 轴朝向夹指前方。摆正后将仿真相机的垂直视场角调整为 120°，以覆盖抓取和放置区域，视觉入口分辨率仍为 1280×960。初始观察姿态下镜头世界位置约为 `(0.375, -0.120, 0.956)` 米；运动时该位置和朝向都会变化。

模型包含可见机壳、双镜头和连接支架。它们是零质量、无碰撞的显示几何，尚未加入实际相机载荷。原桌面上方的固定 `top` 相机已移除；场景仍有 `overview` 观察相机，交互窗口默认使用自由视角，抓取算法读取 `wrist`。

视觉入口的窗口右上角嵌入两幅实时图：左侧腕部 RGB，右侧深度伪彩色图（固定 0.05–1.00 m 色标，无效深度为黑色）。预览采用 320×240、按仿真时间最多 10 帧/秒更新，并随窗口缩放；用于识别的图像仍为 1280×960。运动时更新预览，模型推理和规划暂停仿真期间显示上一帧。`--headless` 不创建预览；`--no-camera-preview` 仅关闭显示，不关闭视觉感知。预览在 `--hold` 阶段继续更新，并在退出时释放渲染资源。

每次拍摄都会把当前光学相机到世界的外参、内参、相机名和所属 body 写入同目录的 `camera.json`。点云反投影和 GraspNet 抓取转换使用该帧的外参快照，之后腕部运动不会改变旧帧的坐标解释。模型生成器也包含安装定义，重新运行 `scripts/build_piperx_model.py` 会保留腕部相机。

夹持判断采用持续的当前双指触觉、夹爪开口及 TCP 实际提起距离；没有把 TCP 位移当作物体真实抬升高度。真实物体的抬升高度在策略结束后单独验收。

当前限制：每次抓取一个 YCB 物体，桌面范围、高度筛选和几何碰撞参数仍针对该场景；逆解与路径检查采用离散采样，没有通用全局避障；每次感知后分段执行动作，并在放置后和失败恢复后更新图像，不是连续视觉伺服。不能据此保证任意物体或任意摆放均能抓取。

### 抓取位姿显示

每次通过完整路径检查后，窗口会显示最终选中的拾取 TCP 位姿：红色 X 轴、绿色 Y 轴（夹爪开合方向）、蓝色 Z 轴（接近方向）、青色夹爪轮廓及黄色接近箭头。左下角列出世界坐标 XYZ（米）、RPY（度，约定 `R = Rz(yaw) Ry(pitch) Rx(roll)`）、检查开口和候选序号；四元数 `wxyz` 同时写入诊断 JSON。

这些标记固定在选中的世界坐标抓取位置，物体搬走后仍表示原拾取位姿。重试重新感知时清除旧标记，选出新位姿后替换。三维标记只存在于 MuJoCo 显示场景，不加入物理模型；腕部 RGB 中的标记根据每次预览时的相机外参重新投影，不参与检测或点云生成。

默认执行前停留 2 秒，方便查看。停留期间暂停物理步进，窗口仍可旋转观察；无窗口时不等待：

```bash
python main_piperx_vision.py --hold --grasp-preview-seconds 10
# 不停留：--grasp-preview-seconds 0
```

每轮 `cycle_00/` 等目录新增 `grasp_pose.json`、`selected_pose_rgb.png`、`selected_pose_3d.png` 和 `selected_pose_closeup.png`，分别记录数值、曝光帧上的投影、三维全景和位姿近景。无窗口运行也会保存这些诊断文件。

### 视觉抓取验收

历史瓶子/红方块场景曾在 0° 安装、120° 视场配置下完成四组视觉抓放验收：默认瓶子、移动位置 `(0.30, 0.08)` 的瓶子、带实时相机画面的实际窗口，以及切回原红方块；当前 YCB 场景尚未重新做端到端视觉验收（本机缺少视觉权重）。四组均在第 0 轮成功，没有触发恢复；物理抬升约 0.099–0.100 m，放置误差分别约 0.76、0.64、0.76、4.34 mm。均无记录到的机械臂自身/桌面接触或 MuJoCo 警告。

实际窗口运行正常退出，右上角 RGB/深度预览更新了 178 帧。另一次运动探针确认相机图像随末端移动而变化。此前方块的旋转和故障恢复历史诊断仍保存在 `docs/vision_runs/gripper_support_*/`，不作为本次瓶子验收结果。

抓取位姿显示另通过了无窗口诊断和实际窗口验收（窗口中抓取前停留 5 秒）。两次均成功抓放，导出的世界位姿与执行所选位姿一致；记录位于 `docs/vision_runs/grasp_pose_default/`、`grasp_pose_gui/`，后者包含带数值的位姿近景图。

汇总见 [piperx_grasp_validation.json](piperx_grasp_validation.json)。最新图像、原始 GraspNet 候选和瓶身拟合结果位于本地 `docs/vision_runs/bottle_body_default/`、`bottle_body_moved/`、`bottle_body_gui/`、`cube_regression/`。窗口预览图可查看 `docs/vision_runs/bottle_body_gui/camera_preview.png`。诊断目录不加入 Git。

包含原有六项检查在内的 29 项仿真/控制/显示测试已全部通过：

```bash
python -m unittest discover -s tests -p 'test_piperx_*.py' -v
```

测试覆盖完整路径规划不改变仿真状态、过期计划拒绝、最短弧姿态插值、松爪后当前接触清除、控制策略禁止读取物体真值、失败后重新感知和规划，以及腕部相机随动、运动后使用旧帧外参转换、初始视野覆盖、瓶子接触抓放、部分圆柱面中心拟合、预览刷新/缩放/退出和无效深度显示。显示检查覆盖坐标轴和开口的世界变换、RGB 投影不修改源图、重试替换/清除标记，以及标记不修改物理状态。

## 依赖和来源

基础演示只需要 NumPy 和 MuJoCo，已在 `piper_mujoco` 的 Python 3.10.22、MuJoCo 3.16.0 上验证。初始使用 NumPy 2.2.6；按用户要求安装视觉依赖后改为 NumPy 1.23.4，六项仿真回归测试仍全部通过。实际 MuJoCo 窗口中的抓取也已通过，并修正了退出时等待渲染线程完成清理的流程。

模型来源：<https://github.com/agilexrobotics/agx_arm_urdf>，当前接入时检查的提交为 `680fa925cd00db7e2628563d2edc5540963045de`。外部仓库保留原样，模型仍依赖其中的 STL 网格，因此请保留 `external/agx_arm_urdf` 目录。模型资源沿用上游 MIT 许可证。

更新来源或重新生成：

```bash
# 首次缺少 external/agx_arm_urdf 时：
git clone https://github.com/agilexrobotics/agx_arm_urdf.git external/agx_arm_urdf
python scripts/build_piperx_model.py
```

生成器读取本体 URDF 和带夹爪的 include-only Xacro，输出 `manipulator_grasp/assets/agilex_piper_x/piperx.xml` 和 `piperx.urdf`，使用相对网格路径，不需要 ROS 或 xacro 包。生成器仅针对当前上游文件结构；若上游加入参数化 Xacro 宏，需要进一步适配。

## 模型和控制约定

- 机械臂采用官方的六关节位置、转轴、限位、质量和惯量，碰撞几何使用上游 STL 的凸包。视觉使用同一坐标系下的 STL，未保留 DAE 材质。
- 上游抽象的 `gripper` 开口指令不作为额外物理关节。保留两个真实滑动指关节，通过 equality joint 约束满足 `q_right = -q_left`。
- `gripper_control(width)` 的参数是总开口，范围 `[0, 0.10]` 米。手指位移各为开口的一半。实际内侧间距还包含添加的薄接触垫厚度。
- 夹爪位置指令使用 1.5 s 的 S 曲线插值后保持 0.1 s，避免位置伺服一步闭合造成冲击；该时间可通过传给 `gripper_control(..., duration=...)` 调整。
- `grasp_link` / `tcp` 位于夹爪基座局部 `(0, 0, 0.125)` 米，处于两指接触垫之间；TCP +Z 是接近方向，Y 是开合方向。该点表示夹持接触参考，不建模腕部触觉片的厚度。
- GraspNet 的平移通常落在可见表面。视觉抓取默认沿候选接近轴再插入 15 mm，使指垫中心进入物体；该偏置可按 `--grasp-insertion-depth` 调整，并按指端与桌面的 2 mm 余量自动截断。它是抓取深度参数，不改变机器人 TCP、相机外参或真机触觉坐标。
- 机械臂通过位置执行器、重力/Coriolis 前馈和时间插值驱动，夹爪通过 tendon 控制。逆解使用 MuJoCo 的 TCP 雅可比和阻尼最小二乘，不依赖 Pinocchio/CasADi。
- 基座在世界坐标 `(0, 0, 0.74)` 米，桌面高 0.74 米。相机参数和变换从实际模型读取，不复用旧 Piper 的标定常量。
- 保留非相邻连杆之间的碰撞，排除机械连接处和左右手指之间的碰撞。演示轨迹已检查无机械臂与桌面、自身之间的接触。
- 当前逆解为局部求解，笛卡尔路径和最短弧姿态插值会在运动前逐点检查，缓存计划在执行前还会检查起始关节是否过期；没有完整的全局避障规划。任意摆放、物体形状、夹爪姿态都需要重新验证。关节限位使当前垂直抓取可用高度有限，演示抬升采用 0.10 米。

## 接入 RGB-D / GraspNet

`PiperXGraspEnv` 已提供：

```python
frames = env.render(camera="wrist", width=1280, height=960)
camera = env.camera_parameters("wrist", 1280, 960)  # 在机械臂下一次运动前保存
T_world_tcp = env.camera_grasp_to_world(
    grasp.translation, grasp.rotation_matrix,
    camera="wrist", T_world_camera=camera["T_world_camera"],
)
env.move_to(T_world_tcp[:3, 3], T_world_tcp[:3, :3])
```

GraspNet 的接近轴是 +X，TCP 的接近轴是 +Z；转换函数同时处理 MuJoCo 相机与光学相机之间 Y/Z 轴方向的差异。腕部相机必须使用拍摄时的外参，不应在机械臂移动后用当前外参解释旧图像。不能直接套用旧 Piper 中写死的姿态修正。

视觉依赖已安装在 `/home/yzc/miniconda3/envs/piper_mujoco`。已在 RTX 4060 Ti 上验证 PyTorch GPU 运算、PointNet++ 采样/聚集、KNN、GroundingDINO 可变形注意力 GPU 算子、GraspNet API/NMS，以及无预训练权重的 GraspNet GPU 前向计算。`pip check` 无依赖冲突，验证报告见 [piperx_vision_validation.json](piperx_vision_validation.json)。

`main_piperx.py` 的基础演示仍使用物体已知坐标；视觉识别和抓取候选选择已通过独立入口 `main_piperx_vision.py` 接入 PiperX 动作流程（见前文“视觉抓取”）。视觉权重的存放位置和验收方式见下文。旧入口另需 Pinocchio（含 CasADi 绑定）、CasADi、spatialmath 等，新的基础仿真不需要这些包；本次未安装真机和 LLM 依赖。

GraspNet 子项目声明 `numpy==1.23.4`，因此本次保持该版本。系统 `/usr/bin/nvcc` 原为 11.5，本次在 Conda 环境内安装了 CUDA 11.8 编译器、运行库和 cuBLAS/cuSPARSE/cuSOLVER 开发包。没有修改系统 NVIDIA 驱动。

重新验收依赖（无需权重）：

```bash
python scripts/check_piperx_vision.py
python -m pip check
```

## 已验证的视觉依赖安装方案

以下组合已在 Python 3.10 / RTX 4060 Ti / 驱动 570.211.01 上验证：PyTorch 2.0.1+cu118 / torchvision 0.15.2+cu118 / CUDA 11.8 / NumPy 1.23.4。PyTorch 安装命令来自 [官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v201)，CUDA 编译工具安装方式来自 [NVIDIA Conda 安装说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html#conda-installation)。以下命令供检查或重建环境使用，当前环境无需重复安装。

前提是本机 `nvidia-smi` 正常，显卡支持 CUDA 11.8，且驱动足以运行该版本。若 `nvidia-smi` 也无法连接驱动，先解决驱动问题；安装 Python 包和 Conda CUDA 工具不会修复系统 NVIDIA 驱动。新于 CUDA 11.8 支持范围的显卡应另选 PyTorch/CUDA 版本。

```bash
conda activate piper_mujoco
cd /media/yzc/F/DOGcode/others/Piper_Mujoco_Sim
nvidia-smi
```

每一步先执行检查。已存在且版本符合本次验证方案时可以跳过对应安装；若已有不同版本，先确认兼容性，不要直接重复安装覆盖。`pip show` 检查安装记录，实际 `import` 检查模块能否加载，两者都需要关注。

1. 检查 GPU 版 PyTorch / torchvision：

```bash
python -m pip show torch torchvision
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

本次验证版本为 `torch 2.0.1+cu118`、`torchvision 0.15.2+cu118`。缺少或确认需要调整时安装：

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

2. 检查视觉依赖文件中的每个包及其已安装版本（只查询，不安装）：

```bash
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

for line in Path('requirements-piperx-vision.txt').read_text().splitlines():
    spec = line.strip()
    if not spec or spec.startswith('#'):
        continue
    name = spec.split('==')[0]
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = 'MISSING'
    print(f'{name:20s} installed={installed:18s} requested={spec}')
PY
```

需要安装时：

```bash
python -m pip install -r requirements-piperx-vision.txt
```

确认 GPU 能实际计算，再进行扩展编译；下面的验证应输出版本 `2.0.1+cu118`、CUDA `11.8` 和 GPU 名称，且没有异常：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda') + 1)"
```

3. 检查 CUDA 编译工具是否存在、版本和位置：

```bash
command -v nvcc
nvcc --version
conda list '^(cuda|libcublas|libcusparse|libcusolver)'
```

本机初始系统 `nvcc` 为 11.5，编译时用 Conda 中的 11.8 工具链以保持一致。已有可用的 11.8 工具链时无需再安装；否则以下安装只针对激活的 Conda 环境，不修改系统驱动：

```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8 libcublas-dev libcusparse-dev libcusolver-dev
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

应确认此处 `nvcc` 输出 11.8，然后逐项检查和安装 GroundingDINO、GraspNet API 和扩展。约束文件防止依赖安装再次升级 NumPy 或 PyTorch。

4. GroundingDINO：

```bash
python -m pip show groundingdino
python -c "import torch; import groundingdino; import groundingdino._C; print('GroundingDINO:', groundingdino.__file__); print('CUDA extension: OK')"
```

缺少或无法加载时再检查失败原因并安装：

```bash
# 使用官方源码。groundingdino-py 0.4.0 的 PyPI 源码关闭了扩展构建。
# 缺少源码目录时才执行 clone。本次构建的官方源码提交为
# 856dde20aee659246248e20734ef9ba5214f5e44。
git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git external/GroundingDINO
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install --no-build-isolation -c requirements-piperx-vision.txt ./external/GroundingDINO
```

5. GraspNet API：

```bash
python -m pip show graspnetAPI
python -c "import graspnetAPI; from graspnetAPI import GraspGroup; print('GraspNet API:', graspnetAPI.__file__)"
```

需要安装时：

```bash
python -m pip install --no-build-isolation -c requirements-piperx-vision.txt -e ./graspnet-baseline/graspnetAPI
```

6. PointNet++ 本地扩展：

```bash
python -m pip show pointnet2
python -c "import torch; import pointnet2._ext as ext; print('PointNet++:', ext.__file__)"
```

需要安装时：

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install --no-build-isolation -c requirements-piperx-vision.txt ./graspnet-baseline/pointnet2
```

7. KNN 本地扩展：

```bash
python -m pip show knn-pytorch
python -c "import torch; import knn_pytorch.knn_pytorch as ext; print('KNN:', ext.__file__)"
```

需要安装时：

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install --no-build-isolation -c requirements-piperx-vision.txt ./graspnet-baseline/knn
python -m pip check
```

`TORCH_CUDA_ARCH_LIST=8.9` 针对本机 RTX 4060 Ti；其他架构应调整该值。本次 PyTorch 使用官方源，普通 Python 依赖使用清华 PyPI 镜像以加速下载；上面省略镜像参数时会使用 pip 默认源。

任一步失败时先保留完整日志，不要跳过错误继续。本方案不安装真机、LLM 和旧 Piper 运动学依赖。

## 本地视觉权重

本机已完成以下权重下载，共约 1.8 GiB。四组预训练模型 GPU 验收均通过，报告见 [piperx_weights_validation.json](piperx_weights_validation.json)。

请从项目根目录运行。权重目录如下，所有权重均保留在项目内，未放入 Conda 环境：

| 用途 | 本地路径 | 官方来源 |
| --- | --- | --- |
| 文本目标检测 | `weights/groundingdino_swint_ogc.pth` | [GroundingDINO SwinT](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| 检测配置 | `weights/groundingdino_swint_ogc.py` | 复制自 `external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| DINO 文本编码器 | `weights/bert-base-uncased/` | [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| 物体分割 | `weights/sam_b.pt` | [Ultralytics SAM-B](https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt) |
| 抓取姿态预测 | `weights/logs/log_rs/checkpoint-rs.tar` | [GraspNet 官方 RealSense checkpoint](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view) |
| 旧流程的 YOLO-World 检测 | `weights/yolov8s-world.pt` | [YOLO-World small](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt) |
| YOLO 自定义类别文本编码器 | `weights/clip/ViT-B-32.pt` | [官方 CLIP](https://github.com/openai/CLIP) 的 `ViT-B/32` 下载地址 |

`checkpoint-rs.tar` 是 PyTorch checkpoint，直接加载，无需解压。DINO 配置中的 `text_encoder_type` 改为 `"weights/bert-base-uncased"`，读取本地 BERT。`cv_process.py` 从配置的权重目录读取 YOLO 权重；目录下存在 `clip/ViT-B-32.pt` 时，设置自定义类别也使用本地 CLIP。

下载工具 `gdown` 和文本编码器包 `clip` 安装在 `piper_mujoco` 中。CLIP 源码保留于 `external/CLIP`，接入时提交为 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。重建环境时，先检查，再按需安装：

```bash
python -m pip show gdown clip ftfy
# 仅缺少时执行；CLIP 官方源码首次缺少时先 clone：
# git clone https://github.com/openai/CLIP.git external/CLIP
python -m pip install -c requirements-piperx-vision.txt gdown ./external/CLIP
```

验收预训练模型：

```bash
conda activate piper_mujoco
cd /media/yzc/F/DOGcode/others/Piper_Mujoco_Sim
python scripts/check_piperx_weights.py --output docs/piperx_weights_validation.json
python -m pip check
```

脚本禁止 Hugging Face 联网加载，并逐个运行 GPU 推理：DINO 在官方猫狗示例图上检测、SAM 以检测框生成掩码、YOLO-World 以本地 CLIP 设置猫狗类别并检测、GraspNet 严格加载预训练参数后对合成点云前向计算。报告记录文件大小和 SHA256，供后续检查文件变化；除 CLIP 下载器按官方 URL 中的哈希验证外，这些记录不是上游发布的独立校验值。该验收不等于 PiperX 视觉抓取闭环已完成。
