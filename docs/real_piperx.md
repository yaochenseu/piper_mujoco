# Piper X 真机视觉接入

本入口把仿真中的 DINO / SAM / GraspNet / VLM 策略复用到真实 Piper X。
视觉进程仍在 `piper_mujoco` Conda 环境运行；ROS 2 驱动、MoveIt 和 D435i
由独立的 `real_bridge/piperx_ros_bridge.py` 进程访问。

## 前置条件

- Piper X + AGX 夹爪，CAN 已配置为 `can0`
- ROS 2 Humble；`real_ws` 或 `~/agx_arm_ws` 已编译
- Intel RealSense D435i，ROS 2 `realsense2_camera`
- `piper_control/tools/calib_results/` 中有当前安装的手眼结果

本机已有 `pyAgxArm` 源码时可直接安装（只安装包本身，不改系统依赖）：

```bash
/usr/bin/python3 -m pip install --user --no-deps \
  /media/yzc/F/DOGcode/others/AGILE-ARM/pyAgxArm-master
```

该本地源码当前是 `4e5715a`；`piper_control` 的安装说明验证的是
`d90e234`。如果驱动启动时报 API/参数不匹配，先在该源码目录切到
`d90e234` 再安装（本机当前安装的是 `d90e234`）。

驱动必须使用 `tcp_offset=0` 启动，桥接层会把它转换成夹爪垫中心 TCP：

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 arm_type:=piper_x effector_type:=agx_gripper \
  speed_percent:=20 tcp_offset:='[0.0,0.0,0.0,0.0,0.0,0.0]'
```

不要同时用 MoveIt 的 `/control/joint_states` 控制流覆盖直连指令。若使用
MoveIt，请按 `piper_control` 的门控方式启动，并且不要用本桥接层直接发运动。

## 启动 D435i

视觉需要彩色图和与之对齐的深度图：

```bash
export ROS_DOMAIN_ID=42
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=1280x720x30 \
  depth_module.depth_profile:=1280x720x30 \
  align_depth.enable:=true
```

话题默认值在 `configs/real_piperx.yaml`：

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/aligned_depth_to_color/camera_info
```

## 构建 ROS 工作区

如果还没有 `~/agx_arm_ws`，在本仓库内构建本地工作区：

```bash
cd /media/yzc/F/DOGcode/my/piper/piper_mujoco
scripts/setup_real_workspace.sh
```

它会创建 `real_ws/src/agx_arm_ros` 和 `real_ws/install`，桥接脚本会自动 source。
还需要按 `piper_control/README.md` 安装 `pyAgxArm` 和 ROS 2 依赖。

## 启动桥接

```bash
cd /media/yzc/F/DOGcode/my/piper/piper_mujoco
scripts/piperx_real_bridge.sh
```

桥接只提供状态、相机、夹爪、门控、急停和 TCP 运动命令。默认 `--execute`
关闭时，所有运动命令都是 dry-run。

## 先只验证视觉

```bash
scripts/run_real_piperx_vision.sh \
  --perception-only \
  --instruction "抓取香蕉" \
  --diagnostics-dir docs/vision_runs/real_perception
```

重点检查：

- `perception/rgb.png`、`depth.png`、`bboxes.png`、`mask.png`
- `perception/camera.json` 中的 `fx/fy/ppx/ppy` 和 `T_world_camera`
- `perception/pointcloud_stats.json` 中的目标点数和世界坐标

如果这里的目标点云不落在桌面上，先检查手眼结果和 D435i 对齐深度，不要执行运动。

## 底层完整 dry-run（两阶段 + 放置）

```bash
scripts/run_real_piperx_vision.sh \
  --instruction "抓取香蕉" \
  --max-attempts 1 \
  --diagnostics-dir docs/vision_runs/real_dry_banana
```

未传 `--execute` 时仍然连接桥接并读取状态，但不会向 `/control/*` 发送运动。
这是底层入口，默认仍为两阶段并执行放置；真机一句话入口的默认见下一节。

## 确认安全后执行

先确认机械臂周围无遮挡，手放在急停按钮上，并使用较低驱动速度。最简单的
入口只需要写一句指令：

```bash
scripts/real_grasp.sh "抓取香蕉"
```

真机入口现在的默认行为是：

- **单阶段**：只用一次 RGB-D（初始或搜索找到目标的视角）生成抓取；
- **grasp-only**：抓起后保持夹住，回到启动时记录的基准，不放置；
- **搜索重试**：任一 attempt 失败时，先回基准，再把 J1 转到下一个搜索角重试；
- 启动时同时记录当前 TCP 和关节角作为基准，成功或失败都会回该基准；
- **RViz 可选**：加 `--rviz` 会在后台启动 RViz2，并随本次抓取进程退出而关闭。

它等价于：

```bash
scripts/run_real_piperx_vision.sh \
  --instruction "抓取香蕉" \
  --execute \
  --open-gate \
  --max-attempts 1 \
  --single-stage \
  --grasp-only \
  --diagnostics-dir docs/vision_runs/real_grasp_<时间戳>
```

把 `香蕉` 换成桌面上的其他物体即可，例如 `scripts/real_grasp.sh "抓取瓶子"`。
如果要做完整抓取+放置，加 `--place`；要恢复两阶段观察，加 `--two-stage`。

### 单阶段与两阶段

`real_grasp.sh` 默认单阶段，适合 D435i 近距离死区较大、第二张近距离 RGB-D
不可用的情况：只使用初始（或搜索找到目标的）RGB-D 生成抓取，然后直接运动
过去。底层 `run_real_piperx_vision.sh` 默认仍是两阶段；需要显式加
`--single-stage` 才会跳过预抓取观察位。

两阶段的流程是：先移动到预抓取观察位，再拍一次 RGB-D 并用 GraspNet 生成最终
抓取。需要恢复两阶段时：

```bash
scripts/real_grasp.sh "抓取香蕉" --two-stage
```

单阶段仍然保留支撑检查、下降、闭合和抬升探针；在默认 grasp-only 模式下不
执行放置。它不会改变速度、限位或使能状态。

### 抓取后直接回基准（不放置）

`real_grasp.sh` 默认就是 `--grasp-only`：抓起后**保持夹住物块**、直接回到启动
时记录的基准位姿，不执行转移和放置。完整抓放需要显式加 `--place`：

```bash
scripts/real_grasp.sh "抓取香蕉" --place
```

`--grasp-only` 的行为：

- 只规划 `approach -> descend -> lift_probe -> lift`，不规划 transfer/place/retreat；
- 抬升探针和完整抬升都确认有物体后，立即结束抓取循环；
- 夹爪保持闭合，随后入口把机械臂用关节空间移回启动时记录的基准关节角；
- 结果 `success=true`，`placement_error_m=null`，`lift_height_m` 和
  `bilateral_contact` 仍会记录；
- `--drop-x/--drop-y` 只在 `--place` 下使用。

底层入口仍然显式接受 `--single-stage`/`--grasp-only`；`real_grasp.sh` 只是把
这两个设为真机默认，并用 `--place`/`--two-stage` 作为反向开关。

### VLM 判定目标不在画面/视角不合适时转 J1 搜索

每个 pickup 视角在 GroundingDINO 选完框之后都会再调用一次 VLM 校验（启动时
还调用一次做指令解析）。VLM 会看到当前 RGB、绿色检测框和指令，返回：

- `target_visible`：指令目标是否真的在当前画面里；
- `selected_matches_target`：DINO 选中的框里是否就是目标；
- `search_yaw_deg`：如果目标不在画面/当前视角不合适，建议**相对当前画面**
  把 J1 再转多少度；正数=向左/逆时针，负数=向右；
- `search_reason`：简要原因。

VLM 判定目标不可见、或 DINO 框不是目标时，立即转 J1 搜索；rotation 优先用
VLM 给出的角度，重复或无效时回退到固定角度表：

- GroundingDINO 没有检测到目标，且 VLM 也确认目标不在画面；
- 目标点云落到桌面工作区后点数太少；
- VLM 看到目标但 DINO 没检测到，或 DINO 框里的物体不是目标。

如果 VLM 确认目标可见且框匹配，但当前视角并不适合抓取（`search_yaw_deg` 非
0），后续 GraspNet 无候选、支撑/可达性预检失败或实际抓取失败时，会按 VLM
建议的角度换视角重试；如果 VLM 认为当前视角已经合适（`search_yaw_deg=0`），
抓取失败后不再搜索，直接回基准。这样既能在抓取失败时探索，又不会盲目转 J1。

VLM 每次校验的完整结果写到对应 `cycle_XX/pickup/vlm_verification.json`；VLM
返回的 target phrase 会用于下一次 DINO。`--semantic-backend rules` 不做图像
校验（rules 无法看图），需要视觉校验时使用默认的 `openai` 后端。

搜索时，如果还停留在感知阶段（机械臂没有执行抓取动作），直接把 J1 转到下一个
搜索角；如果已经发生下探/夹持/搬运，先 `recover` 到启动基准、张开夹爪，再转
J1。然后重新拍 RGB-D、重新检测、重新让 VLM 校验。固定回退角度：

```text
+20, -20, +30, -30, +40, -40, +60, -60, +80, -80, +120, -120 deg
```

VLM 建议角度优先；两者合计最多 13 次 attempt。某个视角 VLM 确认目标后
就停在那个视角继续抓取；单阶段直接用该帧，两阶段用它选择预抓取观察位。抓取
结束后（成功或失败）仍用关节空间回到启动基准。

抓取来源只有 GraspNet：**不做 analytic top-down 兜底**。但“目标可见”不等于
“当前视角可抓”：如果所有候选都因支撑、IK 或路径预检失败（`NoUsableGraspError`），
即使 VLM 返回 `search_yaw_deg=0`，也会继续尝试下一个固定 J1 搜索角，而不是
直接回基准。VLM 的 `yaw=0` 只表示当前画面看起来正常，不能覆盖无可用抓取这一
几何事实。只有运动执行类失败才继续遵守“VLM 没建议方向就不转”的保护。

搜索角度和搜索开关：

```bash
# 固定回退搜索角（VLM 建议仍优先）
scripts/real_grasp.sh "抓取绿色橘子" --search-angles-deg 30 -30

# 不做搜索，只试当前视角
scripts/real_grasp.sh "抓取绿色橘子" --no-search
```

`--max-attempts` 是基础尝试次数；启用搜索时实际次数至少为
`1 + 搜索角度数`。VLM 建议角度限制在 ±120°，固定角度最多 12 个；
速度、限位和使能状态都不变。搜索会转动整条机械臂，执行前要确认底座周围没有
会碰撞的障碍；若桌面空间狭窄，可把默认角度减小到 ±15/±30。

### 目标居中（实验性，默认关闭）

`--center-target` 会在找到目标后，先按当前相机位姿计算一个 J1 转角，把目标
3D 质心投影到图像水平中心，然后重新做一次完整感知和 GraspNet。J1 转角来自
相机投影，不依赖 VLM 的左右判断；18:35 实测把瓶子从 `u=139 px` 调到
`u=648 px`（中心 `644 px`），J1 从 `+26.4°` 转到 `+64.2°`。

但这个默认行为会改变相机相对桌面的方位角。18:35 的居中视角下 GraspNet
得到 11 个无碰撞候选，**全部被 `approach` 条件拒绝**（没有一个顶抓）；而
原先目标在画面左侧的 `+20°` 视角反而能生成顶抓候选。因此该功能默认关闭，
只保留为实验开关：

```bash
scripts/real_grasp.sh "抓取饮料瓶子" --center-target
```

18:36 还试了两阶段 `--two-stage --no-center-target`：预抓取观察位
`[0.088, 0.361, 0.140] m` 的 `move_l` 也超时，没有改善。当前结论是：**仅仅
让物体在图像中央不够，抓取候选的姿态可达性和顶抓视角才是瓶颈**；不要把
`--center-target` 当成默认修复。

### 顶抓优先、侧/斜抓自动兜底

现在的策略不是“只允许顶抓”，而是：

1. 先扫描所有 `approach_cos > 0.8` 的顶抓候选；
2. 顶抓候选全部在支撑/路径预检失败后，`--grasp-only` 自动进入
   `approach_cos > 0.2` 的斜/侧抓兜底；
3. 顶抓候选一旦通过，就绝不会被后面的侧抓候选抢占。

`--grasp-only` 和完整 `--place` 都采用这个顺序，不再需要 `--side-grasps`；
该参数只保留作兼容。兜底只会发生在所有顶抓候选都失败之后，不会抢占顶抓。

真实入口还会用 MuJoCo/URDF 模型对最终 TCP 和预抓取位姿做 IK 预检；无解候选
在运动前标记为 `rejected_ik`，不会再把不可达姿态发给 `move_l`。18:42 的
默认单阶段实机测试中，顶抓候选没有一个通过 approach 门槛，最终选中 fallback
候选 rank 1（`approach_cos=0.780`，仅略低于 0.8），IK 预检通过，抬升
`0.059 m` 且双边接触为真，结果 `success=true`。日志：
`docs/vision_runs/real_grasp_20261028_184221`。这说明“顶抓优先、侧/斜抓兜底”
比原来的硬阈值更符合实际。

兜底模式预抓取点沿 GraspNet 的 approach 轴生成，要求预抓取点高于桌面、
位于工作区内；夹爪抬起仍沿世界 +Z。对平放的螺丝刀这类物体可显式指定：

```bash
scripts/real_grasp.sh "抓取螺丝刀" --grasp-only --side-grasps
```

**2026-10-28 实机测试结果：不建议在当前姿态下使用。** 螺丝刀在 cycle 2
（J1 -20°）选中了一个支撑检查通过的侧抓候选，但 `move_tcp` 在 30 s 后超时
（位置误差 0.307 m，姿态误差 1.08 rad），机械臂停在 joint4=88.51°（接近 ±89°
限位）附近；随后 Cartesian recovery 因为当前 TCP 的 x=0.098 m 略低于工作区
下限而失败。现在已加入“恢复失败时用关节空间回基准”的后备路径，但侧抓本身
仍受手腕限位限制。要抓平放的螺丝刀，优先把它垫高/立起来后用默认顶抓，或换
一个适合顶抓的物体。

### 横放瓶/圆柱的插入深度

2026-10-28 起默认 `--grasp-insertion-depth` 已从 `0.015` 调到 **`0.03 m`**。
横放瓶子的可见上表面到中截面沿接近轴约有 3 cm；旧的 0.015 m 让 TCP 停在瓶身
上部，夹爪在最宽截面之上闭合，实测闭合到 0.0004 m（空夹）。

现在默认 0.03 m 时会选中 rank 2，TCP z 从 0.042 m 降到 0.029 m，夹爪闭合到
0.047 m，抬升 0.058 m 并确认双边接触，结果 `success=true`，机械臂回基准时
保持夹住瓶子。简单入口即可：

```bash
scripts/real_grasp.sh "抓取瓶子"
```

**2026-10-28 18:22 实机结果（球形 workspace + 12 mm 预选中心容差）：**
瓶子在 `x≈0.111, y≈0.402`，距肩关节约 `0.42 m`；第一次搜索到 J1 `+20°`
时目标点云 29105 点，选中 GraspNet rank 2（width `0.0767 m`），
TCP `[0.131, 0.428, 0.148] m`，抬升 `0.0593 m`，双边接触为真，
`policy_success=true`，最后回到启动基准并保持夹住瓶子。日志：
`docs/vision_runs/real_grasp_20261028_182255`。

`bounded_grasp_insertion` 仍会按指尖和桌面做限幅，低矮物体不会因为默认值变大
而把指尖压到桌面上。薄片、抬高放置的小物体或容易压坏的目标，可以显式减小：

```bash
scripts/real_grasp.sh "抓取香蕉" --grasp-insertion-depth 0.015
```

横放圆柱类目标优先用 0.025~0.030 m。

首次实机只建议抓取一个轻物体，保持 `speed_percent:=20`，不要使用
`--test-release-first`。出现异常时用 Ctrl-C；视觉入口会调用桥接的
`/emergency_stop`，它发送的是**原地保持**指令，不会调用
`/enable_agx_arm false`，机械臂仍保持使能。进程退出时只关闭
`/control_enable` 控制门，不关闭电机使能。

## 当前基准位姿

2026-10-24 手动调整后的真机初始基准：

```text
[5.51, 32.98, -74.92, 89.38, -8.68, 1.28] deg
```

桥接查询到的 TCP 约为 `[0.1265, 0.0372, 0.3631] m`。仿真
`HOME_JOINTS_DEG` 使用 `[5.51, 32.98, -74.92, 87.0, -8.68, 1.28]`：
真机 joint4 的 89.38° 超出仿真 ±89° 限位，且该位姿位于腕部奇异点附近，
沿笛卡尔直线起步时局部 IK 无解；87° 保持相同姿态并留出安全余量。

> **注意：** Piper X 的 joint4 固件软件限位是 ±89°。如果当前 joint4 略微超出
> （例如 89.18°），`--execute` 入口现在会先用关节空间把它收到限位内 2° 的安全
> 姿态（joint4≈87°），更新启动基准后再继续，不会放宽固件限位；只有自动恢复
> 失败时才中止。手动恢复仍然可用：
>
> ```bash
> cd /media/yzc/F/DOGcode/my/piper/piper_control
> source /opt/ros/humble/setup.bash
> source ~/agx_arm_ws/install/setup.bash
> export ROS_DOMAIN_ID=42
> export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
> /usr/bin/python3 tools/arm.py j 5.51 32.98 -74.92 87.0 -8.68 1.28 -w 8
> ```
>
> 这是一次约 2° 的 joint4 关节运动，不会关闭使能。运行后可用
> `ros2 topic echo /feedback/arm_status --once` 确认 `arm_status: 0`。

## 工作区（workspace）

`configs/real_piperx.yaml` 里的 `workspace` 是**三维作业范围**，同时用于视觉点云过滤
和 bridge 运动检查，不是物体检测器。Piper X 的可达范围不是一个在桌面上的小矩形，
而是以肩关节（`base_link` 上方 `0.123 m`）为中心的球：

```text
joint2->joint3  0.28503 m
joint3->joint4  0.27364 m
joint4->joint5  0.07466 m
joint5->joint6  0.03500 m
flange/gripper TCP 约 0.1295 m（再加 0.0045 m 夹爪基座）
------------------------------------------------
肩关节到 TCP 约 0.80 m
```

所以配置里同时保留“桌面外框”和“球形可达范围”：

```yaml
workspace:
  table_top_z: 0.0
  reach_center_xy: [0.0, 0.0]
  reach_center_z: 0.123
  reach_radius_m: 0.80
  reach_min_radius_m: 0.02
  x: [0.0, 0.80]       # 桌面外框，不是真正的可达判据
  y: [-0.80, 0.80]
  table_z: [0.0, 0.50]
  object_z: [0.002, 0.30]
  grasp_z: [0.002, 0.25]
```

实际判据是 `sqrt((x-cx)^2 + (y-cy)^2 + (z-0.123)^2) <= 0.80`。桌面高度
`z=0` 时水平可达半径约 `sqrt(0.80^2 - 0.123^2) ≈ 0.790 m`，与 URDF/MuJoCo
采样在桌面上得到的约 `0.78 m` 一致。矩形角点即使落在 `x/y` 外框内，只要超出
这个球也会被拒绝；反过来，外框内的大多数桌面区域（包括 2026-10-28 瓶子所在的
`x≈0.11, y≈0.40`，距肩关节约 `0.42 m`）都能通过。

这套球范围用于三处：视觉点云过滤、`RealPiperXEnv._check_pose` 和 bridge 的
`move_tcp`。bridge 只在启动时读配置，改完 `real_piperx.yaml` 后要重启
`scripts/piperx_real_bridge.sh` 才会生效；重启 bridge 不会关闭电机使能。
`max_command_distance_m` 和固件关节限位仍是第二层检查。目标在相机里可见、但
3D 位置落在球外时，点云目标会被过滤成 0 点；这时应移动物体，而不是把问题当成
DINO/VLM 没看到。**注意：** 2026-10-28 18:06 那次瓶子运行虽然最终失败，但瓶子
已经通过 workspace：目标点云有 26794 点，失败点是 GraspNet 候选的
approach/width/两指支撑检查，不是可达范围。用新球形范围离线重放同一帧，
目标点数仍然是 26794，质心/包围盒不变。

## 可选 RViz 可视化

抓取时一起启动（推荐）：

```bash
scripts/real_grasp.sh "抓取香蕉" --rviz
```

`--rviz` 会在后台启动 `scripts/piperx_rviz.sh`，日志写入本次
`docs/vision_runs/real_grasp_<时间戳>/rviz.log`；抓取进程退出时 RViz 会一起关闭。
即使 RViz 启动失败，抓取本身仍会继续，失败原因看这个日志。

需要让 RViz 单独常驻、不跟抓取进程退出时，直接运行：

```bash
scripts/piperx_rviz.sh
# 等价于
ros2 launch ./scripts/piperx_rviz.launch.py
```

RViz 会显示机器人、相机位姿/视锥、RGB-D 图像、RGB-D 点云和最新抓取位姿。
抓取位姿从 `docs/vision_runs/*/cycle_*/final/grasp_pose.json` 读取，因此上面
`--diagnostics-dir docs/vision_runs/real_grasp_banana` 的输出会自动被看到。
点云/深度显示的近裁剪是 `0.11 m`（D435i Min-Z 约 `0.105 m`，留少量余量），
只影响 RViz 显示；视觉感知使用原始对齐深度，不做这个近裁剪。

## 坐标约定

- 驱动反馈 `/feedback/tcp_pose` 在 `tcp_offset=0` 时是法兰位姿。
- 桥接层使用 `T_flange_tcp = [0,0,0.1295] m + Rz(90°)`，即仿真夹爪垫中心 TCP。
- 手眼结果 `T_flange_cam` 来自 `piper_control/tools/calib_results/`。
- 视觉控制统一使用基座坐标系的 TCP 位姿 `T_base_tcp = T_base_flange @ T_flange_tcp`。

## 故障排查

- `arm_status=4 (target_pos_exceeds_limit)`：先检查当前关节是否已经在软件
  限位外，最常见的是 joint4 的 89.38°；按上面的 joint4 恢复步骤回到 87°。
- 视觉进程报 `ROS bridge request failed: timed out`，随后 bridge 报
  `Connection reset by peer` / socket 不存在：旧版 bridge 在客户端超时断开时
  会被 `BrokenPipeError` 终止。现在 bridge 会捕获断连并继续监听；客户端
  超时改为 45 s，大于 bridge 的 30 s 运动超时；检测到 `arm_status` 为
  2/3/4 且 TCP 在 1.5 s 内没有移动时，会立即返回明确的错误。重启 bridge
  后重跑即可。

## 当前限制

- 夹爪没有独立双指触觉反馈时，`finger_contact_state()` 使用夹持力和开口宽度作为代理，
  不是仿真中的真实双指接触。
- 桥接层目前使用驱动固件的 `move_p` / `move_l`，没有 MoveIt 全局避障预检。
- `workspace`、速度和手眼文件是配置项；更换相机安装或工作台必须先重新标定。
