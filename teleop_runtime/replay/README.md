# DexTeleop Replay

> 本页维护 replay 的完整模块参数。当前离线/真机运行入口统一从
> [`docs/real/README.md`](../../docs/real/README.md#2-replay) 复制。

`teleop_runtime.replay` 用于查看已有数据集，并可在通过检查后把数据集轨迹复现到 RM65 真机。它只做 replay，不创建新 episode，也不写采集数据。

## 启动 GUI

```bash
conda activate dexteleop310
cd /home/zc/HandMotion/Developing/DexTeleop

python -m teleop_runtime.replay \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

服务每次启动都会生成新的 256-bit control token，并将其注入当前页面；页面发出的
`POST /api/control` 会携带 `X-DexTeleop-Control-Token`。token 同时打印在启动终端，
供本机调试客户端使用。控制请求还必须满足同源 `Host`/`Origin` 校验和
`application/json` 类型，单个请求体上限为 64 KiB。

HTTP 服务默认只允许 loopback bind。确需从其他机器访问时，必须显式传
`--unsafe-allow-remote-web`；通配 bind 还必须至少传一个
`--web-allowed-host <hostname-or-ip>`。该模式仍是明文 HTTP，不建议跨不可信网络使用。

也可以启动时直接加载数据集：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

省略 `--episode` 会自动选择第一条可用 episode；若要指定，请使用数据集真实
`episode_index` 或 `episode_id`，例如 `--episode 10` 或 `--episode episode_000010`。

数据集根目录通常需要包含：

```text
meta/episodes.jsonl
data/chunk-*/episode_*.parquet
```

页面中执行：`Load` 数据集 -> 选择 episode -> `Play/Pause` 或拖动时间轴。若 episode 元数据或 parquet 行字段中记录了视频/触觉路径或 inline 触觉字段，摄像头与 DMTAC 触觉会自动显示。

## 真机轨迹 Replay

当前真机 replay 面向 `teleop_runtime.debug.run_vive_rm65_real_teleop`
采集的数据集。主链路始终是数据集 canonical TCP action replay；qpos 只用于
`Move Start` 安全初始化或显式低层 debug，不作为轨迹 replay 默认来源。默认语义是：

- `Move Start`: 只使用首帧 qpos 做安全初始化。优先用
  `action.replay.<side>_arm_joint_target`；没有该字段时退回
  `observation.arm.<side>.current_qpos`，再退回
  `observation.arm.<side>.actual_hardware_qpos`。
  首帧 QPOS 初始化速度可通过启动参数或 GUI 的 `QPOS Init Vel`
  调整；传 1 个 rad/s 值应用到 6 个关节，也可传 6 个逗号分隔的逐关节 rad/s 值。
- `Start`: 按帧读取数据集 26 维 `action` 中 active side 的 7D
  `[x, y, z, qx, qy, qz, qw]` pose。先对 15 Hz 离散关键帧执行
  finite/quaternion/workspace/位姿步长安全限制；优先按 parquet 的完整、严格递增真实
  timestamp 重采样，缺失或非法时整段显式退回 nominal action grid，再生成
  可配置的pose reference（默认60 Hz；xyz linear + shortest-path quaternion SLERP，不是时序平滑）。
  默认必须同时激活左右臂，每个 60 Hz tool pose 由共享双臂 collision-aware IK
  转成 qpos，再执行绝对关节限位、每 reference 步长/速度限制，并对限位后 qpos
  重新计算碰撞距离。只有通过的 60 Hz qpos 才会交给通用 60→120 qcmd 链路，
  真机 backend 不再对该链路独立解逐臂 Cartesian IK。

当前安全边界：每个最终限位后的 60 Hz qpos endpoint 会执行双臂碰撞检查；
但相邻 60 Hz endpoint 之间的 120 Hz 关节插值整段尚未做连续碰撞认证。
调试 metadata 因此始终明确记录 `endpoint_collision_checked=true`、
`segment_collision_certified=false`；不能将该链路描述为连续轨迹已认证。

真实输出在创建/reset planner 时只接受本轮成功读取的双臂硬件 qpos：默认测量年龄不超过
0.25 s、左右测量时间差不超过 0.05 s，并要求 read/measured sequence、timestamp、qpos
一致。硬件读取失败时不会退回 cached `current_qpos`/`last_target_qpos`。coupled planner
通过后的 qpos 会标记为 prelimited，backend 若再次单臂改写或触发 singularity hold，replay
会在同 tick hand 下发前 fault/hold。arm queue 前后及每次 hand I/O 前也会检查两侧 120 Hz
stream readiness/fault/stale 状态。

仅在默认的 coupled position replay 中，若 RM65 位控 backend 的 SDK 判定某侧目标或当前
qpos 处于奇异区，执行器会先按原本有效关节步幅的 `0.2` 计算该侧的慢速 qpos；随后对这组
**实际将发送的双臂慢速 qpos** 再执行同一组 12 对 endpoint 碰撞检查。复核不通过时，
该 reference 不会下发，replay 进入 fault 并保持；只有复核成功的 command 才携带
`singularity_replay_slowdown_preapplied` 标记，backend 才接受该慢速目标，避免二次限幅改写
已检查的双臂组合。此规则不进入 deploy 链路，也不改变 qpos-init 的既有慢速规则。

这一步用于验证后续模型预测的 action 是否能对齐真机下游接口。显式
`qpos_target` 模式仅作为低层兼容/调试路径：生产默认的
`position_collision_required=true` 会明确拒绝它。显式设置
`--no-position-collision-required` 只允许进入不下发动作的旧 qpos/Cartesian
backend 兼容检查；position 模式一旦启用真机输出仍会 fail-closed 拒绝该配置。

## 末端 Replay

默认只 replay 机械臂。需要同时验证手/夹爪接口时，启动时打开
`--end-effectors-enabled`，并为每个 active side 指定末端 backend：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --end-effectors-enabled \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --require-dmtac-config \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

Replay Web GUI 与正式 GUI、collection、deploy 共用
`teleop_runtime/configs/hardware/dmtac/latest.json`。默认会从 assignment 读取 active
side 的 gripper 稳定串口、Modbus ID、波特率和 timeout，但不会连接 tactile。
`--left-gripper-port` / `--right-gripper-port` 仍可逐侧覆盖；对应的 ID、baudrate 和
timeout 使用 `--SIDE-gripper-id`、`--SIDE-gripper-baudrate`、
`--SIDE-gripper-timeout`。完全手工配置时使用 `--no-dmtac-use-config`。可用
`--no-dmtac-validate-usb` 跳过当前 USB 探测，但这不会修复错误的左右 assignment。

### UMI replay 坐标系

v2.3 UMI 数据的 arm pose 先依据数据集中的 `pose_frame_contract` 统一为
canonical `desk`，真机执行边界再使用已检入的 RM65 workspace-near-boundary runtime contract
分别转换到左右 controller base。preflight、position/collision planner 与 impedance
replay 使用同一条转换链；缺失或不完整的 canonical frame contract 会直接阻止回放。

UMI 标定协议把标定板坐标系直接定义为 canonical `desk`。因此标定时标定板必须与
RM65 URDF 的 `workspace_near_boundary_frame` 在物理上同原点、同轴；板的位置或朝向改变后必须
重新标定，不能只复用旧的 `T_vive_world_to_desk`。

`Move Start` 仍然只做机械臂首帧 qpos 初始化，不移动末端。`Start` 阶段
会从数据集 `action` 中读取 active side 的 hand slot。手/夹爪仍在各自 15 Hz
source timestamp 离散下发；机械臂独立按 60 Hz reference 下发，因而不会提前
发送下一帧 hand keyframe，也不会制造中间 hand action。

若临时 tracker hold/FROZEN 帧没有新的 hand command，正式数据集的 26D `action`
hand slot 可能为 NaN；此时 replay 只在对应
`action.replay.<side>_hand_joint_target` 存在、整段 shape 正确且全部 finite 时，使用该列
记录的 backend last-accepted target。主 action 有效时仍优先使用主 action。显式 replay
列缺失或损坏时 preflight 继续 BLOCK，不会把任意 observation/默认零值当作命令。

Impedance replay 的 TCP preflight 比较 active impedance controller 实际加载的
`tcp.pose_flange_m_rad`，不能回退为 arm-endpoint 零 TCP；例如 DMTAC profile 的
`[0, 0, 0.14, 0, 0, 0]` 应与数据集工具 profile 数值一致。

末端维度以 dataset metadata 中的 `hand_action_specs_by_side` 为准。DMTAC
gripper 当前使用 `dmtac_gripper` spec，将 6-slot 兼容 hand target 映射到
1 维 gripper target；如果缺少 spec，resolver 会退回 q6 手部默认语义，
这通常不适合夹爪部署，需要先补齐数据集 metadata 或做后处理转换。

当前通用 replay 动作构造、轨迹 contract 与 15→60→120 schedule 已集中在
`robot_motion/replay` 和 `robot_motion/position_control`。`teleop_runtime/replay`
只保留真机 backend、GUI、preflight 与安全执行壳；MuJoCo replay 复用同一
时间戳 resampling/schedule。

position 频率和 15 Hz keyframe 安全可在启动时配置：

```text
--position-reference-hz 60
--arm-control-hz 120
--position-workspace-min -0.60 -0.80 0.00
--position-workspace-max  0.60  0.80 0.90
--position-max-xyz-step-m 0.10
--position-max-rot-step-deg 15
--position-max-joint-step-rad 0.10472
--position-collision-required
--position-collision-scene-profile dual_arm_hand_proxy_open
--position-collision-min-distance-m 0.03
--position-collision-distance-epsilon-m 0.0001
--position-collision-max-joint-velocity-rad-s 2.967059728,2.967059728,3.316125579,3.665191429,4.188790205,4.188790205
--no-position-collision-debug
```

位控 replay 默认使用正式 `rm65_bimanual_position_v1`：自适应 IK `min=2/max=4`、
orientation cost `0.35`、damping `0.03`、5 ms iteration reserve，并使用
`rm65_bimanual_position_v1_qref` 的逐关节速度上限。数据集主 action 已经是 accepted
qref 的 FK pose；replay 不重复采集前 inactive/tracker/post-IK qref 平滑，只执行主
action 之后的插值、IK、安全限位和下发。命令行仍可显式覆盖逐关节速度值用于 debug，
也可用 `--position-control-profile legacy` 回退。

`--position-collision-max-joint-step-rad` 可选传 1 个或 6 个值；留空时仅使用
`max_joint_velocity / reference_hz` 生成每个 60 Hz reference 的安全步长上限。

`--position-collision-debug` 会在每个 60 Hz command metadata 中附加 raw IK qpos、
限位后 qpos、target/realized endpoint pose、碰撞尝试和耗时明细；关闭时仍保留
status、距离、限位原因、endpoint/segment 认证边界等摘要。controller 同时维护一个
512 条的有界事件环和 tracking/limited/backoff/hold/drop 计数；`/api/state` 返回 last 与
计数，完整有界记录可从
`GET /api/position-collision-debug?limit=100` 读取。只有实际最近一次检查成功且 replay
当前未 fault 时，status 才会报告 `endpoint_collision_certified=true`。

开启 collision debug 时，每个 ring event 还固定保留三个 transport-health 槽：
`before_arm_reference`、`after_arm_reference_before_hand` 和
`immediately_before_end_effector_dispatch`。前两个槽记录每个reference tick的检查；
第三个槽仅在该 reference 实际调度 hand command 时记录，否则为 `not_applicable`。
检查失败会先保存阶段、时间和异常再 fail-closed。该 ring/API 是进程内有界调试窗口，
进程退出后不会作为 sidecar 自动落盘。

这里仍不是跨设备原子事务：左右 RM65 backend 目前是顺序 queue，hand I/O 与 arm
background worker 也没有共同事务。第二侧 queue 才失败时，第一侧可能短暂入队后再执行
group hold；最后一次 arm health 检查之后瞬时发生的 SDK fault 也存在不可彻底消除的极小
竞态。debug transport-health snapshot 只保留已执行 gate 的观察结果；replay ring 当前不保存
完整 backend receipt，因此不能据此宣称双臂与 hand 同时生效。

impedance 模式保留原 action-rate → 40 Hz backend 路径。

Replay 会在连接/预检前比较 episode metadata 的 `arm_control_mode` 与所选
`--arm-control-mode`。不一致会直接拒绝。旧 episode 缺少该字段时，真机上下文按
`impedance` 契约处理，因此不会被默认 position 分支静默接管；无硬件 dry-run 仅给出
无法验证模式的 warning。

单右臂非运动旧兼容示例（显式关闭双臂 collision planner，不能配合真机输出）：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides right \
  --no-position-collision-required \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

双臂示例：

```bash
python -m teleop_runtime.replay \
  --dataset-root /path/to/lerobot_dataset_root \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

GUI 中按顺序操作：

1. 确认 `Hardware Replay` 面板里的 active side、IP、TCP profile。
2. 默认 `Arm Output` 关闭且 Web API 无权自行解锁。需要稍后在 GUI 内开启时，
   启动进程必须预先显式传 `--arm-web-hardware-output`；`--send-actions` 则同时完成
   本机授权并在启动时直接打开输出。
3. 点击 `Preflight`。GUI 会先应用当前表单配置，并在需要时自动 `Connect`。
4. 确认所有阻断项通过后，点击 `Move Start` 执行 qpos init plan 到轨迹首帧。
5. 点击 `Start`，按 `action` pose 复现轨迹并验证下游 action 接口。
6. 需要停止时点 `Abort`；replay 会请求 arm hold 并恢复 replay speed override。

不满足当前状态条件的按钮会自动置灰。例如未选 episode 时不能
`Preflight`，preflight 未通过时不能 `Move Start`，未到 `start_ready` 时不能
`Start`。`Connect` 按钮只保留为手动诊断入口，正常流程不需要单独点击。

## Preflight 检查

真机 replay 会在开始运动前检查：

- 数据集是否有有效轨迹。
- 默认 `action_pose` 模式下，active side 的 `action` pose 是否全帧 finite 且可转换为 pose matrix。
- 是否能为 active side 选择到 finite 的首帧 qpos init target。
- 当前真机 qpos 与选中的首帧 qpos init target 是否在允许范围内。
- 数据集记录的 tool profile TCP 与当前 GUI/CLI 配置的 TCP 数值是否一致。
- 当前 TCP 位姿到轨迹首帧位姿的距离会显示为提示信息。

`Preflight` 本身不执行轨迹运动；它只会连接/读取当前硬件状态并生成阻断项。
真正的机械臂运动从 `Move Start` 开始，真正的 action replay 从 `Start` 开始。

默认不会向真机发送动作。未通过本机进程参数授权时，HTTP 即使持有 control token，
也不能开启 `send_actions`，不能调用 `Move Start` 或 `Start`。授权方式只有启动时显式
添加 `--arm-web-hardware-output`（允许 GUI 稍后开启）或 `--send-actions`
（启动时立即开启）。
