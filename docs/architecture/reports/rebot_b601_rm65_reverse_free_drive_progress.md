# RM65 free-drive → B601 MuJoCo 反向遥操进度

## 当前切片：P0 只读观测、纯 retarget、P1 仿真装配与 P3 前台运行环

本切片面向“RM65 真机手动拖动/free-drive，B601 在 MuJoCo 中跟随”的第一段
链路。它只增加适配层，不改变既有 B601 真机控制、RM65 真机执行、MuJoCo
位置驱动或控制组合 authority。

### 已落地

- `control_station.sources.rm65_free_drive.RM65FreeDriveSource`
  - 复用现有 `RM65ArmBackend.refresh_state_from_hardware()`。
  - 只接受 `observation_only`、`hardware_observation_only`、完整且成功的
    hardware qpos/FK pose、递增 measurement sequence 和 `local_wall_time`。
  - 同时检查 wall-time measurement age 与 monotonic receive age；坏帧、重复
    sequence、未来时间、过期状态和非刚体 pose 均当前 tick fail-closed，不回退
    到上一帧。
  - 不启动后台流、不调用 RM65 命令、不拥有 GUI、dispatch 或 episode authority。

- `robot_motion.rebot_b601_rs.rm65_retarget.RM65ToB601Retargeter`
  - 复用已有 B601/RM65 `CoordinateFrameGraph`，通过图的逆向解析实现
    `RM65 right_arm_base/tool_tcp → B601 base_link/gripper_end`，没有复制矩阵
    约定或另建坐标系。
  - 采用显式 source/target anchor、平移/旋转 gain、freshness 和连续 seed。
  - 复用 `B601DampedLeastSquaresIK`；IK 失败、过期样本、重复 sequence 或
    frame/clock 不匹配只返回 rejection，不产生旧 qpos fallback。
  - 输出仍是纯 qpos candidate；不直接调用硬件或 MuJoCo。后续应用层应把成功
    candidate 交给现有 `B601PositionDriver`/target runtime 的唯一执行 owner。

- `control_station.sim_hil_profile.rm65_b601_free_drive.RM65ToB601FreeDriveBridge`
  - 复用现有 `B601PositionDriver` 的物理步、500 Hz 命令节拍、位置 slew、限位
    和 fault latch；桥接层不写 `MjData`，不复制 position servo。
  - 只负责 `inactive → active → clutched → active/held` 的 anchor、clutch 和
    recenter 状态；激活/重定位都要求一帧新的 RM65 观测。
  - 目标 qpos 只在新样本通过 frame/age/IK 后排队；source、IK 或 driver 拒绝
    会撤销 pending target 并进入 `held`，不会重放上一条 qpos。
  - `step()` 是一个 MuJoCo physics tick；调用方仍负责按现有 driver 的 1 kHz
    physics 时钟推进，不把 RM65 20 Hz 观测误当成新的仿真执行频率。
- `build_rm65_to_b601_free_drive_bridge()` 直接加载现有 B601 模型、现有
  mapping profile、现有 Pinocchio FK/IK 和现有 MuJoCo scene/driver。反向平移
  与旋转 gain 在 `RM65ToB601FreeDriveSimConfig` 中显式给出，不隐式反用正向
  150% profile。

- `control_station.sim_hil_profile.rm65_b601_free_drive_foreground`
  - 复用既有 `B601RM65TerminalCommands` 的无 Enter 单键读取语义，并由新的
    反向前台协调器统一管理 `MujocoPassiveViewerSession` 生命周期。
  - `a/c/r/s/q` 只改变桥接层的 inactive/active/clutched/held 状态；每个
    MuJoCo tick 都调用现有 `RM65ToB601FreeDriveBridge.step()`，不会另起
    source、IK、位置驱动或后台线程。
  - viewer 关闭、Ctrl-C、EOF 和 source/IK admission rejection 都会撤销
    pending target 并退出/保持 held；退出只关闭仿真 viewer，不触碰 RM65 或
    B601 真机。

- `python -m control_station rm65-b601-free-drive`
  - 在同一应用入口中组装 `RM65ArmBackend(observation_only=True)`、只读
    `RM65FreeDriveSource`、反向 B601 MuJoCo bridge 和 native passive viewer。
  - `--validate-only` 只加载 URDF、mapping、MuJoCo 和 Pinocchio；
    `--connect-read-only` 才建立 RM65 观测连接，并要求显式 confirmation、IP
    和 URDF。该模式不启用 RM65、不发送位置/力矩/停止命令，MuJoCo 目标也不
    是真实 B601 执行路径。
  - 连接、freshness、anchor、clutch/recenter、held 和退出事件统一写入传入
    的 JSON/log；不再依赖 shell 维护交互状态。

### 自动化测试证据（不是现场验收）

```text
tests/test_control_station_rm65_free_drive.py     5 passed
tests/test_robot_motion_rm65_retarget.py          5 passed
tests/test_control_station_rm65_b601_free_drive.py 4 passed
tests/test_control_station_rm65_b601_free_drive_foreground.py 3 passed
tests/test_control_station_rm65_b601_free_drive_application.py 2 passed
ruff (changed files)                               PASS
git diff --check                                   PASS
```

这些证据是 fake backend/纯数学测试，不代表 RM65 真机连接、free-drive 状态或
MuJoCo GUI 已验收。

当前候选的合并前相关切片（包含上述应用入口测试、集成、前台和边界测试）为
`66 passed`；该数字仍只表示离线自动化证据，不表示现场硬件验收。

### P2 synthetic MuJoCo 自动化测试

已增加 `tests/test_rebot_b601_rm65_free_drive_integration.py`。该用例使用真实
`reBotArm_control_py` URDF、真实 Pinocchio B601 模型、真实 MuJoCo scene 和
现有 `B601PositionDriver`，但 RM65 输入是确定性的 synthetic observation，不
打开 CAN。它验证：

- 通过现有 mapping profile 的反向 frame graph 生成 RM65 6D 样本；
- anchor 后连续 seed IK 和 position-driver command timing；
- 目标 qpos 最终跟踪误差小于 `5e-4 rad`；
- source rejection 后进入 `held`，撤销 pending command，不重发旧目标。

现场有 vendor 源目录时的显式命令为：

```bash
DEXTELEOP_REBOT_SOURCE_ROOT=/absolute/path/to/reBotArm_control_py \
PYTHONDONTWRITEBYTECODE=1 bash scripts/run_pytest.sh -q -p no:cacheprovider \
  tests/test_rebot_b601_rm65_free_drive_integration.py
```

本候选已用 `/home/zc/HandMotion/Developing/reBotArm_control_py` 作为只读输入，
桥接集成结果为 `2 passed in 1.65s`；加入前台 runner 的组合回归后，
`tests/test_rebot_b601_rm65_free_drive_integration.py` 为 `3 passed in 1.98s`。
该路径没有写入代码或测试默认值。

## P3 自动化测试

新增前台运行环的离线测试：

```bash
PYTHONDONTWRITEBYTECODE=1 bash scripts/run_pytest.sh -q -p no:cacheprovider \
  tests/test_control_station_rm65_b601_free_drive_foreground.py \
  tests/test_control_station_boundaries.py
```

该测试用假的 bridge/viewer，只验证生命周期、无 Enter 命令语义、held/retry
和 viewer 关闭；它是测试，不代表 MuJoCo native GUI 或 RM65 真机已验收。

## 现场验收（尚未执行）

自动化测试通过后，现场验收必须另行执行并保存前台日志、JSON 结果和操作者观察：

1. 在 `user@192.168.70.116` 的桌面会话中，使用
   `python -m control_station rm65-b601-free-drive --connect-read-only`，明确
   传入 RM65 右臂 IP、`rm65_arm.urdf`、B601 source root、task config 和 mapping
   profile；不得猜测现场网络配置，也不得把 RM65 command/stream owner 放进此
   模块。SSH 非交互会话没有 `DISPLAY` 时只可做 validate-only/诊断，不能冒充
   native GUI 验收。
2. 现场操作者先确认 RM65 失能/free-drive、无负载和急停可用，再启动只读观测；
   通过 `a` 激活、手动拖动、`c/r` 离合重定位、`q` 或 Ctrl-C 退出。
3. 验收记录必须包括：RM65 实际反馈连续性、source age、anchor/recenter、
   MuJoCo 末端位移/旋转方向、viewer 关闭、异常/过期帧进入 held，以及退出后
   RM65 未收到运动命令。自动化测试通过不能替代这些现场验收项。

## 真机边界

当前提交没有打开 CAN、没有启动 RM65 free-drive、没有发送 B601 或 RM65 运动
命令。远端隔离 worktree 已完成同一候选的自动化测试，但尚未进行现场验收。后续
真机验收必须由现场操作者前台执行，并单独记录：硬件读状态、observation-only
配置、source age、anchor/recenter、MuJoCo target 轨迹和急停/退出行为；离线通过
不能替代这些证据。

## 当前 frame calibration 固定值

正向 B601→RM65 与本反向 RM65→B601 共用同一份 canonical profile：
`docs/hardware/profiles/b601_rm65_mapping_150_v2.json`。现场已确认并固定：

```text
controlled RPY (roll pitch yaw): 0 90 0
reference RPY (roll pitch yaw): 0 0 90
```

profile 对应的旋转矩阵是 `R_y(+90°)` 与 `R_z(+90°)`，base 平移仍是 URDF-derived
`(-0.2625, 0, 0)`，正向平移 gain 仍为 `1.5 1.5 1.5`。反向 retargeter 通过同一
`CoordinateFrameGraph` 解析严格的逆向边，不把正向矩阵直接套到 RM65 source；反向
translation/rotation gain 由 `RM65ToB601FreeDriveSimConfig` 独立提供。

该固定值只改变 frame calibration，不改变 B601 真机 owner、RM65 真机执行器或任何
CAN/位置/力矩命令。离线反向集成测试必须验证 round-trip、连续 seed 和 rejection
撤销；现场 `--connect-read-only` 仍然只读 RM65，并只驱动 B601 MuJoCo target。
