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
  - 采用显式 source/target anchor、平移/旋转 gain、freshness 和连续 seed；连续
    seed 由桥接层保存的上一条已接受 IK qpos 提供，不使用可能因位置伺服滞后而落后的
    MuJoCo 实测 qpos。
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
  - `a/c/r/s/q` 只改变桥接层的 inactive/active/clutched/held 状态；物理环仍按
    B601 既有节拍推进，native viewer 默认独立以 60 Hz 刷新，不再让 GUI vsync
    把物理环压到约 185 Hz。
  - live application 的 RM65 source poll 按 monotonic wall clock 调度（默认
    20 Hz），不再使用被 viewer 拖慢的仿真时间调度；synthetic tests 默认保留
    确定性的仿真时钟调度。
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
tests/test_control_station_rm65_b601_free_drive.py 6 passed
tests/test_control_station_rm65_b601_free_drive_foreground.py 3 passed
tests/test_control_station_rm65_b601_free_drive_application.py 2 passed
tests/test_rebot_b601_rm65_free_drive_integration.py 3 passed
ruff (changed files)                               PASS
git diff --check                                   PASS
```

这些证据是 fake backend/纯数学测试，不代表 RM65 真机连接、free-drive 状态或
MuJoCo GUI 已验收。

当前候选的合并前相关切片（包含上述应用入口测试、集成、前台和边界测试）为
`66 passed`；该数字仍只表示离线自动化证据，不表示现场硬件验收。

## 现场延迟与 held 诊断（2026-09-19）

现场一次运行中，MuJoCo 日志约 49 秒只推进了 9075 个 physics step（约
185 Hz），而命令请求的是 1000 Hz。原因是旧前台循环在每个 physics step 都调用
native viewer `sync()`，并且每条 JSON 事件都立即 flush；同时 source poll 以
仿真时间计时，使 20 Hz source 在该现场实际只有约 3--4 Hz。RM65 读取和连续 seed
IK 也在同一前台 tick 同步执行，因此会进一步造成可见延迟。

当前实现已将 viewer 刷新和 source poll 时钟解耦，并将日志 flush 批量化；每个
`bridge_step` 仍保留 source/retarget/position/total elapsed 字段，所以不会以
降低证据粒度换取速度。下次现场日志应重点统计这些字段，不能仅凭终端刷新速度
判断 IK 性能。

同一现场日志还显示先有一条 `source_admitted=true`、`retarget_accepted=false`
的 tick，之后桥接器进入 `held`，最后操作者才按 Ctrl-C。它不是自动断开或 RM65
网络断线，而是当前目标被 retarget/IK 门拒绝后按 fail-closed 契约撤销 pending
target。此前前台只打印了泛化的 `bridge rejected the target`，无法知道是
`ik_rejected:*`、source freshness 还是 driver rejection。现在日志和首次/周期性
held 提示会记录 `bridge_rejection_reason`、IK status/iterations/errors、IK seed/结果
qpos、source measurement flags/sequence 以及各阶段耗时；在拿到这一证据前，不应放宽
workspace、freshness 或 IK 安全门。

这次具体拒绝是一次有效样本上的 `ik_rejected:line_search_stalled`：source 已准入、
Jacobian 条件数约为 10，随后才进入 `held`。首个假设是上一版每次用 MuJoCo 实测
qpos 作为 IK seed；位置伺服的动态/目标 slew 滞后使 seed 落后于最近已接受目标，连续
拖动时容易跨入不利的局部数值 basin。桥接器已改为“首次激活或 recenter 使用实测
qpos，之后使用上一条已接受 IK qpos”，不改变 source freshness、joint limits、
driver slew 或 fail-closed rejection 契约。

随后对 2026-09-19 的最新现场日志进行了精确复现：新 seed 已经是上一条已接受目标，
source sequence/时间戳也完整新鲜；失败发生在 IK 第 0 次迭代。该目标的初始位置误差
约 0.0107 m、姿态误差约 0.0681 rad、Jacobian 条件数约 9.6，但 DLS 的主导 J4 步约
为 -0.110 rad。旧的 0.05 rad 数值步长上限把它截断后，全部回溯候选都不能降低 merit，
因此错误地进入 `line_search_stalled`。默认数值步长现调整为 0.10 rad；这不是硬件命令
或 MuJoCo 执行速度限制，物理关节限位、位置 driver slew 和 fail-closed 仍保持不变。
该精确 seed/target 已加入 IK 回归测试。后续若仍出现相同状态，应根据每条事件中的
`ik_seed_qpos_rad`、目标误差和 `ik_iterations` 判断是否是新的数值条件，而不能再把它
解释成 RM65 网络断线。

这次修复仍没有引入后台 source/IK worker：live RM65 SDK 读取和 IK 的同步耗时先
通过上述字段量化。若实测单次 source+IK 超过 source period，再单独设计 latest-only
worker，并保留主线程的 freshness recheck 和 fail-closed admission；不能先用旧
qpos 或静默重试掩盖该问题。

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
