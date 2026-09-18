# ReBot CAN → RM65 MuJoCo GUI 装配进度

本页记录应用接入的源码阶段证据，不修改总路线图。离线验证不等于完成隔离集成或真机验收；
当前 handoff 状态以 Harness Task receipt 为准。它不是新的重力补偿程序，也不证明硬件遥操已经可用。

## 实现与所有权

统一入口为 `python -m control_station b601-rm65`，应用叶子是
`control_station/sim_hil_profile/rebot_b601_rm65_application.py`。

- `assemble_b601_rm65_session()` 只绑定已存在的 backend/model/profile/target；不启动、关闭或驱动注入的资源。
- 复用 `B601ReadOnlyPoseSource`、canonical frame-mapping profile、residual adapter、motion binding/session。
  所有映射必须来自显式 profile；缺少 canonical 模块时在打开 CAN 之前失败，禁止复制实现或默认为 identity。
- 仿真准备复用 `prepare_mujoco_sim_hil_target()`，禁止 task-specific scripted reset；不启动 policy。
  复用 actuated position rig、canonical coupled IK 和原有速度/加速度/关节/碰撞保护，不使用 direct-qpos debug。
- 两臂测量使用 rig 的一次 lock-consistent execution cut 和 private-MjData FK projector。
  保留原测量时间及来源 hash；不是从旧目标或 GUI 缓存反推当前姿态。
- GUI 复用 `MujocoPassiveViewerSession`，由唯一 rig 的 `sync_viewer()` 同步；GUI 不推进物理或重建 IK。
- 前台交互复用已有 terminal loop：`a` 激活或重试、`c` clutch、`r` 重新锚定、`s` 撤销跟随、`q`/EOF 退出。
  在真实控制终端中使用 cbreak 单键输入，按键立即生效，不需要 Enter，也不需要用 shell
  历史重新执行长命令；管道/测试文件描述符仍保留按行协议。初始 inactive，操作者先把
  master 放到合适位置、静置，再显式按 `a` 激活；当前没有反向 haptic channel 或 policy overlay。

反馈能力缺失不会再让这个接收型 GUI 进程退出。已知的
`B601HardwareDependencyError` 会撤销当前 RM65 proposal、保持 session 为 `held`，并在前台提示
修复反馈源后再次按 `a`；旧快照不会被重用，其他异常仍按原边界退出。恢复逻辑属于 Python
session/foreground 状态机，不由 shell 历史命令或后台重启承担。恢复仍需操作者显式按 `a`，
不会在 stale/坏帧后自动恢复跟随，避免数据流刚恢复时未经确认突然产生目标运动。

目标执行 gate 也遵循同一条重试边界。启动 GUI 会先把 RM65 跟随置于 `hold`；操作者按 `a`
时，session 先采集双方当前新鲜姿态，再由 lower target owner 显式执行 `hold → ready`。
切换后必须重新取得 monotonic clock cut，才能看见刚刚准入的 `READY` 事件；不能复用切换前
的旧 cut。若 lower owner 处于 `fault`/`transitioning` 或切换后新鲜度复核失败，目标保持
held、不会提交旧 anchor，前台会记录 gate status/reason 并继续等待操作者修复后重试。该
路径也不触碰独立的 B601 重力补偿/阻尼 owner。

CAN backend 以默认 `allow_motion=False`、`allow_active_reporting=False` 创建，只接收**独立 owner 已开启**的
完整新鲜六轴反馈。它不使能、不切模式、不配置 report、不发送 MIT/position/torque、不操作 J7。
现有 master 重力补偿/阻尼 owner 必须独立运行；应用退出仅撤销仿真跟随并关闭自己的只读连接。
绝不能把 master `safe_stop()` 加入应用的普通 `finally`。

## MotorBridge 运行时能力选择

连接型入口在打开 CAN 前会记录 MotorBridge 运行时来源。若当前 checkout 中同时存在锁定的
`hardware/MotorBridge/bindings/python/src` 和已构建的 `target/release/libmotor_abi.so`，应用会把该
绑定放到 import 路径首位；显式 `MOTORBRIDGE_LIB` 仍是操作者指定的 override。结果 JSON 的
`teleop_contract.motorbridge_runtime_selection` 和 `motorbridge_runtime_evidence` 用于审计实际加载的
来源与 ABI 能力，应用不会把系统 site-package 静默宣称为仓库版本。

`state_observation_metadata_v1` 是 host-fresh B601 反馈的硬门禁。若能力未宣告，GUI 仍可保持
receive-only/held 并允许 `a` 重试，但不会降低 `feedback_timeout_sec`、转用旧 cache 或提交 RM65
目标；应修复/重启到包含该能力的 MotorBridge 绑定后再重试。

## 遥操幅度与 RL 边界

这条入口是 B601 只读位姿到 RM65 MuJoCo 的直接遥操，不是 RL action 执行器。它提交的是
anchor-relative 的 `cartesian_pose`，经已有 coupled IK、关节范围、碰撞、速度、加速度和目标
生命周期门控；`right_tcp_delta_m`、`right_tcp_rotation_delta_rad`、`right_joint_delta_rad`
等 RL 归一化动作尺度不会参与这条路径，也不应修改 RL task JSON 来放大遥操幅度。

位移倍率由外部 canonical frame-mapping profile 的
`translation_gain_target_reference_xyz` 负责。当前 B601 工作空间小于 RM65，现场遥操 profile
应显式使用 `[1.5, 1.5, 1.5]`（旋转倍率仍为 `1.0`），并在结果 JSON 中核对
`translation_gain_target_reference_xyz` 和 profile hash。旧的 `[0.25, 0.25, 0.25]` profile
会缩小运动，只能作为历史对照，不能用于本轮幅度验收。

可在不改动原始标定文件的情况下生成 150% 版本（省略旧 hash，让 canonical loader 重新计算）：

```bash
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
PROFILE_IN=/tmp/b601-rs-commissioning/provisional_b601_rm65_mapping_v0.json
PROFILE_150=/tmp/b601-rs-commissioning/b601_rm65_mapping_150_v1.json
"$PY" - "$PROFILE_IN" "$PROFILE_150" <<'PY'
import json
import sys

source, target = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    payload = json.load(handle)
payload["profile_id"] = "b601-rm65-provisional-lab-150-v1"
payload["translation_gain_target_reference_xyz"] = [1.5, 1.5, 1.5]
payload.pop("semantic_content_hash", None)
with open(target, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(target)
PY
```

### B601 末端工具坐标校正（150% v2）

上面的 `v1` 只解决位移倍率，且把 `gripper_end` 与 `tool_tcp` 的受控坐标系暂时当作
同向（controlled matrix 为 identity）。这会让两个旋转轴在 RM65 上互换。B601 URDF 的
`j_gripper_end` 固定关节相对 `link6` 使用 `rpy="3.1416 -1.5708 0"`，因此当前受控帧
合同必须显式包含：

```text
T_source_controlled_target_controlled =
[[ 0,  0,  1, 0],
 [ 0, -1,  0, 0],
 [ 1,  0,  0, 0],
 [ 0,  0,  0, 1]]
```

该矩阵只作用于 `gripper_end → tool_tcp` 的受控旋转，不改变基座参考矩阵或位移倍率。
因此源端小旋转的离线预期是：`+X → +Z`、`+Y → -Y`、`+Z → +X`；位移仍按
`[1.5, 1.5, 1.5]` 解释。可复用的版本化 profile 位于：

```text
docs/hardware/profiles/b601_rm65_mapping_150_v2.json
```

其 `semantic_content_hash` 为
`6d19eb1e0df109b050ca824d7031721994f54021491304db9e49712185a0365e`。现场验证必须显式
使用该文件，并在结果 JSON 中记录 profile id/hash；不要继续使用 identity 的 `v1`。

仿真只读连接可显式传 `--disable-teleop-residual-envelope`。它仅关闭 B601 binding 的
“单 tick residual 平移/旋转幅度”预检，不关闭 RM65 的 canonical IK、关节、碰撞、速度、加速度、
执行时域或 viewer fault 保护；该选项被限制为 `--connect-read-only`，不得用于真机运动 owner。

## 反馈过期诊断

默认 `feedback_timeout_sec=0.10` 仍属于 B601 backend/真机 owner 的安全契约。20 Hz 报告周期约
50 ms，若一次 report 调度或 USB/CAN 接收间隔丢失，再叠加几毫秒调度抖动，host cache age 就会
从约 50 ms 跳到 100 ms 以上；六个关节同时超时通常是整条 operation-status stream 的间隙，
不是六个电机同时故障。`feedback-stability` 的 p50/p95 通过也不排除单 tick 的最大 age 超时。

为 RM65 仿真跟随，应用提供仅作用于本进程的
`--simulation-feedback-timeout-sec` 覆盖；推荐 `0.20`，同时将
`--maximum-source-age-sec` 设为 `0.25`，保留约 50 ms 的 source TTL 余量。该覆盖通过
`dataclasses.replace` 只修改 GUI 进程内的 backend config，不改 YAML、真机 owner、active
report 或 watchdog。每个 `proposal_admission`/`target_step` 日志事件会带
`feedback_snapshot`，包括每关节 receive age、最大 age、timeout、余量、sequence 和
`fault_reason`，用于区分偶发报告间隙与持续 stale。

`--validate-only` 不打开 CAN/GUI，不启动 worker，只构造、校验契约和模型；
`--connect-read-only` 必须提供 `--confirmation B601_RM65_READ_ONLY_CAN_SIM_TARGET`。
这只是连接确认，不是 motor/report/motion 授权。
日志显式用 `--log` 指定，可追加实际 argv、状态和错误 traceback；结果 `--output` 必须是尚不存在的新文件。
JSON 的 `target_cleanup_completed` 只有 lower runtime 的 `close()` 成功返回才设为 true。

## 已修复的时序缺陷（2026-09-16）

初次标准离线回归为 **50 passed、2 failed**；失败没有被忽略或改成 skip。

1. 激活/重新锚定发生在已到期 tick 时，现有前台循环立刻再次 `session.step()`。
   一次缓存尚未更新就被二次读取，新鲜度推进检查正确拒绝，session 进入 held。
   原前台 owner 现在将激活/recenter 后的首个发布安排到下一个周期；不绕过重复帧检查。
2. `B601RM65MotionSession.step()` 在 residual adaptation **之前**取得默认 target time；
   binding 随后再取真实 monotonic clock，严格的 `now <= target_effective_at` 正确拒绝过去时间。
   synthetic 固定时钟隐藏了此问题。本轮真实 MuJoCo rig/时钟测试已复现。
   现在由原 admission owner 在最终校验时选择默认生效时间；显式时间验证、原 sample TTL 和单次 source poll 均保留。

修复需要原文件：

```text
control_station/sources/rebot_b601_rm65_motion.py
control_station/sim_hil_profile/rebot_b601_rm65_foreground.py
tests/test_control_station_rebot_b601_rm65_motion.py
```

上述路径经用户明确授权后由原生 `scope-extend` 纳入当前 Task；未手改 Task/coordination 元数据，
也没有在 GUI 层增加第二套 session 来绕过门禁。修复后联合回归为 **54 passed**。

当前基线 `37077731` 同时缺少 development 已有的 canonical frame-mapping profile；
最终组合需包含 predecessor 和 profile 两项，不能从其他 worktree 的绝对路径动态导入 tracked 源码。
predecessor 未集成只阻塞最终 integration，不影响当前 scope 内开发。

## 仿真跟随性能与长时运行修复（2026-09-17）

现场日志暴露了两个相互独立的问题：旧候选仍以 15 Hz 运行，且 canonical
arbitration 每个 tick 都会对不断增长的 proposal snapshot 做哈希；后者在长时运行中
把单次 target step 的计算拖慢，最终表现为“移动一段时间后不再跟随”。同一日志中的
feedback age 仍在约 0.01--0.05 s，说明该次停止首先不是 CAN stale，而是运行时保留与
调度路径的问题。

当前候选作了以下收敛，均保持既有安全边界：

- B601→RM65 仿真目标显式使用已验证的 `POSITION_60_120_PROFILE_NAME`，目标/IK 时钟为
  60 Hz，底层 qref/qcmd 仍由已有 60→120 Hz MuJoCo position rig 负责；B601 约 20 Hz
  的重复缓存帧只在 observation sample、时间、姿态、host freshness 和 source-age 均满足
  条件时做有界 zero-order hold，绝不把过期帧当作新样本。
- coupled RM65 planner 复用 `robot_motion.rm65` 中已有的
  `RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S` 和 acceleration profile，避免继承
  面向真实侧的 `0.35--0.50 rad/s` 保守默认值；这只消除仿真侧的额外限速，不改变真实
  B601 owner 的控制参数。
- planner 的 collision minimum distance 从任务 scene contract 读取（当前任务为约
  `2 mm`），碰撞约束仍开启；不再把真实侧的 `30 mm` 余量误用于 MuJoCo，从而减少无碰撞
  位置被提前 hold 的情况。
- B601 目标 runtime 对已过期 proposal 做 owner-side 有界清理，保留 producer sequence
  high-water mark 和不可变快照语义；通用 MuJoCo runtime 默认仍保持原 append-only 行为。
- `target_step` 记录是否复用了 source frame；终端输出管道关闭时，JSON/log 仍会完整落盘，
  避免用空结果文件掩盖真正的运行时故障。

本轮 focused/相关回归为 **154 passed、2 skipped**（未绑定 reviewed RM65 assets 时的
明确可选 skip），并通过 `py_compile`、Ruff（变更的 B601 app/source/test 文件）及
`git diff --check`。尚未运行真实 CAN 或 GUI 人工验收；candidate 仍需在完整组合 checkout
中重新执行 integration Profiles。运行时日志中的 `source_reused`、`target_step.status`、
`feedback_snapshot.max_receive_age_sec` 和 `resolution_failure`（若有）应一并用于下轮判断
是输入时序、IK/碰撞门控还是 viewer/worker 问题。

## 当前离线复核命令

在当前 Task 源码根，使用标准项目 Python 和 approved asset binding：

```bash
bash scripts/run_pytest.sh -q \
  tests/test_control_station_rebot_b601_rm65_application.py \
  tests/test_control_station_rebot_b601_rm65_motion.py
```

当前预期为 **153 passed、2 skipped**（环境没有 reviewed RM65 assets 时保留可选跳过）。
测试覆盖 injected owner 生命周期、完整当前姿态锚点、GUI close/fault、lazy CLI help、
参数/confirmation 拒绝、离线模式不连接 CAN、日志 argv、Ctrl-C、target cleanup，以及真实模型 actuated rig 的 atomic FK measurement。
外部构造在生命周期测试中 mock；坐标 profile 用 synthetic port，不代表真实安装姿态标定。

## 后续验收顺序

1. 在完整 clean commit 上运行 motion/sim 及 impact-added Profiles；全部通过后才可 handoff。
2. 由 integrator 组合 predecessor、canonical mapping profile 和应用候选，重跑 exact combined-source 门禁。
3. 提供现场操作者的完整 foreground 命令：先离线 validation，再独立 master gravity/damping 与新鲜 report 预检。
4. GUI 首先 inactive，确认 RM65 两臂模型及 TCP；`a` 后小幅平移、逐轴旋转，确认只有选中右臂按相对运动跟随。
5. `c` 后移动 master 不应改变 follower；`r` 从双方当前实测姿态重新锚定，不得跳回旧目标。
6. stale/坏帧/越界/GUI fault 撤销仿真跟随并要求重新预检；`s`、`q`、Ctrl-C 不得影响 master 的重力补偿。
7. master 的首次 Ctrl-C protective hold 和再次 Ctrl-C/SIGUSR1 全臂掉能仍只由其原硬件控制 owner 处理。

真机和 GUI 观察均为 **NOT_RUN**。隔离集成的完整组合门禁通过后，才提供包含现场安全预检的
live CAN 验收命令；实际执行仍由现场操作者负责。本 Task worker 不合并、不 push、不安装。
