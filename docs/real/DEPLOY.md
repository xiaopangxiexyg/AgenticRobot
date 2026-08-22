# Deploy 参数说明

Deploy 消费 policy 输出并组装真实 observation/action/backend；运行命令见
[real README](./README.md)。它与 replay 的动作来源、contract 和人工放行条件不同，不能互换。

## 1. Canonical 入口

先按 [real README 的第 3 节](./README.md) 检查 canonical CLI 参数（R0）。

`python -m teleop_runtime.deploy` 是 shell canonical wrapper。
`python -m teleop_runtime.deploy.run_policy` 仍是受支持的直接/程序化入口，OpenPI launcher
当前依赖其 `main`，但人工命令不应无理由绕过 wrapper。

部署 profile、单臂 dry-run、双臂/双手、dataset replay debug 和运行按键见
[`teleop_runtime/deploy/README.md`](../../teleop_runtime/deploy/README.md)。

## Pose Lift 回退边界

Policy pose 到正式 qpos 的提升使用独立的 versioned profile。默认已提升为：

```bash
--pose-lift-profile rm65_pose_lift_v4_qp_constrained
```

V4 保留 V3 的连续等价解与有界腕部搜索，并把正式
qref 逐关节速度和加速度边界放入 Pink QP，并以“上一条正式 admitted qref + 上一条 admitted
velocity”为外层锚点。一次 reference 内的 2–4 次 Pink 调用只是数值迭代，不会重复获得
运动预算；最终 post-QP 关节/碰撞认证仍独立保留。该 qref QP 是部署参考层约束；120 Hz qcmd planner 的既有
速度/加速度安全合同继续独立生效，不能因 V4 而删除。

结构调整前的 `rm65_pose_lift_v1_baseline` 继续作为冻结的原子回退。需要回退时显式传：

```bash
--pose-lift-profile rm65_pose_lift_v1_baseline
```

用于部署 parity 检查的 Pose replay 默认也选择 V4；QPOS replay 是保存 q_ref 的严格复现，
不经过 Pose Lift。V2/V3 仅保留为中间诊断 profile。

### Position source-cut 执行与回退

兼容的双臂 position deploy 默认使用：

```bash
--position-continuous-waypoint-profile formal_third_order
```

真实部署 trace 显示，`async_c2` bounded worker 在整机负载下可与权威 60 Hz
reference worker 竞争，并导致跨过两个 16.7 ms tick。因此正式默认恢复为
third-order governor；这不改变 official all-IK、chunk/stream-start splice、collision owner 或
15/60/120 Hz 频率。

隔离研究 C2 时才显式传：

```bash
--position-continuous-waypoint-profile async_c2
```

该 profile 仍按完整 source cut 原子选择 C2 或 formal，但只作为可回退实验面；
在 CPU 隔离与真实 executor deadline soak 再次通过前，不得用于真机下发。显式请求
`async_c2` 需要 `--debug`，且运行条件不兼容时会 fail closed。历史 experimental/shadow 参数仅用于
`--debug` 研究对照；任一 C2 Active admission 与 `--send-actions` 的组合会在创建
真机 backend 前被拒绝。

RTC frontend 与声明 `arm_hold_output` 的 policy artifact 不会提供已经完成语义解析的
same-chunk successor。两类运行会在 planner/profile 解析前关闭 chunk lookahead，回退到
`pink_pose_lift` 兼容路径，并将 C2 effective profile 标为 `formal_third_order`（即 C2 关闭）；
显式请求 `async_c2` 会 fail closed。这样不会把
尚未解析的未来 arm-hold 状态送入 IK/C2，也不会在日志中把不可用的 C2 标为 effective。

## 2. OpenPI 两进程部署

OpenPI server 使用 OpenPI 环境，DexTeleop client 使用 `dexteleop310`。Checkpoint 目录必须
包含有效 `policy_artifact.json`。

训练路线、环境分工、checkpoint/profile 绑定、恢复和部署副作用统一在
[`deployments/openpi/COMMANDS.md`](../../deployments/openpi/COMMANDS.md) 维护；
现场两终端最短顺序见 [OpenPI 现场部署](./OPENPI.md)。

`--contract-check-only` 才是不连接 server、相机、手部设备或机械臂的静态检查。
“只观测/推理、不发送动作”的 client 仍可能打开真实 observation/backend；它只因缺少
`--send-actions` 而不下发 policy action，不能称为无硬件检查。

只有在完整 preflight、qpos-init 和人工 ready gate 都确认后，才可增加 `--send-actions`
进入 M 级。不要同时加入 `--auto-start` 或 `--start-init_qpos`。

Server 启动、checkpoint 选择和现场按键以
[OpenPI 现场部署](./OPENPI.md)为准；具体已审计 checkpoint/profile 长命令见
[`deployments/openpi/cmd_list.md`](../../deployments/openpi/cmd_list.md)。

## 3. 验收工具（R1）

Absolute/relative 真机 A/B 完成后可离线比较：

```bash
python -m teleop_runtime.deploy.acceptance \
  "${ABSOLUTE_LOG_DIR:?set ABSOLUTE_LOG_DIR}" \
  "${RELATIVE_LOG_DIR:?set RELATIVE_LOG_DIR}" \
  --output "${ACCEPTANCE_JSON:?set ACCEPTANCE_JSON}"
```

该命令只读取已完成的两组日志并写一份离线报告，不连接机械臂。

参数和通过条件见
[`teleop_runtime/deploy/README.md`](../../teleop_runtime/deploy/README.md#v22-rawv23-relative-真机-ab-闭环验收)
与 [v2.3 验收](../data/ACCEPTANCE_V23.md)。
