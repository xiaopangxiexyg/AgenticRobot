# DexTeleop v2.3 验收模板

本页用于组织证据和 gate，不提供可直接复制的真机运动命令。当前运行入口来自
[真实系统](../real/README.md)、[Replay](../real/REPLAY.md)、[Deploy](../real/DEPLOY.md) 和
[OpenPI](../real/OPENPI.md)；验收页不能成为绕过这些入口安全门的第二套配方。

历史站点的完整 17-gate 表和旧 A/B 命令已归档到
[2026-07-22 验收快照](../legacy/acceptance/ACCEPTANCE_V23_site_2026-07-22.md)。

## 0. 先确定本次范围

| 项目 | 填写 |
|---|---|
| 日期、操作员、安全观察员 |  |
| DexTeleop commit 与 dirty-worktree 说明 |  |
| 数据集 schema / pose-frame contract hash |  |
| 站点、左右 RM65、tool/TCP、相机与末端配置 |  |
| control mode | `position` / `impedance` / 不含真机运动 |
| policy/checkpoint/artifact |  |
| evidence 根目录 |  |
| 本次包含的 gate |  |

开始后不得在 absolute/relative、position/impedance 或多次稳定性运行之间切换代码、配置、
checkpoint 或 tool/TCP。非当前交付范围的 gate 标为 `N/A` 并写原因，不要默认为通过。

## 1. 停止规则

H2/M 阶段必须有现场操作员、可用急停和已清空工作区。出现下列任一情况立即停止：

- 左右 side、方向、control mode、tool/TCP 或 workspace 与预期不符；
- qpos、TCP、速度、力或 command/measured 关系突变；
- collision/backend/IK fault 后仍继续消费或下发 action；
- 相机、measured qpos/TCP 等 required observation 缺失但系统没有 fail-closed；
- pause/recovery 后执行了旧 chunk，或出现持续回拉、振荡、queue backlog。

禁止为了通过验收加入 `--auto-start`、`--start-init_qpos`、绕过 ready confirmation 的参数，
或放宽 decoded/command 的硬门槛。

## 2. Gate A：版本与无硬件基线（R0）

```bash
git status --short
git diff --check

PYTHONPATH=. python -m pytest -q \
  tests/test_collection_pose_frame_storage.py \
  tests/test_causal_snapshot_history.py \
  tests/test_dataset_v22_deploy.py \
  tests/test_openpi_common_deploy.py \
  tests/test_umi_collection.py \
  tests/test_position_deploy_executor.py \
  tests/test_replay_execution_checks.py
```

通过条件是退出码为 0；不固定 `passed` 数量。记录 commit、测试输出和所有已知并发改动。

- [ ] PASS
- [ ] FAIL
- 证据：

## 3. Gate B：站点、坐标系与 measured state（H1）

- [ ] 左右 RM65、IP、side、URDF、collision scene 和 tool/TCP 对应正确。
- [ ] canonical desk 原点/轴向经已知物理点检查，左右臂转换一致。
- [ ] `actual_tcp_pose` 来源为真实 measured qpos FK，不是 command cache。
- [ ] commanded 与 measured 在跟踪过程中能够区分。
- [ ] Q6、DMTAC 和三路相机 identity 没有左右交换。

证据至少包含设备只读检查输出、pose-frame hash 和 command/measured 区分记录。

## 4. Gate C：时钟、T0 与因果快照（H1）

对 arm、hand/gripper、tactile、三路相机和 VIVE 分别确认：

```text
source_timestamp <= T0 AND available_timestamp <= T0
```

- [ ] 没有合格历史时 invalid/fail-closed，不回退 current target 或 last command。
- [ ] 发布后的 snapshot 不会被 backend 后续更新原地改变。
- [ ] telemetry 与 tactile sequence 独立，只有新 frame 才刷新 source timestamp。
- [ ] 多次 cold start/episode restart 的首帧均来自有效 hardware poll。

## 5. Gate D：短数据采集与 finalized validator（M）

按 [正式数采入口](../real/README.md#1-human-teleop-与正式数采) 采 3–5 个短 episode。退出并完成
finalize 后，在离线终端执行：

```bash
python -m utils.dataset_postprocess validate-v23 \
  "${DATASET_ROOT:?set DATASET_ROOT}"
```

- [ ] schema、pose-frame/source id、contract hash 一致。
- [ ] action/state/last_action/current/commanded/actual pose 的 frame 语义一致。
- [ ] parquet、video、tactile、depth/mask frame index 与 episode 边界一致。
- [ ] validator 退出码为 0，报告路径已记录。

## 6. Gate E：离线 chunk、relative round-trip 与 artifact（R1）

- [ ] absolute view 保持 canonical desk absolute。
- [ ] relative chunk 使用 chunk 首行的各侧 measured TCP 作为固定 anchor。
- [ ] `absolute → relative → same-anchor absolute` 只有浮点误差。
- [ ] 使用错误/下一帧 anchor 的负对照产生明显误差。
- [ ] artifact 的 schema、pose mode、layout、camera slots、dims 和 checkpoint 与数据一致。
- [ ] OpenPI contract check 通过；server metadata 与本地 artifact 一致。

此 gate 不连接或驱动机械臂。命令入口见 [Deploy](../real/DEPLOY.md) 和
[OpenPI](../real/OPENPI.md)。

## 7. Gate F：真机 replay/deploy A/B（M）

开始前为每次运动单独写运行记录，至少包含 dataset/episode/frame range、初始化帧、control
mode、速度/步长/工作空间限制、停止方式和日志目录。运行命令必须从当前 real 文档生成，不能
从历史验收快照复制。

两次 A/B 只允许改变被比较项。完成后按
[Deploy 验收工具](../real/DEPLOY.md)生成离线 comparison JSON。

- [ ] decoded target、canonical command、source index 和 cleanup/fault 硬门槛通过。
- [ ] measured TCP/qpos 差异在本次预先批准的门槛内。
- [ ] debug drop、collision hold、backend reject 和未解释 IK timeout 为 0。
- [ ] `passed=true`，且没有通过放宽门槛获得。

Position 与 impedance 必须分开验收；一种 control mode 的 episode/结果不能证明另一种通过。

## 8. Gate G：末端、触觉、相机与故障闭环（H1/M）

- [ ] gripper state/action 方向、范围和左右 side 正确。
- [ ] tactile key、shape、dtype、freshness 和外置引用正确。
- [ ] front/left_wrist/right_wrist 与 artifact slots 一致。
- [ ] required observation 丢失、policy timeout、shape/quaternion 错误、IK/collision/backend fault
  均 fail-closed。
- [ ] recovery/queue clear 后只接受 fresh observation 和 fresh chunk。
- [ ] 单侧双臂故障触发约定的 group hold，不继续单侧追踪。

故障注入涉及断开设备或真实状态变化时，必须单独授权；不能因为它属于验收就默认执行。

## 9. Gate H：性能与长期稳定性

分别记录不开 debug、正式数采 debug 档和部署链路的：

| 指标 | 结果/证据 |
|---|---|
| sample / policy / qref / qcmd 实测频率 |  |
| arm/camera/tactile age 与 drop count |  |
| writer queue、policy queue、IK timeout/hold |  |
| CPU、内存、GPU、server latency |  |
| 正常退出后的 dataset integrity |  |

频率短时波动不能只凭体感判定；保留原始 trace、统计窗口和运行配置。

## 10. 最终判定

| Gate | PASS / FAIL / N/A | 证据路径与备注 |
|---|---|---|
| A 版本/测试 |  |  |
| B 站点/frame/measured |  |  |
| C 时钟/因果快照 |  |  |
| D 采集/validator |  |  |
| E chunk/artifact |  |  |
| F 真机 A/B |  |  |
| G 外设/fail-closed |  |  |
| H 性能/稳定性 |  |  |

最终结论只能是 `PASS`、`CONDITIONAL` 或 `FAIL`。`CONDITIONAL` 必须列出未完成 gate、限制、
负责人和关闭条件；任何当前交付范围内的硬 gate 失败都不能标为 PASS。
