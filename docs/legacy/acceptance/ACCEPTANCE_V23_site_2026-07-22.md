# DexTeleop v2.3 全流程真机验收（历史站点版本）

> 本文保留 2026-07-22 时的固定主机、数据集、checkpoint、测试数量和真机 A/B 命令，仅用于
> 追溯。当前可移植且保留人工安全门的模板见
> [`docs/data/ACCEPTANCE_V23.md`](../../data/ACCEPTANCE_V23.md)。不要直接执行本文中的
> `--auto-start` 或绕过 ready confirmation 的历史命令。

> 目标：验证 v2.3 从真机采集、因果快照、坐标系、数据落盘、absolute/relative
> action chunk、replay、模型服务到 RM65 真机执行的完整闭环。
>
> 本文是执行和记录入口。每一阶段均应填写结果和证据路径；任一硬 gate 失败时，停止进入
> 后续真机运动阶段，不能通过关闭校验或放宽 decoded/command 门槛绕过。

## 0. 验收信息

| 项目 | 填写 |
| --- | --- |
| 日期 |  |
| 操作员 |  |
| 现场安全观察员 |  |
| DexTeleop commit / 工作树说明 |  |
| OpenPI commit / checkpoint |  |
| 远端主机 | `user-B650M-AORUS-PRO-AX` |
| DexTeleop 根目录 | `/home/user/zc/DexTeleop` |
| OpenPI 根目录 | `/home/user/zc/openpi` |
| Conda 环境 | `dexteleop310` |
| 左臂 IP | `192.168.101.21` |
| 右臂 IP | `192.168.1.20` |
| 验收日志根目录 | `/home/user/zc/DexTeleop/logs/v23_acceptance` |

最终结论：

- [ ] PASS：所有必选 gate 通过
- [ ] CONDITIONAL：仅非阻塞项未完成，已列明限制
- [ ] FAIL：存在硬 gate 失败

限制和备注：

```text

```

---

## 1. 风险等级和停止规则

| 等级 | 含义 | 示例 |
| --- | --- | --- |
| L0/L1 | 只读、无硬件状态变化 | artifact 检查、数据验证、日志分析 |
| L2/L3 | 非运动写入或进程运行 | 采集日志、启动 OpenPI server、相机 dry-run |
| L4 | 设备状态变化但不计划运动 | 设备 enable/disable、USB reset |
| L5 | 真实运动 | qpos-init、replay、策略闭环、夹爪动作 |

任何 L5 步骤开始前必须满足：

- [ ] 操作员和观察员均在场
- [ ] 工作区无人、无临时障碍物
- [ ] 急停可达且已确认有效
- [ ] 左右臂 IP、side、tool/TCP profile 已复核
- [ ] 速度、步长、力阈值和工作空间限制已审查
- [ ] 已明确本次使用 impedance 还是 position，不能混用 episode/control-mode
- [ ] 已准备随时暂停/退出，机械臂异常时立即急停

统一停止条件：

- 任一机械臂方向与预期不符；
- TCP、qpos、力或速度突变；
- 反复出现“动一下又回去”；
- queue/recovery 循环、持续丢弃新 chunk；
- collision hold、backend reject、通信断开；
- measured TCP 疑似等于 command 而非硬件反馈；
- required 相机/状态缺失但系统仍继续下发。

---

## 2. 环境准备和证据目录

在真机所在远端终端执行：

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310

mkdir -p /home/user/zc/DexTeleop/logs/v23_acceptance
python -m teleop_runtime.deploy --help
python -m deployments.openpi.client.real --help
```

预期：

- 两个入口均正常显示帮助；
- OpenPI client help 包含 `--contract-check-only`；
- 没有连接或初始化机械臂。

### 2.1 代码与并发工作树 preflight

当前若同时存在 v2.4 多源、observation firewall、action arbitration 或 control trace
改动，必须先固定本次验收使用的 commit 和 dirty-worktree 说明。不要在 absolute/relative
两次真机运行之间切换代码、同步分支或继续修改共享部署文件。

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310

git status --short
git diff --check

PYTHONPATH=. python -m pytest \
  tests/test_dataset_pose_chunk.py \
  tests/test_pose_deployment.py \
  tests/test_v23_virtual_deploy_equivalence.py \
  tests/test_v23_real_deploy_acceptance.py \
  tests/test_dataset_v22_deploy.py \
  tests/test_collection_pose_frame_storage.py \
  tests/test_causal_snapshot_history.py \
  tests/test_tactile_external_storage.py \
  tests/test_deploy_canonical_action.py \
  tests/test_deploy_action_emission.py \
  tests/test_policy_observation_view.py \
  tests/test_policy_observation_firewall_activation.py \
  tests/test_openpi_artifact_preflight.py \
  tests/test_control_core_baseline.py \
  tests/test_observation_cut_contracts.py \
  tests/test_action_arbiter.py \
  -q
```

当前本地基线预期为 `128 passed, 3 skipped`。若测试集合本身随后发生了有意修改，数量
可以变化，但必须满足：退出码为 `0`、无 failed/error，并把新数量、commit 和变更原因
记录到本节。`git status --short` 允许显示已知的并发工作，但必须逐项确认与本次验收的
共享路径；`git diff --check` 必须无输出且退出码为 `0`。

特别确认：

- [ ] v2.4 model-facing observation 改动没有改变 relative 解码所用的原始 measured TCP anchor
- [ ] IK recovery 丢弃预测时 replay cursor 事务回滚，恢复后只接收 fresh chunk
- [ ] absolute/relative 两次运行使用完全相同的工作树快照
- [ ] 未通过删测试、放宽门槛或关闭 fail-closed 校验获得通过

结果：

- [ ] PASS
- [ ] FAIL
- 证据/输出：

```text

```

---

## 3. Gate A：站点、坐标系和 TCP

### A1. 站点身份

检查并记录：

- [ ] 左右 RM65 型号、IP、side 映射正确
- [ ] source/profile 属于当前 dual-arm station
- [ ] 左右 tool/TCP profile 正确
- [ ] URDF 与 collision scene 版本正确
- [ ] 左右 DMTAC 序列号和端口未交换
- [ ] front、left_wrist、right_wrist 逻辑相机映射正确

记录：

```text
left arm:
right arm:
left tool/TCP:
right tool/TCP:
left DMTAC:
right DMTAC:
cameras:
URDF/collision profile:
```

### A2. canonical desk

canonical desk 必须是已标定 URDF 的 `workspace_near_boundary_frame`，不是历史
`vive_retarget_frame`。

- [ ] 原点物理位置为桌面表面中心
- [ ] xyz 轴方向与定义一致
- [ ] 左右 raw base→desk 使用各自变换
- [ ] 左右臂观察同一物理点时 canonical desk 坐标一致
- [ ] collection desk contract 与 deployment runtime raw contract 配对正确

建议记录 workspace near-boundary 原点和至少 3 个已知物理点：

| 点 | 物理期望 desk xyz | 左臂转换结果 | 右臂转换结果 | 误差 | 结果 |
| --- | --- | --- | --- | --- | --- |
| center |  |  |  |  |  |
| +x |  |  |  |  |  |
| +y |  |  |  |  |  |
| corner |  |  |  |  |  |

### A3. measured TCP 来源

- [ ] `hardware_qpos/measured_qpos` 来自真实 RM65 poll
- [ ] `actual_tcp_pose` 来源为 `hardware_qpos_fk`
- [ ] 独立 FK 与落盘 `actual_tcp_pose` 一致
- [ ] `actual_hardware_pose_6d` 只作为 controller-base 审计字段
- [ ] commanded pose 与 measured TCP 可在跟踪过程中区分

必须做一次 command/measured 区分观察：目标刚下发但机械臂尚未跟到时，
`actual_tcp_pose` 应连续跟随真实反馈，不能瞬间跳到 command。

结果：

- [ ] PASS
- [ ] FAIL
- 证据路径：

```text

```

**Gate A：**

- [ ] PASS，可以继续
- [ ] FAIL，停止；禁止采集正式 v2.3 或执行 relative action

---

## 4. Gate B：统一时钟、T0 和不可变历史

本阶段连接传感器并记录，尽量不执行任务运动。

### B1. Vive 双机校时

- [ ] 数采前完成 sender/receiver clock-sync
- [ ] sender monotonic capture time 映射到接收机 wall time
- [ ] `source_timestamp` 使用校准后的本机时间域
- [ ] `available_timestamp` 表示本机完整接收、解析、发布时间
- [ ] sender 重启或重连后旧 calibration 不会继续使用
- [ ] offset、RTT、校准状态和 fallback 原因可审计

### B2. 每个 stream 的 causal cut

逐项检查：

| Stream | source_timestamp≤T0 | available_timestamp≤T0 | sequence 可审计 | 缺失时 invalid | 结果 |
| --- | --- | --- | --- | --- | --- |
| left arm |  |  |  |  |  |
| right arm |  |  |  |  |  |
| left hand telemetry |  |  |  |  |  |
| right hand telemetry |  |  |  |  |  |
| left tactile |  |  |  |  |  |
| right tactile |  |  |  |  |  |
| front RGB |  |  |  |  |  |
| left wrist RGB/depth/mask |  |  |  |  |  |
| right wrist RGB/depth/mask |  |  |  |  |  |
| Vive |  |  |  |  |  |

严格规则：

```text
source_timestamp <= T0 AND available_timestamp <= T0
```

不允许：

- 在 observation/writer 组装时直接读取已更新的 mutable backend state；
- 没有合格历史时回退到 current/target/last command；
- 物理采样早但网络到达晚的数据进入当前 T0；
- 重复轮询同一个 tactile/telemetry 缓存却伪装成新 source frame。

### B3. 快照不可变性和性能

- [ ] arm/hand/tactile 历史发布时做深拷贝
- [ ] 发布后的原数组更新不会改变旧 snapshot
- [ ] tactile 仅在新 frame 到达时复制大数组
- [ ] telemetry 与 tactile 使用独立 snapshot sequence
- [ ] 历史容量足够，不发生正常负载下频繁覆盖
- [ ] 运行频率和 CPU/内存没有显著回退

### B4. episode 首帧

至少执行 5 次冷启动和 10 次 episode restart：

| 次数 | 首帧有 hardware poll | qpos 有效 | actual TCP 有效 | timestamp 合格 | commanded fallback=否 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |

其余 restart 记录：

```text

```

**Gate B：**

- [ ] PASS，可以继续
- [ ] FAIL，停止；不能宣称 causal snapshot contract 已验收

---

## 5. Gate C：native v2.3 短数据采集

先采 3～5 个短 episode，每个约 10～30 秒。正式命令应使用当前项目 canonical collection
入口和已审查的设备配置；不要为了验收创建新的 production entrypoint。

### C1. 默认 desk_absolute 数据

- [ ] schema 为 `dexteleop_dataset_v2.3`
- [ ] storage frame 为 `desk_absolute`
- [ ] source id 正确
- [ ] dataset 内 pose-frame contract 不变
- [ ] dataset/episode/row contract hash 一致
- [ ] tactile compression 默认为 `none`
- [ ] measured TCP 未过期且来源为 hardware qpos FK

数据集路径：

```text

```

### C2. 同一帧 pose 副本一致性

以下字段必须在同一 storage frame 中一致：

- [ ] `action`
- [ ] structured arm action pose
- [ ] `observation.state`
- [ ] `observation.arm.pose`
- [ ] `observation.last_action`
- [ ] `current_tcp_pose`
- [ ] `commanded_tcp_pose`
- [ ] `actual_tcp_pose`

以下字段不得进行 raw→desk 转换：

- [ ] `actual_hardware_pose_6d`
- [ ] controller/base-frame 审计字段
- [ ] wrist camera base-frame 审计 pose

### C3. 外置媒体

- [ ] parquet row 与 video frame index 对齐
- [ ] 相机 source/available timestamp 均不晚于 T0
- [ ] tactile sensor key、shape、dtype 正确
- [ ] tactile manifest frame 数与 row 引用一致
- [ ] 无压缩 tactile 文件长度正确
- [ ] wrist depth/mask 不跨 episode、不串帧

### C4. 可选 forced raw 短数据

- [ ] 已采一份短 `raw_absolute` v2.3 数据
- [ ] metadata 明确标记 raw
- [ ] raw 数值未被转换
- [ ] 通过同一 pose contract 转换后与 desk canonical 数值一致

forced raw 数据路径：

```text

```

### C5. finalized dataset validator

对每份正式 v2.3 数据运行：

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310
python -m utils.dataset_postprocess validate-v23 /path/to/v23_dataset
```

预期：退出码 `0`，并通过：

- metadata/hash；
- pose-frame binding；
- measured TCP/state/action 副本；
- 全部 `<=T0` causal cut；
- episode/parquet 计数；
- 外置 artifact 引用。

| 数据集 | validator exit | 报告路径 | 结果 |
| --- | --- | --- | --- |
| desk_absolute |  |  |  |
| raw_absolute（可选） |  |  |  |

**Gate C：**

- [ ] PASS，可以继续
- [ ] FAIL，停止；不得将该数据用于 v2.3 训练交付

---

## 6. Gate D：离线 action chunk 和 round-trip

使用 Gate C 采集的真机数据，不连接硬件。

### D1. absolute view

- [ ] state 是 canonical desk absolute
- [ ] action 是 canonical desk absolute
- [ ] 手部数值未被 pose 转换影响
- [ ] window 不跨 episode
- [ ] 默认 `tail_policy=drop`
- [ ] 不足 horizon 的尾部没有隐式 padding

### D2. relative view

- [ ] anchor 是 chunk 首行 measured `actual_tcp_pose`
- [ ] chunk 内所有 action 共用同一个 anchor
- [ ] 左右臂使用各自 anchor
- [ ] 相邻 chunk 使用各自 chunk-start anchor
- [ ] `last_action` 使用当前 observation anchor
- [ ] 不使用上一行或上一 chunk anchor

### D3. round-trip 和负对照

必须检查：

```text
desk absolute action
  -> relative action
  -> same anchor 解回 desk absolute
```

预期：平移/旋转仅有浮点误差。

负对照：将 relative chunk 用下一帧 anchor 解码，必须产生明显误差，证明测试能够发现
错误 anchor。

| 项目 | 最大平移误差 | 最大旋转误差 | 结果 |
| --- | --- | --- | --- |
| 正确 anchor round-trip |  |  |  |
| 下一帧 anchor 负对照 |  |  |  |

### D4. 虚拟部署等价测试

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310
PYTHONPATH=. python -m pytest tests/test_v23_virtual_deploy_equivalence.py -q -s
```

预期：测试通过，输出 `V23_VIRTUAL_DEPLOY_REPORT=...`；absolute/relative 最终 raw command
误差小于 `2e-6`，错误 anchor 负对照大于 `5 mm`。

**Gate D：**

- [ ] PASS，可以继续真机 A/B
- [ ] FAIL，停止；不要进入机械臂执行

---

## 7. Gate E：policy artifact 和 OpenPI 适配

### E1. native v2.3 模型

- [ ] `pose_training_data_contract.origin=native_v23_dataset`
- [ ] runtime 默认或显式使用 RM65 workspace-near-boundary raw→desk contract
- [ ] artifact pose-frame hash 与训练数据一致

### E2. v2.2 raw→relative 模型

当前目标模型：

```text
model_id: 0715-cube-front-ready-mix-4to1-relative
checkpoint: 149999
```

必须满足：

- [ ] artifact 包含 `pose_training_data_contract`
- [ ] `origin=v22_raw_identity_compatibility`
- [ ] `source_dataset_schema=dexteleop_dataset_v2.2`
- [ ] `expected_runtime_source_id=rm65_real_dual_arm_v1`
- [ ] action/last_action pose mode 均为 `relative`
- [ ] pose-frame hash 是同 source 的 identity contract
- [ ] OpenPI registry 已登记该 model/checkpoint
- [ ] registry state_dim=32、action_dim=16、horizon=50
- [ ] camera slots 为 front/left wrist/right wrist

OpenPI artifact 路径：

```bash
export OPENPI_ARTIFACT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-relative/0715-cube-front-ready-mix-4to1-relative/149999/policy_artifact.json
```

只读 contract 检查：

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310
python -m deployments.openpi.client.real \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --contract-check-only
```

预期 JSON：

```text
ok=true
action_pose_mode=relative
pose_training_origin=v22_raw_identity_compatibility
runtime_pose_frame_contract_source=v22_identity_compatibility
state_dim=32
action_dim=16
```

### E3. websocket metadata

OpenPI server 启动后，在不发送动作时检查：

- [ ] model id 与 artifact 一致
- [ ] checkpoint step 为 149999
- [ ] layout hash 一致
- [ ] state dim=32
- [ ] action dim=16
- [ ] horizon=50
- [ ] 三路 camera slots 一致
- [ ] server artifact 与本地 sidecar 一致

**Gate E：**

- [ ] PASS，可以进入模型 dry-run/闭环
- [ ] BLOCKED：OpenPI artifact/registry 尚未适配
- [ ] FAIL：存在语义冲突，停止

---

## 8. Gate F：真机 absolute/relative impedance A/B replay（L5）

> 本节命令会产生真实机械臂运动，只允许操作员在真机终端现场执行。

### F1. 共同前置条件

- [ ] episode 0/frame 0 和 `[0,20)` 已人工确认安全
- [ ] metadata 声明 impedance
- [ ] 两次运行使用相同代码、数据、初始化、tool 和安全参数
- [ ] 两次运行之间机械臂完全停止并重新初始化
- [ ] 唯一变化是 absolute/relative pose-mode 和 debug 目录

### F2. absolute run

```bash
cd /home/user/zc/DexTeleop
conda activate dexteleop310

python -m teleop_runtime.deploy \
  --send-actions \
  --active-sides left right \
  --end-effector-sides none \
  --camera-mode none \
  --left-arm-ip 192.168.101.21 --right-arm-ip 192.168.1.20 \
  --arm-control-mode impedance \
  --debug-replay-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --debug-replay-episode 0 \
  --debug-replay-start-index 0 --debug-replay-end-index 20 \
  --debug-replay-chunk-size 4 \
  --init-source replay \
  --policy-action-pose-mode absolute \
  --debug-replay-output-pose-mode absolute \
  --debug --debug-dir /home/user/zc/DexTeleop/logs/v23_acceptance/replay_ab/absolute \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --auto-start --no-wait-for-ready-confirm --max-steps 600
```

### F3. relative run

使用完全相同命令，仅替换：

```text
--policy-action-pose-mode relative
--debug-replay-output-pose-mode relative
--debug --debug-dir /home/user/zc/DexTeleop/logs/v23_acceptance/replay_ab/relative
```

### F4. 只读日志比对

```bash
python -m teleop_runtime.deploy.acceptance \
  /home/user/zc/DexTeleop/logs/v23_acceptance/replay_ab/absolute \
  /home/user/zc/DexTeleop/logs/v23_acceptance/replay_ab/relative \
  --expected-actions 19 \
  --output /home/user/zc/DexTeleop/logs/v23_acceptance/replay_ab/acceptance.json
```

通过门槛：

| 检查 | 默认门槛 | 实际 | 结果 |
| --- | --- | --- | --- |
| decoded runtime-absolute target | `2e-6 m / 0.002°` |  |  |
| canonical command target | `2 mm / 0.5°` |  |  |
| measured TCP trajectory | `30 mm / 5°` |  |  |
| measured qpos 单关节最大差 | `0.15 rad` |  |  |
| source index | `1..19` 完全相同 |  |  |
| policy/dispatch/cleanup fault | 0 |  |  |
| debug event drop | 0 |  |  |

要求退出码 `0` 且 `passed=true`。decoded 和 command 是硬门槛，不能放宽。

**Gate F：**

- [ ] PASS
- [ ] FAIL，停止后续模型运动

---

## 9. Gate G：impedance/力控运行行为（L5）

结合 Gate F 日志逐项确认：

### G1. 初始化状态机

- [ ] qpos-init 使用 measured hardware qpos
- [ ] 完成阈值为当前约定 `0.1 rad`
- [ ] 达到阈值后从 final_hold 正确 release
- [ ] replay first frame 初始化和 policy source index 正确
- [ ] `g/A`、auto-start、ready gate 不死锁

### G2. action queue 和 IK recovery

- [ ] chunk 内 anchor 固定
- [ ] 新 chunk 不会拉回旧 anchor
- [ ] 空队列时不重新发送 pose
- [ ] IK recovery 不消耗 replay/model chunk
- [ ] recovery 期间旧预测不会在恢复后执行
- [ ] stable 后等待 fresh policy chunk
- [ ] queue clear/resume 后无陈旧动作
- [ ] 不出现持续 `discarded predicted action chunk`

### G3. backend 跟踪

- [ ] reference target、decoded target、IK qpos、safe command、measured qpos/TCP 均有日志
- [ ] IK timeout 时保持最后安全 command
- [ ] 左右任一侧 fault 时进入 group hold
- [ ] 没有周期性“原地动一下又回去”
- [ ] measured TCP 不等于 command cache

证据路径：

```text

```

**Gate G：**

- [ ] PASS
- [ ] FAIL

---

## 10. Gate H：position 链路独立验收（L5）

> 必须选择 metadata 明确为 position 且人工确认安全的 episode，不能使用 impedance episode
> 绕过 control-mode binding。

position 数据集/episode/frame range：

```text

```

### H1. 频率和因果插值

- [ ] policy/action 实测约 15 Hz
- [ ] pose reference/coupled IK 实测约 60 Hz
- [ ] qcmd 实测约 120 Hz
- [ ] 插值不使用未来 chunk
- [ ] chunk 边界没有固定多一个 action-period 延迟
- [ ] 落后时不补发过期 burst

### H2. authoritative reset

- [ ] 双臂 `hardware_read_success=true`
- [ ] qpos freshness ≤ `0.25 s`
- [ ] 双臂 timestamp skew ≤ `0.05 s`
- [ ] reset 使用 measured hardware qpos
- [ ] 真机不回退 state vector、command 或默认姿态

### H3. coupled IK 和碰撞

- [ ] 双臂 coupled planner 启用
- [ ] joint/step limit 后重新 collision check
- [ ] backend receipt 无 rewrite/reject/singularity hold
- [ ] 任一侧失败触发 group hold
- [ ] `endpoint_collision_checked=true`
- [ ] 明确记录 `segment_collision_certified=false`
- [ ] `debug_event_drop_count=0`
- [ ] `collision_hold_count=0`（正常验收轨迹）

### H4. position absolute/relative A/B

按照 Gate F 的方法执行同一安全 position 轨迹两次，并用同一 acceptance 工具比较；另要求：

- [ ] 无 `position_executor_*fault`
- [ ] 无 dispatch reject
- [ ] 无 collision hold
- [ ] 无 debug event drop

报告路径：

```text

```

**Gate H：**

- [ ] PASS
- [ ] BLOCKED：尚无已确认安全的 position episode
- [ ] FAIL

---

## 11. Gate I：手部、触觉和三相机联合验收

### I1. DMTAC 手部

- [ ] 左右设备未交换
- [ ] state telemetry 来自对应侧硬件
- [ ] action target 范围、方向和开合语义正确
- [ ] hand callback 不阻塞 arm reference loop
- [ ] callback 失败产生明确 fault/hold
- [ ] arm/hand 非原子边界有 debug receipt

### I2. tactile

- [ ] 默认只保存原始 tactile 数组
- [ ] shape、dtype、sensor key 正确
- [ ] 默认 compression 为 `none`
- [ ] 只在新 tactile frame 到来时复制
- [ ] telemetry/tactile snapshot sequence 独立
- [ ] 缺失、过期、晚于 T0 时 availability/mask 正确

### I3. cameras

- [ ] front/left_wrist/right_wrist 逻辑键正确
- [ ] RGB/HWC/uint8 正确
- [ ] artifact 声明的 required views 全部存在
- [ ] source/available timestamp 均不晚于 T0
- [ ] 断流时 required view fail-closed
- [ ] 未声明的相机不会按位置顺序误送给模型

证据路径：

```text

```

**Gate I：**

- [ ] PASS
- [ ] FAIL

---

## 12. Gate J：OpenPI relative 模型 dry-run

本阶段启动 server、真实相机和真实 observation，但不加 `--send-actions`。

### J1. 推理输出

- [ ] server metadata handshake 通过
- [ ] state shape 为 32
- [ ] action shape 为 `[H,16]`，H=50
- [ ] action 全部有限
- [ ] relative quaternion 有效且非零范数
- [ ] camera slots 和 model input keys 一致
- [ ] policy latency 无持续积压
- [ ] action queue policy 符合配置
- [ ] availability mask 正确

运行命令和日志路径：

```text

```

### J2. decoded target 审查

不下发动作，离线审查至少 20 个 chunk：

- [ ] anchor 来自当前 control cut measured TCP
- [ ] chunk 内 anchor 不变
- [ ] decoded runtime absolute target 在安全工作区
- [ ] 没有突然跳回 raw 原点/desk 原点
- [ ] 左右臂 target 没有交换

**Gate J：**

- [ ] PASS，可以进入小范围闭环
- [ ] BLOCKED：Gate E 未完成
- [ ] FAIL

---

## 13. Gate K：OpenPI relative 模型小范围真机闭环（L5）

### K1. 首轮限制

- [ ] 使用已确认安全的 qpos-init frame
- [ ] 速度、step、workspace 使用保守值
- [ ] 只运行少量 chunk/短时长
- [ ] 操作员和安全观察员现场
- [ ] 急停可达

### K2. 运行检查

- [ ] 首个 chunk anchor 正确
- [ ] measured TCP 朝合理方向移动
- [ ] 无突跳、回拉、振荡
- [ ] pause/hold 生效
- [ ] recovery 不消耗旧 chunk
- [ ] resume 只执行 fresh observation/fresh chunk
- [ ] 正常退出后保持安全 command
- [ ] 无 backend/IK/collision/server fault

### K3. 稳定性

至少完成：

- [ ] 3 次 cold start
- [ ] 5 次短闭环
- [ ] 1 次相机短暂缺帧 fail-closed 验证
- [ ] 1 次 policy pause/resume
- [ ] 1 次 server inference timeout/断开验证

日志路径：

```text

```

**Gate K：**

- [ ] PASS
- [ ] BLOCKED：OpenPI 未适配或前置 gate 未通过
- [ ] FAIL

---

## 14. Gate L：故障注入和 fail-closed

仅在不会造成危险运动的阶段执行对应故障；涉及断开机械臂通信、enable/disable 或急停的项目
必须由现场操作员明确安排。

| 故障 | 预期 | 实际证据 | 结果 |
| --- | --- | --- | --- |
| actual TCP 缺失 | 不解 relative、不下发 |  |  |
| measured qpos 无效/过期 | fail-closed |  |  |
| source_timestamp>T0 | 当前 cut 不选用 |  |  |
| available_timestamp>T0 | 当前 cut 不选用 |  |  |
| snapshot sequence 回退 | invalid/fault |  |  |
| quaternion 非有限/零范数 | 拒绝 action |  |  |
| Vive clock-sync 失效 | 明确 invalid/fallback 标记 |  |  |
| required camera 丢失 | 不推理/不下发 |  |  |
| tactile/hand 断开 | mask 或 fault，按 artifact |  |  |
| OpenPI timeout | queue 不执行陈旧 chunk |  |  |
| policy shape 错误 | 拒绝整个 chunk |  |  |
| IK timeout | hold last safe command |  |  |
| collision hold | group hold |  |  |
| backend reject | group hold + receipt |  |  |
| 单侧 RM65 断开 | 双臂停止继续下发 |  |  |
| queue clear/resume | 只接受 fresh chunk |  |  |
| debug event queue 溢出 | 验收失败，不静默通过 |  |  |

对每个故障统一确认：

- [ ] 不回退 commanded/last target 作为 measured state
- [ ] 不继续消耗 replay/model chunk
- [ ] 空闲时不重复发送 pose
- [ ] fault 原因进入 debug
- [ ] 恢复后只使用 fresh observation 和 fresh policy chunk

**Gate L：**

- [ ] PASS
- [ ] 部分完成，未完成项及原因如下
- [ ] FAIL

```text

```

---

## 15. 性能和长期稳定性

至少运行一段包含全部必需传感器的持续采集，以及一段真实部署：

| 指标 | 目标/要求 | 实际 | 结果 |
| --- | --- | --- | --- |
| collection nominal Hz | 与配置一致、无持续下降 |  |  |
| policy Hz | 与配置一致 |  |  |
| arm state age | 小于契约 freshness |  |  |
| camera age | 小于配置 max age |  |  |
| tactile age | 有效且无虚假刷新 |  |  |
| snapshot history drop | 0 或有界且不影响 T0 |  |  |
| debug event drop | 0 |  |  |
| policy queue backlog | 无持续积压 |  |  |
| IK timeout/hold | 正常轨迹为 0 或可解释 |  |  |
| CPU/内存 | 无持续增长 |  |  |
| server latency | 无持续恶化 |  |  |

持续时间：

```text
collection:
deployment:
```

---

## 16. 最终交付判定

### 必须通过

- [ ] Gate A：站点、desk 和 measured TCP
- [ ] Gate B：统一时钟、T0、不可变快照和首帧
- [ ] Gate C：native v2.3 数据采集及 `validate-v23`
- [ ] Gate D：chunk、round-trip 和虚拟部署
- [ ] Gate F：impedance absolute/relative 真机 A/B
- [ ] Gate G：impedance queue/recovery/backend 行为
- [ ] Gate I：本次正式配置中启用的手部、触觉和相机
- [ ] Gate L：关键 fail-closed 故障

### 按交付范围必须通过

- [ ] Gate E/J/K：若交付 OpenPI relative 模型部署
- [ ] Gate H：若交付 position 真机部署

### 最终证据索引

| 证据 | 路径 |
| --- | --- |
| station/desk/TCP 报告 |  |
| clock/T0/snapshot 报告 |  |
| native v2.3 数据集 |  |
| validate-v23 JSON |  |
| chunk round-trip 报告 |  |
| virtual deploy report |  |
| impedance absolute log |  |
| impedance relative log |  |
| impedance A/B acceptance JSON |  |
| position A/B acceptance JSON |  |
| OpenPI artifact preflight JSON |  |
| OpenPI websocket metadata |  |
| OpenPI dry-run log |  |
| OpenPI real closed-loop log |  |
| fault injection report |  |
| performance/stability report |  |

最终签字：

| 角色 | 姓名 | 日期 | 结论 |
| --- | --- | --- | --- |
| 开发/验收 |  |  |  |
| 真机操作员 |  |  |  |
| 安全观察员 |  |  |  |

---

## 17. 相关入口

- 数据集契约：`DATASET_SCHEMA.md`
- 部署契约：`DEPLOY_SCHEMA.md`
- 采集说明：`teleop_runtime/collection/README.md`
- 部署说明：`teleop_runtime/deploy/README.md`
- OpenPI 部署说明：`deployments/openpi/README.md`
- v2.3 数据验证：`python -m utils.dataset_postprocess validate-v23`
- 真机 A/B 日志验收：`python -m teleop_runtime.deploy.acceptance`
- OpenPI artifact 预检：
  `python -m deployments.openpi.client.real --policy-artifact-json ... --contract-check-only`
