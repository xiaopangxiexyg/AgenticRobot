# Chunk-Aware Residual HITL 研究备忘（2026-08-02）

> 历史状态：本文保留当前 HITL migration plan 形成前的研究动机、候选状态机和
> 评估设想，不再作为实施事实源。当前边界、成熟度和下一阶段以
> [`hitl_chunk_relative_migration_plan.md`](../../../todos/hitl_chunk_relative_migration_plan.md)
> 为准；发生冲突时以当前代码、合同和该计划为准。

当前主线应明确调整为：

> **不要求 policy 理解 human。policy 只提供可视、可冻结、可编辑的短期 nominal chunk；human 通过 residual 修改该 chunk；系统负责组合、冲突处理、连续性、重基准和回退。**

这是一种 **human-readable policy + chunk-aware residual control**，而不是 guided policy。

近期 SAPS 也说明，不修改模型、直接在动作层进行 shared autonomy 是可行的模型无关路线；但它主要验证了 blending 和 cosine arbitration，本身并没有完整解决 chunk 版本切换、persistent residual、handback 和弱 policy 反抗等系统问题。([arXiv][1])

---

# 一、确定系统目标

```text
Weak policy
    → 输出短期 nominal chunk
    → 将 policy 计划展示给 human

Human
    → 对 nominal chunk 施加局部 residual
    → 系统产生连续、可执行的 corrected chunk
```

核心角色应定义为：

| 角色                     | 定位                                         |
| ---------------------- | ------------------------------------------ |
| Policy                 | 短期 nominal motion prior，不是绝对正确的 controller |
| Human                  | residual editor、冲突裁决者和 takeover owner      |
| Shared-control runtime | 管理 chunk 生命周期、residual、rebase 和 mode       |
| `control_composition`  | 在确定的时间切面上执行纯数值组合                           |
| `robot_motion`         | authority、安全、reference 和执行                 |

这里最重要的原则是：

> **Policy 是 base，但 human 不是低权重 overlay。human 对其主动修改的方向和 component 拥有优先级。**

---

# 二、最重要的二次修正：base 不能是每帧变化的 next action

如果仍然采用：

[
u_t=\pi(s_t)\oplus r_t^h
]

并在每个控制周期重新使用最新 policy 输出作为 base，那么会出现：

1. human 修改动作；
2. 状态被修改；
3. policy 根据新状态输出另一个动作；
4. base 本身移动；
5. human residual 的参考系不断漂移；
6. human 需要持续与 policy 对抗。

所以合理系统不能只是：

```text
每帧最新 policy action + human residual
```

而应是：

> **Human intervention 激活时，锁定一个带版本号的 nominal policy chunk；human 在一个短期窗口内编辑这个稳定 chunk。新的 policy chunk 暂时只作为 shadow candidate，不能直接替换当前 base。**

这会成为整个系统的基础。

---

# 三、核心控制对象：Versioned Nominal Chunk

设 policy 在时刻 (t_0) 输出：

[
A^\pi_g=
\left{
a^\pi_{g,0},
a^\pi_{g,1},
\ldots,
a^\pi_{g,H-1}
\right}
]

其中 (g) 是 policy generation。

不要只把第一个 next action 交给 composer，而应形成：

```text
PolicyNominalChunkFact
- generation
- request_id
- observation_cut
- component trajectories
- target time lattice
- valid support boundary
- committed prefix
- editable suffix
- produced_at / available_at
```

当 human 开始介入时：

```text
active_base_generation := g
```

随后，即使 policy 根据新 observation 又产生 (g+1,g+2)，在当前 residual session 内也不能无条件替换 (g)。

新的 chunk 只能：

```text
new policy chunk
    → shadow buffer
    → compatibility check
    → rebase boundary
    → 成为新的 active base
```

这一步能显著提高：

* 操作者可预测性；
* residual 参考稳定性；
* chunk 内运动连续性；
* 日志和 lineage 的可解释性；
* handback 的可控性。

---

# 四、Human residual 应修改“轨迹”，而不是混合两个完整动作

## 4.1 基本形式

对 Cartesian pose，human 激活时记录设备 anchor：

[
H_0=H(t_0)
]

计算 human correction：

[
\xi_h(t)
========

S\operatorname{Log}
\left(
H_0^{-1}H(t)
\right)
]

其中：

* (S)：设备到机器人尺度与坐标映射；
* (\xi_h)：6D residual；
* (M)：component/DoF mask。

对 nominal chunk 中的每个 future pose：

[
T^{c}_{g,k}
===========

\operatorname{Exp}
\left(
w_kM\xi_h
\right)
T^\pi_{g,k}
]

其中 (w_k) 是时间方向的平滑权重。

要求：

[
w_0=0
]

或者在当前 authoritative reference 处严格连续；随后在未来若干步逐渐增加：

[
0=w_0<w_1<\cdots<w_r=1
]

这样 human 不是突然把当前 target 移走，而是在未来短期轨迹中逐步加入 correction。

## 4.2 三种 residual temporal support

### A. 整体平移/旋转 chunk

[
w_k\rightarrow 1
]

之后保持为 1。

适合：

* “整条路径向左一点”；
* “抓取位置整体高一点”；
* 保留 policy 原始轨迹形状。

### B. 局部 trajectory warp

权重先增加，再逐渐回到零：

[
w_k:
0\rightarrow1\rightarrow0
]

适合：

* 临时绕开障碍；
* 局部抬高；
* 接触前纠正，但不希望永久改变后续目标。

### C. Rate steering

human 输入表示 residual 的变化率：

[
\xi_{t+1}
=========

\operatorname{clip}
\left(
\xi_t+K\Delta h_t
\right)
]

适合 joystick 式连续调整。

对当前 VIVE/pose 遥操作，建议主模式为：

```text
clutch-relative chunk offset
```

补充模式为：

```text
momentary rate steering
```

---

# 五、Residual 系统不能依靠“50%–50%”

系统应从 symmetric blending 改成 asymmetric composition：

```text
Policy 提供 nominal
Human 提供 correction
Human 在冲突方向优先
```

基本输出可以写成：

[
v_{\mathrm{cmd}}
================

v_\pi^{\mathrm{allowed}}+v_h
]

而不是：

[
v_{\mathrm{cmd}}
================

(1-\alpha)v_\pi+\alpha v_h
]

## 5.1 同向

human 与 policy 同向时：

* 保留 policy；
* human 可以加速、放大或微调；
* 不需要减少 policy 主体能力。

## 5.2 侧向

human 对 policy 做横向修正时：

* 保留 policy 的前进分量；
* 加入 human 横向 residual；
* 形成“policy 推进、human steering”。

## 5.3 反向

human 明确向 policy 相反方向输入时，不能继续平均。

令 human residual 的单位方向为 (\hat r)，policy 在该方向上的投影为：

[
p=v_\pi^\top\hat r
]

若：

[
p<0
]

说明 policy 正在抵抗 human。应先移除 policy 的反向分量：

[
v_\pi'
======

## v_\pi

p\hat r
]

然后：

[
v_{\mathrm{cmd}}
================

v_\pi'+v_h
]

即：

> human 向左纠正时，policy 可以继续保留上下、姿态、夹爪等非冲突能力，但不能继续在同一维度向右拉。

如果冲突幅度大、持续时间长或涉及不同任务模式，则直接：

```text
HUMAN_TAKEOVER
```

Cosine similarity 可以作为冲突诊断和 baseline，但不能单独决定控制权，因为接近零速度、平移与旋转尺度不同、双臂 component 不同都会使单一 cosine 不稳定。SAPS 使用了 cosine-based arbitration，可作为对照实现，而非最终完整语义。([arXiv][1])

---

# 六、建议的完整运行状态机

```text
POLICY_ACTIVE
      │ human clutch / correction
      ▼
RESIDUAL_ACTIVE
      │ correction exceeds envelope
      ├──────────────────────────► HUMAN_TAKEOVER
      │
      │ human release / chunk boundary
      ▼
REBASE_PENDING
      │ compatible new policy chunk
      ▼
HAND_BACK
      │ continuity satisfied
      ▼
POLICY_ACTIVE
```

此外，任一状态都可以进入：

```text
EXPLICIT_HOLD
FAULT
END
```

---

## 6.1 `POLICY_ACTIVE`

* active base 随正常 policy chunk 更新；
* human residual 为 identity；
* UI 显示 nominal policy chunk；
* 执行 policy-only。

必须满足：

[
C(a^\pi,0)=a^\pi
]

即 policy-only 路径应尽量做到数值等价，不经过不必要的平滑或隐式 correction。

---

## 6.2 `RESIDUAL_ACTIVE`

进入时：

1. latch 当前 active policy generation；
2. capture human anchor；
3. capture current authoritative reference；
4. 建立 residual session；
5. 后续 policy chunk 只进入 shadow buffer。

状态内：

* human 编辑 locked nominal chunk；
* policy 不得每帧改变 human 的 base；
* residual 有明确 envelope；
* UI 同时显示 nominal chunk 和 corrected chunk。

这里并不是完全停止 policy inference。policy 可以继续运行，但它的结果只是：

```text
future rebase candidate
```

不是立即执行的 base。

---

## 6.3 `REBASE_PENDING`

human 释放、active chunk 接近结束，或者需要切换到新 policy chunk 时，不能直接清零 residual。

首先对新 chunk 做 compatibility check。

至少检查：

### Temporal validity

* observation 是否过旧；
* chunk 是否仍覆盖目标时间；
* lattice/epoch 是否兼容。

### Seam continuity

新 chunk 第一个有效 reference 与当前 composed reference 的距离：

[
D_{\mathrm{seam}}
=================

d(T_{\mathrm{current}},T^\pi_{\mathrm{new},0})
]

### Directional conflict

新 chunk 是否立即撤销 human correction。

### Chunk stability

连续 policy generations 是否剧烈切换方向或模式。

### Component completeness

新 chunk 是否包含当前需要的 arm、gripper 和 hand component。

---

## 6.4 `HAND_BACK`

若新 policy chunk 可接受，需要做无跳变 rebase。

设当前实际 composed pose 为：

[
T_c
]

新 nominal chunk 起点为：

[
T^\pi_{\mathrm{new},0}
]

计算新的 rebase transform：

[
\Delta_{\mathrm{rebase}}
========================

T_c
\left(
T^\pi_{\mathrm{new},0}
\right)^{-1}
]

随后：

[
T^{c}_{\mathrm{new},k}
======================

\operatorname{Exp}
\left(
\beta_k
\operatorname{Log}
\Delta_{\mathrm{rebase}}
\right)
T^\pi_{\mathrm{new},k}
]

其中：

[
\beta_0=1,\qquad
\beta_k\rightarrow0
]

这样新 policy chunk 从当前真实执行状态连续接管，residual 逐渐消退。

注意：

> 对弱 policy，handback 不应默认假设新 chunk正确。

如果新 chunk 与 human 修正强冲突，应：

* 保持 takeover；
* 显示新 policy ghost trajectory；
* 等待 operator 显式接受；
* 或在更合理的边界再次尝试。

---

## 6.5 `HUMAN_TAKEOVER`

触发条件包括：

* residual 超过平移或旋转 envelope；
* human 与 policy 持续强冲突；
* policy chunk 失效、过期或不完整；
* policy 反复撤销 correction；
* operator 显式接管。

Takeover 可以是 component-wise：

```text
left arm: policy
right arm: human
left gripper: policy
right gripper: human
```

不必一次性切走所有 component。

---

# 七、Chunk 到期时的处理必须明确

这是完整系统中非常关键的边界。

在 `RESIDUAL_ACTIVE` 中，locked base chunk 最终会到达 support boundary。此时不能：

* 无限外推旧 chunk；
* 突然采用最新 policy chunk；
* 静默 HOLD；
* 将最后一个 pose 永久保持而不记录原因。

建议优先级：

```text
1. 存在 compatible shadow chunk
   → rebase 并继续 residual

2. 没有 compatible chunk，但 human 正在主动控制
   → component-wise takeover

3. human 不再主动，且没有可信 chunk
   → explicit HOLD / no_action，由下游 lifecycle 处理

4. policy source fault/stale
   → policy ineligible，不再作为 base
```

PACE 的实验表明，action chunk 应执行多长时间与任务阶段相关，固定 execution horizon 并不可靠。因此，chunk boundary 和 replan boundary 应成为显式运行时对象，而不是固定消费 (N) 步。([arXiv][2])

---

# 八、把策略信息交给 human：Policy Legibility Layer

这一点应成为系统主模块，而不是 GUI 附属功能。

弱 policy 不需要解释自己“为什么这样做”，但必须尽量向 human 暴露：

> 它接下来实际准备怎么运动。

## 8.1 至少展示三条轨迹

### Nominal policy trajectory

policy 原始 chunk：

```text
细线 / 半透明 ghost
```

### Corrected trajectory

加入当前 human residual 后：

```text
实线 / 高亮
```

### Executable committed trajectory

经过当前 composition、admission 和 reference scheduling 后，真正即将执行的近期路径：

```text
最明显、最短的 committed segment
```

三者不能混淆。

---

## 8.2 显示 chunk 的时间结构

建议将 chunk 分成：

```text
[ Executing ][ Committed ][ Editable ][ Unreliable Tail ]
```

### Executing

已经进入 target/reference pipeline，通常不能由普通 residual 撤回。

### Committed

短期内将执行，只有强 intervention 或 takeover 才可改变。

### Editable

human residual 主要修改的部分。

### Unreliable tail

较远期预测，只供参考，透明度更低。

近期 asynchronous/chunk execution 研究指出，实际执行前缀与当前感知、后续 chunk 之间的不一致会造成失败；这说明 UI 和 runtime 都应区分实际执行 prefix、当前 committed prefix 和预测 tail，而不是把整个 action chunk 当作同等可信。([arXiv][3])

---

## 8.3 不要求 policy 输出语义解释

对于弱 policy，不要依赖它可靠输出：

* 目标物体；
* 当前任务阶段；
* 失败原因；
* 语义 intent。

第一版只展示可直接从 chunk 得到的事实：

* 左右 EEF 轨迹；
* 姿态变化；
* gripper open/close 时刻；
* 运动速度；
* 当前 active base generation；
* chunk 剩余时间；
* policy plan 是否刚发生明显跳变。

语义标签可以作为附加信息，但不能作为 shared-control 正确性的前提。

---

## 8.4 Human 的操作语义应对应“编辑可见轨迹”

理想交互不是：

```text
human 猜 policy 会去哪
然后盲目给一个 pose
```

而是：

```text
human 看到 policy ghost chunk
移动控制器
ghost chunk 实时变形
corrected chunk 成为实际候选
```

SHARC 等 shared-autonomy 系统采用虚拟 robot/trajectory preview，让操作者在执行前看到和调整机械臂轨迹，说明 preview 本身可以作为有效的交互层，而不要求底层 policy 理解 human。([IEEE Xplore][4])

---

# 九、不同 component 不能使用同一种 residual 规则

## 9.1 Cartesian arm pose

采用：

* SE(3) residual；
* 明确 world/base/tool frame；
* translation 和 rotation 分开尺度；
* temporal warp；
* conflict clipping。

## 9.2 Gripper

不建议连续比例 blending。

应使用：

```text
policy owner
human override
latched human command
```

例如：

* policy 正在 close；
* human 明确 open；
* 不应得到“半开”。

需要：

* hysteresis；
* command latch；
* takeover/release；
* 明确 discrete lineage。

## 9.3 Hand joint position

可以使用 mask-aware residual，但需要：

* joint limit；
* human mask；
* policy-human conflict；
* synergies/component grouping；
* 不要逐元素随意平均。

## 9.4 双臂

应允许：

* arm-level mode；
* translation/rotation 独立 mode；
* left/right 独立 base generation；
* 同一 chunk 中保持必要的 coordination identity。

如果双臂 policy 强耦合，而 human 只改一臂，必须记录：

```text
cross-arm coordination broken = true
```

必要时降低另一臂 policy 推进、HOLD 另一臂，或进入双臂 takeover，而不能假设单臂 residual 总是无害。

---

# 十、建议新增的系统模块

不建议把全部状态逻辑塞进 `control_composition`。可以增加：

```text
shared_control/
├── contracts.py
├── mode_reducer.py
├── base_chunk_latch.py
├── residual_integrator.py
├── chunk_warp.py
├── conflict_arbitration.py
├── rebase_controller.py
├── policy_chunk_gate.py
├── preview_projection.py
└── observability.py
```

## `base_chunk_latch`

负责：

* active base generation；
* shadow policy chunks；
* support boundary；
* base replacement条件。

## `residual_integrator`

负责：

* human anchor；
* clutch；
* residual state；
* temporal support；
* magnitude envelope。

## `conflict_arbitration`

负责：

* human-policy directional conflict；
* policy opposed-component suppression；
* takeover recommendation。

## `rebase_controller`

负责：

* chunk seam；
* residual transfer；
* handback decay；
* continuity。

## `preview_projection`

负责生成：

* nominal preview；
* corrected preview；
* committed/executable preview。

## `control_composition`

继续只负责：

```text
selected policy base fact
+
selected human residual fact
+
selected shared-control mode fact
→ composed candidate
```

它不维护 chunk latch、human anchor 或 rebase state。

---

# 十一、建议的正式数据契约

## `PolicyNominalChunkFact`

```text
policy_id
artifact_id
generation
request_id
observation_cut
component_schema
time_lattice
actions / poses
committed_prefix_length
support_boundary
produced_at
available_at
```

## `HumanResidualFact`

```text
session_id
anchor_id
residual_mode
residual_se3 / qpos_delta
reference_frame
component_mask
temporal_support
magnitude
active
hold_requested
takeover_requested
effective_interval
```

## `SharedControlModeFact`

```text
mode
active_base_generation
residual_session_id
policy_component_mask
human_component_mask
conflict_state
rebase_state
handback_progress
revision
```

## `PolicyChunkAdmissionDecision`

```text
candidate_generation
accepted / shadow / rejected
seam_difference
directional_conflict
staleness
support_coverage
reason_codes
```

## `CorrectedChunkPreview`

```text
nominal_chunk_identity
residual_identity
composed_chunk
committed_prefix
editable_suffix
support_boundary
```

---

# 十二、系统必须满足的硬性不变量

## I1：Policy identity

```text
human inactive
→ final candidate == policy candidate
```

## I2：Stable intervention base

```text
RESIDUAL_ACTIVE
→ policy 新 generation 不得无条件改变 active base
```

## I3：No opposing policy force

```text
human 对某 DoF 明确纠正
→ policy 不得继续在同一 DoF 施加相反推进
```

## I4：Continuity

在以下边界都不能出现未解释跳变：

* clutch press；
* clutch release；
* policy chunk switch；
* residual mode switch；
* takeover；
* handback。

## I5：Bounded residual

residual 超过：

* 空间范围；
* 旋转范围；
* 持续时间；
* chunk support；

必须升级为 takeover 或 HOLD。

## I6：No double correction

human residual 只能在一个位置作用一次：

```text
human residual
→ chunk warp
```

不能同时又进入 action adaptation 或 reference generator重复叠加。

## I7：Preview fidelity

UI 高亮的 committed trajectory 必须来自实际 authoritative pipeline 的最新可用事实，而不是任意 raw policy prediction。

## I8：Explicit expiration

policy chunk 或 human residual 到期后必须：

* rebase；
* takeover；
* HOLD；
* no_action；

不能使用隐式 last-value forever。

---

# 十三、分层落地规划

## 第一层：Residual 最小正确闭环

范围只做：

* 单臂；
* Cartesian translation；
* free-space；
* explicit clutch；
* policy chunk latch；
* human relative offset；
* residual envelope；
* human inactive = policy identity；
* takeover fallback。

暂不做：

* rotation；
* gripper blending；
* 双臂；
* 自动 handback；
* 模型训练。

验收重点：

* policy 更新不会让 residual reference 漂移；
* human 不再需要追逐变化的 policy base；
* correction 后轨迹连续；
* 强冲突能切 takeover。

---

## 第二层：Chunk-aware residual

增加：

* 整个剩余 chunk 的 warp；
* temporal ramp；
* committed/editable 区分；
* shadow policy buffer；
* chunk support boundary；
* candidate chunk compatibility gate；
* rebase continuity。

这是系统真正从“逐帧动作相加”升级为“轨迹编辑”的关键层。

---

## 第三层：完整 component semantics

增加：

* full SE(3)；
* world/tool frame；
* gripper discrete override；
* hand qpos mask；
* per-arm mode；
* cross-arm coordination diagnostics；
* contact-phase residual envelope。

---

## 第四层：Policy-to-human interface

增加：

* nominal ghost chunk；
* corrected chunk；
* committed executable segment；
* gripper event；
* policy generation change；
* residual saturation；
* current ownership；
* pending handback。

这一层应与第二层同步推进，因为没有 preview 的 chunk residual，操作者仍然难以建立正确 mental model。

---

## 第五层：Robust handback

增加：

* policy shadow evaluation；
* compatibility gate；
* residual transfer；
* explicit/automatic handback；
* weak-policy rejection；
* stale/underrun/fault behavior。

这里不要求 policy 能够恢复，只要求系统判断：

```text
它目前是否适合重新成为 base
```

---

## 第六层：数据闭环，但暂不要求 guided policy

完整记录：

```text
nominal policy chunk
human residual
corrected chunk
actual execution
policy replans
takeover / handback
success / failure
```

这些数据后续可以用于：

* 分析 weak policy 的 failure pockets；
* 训练更好的 base policy；
* 学 residual suggestion；
* 学 intervention trigger；
* recovery data mining。

但当前系统运行不依赖任何这些模型。

---

# 十四、推荐的验证指标

## 控制正确性

### Residual continuity

* clutch press pose jump；
* clutch release pose jump；
* chunk switch pose/velocity jump；
* maximum jerk。

### Base stability

* intervention 期间 active base generation 切换次数；
* shadow chunk 被拒绝比例；
* rebase seam error。

### Human-policy fighting

[
J_{\mathrm{fight}}
==================

\sum_t
\max
\left(
0,-v_h^\top v_\pi
\right)
]

并分别统计：

* 检测到的 conflict；
* 被抑制的 policy opposed component；
* 未解决冲突。

### Residual burden

* correction magnitude；
* intervention duration；
* residual saturation count；
* takeover count。

## 可预测性

* nominal preview 与 raw policy chunk 一致性；
* committed preview 与实际 reference 一致性；
* preview update frequency；
* policy mode/chunk generation 跳变次数。

## HITL 价值

对比：

```text
Policy only
Human only
Absolute 50/50 blend
Per-step residual
Latched-chunk residual
Chunk residual + conflict clipping
完整 residual + rebase
```

指标：

* task success；
* human active time；
* completion time；
* correction 后继续成功率；
* operator 重复反向修正次数；
* takeover 后重新交还成功率。

---

# 最终系统定位

当前最合理的主线不是：

```text
Human teaches policy how to act online
```

而是：

```text
Policy exposes a short-horizon proposal
Human edits the proposal
System guarantees coherent execution
```

具体而言：

```text
Weak policy chunk
    ↓
Versioned nominal trajectory
    ↓       ← Human residual
Chunk warp + conflict suppression
    ↓
Corrected trajectory preview
    ↓
Composition candidate
    ↓
Authoritative execution
```

最重要的三个设计决策是：

1. **Policy base 必须是被锁定、带 generation 的短期 chunk，而不是每帧变化的 next action。**
2. **Human residual 应编辑未来轨迹，并在冲突方向压制 policy，而不是与 policy 对称平均。**
3. **把 nominal chunk、corrected chunk 和 executable committed prefix 展示给 human，由 human承担高层判断，避免依赖弱 policy 理解 human。**

这样既符合 HITL 服务于弱模型的出发点，也能在不修改 policy、不过度依赖模型能力的前提下，形成一个完整、可测试、可追踪、可回退的 residual shared-control 系统。

[1]: https://arxiv.org/abs/2606.15568?utm_source=chatgpt.com "SAPS: Shared Autonomy for Policy Steering by Blending Teleoperation with a Pretrained VLA"
[2]: https://arxiv.org/abs/2606.00537?utm_source=chatgpt.com "PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking"
[3]: https://arxiv.org/abs/2601.20130?utm_source=chatgpt.com "Real-Time Robot Execution with Masked Action Chunking"
[4]: https://ieeexplore.ieee.org/iel8/8860/4359257/10606089.pdf?utm_source=chatgpt.com "A Shared Autonomy System for Precise and Efficient ..."
