# 长程具身任务真机数据采集与标注标准 v0.1

## 一、目标与适用范围

本标准面向多阶段、长程、真实机器人操作任务，目标是建立一套：

* 能够实际执行；
* 不依赖难以稳定获取的真机世界状态；
* 不要求数采员承担过重实时标注；
* 可以支持后续 VLA、VLM–VLA、阶段识别、失败识别和鲁棒性研究；
* 能够从完整任务、局部阶段和人为异常场景中统一生成训练数据；

的数据采集与标注体系。

当前范围包括：

```text
完整长程任务采集
单个任务节点采集
单个任务阶段采集
任务节点与阶段转换采集
人为设计的偏离或失败状态采集
高频机器人与控制机制数据采集
VLM–VLA 局部调用数据的兼容接口
```

当前不纳入：

```text
Human-in-the-Loop 接管
在线 recovery
authority 仲裁
takeover 原因与交还逻辑
复杂世界状态谓词
自动对象关系判定
人工连续 progress 标注
人工 Execution Status 标注
```

这些内容可在后续系统接入时独立扩展，不影响当前核心数据结构。

---

# 二、总体结构

数据分为两类行为语义和一个支持层：

```text
任务语义 Task Semantics
机制语义 Mechanism Semantics
证据与版本信息 Provenance
```

## 1. 任务语义

任务语义描述：

> 当前数据对于这个特定长程任务意味着什么。

包括：

* 当前属于哪个任务节点；
* 当前位于任务节点内部哪个阶段；
* 当前是该节点的第几次实际尝试；
* 本条数据为什么从这个位置开始；
* 是否人为设计了偏离或失败条件；
* 当前局部执行要求是什么；
* 本次局部执行是否产生预期结果。

任务语义通常依赖 Task Profile 和人的任务理解，主要通过任务定义、数采场景配置和后期区间标注产生。

## 2. 机制语义

机制语义描述：

> 机器人、控制系统和传感器客观发生了什么。

包括：

* 时间戳；
* 关节和末端状态；
* 夹爪状态；
* 动作命令；
* 接触和碰撞检测；
* 安全状态；
* 工作空间位置；
* 延迟、掉帧和传感器有效性；
* 人为扰动实际施加的时间和参数。

机制语义原则上不依赖具体任务，可以在线自动采集或离线稳定派生。

## 3. 证据与版本信息

证据层保存：

* 信息来源；
* 标注者；
* 标注置信度；
* schema 版本；
* Task Profile 版本；
* 机器人、控制器、标定和策略版本；
* 数据缺失和时钟同步质量。

它不属于新的行为语义，而是保证数据可追溯和可修订的基础。

---

# 三、唯一术语

为避免一个概念使用多个名称，统一采用以下术语。

| 术语                    | 唯一定义                                 |
| --------------------- | ------------------------------------ |
| `Task Profile`        | 一类长程任务的静态语义定义                        |
| `Task Node`           | 长程任务中一个具有局部目标的任务节点                   |
| `Attempt`             | 在某个 Episode 中对一个 Task Node 的一次实际执行实例 |
| `Phase`               | Attempt 内部的离散进度阶段                    |
| `Collection Scenario` | 描述一类数据如何被布置、触发和采集的配置                 |
| `Episode`             | 一次连续的数据记录，不要求覆盖完整任务                  |
| `Phase Interval`      | 某个 Phase 在 Episode 中的时间区间            |
| `Invocation`          | 上层或操作者向执行器发出的一次有界局部执行要求              |
| `Scenario Event`      | 数采场景中的布置、扰动和目标状态事件                   |
| `Mechanism Stream`    | 连续或高频记录的机器人和系统信号                     |
| `Mechanism Event`     | 接触、碰撞、故障、状态切换等稀疏机制事件                 |

不使用以下歧义字段作为 canonical ground truth：

```text
subtask
stage
atomic_skill
memory
execution_status
continuous_progress
recovery=true
failure=true
```

对应关系为：

```text
subtask          → Task Node
stage            → Task Node 或 Phase
atomic skill     → Invocation Type
memory           → 从真实执行历史派生
progress         → 从 Task Node、Attempt 和 Phase 派生
failure          → Collection Scenario 目标与实际结果
```

---

# 四、数据实体关系

核心关系为：

```text
Task Profile
└── Task Node
    └── Phase Definition

Collection Scenario
└── 规定从哪个 Task Node / Phase 开始
    规定是否人为施加异常条件
    规定采集范围和停止条件

Episode
├── Attempt
│   └── Phase Interval
├── Invocation
├── Scenario Event
├── Mechanism Stream
└── Mechanism Event
```

其中：

```text
Task Node
= 当前要完成什么局部目标

Attempt
= 这次实际执行该局部目标

Phase
= 该次执行内部推进到哪个阶段

Collection Scenario
= 为什么从这里开始，以及是否人为设计了异常

Episode
= 一次实际连续记录
```

---

# 五、Task Profile

Task Profile 是任务标注、局部采集和后续 VLM–VLA 数据导出的共同语义合同。

```yaml
task_profile:
  task_profile_id: store_cup_in_drawer
  version: 1.0

  task_description: 将目标杯子放入抽屉并关闭抽屉

  nodes:
    - task_node_id: pick_cup
      node_description: 拿起目标杯子

      allowed_next_nodes:
        - open_drawer
        - pick_cup

      phases:
        - approach
        - align
        - close_gripper
        - lift
        - stabilize

      entry_guideline: 开始有目的地接近目标杯子
      completion_guideline: 杯子被稳定提离支撑面
      annotation_notes: 抬升后短暂晃动仍属于 stabilize

    - task_node_id: open_drawer
      node_description: 打开目标抽屉

      allowed_next_nodes:
        - place_cup
        - open_drawer

      phases:
        - approach_handle
        - align_handle
        - engage_handle
        - pull
        - release_handle
```

Task Profile 只定义：

* Task Node；
* Phase；
* 合法的节点转移；
* 人工可理解的进入和完成说明；
* 数据标注边界说明。

当前不依赖：

```text
inside(cup, drawer)
grasped(cup)
open(drawer)
```

等难以在真机中稳定获得的自动谓词。

Task Profile 使用 DAG 表达任务结构。严格线性任务只是每个节点只有一个正常后继的特殊情况。

---

# 六、Collection Scenario

Collection Scenario 描述一类数据应如何产生，是当前系统表达偏离和失败数据的主要方式。

```yaml
collection_scenario:
  scenario_profile_id: pick_cup_misalignment_v1
  scenario_version: 1.0
  task_profile_id: store_cup_in_drawer

  collection_purpose: OFF_NOMINAL_STATE

  planned_entry:
    task_node_id: pick_cup
    phase_id: align
    entry_position: MID_PHASE
    entry_condition: OFF_NOMINAL

  setup_recipe:
    description: 将夹爪初始位置设置在杯子中心右侧 4 cm
    parameters:
      lateral_offset_m: 0.04

  perturbation_recipe: null

  target_condition:
    category: MISALIGNMENT

  stop_rule:
    type: FIXED_DURATION
    duration_s: 5.0

  safety_constraints:
    maximum_tcp_speed:
    maximum_contact_force:
```

## 1. collection_purpose

```text
NOMINAL_DEMONSTRATION
PHASE_COVERAGE
PHASE_TRANSITION
NODE_TRANSITION
OFF_NOMINAL_STATE
FAILURE_STATE_GENERATION
ROBUSTNESS_TEST
```

## 2. entry_position

```text
TASK_START
NODE_START
MID_PHASE
```

## 3. entry_condition

```text
NOMINAL
OFF_NOMINAL
```

`entry_condition` 只描述数采场景的设计条件，不要求数采员再标注 DEVIATING、FAILED 等第二条状态轴。

## 4. target_condition

`target_condition` 表示场景希望产生的数据类别，例如：

```text
MISALIGNMENT
EMPTY_GRASP
OBJECT_SLIP
OBJECT_DROP
BLOCKED_MOTION
WRONG_APPROACH
UNEXPECTED_CONTACT
VISUAL_OCCLUSION
CONTROL_LIMIT
```

该类别是场景设计目标，不自动等于实际结果。

## 5. stop_rule

```text
STOP_AT_PHASE_END
STOP_ON_TARGET_CONDITION
STOP_AFTER_PERTURBATION_WINDOW
STOP_ON_SAFETY_TRIGGER
FIXED_DURATION
MANUAL_STOP
```

当前不采集 recovery 时，异常轨迹通常采用：

```text
正常或异常起始状态
→ 人为扰动或执行
→ 目标异常形成
→ 保留短暂后续
→ 停止记录
```

---

# 七、Episode

Episode 是一次连续记录，可以是完整任务，也可以是局部片段。

```yaml
episode:
  episode_id:
  task_profile_id:
  task_profile_version:

  scenario_profile_id:
  scenario_version:

  task_instruction:

  recording_scope:
  context_source:

  entry_context:
    entry_task_node_id:
    entry_phase_id:
    assumed_completed_node_ids:

  scenario_result:
  task_outcome:
  termination_cause:

  start_time:
  end_time:

  robot_config_id:
  controller_config_id:
  sensor_config_id:
  camera_config_id:
  calibration_version:
  policy_version:
  operator_id:
```

## 1. recording_scope

```text
FULL_TASK
NODE_WINDOW
PHASE_WINDOW
PHASE_TRANSITION
NODE_TRANSITION
EVENT_WINDOW
```

含义分别为：

* `FULL_TASK`：覆盖完整任务；
* `NODE_WINDOW`：覆盖一个 Task Node；
* `PHASE_WINDOW`：重点覆盖一个 Phase；
* `PHASE_TRANSITION`：覆盖 Phase A 到 Phase B；
* `NODE_TRANSITION`：覆盖 Task Node A 到 Task Node B；
* `EVENT_WINDOW`：围绕人为扰动或异常状态采集。

## 2. context_source

```text
ACTUAL_FULL_HISTORY
ACTUAL_TRUNCATED_HISTORY
SCENARIO_SETUP
NO_PREHISTORY
```

含义为：

* `ACTUAL_FULL_HISTORY`：Episode 开始前的任务历史完整保留；
* `ACTUAL_TRUNCATED_HISTORY`：真实执行过前置步骤，但当前文件只保留后半段；
* `SCENARIO_SETUP`：人工或自动将场景直接布置为中间状态；
* `NO_PREHISTORY`：该数据本身不依赖前置任务历史。

这能够区分：

```text
真实完成前两步后进入第三步
```

和：

```text
人工直接摆成第三步的起始状态
```

## 3. entry_context

允许 Episode 从任意 Task Node 或 Phase 开始：

```yaml
entry_context:
  entry_task_node_id: place_cup
  entry_phase_id: align_target

  assumed_completed_node_ids:
    - pick_cup
    - open_drawer
```

`assumed_completed_node_ids` 只描述场景 setup 所代表的任务上下文，不是 memory 标签。

## 4. scenario_result

```text
TARGET_ACHIEVED
TARGET_PARTIALLY_ACHIEVED
TARGET_NOT_ACHIEVED
UNEXPECTED_RESULT
INVALID_RUN
UNKNOWN
```

## 5. task_outcome

仅在 `FULL_TASK` 或确实可以判断整体结果时填写：

```text
SUCCESS
PARTIAL
FAILURE
ABORTED
UNKNOWN
NOT_APPLICABLE
```

局部 Phase 片段应使用 `NOT_APPLICABLE`，而不是被迫标记为任务成功或失败。

## 6. termination_cause

```text
TASK_COMPLETED
SCENARIO_COMPLETED
FIXED_WINDOW_END
TIME_LIMIT
MANUAL_STOP
SAFETY_STOP
ESTOP
CONTROLLER_FAULT
SENSOR_FAULT
RESET
UNKNOWN
```

---

# 八、Attempt

Attempt 是某个 Task Node 在当前 Episode 中的一次实际执行实例。

```yaml
attempt:
  attempt_id:
  episode_id:

  task_node_id:
  attempt_index:
  history_complete:

  outcome:
  outcome_reason:

  annotation_source:
  annotation_confidence:
```

## 1. attempt_index

表示当前 Episode 中对该 Task Node 的第几次尝试。

若 Episode 从中间开始，且之前历史未知：

```yaml
attempt_index: null
history_complete: false
```

不得默认写成第一次尝试。

## 2. outcome

```text
SUCCESS
PARTIAL
RETRYABLE_FAILURE
TERMINAL_FAILURE
ABORTED
UNKNOWN
NOT_APPLICABLE
```

对于只截取 Phase 中间部分的 Episode，可以使用 `NOT_APPLICABLE`。

Attempt 的时间范围由关联的 Phase Interval 自动得到，不重复维护起止时间。

---

# 九、Phase Interval

Phase 是当前版本唯一需要人工进行细粒度时序标注的任务子状态。

```yaml
phase_interval:
  phase_interval_id:
  episode_id:
  attempt_id:

  phase_id:

  start_time:
  end_time:

  truncated_at_start:
  truncated_at_end:

  boundary_confidence:
  annotation_source:
  annotation_version:
```

## 1. truncated_at_start

表示 Episode 开始时已经处于该 Phase 中间。

例如轨迹直接从错误对齐状态开始：

```yaml
phase_id: align
truncated_at_start: true
```

## 2. truncated_at_end

表示 Episode 在 Phase 自然结束前终止。

例如人为生成物体滑落后立即停止：

```yaml
phase_id: lift
truncated_at_end: true
```

## 3. Phase 只表达任务进度

即使数据是人为失败场景：

```text
Task Node = pick_cup
Phase = align
Scenario target = MISALIGNMENT
```

也不创建：

```text
phase = failed_align
```

因为 Phase 不负责表达执行质量或失败类型。

---

# 十、Invocation

Invocation 用于兼容 τ₀-VLA、Harness VLA 和 VLA-as-Tools 等 VLM–VLA 架构。

其定义是：

> 上层、操作者或任务脚本向执行器发出的一次具有明确边界的局部执行要求。

```yaml
invocation:
  invocation_id:
  episode_id:
  attempt_id:

  invocation_type:
  invocation_instruction:
  requested_executor:

  issued_at:
  effective_from:
  completed_at:

  outcome:
  termination_cause:

  annotation_source:
  annotation_confidence:
```

## 1. requested_executor

```text
VLA
HUMAN
ANALYTIC_CONTROLLER
REPLAY
OTHER_POLICY
```

当前纯人工数采中可使用：

```text
requested_executor = HUMAN
```

若尚未形成明确的局部调用边界，Invocation 可以暂不在线标注，后续根据 Task Node、Phase 和语言进行离线构造。

## 2. invocation_type

```text
REACH
ALIGN
GRASP
MANIPULATE
TRANSPORT
PLACE
RELEASE
RETREAT
WAIT
OTHER
```

不额外设置 `atomic_skill`。

## 3. 语言层级

只保留三个语言字段：

```text
task_instruction
node_description
invocation_instruction
```

分别表示：

* 整条任务目标；
* Task Node 的稳定语义定义；
* 当前场景下可以直接交给执行器的局部指令。

## 4. invocation outcome

```text
SUCCESS
PROGRESS
NO_PROGRESS
RECOVERABLE_FAILURE
TERMINAL_FAILURE
INTERRUPTED
UNKNOWN
```

## 5. invocation termination_cause

```text
COMPLETION_SIGNAL
END_OF_COMMAND
TIMEOUT
CANCELLED
SAFETY_STOP
CONTROLLER_FAULT
WINDOW_END
UNKNOWN
```

必须区分：

```text
Invocation 为什么停止
```

和：

```text
Invocation 对任务产生了什么结果
```

---

# 十一、Scenario Event

Scenario Event 记录数采场景实际发生的布置和扰动。

```yaml
scenario_event:
  scenario_event_id:
  episode_id:

  event_type:

  planned_time:
  actual_time:

  planned_parameters:
  actual_parameters:

  result:
```

## event_type

```text
SETUP_COMPLETED
PERTURBATION_STARTED
PERTURBATION_COMPLETED
TARGET_CONDITION_REACHED
TARGET_CONDITION_NOT_REACHED
SCENARIO_STOPPED
```

必须同时保存：

```text
计划施加时间
实际施加时间
计划参数
实际参数
```

例如设计降低夹爪力，但控制器实际未达到目标值时，不能只根据场景名称判断异常已经产生。

---

# 十二、Mechanism Stream

Mechanism Stream 保存无法事后可靠补回的连续机器人和系统信息。

逻辑字段如下：

```yaml
mechanism_sample:
  episode_id:
  frame_index:

  monotonic_timestamp:
  wall_timestamp:
  dt:

  robot_state:
    joint_position:
    joint_velocity:
    gripper_position:
    gripper_velocity:

  online_estimates:
    tcp_pose:
    tcp_linear_velocity:
    tcp_angular_velocity:

  action:
    source_action:
    processed_action:
    commanded_action:

  interaction:
    contact_signal:
    collision_signal:
    force_limit_signal:
    torque_limit_signal:

  safety:
    safety_hold:
    estop:
    joint_limit_near:
    workspace_limit_near:
    fault_code:

  timing:
    observation_age:
    inference_latency:
    command_latency:
    control_cycle_jitter:

  quality:
    frame_valid:
    missing_data_mask:
    clock_alignment_status:
```

当前没有多来源仲裁时：

```text
source_action
```

表示数采员、策略或 replay 产生的原始动作；

```text
processed_action
```

表示经过坐标变换、滤波、限速等控制前处理后的动作；

```text
commanded_action
```

表示实际下发给硬件的命令。

HITL 和 authority 接入后，可以在不修改其他结构的情况下扩展动作来源和控制权字段。

---

# 十三、Mechanism Event

Mechanism Event 保存稀疏系统事件。

```yaml
mechanism_event:
  mechanism_event_id:
  episode_id:

  timestamp:
  event_type:

  source:
  signal_value:
  detector_version:

  related_frame_index:
```

推荐事件类型：

```text
CONTACT_STARTED
CONTACT_ENDED
COLLISION_DETECTED

SAFETY_HOLD_STARTED
SAFETY_HOLD_ENDED
ESTOP_TRIGGERED

WORKSPACE_REGION_ENTERED
WORKSPACE_REGION_EXITED

CONTROLLER_MODE_CHANGED
CONTROLLER_FAULT
SENSOR_FAULT
SENSOR_GAP

RESET_STARTED
RESET_COMPLETED
```

若 Mechanism Event 能由连续信号稳定派生，则连续信号是唯一 source of truth，Event 仅作为检索索引。

---

# 十四、数据保存模式

每项数据必须指定保存模式，而不是统一选择“逐帧”或“仅事件”。

```text
DENSE
DOWNSAMPLED_DENSE
CHANGE_POINT
INTERVAL
EVENT
TRIGGERED_BURST
DERIVED
EPISODE_STATIC
```

## 1. DENSE

持续高频记录，例如：

```text
monotonic timestamp
joint position
commanded action
关键夹爪状态
```

## 2. DOWNSAMPLED_DENSE

持续记录但降低频率，例如：

```text
深度图像
非关键诊断量
部分在线估计
```

## 3. CHANGE_POINT

只在值发生变化时记录：

```text
controller mode
fault code
离散工作状态
```

## 4. INTERVAL

保存开始和结束时间：

```text
Phase
contact
safety HOLD
传感器离线区间
```

## 5. EVENT

保存瞬时事件：

```text
collision detected
ESTOP triggered
reset command
```

## 6. TRIGGERED_BURST

平时只保留环形缓存，在事件触发时写出事件前后高频窗口：

```text
高频力/力矩
触觉
电机电流
控制器内部诊断
高帧率辅助相机
```

## 7. DERIVED

由源数据稳定计算，不作为独立 ground truth：

```text
TCP speed
TCP acceleration
jerk
episode elapsed time
workspace cell
distance to workspace boundary
```

## 8. EPISODE_STATIC

一条 Episode 内不变化：

```text
机器人配置
相机配置
标定版本
Task Profile
Collection Scenario
```

---

# 十五、各项数据的推荐保存策略

| 数据              | 保存模式                    |
| --------------- | ----------------------- |
| 单调时间戳           | DENSE                   |
| 关节位置            | DENSE                   |
| 关节速度            | DENSE 或 DERIVED         |
| 最终下发动作          | DENSE                   |
| 原始动作            | DENSE                   |
| 夹爪连续状态          | DENSE                   |
| RGB 视频          | 连续压缩流                   |
| 深度视频            | 连续压缩或 DOWNSAMPLED_DENSE |
| 高频力/力矩          | DENSE 或 TRIGGERED_BURST |
| TCP pose        | DENSE 或 DERIVED         |
| TCP 速度          | DERIVED                 |
| 加速度、jerk        | DERIVED                 |
| workspace cell  | DERIVED 或 CHANGE_POINT  |
| contact         | INTERVAL                |
| collision       | EVENT 或短 INTERVAL       |
| safety HOLD     | INTERVAL                |
| ESTOP           | EVENT                   |
| fault code      | CHANGE_POINT            |
| controller mode | CHANGE_POINT            |
| Phase           | INTERVAL                |
| Task Node       | 由 Attempt 派生            |
| 场景异常类别          | EPISODE_STATIC          |
| 场景扰动实施          | Scenario Event          |

---

# 十六、核心连续数据不能只在错误时保存

以下数据即使绝大部分处于正常状态，也必须持续保存：

```text
时间戳
机器人核心状态
动作命令
关键视觉观测
传感器有效性
```

因为只记录错误时刻会导致：

* 无法分析错误前兆；
* 缺少正常对照样本；
* 无法完整回放；
* 无法判断未记录是正常还是设备掉线；
* 无法训练时间序列模型；
* 无法判断错误是突然发生还是逐渐积累。

合理方式是：

```text
始终保存低成本核心流
+ 稀疏状态使用区间或事件
+ 高带宽辅助数据使用触发窗口
```

---

# 十七、触发式高频数据

高带宽信号建议采用环形缓冲区：

```text
内存持续保留最近 N 秒
→ 发生触发事件
→ 写出事件前窗口
→ 写出事件后窗口
→ 返回普通模式
```

触发源包括：

```text
碰撞检测
接触开始
安全限幅
人为扰动事件
人工异常标记
控制器故障
```

触发窗口必须包含：

```text
pre-trigger context
post-trigger context
trigger source
detector version
```

即使采用事件触发，也需要周期性保存正常高频窗口，避免高频数据集中只有异常样本。

同时记录检测器健康状态：

```text
ACTIVE
DEGRADED
OFFLINE
```

否则“没有碰撞事件”无法区分为“没有发生碰撞”或“检测器没有运行”。

---

# 十八、避免重复保存

优先保存：

```text
无法重建的源数据
系统在线实际消费的数据
系统在线实际输出的数据
```

尽可能派生：

```text
可以通过确定性算法稳定计算的数据
```

例如，若保存了：

```text
qpos
机器人模型版本
运动学版本
基座坐标变换
时间戳
```

并且 TCP pose 可以可靠重建，则 TCP pose 可作为派生缓存。

但如果控制器实际使用的是带滤波和状态估计的在线 TCP pose，则应保存该在线值，因为它代表系统执行时真实消费的输入，而不是重复 ground truth。

---

# 十九、数据采集组成

一个完整的数据集不应只包含完整成功轨迹，也不应只包含人为失败片段。建议由四部分组成。

## 1. 完整成功主干

用于学习：

```text
长程 Task Node 顺序
自然 Phase 演化
Task Node 转换
完整任务上下文
VLM 高层任务推进
真实成功率
```

## 2. Phase 覆盖片段

用于：

```text
补足短时 Phase
均衡 Phase 数据量
训练局部 VLA
训练 Phase 分类器
```

## 3. Transition 片段

分别采集：

```text
Phase A → Phase B
Task Node A → Task Node B
```

防止模型只学会各阶段内部动作，却无法稳定完成阶段切换。

## 4. 人为 off-nominal 片段

用于：

```text
异常状态识别
鲁棒性训练
失败分类
未来 recovery 初始状态库
```

---

# 二十、人为异常场景的设计

熟练数采员自然成功率接近 100% 时，失败和偏离数据应通过 Collection Scenario 主动设计。

异常场景应绑定：

```text
Task Node
Phase
异常施加位置
异常类别
异常强度
目标结果
停止条件
安全约束
```

建议异常设计优先围绕实际部署可能出现的分布：

```text
初始对齐误差
目标物体轻微位移
目标区域变化
抓取位置偏移
夹爪闭合时机偏差
物体滑动或滑落
路径中出现障碍
视觉短暂遮挡
控制限速或限幅
接触提前或延迟
```

避免为了制造失败而设计与真实部署无关的极端状态。

---

# 二十一、Nominal 与 Off-Nominal 配对采集

对于重要异常场景，尽量进行配对采集：

```text
同一 Task Profile
同一 Task Node
同一 Phase
相近场景配置
相近机器人初始状态
```

分别采集：

```text
Nominal Episode
Off-Nominal Episode
```

例如：

```text
place_cup / align_target / 正常对齐
place_cup / align_target / 目标横向偏移 3 cm
```

配对信息通过：

```text
scenario_profile_id
scene_config_id
pair_group_id
```

关联。

必须明确：

> 人为异常数据的比例不代表真实部署中的自然失败概率。

后续训练失败检测器、reward model 或鲁棒策略时，应单独处理采样先验。

---

# 二十二、标注流程

任务标注只要求数采员或后期标注员完成最必要的任务语义。

推荐流程：

## 第一步：确认 Episode 范围

确定：

```text
recording_scope
context_source
entry_context
```

## 第二步：标注 Attempt

确认该段数据属于哪个 Task Node，以及是否拥有完整前置历史。

## 第三步：标注 Phase Interval

只标注：

```text
当前处于哪个 Phase
Phase 起止位置
是否在 Phase 中途开始或结束
```

不同时要求标注 Execution Status。

## 第四步：核对 Collection Scenario

检查：

```text
场景 setup 是否按计划完成
扰动是否实际施加
目标异常是否实际出现
```

## 第五步：可选标注 Invocation

存在明确局部执行指令时，记录 Invocation 边界和结果。

---

# 二十三、证据与质量字段

所有任务语义标注至少保存：

```text
annotation_source
annotator_id
annotation_confidence
annotation_version
schema_version
task_profile_version
created_at
supersedes_annotation_id
```

`annotation_source`：

```text
OPERATOR_ONLINE
HUMAN_POSTHOC
SYSTEM_NATIVE
RULE_DERIVED
MODEL_DERIVED
```

自动机制字段保存：

```text
detector_version
controller_config_id
calibration_version
kinematics_version
```

缺失数据必须使用：

```text
null
UNKNOWN
frame_valid=false
missing_data_mask
```

不得将缺失数据写成 `0` 或 `false`。

---

# 二十四、Canonical 数据实体

当前标准保留九个核心实体：

```text
1. task_profile
2. collection_scenario
3. episode
4. attempt
5. phase_interval
6. invocation
7. scenario_event
8. mechanism_stream
9. mechanism_event
```

唯一数据来源如下：

| 信息               | 唯一来源                |
| ---------------- | ------------------- |
| 长程任务和 Phase 定义   | task_profile        |
| 数据为什么这样采         | collection_scenario |
| 一次实际连续记录         | episode             |
| 当前属于哪个 Task Node | attempt             |
| 当前任务内部阶段         | phase_interval      |
| 局部执行要求与结果        | invocation          |
| setup 和扰动实际发生情况  | scenario_event      |
| 连续机器人和系统状态       | mechanism_stream    |
| 稀疏机制变化           | mechanism_event     |

---

# 二十五、派生数据

以下信息不作为新的 canonical ground truth。

## 1. Task Node 时间线

由：

```text
Attempt + Phase Interval
```

派生。

## 2. Frame-level Phase Label

将 Phase Interval 按时间展开到每帧。

## 3. Progress

由：

```text
Task Node 顺序
Attempt
Phase
Phase 顺序
Invocation Outcome
```

派生。

严格区分：

```text
task_progress
attempt_progress
invocation_progress
```

不要求人工标注连续 0–1 数值。

## 4. Memory

由：

```text
Episode 历史
Attempt 历史
Phase 历史
Invocation 历史
Scenario 历史
```

生成结构化或自然语言摘要。

Memory 不是人工独立标签。

## 5. Failure Dataset

由：

```text
Collection Scenario
Scenario Event
scenario_result
Mechanism Stream
```

构造。

## 6. VLA Invocation Dataset

```text
Invocation 开始前观测
+ invocation_type
+ invocation_instruction
+ 动作序列
+ Invocation Outcome
```

## 7. VLM Decision Dataset

完整任务数据未来可以进一步派生：

```text
task_instruction
+ 当前观测
+ 已完成 Task Node 历史
+ 当前 Phase
+ 上一次 Invocation 结果
→ 下一次 Invocation
```

不要求当前数采阶段额外标注 memory 或 reasoning。

---

# 二十六、推荐实施顺序

## 第一阶段：Task Profile

选择代表性长程任务，定义：

```text
Task Node
Phase
合法节点顺序
Phase 边界说明
局部语言说明
```

## 第二阶段：机制主干

完成：

```text
统一 monotonic clock
机器人核心状态
动作链路
视频时间索引
接触和碰撞信号
安全和故障
数据有效性
```

## 第三阶段：完整成功主干

采集足够的完整成功轨迹，确保自然 Node 和 Phase 转换得到保留。

## 第四阶段：局部覆盖

按照：

```text
Task Node × Phase
Phase Transition
Node Transition
```

检查数据覆盖，并定向补采。

## 第五阶段：异常场景库

为关键 Node 和 Phase 建立 Collection Scenario，并采用 nominal/off-nominal 配对采集。

## 第六阶段：分层存储

实现：

```text
核心连续流
区间状态
稀疏事件
触发高频窗口
离线派生量
```

## 第七阶段：训练数据导出

生成：

```text
VLA 局部训练集
VLM 高层决策集
Phase 分类数据
异常识别数据
失败前兆数据
鲁棒性评估集
```

---

# 二十七、最终原则

整套标准遵循以下原则：

```text
Task Profile 定义任务如何被理解。

Collection Scenario 定义数据为什么从这里开始、为何包含异常。

Episode 允许从任意 Task Node 或 Phase 开始。

Attempt 表示当前实际执行哪个 Task Node。

Phase Interval 是当前唯一人工细粒度时序标签。

人为设计的异常条件与实际出现的异常结果必须分开。

核心机器人状态和动作持续保存，稀疏状态采用区间或事件。

高带宽辅助数据通过环形缓存和触发窗口保存。

可以稳定重建的信息不重复保存。

Memory、Progress 和模型训练视图全部从 canonical 数据派生。
```
