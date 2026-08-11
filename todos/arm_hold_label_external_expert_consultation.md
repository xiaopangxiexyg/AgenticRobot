# Hold Label 学习、部署与采集时序：外部咨询稿

## 1. 咨询目标

我们正在为双臂机器人策略增加逐 action-keyframe、逐侧的 `hold` 语义输出。希望重点判断：

1. 当前 `hold label` 的目标语义和时间对齐是否适合模仿学习；
2. 数据采集中的 mode 切换与 action/实测运动相差一个 keyframe 是否合理；
3. 连续 action loss 与 hold classification loss 应如何共同训练；
4. 部署端应如何消费预测 label，才能兼顾时序一致性、稳定性和安全性；
5. 哪些离线实验足以区分“合理的事件—效果延迟”和“真实的数据错位”。

希望结论尽量模型框架无关，不限定 OpenPI、某个 checkpoint 或特定机器人。

## 2. 当前稳定语义

每个 action keyframe 有左右臂两个 mode target：

```text
teleop      -> 正常策略/遥操控制
fixed_hold  -> 特殊控制器固定该侧机械臂及其已配置末端执行器
unknown / init_pose / stopped -> 不进入当前二分类 loss
```

当前二分类投影为：

```text
fixed_hold = positive
teleop     = negative
shape      = [action_horizon, 2]  # [left, right]
alignment  = same action-keyframe indices
```

这里的 `fixed_hold` 定义为“控制模式/控制权意图”，不是根据关节速度或 action delta 推导出的“物理静止状态”。正常 `teleop` 帧也可以完全静止。

## 3. 数据采集行为

采集侧的简化链路如下：

```text
operator / teleop source
    -> normal or fix-arm controller
    -> effective dispatched command
    -> current sample cut
       ├── observation / measured state / RGB
       ├── action target
       └── arm-mode label derived from effective command metadata
```

进入 `fixed_hold` 时，特殊控制器会锁存该侧的固定参考；锁存范围包括 arm 和已配置的 hand/gripper。label 来自有效 command 上的 `fix_arm_active` 等事实，而不是事后观察 action 是否静止。

非 position-reference 采集路径会将当前 observation 绑定到“最近一次已确认下发的 command”。因此：

- action 与 label 来自同一份 copied dispatched command；
- observation timestamp 可能比 command timestamp 稍晚；
- 数据中可以记录 sample time、command/action time 和 action age，供离线审计。

## 4. 当前观察到的现象

在现有数据的初步离线检查中，mode 切换行记为 `t`：

```text
label_t 已切换到新 mode
但用 action backward delta 得到的 motion/static 状态通常在 t+1 才匹配新 mode
```

即：

```text
motion_from_backward_delta_t = f(action_t - action_(t-1))
```

在进入 hold 时，可能表现为：

```text
t-1: teleop,      action_(t-1) = 旧运动目标
t:   fixed_hold,  action_t     = 新锁存的 hold reference
t+1: fixed_hold,  action_(t+1) = action_t
```

因此 `action_t-action_(t-1)` 仍非零，但 `action_(t+1)-action_t` 才接近零。

退出 hold 时也可能出现第一帧只完成重新锚定，下一帧才产生明显运动。

需要专家判断：这是否应解释为正常的“控制模式事件在 `t` 生效、效果在 `[t,t+1)` 后可见”，还是 action/label 应采用不同索引。

特别说明：目前还不能只根据 action delta 判定存在采集 bug，因为 action target 的边界变化不等于机器人仍由旧 controller 控制。若 label 与 action 实际来自不同 command identity/timestamp，才属于明确的数据错位。

## 5. 当前学习方式

目标模型输出包含：

```text
continuous action chunk: [H, action_dim]
hold probability:        [H, 2]
```

hold head 的基础训练建议为 per-side BCE-with-logits，并只在：

```text
valid label AND not padding
```

上计算。连续 action loss 仍按原训练链路计算；目前没有因为 `fixed_hold` 自动屏蔽 held-side continuous action loss。

初步统计说明 transition 相对稀少，普通 persistence 或多数类预测可以获得很高 aggregate accuracy，却在切换帧上失败。因此正在考虑显式的 class/transition sampling 或 weighting，但不希望通过盲目平移 label 来改善指标。

另有一个独立的可学习性实验：仅使用 `front RGB + current state` 预测左右 hold label。该实验的目的只是检查是否退化为恒定预测以及 label 是否具有可学习信号，不要求证明任务成功率。

这里存在一个关键可观测性问题：操作者按键或 controller-ownership transition 并不一定能从同一时刻的 RGB/state 中看到。若目标是复现操作者事件，则 transition row 可能不可辨识；若目标是让策略自行决定何时 hold，则它又可以被视为任务条件下的 policy decision。

## 6. 当前部署行为

部署端不会把 hold probability 当作只读 annotation。每个 keyframe 正式到期时执行：

```text
decoded action[t] + hold_probability[t]
    -> per-side threshold / hysteresis / consecutive-step state machine
    -> if enter_hold:
         latch current measured arm reference
         latch last accepted end-effector reference
         replace held-side raw action[t]
    -> canonical safety
    -> downstream reference/execution
```

主要性质：

- label/probability 始终附着于同一个 action keyframe；
- raw-RTC 只负责在 timed plan 中保持这种绑定，到正式 due-pop 才消费；
- 进入 hold 时该侧原始模型 action 会被覆盖；
- 持续 hold 时复用最初锁存的 reference，不跟随测量漂移；
- 退出 hold 时恢复执行当前 keyframe 的模型 action；
- enter/exit threshold 和 consecutive-step count 属于 artifact 配置；
- hold resolution 在 canonical safety 之前发生；
- 缺少有效 hold reference 时 fail closed。

因此若把正 label 整体后移一帧，部署端可能会额外执行一个本应被 hold 覆盖的 raw action。反过来，如果现有 label 本来只是“下一帧将 hold”的预告，那么当前部署又会提前锁存。我们需要专家帮助冻结这两个解释中的正确一个。

## 7. 希望专家重点回答的问题

### 7.1 标签的本体语义与时间支持

1. 对 action-chunk policy，更合理的 target 是：
   - `label_t` 表示 action keyframe `t` 应采用的 controller mode；还是
   - `label_t` 表示上一控制区间 `(t-1,t]` 已观测到的 motion state？
2. 如果 hold 在 `t` 锁存新 reference，是否应该允许 `action_t != action_(t-1)`，但要求之后保持稳定？
3. 是否应保持一个 intent label，同时另建只用于分析的 derived motion label，而不是移动 intent label？
4. 对进入和退出 hold，是否应使用相同的索引约定？

### 7.2 连续 action loss 与 hold loss

1. `fixed_hold` 时部署会覆盖 held-side raw action，该侧连续 action loss 是否仍应保留？
2. 如果需要 mask，应该：
   - 屏蔽全部 fixed-hold 帧；
   - 只屏蔽进入 hold 的边界帧；
   - 保留 reference reconstruction loss，但从普通 action loss 中移除；还是
   - 完全不屏蔽，以保持动作建模的一致性？
3. 对严重不平衡且 transition 稀少的标签，推荐使用 class weight、focal loss、transition oversampling、sequence-level sampling，还是独立的 event/onset loss？
4. 是否应对 transition 附近做 soft label 或 temporal tolerance？这会不会模糊部署端需要的明确 controller state？
5. hold head 应独立建模左右侧，还是显式建模 `00/01/10/11` 联合状态？

### 7.3 输入可观测性

1. 仅 `RGB_t + state_t` 是否足以监督当前 intent label，还是必须提供：
   - observation history；
   - previous predicted/executed mode；
   - previous actions；
   - controller event/ownership state；
   - task phase 或 language/context？
2. 如果部署时没有人类按键输入，训练目标应理解为“复现人类切换事件”，还是“根据任务状态自主选择 hold”？
3. 精确 transition 无法从输入观察时，怎样避免模型通过类别先验学成持续 hold 或持续 active？

### 7.4 部署 state machine

1. 预测概率在当前 keyframe 立即作用是否合理？
2. enter/exit 是否应采用不同阈值和不同 debounce 长度？
3. false-positive hold 与 false-negative hold 的代价通常如何权衡？
4. 退出 hold 的第一个 action 应直接执行、限制 step/rate，还是经过显式 rebase/continuity gate？
5. 模型输出 label 已经是 chunk future 时，debounce 应按模型 keyframe 计数，还是按实际 emission/执行 keyframe 计数？

### 7.5 数据采集与审计

1. 每个 sample 最少应保存哪些时间和 identity，才能证明 label/action/observation 因果对齐？
2. label 应来自 operator request、controller admission、effective command，还是 downstream execution acknowledgement？
3. 在 action/control 频率与 camera/state 频率不同的情况下，应如何定义统一 sample cut？
4. 是否需要同时保存：
   - requested mode；
   - effective controller mode；
   - executed/resolved mode；
   - measured motion state？
5. 如果都保存，哪些可以作为训练 target，哪些只能用于审计？

## 8. 我们正在比较的方案

### 方案 A：保持当前 intent 同索引语义

```text
label_t controls action keyframe t
physical effect may appear at t+1
```

优点：与 controller ownership 和当前部署 resolver 一致；不会额外执行一个 raw action。
风险：同帧 RGB/state 对切换事件可能不可观测；按 backward delta 评估会显得提前。

### 方案 B：将 label 整体后移一帧

```text
label_t aligns with backward motion evidence at t
```

优点：更容易与状态/速度的同帧变化匹配。
风险：把 intent 偷换为 measured-motion label；部署可能延迟 hold；episode 边界和 chunk horizon 处理更复杂。

### 方案 C：保留 intent，同时增加独立 motion/audit label

```text
intent_t          = effective controller mode for action[t]
backward_motion_t = f(state_t - state_(t-1))
forward_motion_t  = f(state_(t+1) - state_t)  # offline audit only
```

优点：不混淆控制意图与物理效果，能够明确诊断时序。
风险：字段与评估复杂度增加，必须严格规定哪些字段可进入模型。

当前内部倾向是 **A 作为正式训练/部署合同，C 只作为审计和诊断**；但希望专家独立判断，而不是默认接受该倾向。

## 9. 建议专家给出的结论格式

希望回复至少包含：

1. 推荐的 label 定义和精确时间区间；
2. 是否应移动现有 label，以及理由；
3. hold 行的 continuous action loss 处理；
4. transition sampling/loss 的推荐方案；
5. 部署 enter/exit 状态机建议；
6. 必须补充的数据字段；
7. 能够在不真机执行动作的情况下证明实现无 bug 的最小实验矩阵；
8. 哪些问题只能通过真机 no-action shadow 或受控 motion test 判断。

## 10. 建议随咨询一并提供的证据

为了避免专家只根据文字猜测，建议从真实数据导出一个匿名、只读的小包：

```text
1. 每个方向至少 20 个 transition：teleop->fixed_hold / fixed_hold->teleop
2. 每个 transition 前后各 5~10 个 keyframe
3. 每行包含：
   - episode/frame/timestamp
   - side
   - label + label source
   - action target
   - measured state
   - command/action timestamp and age
   - controller metadata
4. action delta、measured-state delta 和 label 的对齐图
5. 稳定区间与 transition 区间的类别统计
6. 当前部署 resolver 的纯测试 trace：raw action、probability、resolved state、resolved action
```

不得提供真实设备凭证、远端地址、未脱敏图像或可识别操作者的信息。

## 11. 内部实现证据索引

以下路径仅供项目内部复核，外部专家无需访问：

- `dataset_contracts/arm_mode.py`：arm-mode 和训练投影合同；
- `teleop_runtime/collection/arm_mode_labels.py`：从 effective command 构造逐侧 label；
- `teleop_runtime/real_teleop/collection_bridge.py`：当前 observation 与最近 dispatched command 的绑定；
- `dataset_interface/arm_hold.py`：target、valid/loss/transition mask 投影；
- `dataset_contracts/policy_action_semantics.py`：模型输出 sidecar 合同；
- `robot_motion/deploy/arm_hold.py`：部署 resolver、latch、hysteresis 与 action 覆盖；
- `robot_motion/deploy/README.md`：同步和 raw-RTC 消费顺序；
- `tests/test_arm_hold_training_view.py`：训练视图测试；
- `tests/test_deploy_arm_hold_semantics.py`：部署状态机测试；
- `tests/test_openpi_rtc_final_chunk_adapter.py`：OpenPI raw-RTC sidecar 测试。
