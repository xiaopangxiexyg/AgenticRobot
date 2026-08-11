# Writing front-only state dropout 与起始点泛化计划

## 目标与边界

目标是降低 writing front-only 策略对固定起始位姿和少数轨迹模板的依赖，同时保持闭环状态反馈能力。当前只做规划，不修改 DexTeleop/OpenPI 代码、不启动或停止训练。

所有后续训练固定 `num_workers=4`。不再使用 12-worker 建议。

## 已确认的当前事实

- front-only task-ready manifest 有 479 个 episode、78,154 帧，全部来自同一原始数据源；episode 长度 61–321 帧，中位数 161 帧。
- manifest 中 479 个 episode 的 `quality_class` 和 `source_quality` 均为 `unlabeled`。因此普通随机 frame validation 无法证明成功率或起始点泛化。
- relative/absolute front-only 配置当前都使用：
  - `base_0_rgb`；
  - h16；
  - normalized noise `0.02`；
  - `dropout_probability=0.10`；
  - `dropout_granularity="whole_state"`；
  - structured pose noise `1 mm / 0.75 deg`；
  - structured `side/arm/end_effector` dropout 全为 0；
  - `num_workers=4` 和 `shared_worker_v1`。
- 只要任一 state dropout 大于 0，现有实现会启用 `masked_normalized_concat_v1`：16 维状态值后附加 16 维 availability mask。训练时被遮蔽的零值不会和真实零状态混淆；部署时 mask 全为 1。
- structured dropout 在 normalization 前按组件工作；normalized dropout 在 normalization 后工作。两级 mask 会合并。action、last_action、离散状态和 arm qpos 审计列不在该 augmentation 内。

## 核心判断

不建议把 `whole_state` 从 0.10 直接提高到 0.30 或更高。

原因：起始点捷径主要来自左右臂绝对 pose；整状态 dropout 同时移除 pose 和 gripper，而且 absolute 策略在 front-only 图像下难以恢复精确机械臂状态。过高的 whole-state dropout 容易把策略从“少依赖起始点”推向“忽略闭环状态、按图像和训练模板开环输出”。

首选是保留现有 10% whole-state dropout，再增加每侧独立的 structured arm-pose dropout。这样直接打断起始 pose 捷径，仍保留未被遮蔽侧和 gripper 状态，并继续显式提供 mask。

## 推荐候选

### B0：现有基线

- normalized whole-state dropout：0.10
- structured arm dropout：0.00
- structured side/end-effector dropout：0.00

### B1：推荐的第一档

- normalized whole-state dropout：0.10
- structured arm dropout：0.15，每侧独立采样
- structured side dropout：0.00
- structured end-effector dropout：0.00
- 所有 noise 参数保持不变

B1 的理论覆盖率：

- 任一给定 arm pose 被遮蔽：`1 - (1 - 0.10)(1 - 0.15) = 23.5%`；
- 两侧 arm pose 同时不可用：`0.10 + 0.90 * 0.15^2 = 12.025%`；
- gripper 被遮蔽：仍为 10%。

这比当前每个 pose 仅 10% dropout 更强，但不会大幅增加 gripper 缺失，也只小幅增加双臂 pose 同时缺失的比例。

### B2：仅在 B1 仍明显过拟合时测试

- normalized whole-state dropout：0.10
- structured arm dropout：0.25
- 其他项与 B1 相同

B2 中单侧 arm pose 的有效 dropout 为 32.5%，双臂 pose 同时不可用约 15.6%。这是上限候选，不应直接作为默认值。

### 暂不采用

- `whole_state >= 0.20`：全状态丢失过多，对 absolute 风险最高。
- `dropout_granularity="side"` 直接升到 0.20–0.30：会同步丢失该侧 gripper，无法单独验证“arm 起始 pose 捷径”假设。
- 同时增大 noise 和 dropout：会混淆归因。第一轮保持 normalized/SE(3)/gripper noise 不变。
- dropout action 或 last_action：这属于不同的时序条件契约，不能作为本轮 state ablation 顺带修改。

## 分阶段验证

### P0：先建立不会泄漏的泛化划分

禁止随机按帧切分。以 episode 为最小单位，增加版本化 sidecar/manifest，不移动、不复制原始数据：

1. 在 canonical desk frame 中提取每个 episode 的有效起始状态：左右臂位置、方向和 gripper。
2. 提取相对起点的轨迹描述：路径长度、终点位移、主要运动方向、曲率/转折和左右臂协同量。
3. 对 start-state 和 trajectory descriptor 分别聚类。
4. 建立三组 episode：
   - ID：常见起点与常见轨迹；
   - start-OOD：起点在远端分位或留出 cluster，轨迹形态尽量与训练可比；
   - trajectory-OOD：起点尽量与训练可比，但轨迹 cluster 留出。
5. 同一 episode、相邻帧和同一派生来源不得跨 split。

这一步用于区分“起始点过拟合”和“轨迹模板不足”。现有 manifest 全部 unlabeled，所以还应补一个只读评价 sidecar，至少记录 success/suboptimal 或可复核的阶段完成情况；不回写原数据。

### P1：DataLoader 与 mask 预检

对 B0/B1/B2 各抽样至少 10,000 个训练 item，固定 `num_workers=4`，验证：

- 实际 whole/left-arm/right-arm/both-arm dropout 频率接近理论值；
- epoch 变化后同一 episode/frame 的 mask 会变化，四个 persistent worker 均收到新 epoch；
- dropout 后仍是 16 state value + 16 mask 的既有输入契约；
- 未 dropout 的维度和 gripper 未被 arm-only dropout 错误清零；
- deployment/no-augmentation 路径始终产生全 1 mask；
- relative 与 absolute 的 action、last_action 内容和维度完全不变。

若上述任一项不成立，不进入训练。

### P2：低成本训练筛选

relative 与 absolute 分开比较，不能把一个模式的最优概率直接套到另一个模式。

1. 固定数据、seed、h16、学习率、图像增强、训练步数和 `num_workers=4`，唯一变量是 dropout 配置。
2. B0/B1/B2 先训练到 30k，使用新实验名，不覆盖或 resume 既有 checkpoint。
3. 30k 只淘汰明显失败者；B0 与最优候选继续到 60k。
4. 仅当 60k 的 start-OOD 和 trajectory-OOD 指标均满足门槛，才把候选训练到 150k。
5. 最终候选至少复验两个 seed；单 seed 结果只作为筛选证据。

优先顺序建议：先跑 relative 的 B0/B1；若 B1 有改善再加 B2。absolute 从 B0/B1 开始，但采用更严格的状态缺失与终点误差门槛。

### P3：离线指标

所有 loss 都在部署态输入上评估，即 mask 全 1，不在随机 dropout 输入上报告主结果。

- ID、start-OOD、trajectory-OOD 分组 action loss；
- h16 首步、前 4 步和全 chunk 的平移/旋转/gripper 误差；
- absolute：绝对终点误差与相对当前 TCP 的异常大跳；
- relative：chunk 积分后的终点误差、方向错误和累计漂移；
- 小范围一致性测试：在物理可接受范围内扰动 state，检查 action 是否连续，不做与图像严重矛盾的大幅 state swap；
- state ablation sensitivity：正常 state、单侧 arm mask、双臂 arm mask，确认模型降低捷径依赖但没有完全忽略有效 state；
- 按起始位姿距离和轨迹 cluster 绘制误差，而不是只看全局均值。

### P4：晋级门槛

B1/B2 相对同 seed 的 B0 至少满足：

- start-OOD action loss 或 chunk endpoint error 改善至少 10%；
- ID 指标退化不超过 5%；
- trajectory-OOD 不得明显恶化；
- 正常 state 与轻微 state 扰动下无 action 跳变、幅值膨胀或趋向固定模板；
- absolute 的绝对终点/大跳指标不恶化；
- 后续真机按起始点分桶的成功率提高，而不是仅随机测试总体成功率持平。

若 B1 与 B2 都只改善 masked-state 测试、不改善 start-OOD，则停止继续增大 dropout。此时主要问题是数据起点/轨迹覆盖不足，而不是 state shortcut。

## 后续落地时的配置与兼容策略

现有配置名和既有 checkpoint 必须冻结，不能原地改变 0.10 的含义。后续应新增显著区分的配置，例如：

- `writing-phase1-grap-pen-front-only-h16-relative-task-ready-state-dropout-v2`
- `writing-phase1-grap-pen-front-only-h16-absolute-task-ready-state-dropout-v2`

新配置只改变 state dropout 字段，并记录：父配置名、dropout 合同、manifest/hash、norm-stats hash、seed 和 `num_workers=4`。旧配置即为回退路径。

若数据和 normalization 样本集合没有变化，应优先复用并核对原 norm stats hash；不得让随机 dropout 后的状态参与 norm-stat 估计。实际落地前需要用 `compute_norm_stats` 路径确认这一点。

## 最终推荐

第一轮默认候选选 B1：保留 `whole_state=0.10`，新增 `arm_dropout_probability=0.15`，其他 state noise/dropout 不变，worker 固定为 4。

这是一项有针对性的正则化实验，不应被当成解决轨迹覆盖不足的替代方案。最终是否采用，必须由 episode-level start-OOD/trajectory-OOD 和真机起始点分桶结果决定。
