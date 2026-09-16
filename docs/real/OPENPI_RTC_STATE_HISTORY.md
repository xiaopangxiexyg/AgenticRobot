# OpenPI Model-RTC State History

本文说明 Stage3 书写策略的第一版 causal state-history 实现。该实现只增加历史本体状态，不增加历史图像、视觉特征、任务记忆或 execution-start future state；普通 OpenPI 与普通 Model-RTC 默认保持原行为。

## 1. 时间语义

一次推理请求以控制循环观测 `t` 为不可变 anchor：

- 当前图像、当前 state 和历史 state 都截止于 `t`；
- 历史行严格选择 `t-4`、`t-8`、`t-16` 三个控制帧；
- 每行同时携带相对 `t` 的真实秒级 age，不用名义帧率代替真实时间差；
- RTC committed action prefix 描述 `t` 到 `t+d` 已承诺的动作；
- flow loss 仍只作用于未承诺的 action suffix；
- 推理期间新到达的观测不会替换已经提交请求的 anchor 或历史快照。

训练和部署都要求 exact frame offset、同一 episode、同一 continuity segment。时间戳不递增、控制 step 跳变、缺少任一历史行或跨 continuity segment 都会使样本/请求不可用，不会用最近邻、重复当前帧或零值静默填充。

## 2. 输入内容与张量

历史输入为 `history_states: [3, 28]`，行顺序与 `frame_offsets=[4, 8, 16]` 一致；`history_state_ages_sec: [3]` 严格递增。每个 28-D 行按以下固定顺序组成：

1. 左臂 measured joint qpos，6-D；
2. 右臂 measured joint qpos，6-D；
3. 左臂 measured TCP `xyz + quaternion`，7-D；
4. 右臂 measured TCP `xyz + quaternion`，7-D；
5. 左夹爪 opening scalar，1-D；
6. 右夹爪 opening scalar，1-D。

当前 state 仍沿用现有 OpenPI observation 路径，并不复制到 history。qpos 用 `q / pi` 归一化；TCP 和夹爪的 16-D policy-state 部分复用当前策略的 `state` normalization stats。历史本身不使用旧的 VLM/discrete-token 拼接，也不使用旧 history dropout multiplier。

## 3. 模型位置

启用 history 的 Pi0 配置新增一个独立的 `28-D -> action expert width` 投影，并将三个带显式 age sinusoidal embedding 的 token 放在 action expert prefix、action chunk 之前。RTC 的 prefix-mask embedding、hard projection 和 suffix-only loss 路径保持不变。

history 配置默认关闭。关闭时：

- `Observation` 中的 history 字段为 `None`；
- 不创建 `history_state_proj` 参数；
- 不安装部署采样 observer/readiness hook；
- 既有普通训练、普通 RTC、checkpoint 参数树和请求结构不变。

## 4. 训练数据选择

OpenPI v2.3 reader 从同一个 parquet episode 中读取：

- `timestamp`；
- `frame_index`；
- `collection.continuity_segment_index`；
- 左右臂 `observation.arm.<side>.actual_qpos`；
- 现有 16-D `observation.state`。

一个 Stage3 action anchor 只有在三个 exact past rows 全部有效、frame-index 差分别等于 4/8/16、timestamp 严格递增且 continuity segment 相同时才进入训练。历史行允许位于同一 episode/segment 内的前一 stage，因为它们描述到达 Stage3 anchor 前的真实物理轨迹；不允许跨 episode 或跨采集断点。

Stage3 Prepared Skill profile 继续使用现有 by-date 固定数据选择、三视角 `front + left_wrist + right_wrist`、h32、absolute action、final keep-mask、strict media 和 100000 steps。Prepared Skill 本身保持 `standard_non_rtc_v1`；Model-RTC overlay 再启用 `mask_aware_hard_flow_prefix_v2` 及显式 delay/prefix pairs。

## 5. 部署采样

只有 policy artifact 的 metadata 明确包含 `dexteleop_history_state_input_v2` 时，OpenPI RTC backend 才创建 history recorder。recorder 在每个控制 tick 记录 immutable tuple；在连续 16 个历史间隔尚未形成前，RTC frontend 继续运行控制循环但不会提交模型请求。step/timestamp 断点会启动新的内部 continuity segment 并重新 warm up。

该机制不把模型或 checkpoint 固定到 client：metadata 只声明输入的直接字段语义。没有 `contract_hash`、`continuation_contract_hash`、checkpoint step、checkpoint path 或 model/server 身份绑定。训练 profile 中的 Git revision 和源码文件 SHA 仅用于复现训练代码，不参与部署时的 client/server 模型兼容性判断。

## 6. 当前边界

本版有意不实现以下内容：

- 历史图像；
- predicted execution-start state `s_hat(t+d)`；
- history offset jitter；
- repeat-current / shuffled-history 训练分支；
- history dropout；
- 对真实机器人或 camera 的验收。

这些属于后续对照实验或第二版模型，不应通过修改当前 4/8/16 baseline 的运行时语义隐式加入。

## 7. 已完成的离线检查

聚焦检查覆盖：

- contract 解析拒绝自定义 hash 和 checkpoint 字段；
- exact offset、真实 age、continuity reset 和 immutable snapshot；
- 28-D 左右 qpos/TCP/gripper 顺序；
- history normalization；
- action-expert history 参数默认关闭；
- history-enabled policy metadata 与 artifact 直接字段一致性；
- 既有 RTC frontend/runtime/conditioned backend 回归测试；
- Stage3 by-date profile 与原数据、mask、视角、h32、absolute 和质量计数一致；
- Model-RTC overlay 保留 history base config 且不含 checkpoint/continuation hash 绑定。

当前只完成代码与离线测试，没有计算正式 norm stats、运行 GPU smoke 或启动训练。开始训练前仍需在固定远端数据快照上执行 profile audit/prepare、样本图像检查、history 窗口实际计数复核、norm stats 和短 GPU smoke。
