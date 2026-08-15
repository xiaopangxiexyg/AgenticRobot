# DexTeleop RTC 异步 Action Chunk 研究备忘（2026-07-21）

> 历史状态：本文保留 RTC 方案形成前的行业调研和设计动机，不是 DexTeleop RTC 的
> 实施规范。正式边界、阶段、contract 和验收见
> [`dexteleop_rtc_landing_plan.md`](../../architecture/plans/dexteleop_rtc_landing_plan.md)；当前实现、链路和阶段 gate 图见
> [`dexteleop_rtc_visual_guide.md`](../../architecture/dexteleop_rtc_visual_guide.md)。若两者冲突，以架构主线、control-core
> landing plan 和 RTC landing plan 为准；尤其当前 relative pose 使用固定 chunk-start TCP anchor，不使用本文
> 第六节示例中的 chained-delta 累计公式。

截至 2026 年，行业里还没有统一的“异步 VLA 推理标准”，但已经形成了一套比较稳定的事实范式：

> **VLA 低频、异步地产生一段 action chunk；机器人侧以固定高频连续消费动作；在旧 chunk 尚未执行完时提前请求新 chunk；新 chunk 返回后，按时间轴做延迟补偿、过期动作丢弃和轨迹拼接。**

LeRobot 的 `RobotClient + PolicyServer + action queue` 是目前最清晰的开源参考实现；OpenPI 也提供远程 policy server，但其官方示例中的 `client.infer()` 本身仍是阻塞式 chunk 请求，需要机器人运行时自行增加异步线程和缓冲管理。([Hugging Face][1])

## 一、所谓“实时”，不是让整个 VLA 跑到控制频率

一个标准部署通常有三层频率：

```text
机器人内部伺服环
    高频、确定性、本地运行
        ↑
动作执行环
    20–50 Hz 左右，发送 qpos / pose / velocity target
        ↑
VLA 推理环
    根据模型与硬件，数 Hz 到十几 Hz
    每次输出 H 步 action chunk
```

例如 π₀ 所说的“最高 50 Hz 控制”，主要是指模型生成的动作轨迹可以按 50 Hz 采样并执行，不代表完整 VLM/VLA forward 每秒运行 50 次。其关键是一次生成高频 action chunk。([arXiv][2])

因此现在的“异步实时推理”更准确地说是：

> **soft real-time action streaming over a local real-time controller**

网络、Python 和大模型不承担硬实时安全控制；硬实时和最终安全约束必须留在机器人本地。

## 二、标准系统结构

比较典型的结构如下：

```text
 Cameras ─────┐
              ├─> Sensor Synchronizer ─> Observation Snapshot
 Proprio ─────┘                              │
                                             │ non-blocking
                                             ▼
                                   Policy Request Mailbox
                                             │
                                     gRPC / WebSocket
                                             │
                                             ▼
                                      GPU PolicyServer
                                             │
                                      Action Chunk H×D
                                             ▼
                                  Temporal Alignment / Merge
                                             │
                                      Timed Action Buffer
                                             │
                                             ▼
Measured State ─> Safety / Limits ─> Action Executor ─> Robot SDK
```

实际至少分成三个相互独立的线程或进程：

1. **传感器线程**：持续采集相机、关节状态、末端状态并时间同步。
2. **推理线程**：向 GPU server 发送 observation，阻塞等待模型没有关系，因为它不阻塞控制线程。
3. **执行线程**：按固定周期从 action buffer 中取出当前时刻对应的动作，经过安全过滤后发送给机器人。

LeRobot 的参考方案就是机器人侧维护 action queue，同时在另一线程接收新 chunk；PolicyServer 可以位于独立 GPU 机器，通过 gRPC 流式传输 observation 和 action。([Hugging Face][3])

这里一个容易误解的点是：

> “异步推理”不一定意味着使用 Python `async/await`。
> 只要模型推理不阻塞动作执行线程，就是异步运行时。

## 三、一次完整的异步推理过程

假设：

* 动作执行频率为 (f_a)
* 动作周期为 (\Delta t=1/f_a)
* 模型每次输出 (H) 步动作
* 当前 action buffer 中还剩 (R) 步

运行过程是：

### 1. 提前触发推理

当剩余动作数量下降到阈值时：

```text
R <= trigger_threshold
```

机器人立即采集一份最新 observation，并提交给推理线程，但继续执行旧 chunk。

LeRobot 使用的核心参数是：

* `actions_per_chunk`：一次返回多少步，常见为 10–50。
* `chunk_size_threshold`：队列剩余到 chunk 的某个比例时，提前发送新 observation。
* `aggregate_fn_name`：新旧 chunk 重叠部分如何融合。

阈值越高，越早请求新 chunk，控制更闭环，但 GPU 请求更频繁。([Hugging Face][1])

更可靠的工程计算不是拍一个固定比例，而是根据延迟分布决定：

[
R_{\text{trigger}}
\ge
\left\lceil
\frac{L_{p99}+L_{\text{network}}+L_{\text{jitter}}}{\Delta t}
\right\rceil
+G
]

其中 (G) 是安全余量。

例如控制频率 25 Hz，单步 40 ms，推理与网络的 p99 延迟为 320 ms，那么至少需要预留约 8 步，再加几步余量。

### 2. 旧 chunk 继续执行

模型推理期间，执行线程不等待：

```text
tick t     执行 old_chunk[k]
tick t+1   执行 old_chunk[k+1]
tick t+2   执行 old_chunk[k+2]
...
```

这解决了同步推理中 chunk 边界处停顿的问题。LeRobot 报告的异步方案正是通过重叠执行和计算，消除等待帧并提高任务执行速度。([Hugging Face][3])

### 3. 新 chunk 返回后做延迟对齐

这是整个系统最重要的步骤。

假设 observation 在 (t_o) 时刻采集，新 chunk 在 (t_r) 时刻返回：

[
d =
\left\lceil
\frac{t_r-t_o}{\Delta t}
\right\rceil
]

在推理期间机器人已经执行了大约 (d) 步。

因此模型输出：

```text
[a0, a1, a2, ..., aH-1]
```

中的前 (d) 步，通常已经对应过去时刻，不能从 `a0` 开始直接执行。

朴素但正确的处理是：

```text
new_chunk_valid = new_chunk[d:]
```

然后再和当前 buffer 的未来部分拼接。

如果不做这一步，会出现典型问题：

* 机械臂重复执行过去动作；
* chunk 切换时突然回拉；
* 新旧轨迹时间错位；
* 相对动作被错误地重新锚定；
* 高延迟情况下持续振荡。

## 四、不能只用一个普通 FIFO 队列

简单示例经常使用：

```python
action = action_queue.pop(0)
```

但正式系统更适合维护一个**全局时间轴上的轨迹缓冲区**：

```text
global_tick 1540 -> target action
global_tick 1541 -> target action
global_tick 1542 -> target action
...
```

每个 chunk 应携带：

```text
request_id
control_epoch
observation_timestamp
observation_tick
action_start_tick
action_dt
action_horizon
action_representation
action_frame
anchor_state_id
normalization_version
model_version
valid_until
```

这样新 chunk 返回后，运行时可以明确判断：

* 它基于哪一帧 observation；
* 每个动作应当落在哪个实际控制 tick；
* 哪一段已经过期；
* 哪一段已经承诺执行，不能再修改；
* 哪一段可以被新计划替换。

这比“收到 chunk 就 append 到 deque”可靠得多。

## 五、新旧 chunk 如何拼接

现在主要有三种做法。

### 1. 硬替换

保留已执行和即将执行的 committed prefix，之后全部使用新 chunk：

```text
old:  [executed][committed][---------discard---------]
new:                       [-----------use-----------]
```

实现最简单，适合低延迟、低动态任务，但容易在边界产生位置或速度不连续。

### 2. 重叠加权

在新旧 chunk 重叠部分进行融合：

[
a_i =
w_i a_i^{old} + (1-w_i)a_i^{new}
]

切换开始时旧轨迹权重大，随后逐渐增加新轨迹权重。

LeRobot 的异步实现支持 replace 和 weighted blend 等 chunk aggregation。([Hugging Face][3])

对于位控机械臂，最好不要只保证 position 连续，还要尽量保证：

* 位置连续；
* 速度连续；
* 必要时加速度连续。

对于 Cartesian pose：

* 平移可以使用 cubic/quintic interpolation；
* 旋转应使用 SO(3) log/exp 或 SLERP；
* 不应直接线性平均四元数或欧拉角。

### 3. RTC：生成时就约束连续性

对于 π₀、π₀.₅、SmolVLA 这类 flow/diffusion action policy，现在更先进的方案是 Real-Time Chunking：

* 当前 chunk 正在执行；
* 将必然会执行的动作视为 committed prefix；
* 在生成新 chunk 时冻结或约束这段 prefix；
* 对后续部分做 flow inpainting；
* 新 chunk 天生延续当前轨迹，而不是生成后再简单平均。

RTC 解决的是“异步运行后 chunk 不连续”的问题；普通 async 主要解决“等待推理产生停顿”的问题，两者是互补关系。LeRobot 已将二者分别作为独立推理模式，并建议组合使用。([Hugging Face][4])

## 六、相对 pose action 比 absolute qpos 更难处理

这一点对你的统一动作框架尤其重要。

假设模型在 observation 时刻 (t_o) 输出一串相对末端动作：

[
\Delta T_0,\Delta T_1,\ldots,\Delta T_{H-1}
]

它们通常隐含地依赖 observation 时刻的锚点 (T_{t_o})。

如果 chunk 返回时直接做：

[
T_{\text{current}}\Delta T_d
]

实际上改变了模型动作的含义，因为模型训练时的锚点是 (T_{t_o})，而不是当前实测状态。

更稳妥的做法是：

1. 请求推理时保存 `anchor_state = measured_state(t_o)`。
2. 使用该 anchor 将完整 relative chunk 展开成 nominal absolute trajectory：

[
\hat T_i =
T_{t_o}
\prod_{j=0}^{i}\Delta T_j
]

3. 再根据推理延迟丢弃过期的前 (d) 个 target。
4. 将剩余 absolute trajectory 与当前轨迹拼接。
5. 最后才进入 safety 和 robot backend。

因此：

> **relative action 可以作为模型输出协议，但进入异步 action buffer 前，最好先物化为带时间戳的 absolute target trajectory。**

特别是“坐标系相对 + 增量相对”的 action，不能对 chunk 简单切片后从当前 measured state 重新累计，否则会把推理延迟、执行误差和动作语义混在一起。

## 七、一个较标准的伪代码实现

```python
# 固定频率控制线程：绝不能等待模型
def control_loop():
    while running:
        tick = clock.control_tick()
        measured = sensors.latest_state()

        if (
            planner.remaining_steps(tick) <= replan_threshold
            and not inference_state.in_flight
        ):
            request = make_policy_request(
                request_id=next_request_id(),
                control_epoch=source_epoch.current(),
                observation=sensors.synchronized_snapshot(),
                observation_tick=tick,
                anchor_state=measured,
                committed_prefix=planner.committed_future(tick),
            )
            request_mailbox.put_latest(request)
            inference_state.in_flight = True

        target = planner.target_at(tick)

        if target is None or target.expired:
            target = fallback.hold_or_stop(measured)

        safe_target = safety_filter.apply(
            target=target,
            measured_state=measured,
        )
        robot.send_action(safe_target)
        clock.wait_next_tick()


# 推理线程：这里可以阻塞
def inference_worker():
    while running:
        request = request_mailbox.get()

        response = policy_server.infer(request.observation)

        inference_state.in_flight = False

        if response.request_id_is_stale():
            continue

        if response.control_epoch != source_epoch.current():
            continue

        # relative action 先基于 observation-time anchor 展开
        trajectory = materialize_to_absolute_trajectory(
            response.actions,
            anchor_state=request.anchor_state,
            frame_contract=response.frame_contract,
        )

        now_tick = clock.control_tick()

        planner.splice(
            trajectory=trajectory,
            nominal_start_tick=request.observation_tick + ACTION_OFFSET,
            current_tick=now_tick,
            commit_until_tick=now_tick + COMMIT_GUARD,
            merge_mode="rtc_or_weighted_blend",
        )
```

这里 `request_mailbox.put_latest()` 最好是容量为 1 的 latest-value mailbox，而不是无限增长的队列。GPU 忙时，旧 observation 应被新 observation 覆盖，而不是排队等待推理一堆已经过期的画面。

## 八、正式实现必须有的保护机制

### 只允许一个有效 in-flight 请求

单机器人通常只保留一个有效推理请求。即使允许多个请求并行，也必须依靠：

```text
request_id
observation_timestamp
control_epoch
```

丢弃乱序或过期响应。

否则可能出现：

```text
request 10 比较慢
request 11 先返回并被应用
request 10 后返回，又把系统回滚到旧计划
```

### 动作有 TTL

每个 chunk 必须有最大有效时间。超过期限后：

* 不再执行剩余动作；
* 进入 hold、brake 或 safe retract；
* 不应无限重复最后一个 action。

### 控制线程不做重计算

控制线程中不应出现：

* 图像 resize；
* tokenization；
* 网络等待；
* GPU synchronization；
* 大量 tensor allocation；
* 轨迹全量拟合。

这些应在非实时线程完成，执行线程只读取已经准备好的 immutable trajectory snapshot。

### 本地 watchdog

PolicyServer 断线、GPU OOM、相机异常时，本地机器人仍应能够：

* 停止接受旧 chunk；
* 保持最后安全目标；
* 减速或急停；
* 记录明确的 failure reason。

## 九、模型内部的“异步快慢系统”是另一件事

现在还有一条更前沿的路线：

```text
Slow VLM system
    1–3 Hz
    语义理解、子任务判断、推理
          ↓ latent / semantic buffer
Fast action expert
    25–30 Hz
    当前视觉 + proprio + cached latent
    输出连续动作
```

这种架构把 VLM 和 action expert 自身也解耦。VLA-Perf 将其归为 dual-system VLA；2025–2026 年的一些系统开始让慢 VLM 和快 action policy 通过 latent buffer 完全并行运行。([arXiv][5])

但它目前还不应被视为“标准 VLA runtime”，因为它通常要求：

* 修改模型结构；
* 专门进行异步延迟训练；
* 训练 fast pathway；
* 处理 stale latent；
* 建立跨时间尺度数据采样。

对现有 π₀、OpenVLA-OFT 或自研 action-chunk VLA，优先做外部 runtime 异步通常更实际。

## 十、对你当前框架的直接建议

你的第一版最好不是实现一个简单的 `ActionQueue`，而是实现：

```text
PolicyActionChunk
    ↓
TemporalAlignment
    ↓
FrameAndAnchorMaterialization
    ↓
TimedTrajectoryBuffer
    ↓
SourceArbitration
    ↓
SafetyPipeline
    ↓
RobotCommand
```

其中最关键的五个决策是：

1. **所有 policy chunk 都携带 observation-time anchor 和 frame contract。**
2. **buffer 以 global control tick 索引，而不是纯 FIFO。**
3. **推理结果先做时间对齐，再进入 source arbitration 和 safety。**
4. **human takeover 时增加 `control_epoch`，立即使所有旧 policy request 和旧 chunk 失效。**
5. **第一阶段使用单 in-flight、过期前缀丢弃、absolute trajectory 物化和短窗口平滑；模型确认是 flow/diffusion 后再接 RTC。**

最不建议的实现是：

```text
控制循环中同步调用 VLA
→ 返回 50 步
→ 全部执行完
→ 再请求下一段
```

它虽然也使用 action chunk，但本质仍是同步、宽开环控制，不是现在所说的异步实时 VLA。

[1]: https://huggingface.co/docs/lerobot/en/async "Asynchronous Inference · Hugging Face"
[2]: https://arxiv.org/html/2410.24164v1?utm_source=chatgpt.com "𝜋₀: A Vision-Language-Action Flow Model for General ..."
[3]: https://huggingface.co/blog/async-robot-inference "Asynchronous Robot Inference: Decoupling Action Prediction and Execution"
[4]: https://huggingface.co/docs/lerobot/en/rtc "Real-Time Chunking (RTC) · Hugging Face"
[5]: https://arxiv.org/html/2602.18397v1 "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf"
