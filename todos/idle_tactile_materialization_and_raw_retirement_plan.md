# 空闲期触觉解算、压缩与 Raw 退役方案

状态：设计草案；P0 远端只读画像与 raw 无损基线已完成，force 真实值基准仍待纯离线
解算入口校准；尚未授权 active manifest 切换或 raw 退役实现。

## 1. 目标

在不影响真机遥操和正式数采频率的前提下，对当前数据集内已经封存的
episode 执行后台触觉维护：

1. 从 DMTac raw 图像解算标准触觉场；
2. 对解算结果做确定性、可复现的无损存储；
3. 完整校验数据、时间轴和引用关系；
4. 仅在满足明确退役策略后删除 raw；
5. 一旦请求开始遥操或数采，最迟在 2 秒准备窗口内释放后台资源；
6. 功能默认关闭，并能退回目前的实时 raw 采集与现有离线处理流程。

这里的“可回退”必须拆成两层：

- **运行行为可回退**：关闭新参数后不启动任何后台任务，现有采集路径和数据格式
  完全不变。
- **数据可回退**：只要 raw 尚未删除，可以恢复原 manifest 和原始视图；raw 一旦删除，
  `distributed_force` 无法反演为原图，除非存在独立备份或尚在保留窗口中。

因此，“自动解算”和“删除 raw”必须是两个独立 gate，不能由一个布尔参数同时隐式完成。

## 2. 当前实现与量化结论

### 2.1 当前 raw 结构

正式配置使用 `--dmtac-tactile-fields raw`。每个手侧有两个传感器，每个传感器每帧为：

```text
uint8[480, 640] = 307,200 bytes
```

双手四传感器每个触觉快照约 1.172 MiB；15 Hz 时约 63.3 GiB/小时。

本地真实数据集
`datasets/2026-07-30_dmtac_rm65_bimanual_position_v1` 的 episode 0 有 357 帧，四个
raw stream 共约 429.3 MiB。对其中一个 104.6 MiB stream 使用 zstd level 3 做一次
只读内存测量，结果为 65.9 MiB，比例约 0.630，耗时约 0.33 秒（本机结果，不代表远端
硬件上限）。这说明**先做 raw 无损重打包本身就能明显回收空间**。

### 2.2 当前离线解算能力

`utils/dataset_postprocess/DM_force_calculate` 已经能够：

- 逐帧读取 raw 图像；
- 使用固定传感器标定调用 DMTac SDK；
- 输出 `float32[288, 384, 3] distributed_force`；
- 对输出做 shape、finite、全零占位结果检查；
- 使用 chunked zstd，并对每个 chunk 做解压后字节一致性校验；
- 将结果发布到 `tactile_force/` sidecar。

但它目前不能直接作为在线后台退役器，因为：

- 一个 sensor stream 是一个不可抢占的完整任务，没有 2 秒 quiesce 合同；
- 输出是 sidecar，原 episode 的 canonical `tactile_path` 仍指向 raw manifest；
- 没有安全更新 canonical 引用后再删除 raw 的事务；
- source raw manifest 没有完整输入哈希和退役 receipt；
- 没有与 dataset 删除、重编号、切换、finalize 互斥的维护 lease；
- 现有多进程模式可能与控制、相机、实时触觉 worker 竞争 CPU、内存带宽和磁盘。

### 2.3 “解算”不等于“压缩”

一个 force 帧的逻辑大小为：

```text
288 * 384 * 3 * 4 = 1,327,104 bytes
```

是 raw 单帧的约 4.32 倍。force 的 zstd 压缩率可能很好，但必须在真实数据上测量；不能
仅凭逻辑分辨率断言解算后一定省空间。任何自动退役策略都必须记录实际 `raw_bytes` 与
`materialized_bytes`，并支持“结果不省空间则不退役 raw”的 fail-closed 条件。

### 2.4 解算后的紧凑存储不能直接落 float32

`float32[288,384,3]` 应当是 **reader 的逻辑输出合同**，不应等同于磁盘物理布局。
当前项目已经有 `dmtac_force_qint16_zstd_v1` 雏形：每个 chunk、每个 force 通道使用独立
对称 scale 映射到 int16，zstd 压缩后由 `ChunkedTactileStream` 恢复为 float32。合成测试的
NRMSE 小于 `1e-4`。这条路径避免保存 4.32 倍的裸 float32，但仍需真实数据验证体积和误差。

各候选布局的单传感器、单帧理论大小如下，均未计压缩：

| 物理布局 | bytes/frame | 相对 raw `uint8[480,640]` |
| --- | ---: | ---: |
| raw uint8 | 307,200 | 1.00x |
| force float32 288×384×3 | 1,327,104 | 4.32x |
| force float16/int16 288×384×3 | 663,552 | 2.16x |
| force int8 288×384×3 | 331,776 | 1.08x |
| force int16 144×192×3 | 165,888 | 0.54x |
| force int16 96×128×3 | 73,728 | 0.24x |

因此 full-resolution qint16 必须压到其自身的 46.3% 以下，才小于未压缩 raw；根据本地
raw-zstd 约 63% 的测量，它需要压到约 29.2% 以下才能优于 raw-zstd。是否能达到必须由真实
force 数据决定。

推荐按以下顺序评估，避免过早牺牲信息：

#### 候选 F0：无损 float32 predictor + byte-shuffle + zstd

- 每个 chunk 第一帧保存完整 float bits；后续帧对 `uint32` bit pattern 做 XOR temporal
  predictor；
- 对 4 个 byte plane 做 byte-shuffle，再 zstd；
- 解码后必须与 SDK float32 输出 bitwise 相同；
- chunk 独立，保留随机读取和最多一个 chunk 的解码开销；
- 格式仍简单，不引入图像/视频 codec 或模型依赖。

科学浮点场通常具有相同 exponent、高空间/时间相关性，byte-shuffle/XOR 比直接 zstd 更有
机会消除冗余。但在真实 DMTac force 上没有测量前，不能承诺压缩率。

#### 候选 F1：全分辨率 qint16 + temporal predictor + zstd

以现有 `dmtac_force_qint16_zstd_v1` 为基础：

- 继续使用 per-chunk/per-channel scale，避免通道量纲差异和 clipping；
- 量化后的 int16 code 增加可逆的逐像素 temporal delta（对 code 可逆，量化误差不增加）；
- 可再做 byte-shuffle 后 zstd；
- reader 仍返回 `float32[288,384,3]`；
- receipt 保存 scale、max absolute error、NRMSE、分通道误差和 logical hash。

这应作为首个高保真有损候选。相比 float16，同样是 2 bytes/value，但 qint16 能针对每个
chunk/通道利用完整动态范围，不容易损失接近零的力，因此不推荐用 float16 替代它。

#### 候选 F2：2× area pooling + qint16 + temporal predictor + zstd

- 空间从 288×384 降到 144×192，时间频率保持 15 Hz；
- 三个分量使用相同的确定性面积平均规则；
- 未压缩大小已经只有原 raw 的 0.54x；
- 这是更有把握满足空间目标的方案，但会损失高频空间细节，必须作为不同的 dataset
  feature/profile，不能伪装成完整 288×384 原值；
- 如旧模型需要固定 shape，应由 adapter 显式上采样，不能在 manifest 中声称原 shape。

现有 GUI 最终只把 dense force 缩到约 20×30 做可视化，说明 force 场存在较强空间冗余，
但这不能证明训练只需低分辨率。F2 必须通过任务相关指标后才能成为正式训练格式。

#### 暂不推荐

- **float16**：与 int16 同大小，动态范围/近零精度更难控制；
- **qint8**：虽然 full resolution 理论上仅 1.08x raw，但精度下降显著，只有 qint16/F2
  仍不达标时才值得评估；
- **阈值稀疏化**：force 背景通常有低幅噪声，阈值会改变 contact onset，且可变长索引增加
  随机读取复杂度；
- **JPEG/视频 codec**：面向视觉而非有符号三轴物理量，误差不可控；
- **learned codec**：依赖复杂、版本和跨语言 follow 困难，不符合数据格式目标；
- **只存 global wrench/contact flag**：可以作为额外 derived signal，但不能替代 dense tactile。

正式选择不应只比较 bytes，还要同时报告：

```text
per-channel max absolute error / RMSE / NRMSE
integrated force/wrench error
center-of-pressure error
contact-area IoU at fixed physical thresholds
peak magnitude/location error
temporal derivative/contact-onset error
random frame read latency and chunk cache memory
```

硬规则：任何 force 候选若压缩后不小于 raw-zstd，便没有资格触发 raw 退役；如果用户希望
明显降低长期容量，建议进一步要求 `force_stored_bytes <= 0.5 * raw_uncompressed_bytes`。

### 2.5 2026-08-02 远端 raw 无损基准

只读扫描远端代表数据集：

```text
datasets/2026-08-02_dmtac_rm65_bimanual_position_v2_writing_5_stage
```

得到 89 个 episode、356 路 raw stream，总 raw 体积 `71,406,325,760 bytes`
（约 66.5 GiB），每路 122～1231 帧；未发现任何已发布的 `tactile_force` manifest。

在同一 episode 的四个传感器上各取连续 128 帧，比较完全可逆的 predictor + chunked zstd。
所有候选均逐 chunk 解压并与原 `uint8[480,640]` 做精确数组相等验证。聚合结果：

| 候选 | chunk | zstd | 四路平均比例 | 四路范围 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| temporal uint8 modular delta | 32 | 1 | 0.4503 | 0.4344～0.4631 | 最小 |
| temporal+horizontal delta | 32 | 1 | 0.4538 | 0.4370～0.4644 | 更复杂但未更小 |
| temporal+horizontal delta | 8 | 1 | 0.4606 | 0.4436～0.4713 | 随机读折中 |
| temporal uint8 modular delta | 8 | 1 | 0.4727 | 0.4555～0.4872 | 最简单的随机读折中 |
| temporal XOR | 32 | 1 | 0.5060 | 0.4872～0.5229 | 解码快但更大 |

本批数据上 zstd level 1 反而稳定优于 level 3/7，且压缩更快。按全数据集线性估计：

- delta/chunk32/level1：约 29.95 GiB，节省约 36.56 GiB；
- delta/chunk8/level1：约 31.44 GiB，节省约 35.07 GiB；
- XOR/chunk32/level1：约 33.65 GiB，节省约 32.86 GiB。

因此 `LOSSLESS_RAW_REPACK` 的当前首选物理编码是：

```text
dmtac_raw_temporal_delta_zstd_v1
dtype=uint8
predictor=modular_temporal_delta_per_chunk
zstd_level=1
chunk_frames=8（默认随机读折中）
```

若批处理空间优先，可显式选择 chunk 32；不能静默改变默认值。chunk 8 相比 chunk 32 只多约
2.24 个百分点，但单次随机读取最多解码 8 帧而非 32 帧，更适合作为通用 reader 默认。
正式实现还要用优化后的 modular accumulator 重新测量随机首帧/末帧/缓存命中延迟，当前
Python probe 的全批 decode 时间不作为生产性能上限。

### 2.6 force 实测当前阻塞与边界

首次远端临时解算暴露出：现有 `DMForceCalculator` 虽然消费已记录 raw，但构造
`dmrobotics.Sensor` 时仍会扫描并尝试打开 `/dev/video*`。该行为不符合“纯离线维护进程”
合同，因此已停止继续使用这条路径做 benchmark；没有修改源数据集，临时目录已自动回收。

只读反射确认 SDK 内部存在纯计算组件：

```text
FlowTracker
NormalFromFlowCached
Decomposer
ForceEstimator
DMV1 distributed-force composition
```

但 SDK 1.2.10 未公开一个完整、设备无关且能证明与 `Sensor.process()` 等价的 raw→force
构造入口。不能凭内部类名猜测预处理、标定、模式和尺度后将结果当作 canonical force。

因此 P0 的 force gate 调整为：先实现/获得一个**无设备初始化的 SDK adapter**，并在一次
明确授权的对照测试中证明它与现有 SDK 输出在相同 raw/base-frame/config 上逐帧等价；然后
才运行 F0/F1/F2 体积和质量比较。在该 gate 通过前，唯一可推荐落地的是 raw 无损重打包，
不能删除 raw，也不能声称 qint16/pooling force 已验证。

## 3. 推荐将后台操作拆成三种能力

### A. `LOSSLESS_RAW_REPACK`

将无压缩 raw stream 重打包成 chunked zstd，逐 chunk 做精确 round-trip 校验。它不改变
触觉语义、dtype、shape、frame index 或训练输入，是最安全、真正可逆的空间优化。

### B. `FORCE_MATERIALIZE`

从 raw 派生 `distributed_force`，写入 staging/sidecar，验证后发布为一个可选择的逻辑
materialization。此阶段默认保留 raw。

### C. `RAW_RETIRE`

在 force 已经完整发布、标准 reader 已复读通过、保留期或独立备份条件满足后，删除 raw
文件。这是唯一不可逆步骤，必须有独立的显式开关和 receipt。

推荐首版参数语义如下（最终名称在实现前做一次 CLI 校准）：

```text
--idle-tactile-maintenance off                 # 默认，完全沿用当前行为
--idle-tactile-maintenance raw-repack          # 仅无损压缩 raw
--idle-tactile-maintenance derive-force        # 解算并验证，保留 raw
--idle-tactile-maintenance replace-keep-raw    # force 成为 active view，仍保留 raw
--idle-tactile-maintenance replace-retire-raw  # 显式允许不可逆退役
```

同时提供：

```text
--idle-tactile-worker-cpus ...
--idle-tactile-worker-nice ...
--idle-tactile-quiesce-timeout-sec 2.0
--idle-tactile-raw-retention-sec ...
--idle-tactile-min-space-saving-ratio ...
```

参数未出现时不得改变现有 writer、solver、manifest 或 raw 文件。

## 4. 包边界

遵循当前项目所有权，不让新功能重新耦合 `collection_runtime` 与真机：

### `dataset_contracts`

只在合同稳定后保存纯数据定义：materialization 身份、source/output digest、验证 receipt、
commit/retirement receipt。它不运行 SDK、不调度进程、不删除文件。

### `dataset_interface`

负责：

- 发现已封存且可处理的 tactile stream；
- 构造确定性的维护计划；
- 校验 frame/sample/timestamp/sensor 对齐；
- 解析 active tactile materialization；
- 通过标准 reader 复读 staging/committed 输出；
- 提供 raw/force 两种视图和兼容投影。

它不调用机器人、相机或 GUI，也不拥有后台进程。

### `hardware/DMTAC_gripper`（或其已有 SDK 边界）

负责纯 DMTac 图像到 force 的 vendor-specific 计算能力。它接收图像和固定标定，返回
触觉数组，不理解 episode、dataset manifest 或删除策略。

现有 `DM_force_calculate` 的计算核心应复用/抽离到该边界，而不是让生产 runtime import
一个 CLI 脚本。

### `collection_runtime`

只承载 device/UI/control-neutral 的后台维护协调原语：单 worker 生命周期、durable job
journal、maintenance lease、quiesce/ack、重启恢复和资源预算。它不 import DMTac SDK，
也不自己定义数据 schema。

### `teleop_runtime`

作为真机应用适配层：

- 把当前 resolved dataset root 和应用状态提交给维护协调器；
- 将 `teleop start`、countdown、recording、finalizing、dataset switch 转成 typed gate；
- 在 2 秒准备期等待 quiesce receipt；
- 把进度和故障发布给 `control_station`。

它不实现 force 算法，也不自行重写数据协议。

### `control_station`

仅展示状态与发送 operator intent，例如启用模式、暂停、重试、查看失败原因；不直接操作
文件或拥有 worker。

## 5. 哪些时间允许后台运行

不能仅依赖“没有正在写帧”这一条件。推荐首版使用严格 admission：

```text
允许：
  - teleop 尚未进入 RUNNING，且设备启动关键路径已结束；或
  - 明确 Hold、运动 command stream 已 disengage，且没有 open/paused episode；
  - 当前 dataset root 已稳定；
  - 没有 countdown、recording、finalizing、删除、重编号、profile/dataset 切换；
  - 仅处理 meta/episodes.jsonl 中已经 seal 的 episode。

禁止：
  - active teleop；
  - 数采倒计时、warmup；
  - recording active，包括 progress failure 后保持打开的 paused episode；
  - episode finalizing；
  - 相机启动、自检、设备重连和启动归位等资源敏感阶段；
  - 当前仍在写入或尚未 seal 的 episode。
```

停止数采但仍处于双臂遥操时不算空闲。Failure Hold 如果 episode 仍保持打开，也不算可维护
阶段。这样首版不会利用所有理论空闲时间，但状态边界清楚，后续可用实际数据逐步放宽。

## 6. 2 秒 quiesce 合同

开始遥操必须先经过后台维护 admission barrier：

```text
TELEOP_START_REQUESTED
  -> 禁止领取新 job
  -> 向独立 worker 发 quiesce
  -> worker 完成当前原子写入/关闭 staging fd
  -> worker 退出并被父进程 reap
  -> 释放 maintenance lease
  -> 返回 QUIESCED receipt
  -> 才允许遥操 RUNNING
```

2 秒是资源释放 deadline，不是要求完成当前 episode 解算。

现有 solver 不能保证在 2 秒内协作式 checkpoint，而且 SDK 是否依赖跨帧状态尚未被证明。
因此首版推荐：

- 单独子进程、单 job、单 sensor stream；
- 收到停止请求后给短 grace period；
- 未完成则终止隔离子进程，未发布的 `.tmp` 保留为可识别 staging 或安全清除；
- 下次从该 stream 头部重算，先换取可靠抢占；
- 只有证明 SDK 可从任意 chunk 恢复相同结果后，才引入细粒度 checkpoint 续算。

如果 2 秒内无法确认 worker 已退出，必须 fail closed：不允许让后台 worker 与遥操重叠。
可配置的行为应优先为“强制终止并 reap”；若仍无法回收，则阻止进入 RUNNING 并在 GUI
明确报错，而不是悄悄带着后台任务开始控制。

数采 countdown 也应重复检查同一 quiesce receipt，作为 defense in depth。

## 7. 每个 episode 的 durable 状态机

建议每个 side/sensor stream 的持久状态为：

```text
DISCOVERED
  -> CLAIMED
  -> SOLVING
  -> STAGED
  -> VERIFIED
  -> COMMITTED_KEEP_RAW
  -> DELETE_ELIGIBLE
  -> RAW_RETIRED
```

补充状态：

```text
QUIESCE_REQUESTED
FAILED_RETRYABLE
FAILED_PERMANENT
QUARANTINED
```

job journal 采用 append-only fact + 原子生成的当前快照；临时文件永远不使用 canonical
文件名。进程重启时：

- 无 receipt 的 staging 文件不发布；
- `STAGED` 重新完整验证；
- `VERIFIED` 可以幂等 commit；
- `COMMITTED_KEEP_RAW` 先复读 active manifest，再判断是否允许退役；
- `RAW_RETIRED` 仍校验 materialized output 和 retirement receipt；
- 任何不一致进入 `QUARANTINED`，不自动猜测或删除。

## 8. 原子发布与 canonical 引用

当前 force solver 只写 `tactile_force/`，而 parquet row 与 `meta/episodes.jsonl` 仍引用
`tactile/.../manifest.json`。直接删除 raw 会留下悬空引用。

推荐保持 row 中的 manifest 路径稳定，把该 manifest 作为逻辑触觉入口：

1. 保存原 manifest 的完整 bytes/hash；
2. 在唯一 staging 目录写 force stream 和新 manifest；
3. 完整验证；
4. 原子替换原路径上的逻辑 manifest，使其 streams 指向已验证 force materialization；
5. fsync 文件及父目录；
6. 用项目标准 tactile reader 从原 `tactile_path` 复读；
7. 写 commit receipt；
8. raw 进入保留期；
9. 满足退役 gate 后才 unlink raw，并写 retirement receipt。

原 parquet 行内可能保留 capture-time raw shape/field 信息，因此 reader 必须明确：行内字段是
capture provenance，当前物化以 logical manifest 为准。若现有训练 reader 仍把行内 raw shape
作为 authoritative，则在实现 `replace-*` 前必须先修正统一 reader，不能静默制造冲突。

逐 episode 提交会令同一 dataset 在维护过程中暂时同时存在 raw-active 与 force-active episode。
必须在 dataset-level materialization index 中显式记录状态，reader 按 episode 解析；默认训练
视图可要求只选择相同、已验证的 materialization。若禁止任何临时混合，则只能先为全部 sealed
prefix 完成 staging，再做 dataset-level 原子切换，但这会延迟空间回收。

## 9. 删除 raw 前的完整验证清单

每个 sensor stream 至少要求：

1. source episode 已 seal，frame count 非零且与 manifest 一致；
2. 处理前后 source 的 path、device、inode、size、mtime 稳定；
3. 记录 source SHA-256；
4. sensor key 集合、side、frame count 完全一致；
5. 每帧 `sample_frame_index`、`tactile_frame_index`、timestamp、frame id、snapshot sequence
   与 source 一致；
6. 必须存在传感器对应的固定 calibration，使用 `required` 而不是静默无标定；
7. receipt 保存 calibration 内容 hash、SDK/processor revision 和完整参数；
8. output dtype/shape 正确，所有数值 finite；
9. 拒绝全程全零占位结果；
10. 每个压缩 chunk 解压字节相等，完整 logical output SHA-256 一致；
11. 标准 reader 对首帧、中间帧、末帧、所有 chunk 边界复读成功；
12. manifest pre/post hash 和 active stream 文件 hash 全部写入 receipt；
13. raw 路径 link count 被记录；若 `nlink > 1`，明确提示 unlink 不等于实际释放空间；
14. 若策略要求可恢复备份，验证备份为独立 inode，并按项目规则验证设备/inode/link count；
15. 实际输出大小满足空间收益策略，否则保持 raw 或转用 `LOSSLESS_RAW_REPACK`。

任何一项失败：不切换 active manifest，不删除 raw。

## 10. 资源隔离

- 后台解算必须是独立进程，不得放入控制 loop、GUI server、相机线程或 live tactile worker；
- 默认一个 worker，不自动创建 CPU 数量的一半；
- CPU affinity、nice 和 I/O priority 独立可配置，不能复用控制核或 DMTac live worker 核；
- worker 只接受不可变 job spec，并通过 receipt 返回结果；
- 调度器按最老 sealed episode 优先，单次只 claim 一个 stream；
- 内存设置硬上限，禁止整 episode force 常驻内存；
- dataset 切换先 quiesce、release lease，再解析新 root；
- episode 删除/重编号、训练读取和其他 in-place postprocess 必须遵循共享 maintenance lease。

## 11. 推进顺序与可回退 gate

### P0：只读画像与真实数据 benchmark

- 统计多个 episode 的 raw bytes、raw-zstd bytes；
- 离线派生 force，并在完全相同帧上比较：直接 float32-zstd、F0 无损 predictor/byte-shuffle、
  F1 full-resolution qint16 predictor、F2 2× area-pool qint16；
- 对 F0 验证 bitwise round-trip，对 F1/F2 计算分通道及任务相关误差；
- 统计每种候选的尺寸、吞吐、随机读取耗时、CPU 和内存峰值；
- 验证 SDK 跨帧状态与重复运行 bitwise/数值确定性；
- 不改 manifest、不删除任何源数据。

退出条件：至少有一种 force 布局满足约定的空间上限和误差合同；否则不以 force 替换 raw，
只推进 raw 无损重打包。并明确 remote 单 worker 的资源预算。

### P1：纯维护合同与 dry-run planner

- 实现 sealed episode discovery、job plan、lease、journal、receipt；
- `--idle-tactile-maintenance off` 保持默认；
- dry-run 只展示候选、预估空间和阻塞原因。

### P2：raw 无损重打包

- 先落地 `LOSSLESS_RAW_REPACK`；
- 精确 round-trip、原子替换、可恢复 manifest；
- 验证 quiesce 与 crash/restart。

这是风险最低且已从真实数据看到约 37% 空间收益的路径。

### P3：force shadow materialization

- 自动在空闲期解算到 sidecar；
- 完整校验和 receipt；
- 保留 raw，现有训练/采集默认不读取新结果；
- 比较实时性能、解算正确性和空间收益。

### P4：active view 切换但保留 raw

- 标准 reader 和 v2.3/当前数据适配器支持 logical materialization；
- episode-by-episode 原子切换；
- 一键恢复原 manifest；
- 完成训练/部署兼容验证。

### P5：受控 raw 退役

- 仅显式启用 `replace-retire-raw`；
- 默认有保留窗口，先从旧 episode 开始；
- 退役前再次获得 exclusive maintenance lease 并复验；
- 写不可变 retirement receipt 后删除；
- 正式启用前用复制的数据集做断电、SIGTERM、SIGKILL、磁盘满、checksum mismatch 测试。

任一阶段都能通过关闭参数回到现有流程。P0-P4 不要求删除 raw；P5 才产生数据级不可逆行为。

## 12. 验收用例

### 控制与抢占

- 解算中请求开始遥操，2 秒内 worker 被 reap，控制线程无频率下降；
- worker 不响应时，RUNNING 被 fail-closed 阻止且 GUI 明确提示；
- 数采 countdown 再次验证无后台 worker；
- Hold 结束、dataset switch、进程退出均无孤儿 worker。

### 数据完整性

- source/output frame 数、时间轴、sensor keys 完全一致；
- calibration 缺失、全零 force、NaN/Inf、尾 chunk 损坏均不删除 raw；
- manifest commit 前/中/后 kill，重启 resolver 均得到唯一合法状态；
- 标准 reader 经原 tactile path 读取 active materialization；
- v2.3 compatibility adapter 对 raw-active/force-active episode 行为明确；
- 删除 SQLite/cache 后，canonical receipt 能重建 materialization index。

### 删除安全

- hardlink、不同设备备份、只读文件、磁盘满、目标已存在均 fail closed；
- raw 只在 commit receipt 和 retention/backup gate 后删除；
- 删除后统计真实释放块数，而不是只看路径消失；
- 已启动外部 reader 或未取得 exclusive lease 时不退役。

## 13. 实现前需要用户确认的语义

以下选择会影响公开数据语义或不可逆删除，不能由实现自行猜测：

1. **目标触觉表示**：这里的“触觉数据”是否确定为当前 SDK 的
   `distributed_force float32[288,384,3]`，还是还要保存 depth/shear/global force 等字段？
2. **数据级回退要求**：允许在所有验证通过后永久删除 raw，还是必须保留最近 N 个 episode/
   N 小时，或先复制到独立备份设备？推荐首版至少保留一个可配置窗口。
3. **Hold 是否算空闲**：推荐首版只有“无 open episode 的明确 Hold”可运行；progress failure
   导致 episode 保持打开的 Hold 不运行。是否符合预期？
4. **2 秒无法停止时的策略**：推荐强制终止隔离 worker；仍无法 reap 就阻止遥操进入 RUNNING，
   绝不允许重叠。是否接受这种 fail-closed 行为？
5. **临时混合 materialization**：是否允许同一 dataset 在后台迁移期间，一些 sealed episode
   仍为 raw、另一些已为 force，并由 per-episode index 明确区分？若不允许，空间只能在整批
   staging 完成后统一回收。
6. **训练输入兼容**：现有/计划中的训练任务是读取 raw 还是 distributed force？如果既有训练
   仍要求 raw，则 raw 退役必须绑定到明确的新 dataset feature/profile，而不能对旧 profile
   静默替换。

在这些问题确认前，可以安全推进 P0 的只读 benchmark 和 P1 的 dry-run 设计；不应推进 active
manifest 切换或 raw 删除。
