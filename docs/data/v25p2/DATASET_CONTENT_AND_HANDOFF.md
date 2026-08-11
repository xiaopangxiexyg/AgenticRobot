# DexTeleop 数据集内容与交付说明

文中的实物例子是：

```text
DexTeleop_0803/datasets/
  2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage/
```

## 1. 数据集简介

这是由双 RM65、双 DM-Tac 夹爪完成四步毛笔书写任务的真机遥操作数据。每个 Episode
同步组织 15 Hz 低维轨迹、前视与双腕 RGB、双侧触觉、双腕深度范围 mask，并在 v2.3
物理数据上追加 v2.5-p2 Progress 和人工 fine Stage 语义。

这里的版本必须分开理解：

- `dexteleop_dataset_v2.3` 是 Parquet、Episode 元数据和 pose-frame 的物理数据合同；
- `meta/progress/` 是 v2.5-p2 长程任务语义层；
- 目录名中的 `position_v2` 是本次 position 控制/采集配置代际，不是数据格式 v2；
- 目录名中的 `writing_4_stage` 表示四个粗粒度 Progress 子任务，不表示只有四个 fine Stage。

## 2. 示例数据集卡

| 项目 | 只读检查结果 |
|---|---|
| 数据集名称 | `2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage` |
| 采集任务 | 双臂四步毛笔书写 |
| 机器人与末端 | 双 RM65 + 左右 DM-Tac gripper/tactile |
| 控制方式 | 双臂 position，模型主 action 为 accepted-qref FK 的 tool-TCP pose |
| 物理 schema | `dexteleop_dataset_v2.3` |
| 语义扩展 | v2.5-p2 Progress + fine Stage annotation |
| nominal sample rate | 15 Hz |
| Episodes | 355 |
| 正式 sample 数 | 142,576 |
| Episode 帧数 | 最小 10，中位数 407，最大 648 |
| Episode 时长 | 最小 0.605 s，中位数 27.676 s，最大 43.531 s |
| 总正式时长 | 9,807.450 s，约 2 h 43 min 27 s |
| RGB 视角 | `front`、`left_wrist`、`right_wrist` |
| 触觉 | 左右各一组 external binary stream；示例 Episode 每侧各含两个 480×640 `uint8` sensor stream |
| 深度 | 左右腕 480×640 bit-packed depth-range mask；不含 raw depth |
| Progress profile hash | `0874f6b97b3b5255240dd34e6cfac01c76a95c1077df8a20af6b9b3119f3a438` |
| Stage profile hash | `b3e3d19daefa9dbf9fe870c769c062c72b24e1442cfb400d52fae9664dc3934b` |
| policy-I/O layout hash | `sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16` |
| pose-frame contract hash | `sha256:46b5283a5f462aa838754c6321bf425435ea0f080a6b01e0053868ebe2323864` |
| causal snapshot hash | `sha256:f5ff39f205ec1714f3730b9bc0efabb9b591428bdd00e9ae9c59ed0e1add41ad` |

### 2.1 当前磁盘构成

| 子目录 | 文件数 | 大小 | 内容 |
|---|---:|---:|---|
| `data/` | 355 | 0.677 GiB | 每 Episode 一个低维 Parquet |
| `videos/` | 1,065 | 5.081 GiB | 355 × 三视角 RGB MP4 |
| `tactile/` | 2,130 | 157.425 GiB | 双侧 manifest 与未压缩 `uint8` binary streams |
| `wrist_depth/` | 1,420 | 10.369 GiB | 双腕 depth-range mask stream 与 manifest |
| `meta/` | 3,924 | 0.034 GiB | 数据集、Episode、Progress 和 Stage 标注元数据 |
| **合计** | **8,894** | **173.585 GiB** | 完整数据集目录 |

触觉约占总容量的 91%。如果交付物删去触觉，它只能称为一个显式的 reduced-modality
派生版本，不能继续声称是上述完整数据集。

## 3. 数据集名称怎样解读

```text
2026-08-04 _ dmtac _ rm65 _ bimanual _ position_v2 _ writing_4_stage
采集日期       末端/触觉  机械臂   双臂       控制配置代际       任务拓扑
```

名称用于人类检索，不是训练兼容性的权威依据。两个目录即使名称相似，也必须比较
`info.json` 中的物理 schema、policy-I/O、pose-frame、causal snapshot，以及
`meta/progress/profile.cjson`、`stages.cjson` 的内容 hash。

尤其不能做以下推断：

- 不能因为都含 `writing` 就把不同 3/4/5-step profile 直接拼接；
- 不能把 `position_v2` 当成 `dexteleop_dataset_v2.2`；
- 不能按目录修改时间选择最新标注；
- 不能因为目录里有三路视频，就假设每个 sample 的三路图像都有效。

## 4. 四步任务和双臂语义

该数据集绑定线性 Progress profile
`dmtac_rm65_bimanual_position_writing_four_stage_progress` version 2：

| 顺序 | Progress Step | 中文含义 | 左臂 | 右臂 | fine Stage |
|---:|---|---|---|---|---|
| 1 | `right_acquire_brush` | 右手拿取毛笔并送至左手 | `teleop` | `teleop` | `approach`、`grasp`、`lift_and_present` |
| 2 | `left_remove_brush_cap` | 左手取下并放回笔盖 | `teleop` | `fixed_hold` | `remove`、`return` |
| 3 | `right_write` | 右手持笔完成书写 | `fixed_hold` | `teleop` | `write` |
| 4 | `right_return_brush` | 右手调整、放下并归位毛笔 | `fixed_hold` | `teleop` | `align`、`release` |

第一步是明确的双臂协同动作，不能只监督右臂。`fixed_hold` 也不是“删除这一侧 action”：
它表示该侧机械臂和末端执行器保持受控，完整 16D action 仍保留相应目标。

Progress Step 来自采集时在线事实；fine Stage 来自独立的后标注 artifact。两者不能互相
替代，也不应被静默写回 Parquet 成为逐帧标签。

## 5. 目录中每一层有什么

```text
dataset_root/
  data/chunk-000/episode_000000.parquet
  videos/
    observation.images.front/chunk-000/episode_000000.mp4
    observation.images.left_wrist/chunk-000/episode_000000.mp4
    observation.images.right_wrist/chunk-000/episode_000000.mp4
  tactile/
    observation.hand.left.tactile/chunk-000/episode_000000/
      manifest.json
      <sensor-key>.uint8.bin
    observation.hand.right.tactile/chunk-000/episode_000000/
      manifest.json
      <sensor-key>.uint8.bin
  wrist_depth/mask/
    observation.images.left_wrist/chunk-000/episode_000000/
      manifest.json
      stream.bitpack.bin
    observation.images.right_wrist/chunk-000/episode_000000/
      manifest.json
      stream.bitpack.bin
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    modality_validity.json
    devices/
    progress/
      profile.cjson
      stages.cjson
      episodes/<episode_uuid>/
      stage_annotation_drafts/<episode_uuid>/
      stage_annotations/<episode_uuid>/<annotation_set_id>/
```

### 5.1 `meta/` 是索引，不是可选附件

- `info.json`：数据集级 schema、计数、policy-I/O、pose-frame、causal snapshot 和采集配置；
- `tasks.jsonl`：任务 ID 与简述；
- `episodes.jsonl`：每个正式 Episode 的路径、帧数、时间范围、模态引用和 Progress 引用；
- `episodes_stats.jsonl`：Episode 统计；
- `modality_validity.json`：数据集级模态忽略和缺失值策略；
- `devices/`：采集设备的可审计绑定信息；
- `progress/`：长程任务在线事实、fine Stage 草稿和不可变标注集。

这些文件中的相对路径把 Parquet、RGB、触觉、深度和语义层绑定到同一个 Episode。只复制
`data/` 或只复制媒体，会破坏这种绑定。

### 5.2 三种 Episode 身份不要混用

- `episode_index` / `episode_000000`：v2.3 物理文件定位和有序编号；
- `episode_uuid`：Progress artifact 的稳定标识；
- `episode_identity_hash` / capture identity hash：将 Episode record、sample axis 和语义
  artifact 绑定在一起的内容身份。

接收方不应根据列表顺序重新生成 UUID，也不应删除一个 Episode 后重编号剩余 Parquet。

### 5.3 数据集迁移、hardlink 与软链接

真机数据整理、版本投影和历史后处理可能通过 hardlink 复用大文件。因此，
两个不同路径可能指向同一个 inode，并共享完全相同的存储内容。路径名中包含
`backup`、`archive`、`filtered` 或数据版本后缀，都不能证明它是独立备份。

必须遵守以下规则：

- 对 hardlink 的任一路径进行原地写入，会同时改变所有共享该 inode 的数据集。
  视频、Parquet、触觉或深度后处理必须写入全新 inode，完成内容验证后再原子替换；
  不得直接打开 hardlink 目标覆盖写入。
- hardlink 只是空间复用，不是备份。可独立恢复的备份必须位于经核验的独立存储，
  并且其文件与源文件不得具有相同的 `st_dev + st_ino`。
- 完整迁移时，必须把所有可能跨数据集共享 inode 的根目录作为同一个迁移集合。
  分批、逐目录普通复制会丢失跨目录 hardlink 关系，使目标容量接近所有路径大小之和。
- 同文件系统迁移应优先使用保留 inode 的目录重命名。跨文件系统迁移必须显式
  保留 hardlink，并在切换前校验目标端文件集、大小、内容 hash、inode 关系和尾帧解码。
- 旧的数据集根目录可以在目标校验通过后，替换为指向新根目录的“目录级软链接”。
  该软链接只是兼容入口，不是数据副本，也不应计入备份数量。
- 不应把数据集内部的每个 hardlink 机械改为文件级软链接。这不能防止写穿，
  还会引入主文件选择、断链、跨机路径和数据集边界逃逸问题。
- 切换期间不得有 writer、触觉/视频工作进程或后处理进程持有数据集内的可写文件描述符。
  即使目录重命名成功，已打开的文件仍可继续被写入，会造成 manifest 和 payload 失配。

迁移的最小切换门应为：

1. 固化源根目录清单、schema、Episode 数和模态引用清单；
2. 确认没有任何进程以可写方式打开集合内文件；
3. 记录源/目标文件系统、设备号、关键 inode/link count，以及独立备份凭证；
4. 迁移到不覆盖任何现有内容的新目标；
5. 在新目标上重新执行数据合同、媒体和模态引用验证；
6. 验证通过后才建立旧路径的目录级软链接；
7. 重新通过旧路径读取元数据和代表性 Episode，确认兼容入口没有改变。

任何一个门失败都必须停止；不得在“部分目录已迁移、部分目录仍在原地”的
状态下猜测性继续。

### 5.4 2026-08-06 RealmanDMTAC 统一迁移记录

2026-08-06 完成的同文件系统迁移将 46 个完整、非空且模态引用闭合的数据集根统一放到：

```text
/home/user/zc/datasets/RealmanDMTAC/
  DexTeleop/...
  DexTeleop_0803/...
  DexTeleop_StableCollect_0705/...
```

原路径保留为指向上述目标的目录级绝对软链接。旧路径是兼容入口，不是第二份数据；统计容量、
做备份或执行后处理时不得把旧路径与 canonical 目标重复计算。迁移采用同一设备上的目录重命名，
因此原 inode 和跨数据集 hardlink 关系保持不变。

本次迁移的不可变计划、逐步 journal 和完成 receipt 位于：

```text
/home/user/zc/datasets/RealmanDMTAC/.migration/20260806_full_relocation/
  plan.json
  journal.json
  receipt.json
```

计划覆盖 46 个数据集、10,135 个 Episode、156,017 个文件路径和 95,621 个 hardlink 路径。
迁移前后均核对目录 inode/tree 指纹以及 `meta/info.json`、`meta/episodes.jsonl` 的 SHA-256；
旧路径和新路径还分别通过了代表性 v2.3 reader 校验。

以下两个历史数据集仍保留 raw tactile，不得无 receipt 地裁剪或强制压缩：

```text
DexTeleop/datasets/2026-07-27_dmtac_rm65_bimanual_position_v1_old
DexTeleop_0803/datasets/2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage
```

它们各有一个 Episode 的四路 raw 文件长度大于 manifest 声明长度，其中后一数据集还存在非完整帧
尾部。现有 reader 只消费 manifest 声明帧，因此数据合同校验仍可通过；事务压缩器会严格拒绝这些
流。后续修复必须先保存独立备份和修复 receipt，不能猜测性截断尾部。

## 6. 每个 Episode 的模态

### 6.1 低维 Parquet

抽查 `episode_000000.parquet` 得到 479 行、1 个 row group、584 个物理列。584 列中大量是
控制、时钟、传输、传感器新鲜度和诊断证据；它们不是 584 维模型输入。

模型稳定主合同是：

| 字段 | shape | 语义 |
|---|---:|---|
| `action` | 16 | 左 tool-TCP pose 7 + 右 tool-TCP pose 7 + 左 gripper target 1 + 右 gripper target 1 |
| `observation.state` | 16 | 左右 tool-TCP pose 与左右 gripper state |
| `observation.last_action` | 16 | 同一 continuity segment 中的上一 action；段首遵循显式 reset sentinel 合同 |
| `observation.arm.qpos` | 12 | 左右 RM65 各 6 关节的审计/状态向量 |
| `timestamp` | scalar | 正式 sample 的本地 wall-time 时间轴 |
| `episode_index`、`frame_index` | scalar | 物理 Episode 与帧定位 |

Pose 是 `xyz + quaternion(xyzw)`，控制对象是 `tool_tcp`，存储在 canonical
`desk_absolute` 坐标系。当前主 action 是 accepted `q_ref` 经 FK 得到的 pose，而不是原始
VIVE tracker pose。相对 action 是 loader/deployment 的可选投影，不能把磁盘中的 absolute
pose 直接当 relative pose。

DM-Tac 的模型 policy-I/O 是每侧 1D 连续 gripper scalar。采集 hand spec 仍保留兼容的
6-slot 描述，但只有 slot 0 有效，其余是 padding；不要把 6 个槽都解释为独立关节。模型
layout 使用归一化 0–1 语义，采集命令配置还声明了 0–0.9 的有效命令范围，必须通过已绑定
adapter/contract 读取，不能自行按列值猜范围。

其余列按功能分组包括：

- `action.reference.*`：q-ref、raw IK、pose-ref、hold 和 modification reason；
- `observation.arm.*`、`position.transport.*`：commanded/measured 状态、发送与健康证据；
- `observation.hand.*`、`observation.gripper.*`：夹爪状态、触觉引用和时间戳；
- `observation.images.*`：RGB/depth frame index、路径和时钟信息；
- `teleop.*`：VIVE source、arm mode、pose quality 和 channel freshness；
- `dexteleop.*`：行级 schema、pose-frame 和 causal snapshot 绑定。

训练代码应从 `policy_io_layout` 和 `dataset_interface` 构造输入输出，不要使用
`parquet.schema.names` 全列自动入模。

### 6.2 RGB 视频

每个 Episode 有 `front`、`left_wrist`、`right_wrist` 三个 MP4。对三路
`episode_000000` 的实际探测均为：

```text
MPEG-4 / yuv420p / 640×480 / 15 fps / 479 frames
```

每个 Parquet sample 通过 `observation.images.<camera_key>.video_frame_index` 和相应时间戳
关联视频帧。不得只按 `frame_index == video frame index` 的经验假设绕过正式引用和媒体
完整性检查。

### 6.3 DM-Tac 触觉

触觉使用 `binary_stream_v1` 外置存储。每侧 `manifest.json` 描述 sensor key、dtype、shape、
binary path、`sample_frame_index`、tactile frame index，以及 source/available timestamp。

示例 Episode 中，每侧有两个 `480×640 uint8` raw sensor stream，未压缩。sensor key 是设备
身份，不是固定的语义槽位；更换设备或站点后不能靠字符串排序复用左右/区域映射。

触觉 frame count 可以小于 Episode sample 数。例如示例 Episode 有 479 个正式 sample，
左右 tactile manifest 分别记录 445 和 453 帧。缺失 sample 必须按 manifest 映射和 validity
处理，不能重复上一触觉帧后仍标为有效。

### 6.4 双腕深度 mask

当前数据保存 `left_wrist`、`right_wrist` 的 depth-range mask，不保存 raw depth：

- 格式：`dexteleop_wrist_depth_mask_bitpack_v1`；
- shape：480×640；
- dtype 语义：`uint1`，使用 `np.packbits`；
- 深度范围：0.05–0.5 m；
- `1` 表示对齐深度在有效范围内，`0` 表示无效或超出范围。

它是几何有效范围 mask，不等同于物体分割、前景真值或可直接监督的语义 mask。

## 7. Progress、终止和 Stage 标注现状

### 7.1 在线 Progress facts

355 个 Episode 都有 sealed Progress manifest。本次快照共读取到：

| fact 类型 | 数量 |
|---|---:|
| Episode started | 355 |
| Progress advance | 1,039 |
| Episode termination | 355 |
| 显式 Failure fact | 0 |

终止状态为：

| 终止 | Episodes | 含义 |
|---|---:|---|
| `FULL_TASK_COMPLETED / SUCCESS` | 344 | 在线流程完成四步 |
| `MANUAL_STOP / ABORTED` | 8 | 人工停止，不能当完整成功 |
| `SAFETY_STOP / ABORTED` | 3 | 安全停止，不能当正常成功 |

“没有 Failure fact”不等于所有 Episode 都成功；终止事件、异常标注和模态有效性仍需分别
检查。

### 7.2 fine Stage annotation

当前 355 个 Episode 每条至少有一个 sealed Stage annotation set，共 357 个 set；其中 2 个
Episode 有两个版本。357 个 set 中有 9 个带 `BAD_CAPTURE` whole-Episode 异常标记，未发现
局部 abnormal interval。

这些数是 annotation artifact 数量，不是最终训练样本数量：

- 多版本 Episode 必须由训练 profile 显式选择 `annotation_set_id`；
- 禁止按目录时间或字符串最大值静默选“最新”；
- `BAD_CAPTURE`、aborted、Failure/recovery 和缺失模态应由训练 view 统一排除或分流；
- `stage_annotation_drafts/` 是可修改工作状态，`stage_annotations/.../manifest.cjson` 才是
  不可变发布点；
- binding 内可能含 annotator identity，公开交付前应纳入隐私审查。

## 8. v2.3 与 v2.5-p2 的兼容边界

旧 v2.3 reader 可以读取本数据集的 Parquet、视频、action 和 state，因为物理 schema 仍是
`dexteleop_dataset_v2.3`。它会忽略 `meta/progress/`，因此也看不到：

- 四个 Progress Step 和 Attempt；
- Stage 边界；
- aborted/Failure/recovery 的语义区间；
- `BAD_CAPTURE` 和其他训练排除；
- 多版 annotation set 的显式选择。

所以“v2.3 可加载”只代表基础轨迹兼容，不代表普通 v2.3 loader 已正确构建长程任务训练
样本。需要按子任务、Stage、成功/失败或 recovery 训练时，必须使用 p2-aware reader 或
prepared training view。

## 9. 训练准入不能只看目录是否存在

接收方至少应逐层确认：

1. `info.json`、`episodes.jsonl` 计数与实际 Parquet、媒体文件一致；
2. 每条 Parquet 的行数、首尾时间、连续段和 Episode record 一致；
3. 三路视频的 declared/decoded frame count 和尾帧解码通过；
4. tactile/depth manifest 引用存在，frame index 和 available timestamp 合法；
5. Progress profile、Stage profile 和每 Episode manifest hash 一致；
6. 终止为成功，或训练 profile 明确允许相应 aborted/failure/recovery 用途；
7. 显式选择 sealed annotation set，并应用 whole-Episode/局部异常排除；
8. camera slots、policy-I/O、pose-frame 和 causal snapshot 与模型 artifact 一致。

本示例的 `modality_validity.json` 声明数据集级 camera/tactile ignored list 为空，缺失策略是
`frame_absent_valid_false` / `value_absent_valid_false`。这不保证每个 sample 都有全部模态；
loader 仍应读取逐帧有效性和 manifest。基础加载成功不能替代 annotation selection、
训练 profile 和完整数据验收。

## 10. 对外内容

### 10.1 推荐：完整、不可变的数据集根目录

正式交付应整体保留 `data/`、`videos/`、`tactile/`、`wrist_depth/` 和 `meta/`。同时提供：

- 数据集名称和内容快照日期；
- DexTeleop commit 或 release 版本；
- 全目录文件清单、byte size 和 SHA-256 manifest；
- 上表中的 policy-I/O、pose-frame、causal、Progress 和 Stage hash；
- 明确的 annotation-set selection 或 prepared-view manifest；
- 允许的 camera/tactile 组合和缺失模态策略；
- 成功、aborted、Failure、异常和排除数量；
- 数据 owner、联系人、访问级别、许可和允许用途。

“复制到了另一个目录”不等于已有独立备份。若使用 hardlink、部分选择或覆盖式媒体处理，
必须另外验证 device、inode、link count 和独立备份 receipt。

### 10.2 如果只能传递部分模态

不要直接删除大目录后沿用原数据集名称和 `info.json`。应创建显式派生版本，并记录：

- 源数据集 identity/hash；
- 保留和省略的模态；
- 每个 Episode 的选择与排除原因；
- camera constraint，例如三视角训练不得使用 front-only Episode；
- 新的文件清单和完整性 receipt；
- 该派生版本是否仍满足目标训练 profile。

只有低维 Parquet 的交付不能称为“完整 v2.3/p2 数据集”；只有视频而没有 Episode/sample
索引也不能稳定对齐 action。

## 11. 可直接转发的数据集摘要

> `2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage` 是双 RM65 与双 DM-Tac
> 采集的四步毛笔书写真机数据。物理格式为 DexTeleop v2.3，并追加 v2.5-p2 Progress 和
> fine Stage 标注。当前快照包含 355 Episodes、142,576 个 15 Hz samples、三路 640×480
> RGB、双侧 raw tactile 和双腕 depth-range mask，总容量约 173.6 GiB。四个子任务依次为
> 右手拿笔并送至左手、左手取下并放回笔盖、右手书写、右手放回毛笔。344 条在线标记为
> 完整成功，8 条 manual-stop aborted，3 条 safety-stop aborted。Stage 标注存在多版本和
> `BAD_CAPTURE` 标记，训练时必须显式选择 annotation set 并应用终止、异常和模态约束；
> 不能把目录中的全部 Episode 无条件视为可训练成功数据。

## 12. 基础加载代码

面向数据接收方的独立只读示例位于
[`release/phase_dataset_loader`](../../../release/phase_dataset_loader/README.md)。它不依赖
DexTeleop 的机器人、数采、仿真或部署 runtime，可加载：

- 数据集和 Episode metadata；
- 选定的 Parquet 模型主字段；
- sample 引用的 RGB 帧；
- external tactile binary stream；
- 双腕 bit-packed depth-range mask；
- Progress、Stage profile、sealed facts 和显式选择的 Stage annotation set。

```python
from dexteleop_dataset_loader import DexTeleopReleaseDataset

dataset = DexTeleopReleaseDataset(DATASET_ROOT)
episode = dataset.episode(0)
table = dataset.load_parquet(episode.episode_index)
front_rgb = dataset.load_rgb_frame(0, "front", 0)
left_tactile = dataset.load_tactile_frame(0, "left", 0)
left_depth_mask = dataset.load_wrist_depth_mask(0, "left_wrist", 0)

annotation_ids = dataset.stage_annotation_set_ids(0)
annotation = dataset.load_stage_annotation(
    0,
    annotation_set_id=annotation_ids[0],  # 应由 release profile 明确指定
)
```

完整安装、命令行示例、缺失模态行为和验证边界以 release 目录内的 `README.md` 为准。
当一个 Episode 有多版 Stage annotation 时，loader 会要求调用方显式指定，不会自动选择
所谓“最新”版本。
