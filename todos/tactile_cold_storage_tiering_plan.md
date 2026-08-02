# 触觉冷存储分层与无触觉训练视图方案

状态：P1 raw 无损重打包已在本地与远端全部结构完整的封存数据集完成并全量校验；冷存储
export、热端退役和 external-root 绑定均尚未实施。

## 1. 结论

方案可行，但不能把 `tactile/` 直接搬走后仍把剩余目录解释成一个物理自包含、通过完整
v2.3 校验的数据集。

正确语义是：

```text
一个逻辑数据集
  = hot/core artifacts（meta、data、videos、progress 等）
  + cold tactile artifacts（tactile/）
  + 二者之间不可变的 identity/hash binding
```

训练分成两个显式视图：

```text
tactile-free training view
  只要求 numeric/video/progress
  不打开 tactile manifest 或 payload

tactile-enabled training view
  必须绑定 cold tactile root，或先将 tactile 完整恢复到数据集目录
```

触觉暂时离线是**存储可用性状态**，不是采集时缺失、无效或被忽略。不得通过改写
`meta/modality_validity.json` 把已采集触觉声明成 ignored/invalid。

## 2. 当前实现事实

当前 v2.3 外置触觉有两层引用：

- `meta/episodes.jsonl[*].tactile_paths` 指向每侧 manifest；
- Parquet 的 `observation.hand.<side>.tactile_path` 与
  `tactile_frame_index` 指向相同逻辑帧。

`validate_v23_dataset()` 会检查 manifest 和 payload 文件存在，因此简单删除/搬走
`tactile/` 后完整校验必然失败。这是正确行为，不能为了冷存储把 validator 放宽成静默忽略。

另一方面，prepared training view 已有：

```text
modality_selection = numeric | video
```

且其 source inventory 只纳入所选训练模态；现有训练规划也明确要求未启用 tactile 时不打开
tactile manifest 或 binary stream。因此无触觉训练在架构上已经成立，但仍需补齐专门的
fixture，保证构建和训练入口不会先调用“完整物理数据集校验”。

## 3. 不应采用的做法

### 3.1 直接搬走后不留 receipt

这会让触觉离线、传输失败、文件损坏和误删除无法区分，也无法证明放回的是原内容。

### 3.2 删除 episodes/Parquet 中的 tactile 引用

这会改写捕获事实和 episode 内容身份，并让后续恢复无法证明时间轴/帧索引仍属于原数据。

### 3.3 将 tactile 标为 capture ignored 或 invalid

这是错误语义：数据实际采集成功，只是当前物理层不在线。

### 3.4 用同文件系统 hardlink 作为冷备份

不同路径不等于独立备份；相同 inode 的 hardlink 不释放空间，也不能抵抗 inode 损坏。

### 3.5 把绝对冷存储路径写进 portable dataset identity

机器本地 mount/path 会变化。canonical 绑定保存内容身份与逻辑 prefix，物理位置由 workspace
binding 或 CLI 配置解析。

## 4. 推荐物理布局

示例：

```text
/hot/datasets/<dataset_name>/
  meta/
  data/
  videos/
  wrist_depth/
  derived/
  # tactile/ 可以当前不驻留

/cold/dexteleop_tactile/<dataset_identity>/
  tactile/
    observation.hand.left.tactile/...
    observation.hand.right.tactile/...
  tactile_inventory.cjson
  export_receipt.cjson
```

冷存储仍保留原 `tactile/...` 相对结构。首版不额外套一层视频 codec 或 learned archive；raw
先使用 `dmtac_raw_temporal_delta_zstd_v1` 重打包，再迁移目录。这样 reader 简单，恢复后
原逻辑路径不变，也避免重复传输约一半空间。

如冷存储只用于整批恢复，可以在存储系统层另做归档；canonical 内容仍以逐 stream
manifest、payload hash 和 inventory 为准，不能只信 tar/zip 进程退出码。

## 5. 最小合同

首版只需要一个通用的 detachable-modality envelope 和一个 transfer receipt，不扩大成通用
对象存储协议，也不为 camera/tactile 各复制一套近似实体。触觉约束由 inventory 的
`format_profile` 与 `modality_key=tactile` 校验器提供。

### 5.1 `ModalityArtifactInventoryV1`

不可变内容：

```text
dataset base fingerprint
dataset name / task/profile hashes
episodes.jsonl hash
modality key = tactile
logical prefixes = [tactile/]
episode/sides/sensor keys
每个 tactile manifest 的 relative path、size、sha256
每个 payload 的 relative path、size、sha256
总文件数、总字节数
format profile = dmtac_raw_temporal_delta_zstd_v1
inventory content hash
```

它只描述“哪些字节属于该逻辑数据集”，不保存 `/mnt/...` 之类本地路径。

### 5.2 `ModalityStorageTransferReceiptV1`

记录一次复制/验证/退役操作：

```text
inventory hash
source/target storage identity
source/target device id
文件数和总字节数
hash verification result
independent inode/backup verification
copied_at / verified_at
hot source retirement status
tool version
```

物理 locator 使用非 canonical workspace binding：

```text
dataset identity + inventory hash -> /cold/.../<dataset_identity>/tactile
```

未来如接对象存储，可扩展 locator resolver；不改变 inventory 的内容身份。

### 5.3 可用性不是 capture validity

不新增可写的第二套“模态有效性真源”。以下状态由 capture contract、inventory、workspace
binding 和验证结果派生：

```text
ABSENT_AT_CAPTURE       capture contract 明确 ignored
ATTACHED_SELF_CONTAINED tactile/ 在 dataset root 且 inventory 校验通过
ATTACHED_EXTERNAL       显式 cold-root binding 与 inventory 校验通过
DETACHED                inventory 存在，但当前没有可用 binding
INCOMPLETE_OR_CORRUPT   引用、文件数、size 或 hash 不一致
```

`DETACHED` 不能改写为 ignored/invalid；`INCOMPLETE_OR_CORRUPT` 不能被 core-only 模式静默
解释成正常 detached。

## 6. 无触觉训练

默认推荐通过 prepared training view 训练，而不是伪造一个删字段后的第二份 v2.3 数据集：

```text
modality_selection = numeric,video
column_projection = 当前模型所需数值列
```

约束：

1. view manifest 明确 tactile 不在 selected modalities；
2. source inventory 不包含任何 tactile manifest/payload；
3. loader 不实例化 `ChunkedTactileStream`，不 stat/open tactile 路径；
4. 模型 artifact 明确其输入不包含 tactile；
5. 训练启动只验证 prepared view inventory，不先运行完整 v2.3 self-contained validator；
6. 同一训练 run 绑定固定 view hash，不能运行中切换 tactile availability；
7. 原 episodes/Parquet 的 tactile locator 字段允许存在，但未被投影/读取。

这样 cold tactile 完全离线时，numeric/video 训练仍可复现；但源根目录应显示为
`core-online / tactile-offline`，不能显示“完整数据集校验通过”。

## 7. 需要触觉训练时

提供两个模式。

### A. Rehydrate（首版推荐）

1. 读取 cold inventory 和 export receipt；
2. 在 hot 数据集同一文件系统的独立临时目录复制 tactile；
3. 验证文件数、每文件 size/hash、manifest 中 frame count/chunk 索引；
4. 使用标准 reader 复读每路首帧、chunk 边界、中间帧和末帧；
5. 临时目录 fsync 后原子 rename 为 `dataset_root/tactile`；
6. 运行完整 v2.3 validator；
7. 只有全部通过才允许 tactile-enabled training。

不得直接把文件逐个复制进正在被 reader 使用的 `tactile/`。

### B. External root binding（后续优化）

由 `dataset_interface` 接受显式：

```text
logical prefix tactile/ -> verified cold tactile root
```

reader 先校验 inventory hash，再直接从冷盘读取，不必复制回来。旧 v2.3 reader 保持原行为；
这是显式 adapter，不修改 frozen v2.3 locator 语义。

如果冷盘吞吐不足，训练仍应先 rehydrate 到本地 cache，而不是让 dataloader 随机阻塞网络盘。

## 8. 安全迁移流程

任何真实迁移前必须满足项目的数据变换规则。

### Export（默认保留 hot source）

1. 只选择 sealed、无 open writer、无 maintenance lease 的数据集；
2. 完成 raw 无损重打包和 round-trip 校验；
3. 解析所有 episode tactile refs，拒绝缺失、额外未归属或路径越界文件；
4. 记录 source/target `st_dev`、inode、link count、容量；
5. 写入 cold 私有 staging，不能直接 `mv` 跨文件系统；
6. 对所有 payload 计算/复验 hash，并用标准 reader 抽检；
7. 原子发布 cold inventory 和 export receipt；
8. 此阶段 hot `tactile/` 保留，先完成训练/view 验证。

### Retire hot tactile（独立显式 gate）

只有以下条件全部满足才允许：

```text
cold copy 是独立 inode/独立存储
inventory/hash 全部一致
cold restore 演练成功
tactile-free prepared view 训练 fixture 通过
没有 reader/writer/maintenance lease
用户显式启用 retire-hot-tactile
```

退役后记录实际释放块数。删除行为不可由“export 成功”隐式触发。

## 9. 包边界

### `dataset_contracts`

只拥有纯 inventory/receipt identity、hash domain 和验证合同；不复制、不删除、不解析设备。

### `dataset_interface`

拥有：

- tactile artifact inventory 构造与验证；
- prepared tactile-free/tactile-enabled view 选择；
- logical prefix 到显式 artifact root 的只读解析；
- core-only 与 full-dataset 两种验证入口；
- rehydrate 后的标准 reader 校验。

路径解析使用显式 `ModalityArtifactResolver`：先解析 dataset-root 内原相对路径；只有调用者
提供、且已验证 inventory hash 的 external binding 时，才把同一逻辑路径映射到 cold root。
旧 v2.3 reader 默认行为保持不变。禁止在通用 `Path` helper 中偷偷加入全局 fallback。

### `collection_runtime`

不负责长期存储策略。将来只可通过 typed maintenance port 暴露 sealed/idle/lease 状态；不拥有
冷存储路径、复制器或删除策略。

### `teleop_runtime` / `control_station`

只展示状态、发出明确 operator intent，不执行数据复制、不改写 identity、不成为 writer。

### `utils/dataset_postprocess`

承载显式 CLI/application assembly：dry-run、export、verify、rehydrate、retire。具体合同和
reader 逻辑仍来自 `dataset_contracts` / `dataset_interface`。

## 10. 分阶段落地

### P0：只读 audit

- 输出 tactile refs、总字节、压缩状态、缺失/额外文件；
- 生成但不发布 inventory proposal；
- 检查 source/target device、inode/link count 和空间；
- 不写、不复制、不删除。

### P1：raw 无损重打包

- 落地 `temporal delta + zstd level 1 + chunk 8`；
- 保留 logical frame/dtype/shape；
- reader 和 v2.3 validator 全量通过。

### P2：cold export shadow

- 复制到 cold staging；
- 发布 inventory/receipt；
- hot tactile 保留；
- 完成一次从 cold 到临时新目录的 restore 演练。

### P3：tactile-free training gate

- 增加“源 tactile 完全不在线”fixture；
- prepared numeric/video view 构建、校验、loader batch 均通过；
- 验证训练进程没有 open/stat tactile payload；
- policy artifact 明确 tactile-free input contract。

### P4：受控 hot retirement

- 默认关闭；
- 一次只处理一个 sealed dataset；
- 需要独立 receipt、exclusive lease 和显式确认；
- core-only audit 通过，full audit 明确报告 tactile offline，而不是 corrupted。

### P5：cold direct-read/cache（可选）

- 外部 root resolver；
- hash-verified cache；
- cache miss/断连 fail closed；
- 不改变 v2.3 原路径或数据内容。

## 11. 验收用例

- `tactile/` 离线时，完整 v2.3 validator 明确失败或报告 OFFLINE，不静默通过；
- 同一状态下 numeric/video prepared view 构建和训练 batch 成功，且未打开 tactile；
- tactile-enabled view 在无 verified binding 时 fail closed；
- 错误 dataset identity、inventory hash、episode catalog hash、文件数、尾 chunk 任一不符时拒绝；
- restore 中断不会产生半在线 `tactile/`；
- cold storage 断连不会影响已经启动的 tactile-free training；
- rehydrate 后完整 v2.3 validator、首/中/末帧读取和时间索引完全一致；
- 删除 prepared/derived view 后，hot core + cold inventory 仍可重建；
- 不同任务/profile 的 tactile root 不能混绑；
- hardlink 不被误报为独立备份；
- export 成功但未授权 retire 时 hot source 必须仍存在。

## 12. 当前推荐

可以推进，但顺序应为：

```text
raw 无损压缩
→ cold shadow copy（保留 hot）
→ tactile-free 训练 fixture
→ cold restore 演练
→ 最后才考虑 hot tactile 退役
```

首版训练不需要触觉时，使用显式 numeric/video prepared view；不要复制出一个删掉 tactile
引用的伪完整数据集。需要触觉时优先完整 rehydrate，等 direct cold-root adapter 稳定后再省去
复制步骤。

## 13. 2026-08-02 落地状态与下一 gate

已落地的无损 profile：

```text
dmtac_raw_temporal_delta_zstd_v1
logical dtype/shape/frame index 不变
predictor = delta_mod_uint8（每个 chunk 独立）
zstd level = 1
chunk frames = 8
```

本地封存数据集
`datasets/2026-07-30_dmtac_rm65_bimanual_position_v1`：

```text
8 streams / 1732 frames
532,070,400 raw bytes -> 251,752,584 compressed bytes
ratio = 0.4731565...
```

所有 stream 均在删除 raw 前完成逐 chunk round-trip 和全流 SHA-256；提交后又通过正式
`ChunkedTactileStream` 全帧重读及 v2.3 validator。事务收据保存在
`meta/tactile_recompression/`。

远端只读审计与 apply 结果：

- 初始有 7 个结构完整的封存 raw 数据集，888 路、80,482,713,600 bytes；
- 首轮完成 4 个数据集；第五个数据集完成 30/44 manifests 后检测到新的正式遥操进程，运行期
  gate 在下一 manifest commit 前丢弃 staging 并停止；停止遥操后按原 manifest 幂等续跑，已
  提交的 60 路没有重做；
- 最终 7 个结构完整数据集的 raw stream 全部完成，最终只读计划为 0 raw datasets / 0 raw
  streams；整体从 80,482,713,600 bytes 压缩为 38,231,403,191 bytes，ratio=0.475026，
  逻辑字节减少 42,251,310,409；最后一批 71,617,536,000 raw bytes 压缩后为
  34,034,253,327 bytes；
- 对最后批次关联的全部 856 路（含中断前已提交的 60 路）、250,188 帧做正式 reader 全帧
  SHA-256 复验，36,514,669,542 compressed bytes 全部通过；
- 三个最后处理的数据集中两个通过完整 v2.3 validator；另一个只有一项与触觉无关的既存
  缺陷：episode 6 缺失声明的 front MP4，308 路触觉仍全部通过逐帧 hash；
- 另有两个数据集存在 raw 文件大小大于 manifest 声明的问题，其中一个正在写入，另一个需
  单独修复，不能压缩 prefix 后删除整个源文件。

批处理使用精确计划 hash，只有 `size == manifest expected`、link count 为 1、无 v2.4
commit 的 sealed 数据集才能进入；计划发生任何变化都拒绝 apply。两个异常数据集仍原样
保留。压缩器在 operation failure 时写 `failed_partial` 收据；这只补足审计，不把 manifest
间的可恢复分批改成全数据集原子事务。

下一 gate 是 P2 cold export shadow：先实现 inventory/receipt 与恢复演练，hot tactile 保留；
在 tactile-free training fixture 和独立 cold copy 均验证前，不允许执行 hot retirement。
