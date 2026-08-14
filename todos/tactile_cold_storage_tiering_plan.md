# 触觉冷存储分层与无触觉训练视图方案

状态：P1 raw 无损重打包已在本地与远端全部结构完整的封存数据集完成并全量校验；M1-M5 与
whole-prefix W1-W3 已落地，离线事务、故障注入、prepared-view 和 actual-v2.3 fixture 已通过；
尚未对任何真实数据执行 detach，也尚未在独立冷盘执行 W4 canary。
纯合同、只读 inventory、派生可用性和 core-only v2.3 validation 已通过 mixed
raw/zstd、缺失/部分缺失/额外文件、路径越界、symlink/hardlink、catalog/payload 篡改以及
manifest-hot/payload-missing fixture。严格 full v2.3 validator 保持原行为；显式 numeric/video
prepared view 在已验证 detached 状态下可 build/load 且不把 tactile payload 纳入 source
inventory，tactile read capability 明确不可用。后处理 CLI 已实现 dry-run proposal、独立设备
shadow export、逐组 restore drill、journal/quarantine 原子退役与 rehydrate；RealTeleop recorder
已接入 capture-writable admission。尚未生成真实数据 inventory sidecar，也未执行真实冷存储
export、热端退役、rehydrate 或 external-root 绑定。

## 1. 结论

方案可行，但不能把 `tactile/` 直接搬走后仍把剩余目录解释成一个物理自包含、通过完整
v2.3 校验的数据集。

兼容模式的正确语义是：

```text
一个逻辑数据集
  = hot/core artifacts（meta、data、videos、progress 等）
  + hot tactile manifests（很小，保留逻辑时间轴和 payload 图）
  + cold tactile payloads（占主要空间）
  + 二者之间不可变的 identity/hash binding
```

现在另提供显式、默认关闭的 `whole-prefix` 模式：同一 inventory 覆盖 manifest 与 payload，
完整 `tactile/` 在 durable receipt 生效后可以从 hot root 原子退役。两种模式共用同一合同和事务；
默认仍是 payload-only，不改变既有数据或旧命令。

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
  tactile/
    .../manifest.json        # 首版保留
    # 大型 .bin/.zst payload 可以不驻留

/cold/dexteleop_tactile/<dataset_identity>/
  tactile/
    observation.hand.left.tactile/...
    observation.hand.right.tactile/...
  tactile_inventory.cjson
  export_receipt.cjson
```

冷存储保留完整原 `tactile/...` 相对结构，包括 manifest 的校验副本。首版 hot 端只退役
manifest 引用的大型 payload，不删除 manifest。这样 Parquet 的 tactile path/index 仍可和
manifest frame count 做结构校验，恢复时也不需要重写 `episodes.jsonl`、Parquet 或任何
identity。raw 先使用 `dmtac_raw_temporal_delta_zstd_v1` 重打包，再迁移 payload；恢复后原
逻辑路径和字节内容均不变。

把整个 `tactile/` 目录完全移走留作后续能力。它要求 manifest 本身也通过 external resolver
解析，适配面更大，但几乎不比“仅移走 payload”多释放空间，因此不进入首版。

如冷存储只用于整批恢复，可以在存储系统层另做归档；canonical 内容仍以逐 stream
manifest、payload hash 和 inventory 为准，不能只信 tar/zip 进程退出码。

## 5. 最小合同

首版只需要一个通用的 detachable-modality envelope 和一个 transfer receipt，不扩大成通用
对象存储协议，也不为 camera/tactile 各复制一套近似实体。触觉约束由 inventory 的
`format_profile` 与 `modality_key=tactile` 校验器提供。

### 5.1 `ModalityArtifactInventoryV1`

不可变内容：

```text
source binding profile = modality_reference_projection_v1
tactile reference projection hash（episode_index + tactile_paths）
modality key = tactile
logical prefixes = [tactile/]
episode/sides/sensor keys
每个 tactile manifest 的 relative path、size、sha256、frame count
每个 payload 的 relative path、role、size、sha256
总文件数、总字节数
format profile summary（允许 mixed；不能把某一种 codec 作为可拆卸前提）
inventory content hash
```

兼容字段名沿用早期 `dataset fingerprint / episode catalog hash`，但由显式 profile 决定含义；
新 inventory 不再绑定 dataset name、task/profile 或其他模态字段，也不保存一套重复身份。

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
SELF_CONTAINED          manifest/payload 均在 dataset root 且校验通过
EXTERNAL_BOUND          hot manifest + 显式 cold-root payload binding 校验通过
DETACHED_VERIFIED       hot manifest 与 inventory 存在，payload 当前不在线
STALE_INVENTORY         Episode catalog 已在 inventory 生成后改变
INCOMPLETE_OR_CORRUPT   无 inventory 却缺文件，或引用、size、hash 不一致
```

`DETACHED_VERIFIED` 不能改写为 ignored/invalid；`INCOMPLETE_OR_CORRUPT` 不能被 core-only 模式静默
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

### A. Rehydrate（首版唯一正式 tactile-enabled 路径）

1. 读取 cold inventory 和 export receipt；
2. 在 hot 数据集同一文件系统的独立临时目录构建完整 tactile tree；
3. 验证文件数、每文件 size/hash、manifest 中 frame count/chunk 索引；
4. 使用标准 reader 复读每路首帧、chunk 边界、中间帧和末帧；
5. 临时目录 fsync 后把 manifest-only `tactile/` 与完整新树做原子目录交换；
6. 运行完整 v2.3 validator；
7. 只有全部通过才允许 tactile-enabled training。

不得直接把文件逐个复制进正在被 reader 使用的 `tactile/`。

### B. External root binding（后续优化，不阻塞首版）

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

首版 rehydrate 模式不要求任何 reader 使用该 resolver；因此 `ChunkedTactileStream`、v2.3
Parquet reader 和第三方转换器都不需要修改。只有未来 direct cold-read 才在应用边界注入
resolver。

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
- 完成一次从 cold 到临时新目录的 restore 演练；
- hot payload 仍保留，hot manifest 必须保留。

### P3：tactile-free training gate

- 增加“hot manifest 在线、tactile payload 完全不在线”fixture；
- prepared numeric/video view 构建、校验、loader batch 均通过；
- 验证训练进程没有 open/stat tactile payload；
- policy artifact 明确 tactile-free input contract。

### P4：受控 hot payload retirement

- 默认关闭；
- 一次只处理一个 sealed dataset；
- 需要独立 receipt、exclusive lease 和显式确认；
- 仅删除 inventory 精确列出的 payload，保留 manifest；
- core-only audit 通过，full audit 明确报告 tactile offline，而不是 corrupted；
- 成功 retirement receipt 存在时，collection preflight 禁止继续向同一 dataset root 追加；
- rehydrate 后才允许该 root 再次进入 self-contained 状态。

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

## 14. 2026-08-03 兼容优先的最小落地设计

### 14.1 冻结目标

首版只解决两个稳定用例：

```text
DETACHED_VERIFIED
  numeric/video/progress 训练可用
  tactile payload 不占 hot storage

SELF_CONTAINED
  从 cold 完整 rehydrate 后
  所有旧 reader、validator、转换器继续原样工作
```

首版不要求第三方 loader 直接读取 cold root，也不支持同一 dataset root 在 payload detached 时
继续追加 Episode。这两个限制显著缩小适配面，而且都可以在以后通过 additive 能力扩展。

### 14.2 为什么已有 v2.3 数据不需要迁移

以下内容保持逐字节不变：

- `meta/info.json`、`meta/tasks.jsonl`、`meta/modality_validity.json`；
- `meta/episodes.jsonl` 中的 `tactile_paths`；
- Parquet 的 `observation.hand.<side>.tactile_path/frame_index`；
- tactile manifest 的 logical path、frame count、stream path、dtype、shape 和 codec；
- `EpisodeCaptureIdentity`、source episode record hash 和 task/profile binding。

新的 inventory/receipt 作为 additive sidecar 放在新的 `meta/modality_artifacts/` 命名空间。旧数据
没有这些 sidecar 时继续按 `SELF_CONTAINED_LEGACY` 处理；缺文件却没有 verified inventory 时
必须报告损坏，不能猜测成 detached。

新写入 inventory 使用显式 `modality_reference_projection_v1` 绑定：第一个兼容字段保存触觉投影
合同声明 hash，第二个兼容字段只保存 `episode_index + tactile_paths` 的 canonical hash，计数也只统计
存在触觉引用的 Episode。manifest/payload 的路径、size 和 SHA-256 仍由 artifact records 独立绑定。
因此 task 名、视频/Parquet 路径、非触觉 Episode 字段或其他 dataset metadata 的调整不会使触觉
导出失效；任何 `tactile_paths`、tactile manifest 或 tactile payload 变化仍然 fail closed。旧 inventory
保持原 semantic hash 和 reader 兼容；当旧的整数据集身份因无关字段变化而失配时，只允许在当前
触觉引用投影与旧 inventory manifest graph 完全一致时兼容通过。

### 14.3 最少新增的合同

只新增两个持久化合同，不新增可写的 availability 状态真源。

#### `ModalityArtifactInventoryV1`

建议文件：`dataset_contracts/modality_artifacts.py`。

核心字段：

```text
schema / inventory_id / semantic_content_hash
modality_key = tactile
source_dataset_base_fingerprint
source_episode_catalog_sha256
source_episode_count
logical_prefixes = [tactile/]
artifact records:
  logical_path
  role = MANIFEST | PAYLOAD
  size_bytes
  sha256
  episode_index / side / sensor_key（可选索引，不作为路径解析真源）
format summaries（信息性，不限制 codec）
file_count / total_bytes
```

不保存绝对路径，不重复保存完整 tactile manifest，不创建第二套帧时间轴。manifest 的
`frame_count` 只可作为 hash-bound audit summary，正式含义仍来自原 manifest。

#### `ModalityStorageTransferReceiptV1`

同一个 tagged operation 合同覆盖：

```text
EXPORT_SHADOW
RESTORE_DRILL
RETIRE_HOT_PAYLOADS
REHYDRATE
VERIFY_EXTERNAL
```

核心字段：inventory hash、operation id、source/target storage id、文件/字节计数、验证结果、
独立 inode/device evidence、前后状态和时间。不要为每个操作分别创建五种近似 receipt。

绝对 mount path 属于非 canonical workspace binding；receipt 只保存稳定的 storage id。运行时：

```text
storage id + inventory hash -> current local root
```

### 14.4 `dataset_interface` 的最小实现

建议集中在一个包，避免逻辑散落：

```text
dataset_interface/modality_artifacts/
  inventory.py     构建和验证 artifact graph
  availability.py  派生 SELF_CONTAINED/DETACHED/... 状态
  resolver.py      后续 direct cold-read 使用
```

首批正式 API：

```text
build_modality_artifact_inventory(dataset_root, modality="tactile")
verify_modality_artifact_inventory(dataset_root, inventory, artifact_root=None)
resolve_modality_availability(dataset_root, inventory, binding=None)
validate_v23_core_dataset(dataset_root, verified_inventories=...)
```

保留 `validate_v23_dataset(dataset_root)` 的现有严格语义和签名：它仍要求 self-contained，避免
旧调用者静默接受离线模态。`validate_v23_core_dataset` 是新入口，只在 inventory 完整绑定时
允许 tactile payload 离线；它仍检查：

- Episode record 引用的每个 tactile manifest 都存在于 hot；
- Parquet tactile path 与 Episode/manifest 关系一致；
- inventory 精确覆盖所有 manifest 引用的 payload；
- 不存在未归属或重复 logical path；
- capture modality contract 仍表示 tactile 当时有效。

prepared training view 当前 `ALLOWED_MODALITIES={numeric, video}`，inventory 也只加入
episodes catalog、Parquet、所选视频和语义 sidecar；它不读取 tactile。首批不必修改 prepared
view schema，只需新增“源 tactile payload 完全离线”的 fixture，防止未来有人在 view 构建前
无条件调用 full v2.3 validator。

### 14.5 reader 和训练兼容矩阵

| 使用方式 | Self-contained | Detached，无 binding | Detached，有 external binding |
|---|---:|---:|---:|
| 旧 v2.3/full validator | 通过 | 明确失败 | 仍失败 |
| 新 core-only validator | 通过 | `DETACHED_VERIFIED` 通过 | 通过 |
| numeric/video prepared view | 通过 | 通过 | 通过 |
| 旧 tactile reader/第三方 v2.1 转换 | 通过 | 失败 | 失败 |
| tactile-enabled training（首版） | 通过 | 先 rehydrate | 先 rehydrate |
| 新 direct cold reader（后续） | 通过 | 无 binding 时失败 | 通过 |

这保证旧行为没有被放宽：需要触觉的旧工具只在完整 rehydrate 后运行。tactile-free 训练通过
显式 modality selection 工作，而不是捕获 `FileNotFoundError` 后偷偷降级。

### 14.6 后处理命令与默认安全语义

在 `utils/dataset_postprocess/modality_storage.py` 实现，CLI 仍挂到现有
`python -m utils.dataset_postprocess`，不增加生产 entrypoint：

```text
modality-inventory     默认只读，生成 proposal
modality-export        copy -> hash verify -> publish receipt，保留 hot
modality-restore-drill cold -> 临时新树 -> reader/validator verify -> 删除临时树
modality-retire-hot    默认禁用，仅删除 inventory 中 role=PAYLOAD 的文件
modality-rehydrate     cold -> staging complete tree -> 原子目录交换
modality-status        只读派生状态
```

所有 apply 命令必须：

1. 拒绝活跃 writer、reader maintenance lease 和未封存 Episode；
2. 再次核对 source/target `st_dev`、inode、link count 和容量；
3. 使用冻结 plan hash，目标集合变化即拒绝；
4. copy 到新 inode，完整 hash 校验，不信任退出码；
5. retirement 前必须存在 successful export + restore drill receipt；
6. 目录交换前 fsync staging 和父目录；
7. crash 后能从 staging/receipt 判断继续或回滚，不能出现“部分 payload 被视为正常 detached”。

#### 14.6.1 “拆出”实际拆什么

首版拆出的对象不是整个 `tactile/` 目录，而是冻结 inventory 中 `role=PAYLOAD` 的大文件。
`role=MANIFEST` 的小文件继续保留在原相对路径：

```text
hot dataset root
  meta/                         保留
  data/                         保留
  videos/                       保留
  tactile/**/manifest.json      保留
  tactile/**/<large payload>    拆出

cold storage root
  <dataset identity>/<inventory revision>/tactile/**/manifest.json
  <dataset identity>/<inventory revision>/tactile/**/<large payload>
```

cold 副本保存完整 inventory 图（manifest + payload），而 hot retirement 只处理 payload。这样 cold
介质本身具备完整恢复材料，hot 侧仍能解释 Episode 的触觉时间轴以及哪些文件被有意拆出。dataset
identity 和 inventory revision 用 hash 绑定；上面的目录层次是 workspace locator，不应冻结进数据合同。

#### 14.6.2 便捷拆出：一个高层命令，两阶段确认

对操作员提供一个 `modality-detach` 高层工作流，但它只编排同一套 inventory/export/drill/retire
原语，不能复制第二套文件操作实现：

```text
python -m utils.dataset_postprocess modality-detach \
  <hot_dataset> \
  --cold-root <cold_storage> \
  --modality tactile
```

第一次运行永远是只读 proposal，输出：

- tactile binding profile、触觉引用投影 hash 和含触觉引用的 Episode 数；
- payload 文件数、总字节数、预计 hot 释放空间；
- hot/cold `st_dev`、容量、同 inode/hardlink 风险；
- inventory hash 和一次性 plan hash；
- 当前 reader/writer/maintenance lease；
- 将保留的 manifest 列表和将退役的 payload 集合摘要。

真正执行必须显式带回 proposal 的 plan hash 和 dataset fingerprint：

```text
python -m utils.dataset_postprocess modality-detach \
  <hot_dataset> \
  --cold-root <cold_storage> \
  --modality tactile \
  --apply \
  --plan-hash <proposal_hash> \
  --confirm-dataset-fingerprint <fingerprint> \
  --confirm-no-active-readers
```

内部顺序固定为：

```text
冻结 inventory
→ copy 完整 tactile 图到 cold 的新 inode
→ 逐文件 size + SHA-256 验证
→ 验证 cold 与 hot 不是同 inode/hardlink
→ 从 cold 按 manifest/Episode 分组恢复到临时目录并逐组运行 hash + standard reader drill
→ 发布 export + restore-drill receipts
→ 再次确认 catalog/lease/plan 未变化
→ 仅退役 hot payload
→ 验证 hot manifest 和缺失集合恰好等于 inventory
→ 发布 retire receipt
```

任一步在 retirement 前失败，hot dataset 完全不变。retirement 必须采用可恢复 quarantine/transaction，
不能逐个 `unlink` 后寄希望于命令不中断；只有全部 payload 已从 active tree 一致移出且目录 fsync 完成，
才发布成功 receipt。没有成功 receipt 的半完成状态一律判为 `INCOMPLETE_OR_CORRUPT`，不能作为正常
detached 数据集使用。

#### 14.6.3 便捷放回：先构造完整新树，再原子切换

放回命令同样先 proposal、后 apply：

```text
python -m utils.dataset_postprocess modality-rehydrate \
  <hot_dataset> \
  --cold-root <cold_storage> \
  --inventory-hash <inventory_hash>

python -m utils.dataset_postprocess modality-rehydrate \
  <hot_dataset> \
  --cold-root <cold_storage> \
  --inventory-hash <inventory_hash> \
  --apply \
  --plan-hash <proposal_hash> \
  --confirm-dataset-fingerprint <fingerprint> \
  --confirm-no-active-readers
```

不能把 payload 一个一个直接复制进正在使用的 active `tactile/`。正确顺序为：

```text
验证 cold inventory/catalog 身份
→ 在 hot 同一文件系统创建 sibling staging tree
→ 复制 manifest + payload 到 staging 的新 inode
→ 校验全部 size/hash、manifest 引用、帧数和尾部可读性
→ 对 staging 运行完整 v2.3 validator/标准 tactile reader
→ fsync staging 和父目录
→ active manifest-only tree 与 staging 原子目录交换
→ 再次运行完整 validator
→ 发布 rehydrate receipt
→ 成功后清理旧 manifest-only quarantine
```

rehydrate 中断发生在目录交换前时，原 detached hot tree 保持可用；发生在交换附近时，通过 staging、
quarantine 和 receipt 恢复到“完整新树”或“原 detached 树”之一，不能留下部分 payload active。
cold 副本默认继续保留，不因 rehydrate 自动删除。

#### 14.6.4 操作后的可见行为

| 状态 | 完整 v2.3 校验 | core-only 校验 | 纯视觉 prepared view | 触觉读取/训练 | 继续向同一 root 数采 |
|---|---|---|---|---|---|
| `SELF_CONTAINED_LEGACY` / `SELF_CONTAINED` | 通过 | 通过 | 通过 | 通过 | 允许 |
| shadow export 完成、尚未 retire | 通过 | 通过 | 通过 | 仍读 hot | 允许，但 catalog 一旦变化旧 inventory 作废 |
| `DETACHED_VERIFIED` | 明确失败（缺 payload） | 通过 | 通过，且不需要挂载 cold | 明确要求 rehydrate | 首版拒绝 |
| `EXTERNAL_BOUND`（后续 direct reader） | 仍失败 | 通过 | 通过 | 仅新 reader 可直接读 cold | 首版拒绝 |
| `STALE_INVENTORY` | 取决于 hot 是否完整 | 拒绝该 inventory | 启动包装层应拒绝 | 拒绝 binding | 禁止 detach/rehydrate，先重新审计 |
| `INCOMPLETE_OR_CORRUPT` | 失败 | 失败 | 新训练 admission 拒绝 | 失败 | 拒绝 |
| rehydrate 成功后的 `SELF_CONTAINED` | 通过 | 通过 | 通过 | 所有旧 reader 恢复 | 允许 |

这里的“纯视觉 prepared view 通过”必须配套显式 `numeric/video` modality selection 和 core-only
admission。prepared-view reader 本身不访问 tactile payload，但正式训练启动包装层不能绕开 availability
检查，把任意缺文件数据集都误当成正常 detached。

M3/M4 目前已经完成代码与离线 fixture，但还没有通过“独立真实冷盘 + 小型 sealed canary”的操作
验收。在 canary 完成前，不应手工移动真实数据，也不应把 M2 的结构性 `DETACHED_VERIFIED` 当成
production retirement receipt。CLI production apply 额外要求 cold 与 hot 的 `st_dev` 不同，并要求
`--confirm-no-active-readers`；这是因为当前尚无覆盖 OpenPI/第三方脚本的统一 reader lease。

### 14.7 collection 与应用边界

- `collection_runtime` 不拥有 inventory、cold path、复制或删除；无需为首版修改写入热路径。
- `dataset_interface` 提供只读 `assert_dataset_capture_writable()`：若最新成功 receipt 表示 hot
  payload 已 retirement，则拒绝向该 root 追加 Episode。
- `teleop_runtime` 的 dataset 选择/preflight 和未来 UMI adapter 调用该检查；只传 admission
  结果给 collection owner，不复制存储策略代码。
- `control_station` 后续只展示派生状态和发 typed maintenance intent，不直接执行文件操作。

如果已 rehydrate，成功 receipt 把状态恢复为 `SELF_CONTAINED`；之后可以再次采集，但下一次
export 必须生成绑定新 `episodes.jsonl` hash 的新 inventory revision。首版不支持“旧 Episode
在 cold、新 Episode 在 hot”的混合直接读取。

### 14.8 实施顺序

#### M1：纯合同与只读 inventory

- 落地两个合同及 hash round-trip；
- 对现有 raw/zstd/lz4/mixed tactile fixture 构图；
- 缺失、额外、路径越界、hardlink 和 catalog 改变全部 fail closed。

#### M2：core-only validation 与 detached fixture

- 保持 full validator 不变；
- 增加 manifest-hot/payload-missing fixture；
- numeric/video prepared view build/load/batch 在 payload 离线时通过；
- tactile reader 仍明确失败。

#### M3：shadow export 与 restore drill

- 已实现只 copy、不在 drill 前删除 hot；
- 已实现 cold 独立 device/inode、全 hash 和标准 reader 验证；
- 已实现按 manifest/Episode 分组恢复，避免临时驻留第二份完整 tactile；
- actual-v2.3 fixture 在 retirement 前通过 full validator，完整 rehydrate 后再次通过。

#### M4：受控 retirement/rehydrate

- journal/quarantine 原子 retirement 与 rehydrate 已在 fixture 落地；
- retirement 后 core-only 通过、full 明确失败；
- rehydrate 后 full validator 和 tactile artifact hash 重新通过；
- RealTeleop capture append 已对 detached/pending/corrupt root fail closed；
- 下一门槛仍是独立冷盘上的小型 sealed canary，通过后才可扩展到大数据集。

#### M5：direct external binding（可选）

- 只有 rehydrate 的复制时间成为实际瓶颈时再实现；
- resolver 必须显式注入，不修改全局 Path 行为；
- 可以先只适配 `teleop_runtime.replay.ExternalTactileStore`，不要求所有第三方 loader 同时升级。

### 14.9 必须覆盖的 corner cases

- 旧数据无 inventory 且完整：行为与现在完全一致；
- 旧数据无 inventory 但缺 payload：判为 corrupt，不得判 detached；
- inventory 存在但 `episodes.jsonl` 后续追加：`STALE_INVENTORY`；
- 同名 dataset/profile 不同或任务 hash 不同：禁止绑定；
- cold root 来自错误 inventory：禁止 attach/rehydrate；
- hot/cold 文件同 inode 或仅 hardlink：不算独立备份；
- export 中断、receipt 未发布：hot 保留且不可 retirement；
- retirement 中断：staging/quarantine 可判定，不能出现未知缺口；
- rehydrate 尾 chunk 损坏：目录交换前失败，manifest-only hot tree 不变；
- numeric/video 训练期间 cold disk 拔除：训练不受影响且不 stat tactile；
- tactile-enabled 训练启动时 detached：启动前失败，不在 batch 中途报错；
- dataset detached 后尝试继续正式采集：preflight 拒绝并要求新 root 或先 rehydrate；
- rehydrate 后再次采集：旧 inventory 标为历史 revision，新 export 绑定新 catalog hash。

### 14.10 结论

可以兼容已有数据并保持较小改动，前提是首版采用：

```text
保留 tactile manifest
只退役大型 payload
不改 frozen v2.3 引用
无触觉训练走 prepared view
触觉训练先完整 rehydrate
full validator 默认行为不变
```

这样实际必须触碰的生产面只有纯合同、`dataset_interface` 的 inventory/core validation、现有
postprocess CLI，以及 collection 启动前的一处只读 admission check。采集 writer、遥操控制、
Parquet schema、episode identity、已有 tactile codec 和训练数据行均不需要重构。

## 15. 2026-08-03 远端 OpenPI 兼容性复核

对远端 `/home/user/zc/openpi` 做了只读源码审计，未启动训练、JAX/GPU 或设备。当前
`DexTeleopV23MaskedDataset` 的实际数据路径为：

```text
meta/info.json
meta/episodes.jsonl
record.parquet_path
action / observation.state / pose-quality / actual TCP
+ 可选 RGB video_frame_index 与 video_paths
```

其 `required_columns` 不含 tactile path、tactile frame index 或 tactile payload；OpenPI 的
`strict_media_integrity` 只调用 DexTeleop 的 RGB media validator，并显式传入 camera keys。该 validator
只读取 `video_paths`，不遍历 `tactile/`。当前 OpenPI 模型元数据也声明 `tactile_kind=none`。

因此，现有 OpenPI DexTeleop v2.2/v2.3 **不含触觉训练路径**在 hot manifest 保留、payload detached
后不会因为 loader 本身而受影响。这个结论有两个明确边界：

1. 训练启动包装层不能在 OpenPI loader 之前无条件调用严格的 `validate_v23_dataset()`；应调用
   core-only admission 或仅 RGB media admission；
2. 将来增加 tactile-enabled OpenPI profile 时必须先 rehydrate，或显式接入 verified external
   resolver，不能继承当前 `tactile_kind=none` 的兼容结论。

首版需要增加一个无触觉 OpenPI characterization fixture：构造 manifest-hot/payload-missing 数据集，
完成 dataset init、index build 和至少一个 batch，且通过受控 open/stat spy 证明没有访问 tactile
payload。该 fixture 通过后，才把 OpenPI 列入 hot retirement gate。

### 15.1 已有数据的 additive 迁移

已有 sealed 数据不需要转换 Parquet、重写 Episode 或重新生成视频。单个数据集的迁移固定为：

```text
只读 audit
→ 生成并冻结 inventory proposal
→ shadow export 到独立 cold inode
→ 全 hash + standard reader 验证
→ restore drill
→ 发布 inventory/transfer receipts
→ 无触觉 OpenPI/prepared-view fixture
→ 单独显式 retire hot payload
```

迁移工具必须支持多个数据集复用同一流程，但 export 与 retirement 永远不能合成一个默认命令。
已有数据集完整且未启用 sidecar 时继续走 legacy self-contained 分支；只有 successful inventory、
export receipt 和 restore-drill receipt 三者共同绑定相同 catalog hash 时，才允许进入
`DETACHED_VERIFIED`。

## 16. 不解算的近实时触觉无损压缩边界

### 16.1 明确结论

“压缩不在遥操主线程执行”和“压缩完全不影响遥操”不是同一件事。

- 同一台真机主机上，只要压缩在遥操期间运行，就会共享 CPU、内存带宽、page cache、磁盘队列和
  调度器；即使放入独立进程、绑定独立 CPU，也不能承诺物理意义上的零影响。
- 本方案现在明确采用第二条边界：压缩**只在遥操未运行且正式数采完全空闲时存在**；开始遥操前
  必须完成停止、checkpoint、落盘和进程退出屏障。遥操和数采期间机器上不存在压缩 worker。
- 在这个约束下，可以把压缩对遥操/数采的运行期资源影响消除；需要额外验证的是“从压缩到遥操”
  的切换是否真的无残留进程、未完成写盘和 dirty-output 发布，而不是比较两者并发时的性能。

因此不再需要为首版引入外部压缩主机，也不需要让压缩与遥操并发。2 秒 runtime Home 准备窗口可以
用于收尾，但它只是正常预算，不是放宽 admission：未收到完整 quiesce receipt 时不得进入 Home 或
RUNNING，也不得开始正式数采。

### 16.2 当前实现事实

现有 CLI 已支持：

```text
--tactile-compression none | lz4 | zstd
```

但该路径在 `LeRobotV21Writer` 收到每个 tactile job 后，将 32 帧拼成 chunk，并在 writer/collection
执行边界内同步调用 LZ4 level 0 或 Zstd level 1 后写盘。位控正式数采虽然把整个 collection worker
固定到 CPU 8-11、主进程固定到 CPU 0-7、DMTAC reader 固定到 CPU 12-13，但同步压缩仍会：

- 延长 collection worker 的单样本处理时间；
- 增加共享内存 slot 和有界 sample queue 的占用；
- 在压缩追不上时最终暂停正式采集；
- 与相机视频、Parquet 和系统 page cache 竞争内存/磁盘资源。

所以现有参数可用于性能 A/B，但不应直接作为“不影响遥操”的正式默认档。

远端已完成的数据验证给出了体积参考：561,266 个 raw tactile stream frame 共
172,420,915,200 bytes，经可逆 temporal-delta + Zstd level 1 压缩为 81,822,838,275 bytes，
比例 47.46%，并通过逐帧 SHA-256 与完整 v2.3 校验。该结果证明无损重打包有效，但不证明在遥操
期间运行没有资源干扰。

### 16.3 推荐首版：空闲期、上一 Episode 的隔离后台压缩

首版不改每帧采集格式，不把压缩插入 DMTAC reader、遥操循环或实时 sample admission：

```text
DMTAC raw frame
  -> 现有正式采集/写 raw（权威热路径不变）
  -> Episode finalize + durable receipt
  -> 后台进程领取 closed Episode lease
  -> 写新压缩 inode
  -> 标准 reader 逐帧 hash 校验
  -> 原子发布新 manifest/receipt
  -> 最后才退役 raw inode
```

这属于 idle-time、one-episode-lag，不是逐帧 streaming compression。其优点是压缩失败、进程被杀、
遥操重新开始或机器重启时，权威 raw 始终可读，且关闭参数即可完全回到当前行为。

首版压缩只消费 raw bytes 和 shape/dtype/frame index，不运行分布式力、深度、接触、去噪或任何触觉
语义解算。codec 可以有两个显式档：

- `raw_zstd_v1`：严格对 raw chunk 做 Zstd，语义和实现最简单；
- `raw_temporal_delta_zstd_v1`：先做完全可逆的字节预测再 Zstd，仍不是触觉解算，体积效果已验证；
  若“只压缩”被严格解释为禁止 predictor，则不启用此档。

### 16.4 空闲 admission、停止屏障和回退规则

新能力必须默认关闭，并使用独立模式，而不是改变现有 `--tactile-compression none` 的含义：

```text
tactile background compression mode = off | closed_episode_isolated
```

`off` 是完整回退路径。`closed_episode_isolated` 至少满足：

1. 独立 spawn 进程，不进入遥操主进程、DMTAC reader 或 collection worker 的线程池；
2. 任务队列只传 Episode/stream identity 和路径，不复制 frame ndarray 回主进程；
3. 只处理 sealed Episode；active Episode 和未封存 manifest 永远不可领取；
4. 在目标 codec 全量验证前不删除 raw；替换必须遵守临时新 inode、fsync、原子发布和可恢复 receipt；
5. nice/CPU affinity/idle I/O priority 仍作为空闲期的系统友好措施，但不把它们误当成允许与遥操并发
   的理由，也不提前冻结一套依赖特定 CPU 数量的正式 profile；
6. backlog 超限、内存压力或磁盘异常时暂停压缩并保留 raw，不把压缩失败升级成遥操或采集故障。

#### 16.4.1 唯一合法的压缩运行窗口

应用层根据现有权威状态生成一个只读 maintenance admission，不能让 compressor 自己猜 GUI 文案：

```text
startup_and_devices_ready
and (teleop_started == false or hold_active == true)
and episode_recording_active == false
and episode_countdown_active == false
and episode_finalizing_active == false
and dataset_recorder.is_recording == false
and dataset_recorder.finalizing == false
and shutdown_requested == false
```

其中 `is_recording=true, is_paused=true` 的失败 Hold/普通 Hold **不是空闲数采状态**：Episode 仍打开，
压缩不得启动。只有停止 Episode 且 finalize 完成后，Hold/启动等待阶段才允许压缩。

| 场景 | 允许压缩 |
| --- | --- |
| 相机/设备仍在启动 | 否 |
| GUI 已就绪、尚未开始遥操、无 Episode | 是 |
| 遥操 RUNNING、未开始数采 | 否 |
| 遥操 Hold、无 active/finalizing Episode | 是 |
| Hold，但 Episode 只是 paused | 否 |
| Episode 已停止但仍在 finalize | 否 |
| 退出流程 | 否 |

#### 16.4.2 遥操启动的强制 quiesce 屏障

首次开始遥操和 Hold 后重新 Home 都走同一个屏障。最早的“开始遥操/runtime Home” intent 被接受后：

```text
IDLE_COMPRESSING
  -> 禁止领取新任务
  -> cooperative cancel 当前小 chunk
  -> 保存 resumable checkpoint
  -> fdatasync 临时输出与 checkpoint
  -> 关闭全部输入/输出 fd
  -> worker process join
  -> 必要时 syncfs 等待该文件系统既有 dirty write 完成
  -> 发布 QUIESCED receipt
  -> 才允许继续 2 秒准备、Home 和 RUNNING
```

压缩必须按小 chunk 工作并在每个 chunk 间检查 cancel，不能把整条数十 GB stream 作为一次不可中断
的 codec 调用。临时输出在 quiesce 时不成为正式 manifest 的一部分；下次空闲从已验证 checkpoint
继续，或安全丢弃临时输出后重做，raw 始终保留。

如果 cooperative cancel 没有在准备预算内完成，可以升级终止 worker；强制终止后不能假定 close
等于落盘完成，必须在压缩目标文件系统执行并等待明确的 `syncfs`/等价 I/O barrier。只有确认进程
退出、文件描述符关闭且落盘屏障完成后才能继续。超过预算仍未获得 `QUIESCED` 时，应取消本次开始
请求并保持 Hold，提示压缩收尾异常；不能为了“及时开始”让遥操与压缩短暂重叠。

正式数采开始也保留同一条断言：必须已有有效的 compressor `QUIESCED` receipt。正常流程中数采只会
在遥操 RUNNING 后开始，因此它通常是冗余安全检查，但可以防止未来入口绕过遥操 admission。

若只有一个很长 Episode，首版不会提前释放其 raw 空间。将 Episode 内部改成可封存 segment 会扩大
格式、manifest 和 crash-recovery 适配面，待首版真实指标证明必要后再设计。

### 16.5 包边界

- `dataset_contracts`：只定义 codec/transform identity、artifact hash 和完成 receipt；
- `collection_runtime`：拥有 device/UI-neutral 的 closed-artifact 后台作业生命周期、lease、暂停和
  crash recovery；不读取遥操按键、不判断机器人状态；
- `dataset_interface`：拥有标准 reader、逐帧内容等价验证和 manifest 投影；
- `teleop_runtime`：只把 Episode durable 通知和运行健康/admission 信号接到通用端口，默认不启用；
- `control_station`：未来只显示 backlog/paused/verified，并发 typed intent；不执行压缩算法。

当前触觉 writer 仍在兼容路径时，可以用薄 adapter 接入，不在 `teleop_runtime` 复制第二套 compressor。

### 16.6 验收门槛

先在记录数据和 mock collection 上测试，再进行不运动的真机切换 canary。重点不再是让两者并发，
而是证明遥操 admission 后压缩已经完全消失：

- 在最大 chunk 和最慢磁盘条件下请求开始，quiesce 必须可中断且可恢复；
- `QUIESCED` 后 compressor PID/thread 不存在、无打开的 tactile 输入/临时输出 fd；
- receipt 必须晚于最后一次 checkpoint/fdatasync/close/join，不能先发 receipt 后异步收尾；
- 2 秒预算内成功时正常进入 Home；超时时保持 Hold，不能进入 RUNNING；
- Home/RUNNING/正式数采期间无 compressor CPU、磁盘写入或任务领取；
- 普通 Hold + paused Episode、finalizing Episode、设备启动期均不能误启动压缩；
- 遥操 main-loop、DMTAC、相机、CANFD 和 writer 指标继续作为“不存在意外重叠”的监测证据；
- 压缩积压、暂停次数、raw 保留量和逐帧 hash 等价性。

正式启用门槛不是“遥操看起来不卡”，而是状态机和系统观测都证明压缩与遥操/数采没有时间重叠，
且所有中断点都能保留 raw、恢复 checkpoint 或安全重做。若 quiesce 机制不可靠，则保持默认关闭；
不得通过允许短暂重叠、放宽遥操告警或跳过落盘/内容校验来上线。

### 16.7 已落地的可回退首版（2026-08-04）

首版已按上述边界以 `--idle-tactile-compression` 接通，默认关闭；省略参数或显式使用
`--no-idle-tactile-compression` 即完整回退。实现边界如下：

- `collection_runtime.idle_maintenance` 只拥有通用 spawn worker 生命周期、协作取消、join、必要的
  terminate/kill 升级和 `syncfs` quiesce receipt，不 import 真机、GUI、hardware 或 modality；
- `teleop_runtime.real_teleop.tactile_idle_compression` 负责把权威 RealTeleop/recorder 状态投影成准入，
  并调用既有已验证 tactile recompressor；
- 压缩只读取 `meta/episodes.jsonl` 引用的 finalized manifest，使用 8 帧小块
  `delta_mod_uint8 + zstd level 1`，逐块 round-trip 后再做完整 decoded SHA 校验；manifest 原子发布
  并验证后才删除 raw；
- Home intent 在已有 2 秒准备期开始时先发协作停止，准备期结束后必须 join + 文件系统屏障完成；
  快速双踏板 Home、自动 Home、正式 Episode 开始、dataset recorder 重绑定和 shutdown 也有独立的
  fail-closed quiesce 检查；
- 当前正在处理的未提交 manifest 被取消时不发布压缩结果，删除 staging output 并从 raw 安全重做；
  已经提交的 manifest 是恢复粒度。首版没有额外冻结 per-stream checkpoint 合同，避免引入一套暂时
  不必要的 checkpoint schema；
- `is_recording=true`（包括 paused failure Hold）、countdown 或 finalizing 均拒绝启动。控制循环只以
  5 Hz 轮询轻量 controller 状态，RUNNING 前 worker 已不存在。

离线验收已覆盖：默认关闭、状态准入、协作取消、进程 join 后才发 receipt、文件系统屏障、相同
catalog revision 不重复工作、新 revision 可继续、取消时 raw/manifest 字节不变、CLI 冲突拒绝和
Home 准备期的 begin→delay→require 顺序。真机首次启用仍应先做不运动 canary，确认远端 zstd backend、
磁盘余量和日志中的 `quiesced ... forced=False`，再用于正式采集。

## 17. 整个 `tactile/` 可拆装的模块化布局（2026-08-05 规划）

实现状态：W1-W3 已完成；事务 owner 已迁入 `collection_runtime.modality_storage`，历史
`utils.dataset_postprocess.modality_storage` 仅保留薄兼容导出。W4 真实独立冷盘 canary 仍待显式
数据路径和授权。

### 17.1 结论

可以把数据集整理成“整个 `tactile/` 可作为一个模块迁走和迁回”，而且不需要重排现有大文件、
改写 Parquet 或修改触觉 reader 的相对路径。当前 writer 已经满足最重要的物理边界：

```text
dataset_root/
  data/                         # 只保存 tactile 相对引用和 frame index
  videos/
  meta/
    episodes.jsonl             # tactile_paths 指向 tactile/.../manifest.json
    modality_artifacts/        # inventory、receipt、transaction journal
  tactile/                     # 所有 tactile manifest 和 payload 都在这个前缀内
```

因此不新增 `modalities/tactile/`，也不改已有 `tactile/...` 路径。首版只扩展“可拆卸范围”：从当前
的 payload-only 退役，增加 whole-prefix bundle 退役。这样 attached 状态仍是标准、自包含 v2.3；
把目录放回原位后，旧 reader、OpenPI 转换和完整 validator 无需适配。

这里的“直接迁走/迁回”是指工具以整个目录为事务单元，不表示允许操作员对真实数据直接执行
`rm -rf`、跨盘 `mv` 或拖拽文件夹。跨文件系统的 `mv` 本质上仍是不可恢复的 copy+delete；生产操作
继续要求 proposal、独立副本、hash、恢复演练、receipt 和目录原子切换。

### 17.2 不新增冗余 module manifest

现有 `ModalityArtifactInventoryV1` 已经包含：

- `logical_prefixes=["tactile"]`；
- 显式 source binding profile 与触觉引用投影 hash（旧字段名仅为兼容保留）；
- 每个 manifest/payload 的相对路径、size、SHA-256；
- Episode、side、sensor key 和 frame count。

它已经足以定义完整 tactile module。首版不再在 `tactile/` 内增加第二个 module manifest，也不把
每个 Episode manifest 复制到 `meta/`。所需的新增语义只是一种新的完成操作事实，例如：

```text
retire_hot_module_tree
```

该 receipt 表示 inventory 中 `logical_prefixes` 覆盖的完整目录已经在 cold 端验证，并从 hot active
tree 一致退役。旧的 `retire_hot_payloads` receipt 继续表示 manifest-resident 的兼容状态；二者不得
被静默互相解释。这样不修改已生成 receipt 的含义，也不需要增加一套重复身份合同。

### 17.3 whole-prefix 状态判定

availability 必须同时使用 inventory 和完成 receipt，不能只根据目录是否存在猜测：

| hot 状态 | receipt | 结果 |
| --- | --- | --- |
| inventory 全部文件存在且 hash 一致 | 任意历史 receipt | `SELF_CONTAINED` |
| 整个 `tactile/` 不存在 | `retire_hot_module_tree` complete | `DETACHED_VERIFIED` |
| manifest 在、全部 payload 不在 | `retire_hot_payloads` complete | 兼容 `DETACHED_VERIFIED` |
| 手工移走整个目录但没有 receipt | 无 | `INCOMPLETE_OR_CORRUPT` |
| 目录存在但缺一部分文件 | 任意 | `INCOMPLETE_OR_CORRUPT` |
| catalog 的触觉引用投影已变化 | 任意 | `STALE_INVENTORY` |
| 仅其他模态引用或非触觉 metadata 变化 | 任意 | 不改变触觉 inventory 可用性 |
| 目录外出现 inventory 未拥有的 tactile 文件 | 任意 | `INCOMPLETE_OR_CORRUPT` |

whole-prefix detached 时，core-only validator 不再要求 hot manifest；它改为验证：

1. inventory hash 与 schema；
2. source binding profile、触觉引用投影 hash/count；
3. whole-tree detach receipt 和无 pending transaction；
4. inventory 的所有逻辑路径都位于唯一的 `tactile/` 前缀；
5. hot `tactile/` 整体缺失，而不是部分缺失。

严格 `validate_v23_dataset` 保持不变：目录不在时明确失败。纯视觉 prepared view 继续不把 tactile
manifest/payload 放入 source inventory，也不实例化触觉 reader。需要触觉训练时必须先 rehydrate。

### 17.4 新采集数据不需要改变热路径

RealTeleop/UMI writer 当前已经把以下内容全部写入 `tactile/`：

- 左右侧 per-Episode manifest；
- raw/zstd payload；
- manifest 引用的未来兼容 tactile artifact。

Parquet 和 `episodes.jsonl` 只保留相对引用。因此采集热路径不增加逐帧 hash、bundle catalog 或第二套
manifest；这些工作只在 sealed/idle detach proposal 中完成。需要补充的是结构不变量和测试：

```text
所有 tactile_paths 和 tactile storage refs 必须以 tactile/ 开头
manifest 内所有 stream path 必须以 tactile/ 开头
tactile/ 下不能存在 catalog/manifest 未拥有的文件
非 tactile 顶层目录不能保存 tactile payload
```

新采集数据天然 whole-prefix eligible，但只有无 active Episode、无 finalization、无 writer/reader、
触觉引用与触觉 artifact graph 冻结后才能执行 detach；其他模态可以独立维护。detached root 继续
禁止追加会写触觉的 Episode；放回并通过触觉 graph 校验后才恢复 capture admission。完整 v2.3
validator 仍可作为独立的数据集质量检查，但不再是触觉 export/detach/rehydrate 的准入条件。这避免
为了“可搬目录”修改正式数采频率或 Episode lifecycle，也避免无关模态维护阻塞已验证的触觉迁移。

### 17.5 既有数据迁移

#### 已 self-contained 的 v2.3 数据集

无需重写数据：

```text
完整 v2.3 validation
→ 构建现有 inventory
→ cold 复制完整 tactile/ 到新 inode
→ hash + standard reader + whole-directory restore drill
→ 发布 whole-tree receipt
→ hot tactile/ 目录原子退役
```

`data/`、`meta/episodes.jsonl`、所有 Parquet、视频和 tactile 内容字节均不改变。严格来说这不是格式
迁移，而是为原布局发布一份可验证的模块存储状态。

#### 已处于 payload-only detached 的数据集

不能直接升级。必须：

```text
先按旧 inventory rehydrate 到 SELF_CONTAINED
→ 完整 v2.3 validation
→ 再执行 whole-prefix detach
```

这样不会把“cold 端只有部分材料”误认为完整模块。

#### 手工移走过 `tactile/` 的数据集

没有 inventory/receipt 的手工状态不能自动认定为安全 detached。应先把原目录完整放回、通过 full
validation，再生成 whole-prefix inventory/receipt。不得根据同名目录或文件数量补造成功证明。

### 17.6 整目录拆出与放回事务

拆出仍采用跨盘安全顺序：

```text
dry-run proposal
→ copy dataset_root/tactile/ 到 cold staging 的新 inode
→ 校验 exact file set + size + SHA-256 + 无 symlink/hardlink
→ standard reader 首尾/分组恢复演练
→ fsync cold tree 和父目录
→ 发布 export + restore-drill receipts
→ 再次冻结 catalog、inventory 和 reader/writer 状态
→ os.replace(dataset_root/tactile, sibling quarantine)
→ 验证 active tactile/ 整体不存在且 core-only 合法
→ 发布 retire_hot_module_tree receipt
→ 删除 quarantine
```

放回使用相反方向，但不直接从 cold 跨盘 rename：

```text
验证 cold inventory
→ copy 完整 tactile/ 到 hot sibling staging
→ exact file set/hash/reader/full-v2.3 staging validation
→ fsync
→ os.replace(staging/tactile, dataset_root/tactile)
→ full v2.3 validation
→ 发布 rehydrate receipt
→ cold 副本继续保留
```

当 whole-prefix detached 时 active root 没有 `tactile/`，所以 rehydrate 只需把完整 staging 原子发布；
若意外已有同名目录则 fail closed，不能覆盖或合并。

### 17.7 包边界

- `dataset_contracts`：只扩展 transfer operation vocabulary；inventory V1 内容身份保持不变；
- `dataset_interface`：根据 inventory + receipt 解析 whole-prefix availability，提供 strict/core-only
  validation 和 prepared-view admission；
- `collection_runtime`：只拥有 UI/device-neutral 的离线目录事务、lock/journal/quarantine；
- `teleop_runtime` / UMI：保持现有 tactile 写入路径，只调用 capture admission；
- `control_station`：以后可显示 attached/detached/recover-needed，不执行文件复制和删除。

现有 `utils.dataset_postprocess` 继续作为 CLI 入口；具体事务已经由
`collection_runtime.modality_storage` 拥有，CLI 只负责编排 proposal 和显式确认，历史模块路径是薄
兼容导出。payload-only 工具保留为可回退模式，两种 scope 共用 export、hash、restore drill、journal、
quarantine 和 recovery 原语。

### 17.8 分阶段落地与回退

1. **W1 合同/解析**：新增 whole-tree receipt operation，扩展 availability/core-only fixture；不改 CLI
   默认值，不动真实数据。
2. **W2 事务实现**：让现有 detach transaction 支持 `payload-only | whole-prefix`，共用 export、hash、
   drill、journal 和 recovery；默认仍为 `payload-only`。
3. **W3 writer 不变量**：为 RealTeleop 与 UMI fixture 增加“所有 tactile artifact 均位于单一前缀”
   测试；不在实时写入中增加 hash。
4. **W4 canary**：小型 sealed 数据集完成 whole-prefix detach→纯视觉 view→rehydrate→full validation
   字节一致性；再评估把新 CLI 默认切到 `whole-prefix`。

回退路径始终存在：

- 不启用 whole-prefix 时，现有 payload-only 行为不变；
- attached 数据集的物理布局与 reader 完全不变；
- whole-prefix detached 数据集随时可 rehydrate 回标准 self-contained v2.3；
- 不修改冻结的旧 receipt，不对手工缺失目录做宽松推断。

### 17.9 必须覆盖的 corner cases

- hot/cold 同一 inode、同一 `st_dev`、cold 位于 dataset 内部；
- 整个目录缺失但无 receipt、只有 export receipt、只有部分 payload；
- manifest/payload 路径越界、symlink directory、hardlink、extra file；
- proposal 后新增 Episode、修改 catalog、idle compressor 修改 manifest；
- cold copy 完成但 retire 前崩溃；
- active 目录 rename 后 receipt 发布失败；
- receipt 已 durable 但 quarantine 清理失败；
- rehydrate staging 完整但 publish 前崩溃；
- hot 已出现同名 `tactile/` 时拒绝放回；
- detached 数据集误启动 RealTeleop/UMI append；
- whole-prefix detached 的纯视觉 batch 可读，但 tactile reader 明确不可用；
- rehydrate 后 strict validator、随机 tactile frame 和全部原始字节 hash 恢复一致。

## 18. M5：采集期增量完整性与自动 inventory

M5 是默认关闭、可完整回退的优化；不改变 `tactile/` 物理布局、Parquet 引用、帧时间轴或
`ModalityArtifactInventoryV1` 合同。RealTeleop 使用
`--tactile-artifact-integrity`，UMI 使用同名参数或配置字段；关闭时 manifest 与既有 writer 行为
保持不变。

### 18.1 写入与封口

启用后，每个 external tactile stream 在 collection writer 中对**实际交给文件句柄的字节**增量
计算 SHA-256：raw 对 raw bytes，chunked zstd/lz4 对压缩后的 stored bytes。hash 不在机器人控制
循环中运行，也不扫描历史目录。Episode 封口顺序为：

```text
flush final tactile chunk
→ flush + fsync payload handle
→ seal stored_size_bytes / stored_sha256 / stored_hash_profile
→ atomic write + fsync tactile manifest
→ append Episode catalog/stats and update info
→ compose and atomically publish immutable modality inventory
```

inventory 发布失败发生在 Episode 已 durable commit 之后，因此不能让调用方重试 Episode finalize；
writer 保存明确错误并告警，后续 detach 回退到离线全量扫描。每个 catalog snapshot 发布新的 immutable
inventory hash，旧 inventory 不被改写，并会随 Episode catalog 更新自然变为 stale。

### 18.2 采集 admission

共享 `LeRobotV21Writer`、RealTeleop recorder 与 UMI recorder 都在创建数据目录或绑定采集事实之前
执行 admission：

| 数据集状态 | 是否允许 append | 原因 |
|---|---:|---|
| 新/空目录 | 是 | 没有已有模态身份 |
| 无触觉引用的完整数据集 | 是 | 不会制造冷热混合触觉树 |
| `SELF_CONTAINED` | 是 | 当前 hot graph 完整 |
| `DETACHED_VERIFIED`（payload-only/whole-prefix） | 否 | append 会使同一逻辑前缀冷热混装 |
| 已有未封存 tactile 历史 + 本次开启采集期 hash | 否 | 必须新建 root 或关闭参数并最终离线扫描 |
| 手工缺失、部分缺失、extra/symlink/hardlink | 否 | 不能证明是合法 detach |
| active storage transaction | 否 | 必须先离线 recovery |
| 当前 inventory 与物理字节不一致 | 否 | 必须先完成/修复受 receipt 约束的变换 |

当前不支持“拆出旧 Episode 的 tactile 后继续向同一 dataset append 新 Episode”。若未来确有需要，必须
另建 generation/segment 合同，不能复用当前 whole-prefix identity 猜测合并。

### 18.3 数据处理、转换、训练、部署与迁移

- lossless idle recompression 必须为新 stored representation 写入新的 stored hash；如数据集已启用
  inventory tracking，成功提交后发布新 inventory。旧 inventory 仍保留但不再被当作当前状态；
- tactile compact 和 UMI temporal alignment/裁剪会改写 payload，必须重新计算 exact stored-byte hash；
  若原数据集已跟踪 inventory，则变换在提交/切换 staged root 前发布新的 immutable revision；
- 会改变 `tactile_paths`、tactile manifest 或 tactile payload 的工具不得继承“current”身份；仅其他
  模态字段变化不影响触觉 inventory，触觉投影或触觉 artifact 变化则必须发布新 inventory 或让
  admission fail closed；
- 完整 v2.3 validation 在 detached 时继续失败；不能为冷存储放宽；
- numeric/video prepared training view 在 `DETACHED_VERIFIED` 时可读，且 source inventory 不包含
  tactile；tactile-enabled view 必须先 rehydrate 或显式绑定已验证 external root；
- 格式转换若声明包含 tactile，必须要求 `tactile_read_eligible=true`；纯视觉转换只能走明确的
  core-only/modality-selection 路径，不能静默丢弃触觉引用；
- 训练和部署不得根据目录是否存在猜测模态。策略需要 tactile 时 fail closed；不需要 tactile 时不打开
  manifest/payload；
- rsync/数据迁移可以单独迁移 hot core，但 cold locator 不进入 portable identity。装回必须提供匹配
  inventory hash 的 cold root；不能按目录名或文件数量配对。

### 18.4 detach/rehydrate smoke gate

发布前必须通过：capture seal→inventory byte-equivalence→whole-prefix detach→catalog/numeric/video
读取与字节不变→shared writer append 拒绝→rehydrate→exact tactile tree/hash/reader→full validation。
detach apply 仍保留独立 cold inode、copy-time hash、restore drill 和 destructive-boundary full scan；采集期
hash 用于自动生成身份和减少重复准备工作，不替代最终安全校验。
