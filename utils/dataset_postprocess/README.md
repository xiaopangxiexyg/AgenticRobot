# Dataset Postprocess

> 本页维护各后处理子命令的完整参数。当前跨工具运行顺序统一从
> [`docs/data/README.md`](../../docs/data/README.md) 开始。

DexTeleop dataset 后处理工具集合。默认以 dry-run / 写新目录为主，不原地修改原始数据集。
唯一受支持的包路径是 `utils.dataset_postprocess`。原顶层
`dataset_postprocess` 包已移除；脚本、测试和外部调用应统一使用新路径。

## 常用命令

使用带 `pyarrow` 的环境：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
```

### 旧数据腕部相机物理左右归一化

`camera-semantics-materialize` 根据采集时保存的 RealSense 序列号和当前相机角色注册表，逐
episode 判断存储键是否颠倒。dry-run 会列出每个 episode 的
`canonical_to_stored_key`、左右 video index 不一致行数、源 inode/hardlink 数量和预计复制字节数：

```bash
python -m utils.dataset_postprocess camera-semantics-materialize \
  DATASET_ROOT \
  --output-root OUTPUT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json
```

确认计划后，把上一步输出的 `plan_hash` 原样回填：

```bash
python -m utils.dataset_postprocess camera-semantics-materialize \
  DATASET_ROOT \
  --output-root OUTPUT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json \
  --apply \
  --expected-plan-hash PLAN_HASH
```

该工具不原地修改源数据，也不只交换 RGB 图像键。它同时迁移完整的
`observation.images.left_wrist.*` / `right_wrist.*` Parquet 列族、RGB 视频、wrist depth/mask
artifact、episode 路径表和设备快照。单侧历史缺帧会保留在对应的物理相机上；不会借用另一侧
`video_frame_index` 补齐。输出文件均为 fresh inode；每段视频必须通过 Parquet 声明帧数、
ffprobe 解码帧数和 ffmpeg `-xerror` 全段/尾帧解码后才原子发布。dry-run 在给出可 apply 的
plan hash 前也会完整验证源视频，因此截断的 hardlink 血缘会在创建 staging 前被拒绝。应先根据
dry-run 的 `source_byte_count` 预留完整输出空间。输出中的 materialization receipt 会使现有
`DatasetCameraSemanticResolver` 和普通按 canonical key 读取的 loader 都看到 identity，避免
organized provenance 再次把已经换名的数据交换一次。

2026-07-30 已确认的 wrist RGB hardlink write-through 事故数据不能靠左右物化恢复；其中被
截断的视频通常只剩 1--2 帧，而 Parquet 仍声明完整 episode 帧数。相关数据必须先从独立副本
恢复或重新采集，禁止绕过上述源视频门禁。事故范围、根因和恢复边界见
[`deployments/openpi/DATASET_MEDIA_INCIDENT_20260730.md`](../../deployments/openpi/DATASET_MEDIA_INCIDENT_20260730.md)。

相机物化不会读取或复制冷端触觉 payload。含 `tactile_paths` 的数据集只有在 hot 端状态与已
验证 retirement receipt 一致时才允许 `--apply`：`payload-only` 必须只剩完整、独立 inode 的
manifest，`whole-prefix` 必须不存在整个 `tactile/`。dry-run 的 `tactile_detach_audit` 会明确
给出 scope、manifest/payload 数量及 `cold_payloads_read=false`；缺 receipt、仍有 payload、
manifest hash/大小变化、未知热端文件或不同 artifact graph 都会 fail closed。输出继续保留同一
`tactile_paths`、inventory 和 receipt，且发布前重新得到完全相同的 detach audit。因此应先完成并
验证触觉迁出，再物化相机语义；不要把 NAS/cold root 传给相机工具。

完整物化输出在通过严格 v2.3 core 校验和全视频解码后，可用
`camera-semantics-promote` 原子接替原 canonical 路径。晋升只接受同文件系统上相互独立、无
共享 inode 的两个数据集：原路径必须被证明为 `swapped`，新路径必须被证明为 `identity`，
两侧触觉 inventory 必须完全相同。允许的触觉状态组合只有：热→热、冷→热，以及同 inventory、
同 detachment scope 且双方都有生产级 retirement evidence 的冷→冷。热→冷会被严格拒绝，
避免晋升操作无意中移除仍在热端的触觉。批量维护推荐使用冷→冷：相机物化继承 inventory 与
retirement receipts，晋升和回滚都不读取、复制或恢复 NAS/cold payload。原数据不会删除，而是移动到显式
retirement 目录；事务文件记录两侧目录 inode、核心文件 guard、物化 receipt、计划 hash 和
回滚位置：

默认晋升门禁仍要求原生 DexTeleop v2.3。对于明确标注旧 `format` 的历史数据，可在 dry-run
和 apply 中都显式增加 `--allow-legacy-format-preserving`。该路径会逐项验证 `info.json`、
`episodes.jsonl`、全部 Parquet 列和完整视频解码仅发生声明的左右腕相机语义变换；源和纠正
根的 `format` 必须完全一致。它不会写入 v2.3 声明，也不会把旧格式伪装为 schema 升级。

```bash
python -m utils.dataset_postprocess camera-semantics-promote \
  CANONICAL_SWAPPED_ROOT \
  --corrected-root CORRECTED_FULL_ROOT \
  --retirement-root RETIREMENT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json

python -m utils.dataset_postprocess camera-semantics-promote \
  CANONICAL_SWAPPED_ROOT \
  --corrected-root CORRECTED_FULL_ROOT \
  --retirement-root RETIREMENT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json \
  --apply \
  --expected-plan-hash PLAN_HASH \
  --confirm-no-active-readers
```

晋升后应分别复核 canonical 的严格 v2.3、相机 `identity` 和触觉 reader eligibility。若晋升时
旧源已拆离，回滚前必须先把当前 canonical 按相同 inventory 严格拆离；回滚计划会拒绝仍含热端
触觉的 canonical、任何 pending transaction、inventory 变化和非触觉核心文件漂移。随后只使用
事务文件换回旧目录，不手工改名，并在换回后按冷端 inventory 恢复旧 canonical：

```bash
python -m utils.dataset_postprocess camera-semantics-rollback-promotion \
  RETIREMENT_ROOT/.transactions/PROMOTION_PLAN_HASH.json
```

回滚 apply 同样要求冻结的计划 hash 和 `--confirm-no-active-readers`。晋升和回滚都拒绝已有
目标、符号链接、目录 inode 漂移、活跃触觉事务及跨文件系统操作；它们不删除 cold copy、旧
数据或纠正数据。

如果输出只用于数值 + RGB/depth/mask 训练，不需要成为可独立恢复全部模态的完整 v2.3 数据集，
可显式使用默认关闭的 `camera-semantics-prepare-view`。它只选择 episode Parquet、相机 artifact
和训练所需的小型 metadata，不遍历 `tactile/`，也不读取、校验或复制触觉 inventory/payload：

```bash
python -m utils.dataset_postprocess camera-semantics-prepare-view \
  DATASET_ROOT \
  --output-root OUTPUT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json \
  --dry-run

python -m utils.dataset_postprocess camera-semantics-prepare-view \
  DATASET_ROOT \
  --output-root OUTPUT_ROOT \
  --registry robot_motion/configs/rm65_camera_registry.json \
  --apply \
  --expected-plan-hash PLAN_HASH
```

prepared view 会把物理视频、对应相机自身的整组 `observation.images.<role>.*` Parquet 列以及
camera artifact 同时归一化，因此允许历史左右腕视频索引不同；不会拿另一侧索引替换缺帧。
输出使用 fresh inode 并完整解码验证视频，`meta/prepared_views/camera_semantics.json` 明确声明
`full_v23_dataset=false`、`excluded_modalities=["tactile"]` 和源选择清单 hash。它不是完整数据集，
不能用于触觉训练、归档恢复或通用 v2.3 校验。OpenPI corrected wrapper 默认拒绝该投影；只有
调用者同时选择该输出根目录并显式传入 `--allow-camera-prepared-view` 才能训练。原
`camera-semantics-materialize` 的完整克隆和触觉 retirement 门禁保持不变。

prepared view 也不会复制 `.lock` 等运行时文件；manifest 和严格 loader 都会拒绝这类文件。
已经发布但不符合上述合同的试点目录不得继续放在默认可选目录中。先运行
`camera-semantics-quarantine-invalid-view VIEW --audit-root AUDIT_ROOT` 生成冻结计划，随后携带
完全一致的 `--expected-plan-hash --apply`，只会在同一文件系统内原子移动到审计目录。该命令
要求严格 loader 明确返回 contract error；I/O 错误、有效视图、符号链接、跨设备和已存在目标
均 fail closed，并且不会删除或覆盖目录。远端操作使用固定的
`codex_remote/remote_dataset_camera_quarantine.sh`。

若 prepared view 的完整视频解码门禁排除了损坏 episode，应把同一份 hash-bound exclusion
receipt 原子投影回源数据集的 `meta/modality_validity.json`，而不是手工抄 episode 列表：

```bash
python -m utils.dataset_postprocess \
  camera-video-integrity-mark-training-invalid DATASET_ROOT \
  --prepared-view-root CAMERA_PREPARED_VIEW_ROOT

python -m utils.dataset_postprocess \
  camera-video-integrity-mark-training-invalid DATASET_ROOT \
  --prepared-view-root CAMERA_PREPARED_VIEW_ROOT \
  --apply --expected-plan-hash PLAN_HASH
```

该操作只改一个 metadata 文件，保留既有 capture、gripper 等字段；每个坏 episode 会被标成
`training.episodes.<index>.valid=false`，并嵌入完整 exclusion receipt。首次 apply 前必须证明源
camera/numeric inventory 与 prepared view 生成时完全相同；重复运行是幂等的。普通 OpenPI
corrected 入口会 fail closed 拒绝仍包含无效 episode 的完整源根目录，prepared-training view
构建器会按所选 camera 过滤这些 episode。这样损坏视频不能因绕过 camera-only prepared view
再次进入训练/测试。

新正式采集的 thread writer 默认采用 `--storage-queue-backpressure-policy block`：相机存储队列
满时在控制热路径之外背压 sample assembly，不再由每个相机独立丢最旧帧。旧的
`drop_oldest` 仍是显式兼容选项，但可能重新产生单侧 frame gap；正式数据不应使用。process
writer 继续按完整 sample 原子接纳/拒绝，不依赖这一 thread-only 队列。

### 混合触觉数据的一体化入口（推荐）

`tactile-maintain` 是面向既有数据集的状态感知入口。默认只生成 JSON 计划，不写数据；它会从
`episodes.jsonl`、每个 tactile manifest、当前 inventory 和 storage receipt 判断数据集属于：

- 全 raw；
- raw 与 zstd/lz4 混合；
- 已全部压缩；
- `payload-only` 或 `whole-prefix` 已验证迁出；
- 非法部分缺失、未知编码或未完成事务。

目标为 `compressed-hot` 时，只无损压缩仍为 raw 的 stream，已经是 zstd/lz4 的 stream 保持原样。
重复执行会得到 `target_satisfied=true` 的空操作：

```bash
python -m utils.dataset_postprocess \
  tactile-maintain DATASET_ROOT \
  --target compressed-hot
```

确认计划后显式应用：

```bash
python -m utils.dataset_postprocess \
  tactile-maintain DATASET_ROOT \
  --target compressed-hot \
  --apply
```

目标为 `detached` 时，默认先把剩余 raw stream 无损压缩，再调用可续传、可恢复的 whole-prefix
managed move。迁出/放回涉及 reader 边界，因此 apply 必须确认没有活跃 reader：

```bash
python -m utils.dataset_postprocess \
  tactile-maintain DATASET_ROOT \
  --target detached \
  --cold-root COLD_ROOT

python -m utils.dataset_postprocess \
  tactile-maintain DATASET_ROOT \
  --target detached \
  --cold-root COLD_ROOT \
  --apply \
  --confirm-no-active-readers
```

将已迁出的数据恢复为 hot、自包含状态：

```bash
python -m utils.dataset_postprocess \
  tactile-maintain DATASET_ROOT \
  --target hot \
  --cold-root COLD_ROOT \
  --apply \
  --confirm-no-active-readers
```

如果迁出时保留了 raw，再以 `--target compressed-hot --cold-root COLD_ROOT` 应用，工具会先完成
verified rehydrate，再只压缩剩余 raw。每一步继续使用原有 recompression/storage receipt；组合入口
没有第二套进度文件。进程在“压缩后、迁出前”或“装回后、压缩前”中断时，重复同一目标命令会根据
实际 manifest/receipt 从剩余步骤继续。

`--no-compress-before-detach` 是只迁出、不先压缩的显式回退开关。已处于目标状态时不要求 cold root；
detached 数据需要装回或继续压缩时必须显式给出 cold root。任何未知/部分缺失状态都会 fail closed，
不会被当成合法 detached。这个入口只处理 external tactile；没有 external tactile 引用的数据集返回
`NO_EXTERNAL_TACTILE` 空操作。

### 触觉模态冷存储（默认只读）

触觉拆出/放回使用现有 entrypoint，不手工移动 `tactile/`。第一次命令只生成 proposal：

若目标是把整个 `tactile/` 当成一个模块迁走，推荐使用托管 move。第一次不带
`--apply` 只做严格只读计划：

```bash
python -m utils.dataset_postprocess modality-move DATASET_ROOT \
  --cold-root COLD_ROOT \
  --modality tactile
```

确认 hot/cold 路径、不同 `st_dev`、容量和 `ready=true` 后，单次执行：

```bash
python -m utils.dataset_postprocess modality-move DATASET_ROOT \
  --cold-root COLD_ROOT \
  --modality tactile \
  --apply \
  --confirm-no-active-readers
```

这是安全的 rsync 类 move，不是 shell `mv`：冷端 partial tree 由 dataset fingerprint 和
inventory hash 唯一绑定；完整文件按 SHA-256 复用，未完成文件先逐字节验证已有前缀再从断点
续传。SIGINT、进程异常或主机重启发生在 cold export 阶段时，hot `tactile/` 保持不动；再次
执行同一条 apply 命令即可继续。只有独立 cold inode、完整 hash、标准 tactile reader restore
drill 和 receipt 全部成功后，才原子退役 hot `tactile/`。原有 `modality-detach` 的显式
plan-hash/fingerprint 两阶段接口继续保留，作为更严格的审核和完整回退路径。

如果中断发生在热端退役事务中，重复 apply 会先恢复 journal，再重新规划；如果退役 receipt
已经持久化但原调用方没有收到结果，重复 apply 会返回同一个已验证结果，不会创建第二次迁移。
清理事务 quarantine 前会重新核对 receipt/inventory，并重新验证冷端所有 hash 和 tactile
reader；冷盘未挂载、路径不符或任一字节变化时会 fail closed，并保留 quarantine 和 journal，
不得为了“解除卡住”而手工删除这些目录。

既有的标准 v2.3 数据集不要求预先存在 modality inventory：dry-run 会只读构建临时 inventory，
apply 才会发布 sidecar 并执行迁移。由旧格式转换产生、与其他数据集共享 inode 的 hardlink
数据会在 planning/admission 阶段明确拒绝；不能把另一个路径名当作独立备份，也不要直接 unlink
其中任意一份。应先选择 canonical 数据集，或另行完成带独立备份 receipt 的 hardlink 拆分。

新采集和既有标准 v2.3 数据都沿用同一物理布局，不重写 Parquet 或 Episode catalog：

```text
DATASET_ROOT/
  data/                     # 数值帧；保留 tactile 相对引用
  videos/
  meta/
    episodes.jsonl          # 保留 tactile_paths 捕获事实
    modality_artifacts/     # inventory、receipt、事务 journal
  tactile/                  # 全部触觉 manifest 与 payload 的唯一前缀
```

`whole-prefix` 成功拆出后，hot root 中整个 `tactile/` 不存在；inventory 和 durable receipt
留在 `meta/modality_artifacts/`，所以纯视觉视图能够区分“已验证冷存储”和“误删”。放回后恢复为上面的
标准、自包含 v2.3 布局。RealTeleop 与 UMI writer 的回归测试会拒绝把触觉 artifact 写到其他前缀。
事务实现由 `collection_runtime.modality_storage` 拥有；本 CLI 仅负责 proposal、确认和展示结果。

新采集可显式启用 `--tactile-artifact-integrity`。writer 会在写 payload 时增量计算 stored-byte
SHA-256，并在每个 Episode 封口后自动写入 immutable inventory；因此不需要再单独执行“生成 hash”
步骤。旧数据或未启用参数的数据仍由 proposal 走兼容的全量扫描。无论 hash 来自采集期还是离线扫描，
apply 都不会跳过 cold-copy 校验、restore drill 和退役前全量复核。
采集期参数应从新数据集 Episode 0 开始启用；已有未封存 tactile 历史的数据集不会在采集中自动
回填 manifest，而是在 writer 创建前拒绝该参数，避免每次封口重复扫描全部历史 payload。

```bash
python -m utils.dataset_postprocess modality-detach DATASET_ROOT \
  --cold-root COLD_ROOT \
  --modality tactile

python -m utils.dataset_postprocess modality-status DATASET_ROOT
```

兼容默认是 `--detach-scope payload-only`：hot 端保留 manifest。若需要让整个 `tactile/`
成为可拆装模块，proposal 和 apply 都显式使用：

```bash
python -m utils.dataset_postprocess modality-detach DATASET_ROOT \
  --cold-root COLD_ROOT \
  --modality tactile \
  --detach-scope whole-prefix
```

proposal 通过后，apply 必须同时回填输出的 `plan_hash`、dataset fingerprint，并人工确认没有
OpenPI、训练脚本或其他 reader 正在读取数据集：

```bash
python -m utils.dataset_postprocess modality-detach DATASET_ROOT \
  --cold-root COLD_ROOT \
  --modality tactile \
  --detach-scope whole-prefix \
  --apply \
  --plan-hash PLAN_HASH \
  --confirm-dataset-fingerprint DATASET_FINGERPRINT \
  --confirm-no-active-readers
```

production apply 强制 hot/cold 位于不同 `st_dev`。它先完成独立 inode shadow export 和逐
manifest restore drill，最后才以 journal + quarantine 原子退役。`payload-only` 保留 hot
manifest；`whole-prefix` 原子退役完整 `tactile/`。detached root 不能继续正式数采，纯视觉训练
必须走 inventory + durable retirement receipt 驱动的 core-only admission。
RealTeleop、UMI 及共享 headless writer 都会在任何新写入前拒绝 detached/partial/pending-transaction
数据集；工具不会自动创建一个新的 hot `tactile/` 与旧 cold 内容混装。

放回也先 dry-run，再使用相同三项确认：

```bash
python -m utils.dataset_postprocess modality-rehydrate DATASET_ROOT \
  --cold-root COLD_ROOT \
  --inventory-hash INVENTORY_HASH
```

确认 proposal 后再执行：

```bash
python -m utils.dataset_postprocess modality-rehydrate DATASET_ROOT \
  --cold-root COLD_ROOT \
  --inventory-hash INVENTORY_HASH \
  --apply \
  --plan-hash PLAN_HASH \
  --confirm-dataset-fingerprint DATASET_FINGERPRINT \
  --confirm-no-active-readers
```

工具会在 hot 文件系统构造完整 staging tree，通过标准触觉 reader 和完整 v2.3 validation 后才
原子交换；cold 副本不会自动删除。当前实现已通过离线与 actual-v2.3 fixture，但在独立冷盘小型
canary 完成前不得直接用于大规模真实数据。

## 推荐汇总 Pipeline

大批量数据建议按下面顺序处理。结构变换步骤写到新目录；末尾的 tactile compact
是唯一默认的原地步骤，但只有显式 `--apply` 才会执行。

```text
raw dataset
  -> trim-nan-tails / clean-stutters / delete-episodes / split-episodes
  -> label-force-collisions
  -> causal-align
  -> wrist table keep-mask
  -> wrist depth mask postprocess
  -> final wrist keep-mask = postprocessed depth OR table keep-mask
  -> DM_force_calculate（需要 dense force 时，必须在 raw 降分辨率前完成）
  -> tactile compact v1（最后一步，default, offline and in-place）
  -> dataset_interface training-time masked augmentation
```

默认后处理入口会启用 canonical tactile compact v1；不加 `--apply` 时仍然只生成计划：

```bash
python -m utils.dataset_postprocess postprocess DATASET_ROOT
python -m utils.dataset_postprocess postprocess DATASET_ROOT --apply
```

只有明确不需要 compact 时才使用 `--no-tactile-compact`。compact 不在采集 writer/finalize
线程中运行；检测到正式采集进程时，`--apply` 会拒绝执行。默认磁盘余量为 20 GiB。

默认 mask 行为：

```text
final_keep_mask = depth_range_mask_postprocessed OR table_geometry_keep_mask
final 合成阶段不额外添加 proxy/body mask
```

默认 table keep-mask 应只包含桌面；如果上游 table keep-mask 已经渲入 painted body mask，
final keep-mask 会按输入保留这部分像素。

不建议把 random background 离线扩成完整新 dataset。正式训练时读取
`final_keep_mask`，在 DataLoader worker 内在线替换背景；离线视频只用于抽样检查。

结构类工具已经按“大文件 hardlink、可变 meta/data copy”的方式派生数据集：
`trim-nan-tails`、`causal-align` 会复制 `meta/` 和 `data/`，视频/触觉/depth
等大文件 hardlink；`clean-stutters` 会重写 `data/`；`delete-episodes` 和
`split-episodes` 会优先 hardlink 保留的 artifact。输出目录不能放在源 dataset 内部。

## Canonical tactile compact v1

正式后处理统一使用两个固定 profile，禁止为不同数据集临时组合 shape、位深和 codec：

- `dmtac_raw_120x160_q6_delta_zstd_v1`：保持每个 tactile frame、时间戳和采样频率不变；
  原始 `uint8[480,640]` 做确定性的 4x4 整数 area mean，随后做 6-bit 中点重建，逻辑输出
  是 `uint8[120,160]`；每个 chunk 独立做可逆帧差和 Zstd。
- `dmtac_force_qint16_zstd_v1`：`distributed_force` 的逻辑输出仍是
  `float32[288,384,3]`，存储为每 chunk/每通道带 scale 的对称 int16，再做 Zstd。

两者都设置 `frequency_preserved=true`。Parquet 的
`tactile_raw_shape_per_sensor` 和旧 `tactile_shapes_json` 是采集源审计信息，可以继续记录
`480x640`；实际训练/回放数组的 shape、dtype、量化和 source provenance 以 manifest stream
spec 为准。manifest 的逐 tactile-frame `shapes/dtypes` 会更新为 compact 后的逻辑值，并把
原值保存在 `source_shapes/source_dtypes`。

`DM_force_calculate` 只能消费 full-resolution raw，所以需要 dense force 时必须先完成 force
解算，再运行 compact。compact 同时扫描 `tactile/` 和 `tactile_force/`，并在全部完成后写入
`meta/tactile_compaction.json` 和 `meta/info.json.tactile_storage_profile`。

命令默认只读并给出完整计划；任何未知 dtype/shape/encoding 都会令计划 `ready=false`，apply
会 fail closed：

```bash
python -m utils.dataset_postprocess tactile-compact datasets/umi_dual_wrist
```

先限制为一个 manifest 做灰度发布。工具逐 chunk 压缩并立即解压校验，所有 stream staging
成功后才原子提交 manifest，提交后才删除旧流；`--min-free-gb` 是转换期间必须保留的磁盘余量：

```bash
python -m utils.dataset_postprocess tactile-compact datasets/umi_dual_wrist \
  --level 1 --chunk-frames 32 --min-free-gb 20 --max-manifests 1 \
  --apply --delete-originals
```

检测到正式 collection/GUI 采集进程时，`--apply` 会直接拒绝执行。工具不会改视频、
wrist-depth 或触觉频率。批量前后必须用 `dataset_contracts.ChunkedTactileStream` 回读每个
stream 的首/中/末帧，并核对 manifest frame_count 与索引不变。

`tactile-recompress-zstd` 是 full-resolution raw 的无损重打包路径。当前稳定 profile 为
`dmtac_raw_temporal_delta_zstd_v1`：每 8 帧独立做 modular temporal delta，再使用 zstd
level 1；logical dtype、shape、帧数和帧索引不变。命令默认只读：

```bash
python -m utils.dataset_postprocess tactile-recompress-zstd DATASET_ROOT \
  --level 1 --chunk-frames 8 --min-free-gb 20
```

先用 `--max-manifests 1` 灰度；确认计划后才显式增加 `--apply --delete-originals`。工具拒绝
活动 collection、v2.4 committed capture、hardlink source、文件尺寸不符或运行中变化；每路
先写新 inode，逐 chunk round-trip 并全流 SHA-256 校验，再原子提交 manifest，最后删除 raw，
并在 `meta/tactile_recompression/` 写事务收据。它与会改变 raw 空间/量化精度的 compact v1
是不同 profile，不能在同一 stream 上叠加解释。

## Stable/source 与 sorted 数据的 hardlink 去重

对已经生成的 sorted dataset，可把内容完全相同的不可变大 artifact 改为指向 stable/source
文件的 hardlink。默认范围严格限制在路径组件 `videos/`、`tactile/`、`wrist_depth/`，不会处理
`meta/` 或 `data/`。计划阶段先按大小筛选，再对两边完整文件做 SHA-256；文件名和大小相同
不能代替内容校验：

```bash
python -m utils.dataset_postprocess hardlink-dedupe \
  datasets/sorted_v22 \
  --reference-root /path/to/DexTeleop_StableCollect_0705/datasets
```

计划中的 `estimated_reclaimable_bytes` 才是预计实际释放空间。确认采集已经结束并审阅计划后：

```bash
python -m utils.dataset_postprocess hardlink-dedupe \
  datasets/sorted_v22 \
  --reference-root /path/to/DexTeleop_StableCollect_0705/datasets \
  --apply --confirm-replace-targets
```

执行只原子替换 target 目录项，不删除或改写 reference。替换前会再次核对两边 inode、大小和
mtime 是否仍与完整哈希计划一致；任一文件变化即停止。完整哈希扫描和执行在正式采集进程
存在时都会拒绝。hardlink 之后两条路径共享同一 inode：删除任一路径不会删除另一链接，
但禁止对 artifact 做原地写入；需要修改时必须写临时文件后原子替换。

## 旧 UMI workspace-near-boundary 坐标修正（默认关闭）

该工具仅用于标定板位于旧桌面中心的历史 UMI v2.3 数据。新 frame 与旧 frame 方向相同，
但新原点在旧 frame 的 `X=-0.15 m`，所以历史 desk pose 转换到新坐标表达时执行
`x_new=x_old+0.15 m`。它不属于默认 `postprocess` pipeline，也不会被任何采集、训练、
回放或普通后处理命令隐式启用。

只调用子命令会完整扫描 metadata、episode 和 Parquet，但保持只读：

```bash
python -m utils.dataset_postprocess umi-workspace-near-boundary-shift DATASET_ROOT
```

只有确认目标确实是旧 table-center UMI 数据后，显式使用以下专用开关才会先备份
`data/`、`meta/`，再原地迁移：

```bash
python -m utils.dataset_postprocess umi-workspace-near-boundary-shift DATASET_ROOT \
  --apply-workspace-near-boundary-shift
```

工具会写入不可重复的 migration marker；已迁移数据、非 UMI 数据、非 v2.3 数据、
非 `desk_absolute` 数据、hash 不一致数据或正式采集进程运行期间都会拒绝执行。

## 历史 UMI 机械臂 action 时序修正（默认关闭）

旧 UMI 数据把同一采样时刻的 tracker pose 同时写入 `observation.state` 和机械臂
`action`，会形成恒等映射。专用迁移将机械臂部分改为
`action_arm[t] = observation_arm[t+1]`；夹爪 action 仍保留时刻 `t` 的 AngleTrigger
真实命令，不使用下一帧夹爪值。每个 episode 删除无法构造下一时刻 action 的末行，
并同步裁掉末尾双腕视频和对应触觉帧。`observation.last_action` 会按迁移后的 action
重建；首行使用当前机械臂 pose 和当前夹爪命令作因果初始化。

默认命令只执行完整只读预检：

```bash
python -m utils.dataset_postprocess umi-arm-action-next-observation DATASET_ROOT
```

只有明确确认目标是上述旧 UMI 数据后，才可使用专用开关。apply 会先创建同盘硬链接
快照，在 staging 中生成全部替换文件，验证完成后再切换目录：

```bash
python -m utils.dataset_postprocess umi-arm-action-next-observation DATASET_ROOT \
  --apply-umi-arm-alignment
```

该步骤不会被默认 `postprocess` 或其他转换隐式调用；已经带 alignment marker 的数据、
机械臂 action/state 不再完全耦合的数据、媒体计数不一致的数据，以及正式采集进程运行
期间都会被拒绝。

## v2.1 固定 q6 到 DexTeleop v2.2

转换必须提供显式 `PolicyIOLayout` JSON；工具不会猜测 q6 padding 对应夹爪还是灵巧手。
无 `--apply` 时只做完整路径/episode/Parquet 预检并打印计划：

```bash
python -m utils.dataset_postprocess v21-to-v22 datasets/legacy_task \
  --output-root datasets/legacy_task_v22 \
  --layout-json /abs/path/policy_io_layout.json

python -m utils.dataset_postprocess v21-to-v22 datasets/legacy_task \
  --output-root datasets/legacy_task_v22 \
  --layout-json /abs/path/policy_io_layout.json \
  --apply
```

转换永不覆盖源目录或已存在的输出目录。输出的
`meta/v21_to_v22_conversion.json` 记录 source info SHA-256、conversion id、layout hash 和
全部 Parquet 相对路径；原 q6 列保留用于追溯，新的模型字段按 v2.2 layout 生成。

## RM65 v2.2 base-frame 到 canonical desk 兼容版 v2.3

该转换只接受左右臂 reference frame 分别为 `left_arm_base/right_arm_base` 的显式
DexTeleop v2.2 数据，并使用 checked-in RM65 workspace-near-boundary station profile。
默认会完整扫描 Parquet、统计非有限 action/last_action，但不会写盘：

```bash
python -m utils.dataset_postprocess v22-to-v23-desk SOURCE_DATASET \
  --output-root OUTPUT_DATASET
```

确认计划后使用专用开关创建 copy-on-write derivative：

```bash
python -m utils.dataset_postprocess v22-to-v23-desk SOURCE_DATASET \
  --output-root OUTPUT_DATASET \
  --apply-v22-desk-conversion
```

转换后 absolute action/state/last_action 和 measured TCP anchor 存储在 canonical desk；
历史 `*_pose_base`、旧 `teleop.action_source.*_pose_desk` 审计列保持原值。非有限 action
和 episode 首帧 NaN last_action 不会被填充，行中会增加 `training.action_finite` 和
`training.relative_last_action_finite`。产物明确标记为
`v22_canonical_desk_migration`、`native_v23_causal_snapshot=false`，不能当作通过严格
native-v2.3 因果快照验收的数据。

检测 `FAULT/ESTOP + action 全 NaN` 的 episode 尾段：

```bash
python -m utils.dataset_postprocess plan-fault-nan datasets/2026-07-05_vive_rm65_real_teleop
```

检测时间间隔过长的分段，默认阈值建议从 `0.2s` 开始：

```bash
python -m utils.dataset_postprocess plan-stutters datasets/2026-07-05_vive_rm65_real_teleop --max-gap-sec 0.2
```

按长 gap 切分 episode，生成没有内部长卡顿的新 dataset。输出会重建全局
`index`，并把每个 segment 的 `frame_index` 从 0 重新编号：

```bash
python -m utils.dataset_postprocess clean-stutters \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_stutter_clean \
  --max-gap-sec 0.2 \
  --min-segment-frames 15 \
  --apply
```

删除指定 episode。默认只生成计划；真正执行必须加 `--apply`，并写到新目录：

```bash
python -m utils.dataset_postprocess delete-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 0,3,5-8 \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_clean \
  --apply
```

`--episodes` 使用 dataset 中的 `episode_index`，从 0 开始编号；例如 `episode_000000` 对应 `0`。选择器支持逗号和区间，例如 `1,3,5-8`。

自动裁掉旧数据中 `FAULT/ESTOP + action 全 NaN` 的尾段：

```bash
python -m utils.dataset_postprocess trim-nan-tails \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_trim_nan \
  --apply
```

把指定 episode 从原 dataset 中抽出为新 dataset，同时生成删除后的 clean dataset：

```bash
python -m utils.dataset_postprocess split-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 1,17,28,30,43,48,61,62,91,100,105,143,146 \
  --clean-output-root datasets/2026-07-05_vive_rm65_real_teleop_no_collision \
  --extract-output-root datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

抽取后的 dataset 保留原始 `episode_index`，方便回查原数据。

给已有 dataset 补 episode 级力/力矩 hard-stop 标签。默认只扫描；真正写入必须加 `--apply`：

```bash
python -m utils.dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

如果某个 dataset 是人工抽出的碰撞/超限负样本集，可显式覆盖为全 True：

```bash
python -m utils.dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --override all-detected \
  --apply
```

将相机和触觉引用改成因果对齐：每行选择同一 episode 内
`timestamp <= action timestamp` 的最近帧。输出会重建全局 `index`。默认只生成计划；真正执行写到新目录：

```bash
python -m utils.dataset_postprocess plan-causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean

python -m utils.dataset_postprocess causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_causal
```

如果同时做卡顿清理和因果对齐，建议先 `clean-stutters`，再对输出运行
`causal-align`，避免对齐跨过已切开的长 gap。

## 派生数据集组织、物理合并与 mask metadata 迁移

`organize_episodes.py` 用 JSON manifest 从多个源 dataset 选择 episode，生成连续的
`episode_index`，并执行受限的 schema migration。源 dataset 保持不变；构建过程先写入
staging 目录，全部成功后才原子发布，不能把输出目录放在源 dataset 内。已有 manifest 示例：

- `manifests/dexteleop_2026_07_sort.json`
- `manifests/bimanual_cube_new_success_wrist_mask_cleanup.json`
- `manifests/bimanual_cube_new_failure_case_wrist_mask_cleanup.json`

命令行默认只规划；`--apply` 才发布新输出，并且始终拒绝已有输出目录：

```bash
python -m utils.dataset_postprocess organize-datasets MANIFEST.json

python -m utils.dataset_postprocess organize-datasets MANIFEST.json \
  --dataset REVIEWED_OUTPUT_NAME \
  --apply
```

`copy` 与 `reflink_required` 物化在 staging 中逐文件读回并做 byte-for-byte
校验；独立复制若发生瞬时不一致最多重试三次，仍不一致或源文件在复制期间变化时整次构建
fail closed，目标目录不会发布。该检查用于避免只验证文件大小而漏过单字节复制损坏。

也可作为 Python API 使用：

```python
from utils.dataset_postprocess.organize_episodes import build_datasets, plan_dataset_build

plan = plan_dataset_build("/abs/path/manifest.json", dataset_names=["example"])
result = build_datasets("/abs/path/manifest.json", dataset_names=["example"])
```

### 物理合并安全合同

- 每个 source 必须已经具有完整且内部一致的 dataset task binding：
  `meta/info.json.task_id/task_description` 必须与 `meta/tasks.jsonl` 的每一条记录一致。
  source `task_id` 还必须等于 manifest 的 canonical task ID；不允许在物理合并时静默把未知
  task 改名成目标 task。输出只写 manifest 指定的一条 canonical task identity。唯一例外是
  manifest 的 source 显式提供 `reviewed_task_rebinding`：它必须绑定旧 task metadata 的精确
  snapshot SHA-256、目标 task identity 和非空审计原因；只允许没有任何帧级 task 列的
  metadata-only 旧数据。snapshot 变化、Parquet 出现 `task/task_id/task_index/task_description`
  列或目标 task 不一致都会 fail closed。源数据仍保持不变，重绑定 receipt 写入输出 plan/audit。
- 只有 task、数据内容形式以及 progress/stage profile 完全一致的 source 才能写进同一个
  output。目标 `task_description` 可以作为同一 task ID 的规范展示说明统一写入输出，但不能
  用它掩盖 source 内部不一致。Stage1-only 数据不能和完整四阶段数据合并。
- `artifact_materialization_policy=reflink_required` 要求视频、触觉、depth 等大文件生成
  独立 inode 的 CoW reflink；不允许 hardlink，也不在 reflink 不可用时静默退化成整文件复制。
- 文件系统不支持 reflink 时，必须在 manifest 中显式改用 `copy`，并在 apply 前核对预计输出
  体积和剩余空间；`copy` 同样要求独立 inode 与 `nlink=1`，只是会真实占用完整空间。
- `progress_artifact_policy=preserve` 会逐 episode 校验 sealed progress identity、profile hash
  和 annotation binding，然后只白名单复制正式 episode UUID。未封存或未被
  `meta/episodes.jsonl` 引用的临时目录不会进入输出。
- 对采用 native `core + tactile bundle` 分离布局的 source，manifest 必须在该 source 上显式
  指定 `tactile_storage_root`。预检会用标准 bundle reader 校验 dataset/episode identity、commit、
  manifest、artifact ownership、文件尺寸和帧对齐；默认不重复哈希全部大 payload，只有显式设置
  `verify_tactile_payload_hashes=true` 才执行逐 payload SHA-256。未声明 storage root、绑定错误、
  payload 缺失或 source 不是 native bundle 都会 fail closed。
- 合并输出不会继承只对 source UUID 有效的 native bundle binding。所选触觉 artifact 会按
  `artifact_materialization_policy` 物化进 staging，manifest 内部路径随 episode 一起重编号，随后
  输出成为自包含的兼容 external-tactile 数据集；`meta/info.json` 明确标记需要离线 full scan 后
  才能再次拆分。source core 与 source tactile bundle 始终只读且保持原绑定，可直接回退。
- 对采用旧式 inventory/receipt 迁出布局、只在热目录保留 tactile manifest 的 source，不允许
  依赖 storage ID 或目录名猜测 NAS 数据。manifest 必须同时显式指定
  `retired_tactile_inventory_hash` 与指向该 inventory 精确目录的
  `retired_tactile_artifact_root`。预检要求当前 independently-verified retirement receipt、源绑定、
  archive `inventory.json`、精确 artifact 集合、所有尺寸与 SHA-256 全部一致，并拒绝 symlink、部分
  hot 状态、额外文件、过期 inventory 或 native/legacy 双重绑定。验证通过后才可从 archive 直接
  物化到 staging；源 core 和 NAS archive 始终只读。该入口默认关闭且仅对 manifest 中逐 source
  显式钉死的旧数据生效。
- 合并只重排物理 `episode_index` 和 artifact path；原 capture identity、progress fact、
  Attempt/Failure/annotation 语义保持不变。
- manifest source 顺序是输出的规范顺序；同一 source 内按原 episode index 排序，从而能够
  复现合并前 OpenPI 多 source 的样本顺序。
- tactile 路径重编号后，source 的 detachable-artifact integrity sidecar 不再有效。输出保留
  hot tactile payload，但明确标为需要一次离线 full scan 才能再次 detach；不能复制旧 hash
  并冒充已校验。
- 发布后仍须校验 episode/frame 数、progress identity、Parquet 可读性、视频完整解码和尾帧、
  source/output inode 独立性，以及 OpenPI stage eligibility/order 等价性。验证完成前不得删除
  或改写 source。

截至 2026-08-23，0807 的两个完整四阶段分片已按上述合同物理合并到
`/home/user/zc/datasets/RealmanDMTAC/PhysicalCollections/2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage`。
输出包含 331 个 episode / 113,517 帧；Parquet 逐 episode 等价、993 段视频完整解码、progress
identity/annotation 与 source 顺序均已验证，且所有输出 payload 都是 `nlink=1` 的独立 inode。
当前 0804–0807 OpenPI profiles 通过
`LogicalCollections/writing_four_stage_fulltask_2026_08_04_through_08_07_v1/sources/writing4_2026_08_07`
软链接使用该物理输出；旧 source、旧 source 链接与旧 binding 仍保留作回退。
0806 的三个完整四阶段分片已纳入同一 manifest，但目标盘不足以容纳约 71.6 GiB 的全量独立
复制，在释放其已备份 tactile payload 前不得 apply，也不得用 hardlink 规避空间检查。

`relocate_mask_metadata.py` 用于目录迁移后重写 mask JSON/JSONL 中记录的绝对路径，
不会修改 bitpack 等二进制文件。默认是 dry-run；确认输出后才使用 `--apply`：

```bash
python -m utils.dataset_postprocess.relocate_mask_metadata \
  --root /abs/path/masks \
  --replace /old/masks=/new/masks

python -m utils.dataset_postprocess.relocate_mask_metadata \
  --root /abs/path/masks \
  --replace /old/masks=/new/masks \
  --apply
```

## 相机相关工具

RealSense 去畸变：

```bash
python -m utils.dataset_postprocess.real_undistort dataset \
  --dataset-root datasets/source_dataset \
  --output-root datasets/source_dataset_undistorted
```

腕部相机 mask / depth 后处理：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust --help
```

典型 wrist final mask 批处理：

先用 `wrist_cam_adjust dataset --output-kind keep-mask` 生成 `TABLE_MASK_ROOT`；
完整 table keep-mask 命令见 `utils/dataset_postprocess/wrist_cam_adjust/README.md`。

```bash
SESSION=datasets/2026-07-09_vive_rm65_real_teleop
TABLE_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/${SESSION##*/}_table_keep_mask
DEPTH_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess
FINAL_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask

python -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask \
  --session "$SESSION" \
  --all-episodes \
  --output-root "$DEPTH_MASK_ROOT" \
  --jobs 4 \
  --no-preview \
  --overwrite

python -m utils.dataset_postprocess.wrist_cam_adjust compose-final-mask \
  --session "$SESSION" \
  --table-mask-root "$TABLE_MASK_ROOT" \
  --depth-mask-root "$DEPTH_MASK_ROOT" \
  --output-root "$FINAL_MASK_ROOT" \
  --jobs 4 \
  --overwrite
```

## 目录说明

- `plan-fault-nan`：找出故障后继续录制产生的 NaN 尾段。
- `trim-nan-tails`：复制 dataset，并自动裁掉故障后的 NaN 尾段。
- `plan-stutters`：按 timestamp gap 规划分段。
- `clean-stutters`：按 timestamp gap 切分 episode，并重建输出 dataset 的 `index`。
- `delete-episodes`：复制式删除指定 episode。
- `split-episodes`：同时生成 clean dataset 和抽取 dataset。
- `label-force-collisions`：从 parquet 诊断文本聚合 episode 级力/力矩 hard-stop 标签。
- `plan-causal-align` / `causal-align`：把相机和触觉索引改成 latest <= sample timestamp。
- `real_undistort/`：RealSense 视频去畸变。
- `wrist_cam_adjust/`：腕部相机 mask、depth 可视化和后处理。
