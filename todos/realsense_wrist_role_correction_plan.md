# RealSense 左右腕相机语义纠正方案（原始数据零改动版）

## 1. 本版约束与理解

本方案以以下约束为最高优先级：

1. 所有已经采集的原始数据永久只读：不修改 `meta/devices/realsense.json`、`episodes.jsonl`、Parquet、MP4、mask，也不在原始数据目录增加 correction sidecar。
2. 远端当前只有一部分历史原数据，后续会持续补充。因此不能依赖一次性的 503 episode 清单、按日期划界或人工逐批换名。
3. 每次训练读取数据时，都必须根据数据自身记录的 RealSense serial 动态判断 stored key 的物理含义；后来补充的历史数据应自动得到同样处理。
4. 改动尽量限制在代码、运行配置、训练适配和部署 artifact；不做数据迁移。
5. 新采集从切换点开始使用正确物理语义；旧模型继续兼容其训练时的历史语义。

已确认物理事实：

```text
physical left wrist  = serial 260522278055
physical right wrist = serial 261122273924
```

当前旧配置/历史数据记录的 stored mapping 为：

```text
stored left_wrist  -> serial 261122273924 -> physical right wrist
stored right_wrist -> serial 260522278055 -> physical left wrist
```

## 2. 总体决策

采用三段式方案：

```text
采集端：纠正当前 serial 绑定，今后 left/right 表示真实物理左右
训练端：只读原数据，根据 serial 动态建立 canonical role -> stored key
部署端：根据模型 artifact 显式建立 canonical source -> model input slot
```

不采用：

- 原地改 metadata；
- 重命名或交换视频/mask 文件；
- 为每批补数人工生成修复版数据；
- 全局 `swap_left_right` 布尔开关；
- 按目录名、日期、USB 枚举顺序推断左右；
- 把当前 503 episode 固化成长期 allowlist。

## 3. 必须分开的三种身份

| 身份 | 示例 | 含义 |
| --- | --- | --- |
| immutable device identity | `260522278055` | RealSense 硬件 serial |
| canonical physical role | `left_wrist` | 永远指物理左腕 |
| stored/model key | stored `right_wrist`、model `left_wrist_0_rgb` | 历史数据名或某个模型的输入接口 |

项目中的目标不变量为：

```text
canonical left_wrist  == physical left  == 260522278055
canonical right_wrist == physical right == 261122273924
```

stored key 和 model slot 都允许因历史原因与 canonical role 不同，但差异必须显式、有来源、可验证。

## 4. 最小代码端契约

### 4.1 角色绑定契约

增加一个纯契约 `CameraRoleBinding`，内容只表达 serial 和物理角色：

```yaml
schema: dexteleop.camera_role_binding.v1
roles:
  left_wrist:
    serial: "260522278055"
  right_wrist:
    serial: "261122273924"
```

短期可由修正后的共享 RM65 camera registry 提供这份事实，训练启动时通过显式参数传入；不要让 `dataset_interface` 隐式导入 `teleop_runtime` 配置。契约 schema/hash 放在 `dataset_contracts`，解析执行放在所属运行层。

第一阶段不顺带重构整个混合 camera registry。只纠正 serial，并增加跨配置一致性校验；workstation assignment、mount profile、runtime snapshot 的彻底拆分另列后续任务。

### 4.2 读取时解析结果

训练端统一生成只存在于内存和训练输出 receipt 中的结果：

```text
CameraSemanticResolution
  source_dataset_id
  source_device_snapshot_hash
  canonical_to_stored_key
  evidence_by_role
  resolution_status: identity | swapped | ambiguous
```

示例：

```yaml
canonical_to_stored_key:
  left_wrist: right_wrist
  right_wrist: left_wrist
evidence_by_role:
  left_wrist:
    serial: "260522278055"
    stored_key: right_wrist
  right_wrist:
    serial: "261122273924"
    stored_key: left_wrist
resolution_status: swapped
```

它不是写回数据集的修复文件，而是一次训练运行的只读解析结果。

## 5. 数据采集与未来数据保存

### 5.1 当前配置纠正

受控切换时把当前站点绑定改为：

```text
left_wrist  -> 260522278055
right_wrist -> 261122273924
```

需要同步检查并纠正所有仍参与运行的代码侧重复配置，包括：

- `teleop_runtime/configs/devices/realsense.json`；
- `robot_motion/configs/rm65_camera_registry.json` 中对应 serial；
- 仍被正式或兼容入口读取的 UMI/camera 配置。

不要只改其中一个入口，以免 GUI collection、headless collection 和 policy deployment 得到不同语义。

### 5.2 预检

正式采集/部署前打印并校验：

```text
canonical role -> configured serial -> observed serial -> physical mount
```

规则：

- serial 必须存在且左右唯一；
- observed serial 必须与配置相等；
- 禁止将 RealSense 枚举结果按顺序 zip 到左右 key；
- 非交互 `assume_yes` 也不能跳过物理角色校验；
- 不满足条件时 fail closed。

### 5.3 新数据保存

现有 writer 已会把 source 的 device info 写入 `meta/devices/realsense.json`。配置纠正后，新数据会自然保存为：

```text
left_wrist  key 中记录 260522278055
right_wrist key 中记录 261122273924
```

因此第一阶段不需要修改数据 schema，也不需要为新旧数据增加强制迁移字段。数据自身的 stored key + serial 已足够让训练端判断它是 identity 还是 historical swapped。

## 6. 训练端：持续补数下的动态解析

### 6.1 解析算法

训练 loader 请求 canonical `left_wrist/right_wrist` 时执行：

1. 读取本次 episode 所属 source 的 RealSense device snapshot。
2. 建立 `stored_key -> observed serial`。
3. 用 `CameraRoleBinding` 建立 `serial -> canonical physical role`。
4. 反向得到 `canonical role -> stored key`。
5. 用 stored key 定位原视频、mask、内参和有效性，在 dataset 层输出 canonical role。
6. 再由训练 artifact 把 canonical role 映射到该 checkpoint 的 model input key。

两种数据会自动得到不同结果：

| 数据 | snapshot | 解析结果 |
| --- | --- | --- |
| 历史数据 | left=261..., right=260... | swapped |
| 修正后新数据 | left=260..., right=261... | identity |

这使后续补回来的历史数据无需修改，只要其 device snapshot 仍在，就会自动按 serial 被正确解释。

### 6.2 三种数据布局

解析器按以下优先级查找证据：

1. **直接原始数据集**：读取 `<root>/meta/devices/realsense.json`。
2. **整理/合并数据集**：逐 episode 读取 `meta/provenance.jsonl` 确定 source，再读取 `meta/source_devices/<source>/realsense.json`；不能给整个合并根目录套统一 swap。
3. **证据缺失**：只允许训练配置提供显式、source-scoped override；override 必须声明 source identity 和 expected stored mapping。否则该 episode 对 wrist training 为 ambiguous，拒绝读取。

禁止使用日期、路径中出现 `old/new`、episode 编号或当前远端清单推断语义。

### 6.3 映射必须覆盖完整视角

内部读取 API 不应只传字符串 camera key，而应传类似：

```text
ResolvedCameraView(
  canonical_key="left_wrist",
  stored_key="right_wrist",
  source_dataset_id=...,
  serial="260522278055",
)
```

所有原始资源都用 `stored_key` 读取，模型输出字段使用 `canonical_key`：

- RGB MP4；
- wrist depth mask/raw；
- `final_keep_mask`；
- intrinsics；
- frame timestamp/index；
- missing-view 和 modality validity；
- camera-specific augmentation 参数或统计。

这样可以避免“RGB 换了，但 mask/内参仍是另一侧”的隐蔽错误。

### 6.4 持续补数与训练可复现

每次训练启动都重新扫描当前输入 roots，不缓存当前 503 episode 为长期事实。启动前生成审计摘要：

```text
source id | episode count | snapshot hash | identity/swapped/ambiguous | canonical mapping
```

摘要和解析 receipt 只写入训练 run/artifact 输出目录或日志，不写回 source dataset。receipt 至少绑定：

- 本次看到的 source dataset identity/fingerprint；
- episode manifest hash；
- device snapshot hash；
- role binding hash；
- 每个 source 的 canonical mapping；
- identity/swapped/ambiguous episode 数量。

以后补充数据后，新的训练扫描和 receipt hash 会变化，形成新的可复现训练输入身份；旧训练 receipt 仍能说明当时实际使用了哪些数据。

当前已核对的 503 episode 只用于实现后的回归验证：应全部解析为 swapped、0 ambiguous。它不是产品逻辑中的常量。

### 6.5 两种训练模式必须分开

补充历史数据后继续训练现有旧 checkpoint，是近期最实用的路径。它和训练全新 canonical 模型不能混为一谈。

#### A. `legacy_continuation`

先由 resolver 把任意新旧数据恢复成 canonical physical role，再按旧 checkpoint 的训练语义投影：

```text
canonical right_wrist -> observation.images.left_wrist
canonical left_wrist  -> observation.images.right_wrist
```

这样：

- 后续补回的历史 swapped 数据可以继续使用；
- 切换后新采集的 identity 数据也可以加入；
- 两类数据进入旧 checkpoint 时，物理视角含义完全一致；
- 不需要修改任何 source dataset 或 checkpoint 权重。

现有训练启动代码会要求 base artifact 与目标 artifact 的完整 camera contract 相同。实施时应把兼容判定拆成：

- checkpoint-native `model_input_key/server_camera_slot` 必须保持一致；
- `source_key` 允许在显式、双射、带 hash 的 camera semantics route 下重新绑定；
- 未声明的 source permutation 仍然拒绝。

#### B. `canonical_identity`

用于从头训练或经过明确语义迁移的新模型：

```text
canonical left_wrist  -> observation.images.left_wrist
canonical right_wrist -> observation.images.right_wrist
```

旧 checkpoint 若作为 warm start，其输入语义发生了置换，必须作为独立迁移实验验证；不能只改 artifact 名称后宣称为 canonical 模型。

训练命令必须显式选择 `legacy_continuation` 或 `canonical_identity`，不得根据 checkpoint 路径或日期自动推断。

### 6.6 原数据写保护

训练入口增加明确的只读保证：

- source root 仅以读模式打开；
- cache、索引、抽样 manifest、解码缓存只能写入训练工作目录/系统 cache；
- dry-run 对 source root 做 before/after inventory/hash 对照；
- 测试中将 fixture source root 设为只读，确保 resolver/loader 不尝试创建 sidecar。

本方案不再包含“物化修复数据集”的阶段。

## 7. 部署端：旧模型与新模型分流

### 7.1 旧 wrist-only 模型

旧 checkpoint 已经按历史反向标签训练。采集配置纠正后，要保持模型接收到的物理视角不变，应创建新的版本化兼容 artifact：

| canonical runtime source | legacy model input key | server slot |
| --- | --- | --- |
| `right_wrist` | `observation.images.left_wrist` | `left_wrist_0_rgb` |
| `left_wrist` | `observation.images.right_wrist` | `right_wrist_0_rgb` |

现有 `PolicyCameraInput(source_key, model_input_key, server_camera_slot)` 已能表达该关系，不需要在 observation 中做全局 swap，也不修改 checkpoint。

新 artifact 应显式记录：

```text
camera_semantics = legacy_swapped_wrist_capture_v1
base_checkpoint/artifact hash
camera input routing hash
expected runtime CameraRoleBinding hash
```

不要静默覆盖现有 artifact；通过新 artifact id 选择兼容行为。

### 7.2 新训练模型

dataset loader 先输出 canonical physical roles；新模型的 artifact 再采用 identity mapping：

```text
left_wrist  -> observation.images.left_wrist  -> left_wrist_0_rgb
right_wrist -> observation.images.right_wrist -> right_wrist_0_rgb
```

训练生成的 policy artifact 绑定本次训练 receipt 和 camera role binding hash。部署预检必须显示：

```text
physical role -> runtime serial -> source key -> model input key -> server slot
```

并区分 artifact 是 `legacy_swapped` 还是 `canonical_identity`。

## 8. 项目所有权与可读性

- `dataset_contracts/`：`CameraRoleBinding`、解析 receipt 的纯 schema/hash；不读取文件、不拥有设备。
- `dataset_interface/`：只读 `CameraSemanticResolver` 和 `ResolvedCameraView`；统一包装视频、mask、内参和 validity 读取。
- `teleop_runtime/configs/devices/`：当前工作站采集 serial 配置。
- `robot_motion/configs/`：当前共享 RM65 camera registry；本次只做一致性纠正，不展开大重构。
- `deployments/openpi/training/`：接入 resolver、训练启动审计和 training receipt。
- `deployments/openpi/client/` 与 policy artifact：模型特定 source-to-slot 路由。
- `control_station/`：只展示预检结果，不复制 resolver 或拥有相机语义。

统一术语：

- `canonical_role`：物理语义；
- `stored_key`：原数据中实际字段名；
- `serial`：设备身份；
- `model_input_key` / `server_camera_slot`：模型接口。

避免使用 `cam1/cam2`、`swap=true`、`fixed_v2` 等无法单独解释的命名。

## 9. 分阶段实施

| 阶段 | 内容 | 是否修改原数据 | 放行条件 |
| --- | --- | --- | --- |
| P0 | 确认 serial 事实和进程重启边界；历史语义数据可纠正，因此不要求暂停采集 | 否 | 范围确认 |
| P1 | 增加纯契约、serial 驱动 resolver 和只读测试 | 否 | identity/swapped/ambiguous 测试通过 |
| P2 | 接入训练 loader、mask/内参/validity 成组映射和启动 receipt | 否 | 当前 503 episode 全部 swapped、0 ambiguous；source inventory 未变化 |
| P3 | 增加 `legacy_continuation` 训练路由并创建旧 checkpoint 的新兼容 artifact | 否 | 旧模型训练/部署收到的物理图像均与修正前逐项相同 |
| P4 | 原子切换采集 config、共享 registry、预检和部署 artifact | 否 | 本地无硬件测试及远端同步检查通过 |
| P5 | 经授权做 camera-only 实机确认并采全新 pilot dataset | 只产生新数据 | 新数据 snapshot 为 canonical identity |
| P6 | 下次正常完整重启后按正确角色采集；历史补数由训练扫描自动接入 `legacy_continuation` | 否 | 每次训练 0 ambiguous 或显式剔除报告 |
| P7 | 持续扩训旧模型；验证混合 historical/new 数据的物理输入一致性 | 否 | training receipt、旧模型离线评估通过 |
| P8 | 可选：从头训练或显式迁移 canonical identity 模型 | 否 | 独立 artifact、训练 receipt 和评估通过 |

P3 和 P4 必须一起准备：不能先单独纠正运行时相机配置，再让旧 artifact 按同名 source 输入，否则旧模型的物理左右会立刻翻转。

## 10. 验证矩阵

### 10.1 resolver 单元测试

- historical snapshot：解析为 swapped；
- corrected snapshot：解析为 identity；
- serial 缺失、重复、未知或一侧冲突：ambiguous/fail closed；
- snapshot key 顺序变化不影响结果；
- 合并数据集中不同 source 可同时为 identity 和 swapped；
- 后续新增 source/episode 后重新扫描即可接入，无需修改规则。

### 10.2 端到端纯数据测试

用左右不同颜色、不同 mask 和不同内参的 fixture 验证：

```text
source snapshot -> canonical resolver -> video/mask/intrinsics reader
-> augmentation -> OpenPI training sample -> artifact camera contract
```

必须证明 canonical left 始终是 serial `260522278055` 对应的完整模态，canonical right 始终是 `261122273924`。

### 10.3 部署测试

- 旧 artifact：runtime right source 到 legacy left slot，runtime left source 到 legacy right slot；
- 新 artifact：canonical identity routing；
- legacy continuation training 与 legacy deployment 使用同一份 canonical-to-model 语义路由；
- artifact/source/slot 唯一性和 binding hash 校验；
- 缺少预期 runtime serial 时禁止腕部部署；
- 使用 recorded/fake frames 比较切换前后旧模型实际像素输入，而不只比较 key 名。

### 10.4 工程检查

- 使用 `/home/zc/anaconda3/envs/dexteleop310/bin/python -m pytest` 跑最小相关测试；
- `py_compile` 和 `git diff --check`；
- 相关目录已有并行 dirty changes，实施时必须逐文件复核，只修改授权范围；
- 远端只通过 `codex_remote` wrappers 检查/同步，先 dry-run 和 `remote_sync_check.sh`；
- camera-only 确认需要显式授权，机器人运动仍需独立安全范围。

## 11. 回滚

- 原始数据没有任何改动，无需数据回滚。
- resolver 通过训练配置显式启用；回滚只影响训练读取代码，不影响 source root。
- 当前和兼容 policy artifact 都保留，通过 artifact id 切换。
- 采集配置保留版本化回滚文件，但回滚到旧配置时必须同时选择与之匹配的 artifact，禁止独立回滚一侧。
- 所有训练 run 保留 camera resolution receipt，可追溯当时各 source 的 serial 解析结果。

## 12. 最终验收标准

只有同时满足以下条件才算完成：

1. 所有历史 source roots 的文件 inventory/hash 在训练前后不变。
2. 当前已知 503 episode 全部自动解析为 historical swapped，0 ambiguous。
3. 后续新增一份同类历史数据后，无需修改代码/数据，只需重新启动扫描即可纳入训练。
4. 新 pilot 数据自动解析为 canonical identity。
5. 旧模型切换前后的实际物理像素输入一致。
6. 旧模型扩训时，无论输入来自 historical swapped 还是 corrected identity 数据，实际 model slot 的物理含义保持一致。
7. 新模型训练和部署均使用 canonical identity mapping。
8. 日志能完整回答：某一训练样本的 canonical role、stored key、serial、source dataset 和 artifact slot。

## 13. 二次复核结论（2026-08-04）

二次端到端审查补齐了初版遗漏的三个边界：

1. 普通 OpenPI `scripts/train.py` 不经过 model-RTC transform。新增
   `deployments.openpi.corrected_train`，仅允许 `-corrected-v1` config，训练前逐 source
   审计 serial，并通过 `DEXTELEOP_CAMERA_ROLE_REGISTRY` 让 dataset worker 在视频和 mask
   读取边界解析物理角色；历史无后缀 config 仍冻结为显式回滚。
2. compatibility artifact 现在不仅校验 base hash，还严格校验 checkpoint-native model
   input、server slot、全部非相机字段以及精确 wrist route；重复包装、部分 metadata、错误
   route hash 和带正确 base hash 的其他字段篡改均 fail closed。
3. ordinary OpenPI 在选择 frame index 后才进入共享 media pipeline，因此 remapped source
   必须额外证明 canonical/stored frame-index 列逐行相等。wrapper 已把该检查纳入训练 admission。

远端真实数据验证：503 个 remapped episode、132,718 行全部对齐；OpenPI `.venv` 的
`--dry-run --sample-check` 实际读取了 episode 0 的双腕 `480×640×3` 输入并报告 swapped
映射。未启动 trainer、模型权重、GPU server、相机或机器人，也未修改数据集。
