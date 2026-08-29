# Worktree 本地资源与迁移

本仓库的本地状态不只有 `assets/`。离线数据集、采集记录、运行证据、模型权重、厂商目录、
机器路径、密钥、日志、测试输出和缓存都可能被 `.gitignore` 隐藏。它们不出现在
`git status` 中，也不会随分支合并或 `git clone` 自动迁移。

本页与 [`local_resources_manifest_v1.json`](./local_resources_manifest_v1.json) 是这些路径的
版本化登记。清单记录每类资源的忽略规则探针、允许的 tracked 边界和迁移方式；
[`scripts/dev_local_resources.py`](../../scripts/dev_local_resources.py) 只检查路径元数据和 Git
规则，不递归读取资源文件，也不复制、链接、改写或删除数据。

## 1. 分类与默认处理

| 分类 | 典型内容 | 默认迁移方式 |
| --- | --- | --- |
| `shared-read-mostly` | `assets/` 中已审核的机器人模型和模型权重 | 只允许现有 asset-store 工具建立兼容软链接 |
| `external-input` | `datasets/`、`data/`、`remote_data/`、历史离线数据 | 保留仓库外 canonical root，通过 `local_paths.json`、环境变量或 profile 绑定 |
| `mutable-evidence` | `runtime_artifacts/`、未跟踪的 `teleop_records/`、日志和输出 | 停止写入后做 inventory、复制和 hash/count 验证，再切换绑定 |
| `local-vendor` | 模型权重、仿真安装、厂商 checkout | 检查来源与 tracked 边界后逐项绑定或复制 |
| `local-secret` | `.env*`、`local_paths.json`、设备本地配置 | 从 tracked example 或密钥管理重新创建，不进入迁移回执 |
| `generated` | venv、cache、build、coverage、临时输出 | 在目标 worktree 重建，不迁移 |
| `developer-private` | `.codex/`、`.agents/`、`todos/`、编辑器状态 | 保持个人本地，不得成为源码或运行依赖 |
| `local-archive` | 根目录 `legacy/` 本地归档 | 仅在确需保留时做验证复制；版本化历史文档在 `docs/legacy/` |

软链接只解决“多个 worktree 使用同一个稳定、只读资源根”的路径兼容问题，不是隔离或只读
证明。通过链接写入会直接修改共享目标。因此：

- `assets/` 已有 bundle presence 检查和安全链接工具，可以按只读约定共享；
- SAM2/Grounding DINO/VGN 等模型权重必须登记为可选 bundle，保存来源 commit 和 checkpoint SHA-256，不能把外部源码 checkout 放进 `assets/`；
- `runtime_artifacts/`、采集记录、日志和生成输出可能有并发 writer，不允许默认共享链接；
- `teleop_records/`、`model/`、`hardware/VIVE/triad_openvr/` 是 mixed root，内部同时存在 tracked
  文件和 ignored 文件，绝不能整体替换成软链接；
- 数据集应优先放在 worktree 外并显式绑定，避免分支删除、worktree 清理或并发写入影响数据。

## 2. 日常只读检查

从任意 AgenticRobot worktree 根目录运行：

```bash
python scripts/dev_local_resources.py audit
```

该命令验证：

- 清单中的 probe 仍由预期 `.gitignore` 规则处理；
- 应完全不受 Git 跟踪的根目录没有意外 tracked 内容；
- mixed root 仍保留其版本化例外，避免被整体链接或整体迁移；
- 当前 worktree 中哪些登记路径存在、是目录还是软链接。

机器可读输出：

```bash
python scripts/dev_local_resources.py audit --json
```

命令不会枚举目录内容、计算数据大小或读取密钥。若需要容量、文件数、hash 或内容级 provenance，
应由对应 dataset/runtime artifact 工具在已停止 writer 的源上完成。

## 3. Worktree 间迁移计划

先生成无副作用计划：

```bash
python scripts/dev_local_resources.py plan \
  --source-worktree /absolute/path/to/source \
  --target-worktree /absolute/path/to/target
```

计划只比较登记路径是否存在和路径类型，并为每类资源给出策略；它不会执行复制。目标路径已经
存在时会标出 collision review，不能覆盖、合并目录或改写链接。

### 3.1 Development assets

配置一次 clone-shared asset root，然后让标准工具为目标 worktree 建立被忽略的兼容链接：

```bash
python scripts/dev_asset_store.py configure --asset-root /absolute/path/to/assets
python scripts/dev_asset_store.py link-worktree --worktree /absolute/path/to/worktree
```

工具不会替换已存在且指向不同位置的 `assets` 路径。共享资产普通开发只读；修改资产需要单独
认领可写资产工作区，并评审 provenance/hash。

### 3.2 离线数据集

数据集保留在仓库外，按 [`local_paths.example.json`](../../local_paths.example.json) 创建被忽略的
`local_paths.json`，优先设置 `dataset_root`、`real_robot_dataset_root` 等现有键。不要为了让代码
找到数据而把绝对路径写入 tracked profile、测试或源码，也不要把多个可能写入的数据集根做
worktree 软链接。

跨机器迁移时，先在源存储生成数据集自身的 manifest/hash，再复制到全新目标目录并复验，最后
只修改目标机器的本地绑定。Git commit 只能记录数据合同、schema、公开 fixture 或经过评审的
evidence，不能代替数据迁移。

### 3.3 Runtime artifacts 与采集记录

迁移前必须停止所有 producer/writer。`runtime_artifacts/` 使用现有
[`scripts/runtime_artifact_migration.py`](../../scripts/runtime_artifact_migration.py) 的 plan/copy/verify
流程；只有验证通过后才能切换 `runtime_artifacts_root`。其他可变记录也必须遵循“源 inventory →
新目标复制 → hash/count 复验 → 显式重绑定”，不得直接复用一个正在写入的目录。

`teleop_records/` 中的 calibration profiles、hand-zero calibrations 和指定 VIVE coordinate profile
受 Git 跟踪；只迁移被忽略的具体子项，不能复制或链接整个根覆盖这些版本化文件。

### 3.4 模型、厂商目录和秘密配置

`model/` 保留 tracked `README.md` 与 `openpi` gitlink；`triad_openvr/` 保留 portable wrapper。
迁移权重或厂商内容时逐个处理 ignored child，并保留来源、版本和 hash。不得整体替换 mixed root。

`.env*`、`local_paths.json` 和设备配置从 example 或密钥管理重新创建。迁移计划和提交信息不得写入
密钥值、令牌、设备凭据或不必要的本机绝对路径。

`hardware/quest_dev/assets/certs/` 是 mixed root：只跟踪 `.gitkeep`，首次 HTTPS 启动生成的
`.crt` 和 `.key` 由模块内 `.gitignore` 排除。它们是本机开发证书与私钥，应在目标 worktree
重新生成，不复制、不提交，也不通过 assets 软链接共享。根 `/assets` 规则只管理仓库根的
reviewed asset store，不得再次改成会匹配嵌套模块目录的未锚定规则。

## 4. 开发维护规则

新增或修改离线数据、记录、模型、厂商目录、秘密配置、生成输出或个人状态的 `.gitignore`
规则时，必须在同一提交中：

1. 更新 `local_resources_manifest_v1.json`，选择明确分类与迁移方式，并添加不会访问真实内容的
   synthetic ignore probe；
2. 若根目录含 tracked 子项，标记为 `mixed`，说明例外，禁止 whole-root 链接；
3. 更新本页中受影响的操作边界，必要时更新 `local_paths.example.json` 或所属 runtime 文档；
4. 更新 `tests/test_dev_local_resources.py`，运行该 focused test 和
   `bash scripts/verify_dev_base.sh`；
5. 交接时声明哪些本地数据没有被 Git 提交、复制或验证。

只新增 `.gitignore` 条目而不登记，或只新增登记而没有可验证 ignore probe，都会造成后续 worktree
迁移遗漏；统一开发门禁会把现有规则漂移报告为失败。
