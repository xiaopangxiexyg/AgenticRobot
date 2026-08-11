# 后期标注源码投影与便捷迁移规划

## 1. 修订后的目标

目标不是把标注做成一个独立性很强、需要单独发布和维护版本的封装包，而是：

> 标注代码继续以 DexTeleop 主仓为唯一源码，通过一份受测试保护的“标注源码投影清单”，按原目录结构检出或导出到不同标注机器。

这样处理后：

- 日常标注功能仍直接修改 `control_station/refinement_annotation/` 及其现有下游模块；
- 数据合同、reader、store 仍保持当前领域所有权，不复制到 annotation 专用 namespace；
- 标注机器可以频繁 `git pull` 或同步投影差异，不需要等待另一个 wheel/package 发布；
- 现有入口继续是 `python -m control_station refinement-annotation`；
- `.venv-annotation`、`annotation` dependency extra、`uv.lock` 继续复用主仓；
- 投影只是代码部署视图，不是第二套源码，也不拥有独立的数据协议。

本阶段仍只规划，不修改运行代码。

## 2. 当前状态与真正阻碍

仓库已经有适合异机标注的环境基础：

- `pyproject.toml` 有 `annotation` optional dependency；
- `scripts/install_annotation_env.sh` 可创建 Python 3.10 的 `.venv-annotation`；
- 标注 GUI 是 loopback Web 应用，不启动设备和机器人；
- native finalized dataset 和 retrospective workspace 都已接入同一个标注界面。

现在不能直接复制 `control_station/refinement_annotation/`，主要不是目录不够独立，而是 Python 聚合导入扩散。只导入当前标注 CLI，静态分析会触达约 289 个仓库 Python 文件：

| 顶层包 | 被导入文件数 |
| --- | ---: |
| `collection_runtime` | 11 |
| `control_composition` | 35 |
| `control_station` | 16 |
| `dataset_contracts` | 60 |
| `dataset_interface` | 64 |
| `robot_motion` | 95 |
| `utils` | 8 |

主要来源是：

- `control_station/__init__.py`
- `dataset_contracts/__init__.py`
- `dataset_interface/__init__.py`
- `collection_runtime/progress/__init__.py`

它们为了兼容，从包根 re-export 大量与标注无关的对象。Python 导入任意子模块时会先执行父包 `__init__.py`，所以仅靠 sparse-checkout 排除机器人目录会导致导入失败。

因此正确顺序是：先让标注调用链使用具体子模块，并把这些聚合入口改成受兼容测试保护的 lazy export；之后再建立源码投影。无需先移动合同或创建新 package。

## 3. 推荐模式：主仓唯一源码 + 两种投影交付

### 3.1 唯一投影清单

在主仓维护一份声明式清单，例如：

```text
release/annotation_projection.toml
```

它只描述：

- 必须检出的源码路径；
- 静态网页和安装脚本；
- 标注测试与最小 fixtures；
- 禁止出现在运行闭包中的顶层包；
- 系统依赖和启动入口；
- 投影格式版本。

它不复制源码，不包含数据集路径，也不用数据配置 hash 判断 Episode 是否可标注。

建议清单结构：

```toml
schema = "dexteleop.annotation-source-projection.v1"
entrypoint = "python -m control_station refinement-annotation"

include = [
  "control_station/refinement_annotation/**",
  "control_station/__init__.py",
  "control_station/__main__.py",
  "collection_runtime/immutable_canonical_store.py",
  "collection_runtime/progress/stage_annotation_store.py",
  "collection_runtime/progress/failure_trigger_store.py",
  "dataset_contracts/<annotation closure>",
  "dataset_interface/<annotation closure>",
  "scripts/install_annotation_env.sh",
  "pyproject.toml",
  "uv.lock",
  "README_ENV.md",
  "docs/real/LONG_HORIZON_STAGE_ANNOTATION.md",
  "docs/real/WRITING_STAGE1_RETROSPECTIVE_ANNOTATION.md",
]

deny_imports = [
  "robot_motion",
  "control_composition",
  "teleop_runtime",
  "hardware",
  "sim",
  "deployments",
  "model",
]
```

最终 include 不应长期手工猜测。CI 中应从入口计算实际 import closure，再验证：

1. 所有运行依赖都在 projection 中；
2. 没有命中 denylist；
3. 新增 import 后若忘记更新 projection，测试立即失败；
4. 静态资源和文档等非 Python 文件由清单显式补充。

### 3.2 有 Git 访问权限：sparse-checkout，作为首选

标注机器保留一个 DexTeleop 稀疏工作树，目录仍是原始包路径：

```text
DexTeleop-annotation/
  pyproject.toml
  uv.lock
  scripts/install_annotation_env.sh
  control_station/
    __init__.py
    __main__.py
    refinement_annotation/
  collection_runtime/
    ...标注持久化闭包...
  dataset_contracts/
    ...纯合同闭包...
  dataset_interface/
    ...只读适配闭包...
  tests/
    ...标注相关测试...
  docs/
    ...标注文档...
```

更新流程保持接近主仓开发：

```text
主仓修改标注代码
→ 本地标注测试与 projection 检查
→ push/同步同一 commit
→ 标注机 git pull
→ uv sync --frozen --extra annotation（仅 lock 改变时必要）
→ smoke test
→ 重启标注服务
```

优点：

- Git 正确处理新增、修改、重命名和删除，不会残留旧文件；
- 可以直接查看 commit、diff、回滚；
- 不需要新包名、新版本号或 compatibility shim；
- 频繁更新前端、server、reader 时只下载实际变化；
- 标注机仍能对源码做只读诊断，但不需要完整机器人仓库。

### 3.3 无 Git 访问权限：由同一清单生成源码投影

提供一个主仓脚本，例如：

```text
scripts/export_annotation_projection.py
```

它根据同一份 `annotation_projection.toml`，把现有 canonical 文件按原路径复制到一个新的版本化 staging 目录：

```text
annotation-projection-<git-commit>/
  <原目录结构>
  annotation_projection_receipt.json
```

receipt 记录软件 commit、文件列表和文件内容摘要，用于确认更新完整性；这些摘要只用于代码投影校验，不参与数据集或易变配置的 admissibility。

远端不要在一个长期目录上反复 `--no-delete` 叠加，因为文件重命名/删除后会留下旧模块。更安全的部署是：

```text
~/DexTeleop-annotation/
  releases/<commit-a>/
  releases/<commit-b>/
  current -> releases/<verified-commit>
  shared-cache/
```

新投影先进入新的 release 目录，完成 import/smoke test 后再切换 `current`。旧 release 可短期保留用于快速回退；数据集、draft、annotation set 和媒体 cache 不放进 release 目录。

## 4. 为什么不推荐其他方案

| 方案 | 结论 | 原因 |
| --- | --- | --- |
| 完整仓库 + `.venv-annotation` | 现在可用的安全过渡方案 | 更新容易，但源码范围过大，仍受 import fan-out 影响 |
| 手工复制 `refinement_annotation/` | 不可用 | 会遗漏合同、reader、store、父包入口和静态资源 |
| 独立 wheel / `dexteleop_annotation_core` | 不作为主方案 | 容易形成第二个发布节奏、额外版本协调和过强封装 |
| 单独 annotation 分支 | 不推荐 | 容易和主分支长期漂移、反复 cherry-pick |
| Git submodule/独立仓库 | 不推荐 | 标注代码与合同更新仍要跨仓协调 |
| 主仓 sparse-checkout/source projection | 推荐 | 保持唯一源码、原路径和主仓 commit，同时便于频繁更新 |

## 5. 投影中的实际代码闭包

以下是概念闭包。实施时由 import closure 测试生成并锁定准确清单。

### 5.1 应用层

```text
control_station/refinement_annotation/
  __init__.py
  cli.py
  application.py
  server.py
  media.py
  static/index.html
  static/app.js
  static/styles.css
```

继续属于 `control_station`，不移动到新 namespace。

### 5.2 标注持久化层

```text
collection_runtime/immutable_canonical_store.py
collection_runtime/progress/stage_annotation_store.py
collection_runtime/progress/failure_trigger_store.py
dataset_interface/progress_stage_annotation_artifacts.py
```

这些文件仍保持当前 canonical 路径。标注 application 应从具体子模块导入，避免经过 `collection_runtime.progress` 聚合入口加载在线 controller/reducer/store。

### 5.3 只读数据适配层

```text
dataset_interface/progress_annotation_jobs.py
dataset_interface/progress_artifacts.py
dataset_interface/progress_timeline.py
dataset_interface/progress_samples.py
dataset_interface/progress_stages.py
dataset_interface/progress_failure_triggers.py
dataset_interface/sample_axis.py
dataset_interface/retrospective_progress.py
dataset_interface/retrospective_progress_artifacts.py
dataset_interface/episode_curation.py
```

不投影 prepared-training builder、训练视图、控制 trace、图像增强、策略 IO 和机器人动作适配。

### 5.4 纯合同层

```text
dataset_contracts/canonical_encoding.py
dataset_contracts/causal_snapshot.py
dataset_contracts/episode_curation.py
dataset_contracts/progress_facts.py
dataset_contracts/progress_profile.py
dataset_contracts/progress_stages.py
dataset_contracts/progress_stage_annotations.py
dataset_contracts/progress_failure_triggers.py
dataset_contracts/retrospective_progress.py
```

合同继续由 `dataset_contracts` 唯一拥有。projection 只是选择这些原文件，不复制实现、不引入 annotation 专用 schema。

## 6. 标注机器需要的数据

代码投影和数据访问必须分开。Python 源码投影中不包含数据集。

### 6.1 Native finalized dataset

每个可标注 Episode 至少需要：

```text
<dataset>/
  meta/episodes.jsonl
  meta/progress/episodes/<episode_uuid>/
    profile.cjson
    binding.cjson
    facts.cjsonl
    manifest.cjson
    stages.cjson
    stage_entry.cjson
  data/.../<episode>.parquet
  videos/<camera>/<episode>.mp4
```

当前程序继续写：

```text
meta/progress/stage_annotation_drafts/<episode_uuid>/draft.json
meta/progress/stage_annotations/<episode_uuid>/<annotation_set_id>/
  binding.cjson
  annotations.cjson
  failure_triggers.cjson        # 可选
  manifest.cjson
meta/progress/stage_annotations/.writer.lock
```

### 6.2 Retrospective workspace

```text
<workspace>/
  profile.cjson
  stages.cjson
  episodes/<episode_identity_hash>/
    source_binding.cjson
    progress_binding.cjson
    retrospective_window.cjson
    migration_manifest.cjson
```

显式 `--source-root` 指向的 source dataset 还需要：

```text
meta/info.json
meta/episodes.jsonl
<referenced parquet>
<admitted videos>
```

source dataset 保持只读，只有 workspace 的 `meta/progress/` 可写。

`.writer.lock` 是当前文件系统上的 advisory lock，不是跨机器成果；H.264 cache 也位于代码投影和数据协议之外，不迁移。

第一轮只处理代码便捷迁移，不同时引入独立 annotation output root 或跨机器成果合并协议。

## 7. 环境与启动方式

继续复用主仓已有文件：

```text
pyproject.toml
uv.lock
scripts/install_annotation_env.sh
README_ENV.md
```

入口不变：

```bash
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

聚合 import 清理完成后，重新审计 `annotation` extra。目标依赖是：

- Python `>=3.10,<3.11`；
- NumPy；
- PyArrow；
- 系统 `ffmpeg`/`ffprobe`；
- 现代浏览器。

SciPy、OSQP、Pinocchio、OpenCV、机器人 SDK、RealSense、MuJoCo、Torch 等不应进入最终 annotation projection 环境。是否删除依赖必须以 clean-venv import/test 为证据，不能只凭静态判断。

## 8. 分阶段实施计划

### P0：冻结现有行为，不移动文件

- 固定 native、retrospective、failure trigger、二次复核、视频 Range、BrokenPipe、草稿切换保护、catalog 增量刷新、lazy video、LRU 和 writer lock 测试；
- 准备最小 native/retrospective mock fixtures；
- 保存当前 import graph 基线；
- 不创建新 package，不改变入口。

验收：行为测试完整，只有测试/规划变化。

### P1：消除聚合 import fan-out

- 标注代码改用具体子模块导入；
- 将四个聚合 `__init__.py` 改成 side-effect-free 或 lazy compatibility export；
- 为现有包根公共 import 补兼容测试，避免影响其他开发；
- 增加 annotation import denylist；
- 在无机器人依赖的 clean venv 中导入 CLI。

验收：现有入口和协议不变；标注 import closure 不再出现机器人、控制、硬件、仿真、部署和训练模块。

### P2：建立 projection manifest 与闭包检查

- 在 `release/annotation_projection.toml` 维护投影清单；
- 工具根据入口计算 Python import closure；
- CI 校验 closure 被 include 完整覆盖且不触及 denylist；
- 显式加入静态资源、安装脚本、lock、文档和标注测试；
- 加入“从 projection 根运行而不是从完整 checkout 借文件”的测试。

验收：在临时目录中只放 projection 文件，标注 `--help`、native/retrospective `--dry-run` 和 GUI smoke 均通过。

### P3：提供两种更新方式

- 文档化 Git sparse-checkout，作为有仓库权限机器的默认方式；
- 提供 projection exporter，作为无 Git 权限机器的 fallback；
- exporter 只生成 staging/release，不编辑 remote source；
- 生成软件 receipt 并验证文件完整性；
- 版本化 release 通过验证后再切换，不在 live 目录叠加旧文件。

验收：新增、修改、重命名、删除一个标注文件后，两种更新方式均能得到一致目录；可以按 Git commit 回退。

### P4：目标标注机无设备回归

- 从 sparse checkout 或 projection release 安装 `.venv-annotation`；
- 检查 GUI 目录选择、数据集切换、草稿、封存、重开、二次复核、failure trigger、多视角、lazy load 和断连；
- 验证同一 dataset 只能有一个 writer；
- 验证 retrospective source 无写入；
- 不启动采集、机器人或硬件进程。

验收：标注相关测试与真实无设备 smoke 通过，软件 commit/receipt 可追溯。

## 9. 更新与回滚纪律

频繁更新时应遵守：

1. 只在主仓修改 canonical 标注代码；
2. 先通过标注行为测试和 projection closure 测试；
3. 标注机同步同一 Git commit 或该 commit 生成的 projection；
4. 依赖 lock 未变化时不重建环境；
5. 更新期间不覆盖正在运行进程所使用的 release；
6. smoke 通过后重启服务/切换 release；
7. 出现问题按 commit 回滚，不在标注机手改源码；
8. 数据、draft、annotation set 和 cache 始终独立于代码 release。

## 10. 重点风险

1. **聚合入口兼容**：lazy export 影响面较大，必须先盘点包根公共调用者并补测试。
2. **projection 漏文件**：由自动 closure 校验阻断，不能只靠人工 include 列表。
3. **静态资源漏投影**：HTML/JS/CSS、文档、安装脚本必须显式声明。
4. **旧文件残留**：Git sparse 自动处理；无 Git 模式使用新 release，避免 no-delete 叠加。
5. **源码分叉**：禁止独立 annotation branch/repo 中维护实现，标注机不接受长期手改。
6. **合同漂移**：projection 直接选择 `dataset_contracts` 原文件，不复制 schema。
7. **共享文件系统性能**：拆分不得退回全量 catalog 扫描或取消 lazy video/LRU。
8. **锁语义**：第一版仍是 Linux-only `fcntl`；不要为了跨平台静默取消互斥。
9. **并行工作树**：当前标注与 retrospective 仍有并行更新，实施时逐文件保留活动差异。

## 11. 当前实施状态与下一步

截至 2026-08-04，P0、P1、P2 已在当前工作树落地并完成定向验证：

- 标注入口使用具体子模块，四个历史包根通过 lazy compatibility export 保留公共 API；
- 当前标注 CLI 的本地 import closure 为 37 个模块，未进入 denylist；
- `annotation` extra 已缩减为 NumPy/PyArrow；
- `release/annotation_projection.toml` 是当前唯一投影清单；
- `scripts/check_annotation_projection.py` 会重新计算闭包并检查清单覆盖；
- projection 测试会在临时目录中验证入口帮助、loopback GUI 静态资源、native dry-run 和 retrospective dry-run，且拒绝回退读取完整工作树。

截至 2026-08-05，发布目标调整为优先提供可复制的本地源码投影。P3 的 exporter
已经落地：`scripts/export_annotation_projection.py` 从同一 manifest 生成新的发布目录，
写入逐文件完整性 receipt，并可在发布目录中独立复核；首个实体投影位于
`release/refinement_annotation/`。它仍不创建独立 package、不移动合同、不改变入口。

Git sparse-checkout 的专门操作文档仍可作为后续补充；当前无 Git 发布路径已经可用。
更新时 exporter 会拒绝覆盖既有目录，应生成新的版本目录并在验证后切换部署指向，
避免旧文件残留。
