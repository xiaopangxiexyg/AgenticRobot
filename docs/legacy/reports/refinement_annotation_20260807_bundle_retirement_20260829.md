# refinement-annotation 日期化 bundle 退役记录

## 决定

2026-08-29 从版本化源码树删除
`release/refinement_annotation_20260807_39895a67ef52/`。该目录是已经被
2026-08-08 source projection 取代的回滚发布快照，不是 annotation 运行时、GUI、
数据合同或投影工具的权威源。

已删除内容不复制到根 `legacy/`；该根继续保持严格本地、完全忽略。旧快照由 Git
对象和本记录保存可恢复身份。

## 退役前验证

验证基线为 AgenticRobot `development`：

`e9b90e90e5b68fb9895b670af53a8a76a94bcdb1`

退役前在旧快照根执行：

```bash
python scripts/export_annotation_projection.py --verify .
python scripts/check_annotation_projection.py
```

结果为 receipt 验证通过（54 个 source files，release hash 前缀
`0e8117750d6f`）和 import closure 通过（37 个 modules）。同一基线上的当前
2026-08-08 快照也通过 receipt 验证（58 个 source files，release hash 前缀
`734a048108db`）和 import closure（40 个 modules）。

旧快照包含 56 个文件、27,136 个物理行，其中 43 个 Python 文件共 19,826 行，
tracked blob 合计 1,152,199 bytes。与 2026-08-08 快照比较，56 个共享路径中
47 个内容逐字相同；旧快照没有独有路径，后继快照另外包含 4 个 recoverable-failure
相关路径。版本化树中没有生产 import、测试调用或运行入口依赖旧日期目录。

## 精确恢复身份

- 旧 bundle tree：`bf1d36396c4b6872555b0909be46b2263168724d`
- source projection hash：
  `39895a67ef52f0fc3f82507bb7ef6b1a2d3d49d402e62bf2da07de8a966e0f34`
- release hash：
  `0e8117750d6f0bb3219cba79e6c560d6bbded3df50f5b1e25423e42e479893ce`
- 后继 bundle tree：`69e86fae316a3a4298c7f96a178c58e065d48865`
- 后继 release hash：
  `734a048108db902535f591a21372120f586149fbda356abff99f385897d4fb75`

需要审计或恢复旧投影时，应从上述 Git tree 导出到新的临时目录，而不是把副本重新
加入当前源码树。例如：

```bash
mkdir -p /tmp/refinement-annotation-20260807-recovery
git archive --format=tar \
  --prefix=refinement_annotation_20260807_39895a67ef52/ \
  bf1d36396c4b6872555b0909be46b2263168724d \
  | tar -xf - -C /tmp/refinement-annotation-20260807-recovery
```

## 当前替代

- 当前已审阅 bundle：
  `release/refinement_annotation_20260808_c959d1d38695/`
- canonical 投影清单：`release/annotation_projection.toml`
- canonical 检查器：`scripts/check_annotation_projection.py`
- canonical 导出器：`scripts/export_annotation_projection.py`
- disposable handoff 输出：严格位于已忽略的 `dist/` 新目录

## 功能边界

本次退役不修改 `control_station/refinement_annotation/`、`collection_runtime/`、
`dataset_contracts/`、`dataset_interface/`、annotation 数据、当前 release bundle 或
任何部署入口。GUI、媒体、Episode 生命周期、HITL、仿真、硬件及数据持久化语义均不变。

## 2026-08-31 后续：最后版本化投影退役

### 决定

2026-08-31 继续从版本化源码树删除
`release/refinement_annotation_20260808_c959d1d38695/`。该目录也是由
canonical 源码生成的便携交付投影，不是 annotation GUI、数据合同、
持久化逻辑或导出工具的权威源。项目不再保留任何版本化的生成投影；
需要交付时，由 canonical manifest 和 exporter 在已忽略的 `dist/` 或仓库外
临时目录中按需生成。

删除内容不复制到根 `legacy/`。该根继续保持严格本地、完全忽略；
历史身份由 Git 对象和本记录保存。

### 退役前验证

任务基线为 AgenticRobot `development`：

`e994088891e1b449be4d50a6cf62a7713795a5f2`

退役前对版本化投影执行：

```bash
python scripts/export_annotation_projection.py \
  --verify release/refinement_annotation_20260808_c959d1d38695
python scripts/check_annotation_projection.py --json
```

receipt 验证通过：58 个 source files、59 个 receipt 文件记录，release hash
`734a048108db902535f591a21372120f586149fbda356abff99f385897d4fb75`；当前
canonical import closure 检查也通过。被删除目录共 60 个 tracked files、29,167
个物理行。

同一干净任务基线随后通过 canonical exporter 输出到仓库外临时目录，并立即执行
receipt 自校验。新投影包含 71 个 source files、72 个 receipt 文件记录，证明
当前源码仍能生成完整、可验证的独立交付物。最终候选的 source
projection hash 为
`4e62f41dbbf6aa7f512b72eaf8954f41eba76cdfdff2f66af6f46076c1326388`，
release hash 为
`d1c9d5452cc70e0461f26d7ba16d8b6b0de278d3f6d5ae78fe72142691890a72`。
该临时输出不是版本化候选。

### 精确恢复身份

- 退役 bundle tree：`69e86fae316a3a4298c7f96a178c58e065d48865`
- source projection hash：
  `c959d1d38695721dee7f48143ea6f81e893f31c463862945d059017b53e8b5af`
- release hash：
  `734a048108db902535f591a21372120f586149fbda356abff99f385897d4fb75`

需要审计或恢复该历史投影时，应导出到新的临时目录，不要重新加入当前源码树：

```bash
mkdir -p /tmp/refinement-annotation-20260808-recovery
git archive --format=tar \
  --prefix=refinement_annotation_20260808_c959d1d38695/ \
  69e86fae316a3a4298c7f96a178c58e065d48865 \
  | tar -xf - -C /tmp/refinement-annotation-20260808-recovery
```

### 当前替代

- canonical 投影清单：`release/annotation_projection.toml`
- canonical 检查器：`scripts/check_annotation_projection.py`
- canonical 导出器：`scripts/export_annotation_projection.py`
- canonical 回归：`tests/test_annotation_projection.py`
- disposable handoff：忽略的 `dist/` 新目录或仓库外临时目录

### 功能边界

后续退役仍不修改 `control_station/refinement_annotation/`、`collection_runtime/`、
`dataset_contracts/`、`dataset_interface/`、annotation 数据或任何运行入口。GUI、
媒体读取、Episode 生命周期、HITL、仿真、硬件和数据持久化语义均不变。变化只限于
不再把可重建的源码投影作为 29,167 行版本化副本保存。
