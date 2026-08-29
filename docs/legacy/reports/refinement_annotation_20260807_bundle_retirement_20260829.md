# 2026-08-07 refinement-annotation bundle 退役记录

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
