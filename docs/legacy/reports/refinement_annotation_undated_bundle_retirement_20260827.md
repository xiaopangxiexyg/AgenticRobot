# 未日期 refinement-annotation bundle 退役记录

## 决定

2026-08-27 从版本化源码树删除 `release/refinement_annotation/`。该目录是
2026-08-07 的未日期发布投影，不是 annotation 运行时或数据合同的权威源。
已删除内容不复制到根 `legacy/`；该根仍严格本地化并完全忽略。

## 退役依据

在 AgenticRobot `development` 精确提交
`3c40214883b29619e13282dbd10cd99b6a7551b6` 上执行：

```bash
cd release/refinement_annotation
python scripts/export_annotation_projection.py --verify .
python scripts/check_annotation_projection.py
```

receipt 验证因
`docs/real/LONG_HORIZON_STAGE_ANNOTATION.md` 与记录摘要不同而失败；import closure
检查仍通过。因此该目录是结构仍可读、但已不能证明来源完整性的失真快照。
其 43 个 Python 文件共 19,826 行，与已保留的发布投影重复。

同一提交上两个日期化快照均通过 receipt 和 closure 自验证：

- `release/refinement_annotation_20260807_39895a67ef52/`
- `release/refinement_annotation_20260808_c959d1d38695/`

## 当前替代

- canonical 投影清单：`release/annotation_projection.toml`
- canonical 检查器：`scripts/check_annotation_projection.py`
- canonical 导出器：`scripts/export_annotation_projection.py`
- 当前已审阅快照：`release/refinement_annotation_20260808_c959d1d38695/`

临时交付候选应生成到已忽略的 `dist/` 下的新空目录，不再重用无版本身份的
`release/refinement_annotation/`。需要新增 tracked 快照时，目录名必须带日期和 source hash，
并在提交前执行 receipt 与 closure 验证。

## 功能边界

本次退役不修改 `control_station/refinement_annotation/`、`dataset_contracts/`、
`dataset_interface/` 或 annotation 数据。GUI、HITL、Episode 生命周期、媒体路径和部署
入口的运行语义均不变。
