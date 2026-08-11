# 跨层端到端指南

本目录保存已经落地、但横跨数据契约、训练接口和部署运行时的贯通说明。它们用于理解
一项能力如何穿过多个 owner，不是新的生产入口，也不替代各领域 `README.md`。

| 指南 | 覆盖范围 | 当前入口 |
|---|---|---|
| [双臂 Hold action label](./ACTION_LABEL_ARM_HOLD_TRAINING_AND_DEPLOYMENT.md) | 采集标签、训练 target、模型输出和部署状态机 | 真机命令仍从 [`../real/README.md`](../real/README.md) 进入 |

新增指南时必须明确每一层的权威 owner、公开合同和验证边界；单模块参数仍写回对应 package
README，实验结果写入所属领域的 `reports/`。
