# Position A/B 离线分析器退役记录

## 决定

2026-08-28 从版本化源码树删除
`teleop_runtime/debug/analyze_position_ab.py`。该 1,105 行模块是 2026-07 位控
capture 的多层离线分析入口，不是真机遥操、采集、安全、IK 或 dispatch 运行时的
权威 owner。删除的代码不复制到根 `legacy/`；Git 历史仍可恢复原始实现。

## 退役依据

在 AgenticRobot `development` 精确提交
`82547c195dad305f0670d004c2d07be9d0941c50` 上确认：

- 全仓库没有生产 import、测试 import、脚本调用或文档命令引用该模块；
- 模块仅在 2026-07-15 和 2026-07-17 两个历史提交中修改，之后未跟进当前位控与 replay 边界；
- 原始生成的结果已保存为
  `docs/legacy/reports/position_ab_report.md`，该文档已明确标记为历史离线证据；
- 当前命令清单登记了更聚焦的分项评估器，而没有登记该旧入口。

## 当前替代

| 旧职责 | 当前入口 |
|---|---|
| 对齐 `position_reference.jsonl` 与已接受 `position_command_150.jsonl`，查看 target→qref→qcmd 根帧误差 | `python -m teleop_runtime.debug.analyze_position_root_x --control-debug-dir <capture>` |
| 使用数据集已接受 q_ref 作为 teacher，评估当前 Pose Lift、teacher-forced 和 closed-loop parity | `python -m teleop_runtime.debug.evaluate_rm65_pose_lift_parity --no-motion ...` |
| 对当前 Active 位置顺应 pre-IK 管线做离线回放 | `python -m teleop_runtime.debug.replay_position_admittance_active_pipeline --log <action-debug.jsonl>` |
| 查看 2026-07 capture 的已有 A/B 结果 | `docs/legacy/reports/position_ab_report.md` |

上述三个 Python 入口在该精确 `development` 归档上的 `--help` 均返回 0，且都是离线入口。

## 功能边界

本次删除不修改 `teleop_runtime/position_control/debug_recorder.py`，因此
`position_reference.jsonl` 和 `position_command_150.jsonl` 的生成语义不变；也不修改
action-debug 记录、RM65 模型、IK tuning、真机入口、采集、安全或命令派发。
