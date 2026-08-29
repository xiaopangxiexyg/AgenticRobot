# RM65 速度平滑研究探针退役说明（2026-08-29）

## 结论

在精确 `development` 基线
`ad6ec700a6b2dfb481160bb676ad56327f5c5bb0`（tree
`00a13056405ef3a1eae39921ea5dfc2da24cddf6`）上，下列文件组成一条封闭的离线研究链：

- `scripts/probes/rm65_teleop_smoothing_probe.py`：2,624 行；
- `scripts/probes/rm65_vive_velocity_execution_probe.py`：1,240 行；
- `tests/test_rm65_vive_velocity_execution_probe.py`：127 行。

共 3,991 行。它们没有生产、文档、命令映射或 Harness Profile 调用者，也没有版本化的
输入 manifest、输出 evidence 或结论报告。继续保留这条未晋升的实验实现会与现行 RM65
执行边界并列维护，因此从版本化源码树删除，而不是机械拆分或迁移到根 `legacy/`。

## 调用关系与行为复核

退役前的 tracked 全树静态引用只有：

1. VIVE velocity 探针从 smoothing 探针导入候选评估、轨迹模拟和统计 helper；
2. 专属测试通过文件路径动态加载 VIVE velocity 探针；
3. 既有官方全解探针退役说明记录 smoothing 探针曾使用工具标定路径。

两个 CLI 都是只读离线入口：它们消费 `command_trace.npz`，不初始化设备、不 dispatch
机器人命令。受支持的 `dexteleop310` 环境中，两个 `--help` 均成功，专属测试在删除前为
`3 passed`。这些结果证明历史代码在其限定环境中可加载，不构成正式运行时入口或保留理由。

VIVE-derived Cartesian twist、DLS 映射和 velocity-refresh 候选从未进入生产调用图；仓库也
没有保存能够把它们提升为正式能力的版本化实验输入或结论。本次删除的是该未晋升假设及
其自测，不是移除线上 VIVE、IK 或 RM65 位置控制能力。

## 现行所有者

正式执行能力继续由以下活跃模块拥有：

- `teleop_runtime/backends/arms/rm65.py`：RM65 runtime backend，并实际构造
  `SampleAlignedJerkLimitedJointTrajectoryPlanner` 与 `JointHorizonLimits`；
- `robot_motion/position_control/sample_aligned_trajectory.py`：sample-aligned、jerk-limited
  关节轨迹规划；
- `robot_motion/position_control/horizon_governor.py`：位置、速度、加速度和 jerk horizon
  约束及 workspace；
- `teleop_runtime/position_control/official_all_ik_governor.py`：运行时 official all-IK
  governor；
- `scripts/probes/rm65_qref_dynamics_probe.py`：仍存续的 qref 离线动力学研究入口，并继续
  拥有工具标定路径与位姿误差 helper。

相应正式能力由 `tests/test_robot_motion_sample_aligned_trajectory.py`、
`tests/test_robot_motion_position_horizon_governor.py` 和
`tests/test_official_all_ik_governor.py` 独立验证，不依赖本次删除的研究探针。

## 精确身份与恢复

删除前的 Git blob 与 SHA-256 为：

| 路径 | Git blob | SHA-256 |
|---|---|---|
| `scripts/probes/rm65_teleop_smoothing_probe.py` | `e213d97ebbd9285432cb14ce51f2a8c8f6af307c` | `fc9f4d7ee8354fdcbbd9a3f0ee15345b27ef39ac32c32ec165bffaf725b3236f` |
| `scripts/probes/rm65_vive_velocity_execution_probe.py` | `17ad68b1b3b04aaa03df3329eaed1412b2aaa06e` | `1bf2454f08f6beca1a80058391e816933915eaf2e764e9c5259ad6840d920d06` |
| `tests/test_rm65_vive_velocity_execution_probe.py` | `62a0a7f417244b10fd0e587fb8803156e8e406a9` | `2bfb549a7ca0d56a806b02649522b07edb98ceb276c79a51f5fff27b6429210f` |

如需历史复核，可从上述基线精确读取，例如：

```bash
git show ad6ec700a6b2dfb481160bb676ad56327f5c5bb0:scripts/probes/rm65_teleop_smoothing_probe.py
git show ad6ec700a6b2dfb481160bb676ad56327f5c5bb0:scripts/probes/rm65_vive_velocity_execution_probe.py
git show ad6ec700a6b2dfb481160bb676ad56327f5c5bb0:tests/test_rm65_vive_velocity_execution_probe.py
```

根 `legacy/` 继续严格本地、完全忽略；本次没有在其中保留 tracked 副本。整个审计和验证
过程不连接设备、不启动仿真器、不执行运动，也不重新认证硬件或历史实验结论。
