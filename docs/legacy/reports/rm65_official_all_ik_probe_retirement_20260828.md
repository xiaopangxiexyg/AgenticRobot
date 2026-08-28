# RM65 官方全解研究探针退役说明（2026-08-28）

## 结论

在精确 `development` 基线
`8a3c2a84758dc6a19baf5b535f98016ed1c7045b` 上，
`scripts/probes/rm65_official_all_ik_probe.py` 是一个 1,227 行的一次性离线研究探针。
其研究结论已经进入
[Pose Lift 0727 实验报告](../../real/reports/POSE_LIFT_EXPERIMENTS_0727.md)，可执行能力也已由
项目内的 canonical 模块承接，因此从版本化源码树删除。

## 调用关系复核

退役前的全树静态引用复核发现两个 Python helper 消费方：

- `scripts/probes/rm65_qref_dynamics_probe.py` 使用左右工具标定路径和 `_pose_error`；
- `scripts/probes/rm65_teleop_smoothing_probe.py` 在离线碰撞复核中使用左右工具标定路径。

这些值和纯计算 helper 现由 `rm65_qref_dynamics_probe.py` 就地拥有，smoothing probe 从该
仍存续的离线探针导入。位姿误差计算保持原有平移范数与旋转角度语义。除此之外，没有
生产入口、测试、命令映射、Harness Profile 或 runtime 模块调用被退役脚本。

## 现行所有者

相关正式能力继续由以下模块拥有：

- `hardware.rm65.official_all_ik.RM65OfficialAllIK`：官方全解适配；
- `robot_motion.rm65.source_ik_selection`：source IK 候选选择；
- `teleop_runtime.position_control.official_all_ik_governor`：运行时 governor；
- `teleop_runtime.debug.view_rm65_pose_lift_sim_replay`：离线 MuJoCo replay 入口。

## 历史与安全边界

- 完整实验数字、被否决路线和当时结论继续保存在版本化报告中；
- 根 `legacy/` 保持严格本地、完全忽略，本次不在那里保留 tracked 代码副本；
- 如需逐行复核历史实现，应从 Git 精确提交离线读取；
- 本次变更不连接设备、不 dispatch、不启动仿真器，也不构成硬件或科学结论重新认证。
