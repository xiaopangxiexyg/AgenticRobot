# Shared probes and maintenance tools

This directory contains versioned, non-canonical operator tools migrated from
the former Git-tracked `todos/probes/` directory.  Keeping executable tools
here prevents personal planning state from becoming a runtime dependency.

These scripts have mixed risk levels.  Some are read-only audits, while others
can train models, connect to services, transform datasets, or execute an
explicit `--apply` path.  A filename containing `audit`, `probe`, `verify`, or
`smoke` is not proof that a command is read-only.  Inspect the exact script and
its `--help`, identify its source and target paths, and follow the owning
dataset, deployment, simulation, or hardware safety rules before execution.

They are not canonical teleoperation, collection, replay, or deployment
entrypoints.  Those remain documented under `docs/` and in the repository root
`README.md`.

Experiment-specific learner, loss, training, and evaluation implementations
belong under `experiments/`. A launcher kept here for compatibility must stay
thin and delegate to that owner.

## 冻结观测的双臂抓姿重放（纯离线）

`replay_bimanual_grasp_snapshot.py` 加载已留存 RGB-D、mask、qpos 及标定，
`--sides left right` 共用一次感知推理和物理候选池，各臂随后独立做局部链及路径准入。
`shared_candidates.json` 保留共享池与组件诊断；逐臂结果不代表执行或抓取成功。
原单臂入口保留，模型路径和输入要求见该工具 `--help`。它不创建真实硬件 backend；
本次集成验证使用已留存输入或推理替身，没有启动模型服务。

## DMTAC 连续开合包络对照（纯离线）

`compare_rm65_dmtac_sweep.py` 只读加载已审核 CAD 与 sparse proxy URDF，使用保存的
`observation.json` 双臂 qpos，以及可选的离线端点诊断结果。它不启动设备、不覆盖资产，
也不注册或切换生产碰撞 profile：

```bash
python -m scripts.probes.compare_rm65_dmtac_sweep \
  --cad-urdf assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf \
  --proxy-urdf assets/rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf \
  --observation /path/to/frozen/observation.json \
  --output /path/to/new-owned-output
```

依赖已有 SciPy、Pinocchio/FCL、trimesh 环境。工具验证两级反向转动的 mimic 关系、轴、
零偏和关节范围后，由圆弧弦高上界构造连续全开合保守凸包；密集采样仅作实现复核，
不是连续覆盖证明。另以独立 CAD FK 检查坐标，并确认传感器网格被已审核盒体覆盖。
生产碰撞模型仍使用原 box；诊断克隆只替换 terminal geometry，保留原有 20 对检查
及 20 mm 净空要求。这里不包含完整 planner 的附加 D405/点云障碍检查。

CAD 与 proxy 对照必须使用同一个固定场景。输出先列出桌面/相机杆的原始位置和尺寸
差异，然后仅在诊断克隆中把 CAD 固定场景对齐到 proxy；不能把不同桌面高度下的
距离差解释为 proxy 误差。CAD 距离的 101 个开合样本不构成连续距离认证；凸包覆盖
也不证明路径、抓取或真机可执行性。未经几何和物理场景复核，不将诊断凸包自动上线。


只读留存检查可使用 inspect_safe_move_snapshot.py，直接加载安全移动尝试的
snapshot.json 与同批 RGB-D/FK，不重新连接设备或重算采集位姿。输出目录必须不存在；
报告只提供观测表面、碰撞点投影与离线几何证据，不代表路径执行或抓取成功。


## 冻结局部关节路径的分段离线诊断

`python -m scripts.probes.validate_grasp_pipeline --help` 提供独立的离线验证工具。
它消费已经选定的两阶段关节路径，输入必须含 `pregrasp_target`、`constrained_approach`、
`stage_boundary_stops` 和两段动力学证书；不会重新搜索抓姿或 IK，也不会改变当前 whole-plan 默认。
调用方须保证点云、目标标签、夹爪几何与 MuJoCo contract 来自同一冻结 solver-world 场景。
工具会核对留存路径、原 waypoint 与当前碰撞 profile 身份并拒绝不匹配；外部点云、几何与
MuJoCo contract 之间没有自动交叉 hash 绑定，不能据此声称所有输入身份均已自动验证。

工具复用现有 motion port、C2 补接、碰撞与可见点云复核、不可变计划 store 及 force monitor，
检查前段执行、确认 HOLD、连续停稳、新后段授权 fixture 和一次性消费。worker 仅持有本地规划模型；
力输入是零力模拟源，MuJoCo 通过 qpos/forward 做运动学跟随。输出写入指定的新目录。
这里没有硬件 backend、SDK、相机或模型服务启动，结果不构成执行器跟踪、制动或物理抓取验收。

`--budget` 用于规划与后段采纳等待；它不是强制终止底层进程的硬时限。
`Future.cancel()` 无法取消已运行的原生工作，进程池关闭会等待这类工作结束；应在受控 CPU 资源中运行。
冻结状态的精确起点比较和模拟确认不提供真机恢复策略，也不新增生产 staged Skill 或真实 Route/grant。
失败退出时分别关闭 force monitor 与 plan store；优先保留首个中断，否则保留首个普通异常，
附带其他关闭错误的完整信息，不将关闭失败报告为成功。
