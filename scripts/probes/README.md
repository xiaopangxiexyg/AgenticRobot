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
