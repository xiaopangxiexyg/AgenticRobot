# Replay 参数说明

Pose→QPOS 各版本及 0727 离线结果汇总见
[POSE_LIFT_EXPERIMENTS_0727.md](./POSE_LIFT_EXPERIMENTS_0727.md)。

Replay 重放已有数据；运行命令见 [real README](./README.md)。离线浏览与真机执行使用同一
GUI 入口，但只有显式提供真实 backend 并经过人工放行才允许运动。

## 1. Replay GUI：离线浏览（R1）

可直接复制的命令见 [real README 的第 2 节](./README.md)。浏览器访问
`http://127.0.0.1:8765/`。默认只允许 loopback；不要为方便而常开
`--unsafe-allow-remote-web`。

完整 GUI、token、视频/触觉显示说明见
[`teleop_runtime/replay/README.md`](../../teleop_runtime/replay/README.md)。

## 2. 真机 Replay（M）

真机 replay 仍使用同一个入口，但必须显式提供 active sides、RM65 IP、左右 tool profile
以及需要的 end-effector backend。例如 DMTAC 配置骨架：

```bash
set -euo pipefail
: "${DATASET_ROOT:=datasets}"
: "${LEFT_ARM_IP:=192.168.101.21}"
: "${RIGHT_ARM_IP:=192.168.1.20}"
export DATASET_ROOT LEFT_ARM_IP RIGHT_ARM_IP

python -m teleop_runtime.replay \
  --dataset-root "$DATASET_ROOT" \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --end-effectors-enabled \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --require-dmtac-config \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

上面的默认值可在命令前覆盖；启动前还要人工检查所选 episode 的 control mode 与当前
backend 一致。这个骨架不会替你判断轨迹是否适合当前工作空间。

启动服务不等于批准运动。先在 GUI 检查 dataset/episode/control mode，执行 `Move Start`
安全初始化，确认姿态后再人工 `Start`。Position replay 使用数据集 main action，不重复 live
teleop 的 pre-action 平滑；主 action 之后的 IK、安全限位和下发仍保留。

### 两种 Position replay

- `--replay-mode pose`（默认）：`Move Start` 仍使用首帧保存的 accepted q_ref 做 QPOS
  归位；`Start` 后读取保存的 pose action，使用与 deploy 相同的
  `--pose-lift-profile`、双臂 IK 和 admission 链路。它用于检查策略部署时 Pose→QPOS
  复现是否合理，不承诺逐关节严格等于演示。
- `--replay-mode qpos`：`Move Start` 语义不变；`Start` 后严格发送保存的 accepted q_ref。
  该路径不做 IK、不 unwrap、不做 wrist canonicalization，也不修改目标；仍执行绝对限位、
  reference 速度/步长和双臂 endpoint collision 准入。任何准入失败都会 hold/fault，而不是
  自动改写 qpos。

Pose replay 与 deploy 现在默认使用
`rm65_pose_lift_v4_qp_constrained`。结构调整前的
`rm65_pose_lift_v1_baseline` 仍是显式原子回退；Pose replay 与 deploy 必须使用同名
profile，不能只在一侧切换。

`rm65_pose_lift_v3_bounded_continuous` 是后续离线验证 profile：它保留 V2 的合法
unwrap，但把腕部连续性修正改成左右臂独立的确定性有界搜索。候选必须相对原始 IK
保持 pose 等价，并且不能显著恶化原始 IK 已有的目标残差。V3 还使用经 0727
runtime-causal 轨迹扫描验证过 pose/qref parity 的 `ik_posture_cost=0`，避免默认 0.1
posture task 在持续 Pose 跟踪中积累目标滞后；硬关节、步长和碰撞准入不变。该结果不等于
奇异安全验收：0727 的近奇异 episode 中，V3 虽然降低了 qref/FK 复现误差，但个别右臂
样本的归一化 Jacobian 最小奇异值略降、condition number 略升。因此 V3 仍是显式 opt-in
实验 profile，不能替代 baseline 默认值，也不能仅凭关节误差改善用于真机放行。V1/V2 的
语义、metadata 和 fingerprint 不因 V3 改变；V3 验证不通过时仍直接选择 baseline 回退。

`rm65_pose_lift_v4_qp_constrained` 在 V3 上增加 admitted-anchor Pink QP 速度/加速度约束，
实现“先受限求解、再由独立 admission 认证”。0727 episode 16/35/80/81 的离线验证未发现
奇异安全指标回退，目标追踪总体持平或改善，qref 加速度与 jerk 峰值下降；因此 V4 已提升为
deploy/Pose replay 默认并允许经过既有 preflight 的真机输出。选择 V1 可完整回退。

`rm65_pose_lift_v5_sigma_escape` 与 `rm65_pose_lift_v6_structural_branch` 仅用于离线研究。
V5 增加 sigma-min 软逃逸目标；V6 在其上增加 q3/q5 零点单侧分支锁。两者的 profile metadata
均明确 `hardware_dispatch_allowed=false`，部署或真机 dispatch 会拒绝它们；V4 默认和真机命令
不受影响。V6 适合用 episode 96 等结构奇异反例检查“禁止穿零、保留 qdot/qddot/collision
硬约束”的效果，但可能在极深奇异区形成有限单侧平台，不能直接当作跟手性已经解决。

`rm65_pose_lift_v7_structural_escape` 是在 V6 单侧锁上增加 q3/q5 软外移目标的离线反例探针。
它在 episode 96 能进一步降低局部 qref 误差和深奇异暴露，但在 0727 的 8 条 guarded
episode 全局门禁中出现加速度/速度证书回退、parity 回退或 sigma 改善不足，因此保持
`hardware_dispatch_allowed=false`，不得用于部署或真机。V6 是当前离线结构保护基线，V4
仍是默认 profile。

离线比较任意两个 Pose Lift profile 时使用只读 evaluator；它不连接设备、不下发动作，并分别报告
teacher-forced、closed-loop 和 runtime-causal 结果。三种模式都先把保存的 15 Hz pose/qref
重采样到指定部署 reference clock（默认 60 Hz）；区别只在 planner state 是否由上一条
truth 强制重置，不能让 time-aware QP 在 15 Hz 调用间隔中错误地只获得一次 60 Hz 预算：

```bash
python -m teleop_runtime.debug.evaluate_rm65_pose_lift_parity \
  --no-motion \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --pose-lift-profile rm65_pose_lift_v4_qp_constrained \
  --compare-baseline-profile rm65_pose_lift_v3_bounded_continuous \
  --output-json /tmp/rm65_pose_lift_parity.json
```

evaluator 的关节真值始终是数据集保存的 accepted q_ref，不使用 `q_ik_raw` 猜测演示分支。
除关节/FK 误差外，它还报告按固定 characteristic length 归一化的 Jacobian `sigma_min` 与
condition number、q3/q5 结构性奇异裕度、向/离奇异方向计数、q4/q6 null redistribution 与
branch jump，以及 admitted qdot/qddot/jerk。Jacobian 指标只允许在同一模型、TCP、行归一化
和 characteristic length 下横向比较；工具不会臆造一个通用 condition-number 硬阈值。

MuJoCo replay 不属于本页；canonical 与 direct-qpos debug 的边界统一放在
[仿真说明](../sim/MUJOCO.md)。
