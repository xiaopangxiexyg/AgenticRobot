# 仿真运行入口

MuJoCo 是 task-physics 主链路；SAPIEN 只做 preview。详细说明见 [MUJOCO](./MUJOCO.md)。

从仓库根目录、已激活 `dexteleop310` 的桌面会话执行。带 `--viewer` 的命令需要可用显示环境；
无显示服务器先使用第 1 节的只编译检查。

## 1. 编译 canonical profile（R1）

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

## 2. Real-like VIVE position teleop（R1）

先按 [真实系统入口](../real/README.md) 启动 VIVE sender，并让它发送到运行本命令的主机。

```bash
PYTHONPATH=. python -m teleop_runtime.debug.run_vive_mujoco_teleop \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint \
  --viewer
```

## 3. Canonical dataset replay（R1）

```bash
python -m sim.mujoco.replay \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --episode 0 \
  --viewer
```

需要验证 replay-authoritative 的 sim human-in-the-loop capture 时，使用独立的
[HIL 使用说明](../HIL/README.md)。当前 operational profile 是显式、default-off 的
单右臂 PPO + passive VIVE/gripper → MuJoCo 应用；同一文档另行保留历史
replay-authoritative CS-6b1f 验证。它与 RTC 和真机机械臂 deployment 分开。

## 4. Agent TCP-pose Gateway（R1，MuJoCo only）

该入口只启动 DexTeleop 自有的 MuJoCo、下层 motion owner 与本地 Unix Gateway；
不启动 RPent、Planner 或真机设备。socket 父目录必须预先存在、不是符号链接且不可
world-writable；已有 socket 不会被自动删除。历史数据显式落在调用方选择的本地目录。

```bash
export RPENT_ROOT=/absolute/path/to/RPent
mkdir -p "${RPENT_ROOT}/temp/runtime/sim_gateway"
chmod 700 "${RPENT_ROOT}/temp/runtime/sim_gateway"

python -m agent_gateway mujoco \
  --socket-path "${RPENT_ROOT}/temp/runtime/sim_gateway/gateway.sock" \
  --task-config "${PWD}/sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json" \
  --client-id rpent.dexteleop \
  --history-root "${RPENT_ROOT}/temp/runtime/sim_gateway/history"
```

历史命令 `python -m control_station agent-gateway-mujoco ...` 仅保留为薄兼容转发；
Gateway 实现、Skill 定义和 MuJoCo executor 均由顶层 `agent_gateway` 包持有。

ready JSON 会发布 exact target/mode/session/domain 与 Route、station、execution
manifest hashes。RPent 通过同一个 socket attach；未来 real deployment 使用另外的
DexTeleop lower launcher，但上层继续消费同一 Gateway/Skill/Program 接口。退出该进程
统一调用 Gateway close，由下层完成 HOLD；上层无需解释或声明 measured-stop 细节。

当前 MuJoCo motion Route 还会发布一个下层自有、内容寻址的 `standard` TCP-pose
safety envelope。其 absolute workspace、相对位移和相对旋转数值沿用该 Route 已有的
正式限制；`strict` / `relaxed` 不会由上层缩放生成，须待独立审查后再由下层发布。
scene/qpos 初始化完成后，下层从 measured tool TCP 提交稳定的
`SafetyReferenceEpoch`。每次主动 motion SkillAttempt 都先在下层 HOLD 边界绑定所选
envelope 与当前 execution epoch，再开放 READY；普通 HOLD 只撤销本次 operation
绑定，不重设稳定 reference，因此不会因连续规划形成 relative-anchor ratchet。scene
reset 或重新完成物理初始化时，必须由下层先失效旧 reference，再提交新的 measured
reference。最终 TCP-pose filter 位于 pose transform 之后、共享 IK 之前；Agent 只携带
已审查 profile/envelope 的不透明身份，不拥有数值安全、HOLD、IK 或 dispatch 逻辑。
