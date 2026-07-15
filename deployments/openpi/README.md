# OpenPI wrist-only 部署

本目录连接 OpenPI 推理服务与 DexTeleop 真机部署运行时。OpenPI 仅负责模型加载和 websocket 推理；相机、机械臂、夹爪、初始化、限幅和力控均由 DexTeleop 负责。

当前经过闭环 server/client 握手验证的模型是：

| 项目 | 值 |
| --- | --- |
| server model id | `0712-wrist-cube` |
| checkpoint | `checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599` |
| artifact | checkpoint 下的 `policy_artifact.json` |
| 图像 | `left_wrist`、`right_wrist`（RGB） |
| state / action | 16D / 16D，连续 DMTAC gripper 标量 |
| server schema | `dexteleop_deploy_v2` |

`92599` 是当前保存的训练 checkpoint，并不表示 150k 训练已完成或已通过任务成功率评估。部署前应先完成无硬件预检和 dry-run。

## 1. 不可变契约

`policy_artifact.json` 是 checkpoint 的唯一 I/O 权威。客户端传入 `--policy-artifact-json` 后会强制校验：

- server model id 和 checkpoint step；
- v2 schema、16D state/action、layout hash；
- 左右 wrist 的逻辑相机键、OpenPI 输入键和 server slots；
- DMTAC 的连续 gripper 映射。

不要手动将 wrist 图像伪装成 `front`，也不要用位置顺序猜测相机。`0712-wrist-cube` 的 registry key 必须与 artifact 内的 `model_id` 完全相同；否则 websocket 握手会 fail-closed。

服务端 registry 位于 `~/zc/openpi/configs/dexteleop_policy_registry.json`，必须包含：

```json
"0712-wrist-cube": {
  "config": "0712-wrist-cube",
  "checkpoint_dir": "checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599",
  "deploy_schema_version": "dexteleop_deploy_v2",
  "action_key": "actions",
  "action_discrete_key": "actions_discrete",
  "action_dim": 16,
  "action_discrete_dim": 0,
  "state_dim": 16,
  "state_discrete_dim": 0,
  "action_horizon": 50,
  "active_sides": ["left", "right"],
  "camera_slots": ["left_wrist_0_rgb", "right_wrist_0_rgb"]
}
```

## 2. 两套环境

不要混用环境：

| 机器/职责 | 根目录 | 环境 |
| --- | --- | --- |
| OpenPI server / GPU | `/home/user/zc/openpi` | `uv` / `.venv` |
| DexTeleop client / 硬件 | `/home/user/zc/DexTeleop` | `/home/user/miniconda3/envs/dexteleop310/bin/python` |

以下变量应在 **DexTeleop client 机器**上设置；若 server 与 client 是同一台机器，`SERVER_IP=127.0.0.1` 仅适用于本机检查，局域网部署应使用可路由 IP。

```bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export OPENPI_MODEL=0712-wrist-cube
export OPENPI_ARTIFACT=$OPENPI_ROOT/checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599/policy_artifact.json

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

```

## 3. 启动 OpenPI server

在 GPU/server 机器执行。`PYTHONPATH` 必须指向 DexTeleop，因为 OpenPI 的训练配置会加载共享的数据集接口。

```bash
cd /home/user/zc/openpi
export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model 0712-wrist-cube \
  --host 0.0.0.0 \
  --port 8078
```

server 应保持在该终端运行。另开一个终端确认 health 和 metadata：

```bash
curl --fail --silent http://127.0.0.1:8078/healthz
curl --fail --silent http://127.0.0.1:8078/metadata
```

metadata 至少应显示：

```text
model_id=0712-wrist-cube
deployment.checkpoint_step=92599
deploy_schema_version=dexteleop_deploy_v2
local_robot.output_action_dim=16
dataset_adapter.camera_slots=[left_wrist_0_rgb,right_wrist_0_rgb]
```

## 4. 无硬件 websocket 预检

此命令只连接 server、读取 metadata 并验证 artifact；不打开相机、不连接机械臂或夹爪、不发送动作。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -c '
from deployments.client import load_policy_artifact
from deployments.openpi.client.policy import create_policy
import os

artifact = load_policy_artifact(os.environ["OPENPI_ARTIFACT"])
policy = create_policy(
    host=os.environ["SERVER_IP"],
    port=int(os.environ["OPENPI_PORT"]),
    policy_artifact=artifact,
)
print(policy.name)
print(policy.metadata["validation"])
'
```

任何 model id、step、16D action、layout hash 或 camera slot 不匹配都必须先修复；不要用关闭校验的方式绕过。

## 5. qpos-init 与 v2.2 数据集

`--policy-qpos-init-dataset` 用于把机械臂安全移动到数据集的指定帧 qpos，随后才允许策略启动。v2.2 数据集的 `action` 是 16D；运行时现在会根据数据集 `meta/info.json` 内的 `policy_io_layout` 显式桥接成内部 canonical 26D debug/hold action，而不是把 16D 当成旧 action 解析。

实际 qpos 目标不从模型 action 猜测，`--policy-qpos-init-source auto` 按以下优先级选择该帧有限且完整的 6D joint target：

1. `replay_qpos`
2. `observation.arm.qpos`
3. `observation.*.current_qpos`
4. `observation.*.actual_qpos`

对 `bimanual-cube-new-success` 的 episode 0/frame 0，当前会选择 `observation_arm_qpos`，左右臂各为 6D。读取/规划失败时策略保持被阻塞，不能自动退化为任意当前姿态。

可在没有硬件的情况下验证 qpos-init 数据读取：

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -c '
from teleop_runtime.deploy.policy_qpos_init import load_deploy_policy_qpos_init_plan
plan = load_deploy_policy_qpos_init_plan(
    dataset_root="/home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success",
    parquet_path=None,
    episode_index=0,
    frame_index=0,
    action_column="action",
    active_sides=("left", "right"),
    qpos_init_source="auto",
)
print(plan.metadata["action_encoding"], plan.target_sources_by_side)
'
```

预期包含 `v22_layout_to_legacy_q6_bridge` 和两个 qpos source；若不是，停止并检查数据集 metadata、episode 和 layout hash。

## 6. 相机/推理 dry-run（不发送硬件动作）

以下命令会打开两个 wrist 相机、读取真实观测、请求推理、执行动作归一化和安全过滤，但**不**加 `--send-actions`，因此不会向机械臂或夹爪发送命令。先检查两路图像、server validation、动作限幅和 fault 日志。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --max-steps 100 --debug
```

`multi` 配置可以含 front 相机；artifact 会只选择 `left_wrist` 和 `right_wrist`。缺少任一 artifact-required view 会 fail-closed。

## 7. 真机阻抗/力控执行（L5）

下面命令会连接真实 RM65、使能阻抗/力控后端、执行 dataset-frame qpos 初始化，并在启动许可后向双臂和 DMTAC 夹爪发送策略动作。只应在以下条件全部满足且操作者现场确认后执行：

- 人员远离工作空间，急停可用；
- 两臂 IP、tool frame、碰撞场景和 DMTAC assignment 已确认；
- 两路 wrist 相机在 dry-run 中已验证；
- server metadata 与 artifact 已通过第 4 节预检；
- dataset frame 0 是可安全到达的起始姿态；
- 已了解 `g` 会开始 qpos-init/策略流程，`A` 只会在 ready 后放行策略。

```bash
cd "$DEXTELEOP_ROOT"
"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config cameras/configs/rm65/devices/realsense.json \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current --wait-for-ready-confirm \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --left-arm-ip "$LEFT_ARM_IP" --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions --debug
```

`impedance_40hz` 选择经过测试的阻抗分支：policy 15 Hz（默认时）、臂后端 40 Hz、无 CANFD follow。OpenPI 部署会额外显式启用以下双速率 IK；这是部署专用覆盖，**不改变遥操**：

| 周期 | 工作 | 预算/优先级 |
| --- | --- | --- |
| 15 Hz | 独立的完整 Pinocchio IK（每臂一个 worker） | 66.7 ms；只使用 `rm65_pinocchio`，不会并发调用 RM65 SDK IK。超时结果丢弃，不积压旧 pose。 |
| 40 Hz | 力读取、导纳更新、限幅和安全 joint 输出 | 完整 IK 运行时只复用上一条安全命令；否则最多进行 6 ms 的本地 IK。 |

因此 40 Hz 不是“重复执行完整 IK”。完整求解在运行时优先，40 Hz 力控循环仍持续运行；若 pose 落入难解空间，输出保持最后安全关节命令，而不会让陈旧求解阻塞或追赶。不要改为位置分支，也不要添加 `--start-init_qpos`，除非已明确审查其安全影响。

手动启动（不带 `--auto-start`）时，启动日志必须先显示：

```text
[Deploy] impedance prepared without periodic output; press g/H/A to enable it, or use --auto-start.
```

这一步只完成力传感器/安全参考预热，**不会**启动 40 Hz 周期输出、执行 IK 或发送保持命令。按 `g` 才开启阻抗输出并开始 qpos-init；qpos-init 和 startup-init 完成后，按 `A` 才会请求 OpenPI 策略动作。若尚未按 `g` 就看到 `IK hold_last_safe`，说明远端仍在运行修复前代码，不能继续真机流程。

## 8. 失败定位

| 现象 | 含义与处理 |
| --- | --- |
| `must have shape (26,), got (16,)` | 使用了旧 qpos-init 代码；同步 `teleop_runtime/deploy/policy_qpos_init.py` 后重新运行第 5 节只读检查。 |
| `model_id_mismatch` 或 `checkpoint_step_mismatch` | server registry key/step 与 artifact 不一致；修正 registry，不要伪造客户端参数。 |
| `camera_slots` 或 required image key 不匹配 | 相机配置缺少 wrist logical key，或试图用 front 替代 wrist；修正配置。 |
| qpos-init 缺少 target | 指定帧没有有效 qpos audit；换经过审查的 episode/frame，或修复数据集。 |
| `DMTAC ... config is missing` | `latest` assignment、USB 映射或 side 配置不完整；先解决硬件配置，不要允许缺失后继续执行。 |
| `deployment_dual_rate_ik.deadline_miss_count` 增长（已按 `g` 后，`--debug` record） | 15 Hz 完整 IK 没能在 66.7 ms 内完成。该次结果已丢弃，40 Hz 仍保持力/导纳和最后安全 joint command。不要按 `A` 放行策略，不要增大 deadline 来掩盖问题；检查 tool/TCP、当前 qpos/反馈、力传感器偏置、Pinocchio 迭代耗时和策略 pose 是否进入难解空间。 |
| `deployment_dual_rate_hold_last_safe`（已按 `g` 后，`--debug` record） | 本地 6 ms IK 没有安全候选，或完整 worker 正在运行；这是受控保持而非新的策略动作。连续出现时不要按 `A`，先按上一行检查完整 IK deadline 和姿态可达性。 |
| `impedance IK hold_last_safe ... hard deadline`（已按 `g` 后） | 这属于旧的单线程阻抗路径，或说明部署双速率覆盖没有生效。停止真机流程，确认运行的是同步后的 `teleop_runtime/deploy/run_policy.py`，并检查 debug config 中 `deployment_dual_rate_ik_enabled=true`。 |
| server 无法起动 CUDA | 确认在 OpenPI uv 环境启动，且设置了 `JAX_PLATFORMS=cuda`。 |

## 9. 代码边界

```text
OpenPI server
  scripts/serve_dexteleop_policy.py
  configs/dexteleop_policy_registry.json

DexTeleop OpenPI client
  deployments/openpi/client/real.py       thin CLI + artifact forwarding
  deployments/openpi/client/policy.py     websocket contract + v2.2 adapter

Shared real deployment runtime
  teleop_runtime/deploy/run_policy.py
  teleop_runtime/deploy/policy_qpos_init.py
```

OpenPI 不应导入或控制真机 SDK；真机动作只能通过 `teleop_runtime` 的初始化、限幅、状态检查和 dispatch 路径执行。
