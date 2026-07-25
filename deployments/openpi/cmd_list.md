# OpenPI checkpoint 与 DexTeleop 部署命令

本文档记录当前远端 /home/user/zc/openpi 已检查过的 checkpoint，以及在
/home/user/zc/DexTeleop 上启动 OpenPI server、执行契约检查、无动作 dry-run 和
真实力控部署的命令。

以下命令默认在远端机器 dex-remote 上执行。本文档只提供命令，不自动启动
OpenPI、相机、机械臂或夹爪。

## 1. 当前 checkpoint 清单

检查目录：/home/user/zc/openpi/checkpoints

| 模型 | checkpoint | 语义/维度 | 相机 | 状态 |
| --- | --- | --- | --- | --- |
| 0715-cube-front-ready-mix-4to1-h16-relative | .../150000 | v2.3 relative，state 32D/action 16D，h16 | base + left/right wrist | 推荐 |
| 0715-cube-front-ready-mix-4to1-relative | .../149999 | v2.3 relative，state 32D/action 16D，h50 | base + left/right wrist | 可部署 |
| 0715-cube-front-ready-mix-4to1 | .../138257 | v2.2 absolute，state 32D/action 16D，h50 | base + left/right wrist | 可部署 |
| 0712-wrist-cube | .../92599 | v2.2，state 16D/action 16D，h50 | left/right wrist | 已注册、已验证 |
| 0712-cube | .../11768 | v2.2，state 16D/action 16D，h50 | base + left/right wrist | 可部署 |
| dual_rm65_dmgripper | .../99999 | 老 v1，action 26D，h50 | 3 路 | 仅旧 registry 兼容 |

完整路径如下：

~~~text
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-relative/0715-cube-front-ready-mix-4to1-relative/149999
/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1/0715-cube-front-ready-/138257
/home/user/zc/openpi/checkpoints/0712-wrist-cube/0712-wrist-cube-wrist-only/92599
/home/user/zc/openpi/checkpoints/0712-cube/0712-cube/11768
/home/user/zc/openpi/checkpoints/dual_rm65_dmgripper/dual_rm65_dmgripper_lora_b2_lr5e-5_100k_20260706/99999
~~~

以下目录当前不可部署：

~~~text
/home/user/zc/openpi/checkpoints/dual_rm65_dmgripper_v22_masked/bimanual_cube_v22_masked_150k
~~~

该目录为空。writing-phase1-grap-pen-h16-relative 当前也没有训练 step，只有
stats/wandb 文件；writing 模型完成训练并生成 checkpoint 后再补充部署命令。

## 2. 环境变量

OpenPI server 使用 OpenPI .venv；DexTeleop client 使用 dexteleop310 环境。

~~~bash
export DEXTELEOP_ROOT=/home/user/zc/DexTeleop
export OPENPI_ROOT=/home/user/zc/openpi
export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078

export OPENPI_CKPT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000
export OPENPI_ARTIFACT=$OPENPI_CKPT/policy_artifact.json

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
~~~

若 client 不在 server 本机，将 SERVER_IP 改为 OpenPI server 的可路由 IP；不要
把 127.0.0.1 用在跨机器连接上。

## 3. 推荐模型：h16 relative server

该 checkpoint 没有正式 registry entry，因此使用 DexTeleop wrapper 从
policy_artifact.json 生成临时 registry。临时文件位于 /tmp，不会修改 OpenPI
正式 registry 或 checkpoint。

### 3.1 只生成 registry，不启动 server

~~~bash
cd /home/user/zc/DexTeleop

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

/home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --openpi-root /home/user/zc/openpi \
  --config 0715-cube-front-ready-mix-4to1-h16-relative \
  --action-horizon 16 \
  --host 0.0.0.0 \
  --port 8078 \
  --dry-run
~~~

### 3.2 启动 server

~~~bash
cd /home/user/zc/DexTeleop

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

/home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir "$OPENPI_CKPT" \
  --openpi-root /home/user/zc/openpi \
  --config 0715-cube-front-ready-mix-4to1-h16-relative \
  --action-horizon 16 \
  --host 0.0.0.0 \
  --port 8078
~~~

保持 server 终端运行，在另一终端执行：

~~~bash
curl --fail --silent http://127.0.0.1:8078/healthz
curl --fail --silent http://127.0.0.1:8078/metadata | python -m json.tool
~~~

metadata 至少应确认以下内容与 checkpoint 一致：

~~~text
model_id=0715-cube-front-ready-mix-4to1
deployment.checkpoint_step=150000
action_horizon=16
action_dim=16
state_dim=32
~~~

## 4. 无硬件契约检查

该命令不连接 server、不打开相机、不连接机械臂或夹爪，也不发送动作。

~~~bash
cd /home/user/zc/DexTeleop

export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
export OPENPI_ARTIFACT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000/policy_artifact.json

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --contract-check-only
~~~

检查输出应通过 model id、checkpoint step、v2.3 relative pose contract、state/action
维度、layout hash、camera slots 和 action horizon 校验。任何校验失败都不能通过
关闭校验的方式绕过。

## 5. 无硬件 websocket 预检

先启动第 3 节 server，再执行以下命令。该命令只读取 server metadata 并验证
artifact，不打开相机、不连接机械臂或夹爪。

~~~bash
cd /home/user/zc/DexTeleop

export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export OPENPI_ARTIFACT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000/policy_artifact.json

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
~~~

## 6. qpos-init 数据读取检查

该命令不连接硬件，只检查部署前使用的数据集帧是否能够解析出左右臂 6D qpos：

~~~bash
cd /home/user/zc/DexTeleop

export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python

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
~~~

## 7. 相机/推理 dry-run（不发送硬件动作）

该命令会打开两路 wrist 相机、读取实时观测、请求推理并执行安全过滤，但不带
--send-actions，因此不会向机械臂或夹爪发送命令。

~~~bash
cd /home/user/zc/DexTeleop

export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export OPENPI_ARTIFACT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000/policy_artifact.json

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --camera-keep-last 4 --camera-frame-max-age-sec 0.2 \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --max-steps 100 --debug
~~~

## 8. 真实 RM65 阻抗/力控部署（L5）

以下命令会连接真实 RM65，执行 qpos 初始化并发送策略动作。我没有执行该命令。
只有在人员、急停、双臂 IP、tool frame、碰撞场景、DMTAC assignment、相机画面、
server metadata 和前述 dry-run 均确认无误后，才允许现场执行。

~~~bash
cd /home/user/zc/DexTeleop

export DEXTELEOP_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
export SERVER_IP=127.0.0.1
export OPENPI_PORT=8078
export OPENPI_ARTIFACT=/home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1-h16-relative/0715-cube-front-ready-mix-4to1-h16-relative/150000/policy_artifact.json
export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20

"$DEXTELEOP_PYTHON" -m deployments.openpi.client.real \
  --host "$SERVER_IP" --port "$OPENPI_PORT" \
  --policy-artifact-json "$OPENPI_ARTIFACT" \
  --active-sides both \
  --arm-execution-profile impedance_40hz \
  --enable-hands --hand-backend dmtac-gripper --dmtac-config latest \
  --camera-mode multi \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --camera-keep-last 4 --camera-frame-max-age-sec 0.2 \
  --require-camera-frame --require-ready-observation \
  --action-queue-policy replace --min-action-queue 0 \
  --init-source current --wait-for-ready-confirm \
  --policy-qpos-init-dataset /home/user/zc/DexTeleop/datasets/sorted_v22/bimanual-cube-new-success \
  --policy-qpos-init-episode 0 --policy-qpos-init-frame 0 \
  --policy-qpos-init-source auto \
  --left-arm-ip "$LEFT_ARM_IP" --right-arm-ip "$RIGHT_ARM_IP" \
  --send-actions --debug
~~~

impedance_40hz 对应当前部署专用的双速率阻抗链路：policy 默认 15 Hz、臂后端
40 Hz、力读取/导纳/限幅和安全 joint 输出持续运行。不要改成 position 分支，也不
要额外添加 --start-init_qpos。

## 9. 其他已存在模型

### 9.1 v2.3 relative，h50

该模型已经在 OpenPI registry 中：

~~~bash
cd /home/user/zc/openpi

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model 0715-cube-front-ready-mix-4to1-relative \
  --host 0.0.0.0 \
  --port 8078
~~~

### 9.2 v2.2 0712-wrist-cube

~~~bash
cd /home/user/zc/openpi

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model 0712-wrist-cube \
  --host 0.0.0.0 \
  --port 8078
~~~

### 9.3 v2.2 0712-cube

~~~bash
cd /home/user/zc/openpi

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model 0712-cube_11768 \
  --host 0.0.0.0 \
  --port 8078
~~~

### 9.4 v2.2 absolute 0715-cube-front-ready-mix-4to1

该 checkpoint 没有正式 registry entry，使用 wrapper：

~~~bash
cd /home/user/zc/DexTeleop

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

/home/user/zc/openpi/.venv/bin/python -m deployments.openpi.server \
  --checkpoint-dir \
  /home/user/zc/openpi/checkpoints/0715-cube-front-ready-mix-4to1/0715-cube-front-ready-/138257 \
  --openpi-root /home/user/zc/openpi \
  --config 0715-cube-front-ready-mix-4to1 \
  --action-horizon 50 \
  --host 0.0.0.0 \
  --port 8078
~~~

### 9.5 老 v1 dual_rm65_dmgripper

该 checkpoint 没有 policy_artifact.json，只能走旧 registry 兼容入口，不能使用
当前 artifact 契约检查流程：

~~~bash
cd /home/user/zc/openpi

export PYTHONPATH=/home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

uv run scripts/serve_dexteleop_policy.py \
  --model dual_rm65_dmgripper_latest \
  --host 0.0.0.0 \
  --port 8078
~~~

## 10. 训练完成后的 writing 模型

当前 writing relative checkpoint 尚未生成，不能执行部署。训练时如果 checkpoint
目录已经存在，默认使用 --resume，不要直接重复运行不带 --resume 的命令：

~~~bash
cd /home/user/zc/openpi

uv run scripts/compute_norm_stats.py \
  --config-name writing-phase1-grap-pen-h16-relative

uv run scripts/train.py writing-phase1-grap-pen-h16-relative \
  --exp-name=writing-phase1-grap-pen-h16-relative \
  --resume
~~~

生成有效 checkpoint 和 policy_artifact.json 后，应把第 3 节中的 OPENPI_CKPT、
--config 和 --action-horizon 替换为 writing 模型对应值，再执行第 4～8 节的检查和
部署流程。
