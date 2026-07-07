# teleop_runtime.deploy

`teleop_runtime.deploy` 是独立于采集主循环和左臂实验模式的策略部署入口，面向 RM65 单臂/双臂，可选 InspireTac 手、DMTac 1D gripper 和 RealSense。默认 dry-run；真实下发必须显式加 `--send-actions`，且默认要求先用数据集帧 qpos 初始化。

当前推荐入口是 `python -m teleop_runtime.deploy`。兼容入口
`python -m teleop_runtime.deploy.run_policy` 仍然保留。

## 当前执行链路

deploy 主线已经收敛到 canonical action path：

```text
policy output
  -> NormalizedAction
  -> filter_normalized
  -> RobotAction
  -> RobotActionResolver
  -> TeleopCommand
  -> launcher/backend dispatch
```

代码边界：

- `teleop_runtime/action/vector_pipeline.py`: dense action vector 到 `RobotAction` 的通用适配层；后续目标是迁到 `robot_motion/action`。
- `teleop_runtime/deploy/canonical_action.py`: policy action 过滤、canonical `RobotAction` 构造和 `TeleopCommand` 解析；保持硬件无关，后续目标是迁到 `robot_motion/deploy`。
- `teleop_runtime/deploy/dispatch.py`: 当前真机 launcher 的 `TeleopCommand` 到 arm/hand backend 分发；这是 real-specific 代码，不应迁到 `robot_motion`，sim launcher 应实现自己的 dispatcher。
- `teleop_runtime/deploy/actions.py`: 仍保留旧 `ActionSafetyFilter.filter()` direct-command path 作为 legacy 封存入口；新代码应使用 `filter_normalized()`。

因此模型部署、dataset replay debug 和未来 sim deploy 都应以 `RobotAction` 作为语义边界，
以 `TeleopCommand` 作为下发边界，不再让 policy adapter 直接生成 backend command。

后续迁移到 `robot_motion` 时，建议只搬硬件无关部分：

- 搬迁：`RobotAction` schema/spec、vector -> `RobotAction` 适配、canonical deploy preparation、value-level safety filter、`RobotActionResolver`。
- 不搬迁：CLI 参数、键盘控制、相机/触觉采集、真机 backend 构造、`dispatch.py`、fixed-init 的真机执行细节。
- 迁移后：`teleop_runtime.deploy.run_policy` 和未来 `sim` deploy runner 都调用同一套 `robot_motion` deploy preparation，再分别 dispatch 到真机或仿真 backend。

部署频率兼容旧接口：

- 默认仍可用 `--hz` 指定控制/发送频率。
- 如果提供 `--debug-replay-dataset` / `--debug-replay-parquet`，且没有显式传 `--action-hz`，部署器会优先从数据集实际记录的 action 行间 `timestamp` / episode stats / `control_hz` 推断真实 action 执行频率，并按该频率发送 action。
- 也可以用 `--action-hz-dataset` 单独指定一个参考数据集来推断发送频率，而不进入 replay 模式。
- `--policy-hz` 可以省略；省略时，普通部署保持历史默认 `10Hz`，而 replay / 参考数据集频率模式会默认跟随推断出的 action 频率。

## Action / Observation Layout

部署器兼容 `teleop_runtime.collection` 的数据布局：

- `action`: 26 维 `[left_arm_pose7, right_arm_pose7, left_hand_q6, right_hand_q6]`
- `observation.state`: 38 维 `[left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]`
- arm pose 是 base frame 下的 `[x, y, z, qx, qy, qz, qw]`，控制对象是
  `tool_tcp`，也就是当前臂/末端 profile 绑定的软件 TCP。
- RM65 真机执行会把 `tool_tcp` 目标转换到 SDK/IK 所需的 hand_base 或 arm_link6
  内部 frame；policy action 不应直接使用这些内部 frame。
- payload、重力补偿和 impedance 参数同样由部署/真机执行 profile 解释。
- 如果部署旧版位控模型、且不希望依赖力控标定 profile，可以给对应 side 传
  `--left/right-tool-profile position_endpoint`。该 profile 将 `tool_tcp` 固定在 RM65
  机械臂末端，且不携带 payload、重力补偿或 impedance 参数。
- 部署 observation/action metadata 默认携带 `arm_action_specs_by_side`、
  `arm_controlled_frame_by_side`、`arm_reference_frame_by_side`。下游模型或 adapter
  必须按 spec 解释 arm pose。
- 旧 row/replay 数据如果缺少 canonical action metadata，会由 row replay 兼容入口按历史
  `arm_endpoint` pose 解释；新数据和 policy 输出按 `tool_tcp` spec 解释。

策略输出支持：

- `np.ndarray` shape `(26,)` 或 `(chunk, 26)`
- 单臂 active 时 shape `(7,)` 或 `(chunk, 7)`，表示该侧 arm pose
- 双臂 active 时 shape `(14,)` 或 `(chunk, 14)`，顺序 `left,right`
- hand-only shape `(12,)`，顺序 `left,right`
- dict: `{"action": ...}` 或 `{"action_chunk": ...}`
- dict: `{"left_arm_pose": pose7, "right_arm_pose": pose7, "left_hand": q6, "right_hand": q6}`
- dict: `{"left_arm_qpos": q6, "right_arm_qpos": q6}` 可直接下发 joint target

注意：`DEPLOY_SCHEMA.md` 中的完整 canonical JSON 是目标契约，不是当前
`normalize_policy_output()` 可直接接收的返回格式。policy wrapper 如果内部使用该结构，
需要在返回前转换成上面的 short form、`NormalizedAction` 或 `RobotAction` 兼容路径。

## Policy 前数据集帧 QPOS 初始化

真实策略部署可以先选定一个已采集数据集 episode/frame，用该帧记录的 qpos 初始化机械臂，
然后再进入 startup init 和 policy inference。这用于让模型部署从指定数据集状态对齐启动。

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --policy-qpos-init-source auto \
  --policy-qpos-init-arm-max-angvel 0.25,0.25,0.25,0.35,0.35,0.35 \
  --init-source current \
  --wait-for-ready-confirm \
  --send-actions
```

启动顺序：

- 启动后按 `g`，执行 dataset frame qpos init。
- 到达目标后进入 startup init；完成后如果启用 ready confirm，会显示 `ready=wait:A`。
- 再按 `A`，才开始 policy inference / action execution。

`--policy-qpos-init-source auto` 和 replay 一致，优先使用记录的
`action.replay.<side>_arm_joint_target`，缺失时依次回退到
`observation.arms.<side>.current_qpos`、`observation.arms.<side>.actual_qpos`。
初始化轨迹规划同样复用 replay qpos init 的慢速、双臂碰撞检查和奇异位形保护。
需要临时调整首帧 QPOS 初始化速度时，使用 `--policy-qpos-init-arm-max-angvel`；
可传 1 个 rad/s 值应用到 6 个关节，也可传 6 个逗号分隔的逐关节 rad/s 值。
该路径和旧 `--deploy-fixed-init-*` 互斥；后者保留为 legacy fixed T 轨迹入口。

安全门：

- 真实 arm 下发时，默认必须提供 `--policy-qpos-init-dataset` 或
  `--policy-qpos-init-parquet`。
- 如果确实要从当前 qpos 原地开始推理/执行，必须显式加 `--start-init_qpos`。
  这是强制覆盖模式，日志会打印 override 提示；只应在确认当前姿态、末端、工作空间和
  policy 首帧都安全时使用。
- dry-run 不受该限制，可以不提供 qpos init 数据集。

## 运行时操作和按键

脚本启动后会打印当前模式、是否真实下发、是否需要 ready confirm，以及可用按键。
按键只在运行脚本的终端处于前台且 stdin 是 TTY 时生效；大小写等价。

常用按键：

- `g`: start/resume。普通部署中启动或恢复 policy；如果配置了
  `--policy-qpos-init-*` 且尚未完成，则先执行选定数据集帧的 qpos init。
- `A`: ready confirm。`--wait-for-ready-confirm` 开启时，startup init 完成并出现
  `ready=wait:A` 后按 `A`，才开始 policy inference / action execution。`A` 不会启动
  policy qpos init；qpos init 必须用 `g` 开始。
- `h` / `H`: 普通部署中暂停 policy 并清空 action queue。policy qpos init 正在执行时不会用
  `h` 中断轨迹；如需真实急停，使用外部安全流程。
- `q`: quit。退出 deploy loop 并关闭已打开的 backend/camera。它不是急停。
- legacy fixed-init 下同一个 `h` / `H` 键有特殊含义：配置 `--deploy-fixed-init-*` 时，
  第一次按键移到 T 轨迹首帧 qpos，出现 `fixed_init=ready:H` 后第二次按键执行固定 T 轨迹。

启动状态：

- 默认不加 `--auto-start` 时，policy 初始为 paused，需要按 `g`。
- 加 `--auto-start` 且没有 qpos/fixed init gate 时，启动后自动进入 policy。
- 加 `--auto-start` 且配置了 `--policy-qpos-init-*` 时，启动后自动进入 qpos init；
  qpos init 完成后才释放 startup init / policy。
- 加 `--send-actions` 才会真实下发；真实 arm 下发还需要数据集 qpos init，或显式
  `--start-init_qpos` 覆盖。不加 `--send-actions` 时是 dry-run，只构造 observation、过滤动作并打印状态。

状态行常见字段：

- `policy=active|paused`: policy 主循环是否放行。
- `send=on|dry-run`: 是否真实发到硬件。
- `queue=N`: 待执行 action chunk 数量。
- `policy_qpos_init=wait:g|planning|qpos:i/N|final_hold|failed:g|timeout:g|released`: policy 前 qpos init 状态。
- `init=wait|xx%|done:<source>`: startup init 状态。
- `ready=wait:A`: startup init 已完成，正在等待 `A` 确认。
- `state_valid=a/b`: 当前 observation state 中有效元素数量。

常见 qpos init 失败：

- `qpos init planning failed before execution` / `no collision-safe interpolation step was found`：
  规划阶段没有找到满足双臂碰撞约束的插值路径，命令不会下发，policy 会保持 blocked。
  优先换一个更安全的数据集 frame，或先把机械臂手动移动到更安全的 pre-start qpos。
  如果确认是 strict proxy 过于保守，可在检查真实间隙后尝试
  `--policy-qpos-init-collision-scene-profile dual_arm_hand_proxy_open`。
- `timed out while moving to dataset frame`：轨迹已经开始执行，但实际 qpos 未在
  `--policy-qpos-init-timeout-s` 内进入 `--policy-qpos-init-qpos-tolerance`。
  日志会给出最大关节误差。先确认机械臂是否仍在运动、是否被限速/保护挡住；必要时增大
  timeout，或在确认可接受的前提下放宽 qpos tolerance。
- `targets={'left': 'observation_current_qpos', ...}` 表示该帧没有可用的
  `action.replay.<side>_arm_joint_target`，`--policy-qpos-init-source auto` 回退到了记录的
  observation qpos。若必须使用 replay qpos，改用
  `--policy-qpos-init-source replay_qpos`，缺失时会直接报错。

### hand16 / 空槽部署语义

后续 hand16 action layout 中，手部固定槽位用于统一夹爪、InspireTac 和高自由度
灵巧手的接口长度。固定槽位不代表所有槽都有效：

- 有效槽由 `hand_active_mask_by_side` 或本体 `HandActionSpec` 定义。
- 无效槽值约定为 `NaN`，表示 padding/unused slot。
- 部署侧不得把 `NaN` 槽解释为 0，也不得把它们下发给硬件。
- policy 输出、teleop 输入和 dataset replay 都必须在进入 safety filter/backend 前按 mask
  只取有效槽，再由具体 hand adapter 映射到本体命令。

因此，模型 checkpoint 或部署脚本如果使用 hand16，必须随同声明 hand spec；仅凭固定
action 维度不足以判断哪些 hand 槽可执行。旧模型/旧数据集应通过自己的 adapter 映射，
部署主线不为旧 state/action 维度保留额外兼容口径。

## 单臂 Dry-Run

```bash
python -m teleop_runtime.deploy \
  --policy hold \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --max-steps 200
```

未传 `--auto-start` 时启动后按 `g` 才执行策略；按 `h` 暂停并清空 action chunk，按 `q` 退出。
如果同时使用 `--wait-for-ready-confirm`，startup init 完成并出现 `ready=wait:A`
后还需要再按 `A`，才会真正开始 policy inference / action execution。

## 单臂真实下发

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-tool-profile position_endpoint \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --send-actions
```

## 双臂 + 双手

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --enable-hands \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-tactile-regions all \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --send-actions
```

## 双臂 + DMTac Gripper

先按根目录 `README.md` 的“GRIPPER 和 DM TAC 分开标注”保存
`teleop_records/dmtac_hardware_configs/latest.json`；部署时读取同一份 latest：

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:create_policy \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --policy-qpos-init-dataset /abs/path/to/dataset \
  --policy-qpos-init-episode 20 \
  --policy-qpos-init-frame 123 \
  --send-actions
```

没有 DM TAC 触觉时加 `--dmtac-no-tactile`。DMTac gripper 仍占用 6 维 hand action
兼容槽，但部署 metadata 会把 `HandActionSpec` 标成 `dmtac_gripper`，只有
`--gripper-output-dim` 指定的槽是真实 gripper close command。

## 外部模型 Client

`teleop_runtime.deploy` 只负责真机侧 observation、统一动作解析、安全检查和硬件下发。
模型 server 的连接、协议校验、输入/输出适配放在 `deployments.client` 以及对应模型族目录。

OpenPI 的入口位于 `deployments/openpi/client`。例如 dry-run：

```bash
python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8000 \
  --model dual_rm65_dmgripper_latest \
  --active-sides both \
  --hand-backend dmtac-gripper \
  --dmtac-config latest
```

该入口会把 OpenPI-specific 参数展开成普通的 `--policy
deployments.openpi.client.policy:create_policy`，因此 real runtime 不 import
也不绑定 OpenPI。真实下发仍需显式加 `--send-actions`。

## Dataset Replay Debug

可以直接 replay `teleop_runtime.collection` 采集出的 action。默认仍是 dry-run；真实下发仍需 `--send-actions`。
debug replay 在 `--action-queue-policy auto` 下会自动 append 预取的 action
chunk，避免按 `--min-action-queue` 补队列时跳过轨迹尾部；普通模型 policy 仍默认
replace pending queue。

```bash
python -m teleop_runtime.deploy \
  --debug-replay-dataset /abs/path/to/dataset \
  --debug-replay-episode 0 \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --auto-start
```

上面命令在未显式传 `--action-hz` 时，会自动读取该 episode 的真实频率并按接近真机的节奏发送/消费 action。

## 用真机数据集校准部署频率

```bash
python -m teleop_runtime.deploy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --action-hz-dataset /abs/path/to/real_dataset \
  --action-hz-episode 12 \
  --policy-qpos-init-dataset /abs/path/to/real_dataset \
  --policy-qpos-init-episode 12 \
  --policy-qpos-init-frame 0 \
  --send-actions
```

## 安全边界

- arm cartesian pose 会按 `--arm-workspace-min/--arm-workspace-max` 裁剪位置。
- arm pose 每 tick 默认限速：`--max-arm-xyz-step-m 0.03`、`--max-arm-rot-step-deg 8`。
- arm joint target 每 tick 默认限速：`--max-arm-joint-step-rad 0.08`。
- hand q6 当前只会 clip 到 `[0, 1]`；`--max-hand-step` 是保留参数，hand rate-limit 暂未启用。
- `--require-ready-observation` 会在 active state 或图像缺失时跳过策略和下发。
- `--inspire-hand-max-force` 默认继承 `teleop_runtime` 的 `800`。
