# RM65 初始位姿录制与复用流程

> 本文包含的 SAPIEN 可视化命令属于冻结历史诊断，仅供旧流程复现。当前若只需要初始状态
> 记录，应使用非渲染/中立路径；不要把 SAPIEN preview 作为新的流程门禁或验证证据。

本流程只读取真机状态并保存初始对齐 JSON，不会让机械臂运动。真实遥操作或任何
需要下发动作的命令，仍必须按现场安全流程单独确认。

## 1. 准备环境和标定文件

在仓库根目录进入真机环境：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

准备已经验证过的 VIVE/desk 标定 profile。它至少应包含左右 RM65 的 `T_D_B`；例如：

```text
teleop_records/calibration_profiles/20260702_090000_vive_rm65_virtual_desk.json
```

如果还没有 profile，先按 `teleop_runtime/CalibrationREADME.md` 完成 VIVE 虚拟 desk
标定，不要用默认值替代已经安装好的真实机械臂安装关系。

## 2. 把机械臂摆到要复用的安全起始姿态

由现场人员在断能/低风险条件下把左、右 RM65 和可选 InspireTac 手放到本次运行希望的
起始姿态。该工具只读取当前位置，不会执行回零、MoveJ、servo 或任何补偿动作。

确认：

- 机械臂没有处于碰撞、急停或关节限位附近；
- 左右物理 IP 与逻辑 side 对应正确；
- 如果连接 InspireTac，手部也处在希望保存的开合状态。

## 3. 启动只读状态查看器

双臂、带 SAPIEN viewer 的完整命令：

```bash
python -m teleop_runtime.debug.rm65_real_debug.run_rm65_real_state_viz \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --save-initial-alignment teleop_records/initial_alignment_states
```

只看终端、不创建 viewer 时追加 `--skip-renderer`。单臂只提供对应一侧的
`--*-arm-ip` 即可。若物理左右与逻辑左右确实相反，才追加 `--swap-arm-sides`。

## 4. 逐项确认读取结果

启动日志和状态行应确认：

1. `left_arm_ip` / `right_arm_ip` 到 logical side 的映射正确；
2. `q=[...]` 是当前真实六关节角（弧度），没有错误码或异常跳变；
3. `tcp_xyz`/`tcp_rpy_deg` 位于标定 profile 预期的 desk 坐标位置；
4. viewer 中 `left_tcp_actual`、`right_tcp_actual` 与真实安装方向一致。

发现 side、TCP 或标定方向错误时，先退出并修正 profile/IP，不要在错误状态下保存。

## 5. 保存 JSON

确认状态稳定后，在状态查看器终端按一次 `s`。程序会在指定目录生成带时间戳的文件，
例如：

```text
teleop_records/initial_alignment_states/20260721_153000_rm65_initial_alignment.json
```

也可以直接给一个文件路径：

```bash
--save-initial-alignment teleop_records/initial_alignment_states/rm65_start.json
```

保存内容包括：

- 每侧 `arm_qpos`（六关节角）；
- `tcp_pose_base` 和标定 desk 坐标下的 `tcp_pose_desk`；
- 可用时的手部 `hand_qpos_q6`；
- 保存时间、profile、IP、side swap 等 metadata。

终端出现 `saved initial alignment -> ...` 后才算写盘完成。随后按 `q` 退出只读查看器。

## 6. 离线检查保存文件

不连接设备即可检查格式和两侧数据：

```bash
python - <<'PY'
from pathlib import Path
from teleop_runtime.initial_alignment import TeleopInitialAlignment

path = Path("teleop_records/initial_alignment_states/your_alignment.json")
alignment = TeleopInitialAlignment.load_json(path)
print("tcp_frame:", alignment.tcp_frame)
for side in ("left", "right"):
    state = alignment.get_side(side)
    print(side, None if state is None or state.arm_qpos is None else state.arm_qpos.tolist())
PY
```

双臂运行时应同时存在左右 `arm_qpos`，每侧长度为 6 且数值有限。缺失一侧时不要把
文件用于双臂启动检查。

## 7. 在遥操作中复用

真实遥操作入口可以加载该文件做启动前的起始关节检查：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --load-initial-alignment \
    teleop_records/initial_alignment_states/your_alignment.json \
  --gui
```

这里的 `--load-initial-alignment` 只提供保存的 robot 起始 qpos/TCP 作为检查和 seed；
按 `p` 开始遥操作时，VIVE tracker home 仍会重新采集，避免复用过期的人手姿态。
如果当前只连接 2 个踏板，入口会自动关闭 init-arm，不会读取或要求 init-arm qpos
profile；fix-arm 仍可正常使用。

当前真机会话的 preview/IK seed 也使用同一份 initial-alignment 文件；历史
SAPIEN virtual teleop 入口已退役。

## 8. 每次运行前的最小验收

- 文件路径可读且 JSON 能通过 `TeleopInitialAlignment.load_json`；
- 保存文件的 side/IP 与当前接线一致；
- 当前机械臂与保存 qpos 的差异在现场允许阈值内；
- profile、tool/TCP frame、desk 坐标约定没有改变；
- 只有在上述检查通过后，才开始真实遥操作或数采。
