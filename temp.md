我按当前代码读了一遍，没有改代码。现在的主线流程是：VIVE 输出 `left_wrist/right_wrist`，retarget 生成统一 `tool_tcp` action，真机默认用外环 pose impedance 执行；TCP、payload/CoM、重力补偿相关配置和 impedance 参数都来自 RM65 tool profile。

**当前 Action 语义**
现在 arm action 主 frame 是 `tool_tcp`，不是 `arm_endpoint`：

- 统一 action：`T_arm_base_tool_tcp`
- VIVE retarget、数据集、policy 输出、真机执行都按 `tool_tcp` 理解
- `arm_endpoint` 只是兼容机械臂末端 frame
- profile 里的 `tcp.pose_flange_m_rad` 定义 flange 到 tool TCP
- 真机 impedance 后端要求 `cartesian_pose`，默认 `--arm-control-mode impedance --impedance-mode pose`

对应代码主要在 [arm_frames.py](/home/password-1/DexTeleop/teleop_runtime/arm_frames.py:1)、[pose_to_arm.py](/home/password-1/DexTeleop/teleop_runtime/retargeters/pose_to_arm.py:1)、[rm65_impedance.py](/home/password-1/DexTeleop/teleop_runtime/backends/arms/rm65_impedance.py:1)。

**1. VIVE Sender**
只需要两个 tracker：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host <teleop_runtime机器IP> \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist
```

第三个 `tracker_3:auxiliary` 仍可选，但 virtual desk 标定可以直接用 left/right wrist tracker。

**2. VIVE/工位标定**
第一次上手先跑 station/tracker 标定，不下发真机动作：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999
```

流程按终端提示走：

1. `LINK_CHECK`：确认左右 wrist channel 都有数据，按 `n`
2. `VIRTUAL_DESK_X/Y/Z`：按 `c` 后分别做后到前、右到左、低到高 sweep
3. `WRIST_COARSE`：双手放 canonical neutral pose，按 `c` 静止采样
4. 可选 `--show-tracker-frame-viz` 后用 `j/l/i/k/u/o` 微调 wrist 姿态
5. `SAVE` 写出 `teleop_records/calibration_profiles/<timestamp>_vive_rm65_virtual_desk.json`

之后复用：

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

注意：这个 profile 只包含 virtual desk 和 wrist tracker 外参，不包含每次 session 的 home。

**3. RM65 Tool Profile/TCP/重力补偿准备**
每个 arm + tool 都需要一个独立 profile：`hardware/rm65/utils/configs/calibrations/rm65_{arm}_{tool}.json`。

推荐顺序：

```bash
python hardware/rm65/utils/scripts/rm65_official_force_sensor_calibrate.py \
  --arm right \
  --expected-tool empty \
  --check-current \
  --move-to-first \
  --execute \
  --confirm CALIBRATE_RM65_FORCE_SENSOR
```

然后做 payload/CoM 拟合、写入 controller/tool profile、验证：

```bash
python hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py \
  --arm right \
  --expected-tool empty

python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm right \
  --expected-tool empty \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD

python hardware/rm65/utils/scripts/rm65_validate_tool_payload.py \
  --arm right \
  --expected-tool empty \
  --checkpoints 4 \
  --samples 20 \
  --interval-sec 0.05 \
  --evaluate-source tool
```

再确认/调整 TCP：

```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile empty \
  --arm right
```

最后建议做 pose impedance 固定点调参并保存到 profile：

```bash
python hardware/rm65/utils/scripts/rm65_fixed_point_impedance_tune.py \
  --profile empty \
  --arm right \
  --mode pose \
  --axes all \
  --control-hz 40 \
  --action-hz 20 \
  --save-to-profile \
  --execute \
  --confirm TEST_RM65_FIXED_POINT_IMPEDANCE
```

当前 strict impedance 要求 profile 里有：`tool_name`、`tcp.pose_flange_m_rad`、`tcp.confirmed=true`、`payload.mass_kg`、`payload.com_tool_m`、`impedance.pose`。

**4. 仿真/只读验证**
先用真实 VIVE 输入跑 SAPIEN，不给真机下发动作：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode virtual \
  --control-mode absolute \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

如果要读取真机当前姿态作为 preview 起点，但仍不下发：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode real \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --active-sides right \
  --control-mode absolute \
  --load-profile teleop_records/calibration_profiles/your_profile.json
```

启动后按 `p` 或回车做 runtime home。这里验证前/左/上方向、左右映射、姿态跟随、workspace、碰撞规避。再切 `--control-mode delta_clutch --keyboard-clutch` 验证相对控制手感。

**5. 真机单臂测试**
先右臂、低速、小范围、不接手部末端控制：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_empty.json \
  --no-collision
```

流程：

1. 程序连接 RM65，strict 校验 active tool/profile
2. 按 `p` 或回车做 runtime home：采当前 wrist home + 当前真机 tool TCP home
3. 程序提示保持末端静止
4. 执行 `--force-tare-sec 3.0` 的 force tare
5. 启动 pose impedance loop
6. `arm_hardware()` 后开始真正下发

旧位控 fallback 是：

```bash
--arm-control-mode position
```

但这不走默认外环力控/impedance。

**6. 真机双臂测试**
双臂需要左右两边 profile 都齐全：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides both \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_<tool>.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_<tool>.json
```

默认双臂会尝试 collision-aware retargeter；如果环境缺 `pinocchio/pink/qpsolvers` 或先做保守测试，加 `--no-collision`。

**当前明显阻塞点**
我在当前仓库只看到一个正式 tool profile：

```text
hardware/rm65/utils/configs/calibrations/rm65_right_empty.json
```

你 IDE 里开的 `rm65_right_inspire.json` 当前路径下不存在。也就是说：按默认 strict profile，右臂 empty 可以走；right inspire、left inspire、双臂都需要先生成/确认对应 `rm65_{arm}_{tool}.json`，否则真机 impedance 入口会在启动阶段拒绝。












`left_inspire` 的重力标定本身看起来不是问题，问题是它没有形成当前真机遥操 strict 模式要加载的运行时 profile。

我看到已有这些结果：

- `hardware/rm65/utils/results/rm65_left_inspire_force_measurements_calibration_fit.json`
  - `mass_kg_abs = 0.933815`
  - `com_tool_m = [0.001263, -0.001820, 0.101409]`
  - corrected force/torque residual很小，fit 质量看起来正常
- `hardware/rm65/utils/results/rm65_left_inspire_tool_payload_validation.json`
  - controller 当前 tool 是 `inspire`
  - payload readback `0.933`
  - CoM readback 和 fit 误差约 `1e-6 m`
- `hardware/rm65/utils/results/rm65_left_inspire_official_force_sensor_calibration_report.json`
  - official force sensor calibration 也有记录，`Manual_Set_Force` return code 都是 0

缺的是：

1. **没有 `hardware/rm65/utils/configs/calibrations/rm65_left_inspire.json`**
   当前 `configs/calibrations/` 里只有 `rm65_right_empty.json`。真机 impedance backend strict 模式会按 active tool name `inspire` 去加载 `rm65_left_inspire.json`，找不到就启动不了。

2. **没有已确认的 profile TCP**
   即使生成 profile，`inspire` 的 TCP 初值默认会偏向 `hand_base`，还需要用 TCP GUI 保存确认，让：
   ```json
   "tcp": {
     "frame_name": "tool_tcp",
     "pose_flange_m_rad": [...],
     "confirmed": true
   }
   ```

3. **profile 里需要汇总 payload 和 impedance.pose**
   payload 数据已有，但还没写进 `rm65_left_inspire.json`。`impedance.pose` profile 创建后会有默认值，但如果要更稳，最好再跑一次 fixed-point pose impedance tune 并 `--save-to-profile`。

建议补齐顺序：

```bash
python hardware/rm65/utils/scripts/rm65_apply_tool_payload_from_fit.py \
  --arm left \
  --expected-tool inspire \
  --fit-json hardware/rm65/utils/results/rm65_left_inspire_force_measurements_calibration_fit.json \
  --execute \
  --confirm APPLY_RM65_TOOL_PAYLOAD
```

然后确认 TCP：

```bash
python hardware/rm65/utils/scripts/rm65_tool_tcp_adjust.py \
  --profile inspire \
  --arm left
```

保存后再检查：

```bash
python hardware/rm65/utils/scripts/rm65_calibration_profile.py show \
  --name inspire \
  --arm left
```

一句话：`left_inspire` 的重力/载荷标定已经做过，缺的是把这些结果落到 `rm65_left_inspire.json`，并确认 `tool_tcp`。