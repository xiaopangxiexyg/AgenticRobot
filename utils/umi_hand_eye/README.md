# UMI Vive→desk 标定

本工具标定一套刚性连接的 RealSense 相机和 Vive tracker。标定板固定不动，且其
坐标系直接定义为 v2.3 canonical `desk`。采样时手持整套 UMI 改变位置和姿态；不要
让相机与 tracker 发生相对移动。

矩阵统一采用 `p_B = T_A_to_B @ p_A`。每组样本满足：

```text
T_tracker_to_vive_world @ T_camera_to_tracker @ T_board_to_camera
    = T_board_to_vive_world
```

因此输出的 v2.3 变换为：

```text
T_vive_world_to_desk = inverse(T_board_to_vive_world)
```

这只标定 UMI/Vive 与 desk，不标定 RM65 base、末端或 TCP。真机采集仍使用自身的
URDF desk 配置。

## 配置

默认配置是 `configs/rm65_dual_wrist_umi_v1.json`：

- 左腕相机：`261122273924`，tracker：`tracker_1`
- 右腕相机：`260522278055`，tracker：`tracker_2`
- 标定有效期：3600 秒

两套 UMI 都在配置中声明，但第一版一次选择一侧求解。Vive sender 仍发送现有两只
tracker，不需要第三只 tracker。

## 在线采集

先启动现有 Vive sender，然后在采集电脑运行：

```bash
python -m utils.umi_hand_eye --side left
```

窗口按键：

- `r`：在 UMI 静止时记录一个 2 秒 burst；
- `c`：达到最少样本数后求解；
- `q` / `Esc`：退出。

建议至少 25 组、推荐 40 组。必须覆盖不同位置，并绕至少两个轴产生明显旋转；只有
平移、始终正对标定板或非常相近的姿态会导致手眼方程退化。

结果默认写入 `teleop_records/umi_hand_eye/`：

- `*_samples.json`：原始配对样本，可离线重算；
- `*.json`：标定结果、残差、设备绑定和过期时间；
- `*_pose_frame_contract.json`：v2.3 UMI 数采使用的 `desk_absolute` contract。

## 离线重算与有效期检查

```bash
python -m utils.umi_hand_eye \
  --side left \
  --load-samples teleop_records/umi_hand_eye/example_samples.json

python -m utils.umi_hand_eye \
  --side left \
  --check-result teleop_records/umi_hand_eye/example.json
```

`--check-result` 会同时检查 1 小时有效期、camera serial 和 tracker name。过期、SteamVR
tracking space 重建、基站移动、相机/tracker 装配变化后都必须重新标定。

## 时间配对限制

第一版不是运动中硬同步。每个视觉检测帧读取最新 Vive pose，并要求其接收年龄默认不
超过 250 ms；随后在整个 burst 上检查 tracker 和视觉 target 都保持静止。该策略适合
人工摆位后的静止标定，不适合边移动边连续采样。未来双相机同时标定可以沿用相同样本
协议，并增加按 clock-sync capture timestamp 的联合优化。
