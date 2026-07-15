# RM65 与相机手眼标定

这个目录是从 `hardware/old_headeye` 迁移后的新入口，用于 **眼在手外** 标定：

- 相机固定在工位上。
- ArUco/ChArUco 标定板固定在 RM65 末端。
- 求解 `T_camera_to_base`，即 `p_base = T_camera_to_base @ p_camera`。
- 平移单位统一为米，RM65 姿态使用 `[x, y, z, rx, ry, rz]`，角度为弧度。

新代码使用 `hardware.rm65.rm_api2_adapter.Arm` 读取 API2 固件下的 RM65 状态，旧脚本里 SDK 相关的 TODO 已经不需要再手动改。
命令支持 `--arm left/right` 自动选择左右臂 IP；`--arm-ip` 仍可手动覆盖。
手眼标定默认不绑定具体末端执行器；只有在你明确需要检查 RM65 当前 active tool frame 时，才额外使用 `--expected-tool`。

## 快速开始

先枚举 RealSense：

```bash
python -m utils.head_eye --list-cameras --show-profiles
```

运行标定：

```bash
python -m utils.head_eye \
  --camera 0 \
  --arm left \
  --target aruco \
  --marker-length-m 0.10 \
  --output-dir teleop_records/headeye
```

自动记录模式：

```bash
python -m utils.head_eye \
  --camera 0 \
  --arm left \
  --target aruco \
  --marker-length-m 0.0997 \
  --auto-record \
  --auto-stable-sec 4.0 \
  --output-dir teleop_records/headeye
```

自动模式下，拖动机械臂到新姿态后松手；如果机械臂静止超过 `--auto-stable-sec`，并且相机能看到标定板，程序会自动记录一组稳定 burst。窗口底部会显示当前自动记录状态。

### RealSense profile 兼容

默认先严格使用 `--resolution` / `--fps` 请求的 color + depth profile。仅当这个请求被相机拒绝时，才会自动选择一个同时存在的、分辨率和帧率匹配的 `bgr8` color / `z16` depth profile；可用 `--no-auto-select-supported-profile` 关闭该兜底并直接报错。

`--list-cameras --show-profiles` 可在实际设备上列出可用 profile。新保存的 samples 与结果 JSON 会记录请求 profile、是否发生 fallback，以及实际选择的 profile。该字段只用于溯源：不会改写既有 JSON/NPZ/NPY 标定结果，旧结果仍可直接加载。

窗口按键：

- `r`：手动记录一组稳定采样。程序会在短时间内连续采相机和机械臂 pose，并拒绝明显晃动的数据。
- `d`：删除上一组。
- `s`：保存当前 raw samples。
- `c`：计算并保存标定结果。
- `q` 或 `Esc`：退出。

输出文件：

- `<name>.json`：主结果，包含矩阵、质量指标、全部 raw samples。
- `<name>.npz`：程序读取更方便的矩阵包。
- `<name>_cam2base.npy`：兼容旧流程的单矩阵文件。
- `<name>_samples.json`：自动保存的原始采样，可离线重算。

每份新 JSON 的 `metadata.realsense_settings_report` 还会保存相机 stream 选择记录，便于判断该次标定的内参与 profile 是否与其他结果一致。

离线重算：

```bash
python -m utils.head_eye \
  --load-samples teleop_records/headeye/你的_samples.json \
  --method AUTO
```

离线重算会保留 samples 中原有的相机 profile 溯源，并在新结果中标记 `source_samples_metadata`；它不会改写输入 samples 或任何既有结果。

可视化点云到 RM65 base 坐标系：

```bash
python -m utils.head_eye.visualize_handeye \
  --calibration teleop_records/headeye/你的.json \
  --camera 0
```

可视化默认使用 OpenCV 三视图投影和 NumPy 深度反投影，不依赖 Open3D。窗口中 `q` 或 `Esc` 退出。

如果只标定了其中一只机械臂，也可以用双臂 URDF 中的固定 base 外参切换显示坐标系。比如已完成 right 标定，希望按 left base 查看同一个相机点云：

```bash
python -m utils.head_eye.visualize_handeye \
  --calibration teleop_records/headeye/你的_right标定.json \
  --calibration-arm right \
  --view-arm left \
  --camera 0
```

也可以把派生出的 left headeye 结果保存成独立文件：

```bash
python -m utils.head_eye.dual_arm \
  --calibration teleop_records/headeye/你的_right标定.json \
  --source-arm right \
  --target-arm left \
  --output-dir teleop_records/headeye
```

该转换默认读取 `assets/rm65_inspire_tac/rm65_dual_arm.urdf`：`T_camera_to_left_base = T_right_base_to_left_base @ T_camera_to_right_base`。反向同理。

## 高精度建议

推荐使用 ChArUco 板，而不是单个 ArUco：

```bash
python -m utils.head_eye \
  --camera 0 \
  --arm left \
  --target charuco \
  --charuco-squares-x 5 \
  --charuco-squares-y 7 \
  --charuco-square-length-m 0.035 \
  --charuco-marker-length-m 0.026
```

采样建议：

- 至少 15 组，推荐 20 到 30 组。
- 不要只平移末端；每个区域都加入明显 wrist 旋转，绕不同轴倾斜标定板。
- 让标定板覆盖画面中心、四角、近处和远处。
- 每次记录前先让机械臂和标定板静止；自动模式会默认等待 3 秒稳定。
- 避免运动模糊、过曝、反光和标定板弯曲。
- `c` 后看 residual：平移 RMS 越低越好。若超过几毫米，优先删除模糊/角度重复/边缘检测不稳的采样再重算。

## 结果约定

JSON 中的关键矩阵：

- `T_camera_to_base`：相机坐标到 RM65 base 坐标，遥操作/点云落 base 通常用这个。
- `T_base_to_camera`：上面的逆。
- `T_target_to_gripper`：标定板坐标到末端坐标，用于质量评估。

这里的 `gripper` 是程序读取到的 RM65 末端位姿坐标系。若后续确认当前 SDK 返回值会随 active tool frame 改变，则应固定同一个 frame 重新采样；`--expected-tool` 只作为可选一致性检查，不是手眼标定流程的必需参数。

质量指标：

- `target_in_gripper_translation_rms_m`：同一块标定板反算到末端后的平移一致性。
- `target_in_gripper_rotation_rms_deg`：同一块标定板反算到末端后的旋转一致性。
- `robot_translation_span_m` / `robot_rotation_span_deg`：采样覆盖度。
- `target_reprojection_rmse_px_*`：视觉检测质量。

如果 `quality_warnings` 非空，建议按提示补采样或清理坏点后重新计算。
