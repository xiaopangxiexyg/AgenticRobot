# RM65 腕部相机手眼标定

这个目录用于 **眼在手上** 标定：

- RealSense 相机固定安装在 RM65 手腕或末端执行器上。
- ArUco/ChArUco 标定板固定在场景中，采样期间不要移动。
- 不断移动机械臂末端，让相机从不同位置和角度观察标定板。
- 求解 `T_camera_to_gripper` 和 `T_gripper_to_camera`。
- 平移单位统一为米，RM65 姿态使用 `[x, y, z, rx, ry, rz]`，角度为弧度。

矩阵约定：

- `T_camera_to_gripper`：`p_gripper = T_camera_to_gripper @ p_camera`。这是 OpenCV hand-eye 的直接输出，也是把相机点云落到机械臂 base 时最常用的矩阵。
- `T_gripper_to_camera` / `T_end_effector_to_camera`：`p_camera = T_gripper_to_camera @ p_gripper`。这是上面的逆矩阵，也就是“末端到摄像机”的变换。
- 当前时刻相机到 base：`T_camera_to_base = T_gripper_to_base @ T_camera_to_gripper`。

代码使用 `hardware.rm65.rm_api2_adapter.Arm` 读取 API2 固件下的 RM65 状态。命令支持 `--arm left/right` 自动选择左右臂 IP，也可以用 `--arm-ip` 手动覆盖。

## 快速开始

先枚举 RealSense：

```bash
python -m utils.hand_eye --list-cameras
```

如果遇到 `RuntimeError: Couldn't resolve requests`，先查看相机支持的 profile：

```bash
python -m utils.hand_eye --list-cameras --show-profiles
```

程序默认会在请求的 `--resolution/--fps` 不支持时自动选择最接近的 color profile，并在启动后打印 `fallback_color_profile`。如果想手动指定更保守的 profile，可以用：

```bash
python -m utils.hand_eye \
  --camera 0 \
  --arm left \
  --resolution 640 480 \
  --fps 30
```

运行标定：

```bash
python -m utils.hand_eye \
  --camera 0 \
  --arm left \
  --target aruco \
  --marker-length-m 0.10 \
  --output-dir teleop_records/hand_eye
```

`--marker-length-m` 是 ArUco 黑白码单个 marker 的边长，单位是米。比如 99.7 mm 应写 `0.0997`，不是 `0.997`。

也可以把相机和左右臂绑定，避免 USB 顺序变化导致选错相机：

```bash
python -m utils.hand_eye \
  --arm left \
  --left-camera serial:261122273924 \
  --right-camera serial:213222079798 \
  --target aruco \
  --marker-length-m 0.10
```

显式 `--camera 0` 或 `--camera-serial 261122273924` 会优先于 `--left-camera/--right-camera`。`--left-camera` 和 `--right-camera` 支持 `index:0`、`serial:261122273924`、短数字 `0`、或直接写完整 serial。

默认就是高精度采集配置：

- color auto exposure 关闭，默认 `--color-exposure-us 8000`、`--color-gain 16`。
- color auto white balance 关闭，默认 `--color-white-balance 4600`。
- depth 默认关闭：`--no-enable-depth`。标定板识别只用 RGB；需要同时看深度流时再加 `--enable-depth`。
- depth emitter/projector 默认关闭：`--no-depth-emitter-enabled --depth-laser-power 0`。
- 单样本 burst 默认 2 秒，至少保留 20 帧有效检测。
- 默认按重投影误差剔除坏帧：`--max-reprojection-rmse-px 0.8`。
- 默认 OpenCV 初值后执行非线性精修：`--refine --refine-loss huber`。
- 默认自动离群点剔除：`--auto-reject-outliers --outlier-reject-rounds 2 --outlier-reject-fraction 0.10 --outlier-min-samples 20`。

自动记录模式：

```bash
python -m utils.hand_eye \
  --camera 0 \
  --arm left \
  --target aruco \
  --marker-length-m 0.10 \
  --auto-record \
  --auto-stable-sec 3.0 \
  --output-dir teleop_records/hand_eye
```

自动模式下，拖动机械臂到新姿态后松手；如果机械臂静止超过 `--auto-stable-sec`，并且相机能看到固定标定板，程序会自动记录一组稳定 burst。窗口底部会显示当前自动记录状态。

窗口按键：

- `r`：手动记录一组稳定采样。程序会在短时间内连续采相机和机械臂 pose，并拒绝明显晃动的数据。
- `d`：删除上一组。
- `s`：保存当前 raw samples。
- `c`：计算并保存标定结果。
- `q` 或 `Esc`：退出。

输出文件：

- `<name>.json`：主结果，包含矩阵、质量指标、全部 raw samples。
- `<name>.npz`：程序读取更方便的矩阵包。
- `<name>_gripper2camera.npy`：单矩阵文件，保存 `T_gripper_to_camera`。
- `<name>_cam2gripper.npy`：单矩阵文件，保存 `T_camera_to_gripper`。
- `<name>_samples.json`：自动保存的原始采样，可离线重算。

离线重算：

```bash
python -m utils.hand_eye \
  --load-samples teleop_records/hand_eye/你的_samples.json \
  --method AUTO \
  --refine \
  --auto-reject-outliers
```

可视化腕部相机点云到 RM65 base 坐标系：

```bash
python -m utils.hand_eye.visualize_handeye \
  --calibration teleop_records/hand_eye/你的.json \
  --camera 0 \
  --arm left
```

可视化会实时读取机械臂当前末端 pose，并使用 `T_gripper_to_base @ T_camera_to_gripper` 把深度点云投影到 base 坐标系。窗口中 `q` 或 `Esc` 退出。

## 采样建议

推荐使用 ChArUco 板，而不是单个 ArUco：

```bash
python -m utils.hand_eye \
  --camera 0 \
  --arm left \
  --target charuco \
  --charuco-squares-x 5 \
  --charuco-squares-y 7 \
  --charuco-square-length-m 0.035 \
  --charuco-marker-length-m 0.026
```

采样建议：

- 至少 25 组，推荐 40 组。
- 标定板必须固定在桌面、夹具或场景中，采集期间不要碰。
- 不要只平移末端；每个区域都加入明显 wrist 旋转，绕不同轴倾斜相机。
- 距离覆盖近/中/远；画面覆盖中心、四角、边缘。
- 每次记录前先让机械臂和相机静止；自动模式会默认等待 3 秒稳定。
- 避免运动模糊、过曝、反光和标定板弯曲。
- `c` 后看 residual：平移 RMS 越低越好。若超过几毫米，优先删除模糊/角度重复/边缘检测不稳的采样再重算。

程序每记录一组都会打印 `[plan]`：

- `dist near/mid/far`：近/中/远距离采样数量。
- `grid`：标定板中心落在画面九宫格中的覆盖格数。
- `wrist_rpy_span`：RM65 末端 RPY 三个轴的采样跨度。

结果 JSON 的 `metrics.sample_plan` 会保存同样的覆盖度分析，并把缺项写入 `quality_warnings`。

## 高精度求解

`c` 计算时会先调用 OpenCV `calibrateHandEye` 得到初值，然后用 `scipy.optimize.least_squares` 做全局精修。优化变量是：

- `T_camera_to_gripper`
- `T_target_to_base`

目标函数包含：

- 固定标定板反算到 base 后的 SE(3) 一致性。
- 如果 raw samples 中保存了角点，额外加入图像角点 reprojection residual。
- 默认鲁棒 loss 是 Huber，可用 `--refine-loss cauchy` 等方式切换。

如果当前环境没有 SciPy，程序会自动回退到 OpenCV 结果，并在 `metrics.nonlinear_refinement` 里记录 skipped。

自动离群点剔除流程：

- 按每个 sample 的 `target_in_base` 残差排序。
- 每轮删除最差的 `--outlier-reject-fraction`，默认 10%。
- 默认最多 2 轮。
- 至少保留 `--outlier-min-samples`，默认 20。
- 原始 samples 仍会 autosave 到 `<name>_samples.json`，结果 JSON 只包含最终用于求解的样本。

## 结果约定

JSON 中的关键矩阵：

- `T_camera_to_gripper`：相机坐标到 RM65 末端坐标。
- `T_gripper_to_camera` / `T_end_effector_to_camera`：RM65 末端坐标到相机坐标。
- `T_target_to_base`：固定标定板坐标到 RM65 base 坐标的估计，只用于质量评估和排查。

质量指标：

- `target_in_base_translation_rms_m`：同一块固定标定板反算到 base 后的平移一致性。
- `target_in_base_rotation_rms_deg`：同一块固定标定板反算到 base 后的旋转一致性。
- `robot_translation_span_m` / `robot_rotation_span_deg`：采样覆盖度。
- `target_reprojection_rmse_px_*`：视觉检测质量。
- `sample_plan`：近/中/远、画面九宫格、wrist RPY 跨度。
- `nonlinear_refinement`：非线性精修状态、loss、迭代次数和残差变化。
- `outlier_rejection`：自动剔除的轮次和被删除样本。

如果 `quality_warnings` 非空，建议按提示补采样或清理坏点后重新计算。
