# LeftArmFixed 采集模式

这里放两条复用 `teleop_runtime.collection.run_collect` 的左臂非操作者采集链路：

- `left-arm-fixed`：左臂固定在按 `p` 时读取到的当前位置。
- `left-arm-jitter`：左臂以按 `p` 时的位置为中心，末端做小范围随机平滑抖动，并和右臂一起走双臂 IK。

## 固定左臂行为

- 左臂：按 `p` 时同步真实 RM65 当前关节和 TCP pose，之后每帧发送同一个 `left_arm` joint hold command。
- 右臂：继续使用 `right_wrist_pose -> right_arm` 遥操作。
- 左手/右手：继续使用现有 q6 到 InspireTac 的遥操作和 telemetry/tactile 采集。
- 数据：仍使用双臂 schema，`action` 26 维、`observation.state` 38 维，左臂 qpos、pose、replay joint target 都保留，不会退化成单臂空槽。

固定左臂模式会关闭双臂 joint-space collision-aware retargeter，右臂走独立 `PoseToArmRetargeter`。运行前需要确认固定左臂不会处在右臂工作路径里。

## 固定左臂启动

推荐用专用入口，它会自动打开 `--left-arm-fixed`：

```bash
python -m teleop_runtime.LeftArmFixed.run_collect \
  --dataset-root /path/to/datasets \
  --task-name grasp_demo_left_arm_fixed \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --no-camera \
  --no-video
```

也可以直接用原 collector：

```bash
python -m teleop_runtime.collection.run_collect --left-arm-fixed ...
```


## 相机可视化

`LeftArmFixed` 两个入口都会直接复用 `teleop_runtime.collection.run_collect` 的相机链路，因此支持：

- `--camera-config`：加载 RealSense 配置。
- `--vis_cam` / `--vis-cam`：打开实时相机预览窗口。
- `--vis-cam-key` / `--vis-camera-key`：指定预览哪一路 camera key；不指定时优先预览 `front`，没有 `front` 时预览配置里的第一路。
- `--no-video`：只预览、不导出视频；保留视频时会继续按双臂 schema 写入数据集。

固定左臂模式带相机预览示例：

```bash
python -m teleop_runtime.LeftArmFixed.run_collect \
  --dataset-root /path/to/datasets \
  --task-name grasp_demo_left_arm_fixed_camera \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --vis_cam \
  --vis-cam-key front
```

抖动左臂模式同样支持相机预览，只需要在 `run_collect_jitter` 命令里加入相同的 `--camera-config --vis_cam --vis-cam-key front` 参数。

## 左臂小范围抖动 + 双臂 IK

这个入口会自动打开 `--left-arm-jitter`：

```bash
python -m teleop_runtime.LeftArmFixed.run_collect_jitter \
  --dataset-root ./test_datasets \
  --task-name grasp_demo_left_arm_jitter \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --left-arm-jitter-position-m 0.01 \
  --left-arm-jitter-rotation-deg 3 \
  --left-arm-jitter-refresh-sec 1.2 \
  --left-arm-jitter-smoothing-sec 0.45 \
  --vis_cam \
  --vis-cam-key front \
  --visualize-tactile \
  --hand-tactile-regions all
```

`--dataset-root` 是基础输出目录，实际数据集目录会自动命名为
`<dataset-root>/<YYYY-MM-DD>_<task-name>`；旧的 `--task` 仍可用作兼容入口。

抖动模式行为：

- 左臂：按 `p` 时记录当前 TCP pose 作为 anchor，之后生成围绕 anchor 的随机平滑 cartesian target。
- 抖动启停：只有按 `z` 开始录制后才推进随机抖动；录制前以及按 `c`/`d` 结束后，左臂保持静止。`c`/`d` 会把左臂停在当时当前位置，不会拉回 `p` 时的 anchor。
- 双臂 IK：左臂 jitter target 和右臂 wrist target 一起输入 `DualArmCollisionAwareRetargeter`，输出左右 arm joint command。
- 左手：继续用左 q6 控制左 InspireTac hand。
- 右臂：继续用右 wrist tracker 遥操作。
- 右手：继续用右 q6 控制右 InspireTac hand。
- InspireTac 启动 force：继承 collector 参数 `--inspire-hand-max-force N`，默认 `800`；兼容 `--max-fingertip-force` 和 `--max-force`。
- 数据：仍是双臂双手 schema，左臂 action/observation/replay joint target 来自双臂 IK 后的左臂命令和 backend 状态。

抖动参数：

- `--left-arm-jitter-position-m`：TCP 平移抖动最大半径，单位米。
- `--left-arm-jitter-rotation-deg`：TCP 姿态抖动最大角度，单位度。
- `--left-arm-jitter-refresh-sec`：多久采样一个新的随机目标。
- `--left-arm-jitter-smoothing-sec`：一阶平滑时间常数；越大越慢。
- `--left-arm-jitter-seed`：随机种子；负数使用系统随机源。

抖动模式需要双臂 IK，因此不能和 `--disable-dual-arm-collision-avoidance` 同时使用。

## 按键流程

按键和普通 collector 一致：

- `p`：同步左右臂当前真机状态；右臂采集 wrist home；左臂锁定当前位置。
- `z`：开始 episode。
- `c`：保存 episode。
- `d`：丢弃当前 episode。
- `h`：停止跟随并清除 home/fixed/jitter state。
- `q`：退出。

这两种模式都只需要右臂 wrist tracker 正常输入；左 wrist tracker 不参与运行时 home，也不是安全必需 channel。如果发送端仍然提供左 wrist tracker，collector 会把它作为非关键 channel 记录下来。
