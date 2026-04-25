# Dex Inspire Teleop 数据采集 Pipeline

本文档是当前仓库内从设备连接、虚拟验证、真机 teleop 验证、带单相机采集，到 SAPIEN+视频回放的完整流程。

默认假设在采集电脑上执行，工作目录固定为：

```bash
cd /home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy
```

## 0. 下载解压assets到项目路径: https://disk.pku.edu.cn/link/AA669C31FE27A74A01A2A3BADD61ADF559
文件名：assets.zip
有效期限：2026-06-08 14:10
提取码：YIlq


## 1. 固定设备配置

当前 README 中使用的设备配置如下：

| 设备 | 参数 |
| --- | --- |
| 左 RM65 | `192.168.101.21` |
| 右 RM65 | `192.168.1.20` |
| 左 InspireTac | `192.168.12.210` |
| 右 InspireTac | `192.168.11.210` |
| 左 q6 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00` |
| 右 q6 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00` |
| RealSense RGB | serial `213222079798`，camera key `front` |
| VIVE UDP | receiver 监听 `0.0.0.0:9999`，sender 目标使用采集电脑 IP，例如 `192.168.20.90` |

推荐先设置这些变量，后面命令直接复用：

```bash
export REPO=/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop_new
export PROFILE=$REPO/teleop_records/calibration_profiles/20260423_164018_vive_rm65_virtual_desk.json
export CAMERA_CONFIG=$REPO/teleop_runtime/configs/devices/realsense.json
export DATASET_ROOT=$REPO/dataset_collection/demo_bimanual_single_camera
export VIVE_TARGET_HOST=192.168.20.90

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
export LEFT_HAND_IP=192.168.12.210
export RIGHT_HAND_IP=192.168.11.210
export LEFT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00
```

## 2. 单摄像头配置

默认单相机配置文件已经固定到：

```text
teleop_runtime/configs/devices/realsense.json
```

当前内容是单路 `front`：

```json
{
  "front": {
    "serial": "213222079798",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "metadata": {
      "mount": "single_front"
    }
  }
}
```

采集时不要传 `--no-camera`，并保留默认 `--write-video`，writer 会输出：

```text
videos/observation.images.front/chunk-000/episode_000000.mp4
```

如果相机启动失败，先确认系统能看到该 serial：

```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.query_devices():
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

## 3. 启动 VIVE Sender

在 VIVE 机器或同一台采集电脑的独立终端(Windows)启动 sender：

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python hardware/VIVE/mocap/vive_data_sender.py \
  --host "$VIVE_TARGET_HOST" \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist \
  --tracker tracker_3:auxiliary
```

如果 sender 和 receiver 在同一台机器上临时自测，可以把 `VIVE_TARGET_HOST` 改成 `127.0.0.1`。



## 4. Virtual SAPIEN 验证

这一步只验证 VIVE profile、左右 wrist 映射、q6 手部映射和双臂 SAPIEN preview，不向真机发送运动。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode real \
  --control-mode absolute \
  --load-profile "$PROFILE" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --calibration-timeout 15 \
  --calibration-samples 20
```

检查点：

- 启动后按 `p` 做本次 session home capture。
- 左右 wrist tracker 映射正确，左手控制左臂，右手控制右臂。
- 桌面坐标方向正确，前/左/上不会反。
- q6 控制的 SAPIEN 手部开合方向正确。
- 如果渲染设备不可用，先加 `--skip-renderer` 验证主链路日志。

## 5. Real Teleop 真机短测

这一步接入双臂双手真机，但还不采集 LeRobot dataset。先做短时 teleop 验证，确认随动、触觉读取和频率稳定。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --load-profile "$PROFILE" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions thumb_1,index_1,middle_1,ring_1,little_1,palm \
  --hz 30 \
  --status-sec 1 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --visualize-tactile
```

检查点：

- 按 `p` 前双臂必须在安全起始位。
- `p` 后才进入本次 session 的 home 对齐和随动。
- 先小范围移动 wrist tracker，确认机械臂方向和速度安全。
- 手部 q6 能下发到 InspireTac。
- 触觉读取不要一开始就全 17 区域高频压力测试；先用上面少量区域和 `10Hz`。

## 6. 带单相机采集数据

确认上面两步稳定后，再启动正式采集。关键点是不要传 `--no-camera`，并显式传入单相机配置。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.collection.run_collect \
  --dataset-root "$DATASET_ROOT" \
  --task grasp_demo_camera \
  --task-description "bimanual InspireTac teleop with single RealSense front camera" \
  --active-sides both \
  --load-profile "$PROFILE" \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions thumb_1,index_1,middle_1,ring_1,little_1,palm \
  --visualize-tactile \
  --tactile-vis-value-scale 0.02 \
  --camera-config "$CAMERA_CONFIG" \
  --write-video \
  --hz 30 \
  --nominal-fps 30 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --status-sec 1
```

首次带相机短测可以临时追加：

```bash
--max-steps 300
```

采集按键流程：

| 按键 | 行为 |
| --- | --- |
| `p` | 起始位置标定，进入随动 |
| `z` | 开始录制当前 episode |
| `c` | 保存当前 episode |
| `d` | 丢弃当前 episode |
| `h` | 锁定机械臂并清除本次 `p` 标定 |
| `q` | 退出；若正在录制会先保存 |

采集后检查：

```bash
find "$DATASET_ROOT" -maxdepth 5 -type f | sort
```

至少应看到：

```text
meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/calibration/teleop_profile.json
meta/devices/realsense.json
data/chunk-000/episode_000000.parquet
videos/observation.images.front/chunk-000/episode_000000.mp4
```

## 7. 回放采集数据

SAPIEN 复刻动作、同步播放采集视频，并逐帧打印左右手 tactile sum：

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --video-mode window \
  --video-camera-key front
```

如果只想看采集视频和触觉，不启动 SAPIEN：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --skip-renderer \
  --video-mode window \
  --video-camera-key front
```

如果当前机器没有显示环境，只检查 parquet 和触觉：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --skip-renderer \
  --video-mode off \
  --max-frames 20 \
  --no-sleep
```

回放时默认使用：

- arm：`action.replay.<side>_arm_joint_target`
- hand：`action.replay.<side>_hand_joint_target`
- tactile：`observation.hand.<side>.tactile`
- video：`meta/episodes.jsonl` 中的 `video_paths` 或 parquet 中的 `observation.images.front.path`

每帧日志示例：

```text
[Tactile] replay_i=000000 frame=000000 t=0.000s left_sum=0.022 right_sum=0.242 total=0.264
```

## 8. 常见阻塞点

| 现象 | 优先检查 |
| --- | --- |
| 按 `p` 超时 | VIVE sender 是否在发 `tracker_1:left_wrist` 和 `tracker_2:right_wrist`，receiver `--host/--port` 是否匹配 |
| 左右臂反了 | tracker 名称、`--left-tracker-name`、`--right-tracker-name`、profile 是否匹配 |
| 机械臂方向不对 | 先回到 virtual SAPIEN 验证 profile，不要直接在真机上调 |
| 相机启动失败 | `realsense.json` serial 是否为 `213222079798`，`pyrealsense2` 是否能枚举设备 |
| 没有 mp4 | 是否误传 `--no-camera` 或 `--no-video`，`meta/episodes.jsonl` 的 `video_paths` 是否为空 |
| 回放没有视频窗口 | 使用 `--video-mode window`，并确认当前终端有 `DISPLAY` 或 `WAYLAND_DISPLAY` |
| 采集频率下降 | 降低 tactile 区域/频率，先用 `--hand-telemetry-hz 10`，相机长测前先做短测 |

## 9. 当前实现状态

已接入：

- 单 RealSense RGB 配置：`front`，serial `213222079798`
- 采集写出：parquet、meta、calibration、device config、mp4
- SAPIEN 回放：双臂双手 action replay
- 视频回放：OpenCV window，按 parquet 的 `video_frame_index` 同步
- 触觉统计：逐帧左右手 sum 和 total，结束时累计 sum

未覆盖的高风险测试：

- 长时间带相机真机采集压力测试
- 全 17 tactile 区域高频读取压力测试
- 多相机同步采集
