# Quest Dev Visualization Feature Protocol v1

连接、配对和 XR 生命周期由 core protocol v2 定义。visualization feature v1 目前只有
host → Quest 的 `robot_tcp` 下行；Quest → host 的手柄观测属于独立的
`controller_pose` feature，两个方向都没有控制消息。

## 本地几何体

黄色线框立方体由 browser feature 本地生成，不需要网络资源。它在每个 reference-space
generation 的首次有效 viewer pose 前方锚定。controller axes 直接使用同 generation
中的 `T_reference_from_grip`。发生 `reset` 时旧 marker、robot message 和任何对齐状态
立即失效。

## `robot_tcp`

可选只读 adapter 以配置频率发送。4x4 matrix 均为 column-major，平移位于索引
12、13、14，可直接传给 WebGL `Float32Array`。

```json
{
  "type": "robot_tcp",
  "feature": "visualization",
  "protocol_version": 2,
  "feature_protocol_version": 1,
  "server_timestamp": 1710000000.0,
  "robots": {
    "ar5_1": {
      "robot": "ar5_1",
      "key": "ar5:feedback:ar5_1",
      "fresh": true,
      "control_ready": true,
      "age_s": 0.02,
      "mode": "rt_joint_position",
      "frame": "robot_world",
      "position": [0.1, 0.2, 0.3],
      "orientation_quat_xyzw": [0, 0, 0, 1],
      "matrix_robot_from_tcp": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.2, 0.3, 1]
    }
  }
}
```

反馈缺失、过期、机器人身份不匹配或 pose 非法时，robot item 为 `fresh: false`，包含
`error`，且不含 position/quaternion/matrix。Redis 不可达时顶层含 `source_error`，
`robots` 为空。status 只记录 robot count，不记录 pose body。

## 坐标与绘制门槛

`matrix_robot_from_tcp` 仍在机器人 reference frame，不能直接在 Quest 中渲染。固定
手柄标定需要同时考虑未知 mount：

```text
T_quest_from_controller(i) = T_quest_from_robot × T_robot_from_tcp(i) × T_tcp_from_controller
```

只有经独立验收且绑定同一 robot epoch、tool profile、Quest session/reference generation
的 `T_quest_from_robot` 才能用于绘制。当前 R0 只输出 fit evidence，尚未定义或安装
acceptance profile，因此 robot TCP 绘制刻意保持关闭。详见
[paired-pose-calibration.md](paired-pose-calibration.md)。
