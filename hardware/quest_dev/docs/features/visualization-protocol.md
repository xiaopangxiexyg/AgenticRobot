# Quest Dev Visualization Feature Protocol v1

所有消息均为 UTF-8 JSON WebSocket text frame。连接和生命周期消息由 core protocol 定义；机械臂数据只允许 visualization feature 从主机向 Quest 下行。

## `robot_tcp`

主机以配置的频率发送。所有 4x4 matrix 均为 column-major，平移位于索引 12、13、14，可直接传给 WebGL `Float32Array`。

```json
{
  "type": "robot_tcp",
  "feature": "visualization",
  "protocol_version": 1,
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

当反馈缺失、过期、机器人身份不匹配或 pose 非法时：

- `fresh` 为 `false`；
- 包含 `error`；
- 不包含 position、quaternion 或 matrix，避免客户端误绘制旧位姿。

Redis 自身不可达时，顶层消息包含 `source_error` 且 `robots` 为空。

该 feature 当前没有 Quest → host 的专用消息，也没有机械臂控制消息。通用生命周期消息属于 core protocol。

## 坐标契约

`matrix_robot_from_tcp` 仍处于机器人世界，不能直接在 Quest 中渲染。只有得到验证过的空间标定后才能计算：

```text
T_quest_from_tcp = T_quest_from_robot × T_robot_from_tcp
```

WebXR reference space 发生 `reset` 后，已有 `T_quest_from_robot` 必须失效。
