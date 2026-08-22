# Quest Dev Controller Pose Feature Protocol v1

该 feature 只上传当前 Quest WebXR session 中的手柄 `gripSpace` 观测，不发送动作、
目标或控制命令。消息只有在 core protocol v2 配对认证成功后才会被 server 接受。

## Session identity

浏览器为每个页面实例、XR session 和 `XRInputSource` 分配 session-local ID。WebXR
不提供可依赖的手柄硬件序列号，因此这些 ID 不能跨重连、session 或设备安装复用。

`xr_session_state=started` 建立：

- `client_instance_id`；
- `xr_session_id`；
- 实际取得的 `reference_space_type`；
- 初始 `reference_space_generation=0`。

reference space `reset` 必须把 generation 严格加一。server 会立即清除上一代的
latest pose，并拒绝旧 generation 的后续样本。session end 或 WSS disconnect 同样
清除 latest pose。

## `controller_pose_sample`

```json
{
  "type": "controller_pose_sample",
  "feature": "controller_pose",
  "protocol_version": 2,
  "feature_protocol_version": 1,
  "client_instance_id": "browser-...",
  "xr_session_id": "xr-...",
  "reference_space": {"type": "local-floor", "generation": 0},
  "sample_sequence": 42,
  "xr_frame_time_ms": 123456.7,
  "controllers": [
    {
      "source_session_id": "input-1",
      "handedness": "right",
      "target_ray_mode": "tracked-pointer",
      "profiles": ["oculus-touch-v3"],
      "emulated_position": false,
      "matrix_reference_from_grip": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 1.2, -0.5, 1]
    }
  ]
}
```

矩阵为 column-major `T_reference_from_grip`。host 严格校验 finite、齐次末行、
正交旋转和 determinant `+1`，限制 controller 数量、profiles 长度、序号及到达频率。
runtime 只保留每个认证客户端的 latest sample；不写文件、不形成轨迹，也不调用任何
机器人、Redis 或控制接口。需要 pairing bearer 的 `/api/status` 只暴露
session/generation/sequence/count 摘要，不暴露 pose matrix。
