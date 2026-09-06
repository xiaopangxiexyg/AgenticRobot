# Quest Dev Visualization Feature Protocol v2

连接、配对和 XR 生命周期由 core protocol v2 定义。visualization feature 只有
host -> Quest 的只读 robot_tcp 下行；Quest -> host 的手柄观测属于独立的
controller_pose feature，两个方向都没有控制消息。

## 本地几何体

黄色线框立方体由 browser feature 本地生成，不需要网络资源。它在每个
reference-space generation 的首次有效 viewer pose 前方锚定。controller axes 直接使用
同 generation 中的 T_reference_from_grip。发生 reset 时旧 marker、robot message 和
任何对齐状态立即失效。

## robot_tcp

可选只读 adapter 以配置频率发送。4x4 matrix 均为 column-major，平移位于索引
12、13、14，可直接传给 WebGL Float32Array。

未配置 reviewed frame binding 时，robot item 仍可包含机器人 reference frame 的诊断 pose，
但不会包含 Quest-space 矩阵：

~~~json
{
  "type": "robot_tcp",
  "feature": "visualization",
  "protocol_version": 2,
  "feature_protocol_version": 2,
  "server_timestamp": 1710000000.0,
  "robots": {
    "ar5_1": {
      "robot": "ar5_1",
      "key": "ar5:feedback:ar5_1",
      "fresh": true,
      "alignment_ready": false,
      "control_ready": false,
      "frame": "robot_world",
      "matrix_robot_from_tcp": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.2, 0.3, 1]
    }
  }
}
~~~

只有显式加载 accepted_frame_binding.v1 且机器人 pose 的 nested pose_binding 元数据
逐字段匹配时，host 才附加：

~~~json
{
  "alignment_ready": true,
  "matrix_quest_from_tcp": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1.1, 2.2, 3.3, 1],
  "frame_binding": {
    "binding_id": "binding-1",
    "acceptance_receipt_sha256": "…",
    "quest_session_id": "xr-session-1",
    "reference_space_type": "local-floor",
    "reference_space_generation": 0,
    "quest_reference_epoch_id": "xr-session-1:local-floor:0",
    "reference_frame_id": "quest-local-floor"
  }
}
~~~

反馈缺失、过期、机器人身份不匹配、pose 非法、binding 缺失或 source/tool/FK/
coordinate-binding identity 不匹配时，robot item 为不可投影状态，且不含
matrix_quest_from_tcp。Redis adapter 不会把旧的裸 Redis payload 当作 canonical
机械臂 pose；canonical passive pose journal publication 通过
project_pose_publication() 投影，Redis 传输若要启用 Quest 投影必须携带同一份
nested pose_binding identity。

## 坐标与绘制门槛

固定手柄标定的观测模型仍为：

~~~text
B_i = X · A_i · Y

A_i = T_robotReference_from_tcp(i)
B_i = T_questReference_from_controllerGrip(i)
X = T_questReference_from_robotReference
Y = T_tcp_from_controllerGrip
~~~

后续 TCP 绘制不需要把 controller mount 偏置重新乘进去，只使用：

~~~text
T_questReference_from_tcp(i) = X · A_i
~~~

hardware.quest_dev.features.visualization.frame_binding 定义并严格解析 reviewed
quest_dev.visualization.accepted_frame_binding.v1。binding 同时绑定：

- acceptance receipt、capture evidence、diagnostic fit、dataset 和 nested fit hashes；
- Quest session、reference-space type/generation、reference epoch 和 controller source；
- robot source instance/epoch、tool profile、FK contract、coordinate binding hash、
  reference/moving frames 和 pose schema；
- T_questReference_from_robotReference 的 nested 与 column-major 两种表示。

该 parser 验证 canonical hash 和所有冗余矩阵表示，但不会创建 acceptance receipt，也
不会把 candidate-only fit 自动升级为 accepted binding。当前默认配置中的
frame_binding_paths 和 frame_bindings 均为空。

browser 只在以下条件同时满足时绘制 TCP axes：

1. message protocol version 是 v2；
2. robot item fresh=true、alignment_ready=true 并带 16 元素有限
   matrix_quest_from_tcp；
3. frame_binding.quest_session_id 等于当前 XR session；
4. reference-space type、generation 和 derived quest_reference_epoch_id 全部匹配；
5. visualization feature 未关闭 TCP axes。

reference-space reset、XR session end、协议不匹配或 binding identity 不匹配会立即清空
旧 robot message，停止绘制。这个路径始终是只读观测，不发送机械臂动作，也不产生
future trajectory 或控制目标。
