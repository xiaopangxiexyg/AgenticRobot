# B601 MuJoCo 接触力观测开发进度

> 本文只记录 B601 Real2Sim 的 MuJoCo 接触场景与末端 wrench 观测切片。长期目标、阶段顺序和
> 真机门禁继续以
> [总路线图](../plans/rebot_b601_rm65_real2sim_hitl_roadmap.md) 为准；本文不授予真机运动、
> 合并、push 或安装权限。

## 1. 本批目标和边界

本批把 B601 的默认无碰撞验证模型扩展为一个**显式 opt-in** 的接触观测场景，形成后续
`simulated follower contact -> haptic feedback` 所需的原始输入契约。它不修改默认 position/
effort 场景，不引入第二套 simulator，也不把接触力直接发送给控制器或真机。

范围固定为：

- 复用现有通用 `SceneSpec`、ground、free-body object、collision filter 和接触材质编译器；
- 在 B601 场景中按需加入一个 MuJoCo 原生 contact sensor；
- 输出作用于 `gripper_end` 子树、在世界/基座轴方向表达、关于 `gripper_end` 原点的六维 wrench；
- 无接触返回严格零，缺失/畸形/非有限传感器数据 fail closed；
- 不进行滤波、缩放、passivity、触觉回传、绝对力标定或硬件 I/O。

## 2. 上游复用判断

审阅的 Seeed RS Web/MuJoCo 上游为 `Yang-Ci/ReBot_Arm_web_RS` 的
`dadefb0d0681501c41e6311ccc09045f836decd6`。该项目证明 RS physics scene 可用于 gravity、
contact、桌面物体和抓取，但其 ROS/Fake Driver 同步链没有可直接复用的、关于 B601 末端原点且
frame 明确的 contact-wrench 输出。其模型还来自另一个仓库，审阅版本未发现可支持复制资产的
明确 license，因此本批不复制 MJCF、URDF 或 mesh。

实际实现继续使用项目已固定的 `reBotArm_control_py` gitlink 和已有通用 MJCF 编译器，只复用
MuJoCo 原生 `sensor/contact` 能力。传感器限定 `subtree2="gripper_end"`，使方向明确为作用在
末端子树上的接触反力；`reduce="netforce"` 将多个接触归并为一个世界方向的等效 wrench。

## 3. 已实现契约

新增 `b601_contact_scene_spec()` 作为显式 contact profile 工厂，默认启用：

- SI 制、右手系、`Z-up`；
- `gravity=(0, 0, -9.81) m/s^2`；
- ground 和 B601 URDF collision geometry；
- self collision 与 parent-child exclusion；
- 可选复用现有 `FreeBodyObjectSpec` 添加物体。

`build_b601_mjcf_xml()` 与 `build_b601_mujoco_scene()` 新增两个默认关闭的参数：

```python
scene_spec = b601_contact_scene_spec()
scene = build_b601_mujoco_scene(
    rebot_source_root,
    gravity_mps2=scene_spec.gravity_mps2,
    actuation_mode="effort",
    scene_spec=scene_spec,
    enable_contact_observation=True,
)
```

场景 gravity 必须与 `SceneSpec` 完全一致；启用 contact observation 时必须提供 collision-enabled
scene。省略新增参数时，既有无碰撞 B601 position/effort 场景、actuator 和 metadata 路径保持
不变。

读取函数 `read_b601_contact_observation(scene)` 返回：

- `matched_contact_count` 和 `in_contact`；
- 生成该观察的 MuJoCo `physics_time_sec`，不伪造设备 source timestamp 或 host receive time；
- MuJoCo 报告的世界系等效作用点；
- `B601SpatialWrench`，frame 固定为
  `base_link/world_aligned@gripper_end`；
- `force_calibrated=false`，防止仿真数值被误称为真机绝对力标定结果。

MuJoCo `netforce` 给出的 moment 关于等效接触点，本批按下式平移到末端原点：

```text
tau_ee = tau_contact + (p_contact - p_ee) x force
```

该输出可以在后继批次进入已有 wrench filter/limit/frame mapping，再由唯一 torque owner 做
`J.T @ wrench`；当前观察器自身不拥有控制权。

## 4. 当前验证

- 合成接触场景验证：障碍物位于末端 `+X` 侧时，`subtree2` 末端收到 `-X` 反力；多接触归并、
  作用点和力矩平移均可重复。
- 合成无接触场景严格输出零 wrench，不伪造作用点。
- 缺失 sensor、错误 sensor dimension 和非有限 sensordata 均被拒绝。
- 使用外部只读、固定 revision 的官方 B601 资产完成一次实际 MJCF 编译：`nbody=11`、
  `ngeom=35`、`nsensor=1`；默认姿态无接触，输出严格零。
- Task 工作树物化固定 gitlink 后，接触新增测试和既有 B601 MuJoCo、MIT、Real2Sim、haptics、
  control-composition 聚焦集合为 `175 passed`；最终门禁数字以 Harness Profile 为准。

以上证据只证明 simulator contact observation 的结构、方向和 frame/origin 契约，不证明真实
接触力精度、触觉稳定性、passivity 或真机安全。

## 5. 后续切片

1. 在固定 B601 初态加入可复现的 virtual wall/box 接触场景，并记录接触、wrench、q/dq 与
   physics timestamp 的同一 tick trace。
2. 复用现有 haptics limit/filter 与 frame transform，将原始 contact wrench 变成独立的
   `HapticWrenchSample`；保持 motion channel 与 haptic channel 分离。
3. 建立 `F_slave -> F_master -> J_master.T @ F_master` 的纯仿真链，验证方向、功率不变量、
   saturation、slew 和 delay/jitter 下的有界能量。
4. 在完成仿真与 shadow 门禁前，不把本切片接入真机 torque command；绝对 force scale、摩擦、
   payload 和触感继续等待独立真机证据。
