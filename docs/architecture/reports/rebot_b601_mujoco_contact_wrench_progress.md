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

### 采样时序修正（2026-09-15）

MuJoCo `mj_step` 先执行 forward dynamics，再积分 `qpos/time`。因此直接在步进后读取
`sensordata/xpos` 并标注当前 `data.time` 会错配一个物理步长；接触刚进入或刚脱离时尤其明显。
读取函数现在用原生 `mj_copyData` 复制完整数据，并在独立副本上调用 `mj_forward`，从同一副本
读取接触力、作用点、末端原点和时间。主场景的积分、控制输入、solver working state、sensor 和
Cartesian cache 均不被改写；不通过从时间戳减去固定 `dt` 猜测采样时间。

调用必须位于 scene owner 线程，不能与步进并发。已注册 control callback 的模型被明确拒绝，
避免只读观测再次调用控制逻辑；当前 B601 场景通过外部设置 actuator inputs，不使用该 callback。
非有限完整 integration state、坏缓存 sensor、非法 contact count 或坏补算 sensor 均 fail closed，
不能借 forward 补算掩盖异常。本函数是按需诊断观测，不宣称副本分配/补算已满足高频 haptic servo
的性能预算；实时闭环需由后继控制 owner 明确安排采样阶段。

## 4. 当前验证

- 合成接触场景验证：障碍物位于末端 `+X` 侧时，`subtree2` 末端收到 `-X` 反力；多接触归并、
  作用点和力矩平移均可重复。
- 合成无接触场景严格输出零 wrench，不伪造作用点。
- 缺失 sensor、错误 sensor dimension 和非有限 sensordata 均被拒绝。
- 接触进入/脱离的步进后回归：旧实现两例均失败；修正后与当前积分状态独立 forward 的结果
  一致，且主场景数据逐字段保持不变。新增坏积分输入、control callback 和坏补算结果拒绝测试。
- 使用外部只读、固定 revision 的官方 B601 资产完成一次实际 MJCF 编译：`nbody=11`、
  `ngeom=35`、`nsensor=1`；默认姿态无接触，输出严格零。
- Task 工作树物化固定 gitlink 后，接触新增测试和既有 B601 MuJoCo、MIT、Real2Sim、haptics、
  control-composition 聚焦集合为 `175 passed`；最终门禁数字以 Harness Profile 为准。
- 采样时序修正后的接触聚焦集合为 `17 passed`；含上述既有行为及 MuJoCo haptic demo 的七文件
  回归集合为 `198 passed`（仅已有 `hppfcl` deprecation warning），Ruff、source compile 和
  `git diff --check` 通过。正式交接仍以修正 commit 上重新生成的 Harness Profile evidence 为准。

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
