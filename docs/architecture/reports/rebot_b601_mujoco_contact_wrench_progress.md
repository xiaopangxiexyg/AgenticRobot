# B601 MuJoCo 接触力观测开发进度

> 本文只记录 B601 Real2Sim 的 MuJoCo 接触场景与末端 wrench 观测切片。长期目标、阶段顺序和
> 真机门禁继续以
> [总路线图](../plans/rebot_b601_rm65_real2sim_hitl_roadmap.md) 为准；本文不授予真机运动、
> 合并、push 或安装权限。

## 1. 本批目标和边界

本批把 B601 的默认无碰撞验证模型扩展为一个**显式 opt-in** 的接触观测场景，形成后续
`simulated follower contact -> haptic feedback` 所需的原始输入契约。它不修改默认 position/
effort 场景，不引入第二套 simulator，也不把接触力直接发送给控制器或真机。

初始 contact observer 切片的范围为（后续独立反馈装配见第 6 节）：

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

## 5. 物理接触闭环与同状态 trace（2026-09-15）

已有 `sim.mujoco.rebot_b601_rs.haptic_demo` 新增 opt-in `contact-box`，复用唯一的
`B601MujocoEffortDriver` 和原有 spring、gravity、damping、限幅/限速及状态门禁，不新增控制器或
生产入口。启动前将 2 cm、50 g 的自由 box 放到固定 B601 `gripper_end` 碰撞 mesh 的世界系最高
顶点上，初始几何重叠为 0.5 mm；这是可重复的仿真验收输入，不是真机 payload 或摩擦标定。
spring anchor 使用仿真初态的当前 FK，不另外规划一段机械臂运动。默认无碰撞数学 fixture 保持不变。

MuJoCo 约束求解器已经把接触反力施加到 plant。本切片**不将测得的 `J.T @ contact_wrench` 再加回
同一机械臂**，否则会重复施力。该投影只用于方向和 `F.T @ V == tau.T @ dq` 的诊断，真正送入
driver 的 overlay 仍然只有已有 spring。该单臂 fixture 不做 force reflection；独立双 plant
装配见第 6 节。

`--trace-jsonl PATH` 创建新的 JSONL（已有文件拒绝覆盖），按参考 tick 流式记录：

- 当前积分状态的 native physics time、driver tick time、q/dq、末端 FK pose、contact count；
- frame/origin 明确的物理 wrench、作用点及只读 `J.T @ wrench` 投影和两种 power；
- 求解该观察时尚未更新的 actuator ctrl、下一参考的 spring overlay、发送 admission 和实际
  actuator torque、该物理步的 interval end time。

采样阶段明确为 `current_state_before_reference_admission`：wrench 对应当前 q/dq 及上一安装的
ctrl，不冒充新参考执行后的接触力；新参考和后续物理步单独记录。native time 累加 dt，driver
time 来自整数 tick，比较只允许浮点 roundoff。主 JSON 只保留首尾 sample 和有界大小的统计，不把
完整 trace 装进内存。功率恒等式只验证投影数学，不宣称 passivity、实测功率分解或触觉稳定性。

从交接候选的 AgenticRobot worktree 根目录运行（纯仿真，不访问 CAN；不添加 `&`）：

```bash
(
  RUN_DIR="$(mktemp -d /tmp/b601-contact-box-XXXXXX)" || exit 1
  /home/zc/anaconda3/envs/dexteleop310/bin/python3.10 \
    -m sim.mujoco.rebot_b601_rs.haptic_demo \
    --rebot-source-root hardware/reBotArm_control_py \
    --fixture contact-box --duration-sec 1.5 \
    --trace-jsonl "$RUN_DIR/contact.jsonl" > "$RUN_DIR/result.json"
  printf 'sim exit=%s; results=%s\n' "$?" "$RUN_DIR"
)
```

需 GUI 时在同一入口追加 `--viewer`。预期看到机械臂保持在预置仿真姿态附近、轻载 box 与真实
collision mesh 接触并自由运动；未人工验证 GUI。关闭 viewer 或 Ctrl-C 只影响仿真进程，不改变
任何真机状态。要求无 fault、无 torque/overlay 限幅、至少出现非零物理接触 wrench、projection
power error < `1e-9 W`；同状态 trace 不应跳帧或混用步前/步后 q。

最小 1.5 s 实验：1500 physics ticks、750 reference ticks，接触力峰值约 `0.74 N`，
速度峰值约 `0.047 rad/s`，加速度峰值约 `10.35 rad/s^2`，无 driver fault 或 torque/overlay
限幅。这只是当前 seed 的观测结果，不是新的安全默认限制。新增 trace/单 torque-owner 测试、
无接触负对照、确定性重放与旧证据保护；新旧 contact/haptic 两文件聚焦验证为 `34 passed`。
七文件 B601 MuJoCo/MIT/Real2Sim/haptics/composition 回归为 `202 passed`（仅已有 `hppfcl`
deprecation warning），现有 CLI `--help`、Ruff 与 diff check 通过。正式候选状态和 Run ID 由
本次最终 Harness 交接给出，不能沿用前一候选的 Profile；排除 optional 的 fast/motion/sim-feedback
Profile 不单独证明新增 physical-contact 测试，以上聚焦结果与固定 gitlink 的实际模型仿真证据需要
独立保留。

## 6. 独立 follower -> master 力反馈切片（2026-09-15）

`robot_motion.rebot_b601_rs.haptics` 新增 `B601HapticWrenchSample` 和小型
`B601HapticWrenchChannel`，复用已有 wrench 类型与 co-adjoint 变换，不依赖 MuJoCo、真机或
teleop runtime。样本显式包含 sequence、source/receive timestamp、frame 和 valid；channel
只拥有单条反馈的 admission/filter memory，不拥有 actuator、gravity、motion source 或 dispatch。

通道顺序为：frame/sequence/time/age 验证 -> co-adjoint -> 非负 scalar gain -> 六维逐轴 clip
-> 按 consumer dt 计算的 LPF -> slew -> 最终逐轴 bound。首个输出从零开始。
`update(None)` 只在原始 source age 合法时保持样本，不能重写时间戳。wrong-frame、invalid、
out-of-order、future、expired 或非有限映射立即把反馈清零并清除滤波记忆，恢复需新合法样本；
这不改变 driver 的独立 gravity/damping/watchdog。反馈撤销不意味着 actuator 总力矩 slew 被绕过。

现有 CLI 新增 `--fixture force-reflection`，内部装配位于 `reflection_demo.py`；没有第二个
生产入口。两个独立 B601 ideal-effort plant 各复用唯一 `B601MujocoEffortDriver`：

```text
synthetic B601 follower + native mesh/box contact
    -> immutable current-state contact wrench
    -> optional local simulated delay
    -> bounded wrench channel
    -> existing master J.T mapping
    -> master spring + feedback overlay
    -> existing gravity/damping + final MIT clip/slew owner
```

follower 只执行自己的已有 spring，不重复施加 native contact reaction，也不受 master 位移
影响。这里的 follower 是用于验收的 B601 合成源，不是 RM65；RM65 后续仍必须通过其已有
canonical target 接入。当前无 teleop/policy/motion return channel，因此没有把反力导致的
master 位移冒充 human intent，也不宣称已实现 intent observer。

固定验收耦合为 tip-origin 的 `Ry(+90 deg)`、translation=0：follower 的 `-Z` 接触反力映射到
master 的 `-X`。不是把两个 base 的位置差塞进 wrench transform 产生伪 lever-arm torque。
通道使用每轴 force `2 N`、moment `0.2 Nm`、force slew `5 N/s`、moment slew `2 Nm/s`、
LPF tau `0.02 s`、source age ceiling `0.1 s`；这些只是明确的离线 seed，未改任何真机默认值。

trace 区分当前两端 q/dq、当前 source wrench、仍 active 的旧样本 source/receive time 和本
tick 新到达事件，随后记录 master spring、feedback、合成 overlay 与 driver admission。
co-adjoint power test 使用对应变换的**虚拟 source twist**；两个独立 plant 的实际速度没有被
强制设为相等。scalar gain、clip/LPF/slew 不保持 power invariance。名义 feedback work 按
held requested feedback torque 和步前 control qvel 累加，不能当成最终总力矩 slew 后的独立
功率分解。有限时域 `sum(tau_overlay_limit * dq_state_limit) * elapsed` 仅为有界输出且无
fault/clip 的本次 run 提供宽松上界，不是 passivity observer、energy tank 或长期触觉稳定性保证。

纯仿真前台命令（从最终候选 worktree 根目录运行，不访问 CAN）：

```bash
(
  RUN_DIR="$(mktemp -d /tmp/b601-force-reflection-XXXXXX)" || exit 1
  LOG="$RUN_DIR/command_and_output.log"
  CMD=(/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
       -m sim.mujoco.rebot_b601_rs.haptic_demo
       --rebot-source-root hardware/reBotArm_control_py
       --fixture force-reflection --duration-sec 1.5 --feedback-delay-sec 0.04
       --trace-jsonl "$RUN_DIR/feedback.jsonl")
  { printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'; } > "$LOG"
  env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl "${CMD[@]}" \
    > "$RUN_DIR/result.json" 2>> "$LOG"
  SIM_STATUS=$?
  printf 'sim exit=%s; results=%s\n' "$SIM_STATUS" "$RUN_DIR"
)
```

需要 GUI 时，移除 `MUJOCO_GL=egl` 并追加 `--viewer`。该 viewer **只显示 master**，不是两个
场景拼接显示；follower 的物理接触见 result/trace 或第 5 节单臂 `contact-box --viewer`。
固定 seed 位移很小，不能用“肉眼看起来没有动”替代 trace 验收。GUI 尚未人工验证。

聚焦覆盖保留了旧 fixture 行为，并新增：源数据 freshness/sequence/frame 拒绝、即时清零与
重启、LPF/slew、带 rotation/lever-arm 的映射后 clip、非法参数/metadata 和映射 overflow；
两个独立 driver 的实际发送 spy、0/40 ms 延迟的完整双 plant trace、零 gain 的无反馈负对照、
120 ms 延迟过期拒绝，以及现有 CLI 的正向/非法选项路径。七文件回归 `229 passed`，Ruff 与
diff check 通过。40 ms seed 两端各 1500 physics ticks、750 reference ticks，无 fault 或
torque/overlay clip；master 反馈力矩峰值约 `0.25 Nm`，名义 delivered work 约 `0.0041 J`。
这些结果不是硬件或辨识证据；正式 Profile 必须在本次最终 commit 上重跑。

## 7. 后续切片

1. 当前 box 接触 trace 已完成；后续扩展更明确的接触方向/工作空间对照及固定障碍物场景时，
   继续复用通用 scene compiler，不能为了测试重复施加 native 接触反力。
2. 原始 contact -> timestamped bounded wrench channel -> 独立 master J.T 的合成仿真链已落地。
   后续按总路线图将 RM65 contact/source adapter 接入 canonical target，不复制 RM65 execution loop。
3. 继续验证 jitter/dropout、更多接触方向和物理参数下的行为；需要能量稳定性声明时独立实现
   passivity/energy-tank 门禁，不能用本次有限时域 work 上界替代。
4. 在完成仿真与 shadow 门禁前，不把本切片接入真机 torque command；绝对 force scale、摩擦、
   payload 和触感继续等待独立真机证据。
