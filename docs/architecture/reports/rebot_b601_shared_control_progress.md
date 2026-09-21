# B601 共享控制开发进度

> 总目标和跨阶段门禁见
> [ReBot B601-RS 与 RM65 双向 Real2Sim / HITL 路线图](../plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)。
> 本文只记录增量事实和证据，不声明真机验收、合并、部署或硬件安全已经完成。

## 1. 当前控制边界

目标控制链保持单一真实执行 owner：

```text
timestamped policy intent
    -> latest-value admission
    -> bounded policy-assist torque
    -> existing gravity + total-torque clip/slew
    -> existing MIT dispatch/protective hold owner
    -> measured B601 motion
    -> existing B601 -> RM65 mapping
```

Policy 不持有 CAN/backend，不直接发送 MIT 帧；人工推动也不是第二个命令源。Policy stale、
wrong-frame、无效、重放或显式 clutch 时，assist 归零，重力补偿和被动阻尼继续由原 owner
负责。J7 仍是独立通道，不进入六轴共享控制。

## 2. 已完成增量

### P0：纯控制语义

`robot_motion/rebot_b601_rs/shared_control.py` 提供：

- 带 sequence、source/receive timestamp、frame 和 validity 的 policy joint intent；
- latest-value buffer，不积压旧 action；
- target error、assist torque、assist slew 和 authority ramp 四层限制；
- 显式 policy clutch；
- override 后必须收到更高 sequence 的重新锚定指令，旧目标不能恢复；
- stale/future/invalid/replayed command 退回零 assist。

### P1：MIT overlay 适配

`robot_motion/rebot_b601_rs/shared_control_runtime.py` 将 policy intent 转换为：

```text
qref = 当前测量 q
qvel_ref = 0
vendor/software position Kp = 0（由 owner 契约保证）
joint_torque_overlay = bounded policy assist
```

该模块无 CAN、MotorBridge 或 MuJoCo 依赖。最终 gravity、driver-local damping、总 torque
clip/slew 和 dispatch 仍由实际 owner 完成。

### P2：MuJoCo 生命周期验收

现有 `sim.mujoco.rebot_b601_rs.haptic_demo` 增加 opt-in `policy-assist` fixture，没有创建
第二个 simulator/入口。Fixture 覆盖 passive、policy assist、human override、等待新锚点、
recovery 和 stale 回退。

理想 URDF plant 使用 raw-model gravity baseline，以隔离共享控制逻辑；操作者标定的
gravity scale/bias 仍属于真机 controller profile。自动验收额外要求：

- clutch 和等待重锚期间 assist 为零；
- J6 assist 不反向，避免把 fixture 新命令误看成旧目标回弹；
- J1--J5 非目标最大漂移不超过 `0.005 rad`；
- 最终状态为 passive，driver 无 fault；
- hardware claim 始终为 false。

`--output-root` 会建立唯一运行目录，将完整 JSON 写入 `result.json`，终端只显示摘要。

人工 MuJoCo 观察已经确认 J1--J5 无明显运动；第一版 recovery 反向增量造成的视觉“回弹”
已定位并移除。修复后的同向/无回拉行为仍需作为本阶段人工验收项保留。

### P3：现有真机 owner 的注入边界

`scripts/hardware/rebot_b601_rs_low_gain_probe.py` 现在允许上层以显式关键字参数注入
`B601SharedControlRuntime`。该路径：

- 只允许复用现有 `hold` owner；
- 强制 software Kp/Kd 为零；
- 禁止与 Cartesian/pose-axis controller 叠加；
- 使用当前测量 q 作为无刚度 MIT reference；
- 将 assist 放入现有 overlay，再经过原 gravity、总力矩 absolute/slew conditioner；
- assist 独立上限不得超过最终软件 torque envelope；
- 每 tick 记录 mode、authority、reason、sequence、target error 和 limit 状态；
- 普通 CLI 不构造 runtime，因此既有 hold/commissioning 行为不变。

这只是程序内接线边界，还不是可运行的真机 shared-policy application。当前没有新增 socket、
JSONL 轮询、第二个 CAN owner 或后台进程。

### P4：复用 RM65→B601 重定向作为 policy 意图

`robot_motion/rebot_b601_rs/rm65_policy_assist.py` 复用既有
`RM65ToB601Retargeter` 的坐标变换、anchor-relative mapping 和连续种子 IK，不另建映射或
求解器。它只把当前 tick 接受的 B601 qpos 目标发布到同进程
`B601SharedControlRuntime`：

- policy sequence 直接继承已验证递增的 RM65 sample sequence；
- policy source timestamp 使用 host-local monotonic receive timestamp，不混用设备 wall clock；
- stale、frame、replay、IK 或 policy publish 拒绝会立即撤销上一条 assist；
- 撤销保留已接受 sequence 高水位，迟到旧包不能重新激活；
- policy clutch 同时撤销 assist 并清空 retarget anchor，释放后必须显式 recenter；
- bridge 不持有 CAN、MotorBridge、MuJoCo 或 gravity/dispatch 生命周期。

该阶段建立的是“RM65 运动意图 → B601 bounded assist”的纯软件接线，并不表示已经接入
真实 policy provider 或真机执行。

### P5：真机单关节助力调试协议（待人工验收）

现有 `rebot_b601_rs_owner.py` 增加默认关闭、二次确认的一次性单关节助力协议，仍由原
gravity/damping owner 独占 CAN。协议不复用会固定启动位置的 `slow-step`：

- 到达 `STARTUP ACTIVE` 后仍保持纯重力补偿和 driver-local damping；
- 操作者释放机械臂并按 `A` 后，才从当帧完整、新鲜且静止的实测 q 捕获相对目标；
- 倒计时内检测到运动会拒绝该锚点，保持零助力并要求重新按 `A`；
- 助力只作用于显式目标关节，并经过 target-error、assist-torque、assist-slew、authority
  ramp 以及原 owner 的最终总力矩限制；
- `C`、试验时长完成、stale/fault 或 owner 退出都会撤销 policy assist；撤销后 reference
  继续跟随当前实测 q，不生成返回捕获位置的目标；
- 原第一 Ctrl-C/protective hold、第二 Ctrl-C/SIGUSR1 全失能语义不变。

首轮默认面向 J6：相对偏移 `0.10 rad`，误差上限 `0.05 rad`，助力上限 `0.05 Nm`，
助力力矩变化率 `0.10 Nm/s`，权限上升 `0.5/s`，持续 `4 s`。这些是可调整的调试参数，
不是硬件认证限值。自动测试不连接 CAN，也不构成真机通过证据。

首轮现场运行 `/tmp/b601-rs-commissioning/b601-owner-efstt0pz` 证明六轴启动、20 Hz
gravity/damping 控制、protective hold、驱动器限矩恢复和最终 safe stop 完整；但协议事件为空、
状态停留在 `waiting`，因此没有发送 J6 assist，不能作为助力验收。顶层 `FAIL` 来自
`commissioning_acceptance=INCOMPLETE`，不是控制或 cleanup 失败。

该次还暴露了前台 launcher 的 SIGINT 缺陷：父子进程同属前台 process group，均已直接
收到 Ctrl-C，父进程又重复转发并提前关闭 stdout，导致子进程结果输出出现 broken pipe，
表现为退出码 120。修复后父进程只观察 SIGINT，由子进程独占两阶段停止语义，并持续读取
输出至 EOF；控制时限届满但助力未完成时也会显式撤销 policy 并显示
`ASSIST INCOMPLETE`。一次性助力正常完成后会直接进入 protective hold，不再要求操作者
等待总 duration；duration 仅作为等待 `A`/完成试验的最长窗口。

第二轮现场运行 `/tmp/b601-rs-commissioning/b601-owner-6s8jamp5` 已完成按键、倒计时、
assist 注入、撤销、protective hold 和最终 safe stop。控制环为 `20.012 Hz`，81 个 active
sample 中 J6 assist 确实达到 `0.05 Nm`，无 torque/slew/overlay limit，所有完整六轴命令
均已发送。但 J6 峰值位移仅 `0.000383 rad`（`0.022 deg`，一个 Type-24 位置量化台阶），
末五帧均值仅 `0.000153 rad`（`0.009 deg`）。因此这不是按键或 dispatch 故障，而是该
姿态下 `0.05 Nm` 未克服 J6 的静摩擦/死区。

该证据同时暴露了原验收口径缺陷：状态机进入 `COMPLETE` 曾被直接写成 `PASS`。当前候选
已将协议完成与机械验收分离；真机 PASS 还要求助力实际达到配置比例、目标关节产生至少
`0.1 deg` 的正确方向稳定响应、响应不过量且 J1--J5 漂移均不超过 `0.5 deg`。终端只显示
一行 `ASSIST RESPONSE` 摘要，完整测量保存在结果 JSON。旧证据按新口径为
`INSUFFICIENT_RESPONSE`，不能再冒充真机助力通过。
相同边界也在 active 阶段逐帧生效：错误方向、目标关节过量位移或任一支撑关节过量漂移
会先撤销 assist，再进入既有 protective-hold 故障路径，而不是等试验结束后才报告。

后续两档现场证据进一步关闭了“命令未送达”的假设：`0.15 Nm` 时 J6 峰值响应约
`0.000920 rad`（`0.053 deg`），仍低于 `0.1 deg` 验收门槛；`0.30 Nm` 时发生明确运动，
并在 J6 相对捕获位置达到 `0.181398 rad` 时被 `0.15 rad` active guard 撤销。后者证明
静摩擦已经被跨过，也证明过量运动保护路径有效，但它是保护触发的 FAIL，不是可接受的
单关节助力参数。该结果终止继续按关节力矩加档，下一阶段改为任务空间受限助力。

### P6：基于 TCP 空间位置的受限助力（离线与 MuJoCo 已通过，真机未运行）

一次性 Cartesian assist 继续复用同一 gravity/damping owner、现有
`B601TranslationalVirtualSpring`、`LOCAL_WORLD_ALIGNED` Jacobian 和
`J.T @ wrench` 映射，不创建第二个硬件 owner。按 `A` 后从新鲜、静止的完整六轴状态
捕获 TCP；目标是捕获点沿 `base_link` 世界对齐 X/Y/Z 的显式小偏移，而不是任一关节角
目标。MuJoCo Viewer 也使用相同的显式触发语义：启动后助力严格为零，操作者可先把模型
放到有足够运动空间的位置并等待静止，再在启动命令所在终端按 `A`（无需 Enter）捕获当时
TCP；运动未静止时该次 `A` 被拒绝且可重试，`C` 随时撤销，`Q` 撤销并退出。Viewer 自身的
`A/C` 回调保留为备用，但不再要求窗口焦点。终端 cbreak 生命周期复用现有遥操前台的同一
公共实现，退出时恢复终端设置；无界面的确定性回归才会在固定 settle 时间后自动触发，不能
用它替代人工 Viewer 验收。当前保守默认值为：

- 目标轴偏移 `10 mm`，刚度/阻尼 `50 N/m`、`3 N.s/m`；
- 两个非目标平移轴保持捕获位置，刚度/阻尼 `30 N/m`、`2 N.s/m`；
- 每轴 Cartesian force 上限 `0.5 N`；经 Jacobian 映射后，每关节附加力矩再独立限制为
  `0.15 Nm`，变化率 `0.10 Nm/s`，authority 以 `0.5/s` 上升；
- active guard 同时检查目标轴方向与最大响应、横向漂移 `5 mm` 和姿态漂移 `5 deg`；
- `C`、完成或任何 guard 拒绝都会立即把 assist 清零，不生成返回捕获点的关节目标，随后
  保持原重力补偿、阻尼和 protective-hold 语义。

横向保持力是 MuJoCo 暴露出的必要结构，而不是为了通过测试放宽门槛：只在 X 轴施力时，
理想模型的横向漂移达到约 `16.1 mm`，会被同一 `5 mm` 真机 guard 正确拒绝。加入较弱的
Y/Z 空间保持后，确定性 6 秒 MuJoCo 结果为：X 正向最终位移约 `9.987 mm`、最大横向漂移
约 `4.813 mm`、最大姿态漂移约 `0.820 deg`、峰值映射关节附加力矩约 `0.0321 Nm`，无
overlay/总力矩限幅或 driver fault；6 秒助力结束后 overlay 已撤销，额外 1 秒 passive
观察期间没有返回旧锚点。加入按键触发后，无界面回归仍得到相同结构的通过结果，完整结果
位于
`/home/zc/HandMotion/Developing/temp/b601-rs-commissioning/b601-cartesian-assist-l37v_gdn/result.json`。

空间助力 guard 以及既有单关节 guard 的拒绝不再通过异常路径丢失试验上下文；完整候选
会记录 guard 原因、fresh feedback 时间、实测 q/qvel、捕获锚点/目标和
`command_dispatched=false`，然后进入既有 protective hold。该调整只改善故障证据，不
放宽任何保护阈值。

Viewer 的目标轴、带符号位移、轴向刚度、每轴力上限、每关节投影力矩上限和横向保持参数
均可通过独立 CLI 参数调整。单独把目标位移从 `10 mm` 放大到 `30 mm` 会在当前理想模型
产生约 `13.6 mm` 瞬态横向漂移，不能通过既有 `5 mm` 边界；因此大范围可视化参数不能只
改位移。采用 `30 mm`、`1.0 N`、`0.30 Nm`，并把横向保持刚度/阻尼调整为
`120 N/m`、`4 N.s/m` 后，正式 CLI 的无界面证据达到目标轴 `30.000 mm`、最大横向漂移
`4.330 mm`、最大姿态漂移约 `1.047 deg`，结果保存在
`/home/zc/HandMotion/Developing/temp/b601-rs-commissioning/b601-cartesian-assist-suk73xpk/result.json`。

## 3. 当前证据边界

- 纯控制、runtime、MuJoCo 和低增益 probe 单元/契约测试通过；
- MuJoCo 只证明方向、生命周期、边界和定性运动；
- P3/P4 尚未连接真实 policy provider；P5 的单关节命令链已证明可产生运动，但
  `0.30 Nm` 试验因过量响应被 guard 拒绝；
- P6 的空间助力仅有纯控制/确定性 MuJoCo 证据，尚未发送任何真机 Cartesian wrench；
- 没有自动人工外力检测、policy replan、真实 torque sensing 或 passivity 声明；
- 当前候选不合并、不 push、不安装。

## 4. 下一步

1. 先由操作者在 MuJoCo viewer 中确认 P6 的 +X 空间运动、非目标轴稳定和退出行为；当前
   候选保持未合并。
2. Viewer 验收后再生成一次独立、前台、+X、`0.5 N` 上限的真机命令。首轮只测试 X，
   不同时测试 Y/Z；任何 guard、stale feedback 或异常都会撤销助力并保留 protective hold。
3. X 方向真机证据通过后才按相同流程逐轴验证 Y/Z；不同轴的 Jacobian 投影力矩不同，不能
   从 X 的证据外推。
4. 在 application leaf 中复用 P4 bridge 装配真实 policy producer、显式 policy clutch 与
   同进程 runtime；不得让 producer 获得 backend/CAN handle。
5. 增加自动人工干预检测 shadow，验证符号、量级、freshness 和 deadline 后才允许其改变
   authority。
