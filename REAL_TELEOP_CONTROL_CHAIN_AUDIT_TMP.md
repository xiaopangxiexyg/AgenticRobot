# 当前遥操链路审计（临时文档）

> 快照日期：2026-07-17  
> 范围：当前工作区中面向真实 RM65 的 VIVE 遥操 GUI 主链路。  
> 目的：整理位控/“力控”、单臂/双臂、坐标系和 IK 的实际调用关系，作为后续对齐单臂解 IK 的依据。  
> 本轮只新增本文件，没有修改 Python、配置、环境或锁文件。由于当前工作区本身已有较多未提交改动，下面的行号以本次审计时的工作区快照为准。

## 1. 先给结论

当前系统不是一条“VIVE 位姿 → 一个 IK → 机械臂”的单链路，而是三个相互独立的开关叠加：

1. `active_sides`：只启用左臂/右臂，还是同时启用双臂。
2. `--no-collision`：双臂时是否启用共享的双臂耦合解算器。
3. `--arm-control-mode`：`position` 位控，还是 `impedance` 阻抗/力控。

另外，`--control-mode` 的 `absolute` / `delta_clutch` 只决定操作者位姿如何映射为目标位姿，不决定机械臂最终使用哪一个 IK 或控制器。

当前真实 GUI 主链路可以概括为下表：

| 拓扑 | 执行模式 | 上游目标/IK | 实际执行后端 | 最终输出 | 是否共享双臂约束 |
|---|---|---|---|---|---|
| 单侧启用 | `position` | `PoseToArmRetargeter` 输出笛卡尔目标；`RM65ArmBackend` 内部做单臂 Pinocchio/Jacobian IK | `RM65ArmBackend` | 关节目标，经位置插值/加速度规划后发 CANFD | 否 |
| 双侧启用 + `--no-collision` | `position` | 左右各自走 `PoseToArmRetargeter`；每个 `RM65ArmBackend` 各自做单臂 IK | 两个 `RM65ArmBackend` | 两个独立关节目标 | 否；只是两个单臂链路并行 |
| 双侧启用 + 默认碰撞保护 | `position` | 两个 pose adapter → 一个 `DualArmCollisionAwareTargetSolver`，共享 12 维双臂状态并联合求解 | 两个 `RM65ArmBackend` | 双臂 solver 直接给出的两个 `joint_vector` | 是 |
| 单侧启用 | `impedance` | `PoseToArmRetargeter` 输出笛卡尔目标；阻抗控制器内部做力偏移后的 IK | `RM65ImpedanceArmBackend` + `hardware/rm65/impedance.py` | 经过 admittance/IK resolver 后的关节 CANFD 命令 | 否 |
| 双侧启用 + `--no-collision` | `impedance` | 左右各自输出笛卡尔目标；两个阻抗控制器独立做 admittance + IK | 两个 `RM65ImpedanceArmBackend` | 两个独立的力控关节命令 | 否 |
| 双侧启用 + 默认碰撞保护 | `impedance` | 双臂 solver 先对名义目标做共享 IK；每侧阻抗控制器再根据力/位移偏移独立完成最终 IK | 两个 `RM65ImpedanceArmBackend` | 双臂名义目标 + 每侧力控后的关节命令 | 只在名义目标阶段共享；最终力控修正是逐臂的 |

因此，当前“单臂解 IK 混乱”并不是一个函数内部的小问题，主要原因是代码中同时存在三种不同职责的 IK：

- 位控单臂：`robot_motion/rm65/single_arm_ik.py` 的单步 IK，由 `RM65ArmBackend` 调用。
- 双臂耦合 IK：`robot_motion/rm65/dual_arm_ik.py` / `dual_arm_target_solver.py`，由双臂 retargeter 调用。
- 阻抗力控 IK：`hardware/rm65/impedance.py` 中的 `IKSafetyResolver` 及 SDK/Pinocchio provider，由阻抗控制器调用。

`PoseToArmRetargeter` 本身不是 IK；它负责把 VIVE 位姿映射、标定、工作空间裁剪后包装成 `EffectorCommand`。

## 2. 当前真实 GUI 的总调用图

当前应以 `teleop_runtime/debug/run_vive_rm65_real_teleop.py` 为真实 GUI 入口。`teleop_runtime/collection` 主要是 headless/兼容路径，不应直接作为当前 GUI 控制链路的唯一依据。

```text
VIVE tracker / operator input
        │
        ▼
Input source polling + filter + safety
        │
        ▼
TeleopSession.run_step()
        │
        ▼
PoseToArmRetargeter（每个逻辑侧各一个）
        │
        ├─ 双臂且未 --no-collision
        │       ▼
        │   DualArmCollisionAwareRetargeter
        │       ▼
        │   DualArmCollisionAwareTargetSolver
        │       ├─ 共享双臂 qpos
        │       ├─ 双臂末端任务
        │       ├─ 关节限位/速度步长
        │       ├─ 碰撞 barrier / posture task
        │       └─ 根据 arm_control_mode 输出 cartesian_pose 或 joint_vector
        │
        └─ 单侧，或双臂 + --no-collision
                ▼
            每侧独立的 PoseToArmRetargeter
                ▼
            cartesian_pose

TeleopCommand
        │
        ▼
BimanualRig.send_commands()
        │
        ├─ position   → RM65ArmBackend
        │                cartesian_pose → 单臂 IK
        │                joint_vector   → 接受 solver qpos
        │                → 插值/规划 → Movej_CANFD
        │
        └─ impedance  → RM65ImpedanceArmBackend
                         → PoseAdmittance / wrench
                         → IKSafetyResolver / SDK / Pinocchio provider
                         → Movej_CANFD 或 follow 流
```

关键入口和分发位置：

- CLI 模式定义：`teleop_runtime/debug/run_vive_rm65_real_teleop.py:2477-2502`。
- runtime 根据 active sides、控制模式和碰撞开关组装组件：`run_vive_rm65_real_teleop.py:7357-7617`。
- 每个周期从 source 到 command、guard、rig dispatch：`teleop_runtime/session.py:645-664,686-927`。
- rig 只负责把已经生成的 `EffectorCommand` 发给各侧 backend，本身不做 IK：`robot_motion/runtime/rig.py:69-142`。

## 3. 三个容易混淆的模式开关

### 3.1 `--control-mode`：操作者位姿映射方式

当前值为：

- `absolute`：根据校准时的人手 home 姿态和机械臂 home 姿态做绝对映射。
- `delta_clutch`：以离合锚点为基准，把重新接管后的操作者相对运动映射到机器人。

这个开关发生在 `PoseToArmRetargeter.retarget()` 的输入映射阶段，主要影响平移 home delta、姿态映射和 clutch anchor。它不等于位控/力控，也不等于单臂/双臂。

相关实现：`teleop_runtime/retargeters/pose_to_arm.py:300-460`。

#### 3.1.1 当前默认值核对与 `delta_clutch` 封存建议

当前不能简单说“所有入口都默认 `absolute`”，准确情况如下：

| 位置 | 当前默认 | 判断 |
|---|---|---|
| 真实 RM65 GUI `run_vive_rm65_real_teleop.py` | `absolute` | 当前真机主入口已经是 absolute |
| SAPIEN 虚拟入口 `run_vive_sapien_virtual.py` | `absolute` | 当前推荐虚拟验收入口已经是 absolute |
| 数据采集 `collection/run_collect.py` | `absolute` | 新采集默认是 absolute，且 metadata 会记录 `control_mode` |
| MuJoCo VIVE 入口 `run_vive_mujoco_teleop.py` | `delta_clutch` | 仍然是旧默认，需要纳入统一 |
| `PoseToArmRetargeterConfig` 的 dataclass 默认 | `delta_clutch` | 直接构造而不传 mode 时仍会进入 delta 分支 |
| `DualArmCollisionAwareTargetSolverConfig` 默认 | `delta_clutch` | 低层/兼容调用仍可能隐式进入 delta |
| `teleop_runtime/configs/builders.py` | 未显式传 mode，继承上述默认 | 存在隐式默认风险 |
| `teleop_runtime/README.md`、`blueprint.md` | 部分文字仍写 default `delta_clutch` | 文档和当前 canonical GUI 不一致 |

真实 GUI 会把 CLI 的 `args.control_mode` 显式传给左右 `PoseToArmRetargeter` 和双臂 retargeter，因此真实 GUI 的实际默认确实是 `absolute`：`run_vive_rm65_real_teleop.py:2477-2480,7905-7929,7967-7970`。但低层配置默认仍是 `delta_clutch`，所以如果要正式“封存 delta”，不能只改真实 GUI 的 argparse 默认值。

需要特别区分两件事：

- `absolute` 不是把 VIVE 原始 pose 生硬地直接当作机器人 pose。已有 runtime home 后，它使用 `home_robot`、`home_human` 做平移和姿态对齐；tracker 在 human home 时，目标会回到 robot home，代码在 `pose_to_arm.py:449-460`。
- `delta_clutch` 的主要独立价值是“松开后重新摆手，再按 clutch 继续操作”，也就是在操作者工作空间有限时进行 re-grasp。它不是当前单臂/双臂 IK 的必要组成部分，也不改变后续 position/impedance IK owner。

因此建议把控制模式调整为以下产品语义：

```text
正常主路径：absolute
历史兼容路径：delta_clutch（显式、隐藏/弃用、仅调试或复现旧记录）
```

推荐分两阶段封存：

**阶段 A：软封存，保留可恢复性。**

1. 所有 CLI、dataclass 和 builder 的隐式默认统一为 `absolute`。
2. `delta_clutch` 仍可被显式指定，但从正常帮助文本和推荐命令中移除，标记为 legacy/deprecated。
3. `--keyboard-clutch` 只在显式 legacy `delta_clutch` 时启用；absolute 下应明确提示“该参数无效”或直接拒绝，避免操作者以为 clutch 会改变 absolute 行为。
4. 保留 `DeltaClutchController`、旧 episode 的 `control_mode` metadata 解析和针对 delta 的回归测试；不要把旧 delta 记录静默解释成 absolute。
5. 新的真实运行、SAPIEN/MuJoCo 调试和数据采集 metadata 一律写入 `control_mode="absolute"`。
6. 同步修正文档：`README.md`、`README_control_modes.md`、`blueprint.md` 中把 absolute 写成默认，把 delta 章节改成“legacy compatibility / archived experimental mode”。

**阶段 B：确认无依赖后硬封存。**

只有在确认以下内容后，再删除运行分支和 clutch 输入链：

- 没有新的采集脚本或真机脚本显式使用 `delta_clutch`；
- 旧数据若需要重放，已经完成离线转换，或 replay 明确保留独立兼容 reader；
- `tests/test_teleop_recovery.py` 等 delta 行为测试已转移到归档兼容测试，或确认不再需要；
- `run_vive_mujoco_teleop.py`、低层 config、builder 和文档不再保留 delta 的隐式入口。

本次硬封存已完成：`PoseToArmRetargeter.retarget()` 的 delta 分支、`DeltaClutchController` 的 active 导出、delta recovery 分支和 active CLI choice 均已移除；`control_mode` metadata 仍固定写入 `absolute`。旧 replay/raw-input 记录若声明 delta，将不再被 active runtime 静默转换，需使用历史版本或单独迁移。

建议后续实现前先加一组默认值和兼容性检查：所有正式入口无参数解析为 `absolute`；低层直接构造 config 也解析为 `absolute`；只有显式 legacy 参数才允许 `delta_clutch`；旧记录中若 metadata 为 delta，则必须显式进入兼容路径。这样可以先解决“默认路径不统一”的问题，再决定是否彻底删除 delta 实现。

### 3.2 `--arm-control-mode`：机械臂执行方式

当前值为：

- `position`：目标最终需要转换成关节轨迹/关节目标，由位置后端执行。
- `impedance`：目标作为 TCP 位姿参考进入阻抗控制器，控制器读取力/力矩并产生 admittance 位姿修正，然后再进行最终 IK 和关节发送。

CLI 解析阶段会把 `args.arm_control_mode` 对齐到 execution config：`run_vive_rm65_real_teleop.py:3977-4031`。因此这里的 `position`/`impedance` 是执行层模式，而非输入层的 absolute/delta 选择。

### 3.3 `active_sides` 和 `--no-collision`：拓扑方式

双臂共享 solver 的条件是：左右两侧都 active，且没有传 `--no-collision`。一旦只有一侧 active，或者双臂显式 `--no-collision`，就会退化成每侧一个独立的 `PoseToArmRetargeter`。

这意味着“单臂”有两个实际含义：

1. 只启用一侧物理机械臂。
2. 双臂都启用，但禁止双臂耦合；此时是两个单臂链路并行，不是一个共享双臂解算器。

`--swap-arm-sides` 交换逻辑侧到物理机械臂/IP 的映射；`--swap-operator-sides` 只交换 VIVE tracker 输入。两者不应混为一个“左右手交换”开关：`run_vive_rm65_real_teleop.py:2460-2474`。

## 4. 输入位姿、标定和坐标系

### 4.1 VIVE 到机械臂 base 的主路径

`PoseToArmRetargeter.build_command_from_target_pose_desk()` 的输入主语义是 desk/retarget frame 下的目标姿态：

```text
tracker pose
  → world/operator/desk 归一化
  → tracker local offset
  → absolute 或 delta_clutch 映射
  → workspace / orientation policy
  → target_pose_desk
  → calibration.desk_to_base()
  → target_pose_base
  → EffectorCommand(frame_name="arm_base")
```

代码位置：`teleop_runtime/retargeters/pose_to_arm.py:205-274,376-460`。

按照当前 `TeleopCalibration` 的实现，若输入来自 `world`，实际计算为：

```text
pose_operator = inv(T_W_O) @ pose_world
pose_desk     = inv(T_O_D) @ pose_operator
pose_base     = inv(T_D_B) @ pose_desk
```

其中：

- `T_W_O`、`T_O_D` 是全局 operator/desk 标定。
- `T_D_B` 是每个机械臂自己的 desk-to-arm-base 标定对象，但代码实际使用 `inv(T_D_B) @ pose_desk`。
- 左右臂分别从 `calibration.arms` 取自己的 `T_D_B`，所以左右臂不是共用一个 arm base 坐标系。
- `desk` 是 VIVE retarget/calibration frame，不应直接称为 solver world。

实现位置：`teleop_runtime/calibration.py:256-291`；标定字段定义在 `calibration.py:88-126,141-169`。

### 4.2 控制 frame 和 solver frame 不是同一个名字

当前命令通常携带：

- 数值 `pose`：在每侧 `arm_base` 中表达。
- `frame_name="arm_base"`：说明 pose 的参考坐标系。
- metadata `arm_controlled_frame="tool_tcp"`：说明这个 pose 控制的是哪个末端物理 frame。
- `target_pose_desk` / `target_pose_base`：保留 retarget 阶段的审计信息。

常用 frame 的职责如下：

| Frame | 职责 | 当前主要使用位置 |
|---|---|---|
| `vive_world` / `world` | VIVE/追踪器输入的世界参考 | 输入解析、标定 |
| `operator` | 操作者参考坐标系 | 输入解析、标定 |
| `desk` | VIVE retarget/calibration 中间坐标系 | retarget 输出和 dual solver 的目标元数据 |
| `arm_base` | 每个 RM65 自己的 base frame | `EffectorCommand` 的 pose 数值 frame |
| `arm_endpoint` / `arm_link6` | 机械臂末端法兰/第 6 轴附近的 IK frame | Pinocchio/双臂 solver 内部候选 frame |
| `hand_base` | 手部安装基座或默认 TCP 参考 | RM65 URDF 和 TCP 转换 |
| `tool_tcp` | 当前实际控制的工具 TCP | command metadata、阻抗控制、TCP profile |
| joint qpos | 6 关节配置 | position backend、dual solver、CANFD |

统一 frame 别名和 TCP 变换定义在 `robot_motion/frames.py:11-83`。其中 `tool_tcp` 不是固定等于 `arm_link6`；实际工具安装偏置由 active TCP profile 决定。

### 4.3 当前最值得关注的坐标系风险

当前 dual solver 如果拿到 `target_pose_desk`，会优先使用 desk 目标；否则才可能从 command 中的 arm-base pose 和每侧标定重新恢复 solver target。随后还要把 `tool_tcp` 转换到 solver 使用的 endpoint frame。

这带来一个需要后续测试固定的约定：

```text
同一个目标的审计字段：target_pose_desk
同一个目标的 command 数值：target_pose_base
dual solver 的内部目标：solver frame / endpoint frame
单臂 position backend 的内部目标：configured TCP → hand_base/arm_link6
单臂 impedance backend 的内部目标：configured TCP → controller IK frame
```

现在这些层次都存在，但不是由一个统一的 `TargetPose`/`IKRequest` 接口串起来的。特别是 `target_pose_desk` 和 `target_pose_base` 同时存在时，dual solver 的优先级规则必须保留在对齐测试中，否则以后修改标定或 metadata 很容易出现“显示的目标”和“实际解的目标”不一致。这里是审计风险，不据此断言当前一定有数值 bug。

## 5. 双臂链路：当前相对完整的实现

### 5.1 双臂 retargeter 不是两个单臂 IK 的简单拼接

当左右都 active 且没有 `--no-collision` 时，主程序创建两个 `PoseToArmRetargeter`，再把它们注入一个 `DualArmCollisionAwareRetargeter`：

- 两个 pose adapter 分别处理左右 tracker、左右标定和左右工作空间。
- adapter 只产生左右笛卡尔目标。
- `DualArmCollisionAwareTargetSolver` 再读取左右目标和当前左右 qpos。
- solver 内部保持一个完整的双臂配置，联合处理左右末端任务、关节限位、速度步长、姿态/posture task 和碰撞 barrier。

主程序组装位置：`run_vive_rm65_real_teleop.py:7899-8092`。包装关系：`teleop_runtime/retargeters/dual_arm_collision_aware.py:1-85`。

### 5.2 双臂 solver 的目标和输出

共享 solver 的核心目标处理在 `robot_motion/rm65/dual_arm_target_solver.py:922-984,1666-1842`：

1. 先分别调用左右 pose adapter。
2. 把 desk/base/tool TCP 目标转换成 solver endpoint frame。
3. 以当前双臂 qpos 为 seed，执行共享的双臂 IK step。
4. 对候选解、限位、步长和碰撞约束进行处理。

主程序根据 `arm_control_mode` 选择输出类型：`run_vive_rm65_real_teleop.py:8027-8031`。

#### 双臂位控

```text
左右 target_pose
  → shared dual IK
  → limited solver qpos
  → left/right EffectorCommand(command_type="joint_vector")
  → RM65ArmBackend
  → position interpolation / CANFD
```

双臂位控输出的 qpos 已经是 solver 处理过的结果，metadata 中会带 `action_pose_source="limited_solver_output_qpos"` 等信息。`RM65ArmBackend` 通过 `_command_is_prelimited_solver_qpos()` 识别这类命令，避免把同一份 solver 步长再次作为普通 cartesian command 做一遍相同的 max-step clipping：`teleop_runtime/backends/arms/rm65.py:198-204`。

因此，双臂位控的实际 IK owner 是 dual solver，而不是每侧 `RM65ArmBackend` 的单臂 IK。

#### 双臂阻抗/力控

```text
左右 target_pose
  → shared dual IK（得到名义双臂 qpos）
  → 每侧 cartesian_pose + solver_output_qpos hint
  → 两个 RM65ImpedanceArmBackend
  → 每侧读取 wrench
  → PoseAdmittance 得到 pose offset
  → 每侧 IKSafetyResolver
  → 最终每侧 qpos CANFD
```

双臂阻抗输出仍保留 `solver_output_qpos`，但它是“名义目标的首选关节 hint”，不是最终必然发送的 qpos。力/力矩产生的 admittance offset 可能使每侧控制器重新走 IK，因此双臂的碰撞耦合只保证在名义目标解算阶段；力控修正后的两侧最终 qpos 并没有再次经过同一个双臂耦合 solver。

双臂输出构造位置：`robot_motion/rm65/dual_arm_target_solver.py:1395-1631`。双臂 IK backend：`robot_motion/rm65/ik.py:218-269` 和 `robot_motion/rm65/dual_arm_ik.py:11-17,297-360`。

### 5.3 双臂链路的边界

双臂 solver 属于“目标层/名义目标层”；它不直接取代每侧执行 backend 的全部职责。每侧 backend 仍负责：

- hardware state / command validity；
- TCP profile 处理；
- 最终关节限制和执行保护；
- position 的插值与 CANFD 发送，或 impedance 的力控循环。

这解释了为什么双臂代码看起来已经有“解 IK”，但 force/impedance 模式仍然会在每侧控制器内部出现另一套 IK。

## 6. 单臂链路：当前两种模式并不共用同一套 IK

### 6.1 单臂位控：`cartesian_pose → RM65ArmBackend → 单臂 IK`

单臂位控，或双臂 `--no-collision` 时每侧的实际路径是：

```text
VIVE
  → 单侧 PoseToArmRetargeter
  → cartesian_pose（arm_base frame，controlled frame=tool_tcp）
  → RM65ArmBackend._solve_cartesian_command()
  → solve_rm65_single_arm_ik()
  → limited qpos
  → position background stream / direct Movej_CANFD
```

具体职责：

1. `PoseToArmRetargeter` 完成 tracker→desk、absolute/delta、desk→base 和工作空间裁剪：`teleop_runtime/retargeters/pose_to_arm.py:205-274,300-460`。
2. `RM65ArmBackend` 收到 `cartesian_pose` 后，以自身读取的当前 qpos 为 seed 解单臂 IK：`teleop_runtime/backends/arms/rm65.py:1882-1950,2380-2410`。
3. solver 由 `robot_motion.rm65.ik.build_rm65_single_arm_ik_solver()` 创建：`rm65.py:2229-2248`。
4. solver 通过 Pinocchio URDF frame 建立 FK/Jacobian，并执行一次受关节限位、阻尼、步长约束的加权 Jacobian 更新：`robot_motion/rm65/single_arm_ik.py:30-83,98-155`。
5. 之后做 wrist-flip 等候选分支归一化，选择和 seed 距离较近的解：`single_arm_ik.py:208-247`。
6. 若启用 position background streaming，IK 结果先进入 interpolator/acceleration planner，再由后台 CANFD 周期发送；否则直接发送：`rm65.py:2040-2169,3467-3486`。

这条链路中，`RM65ArmBackend` 的单臂 IK 是最终 qpos 生成者。

### 6.2 单臂阻抗/力控：`cartesian_pose → admittance → 另一套 IK`

单臂阻抗的实际路径不是上面的 `RM65ArmBackend._solve_cartesian_command()`：

```text
VIVE
  → 单侧 PoseToArmRetargeter
  → cartesian_pose
  → RM65ImpedanceArmBackend
  → RM65PoseImpedanceController.step()
  → 读取 wrench / PoseAdmittance
  → reference pose + admittance offset
  → IKSafetyResolver（SDK / Pinocchio 等 provider）
  → final qpos
  → Movej_CANFD 或 follow stream
```

实现分层如下：

- backend 启动时连接 RM65、加载/校验/同步 TCP profile，并构造 `RM65PoseImpedanceController`：`teleop_runtime/backends/arms/rm65_impedance.py:376-526`。
- `send_command()` 接受 cartesian pose；如果收到 joint vector，通常先通过 FK 转成 TCP pose，之后仍以 pose reference 驱动阻抗控制器：`rm65_impedance.py:721-843`。
- command 的 controlled frame 会被统一转成实际 `tool_tcp`：`rm65_impedance.py:1000-1030`。
- 控制器 `step()` 依次处理参考位姿、wrench、admittance 位移和安全 IK，再形成关节命令：`hardware/rm65/impedance.py:3131-3382`。
- 经过力控偏移后的 pose 由 `IKSafetyResolver` 处理，内部可能选择 SDK 或 Pinocchio provider，并以 preferred qpos 作为分支/连续性提示：`hardware/rm65/impedance.py:1933-2085,4738-5051`。

所以“单臂力控解 IK”与“单臂位控解 IK”当前不是同一个类、同一个 seed 策略或同一个输出 contract。单臂 impedance 模式中，`robot_motion/rm65/single_arm_ik.py` 并不是主执行链路的最终 IK owner；它最多可能作为某个 provider 或预览/兼容路径的一部分。

### 6.3 当前代码中“单臂 IK”这个词的四种指代

后续讨论时建议明确写出完整名称，否则很容易把不同层混在一起：

| 名称 | 是否真做 IK | 所属阶段 | 主要用途 |
|---|---:|---|---|
| `PoseToArmRetargeter` | 否 | 输入 retarget | 把操作者 pose 变成 arm-base 下的笛卡尔目标 |
| `RM65ArmBackend` + `solve_rm65_single_arm_ik` | 是 | 位控执行前 | 单臂 cartesian target → qpos |
| `DualArmCollisionAwareTargetSolver` | 是 | 双臂目标层 | 两臂 cartesian target → coupled qpos |
| `RM65PoseImpedanceController` + `IKSafetyResolver` | 是 | 力控执行循环 | admittance 修正后的 pose → final qpos |

此外，`robot_motion/rm65/position_collision.py` 的 `RM65PositionCollisionPlanner` 是另一条较高层的双臂 position planner，主要被 sim/replay/deploy/collection 等路径使用；它不是当前真实 GUI 中 `DualArmCollisionAwareTargetSolver` 的同一个对象，不能拿来直接证明 GUI 的单臂或双臂链路已经对齐。

## 7. 位控和阻抗力控的差异

### 7.1 位控的 qpos 所有权比较清晰

位控有两种 qpos 生成位置：

- 单臂/独立侧：`RM65ArmBackend` 的单臂 IK 生成 qpos。
- 双臂耦合：`DualArmCollisionAwareTargetSolver` 生成 qpos，`RM65ArmBackend` 直接接受 `joint_vector`。

双臂位控的 metadata 会标识 solver 已经做过限步，backend 不应把它重新当成普通 Cartesian command 再做一次独立单臂 IK。这是当前 dual position 能够保持双臂耦合的关键。

### 7.2 阻抗力控的 qpos 所有权是两级的

阻抗模式的 qpos 生成分为两级：

1. dual 场景下，双臂 solver 生成名义目标 qpos；
2. 每个 impedance controller 根据当前 wrench 和 admittance offset 生成最终 qpos。

单臂场景没有第 1 级 dual solver，直接从 pose reference 进入第 2 级。

因此，双臂 impedance 目前不是“完整双臂 force IK”，而是“名义双臂耦合 IK + 每侧独立力控实现”。如果以后要求力控后的两臂仍然满足严格的实时互碰约束，需要单独设计 force-adjusted dual solve 或其他跨臂安全层，不能仅靠现有单臂 resolver 推断已经满足。

### 7.3 当前用户所说的“力控”不是旧版 Force-Position SDK 分支

`RM65ArmBackendConfig` 仍保留 `force_position_control`、传感器、方向、力值等旧字段；当该 flag 为真时，backend 会走 SDK 的 `Force_Position_Move_Joint`。但当前 canonical real GUI 的构造路径没有给这个字段赋值，当前 `--arm-control-mode impedance` 使用的是：

```text
RM65ImpedanceArmBackend
  → RM65PoseImpedanceController
  → PoseAdmittance + IKSafetyResolver
```

因此文档和调试日志中最好把它称为“impedance/admittance 力控”或“阻抗力控”，把旧的 `Force_Position_Move_Joint` 称为“legacy force-position SDK 分支”，避免用同一个“force control”名称指代两套行为。

## 8. 后端对象的重复不是重复控制链

当前组件组装时，每个侧可能同时存在三个不同用途的 arm backend：

| 对象 | 典型类型 | 是否驱动真实机械臂 | 用途 |
|---|---|---:|---|
| preview backend | `RM65ArmBackend`，dry-run | 否 | 本地 FK/IK 预览、候选姿态和界面显示 |
| state backend | `RM65ArmBackend` | 否（读状态） | 连接硬件读取 qpos/状态、刷新当前状态 |
| command backend | `RM65ArmBackend` 或 `RM65ImpedanceArmBackend` | 是 | 接收 rig dispatch 并执行实际 position/impedance |

主程序的相关组装位置：`run_vive_rm65_real_teleop.py:8292-8454`；state backend 单独启动在 `run_vive_rm65_real_teleop.py:12087-12096`。

这会产生一个非常容易误读的现象：即使当前真实执行模式是 impedance，代码里仍然可以看到一个 `RM65ArmBackend` 在做 dry-run 单臂 IK。那个对象属于 preview 或 state 路径，不等于真实电机 command backend 的 IK 路径。

建议后续排查日志时至少区分以下前缀/语义：

```text
preview_*       本地预览解
state_*         读到的硬件状态
command_*       实际送给硬件的 command
solver_output_* dual/position 名义 solver 输出
impedance_*     admittance 后的最终执行输出
```

## 9. 当前“单臂没有对齐”的具体表现和原因

下面是从调用链能直接确认的结构性差异；它们是后续对齐的重点，不是本轮要修改的代码：

### 9.1 单臂位控和双臂位控的 IK owner 不同

- 单臂位控：pose 先进入每侧 `RM65ArmBackend`，由单臂 solver 根据每侧当前 qpos 解。
- 双臂位控：pose 进入共享 dual solver，直接输出两个 qpos，后端不再做单臂 IK。

两者即使输入的 `target_pose_base` 相同，也不保证得到相同的 qpos 分支、阻尼行为、posture 偏好或步长轨迹。双臂 solver 的 seed 是共享双臂配置；单臂 solver 的 seed 是单侧 backend 当前状态。

### 9.2 单臂 impedance 和双臂 impedance 也不是同一层级

- 单臂 impedance：每侧直接从 pose reference 做 admittance + IK。
- 双臂 impedance：先做一次双臂名义 IK，再把每侧结果作为 pose/qpos hint 交给各自 impedance controller。

因此双臂中 `solver_output_qpos` 是 hint，单臂中通常没有对应的 dual hint；二者的分支连续性和最终 qpos 选择不完全对齐。

### 9.3 四种 frame contract 没有集中到一个接口

当前至少有以下转换入口：

- `PoseToArmRetargeter`：desk → arm_base；
- dual solver：desk/base/tool_tcp → shared solver endpoint frame；
- `RM65ArmBackend`：tool_tcp/endpoint → configured TCP 对应的 hand_base/arm_link6；
- impedance backend/controller：command frame → tool_tcp → controller IK frame。

如果某侧 TCP profile、`hand_base_to_configured_tcp`、`tool_tcp` metadata 或标定对象不一致，就可能出现“界面目标一致但单/双臂解不同”的现象。当前代码有各自的转换逻辑，但没有一个跨 `position`/`impedance`/single/dual 共用的最终 IK request contract。

### 9.4 命令类型不同，容易误以为后端行为不同是 bug

当前约定是：

- dual + position：solver 输出 `joint_vector`；
- single + position：retargeter 输出 `cartesian_pose`，backend 才解单臂 IK；
- impedance：通常都以 `cartesian_pose` 进入阻抗 backend，dual 额外携带 `solver_output_qpos` hint。

所以要比较单臂和双臂，不能只比较 `EffectorCommand.value` 的数值；必须同时比较 `command_type`、`frame_name`、`arm_controlled_frame`、`target_pose_desk`、`target_pose_base`、`solver_output_qpos` 和 `action_pose_source`。

### 9.5 速率和执行时钟不是一个概念

当前默认 profile 大致是：

- VIVE/数据采样：15 Hz；
- position 外层/reference/dual solve 调用预算：约 75 Hz；
- position CANFD background stream：约 150 Hz；
- impedance outer/controller 与 RM65 command rate：由 impedance profile 分别配置，默认不等同于 position 的 150 Hz stream。

配置解析还明确说明 `dual_arm_ik_hz` 是 nominal solver budget，实际 solver 调用受 outer loop 时钟约束：`teleop_runtime/position_control/config.py:596-605`。因此不能用 CANFD 发送频率直接判断 IK 是否以同样频率重新计算。

## 10. 推荐的“对齐基准”

这部分只是后续实现顺序建议，本轮不修改代码。

### 10.1 先固定四维行为矩阵

建议把以下组合明确写成测试/日志矩阵：

```text
                 position                         impedance
single       single-arm IK → qpos           pose → admittance → IK
dual         coupled dual IK → qpos         coupled nominal IK
             → position backend             → per-arm admittance → per-arm IK
```

再分别覆盖：

- 单侧 active；
- 双侧 + `--no-collision`；
- 双侧 + collision-aware dual solver；
- `absolute` / `delta_clutch`；
- `swap-arm-sides` / `swap-operator-sides`。

### 10.2 给所有 IK 统一一个输入/输出 contract

后续单臂对齐时，最应该先统一的不是某一个数值参数，而是以下字段：

```text
IKRequest
  target_pose
  target_frame              # arm_base / solver frame
  controlled_frame           # tool_tcp / endpoint
  current_qpos               # seed
  preferred_qpos             # 可选，来自 dual solver 或 home
  tcp_profile
  joint_limits
  max_joint_step
  branch_policy
  collision_policy

IKResult
  qpos
  realized_pose
  solver_frame
  controlled_frame
  branch_id / branch_reason
  safety / limit metadata
```

目前这些字段分散在 command metadata、`RM65SingleArmIKConfig`、dual solver config 和 impedance `IKResolveRequest` 中。没有统一 contract 时，单臂 position 与单臂 impedance 很难证明已经使用了同一坐标系、同一 TCP 和同一 branch continuity 规则。

### 10.3 明确 dual impedance 的安全语义

需要先决定以下产品/控制语义：

- `solver_output_qpos` 只是 seed/hint，还是必须作为最终 qpos 的硬约束？
- admittance offset 后是否允许每侧独立解到会破坏双臂碰撞约束的状态？
- 如果不允许，是否需要 force-adjusted dual solver 或跨臂 final safety check？
- 单臂 impedance 是否需要复用 dual solver 的 branch/seed 规则，还是允许使用独立 resolver？

这些问题不是通过把某个单臂 IK 函数名改成 dual IK 就能解决的。

### 10.4 先做离线/dry-run 对齐验证

在真正接触硬件前，建议用同一批固定 pose 和 qpos seed 做以下比较：

1. 单臂 position solver 与 dual solver 固定另一侧后的 qpos 差异。
2. 单臂 position 与 impedance resolver 在零 wrench、零 admittance offset 时的 qpos 差异。
3. `tool_tcp`、`hand_base`、`arm_link6` 三种 target frame 相互转换后的 FK 误差。
4. 左右 side calibration/TCP profile 交换前后的结果。
5. 首次解、连续解、wrist-flip 附近的 branch 选择和 qpos step。
6. dual position 的 prelimited qpos 是否被 backend 二次限步。
7. dual impedance 的 qpos hint 在 admittance offset 为零/非零时是否按预期使用。

本轮没有执行真实运动、回放、force enable 或硬件控制；这些验证应保持 dry-run/仿真或明确的安全流程。

## 11. 位置索引

以下文件构成当前审计的主要证据链：

- 主 GUI 入口和组件组装：`teleop_runtime/debug/run_vive_rm65_real_teleop.py`。
- session 生命周期和 command 构造：`teleop_runtime/session.py`。
- command 分发：`robot_motion/runtime/rig.py`。
- VIVE pose retarget：`teleop_runtime/retargeters/pose_to_arm.py`。
- 双臂 retarget adapter：`teleop_runtime/retargeters/dual_arm_collision_aware.py`。
- 双臂目标 solver：`robot_motion/rm65/dual_arm_target_solver.py`。
- 双臂 IK/Pink 封装：`robot_motion/rm65/dual_arm_ik.py`、`robot_motion/rm65/ik.py`。
- 单臂 position backend：`teleop_runtime/backends/arms/rm65.py`。
- 单臂 IK：`robot_motion/rm65/single_arm_ik.py`、`robot_motion/rm65/ik.py`。
- impedance backend：`teleop_runtime/backends/arms/rm65_impedance.py`。
- impedance/admittance 和 resolver：`hardware/rm65/impedance.py`。
- 坐标系/TCP 别名：`robot_motion/frames.py`。
- 标定变换：`teleop_runtime/calibration.py`。
- position/impedance profile 和频率：`teleop_runtime/position_control/config.py`、`teleop_runtime/real_teleop/control_profile.py`、`teleop_runtime/defaults.py`。

## 12. 最终判断

当前双臂位控链路已经形成了清晰的“共享 dual IK → qpos → position backend”结构；双臂 impedance 也已经形成“共享名义 IK → 每侧阻抗执行”的结构，但最终力控修正仍是逐臂完成。

当前单臂之所以显得没有对齐，核心不是“没有单臂 IK”，而是：

1. 单臂 position 使用 `RM65ArmBackend` 的 Pinocchio/Jacobian 单臂 IK；
2. 单臂 impedance 使用硬件阻抗控制器内部的 `IKSafetyResolver`；
3. 双臂 position 使用共享 dual solver，跳过每侧 backend 的单臂 IK；
4. 双臂 impedance 又把 dual solver qpos 作为每侧 resolver 的 hint；
5. 不同层各自处理了 desk/base/TCP/solver frame 和不同的 branch/limit/seed 逻辑。

因此后续工作的第一目标应是统一“目标 pose 的 frame、controlled TCP、seed/preferred qpos、限步/分支策略和结果 metadata”的 contract，再决定哪些 IK 实现可以复用；不建议在没有固定这套 contract 前直接修改某一个单臂 IK 函数。
