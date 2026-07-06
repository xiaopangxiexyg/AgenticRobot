# VIVE 到 RM65 的相对控制与绝对控制说明

本文档只说明当前仓库里**已经落地、能对上代码实现**的 arm teleop 控制模式，也就是：

- `absolute`：绝对控制
- `delta_clutch`：相对控制 / 带 clutch 的增量控制

这里讨论的是 `VIVE -> PoseToArmRetargeter -> RM65` 这条链路，**不包含** `q6 -> hand/gripper` 手部链路。
当前真正可运行的调试入口是：

- [teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py)
- [teleop_runtime/debug/run_vive_rm65_real_teleop.py](./debug/run_vive_rm65_real_teleop.py)

前者使用真实 VIVE 输入，但默认只驱动 SAPIEN 中的双 RM65 虚拟模型；后者接入 RM65 真机。

## 1. 先说结论

当前两种模式的差别可以先粗略理解为：

| 模式 | 核心行为 | 是否需要 clutch | 更适合什么场景 |
| --- | --- | --- | --- |
| `absolute` | VIVE 位姿直接映射为机器人目标位姿 | 否 | 坐标系调试、标定检查、快速验证 |
| `delta_clutch` | 先记住一个锚点，再用“人手相对锚点的位移”去推动机器人 | 是，或默认常按住 | 真正遥操作、更安全的重抓取、避免大跳变 |

更具体地说：

- `absolute` 更像“你把 tracker 放到哪里，机器人 TCP 就想去哪里”。
- `delta_clutch` 更像“你按住 clutch 时才让手的运动增量生效，松开后机器人停在原地，你可以重新摆手再继续”。

## 2. 当前控制链路到底走哪些文件

### 2.1 当前主链路

当前相对控制和绝对控制，主链路是下面这条：

1. [hardware/VIVE/mocap/vive_data_sender.py](../hardware/VIVE/mocap/vive_data_sender.py)
   每帧取一次 OpenVR pose snapshot，给所有 tracker 复用后发 UDP JSON。
2. [teleop_runtime/sources/pose/vive.py](./sources/pose/vive.py)  
   接收 UDP，解析成 `ObservationChannel`。
3. [teleop_runtime/session.py](./session.py)  
   聚合 source，生成 observation，调用 retargeter。
4. [teleop_runtime/retargeters/pose_to_arm.py](./retargeters/pose_to_arm.py)  
   在这里分流到 `absolute` 或 `delta_clutch`。
5. [teleop_runtime/backends/arms/rm65.py](./backends/arms/rm65.py)  
   把笛卡尔目标位姿转成关节目标；真机模式下调用 `Movej_CANFD`。
6. [teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py)  
   当前现成入口。它每一帧从 `session.run_step()` 取命令，再喂给虚拟 RM65 backend 和 SAPIEN 渲染器。

### 2.2 VIVE 接收器在当前链路里的位置

仓库只保留一个 VIVE sender 入口：[hardware/VIVE/mocap/vive_data_sender.py](../hardware/VIVE/mocap/vive_data_sender.py)。
当前真正参与控制闭环的接收器是 [teleop_runtime/sources/pose/vive.py](./sources/pose/vive.py)；
旧的 float-only `vive_data_receiver.py` / `vive_visualizer.py` 已移除，避免和 JSON UDP 主链路混用。

## 3. 共享的数据流和坐标变换

不管是 `absolute` 还是 `delta_clutch`，前半段流程完全一样。

### 3.1 VIVE 发送端

[hardware/VIVE/mocap/vive_data_sender.py](../hardware/VIVE/mocap/vive_data_sender.py) 默认发送两个 wrist tracker：

- `tracker_1:left_wrist`
- `tracker_2:right_wrist`

第三只 `tracker_3:auxiliary` 可选；virtual desk 标定可以直接使用左/右 wrist tracker。

每个 tracker 打包内容包括：

- `tracker_name`
- `tracker_role`
- `valid`
- `pose_matrix_4x4`
- `sender_timestamp`

### 3.2 Runtime 接收端

[teleop_runtime/sources/pose/vive.py](./sources/pose/vive.py) 收到数据后，会把每个 tracker 变成 `ObservationChannel`，关键字段是：

- `channel_name`：例如 `left_wrist_pose`
- `frame_name="vive_world"`
- `value=4x4 pose matrix`
- `metadata.tracker_role`

### 3.3 `PoseToArmRetargeter` 先统一到 desk frame

[teleop_runtime/retargeters/pose_to_arm.py](./retargeters/pose_to_arm.py) 不会直接拿 `vive_world` 去控机器人，而是先走标定对象 [teleop_runtime/calibration.py](./calibration.py)：

- `world_to_operator`
- `operator_to_desk`
- `desk_to_base`

当前公式是：

```text
pose_operator = inv(T_W_O) @ pose_world @ T_tracker_offset
pose_desk     = inv(T_O_D) @ pose_operator
pose_base     = inv(T_D_B) @ pose_desk
```

也就是说，控制模式的分歧发生在 `pose_desk` 这一层。

### 3.4 当前虚拟跑法默认标定

[teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py) 默认调用：

- [teleop_runtime/configs/presets.py](./configs/presets.py) 里的 `build_annotated_dual_rm65_calibration(...)`

这个默认标定的关键点是：

- `T_W_O = I`
- `T_O_D = I`
- 左右臂底座距离默认 `0.525 m`
- 左臂底座在 desk 下约为 `y = +0.2625 m`，偏航 `-90°`
- 右臂底座在 desk 下约为 `y = -0.2625 m`，偏航 `+90°`

因此在默认虚拟跑法里，preset 本身仍然是：

- `T_W_O = I`
- `T_O_D = I`

但脚本启动后会先进入等待状态，等你按 `p` 或回车后，再做一次 **runtime home calibration**：

- 等待 `left_wrist_pose` 和 `right_wrist_pose`
- 对每侧收集若干帧稳定样本
- 记录“人手 home pose”和“机器人当前 home TCP pose”
- 之后无论 `absolute` 还是 `delta_clutch`，都先以这个启动 home 作为对齐锚点

这意味着：

- **同一次启动内**，只要你在开机时重新做 home calibration，VIVE 基站位置变化不会直接导致机器人瞬间跑到旧坐标
- 但它**还不是**完整的 `vive_world -> operator -> desk` 显式外参标定流程
- 所以它更准确地说是“每次启动的 home 对齐”，而不是“桌面世界系完全求解完毕”

## 4. 绝对控制 `absolute` 的当前实现

### 4.1 代码入口

绝对控制在 [teleop_runtime/retargeters/pose_to_arm.py](./retargeters/pose_to_arm.py) 里的逻辑非常直接：

1. 从 observation 取 `left_wrist_pose` / `right_wrist_pose`
2. 根据标定把输入 pose 变到 `desk`
3. 调 `_apply_absolute_mapping(pose_desk)`
4. 做姿态策略和 workspace 裁剪
5. 转到 `base`
6. 输出 `command_type="cartesian_pose"` 的 `EffectorCommand`

对应的核心行为现在是：

```text
delta_translation  = current_human.translation - home_human.translation
target_translation = home_robot.translation + translation_scale * delta_translation
target_rotation    = home_robot.rotation * inv(home_human.rotation) * current_human.rotation
```

当前默认配置下：

- `orientation_mode="full"`
- `translation_scale=1.0`
- `workspace_clip=True`

所以绝对控制默认就是：

- 启动时先把“人的 home 位姿”和“机器人的 home TCP 位姿”对齐
- 之后在 `desk` 全局坐标中跟踪人手相对 home 的平移变化
- 末端方向默认跟踪 tracker 相对 home 的姿态变化，因此 tracker 保持在 home 不动时机器人也会保持在 robot home
- 超出工作空间的平移会被裁剪回安全范围

只有在**没有建立 home anchor** 的情况下，`_apply_absolute_mapping(...)` 才会退回旧的“直接把当前 pose 当作目标 pose”行为。  
但当前 [run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py) 会在进入主循环前等待你手动触发 runtime home calibration，所以虚拟跑法默认不会走这个 fallback。

### 4.2 默认双臂碰撞规避层

当前 [run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py) 在双臂模式下默认会先走一层
`DualArmCollisionAwareRetargeter`：

1. 仍然先用左右两侧 `PoseToArmRetargeter` 生成 `cartesian_pose`
2. 再把左右 TCP 目标一起送进 joint-space 双臂多任务 IK
3. 最终输出两条 `command_type="joint_vector"` 的机械臂命令

默认参数是：

- `collision_scene_profile="dual_arm"`
- `collision_avoidance_mode="boundary_slide"`
- 运行依赖：`pinocchio + pink + qpsolvers`

这层行为有两个工程模式：

- `boundary_slide`：目标继续推向碰撞区时，解会沿可行边界继续移动
- `freeze_on_contact`：一旦进入接触阈值附近，就保持最近安全关节位姿

如果你显式加：

```bash
--no-collision
```

才会退回到旧的“两条独立 `cartesian_pose` -> 两个单臂 backend”路径。
旧参数 `--disable-dual-arm-collision-avoidance` 和 `--disable-collision-avoidance` 仍然兼容。

### 4.3 后端如何执行

目标命令进入 [teleop_runtime/backends/arms/rm65.py](./backends/arms/rm65.py) 后：

1. 如果上游输出的是 `cartesian_pose`，backend 自己做单臂 IK
2. 如果上游输出的是 `joint_vector`，backend 只做关节步长限制
3. 真机模式下再调用 `Movej_CANFD(...)`

也就是说：

- 旧路径：`cartesian_pose`，单臂 IK 在 backend 内完成
- 新默认双臂路径：`joint_vector`，联合 IK 已经在 retargeter 层完成

### 4.4 启动后的预期效果

`absolute` 模式下，预期看到的行为是：

- VIVE tracker 在 `desk` 中向前/左/上移动时，机器人 TCP 也按同一全局方向移动
- VIVE tracker 当前姿态变化，会直接映射成末端姿态变化
- 不需要 clutch，就会持续响应输入
- 如果 `T_W_O / T_O_D / T_D_B` 不准，机械臂目标会明显偏、歪，或者一开始就在不合理位置
- 如果输入 tracker 瞬时跳变，目标也会跟着跳，只是后端 IK 和关节步长限制会把执行动作稍微抹平

### 4.5 适用场景

当前 `absolute` 更适合：

- 检查坐标系有没有对上
- 检查 tracker 角色映射有没有写反
- 检查左右臂基座外参是否合理
- 调试当前统一的 `tool_tcp` action frame

它不一定适合作为第一选择的真机精细遥操作模式，因为对标定质量非常敏感。

## 5. 相对控制 `delta_clutch` 的当前实现

### 5.1 代码入口

相对控制由两部分组成：

- [teleop_runtime/retargeters/pose_to_arm.py](./retargeters/pose_to_arm.py)
- [teleop_runtime/clutch.py](./clutch.py)

`PoseToArmRetargeter` 先决定当前 clutch 是否按下，再把数据交给 `DeltaClutchController.update(...)`。

### 5.2 当前 clutch 的定义

默认会读取这些按钮 channel：

- `left_clutch_button`
- `right_clutch_button`
- 兼容通用名 `clutch_button`

在当前现成入口 [run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py) 里，这些按钮来自：

- [teleop_runtime/sources/misc/keyboard.py](./sources/misc/keyboard.py)

如果启用 `--keyboard-clutch`，终端键位是：

- `z`：切换左手 clutch
- `x`：切换右手 clutch
- `c`：左右一起切换
- `r`：全部释放
- `q`：退出

### 5.3 `delta_clutch` 的核心机制

`DeltaClutchController` 为每一侧维护一个 anchor：

- `human_pose_reference`
- `robot_pose_target`
- `timestamp`

控制逻辑是：

1. **第一次按下 clutch 的瞬间**

   记录一个 anchor：

   - 当前人的参考 pose
   - 当前机器人的目标 pose

2. **clutch 按住期间**

   只取“人相对 anchor 的位移增量”：

```text
delta_translation =
    (current_human.translation - anchor.human.translation) * scale

target.translation =
    anchor.robot.translation + delta_translation
```

姿态部分当前实现是：

```text
target.rotation = current_human.rotation
```

也就是说，当前 `delta_clutch` 不是“只增量平移、姿态冻结”的版本，而是：

- 平移：相对 anchor 做全局平移增量
- 姿态：按住 clutch 时直接跟随当前 tracker 的全局姿态

3. **clutch 松开期间**

   直接返回上一帧机器人目标位姿，不再响应当前手的移动。  
   这意味着你可以松手、重新摆姿势、再按下继续。

### 5.4 为什么第一次必须先按一下 clutch

当前实现里，如果：

- `control_mode="delta_clutch"`
- 还没有 `_last_target_pose_desk`
- 当前 clutch 又没按下

那么 retargeter 会直接返回空命令，并标记：

- `skipped = "waiting_for_clutch_anchor"`

这是一个刻意保留的安全行为，目的就是：

- 避免系统刚启动时，因为还没有 anchor，直接把当前 tracker 绝对位姿当成机器人目标，导致瞬时跳变

### 5.5 启动后的预期效果

`delta_clutch` 模式下，预期行为是：

- 没按 clutch 前，系统可能不出 arm 命令，直到你先建立 anchor
- 按住 clutch 时，手往前推一点，机器人也沿当前映射方向推进一点
- 松开 clutch 后，机器人停在当前目标位姿
- 你可以把手撤回来、重新摆舒服，再按下 clutch 继续推
- 如果键盘 clutch 被一直设为 `True`，那它会表现成“连续开启的相对控制”，本质上少了重新抓取这一步

### 5.6 当前实现的一个重要细节

当前 `delta_clutch` 的 orientation 处理是：

- **按住时**：姿态跟当前 tracker
- **松开时**：姿态停在上一帧目标

因此它更准确地说是“带 clutch 的平移增量控制 + 按住期间姿态直跟”。

## 6. 当前现成的启动方式

### 6.1 推荐环境

仓库现有 README 里推荐：

```bash
conda activate dexteleop310
```

如果只做虚拟跑法，至少要保证：

- `triad_openvr`
- `numpy`
- `SAPIEN` 相关依赖

可用。

### 6.2 先启动 VIVE 发送端

同机调试：

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host 127.0.0.1 \
  --port 9999 \
  --hz 60 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist
```

跨机器调试时，把 `--host` 改成运行 `teleop_runtime` 接收端那台机器的 IP。

### 6.3 启动绝对控制

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --control-mode absolute
```

建议先从这个模式开始，因为更容易发现：

- tracker 映射是否正确
- 左右 arm 是否对称
- 坐标轴朝向是否正确

正常工作时，终端会周期性打印类似：

```text
[Calibration] expecting left_wrist_pose<-tracker_1, right_wrist_pose<-tracker_2
[Calibration] HOME CAPTURE IS ARMED. Position robot and trackers first, then press 'p' or Enter.
[Calibration] waiting_for_key tracker_ready={'left_wrist_pose': True, 'right_wrist_pose': True} ready_key='p/Enter'
[Calibration] observed left_wrist_pose <- tracker_1:left_wrist frame=vive_world
[Calibration] observed right_wrist_pose <- tracker_2:right_wrist frame=vive_world
[Calibration] runtime home captured left_samples=20 right_samples=20
[VirtualRun] step=123 hz=29.9 state=ARMED channels=['left_clutch_button', 'left_wrist_pose', 'right_clutch_button', 'right_wrist_pose'] left_clutch=False right_clutch=False faults=[] collision=tracking
```

如果左右腕部通道没进来，常见现象是：

- calibration 阶段超时，或者进入运行态后直接 `FROZEN`
- `faults=['missing_channel:left_wrist_pose']` 或 `faults=['missing_channel:right_wrist_pose']`
- 机械臂不动
- 终端只打印到单侧 `observed ...`

### 6.4 启动相对控制

带键盘 clutch 的推荐方式：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --control-mode delta_clutch \
  --keyboard-clutch
```

启动后先按一次对应侧的 clutch 建 anchor，再开始移动。

如果你只是想快速看连续效果，也可以不加 `--keyboard-clutch`：

```bash
python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --control-mode delta_clutch
```

这时脚本会把左右 clutch 默认设成 `True`，表现会更接近“持续开启的相对控制”。

### 6.5 常用附加参数

- `--translation-scale`：调整平移缩放比
- `--collision-avoidance-mode boundary_slide|freeze_on_contact`：双臂推向碰撞区时的默认行为
- `--collision-scene-profile dual_arm|dual_arm_hand_proxy_open|dual_arm_hand|dual_arm_cylinder_table`：当前 joint-space 双臂碰撞场景 profile，默认先用 `dual_arm_hand_proxy_open`
- `--no-collision`：退回旧的两条独立单臂链路
- `--tcp-mode hand_base`：历史兼容参数；当前 action 主 frame 是 `tool_tcp`
- `--skip-renderer`：不启动 SAPIEN，只跑控制主链
- `--headless`：创建渲染器但不弹 viewer
- `--hz`：主循环频率
- `--print-hz`：终端打印频率
- `--max-steps`：跑固定步数后退出

## 7. 真机现在怎么接

当前真机入口是：

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --load-profile teleop_records/calibration_profiles/your_profile.json \
  --control-mode absolute
```

正式采集时优先使用同一个入口加 `--gui`、相机参数和需要的 hand/gripper backend；不要再新写一套绕开现有状态机的真机 runner。

## 8. 两种模式分别会看到什么现象

### 8.1 绝对控制的典型现象

- 机械臂会比较“黏”着 tracker
- 一旦你的手进入奇怪姿态，机器人也会努力跟过去
- 标定稍有问题，就容易体现为整体偏移、左右镜像不对、姿态扭转
- 更容易暴露“控制是通了，但坐标不对”的问题

### 8.2 相对控制的典型现象

- 机器人不会因为你松手重摆而乱跑
- 适合操作空间有限、操作者姿势不舒服的情况
- 比 `absolute` 更适合长时间操作
- 但如果 clutch 使用逻辑不清楚，会误以为“为什么机器人不动”，实际通常是还没建 anchor 或 clutch 没按下

## 9. 当前版本的限制和注意事项

### 9.1 这是 arm 模式，不是 hand 模式

当前 `absolute / delta_clutch` 只实现于：

- [teleop_runtime/retargeters/pose_to_arm.py](./retargeters/pose_to_arm.py)

不影响 `JointsToDexHandRetargeter` 那条手部链路。

### 9.2 现成入口的边界

当前可以直接跑的是：

- [teleop_runtime/debug/run_vive_sapien_virtual.py](./debug/run_vive_sapien_virtual.py)
- [teleop_runtime/debug/run_vive_rm65_real_teleop.py](./debug/run_vive_rm65_real_teleop.py)

前者用于虚拟验收，后者用于真机短测和 GUI 采集。

### 9.3 默认虚拟标定更像调试预设

默认 `T_W_O=I, T_O_D=I`，所以现在看到的“绝对控制”本质上更接近：

- VIVE 世界系下的 tracker pose，直接映到 desk，再映到各 arm base

它适合调试，不代表真实工位下已经完成 runtime calibration。

### 9.4 `delta_clutch` 当前对姿态不是纯增量

再次强调，当前实现不是“整个位姿都做相对增量”，而是：

- 平移：相对 anchor 增量
- 姿态：按住 clutch 时直接跟当前 tracker

如果后续你想要“平移和姿态都做增量”或者“姿态只保留 yaw”，需要继续改 [pose_to_arm.py](./retargeters/pose_to_arm.py) 里的策略。

## 10. 建议的实际调试顺序

建议按这个顺序排查：

1. 先跑 `hardware/VIVE/mocap/vive_data_sender.py`，确认 VIVE tracker 本身正常输出
2. 再跑 `run_vive_sapien_virtual.py --control-mode absolute`
3. 确认左右 tracker 映射、坐标方向、TCP frame 都合理
4. 再切到 `--control-mode delta_clutch --keyboard-clutch`
5. 确认 clutch anchor、释放、重抓取符合预期
6. 最后用 `run_vive_rm65_real_teleop.py` 做真机短测或 GUI 采集

如果你只是想判断“模式是不是工作了”，最直观的判断标准是：

- `absolute`：不按任何键，机械臂持续跟 tracker 走
- `delta_clutch`：只有按住 clutch 时机械臂才继续响应；松开后停住
