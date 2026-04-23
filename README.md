# DexTeleop 采集链路说明（run_collect）

本文档总结 `python -m teleop_runtime.collection.run_collect ...` 进行采集时，
**VIVE tracker / q6 输入如何进入主控制循环并驱动 RM65 + InspireTac**，以及
**RM65 / InspireTac 状态如何回传给主进程**。

---

## 1. 总体架构

`run_collect` 采用**单进程 + 多线程**架构：

- 主线程：采集主循环（控制周期 `--hz`）
- Source 线程：VIVE UDP 接收线程、q6 串口读取线程、相机线程
- Backend 线程：InspireTac telemetry 线程、RM65 可选后台流式发送线程

> 关键点：这里没有独立子进程 IPC 队列；数据通过对象内部共享状态（加锁）被主线程拉取。

---

## 2. 输入链路：VIVE / q6 -> 主进程

### 2.1 组件装配入口（run_collect）

- `build_collect_app()`
  - 文件：`teleop_runtime/collection/run_collect.py`
  - 函数：`build_collect_app`（约 L960）

在这里会构建并注册：

- VIVE source：`_build_vive_source`（约 L379）
- q6 source：`_build_hand_source`（约 L418）
- session：`TeleopSession(...)`

### 2.2 VIVE 数据接入

- 文件：`teleop_runtime/sources/pose/vive.py`
- 类：`VivePoseSource`
- 关键接口：
  - `start()`：绑定 UDP host/port，启动接收线程
  - `_recv_loop()`：持续 `recvfrom()` 收包并解析
  - `get_latest_channels()`：主线程读取最新 `ObservationChannel`

输出到 session 的通道名：

- `left_wrist_pose`
- `right_wrist_pose`

### 2.3 q6 数据接入

- 文件：`teleop_runtime/sources/hand/q6_serial.py`
- 类：`HandQ6Source`
- 关键接口：
  - `start()`：为左右手创建 reader 与读取线程
  - `_reader_loop()`：循环调用 `reader.read_frame()`
  - `get_latest_channels()`：主线程读取最新 q6 通道

q6 底层读取接口：

- 文件：`hardware/teleop_hand/hand_q6_reader.py`
- 类：`SingleHandQ6Reader`
- 函数：`read_frame()`

输出到 session 的通道名：

- `left_hand_q6`
- `right_hand_q6`

---

## 3. 主控制链路：主进程如何下发到 RM65 / InspireTac

### 3.1 主循环调用关系

- 文件：`teleop_runtime/collection/run_collect.py`
- 函数：`main()`（约 L1124）
- 在 home ready 后，每轮调用：`runtime.step(now=loop_started_at)`

`runtime.step` -> `session.run_step`：

- 文件：`teleop_runtime/collection/runtime.py`
  - `CollectionRuntime.step()`（约 L49）
- 文件：`teleop_runtime/session.py`
  - `TeleopSession.run_step()`（约 L227）
  - `poll_sources()`（约 L182）汇总 VIVE/q6/camera 等通道
  - `_build_command()`（约 L217）调用 retargeter 生成动作命令

### 3.2 重定向（retarget）

1) VIVE wrist pose -> RM65 末端位姿命令

- 文件：`teleop_runtime/retargeters/pose_to_arm.py`
- 类：`PoseToArmRetargeter`
- 关键函数：
  - `retarget()`（约 L217）
  - `build_command_from_target_pose_desk()`（约 L173）
- 生成 `EffectorCommand(command_type="cartesian_pose")`，目标 effector：`left_arm/right_arm`

2) q6 手部关节 -> InspireTac 命令

- 文件：`teleop_runtime/retargeters/joints_to_dexhand.py`
- 类：`JointsToDexHandRetargeter`
- 关键函数：
  - `retarget()`（约 L52）
- 生成 `EffectorCommand(command_type="joint_vector")`，目标 effector：`left_hand/right_hand`

### 3.3 下发执行

- 文件：`teleop_runtime/session.py`
- 函数：`TeleopSession.run_step()`
- 在 `ARMED/RUNNING` 状态调用：`self.rig.send_commands(command)`

rig 分发：

- 文件：`teleop_runtime/rigs/bimanual.py`
- 类：`BimanualRig`
- 函数：`send_commands()`（约 L42）
- 将每个 effector 命令转发到对应 backend 的 `send_command()`

具体硬件 backend：

- RM65
  - 文件：`teleop_runtime/backends/arms/rm65.py`
  - 类：`RM65ArmBackend`
  - 函数：`send_command()`（约 L190）
  - 实际下发：`Movej_CANFD(...)`

- InspireTac
  - 文件：`teleop_runtime/backends/hands/inspire_tac.py`
  - 类：`InspireTacHandBackend`
  - 函数：`send_command()`（约 L84）
  - 实际下发：`hand.set_joint_pos(...)`

---

## 4. 状态回传链路：RM65 / InspireTac -> 主进程

## 4.1 RM65 qpos 回传

- 文件：`teleop_runtime/backends/arms/rm65.py`
- 关键接口：
  - `refresh_state_from_hardware()`（约 L502）
    - 调 RM SDK：`Get_Current_Arm_State()`
    - 解析并更新 `_current_qpos`
  - `get_state()`（约 L544）
    - 返回 `current_qpos`、`hardware_pose_6d`、错误码等

`run_collect` 中调用点：

- 文件：`teleop_runtime/collection/run_collect.py`
- 函数：`_sync_arm_backends_from_hardware()`（约 L657）
  - 启动与标定阶段调用 `backend.refresh_state_from_hardware()`
  - 读取 `state["current_qpos"]` 用于同步/home 锚定

## 4.2 InspireTac qpos / tactile 回传

- 文件：`teleop_runtime/backends/hands/inspire_tac.py`
- 关键接口：
  - `_telemetry_loop()`（约 L137）
    - 周期读取 `read_joint_pos()` 与 `read_tac_data()`
  - `get_state()`（约 L169）
    - 返回 `joint_pos`、`tactile`、时间戳、错误信息
  - `read_joint_pos_now()`（约 L124）
    - 需要时可同步读当前关节

`run_collect` 中显式使用点：

- 文件：`teleop_runtime/collection/run_collect.py`
- 函数：`_update_tactile_visualizer()`（约 L626）
  - 通过 `hand_backend.get_state().get("tactile")` 获取触觉并显示

> 说明：当前采集闭环里，手部控制输入主通道是 `q6`；InspireTac 的 `joint_pos` 已有回传接口，但在 `run_collect` 主逻辑中主要用于状态可读/监控，而非替代 q6 作为控制输入。

---

## 5. 一页调用路径（简版）

1. `run_collect.main()` 循环调用 `runtime.step()`
2. `CollectionRuntime.step()` 调 `TeleopSession.run_step()`
3. `poll_sources()` 拉取：
   - `VivePoseSource.get_latest_channels()`
   - `HandQ6Source.get_latest_channels()`
4. `_build_command()` 调 retargeter：
   - `PoseToArmRetargeter.retarget()` -> `left/right_arm cartesian_pose`
   - `JointsToDexHandRetargeter.retarget()` -> `left/right_hand joint_vector`
5. `rig.send_commands()` -> backend：
   - `RM65ArmBackend.send_command()`
   - `InspireTacHandBackend.send_command()`
6. 状态回读：
   - RM65：`refresh_state_from_hardware()` / `get_state()`
   - Inspire：`_telemetry_loop()` 写缓存，`get_state()` 读缓存

---

## 6. 与命令行参数的对应关系（你给的命令）

- `--host/--port` -> `VivePoseSource` UDP 接收
- `--left-hand-port/--right-hand-port` -> `HandQ6Source` 串口 q6 输入
- `--left-arm-ip/--right-arm-ip` -> `RM65ArmBackend`
- `--left-hand-ip/--right-hand-ip` -> `InspireTacHandBackend`（有 IP 才连接硬件与 telemetry）
- `--hand-telemetry-hz` / `--hand-tactile-regions` -> Inspire telemetry 采样频率与触觉区域
- `--hz` -> 主控制频率与 RM65 后台流式发送频率基准

