# 设备与标定参数说明

运行命令见 [hardware README](./README.md)。任何 H1/H2 工具运行前，先停止使用同一设备的
teleop、collection、replay、deploy 和独立可视化进程。

## 1. 统一设备检查

H1 只读检查命令见 [hardware README 的第 1 节](./README.md)。

H2，处理串口冲突并修改当前设备节点权限：

```bash
sudo -E "$CONDA_PREFIX/bin/python" \
  -m teleop_runtime.configure_devices \
  --fix-serial-conflicts \
  --fix-permissions
```

## 2. VIVE

H1，sender 持续读取 OpenVR 并发送 UDP；当前启动命令见
[real README 的第 1.1 节](../real/README.md)。

协议、校时、benchmark 和 legacy receiver 见：

- [`hardware/VIVE/mocap/README.md`](../../hardware/VIVE/mocap/README.md)
- [`hardware/VIVE/mocap/CLOCK_SYNC.md`](../../hardware/VIVE/mocap/CLOCK_SYNC.md)

## 3. Teleop calibration

H1，读取 tracker、交互生成 virtual desk/wrist profile，不发送 RM65 运动。当前命令见
[hardware README 的第 2 节](./README.md)。

已有 profile 可显式增加：

```bash
--load-profile "${CALIBRATION_PROFILE:?set CALIBRATION_PROFILE}"
```

上面是追加到 calibration 命令后的参数片段，不是独立命令。

动作要求、按键和质量门槛见
[`teleop_runtime/CalibrationREADME.md`](../../teleop_runtime/CalibrationREADME.md)。Session home
仍需每次运行重新采集，不能由旧 profile 代替。

## 4. Q6 与 AngleTrigger

H1 扫描命令见 [hardware README 的第 3 节](./README.md)。

H2，保存左右 assignment：

```bash
python -m hardware.teleop_hand.q6_assignment configure \
  --left "${LEFT_Q6_BY_ID:?set LEFT_Q6_BY_ID}" \
  --right "${RIGHT_Q6_BY_ID:?set RIGHT_Q6_BY_ID}"
```

AngleTrigger：

```bash
# H1：枚举
python -m hardware.angle_trigger --list

# H2：重分配或重做角度端点
python -m hardware.angle_trigger --reassign
python -m hardware.angle_trigger --recalibrate-angles
```

Q6 是输入设备；DMTAC gripper 是输出设备，不能复用彼此的串口 namespace。

## 5. DMTAC gripper / tactile

独立 CLI 的源码包位于子目录。从仓库根设置源码路径和进行只读扫描的当前命令见
[hardware README 的第 4 节](./README.md)。

这里只保留副作用分级：`list-*`、`--scan-only` 和 `verify-*` 是 H1；
`configure-usb ... --save` 属于 H2；`open/close/move/set-speed/set-torque` 属于 M。
只读组合、左右分配、tactile baseline、micro-jog 和配置覆盖规则统一由
[DMTAC 配置说明 §4](./DMTAC.md#4-dmtac-维护-cli)维护。

## 6. RM65 初始化、TCP、payload 与力传感器

R0、只生成 qpos 初始化 plan 且不连接真机的命令见
[hardware README 的第 5 节](./README.md)。

真机执行 qpos sequence、TCP/payload 标定、力传感器标定、teach/replay 和 impedance tune
全部属于 M 或 H2，不在统一索引中复制长命令。执行前阅读：

- [`robot_motion/rm65/arm_init/README.md`](../../robot_motion/rm65/arm_init/README.md)
- [`hardware/rm65/README.md`](../../hardware/rm65/README.md)
- [`hardware/rm65/utils/README.md`](../../hardware/rm65/utils/README.md)

旧 `hardware/rm65/utils/init_arm` Cartesian standard-pose 路径不是当前初始化主线。

## 7. UMI hand-eye

生成/检查标定命令只在 [UMI 数采的第 1 节](../real/UMI.md)维护。它处理 wrist
camera/tracker/tool 几何，不修改 DMTAC USB assignment，也不控制夹爪。完整工具参数见
[`utils/umi_hand_eye/README.md`](../../utils/umi_hand_eye/README.md)。
