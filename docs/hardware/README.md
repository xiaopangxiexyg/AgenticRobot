# 硬件与标定运行入口

详细副作用和设备专题链接见 [设备与标定说明](./DEVICES_AND_CALIBRATION.md)。
带日期的设备角色或导出目标快照集中在
[hardware evidence receipts](./evidence/README.md)；这些收据只证明其写明的范围，不是设备连接、
配置修改或运动授权。

从仓库根目录执行。H1 命令虽然不发送目标，也会打开设备并可能与遥操/数采争用端口；运行前
停止使用同一设备的其他进程。

## 1. 统一设备只读检查（H1）

```bash
python -m teleop_runtime.configure_devices --check
```

## 2. Teleop calibration（H1，不发送 RM65 运动）

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999
```

## 3. Q6 扫描（H1）

```bash
python -m hardware.teleop_hand.q6_assignment scan --probe-all-serial
```

## 4. DMTAC CLI 检查（H1）

从仓库根执行：

```bash
export PYTHONPATH="$PWD:$PWD/hardware/DMTAC_gripper:${PYTHONPATH:-}"
python hardware/DMTAC_gripper/run_gripper.py --help
python hardware/DMTAC_gripper/run_gripper.py configure-usb --scan-only
```

## 5. 夹爪状态微动（M）

默认只生成左右夹爪（`all`）的预览，不连接设备；也可用 `--side left` 或
`--side right`。默认使用归一化正常力度 `0.20`、速度 `0.20`，在实测当前位置附近
做总跨度 `0.04` 的低位/高位/原位顺序微动，并在正常完成后恢复原始速度、力度和
continuous-grasp 设置。

```bash
python scripts/check_dmtac_gripper_health.py
```

实际微动属于 M：必须先停止 teleop/数采及其他夹爪进程，清空夹爪工作区，确认现场
操作员和急停可用，再显式解锁。两侧按 left -> right 顺序执行，不会同时运动；若运动
阶段异常，脚本请求该侧夹爪 stop，并保留错误供排查。

```bash
python scripts/check_dmtac_gripper_health.py \
  --side all \
  --execute \
  --confirmation GRIPPER_HEALTH_CHECK
```

## 6. RM65 信息与状态读取（H1，只读）

默认从 `latest` RM65 IP assignment 顺序读取两侧；支持 `--side left/right` 和
`--left-ip` / `--right-ip` 覆盖。输出包括机械臂基本信息、软件版本、当前关节/末端
状态、关节使能、温度、电流、电压、关节错误及抱闸状态。命令只调用读取接口和关闭
连接，不发送运动、使能、停止或配置写入，但仍会占用机械臂连接。

```bash
python scripts/read_rm65_status.py
python scripts/read_rm65_status.py --side left
```

## 7. 生成 RM65 qpos 初始化 plan（R0）

该命令不连接真机：

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_init \
  --current-qpos "${CURRENT_QPOS_JSON:?set CURRENT_QPOS_JSON}" \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/bench_init_pose.json \
  --arm both \
  --output /tmp/rm65_qpos_init_plan.json \
  --include-steps
```

`CURRENT_QPOS_JSON` 必须符合 RM65 qpos target schema；格式和执行器边界见
[`robot_motion/rm65/arm_init/README.md`](../../robot_motion/rm65/arm_init/README.md)。生成 plan
不等于批准执行该 plan。

## 8. Quest Browser/WebXR 开发基座（R0 测试）

[`hardware/quest_dev`](../../hardware/quest_dev/README.md) 是自包含的通用 HTTPS/WSS 与 WebXR
开发模块。默认 robot feedback 关闭；可选 adapter 只读 Redis，不发布控制目标，也不会由
AgenticRobot 自动启动。无需启动服务即可运行离线单元测试：

```bash
bash scripts/run_pytest.sh -q hardware/quest_dev/tests
```

实际启动会绑定网络端口，并可能生成本机自签名证书；应按模块 README 显式执行，不属于统一
开发门禁或硬件激活步骤。

## 7. B601-RS 力觉离线验收与真机验收计划

[ReBot CAN → RM65 GUI 装配进度](./rebot_b601_rm65_can_gui.md) 记录只读输入、
单源仿真执行的应用边界、已复现 timing 缺陷与修复门禁；当前仍在开发中，不可真机验收。

[ReBot → RM65 MuJoCo 执行接入进度](../architecture/reports/rebot_b601_rm65_motion_execution_progress.md)
记录单源 human proposal、目标端 IK/dispatch 的离线验证与尚待装配的真机遥操边界。

本轮有线后端预检、反馈/计算/发送/力觉独立频率记录，以及真机分阶段操作边界见
[`B601-RS hardware commissioning`](./rebot_b601_rs_hardware_commissioning.md)。该文档中的
position probe 仍属于 M 级操作，只能由现场操作者在完成安全检查后显式运行。

[`rebot_b601_rs_haptic_acceptance.md`](./rebot_b601_rs_haptic_acceptance.md) 给出无设备的
数学/闭环效果验收命令，以及机械臂到货后从只读反馈、急停/watchdog、单关节 MIT、重力阻尼
到低幅弹簧/虚拟墙的分阶段验收。当前只能执行 H0；H1–H8 均为 `NOT_RUN`，不构成真机运动
授权。
