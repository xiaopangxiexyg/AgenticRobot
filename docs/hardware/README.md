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

## 5. 生成 RM65 qpos 初始化 plan（R0）

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

## 6. Quest Browser/WebXR 开发基座（R0 测试）

[`hardware/quest_dev`](../../hardware/quest_dev/README.md) 是自包含的通用 HTTPS/WSS 与 WebXR
开发模块。默认 robot feedback 关闭；可选 adapter 只读 Redis，不发布控制目标，也不会由
AgenticRobot 自动启动。无需启动服务即可运行离线单元测试：

```bash
bash scripts/run_pytest.sh -q hardware/quest_dev/tests
```

实际启动会绑定网络端口，并可能生成本机自签名证书；应按模块 README 显式执行，不属于统一
开发门禁或硬件激活步骤。
