# B601 ↔ RM65 夹爪映射进度（独立于总路线图）

## 当前阶段

本候选实现公共 scalar contract、J7 只读采样，以及本地真 B601 → RM65
MuJoCo（机械臂+夹爪）的可选接入：

```text
B601 J7 read-only source --calibrated--> normalized_close
                                         \
                                          --> RM65 MuJoCo gripper
RM65 DMTAC read-only source -----------> normalized_close
                                          \
                                           --> B601 MuJoCo gripper
```

`normalized_close=0` 表示打开，`1` 表示闭合。映射只处理夹爪，不进入
B601 J1–J6 的 FK、重力补偿、阻尼、保护保持或故障兜底状态机。

当前候选已在独立 worktree 完成离线验证，不代表已合并或已完成真机夹爪验收：

- B601→RM65 适配器只接受 `b601_rs_gripper_motor_position_rad` 经过显式
  open/closed 标定后的样本；RM65→B601 适配器只接受
  `rm65_dmtac_normalized_close` 样本。
- `b601-rm65` application 的 `--enable-b601-gripper` 是显式 opt-in：它为
  J7 创建独立 MotorBridge 只读 reader，只调用 `mechPos (0x7019)` 参数读取，
  不调用 J7 enable、mode、MIT/POS-VEL、active-report 或 disable。J1–J6
  仍由原有 owner 独立负责。
- 夹爪样本与右臂 residual 在同一份 human proposal 中提交；RM65 的双臂
  atomic selection group 因此同时携带测量得到的 left-arm baseline，避免
  “只发 right-arm+right-hand”被目标 arbiter 拒绝。目标仍由既有 MuJoCo
  compound hand resolver 执行，夹爪值使用 `HandActionSpec.dmtac_gripper`
  的 normalized scalar。
- 适配器对非法样本、错误 source-id、未来/过期时间戳和不递增 sequence
  fail-closed，拒绝时不改变仿真目标。
- B601 端点范围默认仍严格拒绝；只读 MuJoCo 入口可通过显式
  `--gripper-calibration-tolerance-rad` 允许手工开合标定端点附近的小幅
  越界，并钳制到对应端点。容差不会传入 B601 owner，也不会改变真实电机
  的安全边界；超出容差仍会使整份 arm+hand proposal 进入 hold。
- 样本默认声明 `clock_domain_id=local_wall_time`；设备单调时钟或其他时钟域
  必须在 source adapter 内先转换，不能由仿真 target 猜测。
- B601 仿真夹爪是显式 opt-in；默认场景和默认 `B601PositionDriver` 仍保持
  原来的固定张开行为。

## 复用和边界

- RM65 侧复用已有 `HandActionSpec.dmtac_gripper`、`GripperSpec` 和 DMTAC
  backend 的 normalized scalar 语义；没有新增串口驱动。
- B601 侧复用 vendor 配置中的可选 `gripper` 组，但当前六轴 hardware
  owner 仍明确排除 J7；这里不读取 CAN，也不发送 MIT/POS-VEL 命令。
- B601 MuJoCo 默认仍为固定打开。只有显式
  `build_b601_mujoco_scene(..., enable_gripper=True)` 才注册两个辅助
  prismatic joints 的 `b601_gripper` spec，并由映射 adapter 进行可视化更新。
- ideal-effort B601 scene 不支持该 opt-in gripper，继续使用既有连续 equality
  约束；这避免把未识别的 J7/手指动力学伪装成重力或力控证据。

## 尚未声称完成

- B601 J7 的真实 open/closed motor-angle 标定；profile 中保持 `null`。
- RealMan DMTAC 实时读取接入 RM65↔B601 GUI；这需要在不复制现有 DMTAC
  backend 的前提下增加独立 observation port，并且必须先做 read-only 现场验收。
- B601 real↔RM65 sim 的夹爪路径已接入现有 `b601-rm65` receive-only GUI；
  反向 RM65→B601 目前仍只保留 MuJoCo adapter，尚未接入真实 B601 J7 写入。
- 任何真实夹爪运动、串口写入、CAN 写入或硬件验收。

## 已完成的离线证据

在候选 worktree、Python 3.10 `dexteleop310` 环境中：

```text
focused gripper + affected B601/RM65 suites: 42 passed, 2 warnings
boundary + affected motion/model suites: 255 passed, 5 warnings
ruff: PASS
py_compile: PASS
git diff --check: PASS

端点容差回归及受影响组合：148 passed, 1 warning；ruff、py_compile 和
git diff --check 继续通过。

另外完成了一次组合 proposal 运行时检查：在 `enable_gripper=True` 的实际
RM65 MuJoCo target 中，右臂 residual + J7 normalized close 同一 proposal
被接收，target `runtime.step()` 返回 `dispatched`。这只是离线证据，不是
真机夹爪运动证据。
```

其中 1 个 gripper test 使用现有 RM65 MuJoCo task scene 和真实 B601 URDF
资产验证双向目标写入；这仍是仿真证据，不是 RealMan 或 RobStride 现场证据。

## 下一步顺序

1. 运行 focused offline/MuJoCo tests，确认默认 scene 不变、双向 scalar 单调、
   过期/非法样本 fail-closed。
2. 在现场对 B601 J7 只读采集 open/closed 两端和中间点，填写命令中的
   `--gripper-open-position-rad`、`--gripper-closed-position-rad`，再用独立
   JSON 做方向验收。
3. 先执行本地 B601→RM65 MuJoCo（臂+夹爪）前台验收；确认 scalar 方向和
   双臂 baseline 后，再规划 RM65→B601 的只读仿真路径。
4. 最后才规划单独的真机夹爪命令与安全/限速验收。夹爪命令不能借用六轴
   B601 owner 的 protective-hold 假设。
