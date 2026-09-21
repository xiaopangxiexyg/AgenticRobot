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

## v4/4b2 目标资产与 J7 阻尼边界（本候选）

`b601-rm65` 的 MuJoCo target 现在默认选择已审核的 v4/4b2 组合：

- URDF：`assets/rm65_inspire_gripper_v4/rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf`
- camera registry：`robot_motion/configs/rm65_camera_registry_v4.json`
- gripper profile：`dmtac_bimanual_position_eg2_4b2_cad_20260818.v1`

三个值都可以通过 `b601-rm65` 的 `--target-urdf`、
`--target-camera-registry` 和 `--target-gripper-profile` 显式覆盖，方便离线
比较而不复制或修改 task JSON。旧 task/mapping 中的
`left_gripper`/`right_gripper` 逻辑名在场景只有一个对应侧 gripper 时会解析到
v4/4b2 的实际 CAD 名称；多个候选时仍然拒绝猜测。

J7 是夹爪，不属于当前 B601 J1--J6 gravity/damping owner。当前只读 J7
reader 不发送任何 MIT/POS-VEL 帧，因此修改六轴 owner 的 `--driver-kd` 不会
改变 J7，也不会让 J7 进入 protective hold。官方 `reBotArm_control_py` 路径
中，J7 的阻尼来自 `config/rebotarm_rs.yaml` 的
`joints[name=gripper].MIT.kd`（当前为 `4.0`），并在每个
`gripper.send_mit(..., kd=...)` 帧中生效；`JointGroup.mode_mit(kd=...)` 可在
运行时覆盖它。建议先以 `kd=1.0` 做现场小范围验收，确认无明显回弹后再考虑
`0.5`，不应把 `0x700B` 当作阻尼参数（它是 torque limit）。

由于该配置位于 pinned vendor submodule，当前候选没有直接改写厂商文件；需要
真实 J7 控制时，应在官方 gripper owner 中显式设置运行时 `kd`，或另建一个有
 独立审查记录的 vendor 子模块提交。B601→RM65 read-only 遥操路径不会因为
 v4/4b2 目标切换而隐式使能或控制 J7。

## J7 运行时 Kp/Kd 与配置路径

本候选新增了独立的前台 `scripts/hardware/rebot_b601_rs_gripper_owner.py`。
它只注册 RobStride motor 7，启动前读取当前 `mechPos` 作为保持参考，调用单电机
`ensure_mode(MIT)`、`enable`、`send_mit`，退出时只对 motor 7 `disable`；绝不调用
`Controller.enable_all()`、`disable_all()`，也不进入六轴 B601 owner 的 protective
hold。运行时参数为：

```text
--gripper-kp 50.0
--gripper-kd 1.0
```

省略时从 `joints[name=gripper].MIT.kp/kd` 读取（当前 `50.0/4.0`）。这些参数
只作用于本次 MIT 帧，不写回驱动器非易失参数，也不会改变 J1–J6 的重力补偿。
命令仍必须显式给出 `--allow-motion --auto-disable
--confirmation B601_RS_J7_MIT_GRIPPER`；当前只完成离线/模拟验证，未执行真机。

例如只把本次 J7 阻尼从配置中的 `4.0` 改为 `1.0`，而不改 YAML：

```bash
cd <AgenticRobot>
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_gripper_owner.py \
  --config "$PWD/hardware/reBotArm_control_py/config/rebotarm_rs.yaml" \
  --channel can0 \
  --gripper-kp 50.0 \
  --gripper-kd 1.0 \
  --duration-sec 30 \
  --rate-hz 20 \
  --allow-motion \
  --auto-disable \
  --confirmation B601_RS_J7_MIT_GRIPPER
```

该命令是前台、单 J7、短时 hold；它不是 B601 六轴 owner，也不是夹爪开合
轨迹播放器。现场执行前必须确认没有其他进程拥有 J7，并保持物理急停可用。

此前命令显示的
`/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml`
是历史 commissioning 流程使用的 sibling checkout。它与项目内的
`hardware/reBotArm_control_py` 是两个独立工作树，虽然当前都固定在同一个上游
commit；外部路径不是运行时必须依赖，也不应成为项目默认。当前候选已把 owner 和
MuJoCo launcher 的默认值切换为项目内 pinned submodule：

```text
<AgenticRobot>/hardware/reBotArm_control_py/config/rebotarm_rs.yaml
```

这样可由父仓库 gitlink 固定 vendor 版本、避免复制厂商代码，同时保留显式
`--config`/`--rebot-source-root` 覆盖，方便现场使用已安装的 sibling checkout。
因此外部路径仍可用于人工验收，但不再是代码隐含的默认路径。
