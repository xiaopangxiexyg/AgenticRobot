# 真实系统运行入口

真实输出前必须完成设备检查、标定、工作空间和急停确认。遥操与数采参数见
[TELEOP_COLLECTION](./TELEOP_COLLECTION.md)。

| 专题 | 用途 |
|---|---|
| [TELEOP_COLLECTION](./TELEOP_COLLECTION.md) | 启动顺序、正式 profile 参数、数采验收和回退 |
| [IK 抖动诊断](./DEBUG_IK_0827.md) | 显式启用的被动诊断记录、输出边界和回退方式 |
| [原生触觉分离存储](./NATIVE_TACTILE_BUNDLE.md) | 采集期双根写入、可迁移身份、校验附着与原子恢复 |
| [长程 Progress Profile](./TELEOP_COLLECTION.md#38-可选长程进度语义与独立-gui) | GUI 选择、数据集唯一绑定、Failure/Retry 和回退方式 |
| [长程细粒度阶段标注](./LONG_HORIZON_STAGE_ANNOTATION.md) | 已封存 Episode 的视频时间轴、Failure/stage 标注和训练排除 |
| [REPLAY](./REPLAY.md) | 离线/真机 replay 参数和安全门 |
| [DEPLOY](./DEPLOY.md) | 通用 deploy 合约和验收 |
| [OPENPI](./OPENPI.md) | OpenPI 两进程部署 |
| [OpenPI RTC 状态历史](./OPENPI_RTC_STATE_HISTORY.md) | Stage3 关节、TCP 与夹爪历史输入的训练/部署合同和兼容边界 |
| [UMI](./UMI.md) | UMI 配置、标定和数采 |
| [离线评估报告](./reports/README.md) | 带日期、数据来源、验证边界和回退结论的历史结果 |
| [Tracker-Hand 旧 RPY 判定退役记录](../legacy/reports/tracker_hand_legacy_predicate_retirement_20260825.md) | 全仓无调用的旧 canonical RPY 判定移除、保留的校准接口和验证边界 |
| [共享实施计划](./plans/README.md) | 被当前报告或架构文档引用、但尚未完成的真实系统工作 |

## 执行模式边界

- canonical human teleop、`teleop_runtime.deploy` 和 `teleop_runtime.replay` 默认分别使用自己的
  source、生命周期、safety/executor 与真实 backend，不经过 `control_composition`。
- replay 的真机模式仍由 replay executor 推进唯一 cursor/plan，并由 `ReplayHardwareRig` 派发；
  GUI 中启用被动 shadow 或 trace 不会创建第二个 authority/dispatch owner。
- 两个或更多 human/policy/replay source 同时参与选择、融合或 handoff 时，必须由显式
  Control Station composition/profile 装配，并继续把 candidate 交给原 target 的唯一
  canonical safety/dispatch owner；当前页面不宣称一个通用的真实多源启动命令。

完整依赖方向和验证范围见[运行域与可拆分边界](../architecture/runtime_domain_boundaries.md)。

## 0. 运行前确认

- 从仓库根目录使用 Bash 或 zsh，并已按 [统一环境前缀](../README.md#命令约定) 激活
  `${DEXTELEOP_ENV_NAME:-dexteleop310}`。
- 如果使用复制的 checkout，先把 `DEXTELEOP_ROOT` 指向该副本；下面命令默认使用
  `/home/user/zc/DexTeleop`，不会猜测副本目录名。例如远端当前副本实际是
  `/home/user/zc/DexTelop_0726`（`DexTelop` 少一个 `e`）。
- 已完成 [设备只读检查](../hardware/README.md)；左右 side、RM65 IP、tool/TCP、Q6、DMTAC
  和相机配置与现场一致。
- [`teleop_args.sh`](./teleop_args.sh) 是当前站点参数，不是通用默认值。正式数采前检查其中的 CPU 编号、数据目录
  和设备默认值；当前远端 Ryzen 9 9950X 是 16 个物理核心/32 个逻辑 CPU，
  配方使用 `0-15` 中每个物理核心的一个线程，`16-31` 留给系统和其他进程。
- H2/M 命令必须有现场操作员、可用急停和已清空的工作空间。

## 1. Human teleop 与正式数采

### 1.1 VIVE sender（H1）

```bash
python hardware/VIVE/mocap/vive_data_sender.py \
  --host "${VIVE_RECEIVER_IP:?set VIVE_RECEIVER_IP}" \
  --port 9999 \
  --hz 300
```

`VIVE_RECEIVER_IP` 是运行接收端的主机地址；sender 与接收端不在同一机器时尤其不能沿用
别人的 IP。

### 1.2 GUI real-teleop root（M）

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui
```

进入 runtime Home 前，系统会用已标定的 VIVE desk 横向轴检查左右 wrist tracker：
三轴标定中的“身体右侧 → 身体左侧”定义为操作者坐标 `+Y`。检查会先去除配置的
`virtual_desk_post_yaw_deg`，因此默认 `180°` 场景旋转不会把左右判断颠倒。
正常情况要求 logical left tracker 的操作者坐标 Y 大于 logical right；若顺序相反，
或双手横向间距小于 5 cm 无法可靠判断，GUI 会在 Home 真正开始前弹窗并要求二次
确认。建议取消后点击“交换左右 Tracker”并重试；选择继续不会自动交换机械臂、
Q6、真手或任何持久配置。
双侧样本还必须有效且接收时间不超过 0.5 秒；按下开始后会等待首个可判定的新鲜
结果，并在准备倒计时结束、真正采集 Home 前再复核一次。运行中双臂映射已锁定时，
弹窗不会提供不可用的“交换”建议，而是要求恢复左右手位置，或安全退出后重新启动。
左右关系检查只在当前 GUI 进程第一次开始遥操时执行一次：首次通过或经人工二次
确认后，本进程后续的普通 Home、Q6 重采和 2/5 踏板切换不再重复计算左右关系，避免
把它变成持续运行监控。后续 Home 仍会独立检查 Tracker 新鲜度和稳定性。首次操作
若来自 2/5 踏板 Hold 恢复，则左右检查通过后仍走原有快速 Home。2/5 踏板切换的稳定判定为
平移速度 ≤90 mm/s、旋转速度 ≤15°/s、连续稳定 120 ms，窗口漂移 ≤6 mm/1.5°；普通
runtime Home 仍使用更严格的 15 mm/s、2.5°/s、250 ms、3 mm/0.75°。Init 单臂
到位判定另采用平移速度 ≤30 mm/s、旋转速度 ≤5°/s、连续稳定 2 s；这些阈值不改变
Tracker/相机新鲜度要求。
进入指定单臂时只采集即将 Fix 的目标侧 Home，另一侧沿用已有 runtime anchor，不会因其
Tracker 仍在运动而阻塞切换；从单臂恢复双臂时只采集需要释放的固定侧，Hold 直接恢复
双臂时仍采集双侧。
Tracker 等待与稳定 Home 采样共享 1 秒总时限；超时会取消切换并显示 3 秒红色提醒。人工处理首次左右
警告弹窗的时间不计入，二次确认后重新开始这 1 秒窗口。普通 Home 与 Q6 Home
不会被 2/5 踏板模式误识别为快速 Home。

位控的 `single_step_singularity_aware` 路径还会在 q5 接近腕部共轴区域时禁止等价的
`±pi` 腕部翻转候选参与逐帧竞争（默认 guard 为 12°），避免候选步长裁剪后在两个
等价解之间抖动。候选层还会拒绝任何辅助姿态/腕部卸载候选在局部步长裁剪后跨入
该 guard 的另一侧；q5 已经逃离 3.25° SDK 边界后，在 ±6° release band 内的
向内候选会保持上一侧的 q5，避免 `singularity_escape` 候选在边界两侧开关。可用
`--position-wrist-q5-hysteresis-release-deg 0` 仅回退 q5 回滞，或用
`--position-wrist-branch-switch-guard-deg 0` 回退分支 guard；两者均不影响旧的
`legacy_iterative` 路径。当前稳定基线为 single-step、guard=12°、q5 回滞 release=6°，
并保留上述两个参数作为逐项回退开关。

当当前 admitted 位姿到目标的误差达到 0.10 m 或 20° 时，稳定基线会暂时放松
q5 release-band 的“保持上一侧”条件，并启用 `[1,2,4]` 半径的奇异逃逸候选搜索。
这只扩大搜索和解除粘滞，不改变单周期关节速度、加速度、位置/碰撞约束；最终 qref
仍必须通过原有 admission。对于宽搜索且确有奇异度进展的候选，允许最多 0.05 m/18°
的暂时 Cartesian 误差变差，以免“有逃逸方向但被跟踪评分拒绝”；这个放宽不适用于
普通 IK 或近目标候选。目标回到近距离后自动恢复 6° 回滞。宽搜索由
`dual_arm_teleop` safety profile 控制，关闭它即可回退单半径搜索。

双踏板模式以及五踏板模式下的 B 都是“双臂遥操 / Hold”状态键，并不始终等同于普通
“开始遥操”：双臂遥操时按 B 进入 Hold；Hold 时按 B 经过快速 Home 恢复双臂遥操；
任一单臂 Fix 状态按 B 时，当前运动臂和数采保持连续，固定臂继续冻结到后台 Home
就绪后再原子释放为双臂遥操，不触发全局 Hold，也不暂停数采。Init 移动、精确 qpos
或 Init Home 重采期间不会接受 B/踏板 1/2 的模式切换；此时 C 仍始终可用，用于先进入
常规 Hold，再重新选择单臂/双臂遥操。五踏板的 A 仍为开始/停止数采，B/C 的派发和
双踏板专用 B/C 完全一致；F15/F16/F17 仅承担进度推进/失败/Hold/撤销。

### 踏板:

```bash
sudo -E "$(command -v python)" -m teleop_runtime.configure_devices --fix-permissions

python -m hardware.foot_pad --count 2
```

### 1.3 正式 position 数采（M）

新建的正式 RealTeleop/headless 数据集统一写入
`/home/user/zc/datasets/RealmanDMTAC/FormalCollection`；collector 继续在其下创建
`<YYYY-MM-DD>_<collection-task-name>`。`DATASET_BASE` 的显式覆盖仍然保留给旧数据兼容、
离线测试和受控回退，但不会迁移已有数据，也不应作为新的正式采集位置。

```bash
: "${DEXTELEOP_ROOT:=/home/user/zc/DexTeleop}"
(
  if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
    printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
    exit 1
  fi
  cd -- "$DEXTELEOP_ROOT"
  export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  eval "$(conda shell.bash hook)"
  conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
  set -euo pipefail
  : "${LEFT_ARM_IP:=192.168.101.21}"
  : "${RIGHT_ARM_IP:=192.168.1.20}"
  : "${DATASET_BASE:=/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"
  export LEFT_ARM_IP RIGHT_ARM_IP DATASET_BASE
  source docs/real/teleop_args.sh

  taskset -c 0-7 \
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
    "${COMMON_REAL_TELEOP_ARGS[@]}" \
    "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
    "${DMTAC_A_ARGS[@]}" \
    "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
    "${DATASET_COMMON_ARGS[@]}" \
    "${POSITION_FORMAL_V1_ARGS[@]}" \
    "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
    "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}" \
    --dataset-root "$DATASET_BASE" \
    --collection-task-name dmtac_rm65_bimanual_position_v1 \
    --collection-task-description "formal rm65 bimanual position v1: accepted-qref FK action, IK 1.0/0.35 min2/max4, 15/60/60/120" \
    --q6-config latest --q6-config-required \
    --dmtac-config latest --require-dmtac-config \
    --camera-config teleop_runtime/configs/devices/realsense.json \
    --camera-color-format bgr8 \
    --write-parquet --write-video
)
rc=$?
printf '[DexTeleop] command exited with status=%s\n' "$rc"
```

#### 1.3.1 固定的 60 Hz 阻抗数采 + 四阶段标注 + 五踏板命令

正式 writing 数采使用仓库内固定 launcher
[`run_formal_60hz_writing_collection.sh`](../../scripts/run_formal_60hz_writing_collection.sh)。
它只组合现有、已准入的参数数组和 canonical RealTeleop 入口，不维护第二份控制参数。
默认 dataset task name 是拼写正确的 `4_stage_writing_detail`；在线 Progress profile
固定为 `dmtac_rm65_bimanual_position_writing_four_stage_progress`。这个默认入口选择
coarse capture profile，采集后可以在标准 annotation 中独立选择 writing detail overlay；
这只是默认 capture 选择，不是对 detailed profile 的禁用，detail 标注选择也不会反向改写
采集事实。

如果是全新的、尚未绑定 stage profile 的 dataset，也可以在 progress GUI 中显式选择
`dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed`。这个选择是正常的
capture profile，会直接把 `pen_down / write / pen_up` 三个 fine stage 绑定到新 dataset，
writer、机械臂控制和 progress manifest 都沿既有采集路径运行；已有 coarse dataset 不能
在原目录内混用两种 stage hash。8 月 24 日这类已经用 coarse profile 采集的 writing
Episode，继续使用上面的 coarse 入口即可，进入标准 annotation 后再选择并封存 additive
detail overlay。

先在终端 1 启动五踏板转发服务：

```bash
cd /path/to/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/run_formal_60hz_writing_collection.sh foot-pad
```

再在终端 2 启动正式数采；触觉目录通过参数显式选择：

```bash
cd /path/to/DexTeleop
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
bash scripts/run_formal_60hz_writing_collection.sh collect \
  --tactile-bundle-root /absolute/path/to/DexTeleopTactile
```

触觉根目录必须预先存在、不能是符号链接，并且必须与 dataset root 保持目录隔离；
两者可以位于同一文件系统。
launcher 默认保留 200 GiB 空闲空间，可用 `--tactile-min-free-gb` 显式调整。dataset 根目录
和 task name 也可以分别用 `--dataset-root`、`--task-name` 覆盖；等价的环境变量是
`TACTILE_BUNDLE_ROOT`、`TACTILE_BUNDLE_MIN_FREE_GB`、`DATASET_BASE` 和
`COLLECTION_TASK_NAME`。task name 只决定正式 dataset 的任务名/目录名，不会隐式改变
progress 语义；要选择三阶段或 detail 变体，应显式使用 launcher 的
`--progress-profile` / `--stage-profile`，或设置 `COLLECTION_PROGRESS_PROFILE` /
`COLLECTION_STAGE_PROFILE`。例如直接选择现有 detail 变体：

```bash
COLLECTION_TASK_NAME=3_stage_writing_detail \
COLLECTION_PROGRESS_PROFILE=catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed \
bash scripts/run_formal_60hz_writing_collection.sh collect \
  --tactile-bundle-root /absolute/path/to/DexTeleopTactile --dry-run
```

launcher 会把 `*_detailed` catalog ref 展开为 coarse `--collection-progress-profile` 加
匹配的 `--collection-stage-profile`，并在 dry-run/启动诊断中打印最终 task name、profile 和
stage。若当前 shell 已经继承旧的 `COLLECTION_TASK_NAME`，写法
`COLLECTION_TASK_NAME="${COLLECTION_TASK_NAME:-3_stage_writing_detail}"` 会保留旧值；
需要强制改名时请直接写 `COLLECTION_TASK_NAME=3_stage_writing_detail` 或传
`--task-name 3_stage_writing_detail`。正式启动前可在任一命令末尾追加 `--dry-run`，只打印完整解析结果，
不会连接踏板、机械臂、相机、夹爪或触觉设备。

正式 writer 的子进程 ready 握手不属于全局设备预检；它会在按下开始数采后，针对当前 dataset
路径、task/profile 和模态合同一次性初始化。初始化失败只会拒绝本次数采并保留遥操/维护 GUI；
重试会创建全新的 writer 子进程，不会复用已经失败的 `multiprocessing.Process`。

需要临时降低 reserve 时，优先使用显式参数，例如
`--tactile-min-free-gb 2`。Shell 写法
`TACTILE_BUNDLE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-2}"` 只会在变量未设置时
采用 2；若当前终端已经继承了 200，它仍会保留 200。launcher 和 canonical Python
入口都会报告最终生效值及其来源，实际采集前应核对该诊断；降低 reserve 不会增加磁盘
容量，仍需按预计采集量保留余量。

该 collect 命令固定展开
`POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS`，即 15 Hz
操作输入、60 Hz IK/qref、120 Hz qcmd，以及双臂平移/旋转 bounded force-admittance
overlay；同时启用 Progress GUI、external tactile、artifact integrity 和空闲期无损压缩。
该 profile 保留 fix-arm 控制器；active admittance 在固定侧通过运行时 side-activity
门控暂停 modifier，在活动侧继续使用相同的 bounded overlay，因此不会因五踏板侧切换而
丢失原有 60 Hz 控制配置。
五踏板 1/2 继续使用单臂 Fix 语义，3/4/5 分别为 Advance、Failure、Undo；踏板服务必须
与数采进程分终端运行。

当前 `POSITION_FORMAL_V1_ARGS` 已内置稳定的 single-step 奇异保护（guard=12°、
q5 回滞 release=6°），因此上面的正式命令就是稳定版命令，不需要另加实验数组。
如需逐项回退，显式追加对应参数即可；完整旧链路回退使用
`--position-ik-step-mode legacy_iterative`。

针对“非奇异区 post-IK EMA 仍然不跟手”的修订以独立、可回退的
`rm65_bimanual_position_v2_responsive` profile 提供。验证 V2 时，将上面命令中的
`"${POSITION_FORMAL_V1_ARGS[@]}"` 替换为
`"${POSITION_FORMAL_V2_RESPONSIVE_ARGS[@]}"`，并将 task name 改为
`dmtac_rm65_bimanual_position_v2_responsive`。V2 仅在单侧状态为 `tracking`、tracker
运动窗口有效、q3/q5 均在 4° guard 外且腕部 null projection 未介入时，把该侧
post-IK alpha 提升到 0.70；进入 guard、发生 hold/boundary 或 q5 escape latch 仍使用受保护
路径。恢复 `POSITION_FORMAL_V1_ARGS` 即完整回退，不能把 V1/V2 episode 写入同一个
task/profile 数据集合。

`POSITION_FORMAL_V2_BALANCED_ARGS` 是已经过真机 A/B 的冻结稳定基线。它只把上述
普通区 alpha 从 0.70 降为 0.55；IK、限位/分支保护、力急停和 120 Hz 执行配置均与
V2 相同。若继续验证残余约 30 Hz 微振动，使用默认关闭的
`POSITION_FORMAL_V2_SMOOTH_ARGS`；它只把该 alpha 进一步降到 0.45。切回 BALANCED
即完整回退，RESPONSIVE/BALANCED/SMOOTH 不得写入同一个 task。

J3/J5 物理限位附近的 `joint_limit_escape` 仍保留在 `legacy` 回退链路中。验证不制造
固定向外步的边界保护时，在 V1 或 V2 数组之后显式追加
`--position-joint-limit-guard-mode boundary_hold` 和
`--position-joint-limit-boundary-margin-deg 1.5`，并使用独立 task name。
`boundary_hold` 会从候选池移除旧的人工 limit-escape 分支；只有普通候选进入 1.5°
硬限位带且继续向内时，才把对应关节分量投影为零步，向外运动立即放行。投影后仍会
复检普通误差界、关节动态和碰撞；在正常误差界内保留其他关节/另一侧运动，超界才
退回整对 reference hold。删除这两个命令行覆盖即可原子回退到数组内显式声明的
`legacy`，不要通过把 boundary margin 扩大到旧 8° escape 激活带来模拟该模式。

若活动遥操帧最终仍被 RM65 发送前的绝对关节限位门禁拒绝，运行时把它作为独立的
`joint_hard_limit` 安全事件处理：请求双臂 `Arm_Stop`、结束当前 episode 并进入 Hold，
但保留 GUI 和设备连接，供人工调整后重新采集 runtime Home。该事件不会伪装成碰撞，
也不会写入 `force_collision_detected`；启动/归位阶段及不符合该精确错误格式的异常仍会
正常抛出，避免隐藏配置或程序错误。上游允许 `1e-8 rad` 纯数值端点误差，发送门禁以
`2e-8 rad` 提供两倍比较余量；所有通过的目标都会投影到物理限位内侧 `1e-8 rad` 后
再交给轨迹器/SDK。这只吸收浮点换算误差，不会向外放宽 RM65 的物理关节范围。

遥操侧的 admitted-anchor QP 速度/加速度约束保持 default-off。测试时将
`POSITION_FORMAL_V1_ARGS` 替换为 `POSITION_FORMAL_V3_QP_CONSTRAINED_ARGS`；该数组只在
稳定 V1 上增加 `--position-ik-admitted-motion-limit`，不会改变 source shaping、碰撞或
受力急停。恢复 `POSITION_FORMAL_V1_ARGS` 即完整回退。

真机多相机默认使用 `bgr8` 颜色传输；它不改变 640×480 等请求分辨率，只避免多台
RealSense 同时协商 `yuyv` 时出现首帧迟到和底层 SDK 重启。启动首帧等待（5 秒）与
运行期新鲜度等待（1 秒）分开。现在每台
相机成功启动时还会打印 `pipeline_start_sec`、`sensor_configure_sec` 和 `warmup_sec`：
前者偏大表示 USB/UVC/设备内部协商耗时，后者偏大表示首帧等待；这三段都发生在相机
源启动阶段，与 parquet/video writer 无关。只有在已验证的单站环境中才应显式改用
`--camera-color-format yuyv`。

相机协商仍必须串行。默认的
`--camera-startup-warmup-policy bounded_first_frame` 会让每台相机在 `pipeline.start()` 后
只等待第一帧，且每次 pipeline 尝试受 `startup_frame_grace_sec`（当前默认 5 秒）约束；随后仍由
原有的三相机连续新帧自检完成正式 admission。这个开关不会更改视频内容、分辨率、
颜色格式或全局自检，也不会并行启动 native pipeline。单次 SDK wait 使用运行期等待上限
（当前默认 1 秒），不再出现一次 `Frame didn't arrive in 4999` 吞掉完整预算；每次 pipeline
最多连续尝试 5 次。首次仍无首帧时，只关闭并重新打开这一台失败相机的 pipeline 一次，
不做 USB hardware reset，也不影响已启动的其他相机。第二次仍失败才进入 GUI 等待人工重连。
人工重连成功后会重新执行三台相机的连续新帧自检，并用新结果替换 GUI 中旧的“启动自检失败”
快照。显式使用 `--camera-startup-warmup-policy legacy` 可原子回退到旧的 10 帧同步 warmup 与 reset
retry。若日志中偏大的是 `pipeline_start_sec` 而非 `warmup_sec`，该开关不会缩短 SDK
自身的 USB/UVC 协商时间，应继续检查 USB 拓扑、独占占用和设备状态。

`POSITION_FORMAL_V1_ARGS` 会强制启用独立的双臂位控受力急停：通过左右 RM65
状态连接以 20 Hz 读取 tool-zero 六维力，默认任一侧 `|F| > 30 N` 或
`|T| > 5 Nm` 即锁存并并行请求左右臂 `Arm_Stop`。需要恢复此前更保守的完整
`15 N / 2 Nm` 配置时，在所选正式 position 数组后追加
`"${POSITION_FORCE_GUARD_CONSERVATIVE_ARGS[@]}"`；该数组同时保留 tool source、20 Hz、
单样本触发、3 次读取错误上限和 0.4 s stale watchdog。启动时必须看到
`force_limits_armed=true stale_watchdog=deferred_until_camera_ready`，随后在相机启动
完成且任何归位/遥操动作前看到
`stale_watchdog=armed_after_fresh_post_camera_samples motion_admission=ready`。
相机初始化期间真实超力和连续读取错误仍会 fail-closed；只有数据陈旧 watchdog
延后启用，且启用前强制要求左右各产生一帧新的有效力样本。运行期力读取连续失败、
数据陈旧或启动后无法取得双侧新鲜样本都会 fail-closed。若故障发生在已进入遥操的运行期，
双臂仍会锁存急停，但当前 episode 会先结束并保留受力碰撞标签，GUI、相机、tracker、手部
和力监控连接保持在 Hold；人工调整机械臂后点击“开始遥操”会刷新真实 qpos、重启 CANFD
worker 并重新取得双侧新鲜力样本。只有 Arm_Stop 未确认、后台 worker 无法退出或启动阶段
故障才要求检查设备并重启进程。这里的保护与 impedance 使用的 `--force-tare-sec` 无关。

### 1.4 Impedance/力控 15→40 Hz 数采（M）

力控控制器的 `hard_stop_force_n`/`hard_stop_torque_nm` 触发时同样按碰撞处理：
左右臂并行请求 `Arm_Stop`，当前 episode 结束并写入 `detected_collision`，但不会关闭
GUI、相机、tracker、手部或其他设备连接。Arm_Stop 均确认后，GUI 会进入 Hold；人工调整
机械臂并再次点击“开始遥操”时，系统会以实时 qpos 重建力控/跟随线程并重新采集 runtime
Home。若任一 Arm_Stop 未确认或控制线程无法退出，则保持 fail-closed 并要求重启，不能
通过普通 Hold 清除急停锁存。

```bash
: "${DEXTELEOP_ROOT:=/home/user/zc/DexTeleop}"
(
  if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
    printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
    exit 1
  fi
  cd -- "$DEXTELEOP_ROOT"
  export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  eval "$(conda shell.bash hook)"
  conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
  set -euo pipefail
  : "${LEFT_ARM_IP:=192.168.101.21}"
  : "${RIGHT_ARM_IP:=192.168.1.20}"
  : "${DATASET_BASE:=/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"
  export LEFT_ARM_IP RIGHT_ARM_IP DATASET_BASE
  source docs/real/teleop_args.sh

  taskset -c 0-11 \
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
    "${COMMON_REAL_TELEOP_ARGS[@]}" \
    "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
    "${DMTAC_A_ARGS[@]}" \
    "${DATASET_COMMON_ARGS[@]}" \
    "${IMPEDANCE_40HZ_ARGS[@]}" \
    "${CONTROL_DEBUG_SUMMARY_ARGS[@]}" \
    --dataset-root "$DATASET_BASE" \
    --collection-task-name dmtac_impedance_15_40hz \
    --collection-task-description "full-device bimanual impedance 15Hz capture / 40Hz backend" \
    --q6-config latest --q6-config-required \
    --dmtac-config latest --require-dmtac-config \
    --camera-config teleop_runtime/configs/devices/realsense.json \
    --write-parquet --write-video
)
rc=$?
printf '[DexTeleop] command exited with status=%s\n' "$rc"
```

### 1.5 无 collection、无 debug 写盘的卡顿基线（M）

这个命令保留双臂、Q6、DMTAC 夹爪/触觉和 RGB 相机，但不创建 dataset
writer、不启动腕部 depth stream，也不写 standalone debug 文件。它只在终端输出
超过 66 ms 的慢帧和主循环 stall，用于判断卡顿是否仍存在于控制/传感器链路。
慢帧文本通过有界后台队列输出，终端或 SSH 反压不会在遥操控制线程内执行
同步 `flush`。

```bash
: "${DEXTELEOP_ROOT:=/home/user/zc/DexTeleop}"
(
  if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
    printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
    exit 1
  fi
  cd -- "$DEXTELEOP_ROOT"
  export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  eval "$(conda shell.bash hook)"
  conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
  set -euo pipefail
  : "${LEFT_ARM_IP:=192.168.101.21}"
  : "${RIGHT_ARM_IP:=192.168.1.20}"
  : "${DATASET_BASE:=/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"
  export LEFT_ARM_IP RIGHT_ARM_IP DATASET_BASE
  source docs/real/teleop_args.sh

  taskset -c 0-7 \
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
    "${COMMON_REAL_TELEOP_ARGS[@]}" \
    "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
    "${DMTAC_A_ARGS[@]}" \
    "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
    "${POSITION_FORMAL_V1_ARGS[@]}" \
    "${NO_COLLECTION_SENSOR_BASELINE_ARGS[@]}" \
    --q6-config latest --q6-config-required \
    --dmtac-config latest --require-dmtac-config \
    --camera-config teleop_runtime/configs/devices/realsense.json \
    "${CONSOLE_TIMING_ARGS[@]}" \
    "${NO_DEBUG_DISK_WRITE_ARGS[@]}"
)
rc=$?
printf '[DexTeleop] command exited with status=%s\n' "$rc"
```

不要在这个基线命令中加入 `DATASET_COMMON_ARGS`、
`POSITION_COLLECTION_PROCESS_ARGS`、`--write-parquet` 或 `--write-video`。
`CONSOLE_TIMING_ARGS` 已由 `teleop_args.sh` 定义；不要用未定义的 Bash 数组占位。

### 1.6 Headless/兼容 collection（先 R0）

先看帮助：

```bash
python -m teleop_runtime.collection --help
```

### 1.7 UMI 配置验证（R0）

要求左右 hand-eye 标定已存在：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --dataset-root /home/user/zc/datasets/RealmanDMTAC/FormalCollection/umi_dual_wrist \
  --validate-only
```

详细 UMI 流程见 [UMI](./UMI.md)。

## 2. Replay

离线 Replay GUI（R1）：

```bash
python -m teleop_runtime.replay \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

真机 replay 仍使用同一入口，但必须显式配置 arms/tools/backend，并在 GUI 中人工执行
`Move Start` 和 `Start`。参数和安全边界见 [REPLAY](./REPLAY.md)。MuJoCo replay 放在
[仿真入口](../sim/README.md)，不混在真实系统命令中。

## 3. Deploy

Canonical CLI 帮助（R0）：

```bash
python -m teleop_runtime.deploy --help
```

OpenPI artifact contract 检查（R0）：

```bash
python -m deployments.openpi.client.real \
  --checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" \
  --contract-check-only
```

实机观测和推理、不发送动作（H1）：

```bash
python -m deployments.openpi.client.real \
  --checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" \
  --deployment-profile rm65_bimanual_dmtac \
  --host "${POLICY_SERVER_HOST:?set POLICY_SERVER_HOST}" \
  --port 8078 \
  --max-steps 100
```

只有完成 qpos-init、ready gate 和人工确认后才允许增加 `--send-actions`。通用说明见
[DEPLOY](./DEPLOY.md)，OpenPI 两进程流程见 [OPENPI](./OPENPI.md)。
