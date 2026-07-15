# DexTeleop 当前复核（临时）

> 复核日期：2026-07-15（本地工作区快照）  
> 复核方式：阅读当前目录、主要 README/schema、Git 修改与新增模块，并运行本地无硬件测试。未连接远端、相机、VIVE、RM65、夹爪或触觉设备，未执行任何运动、设备复位或权限修改。  
> 历史版本：[legacy/README_REVIEW_TMP_20260711_pre_review.md](legacy/README_REVIEW_TMP_20260711_pre_review.md) 仅保留当时背景；本文件是当前交接依据，但不是安全放行文档。

## 1. 结论先行

仓库已从“若干真机入口各自携带完整实现”进一步转向分层架构：

```text
dataset_contracts     稳定、硬件无关的数据与 policy artifact 契约
dataset_interface     训练时视频/mask/state 输入与增强
robot_motion          sim/real 共用动作、位控、回放、运行时和安全领域逻辑
sim                    MuJoCo 物理/回放及跨域仿真契约
teleop_runtime         真机设备组装、采集、遥操、回放和部署
deployments            模型客户端与 OpenPI 适配
utils                   离线后处理、标定和可视化
legacy                  已封存实现和历史材料
```

相较 2026-07-11 版本，当前工作区已经有几项实质进展：根目录测试边界已建立；数据契约扩展到 v2.2/v2.3、typed end-effector 和 policy artifact；15→75→150 Hz 位控链贯通采集、MuJoCo replay、真机 replay/deploy；VIVE 增加速度、运动意图和可选时钟同步；真机遥操增加 angle trigger fallback、foot pad 控件和更多可测试组件。

但当前工作区仍不应作为可发布统一基线：

1. 工作树包含 **191 个未暂存修改、101 个删除记录和 171 个未跟踪项**，目录迁移、功能开发、参数调优、测试和临时文档交织在一起。
2. 根测试已可收集 **1,498** 项；恢复无硬件基线后，全量结果为 **1,464 passed、34 skipped、0 failed**。真实 loopback UDP 用例已显式标记为 integration，并在当前环境不允许 UDP socket 时跳过。
3. 新位控链只认证到所选场景中最终 75 Hz qpos endpoint；75→150 Hz 插值整段尚未做连续碰撞认证，自定义末端工具几何也不能由 fixed proxy 自动代表。
4. `v22_native` typed action 的契约载体已经存在，但当前真机 deploy 的 native safety/backend adapter 尚未完整，设计上应继续 fail-closed。
5. 真机主入口仍有 18,772 行，采集/部署/回放入口也继续增长；已抽出的组件降低了局部耦合，但尚未改变组装入口过大的事实。

## 2. 本轮核实的工程基线

### 2.1 工作区与测试入口

| 项目 | 当前结果 | 判断 |
| --- | --- | --- |
| 本地解释器 | `/home/zc/anaconda3/bin/python`，Python 3.12 | 不是 README_ENV 指定的 `dexteleop310` / Python 3.10，不能替代目标环境验收。 |
| Git 状态 | 191 modified、101 deleted、171 untracked | 其中很多是“删旧目录 + 新目录未跟踪”的迁移表示，不应简单解释为功能删除。 |
| 顶层测试配置 | 新增 `pytest.ini`，`testpaths=tests`，排除 OpenPI 子树并声明 unit/contract/integration/optional/hardware markers | 旧报告中的“根 pytest 会误收集 OpenPI、没有 pytest 配置”已经解决。 |
| 收集 | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --collect-only -q` → 1,498 collected | 根入口现在可复现收集主测试集合。 |
| DMTAC 定向复测 | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dmtac_impedance_tuning.py` → 8 passed | 已覆盖的 profile 参数、派生阈值和单步回中结果一致。 |
| 最终全量复测 | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` → 1,464 passed、34 skipped、0 failed | 当前无硬件默认基线全绿；skip 包含可选依赖及环境不允许的真实 UDP integration 用例。 |

根目录仍没有统一的 `pyproject.toml`/锁文件或 CI workflow。`pytest.ini` 解决了收集边界，但没有解决 Python 版本、核心/可选依赖和 CI 矩阵的可复现性。

### 2.2 首次 14 项失败及处理结果

| 分类 | 数量 | 当前证据 | 建议 |
| --- | ---: | --- | --- |
| 受限环境 UDP | 6 | VIVE clock sync/sender/source 的真实 loopback 测试需要 UDP socket | 已标记为 integration；共享 fixture 在 UDP 不可用时明确 skip，在普通允许 loopback 的环境仍运行真实测试。 |
| device preflight 契约漂移 | 2 | 旧测试未跟随 angle trigger fallback 和独立 impedance 外环频率 | 已对齐：fallback 时不强制 Q6；自动 minimum control Hz 为 15；arm step dt scale 保留 3.0 安全下限。 |
| DMTAC impedance 调优漂移 | 5 | 已确认当前左右 profile 是权威基线；测试原先仍断言中间调优值 | 已同步测试覆盖的 compliance、offset cap、return cap/rate、派生饱和阈值和 40 Hz 单步回中结果。未覆盖参数保持未测试。 |
| GUI 状态文本漂移 | 1 | gripper 状态已泛化为输入源、有效性、预览 target 和实际 send target | 测试已分别断言 `input=q6`、`valid=ok`、preview target 与 `send_target`，不再把两者混为同一字段。 |

上述 14 项首次失败已经逐类处理：DMTAC 参数测试以确认的 profile 为权威，3 项当前行为契约完成同步，6 项真实 UDP 用例具有显式 integration/skip 边界。当前默认无硬件测试不再有失败。

## 3. 当前框架与新增能力

### 3.1 数据契约、训练输入和 v2.3 位姿框架

`dataset_contracts` 已不只是 v2.1 兼容层，当前包含：

- v2.1 compatibility、v2.2 typed layout、v2.3 pose-frame metadata；
- Parquet 行契约、episode 统计、触觉和 wrist-depth codec；
- `PolicyIOLayout`、layout hash、policy artifact、state input mode；
- 细粒度 `source_id`、raw absolute frame、`raw_to_desk_transform` 及每侧 override。

v2.3 默认以 `desk_absolute` 存储/解释 pose；relative action 是 loader/replay 的 chunk-local SE(3) 表示，不是新的 Parquet 向量布局。v2.2 与更旧数据仍以带来源标记的 legacy raw absolute 暴露，禁止靠列名或当前机器标定猜 frame。

`dataset_interface` 新建为训练输入层，负责：

- RGB 视频与 wrist final keep-mask 读取；
- front 固定边界背景替换、全图非几何 photometric augmentation；
- 归一化后连续 state 的 noise/dropout 与 masked input；
- modality validity 过滤、Torch adapter 和 pose/action window。

边界设计是正确的：离线生成/修复仍在 `utils.dataset_postprocess`，训练时读取和增强在 `dataset_interface`，稳定物理格式在 `dataset_contracts`。现有 dependency-boundary 测试覆盖了 runtime/sim 不反向导入 offline implementation、旧路径 re-export 和数据包隔离导入。

### 3.2 动作、末端和部署契约

动作领域实现正在从 `teleop_runtime.action` 下沉到 `robot_motion.action`；旧路径仍作为兼容层。当前 v2.2 低维契约具有：

- 固定双臂 `left_pose7 + right_pose7` 核心；
- 左右独立的 continuous/discrete/none end-effector spec；
- float state/action 与 int64 token 完全分离；
- dataset、checkpoint/server、replay 和 deploy 共用 layout hash；
- camera logical key → model field → server slot 由 artifact 绑定。

执行路径需要继续区分：

```text
v22_q6_bridge  typed output -> 显式 legacy q6 slot 投影 -> 既有 26D safety/dispatch
v22_native     typed output -> typed RobotAction/TeleopCommand -> native adapter
```

前者已经有较完整契约测试；后者的类型载体存在，但真机 `run_policy` 尚未完整接通 native safety/backend，当前 fail-closed 是合理边界。不能通过 policy adapter 直接构造 backend command 绕开它。

### 3.3 15→75→150 Hz 位控与因果采集

`robot_motion.position_control` 新增不可变 reference/command contract、因果 timeline、pose/qpos 重采样、插值、trajectory、安全与审计。当前频率语义已拆分为：

```text
15 Hz   policy action / dataset sample
75 Hz   position reference、双臂 coupled IK 与 endpoint safety
150 Hz  RM65 CANFD joint command
40 Hz   impedance backend（独立分支）
```

采集链将 `control_step()` 与 `record_step()` 分开，15 Hz 样本只选择 sample timestamp 之前、未过期且 epoch/sequence 完整的 reference；缺失时丢样，不用 backend 当前命令伪造 action。observation 优先记录因果实测关节/FK，同时保留 commanded/reference 审计。

real replay/deploy 的 position 主链现为：

```text
15 Hz target pose
  -> keyframe safety
  -> 75 Hz pose interpolation
  -> coupled dual-arm collision IK
  -> joint limits
  -> endpoint collision recheck
  -> shared 75→150 qcmd interpolation
  -> sim/real sink
```

重要边界：online deploy 为因果插值引入一个 15 Hz action period 延迟；手/夹爪仍按 15 Hz 离散调度；arm/hand 的故障处理是 fail-stop，不是跨设备原子事务。真机 position 模式要求完整、成功且新鲜的双臂实测 qpos，不允许猜被动臂默认姿态。

### 3.4 安全、碰撞与审计

已确认并应保留的边界：

- 默认双臂场景使用明确的 12 对碰撞集合，包括左右 link3 proxy 与对应 camera pole 的对偶检查；
- impedance teleop 不依赖不可配置的 SDK 奇异检测；普通 position teleop 保留进入奇异区 hold；
- position replay 的奇异慢速目标必须先以实际双臂 qpos 重新做批准碰撞检查，再允许 dispatch；
- backend 区分 attempted、accepted、measured，SDK 失败不推进 accepted anchor；
- reference/command/timing 支持 bounded、非阻塞审计，`--control-debug` 不应改变控制语义。

仍未闭合的安全项：75→150 插值段连续碰撞认证、真实安装几何验收、自定义工具 profile，以及双臂 dispatch 的非原子窗口。DMTAC impedance 已确认以当前左右 profile 为参数权威，但这不等于未经执行的真机行为已经验收。

### 3.5 VIVE、控制输入和真机遥操组件化

VIVE sender/source 现包含 300 Hz 默认发送、OpenVR snapshot velocity、序列/重复包诊断、time-based history、motion intent，以及独立控制端口上的可选四时间戳 clock sync。legacy sender 保持 v1 packet shape 和 clock control off，兼容方向清楚。

真机输入新增或强化：

- Q6 优先、未发现时回退已标定 angle trigger；
- foot pad 控件；
- tracker loss grace/manual hold、reacquisition baseline reset；
- wrist-depth freshness、Q6 hold gesture、temporary dual-arm limiter；
- control profile、side mapping、watchdog、debug recorder；
- GUI controls/tracker/special-arm/VIVE calibration 等叶子组件。

这些模块已放入 `teleop_runtime.real_teleop` 并有 component tests，是合理的渐进抽离。不过主入口从上一版记录的 17,144 行增长到 **18,772 行**，说明新增能力仍大量汇聚在旧入口，组件化尚未转化为入口收缩。

### 3.6 MuJoCo 与 sim/real 边界

旧 `sim/mujoco_real2sim` 已迁到 `sim/mujoco`，并拆出：

- `sim/contracts`：action、observation、timing、scene 和 manifest；
- `sim/mujoco/modeling`：geometry/collision/world/object 组装；
- `sim/mujoco/replay`：数据回放、v2.2 writer、RGB distortion；
- position schedule 与 real 共用 `robot_motion.position_control`。

MuJoCo 是 canonical task-physics backend，SAPIEN 仍是 teleop/debug preview。当前物理边界只有碰撞几何、过滤、重力、ground 和 freejoint debug object；RM65 actuator model 尚未完成，因此完整机械臂在重力下 `mj_step` 不能被表述为有效动态 rollout。kinematic direct-qpos replay 也只能作为相机/几何诊断。

### 3.7 目录迁移状态

当前磁盘上的主要迁移是：

| 旧路径 | 当前路径/状态 |
| --- | --- |
| `dataset_postprocess/` | `utils/dataset_postprocess/`，旧路径删除 |
| `utils/wrist_cam_fix_mask/` | `utils/wrist_cam_fixed_mask/`，命名和输出边界整理 |
| `sim/mujoco_real2sim/` | `sim/mujoco/` + `sim/contracts/` |
| `portable_init_UV/` | 封存到 `legacy/` |
| `teleop_runtime.action` 的通用领域逻辑 | 下沉到 `robot_motion.action`，旧导入保留兼容 |

Git 当前以 deleted/untracked 显示这些变化，合并前应显式 stage 后检查 rename detection、兼容 import 和被忽略产物，不能只依据 `git status` 的删除数判断。

## 4. 当前高优先级问题

### 已处理的 P0：DMTAC impedance 参数权威性

已确认当前左右 JSON profile 是权威参数源。原测试覆盖的字段已按 profile 对齐：线性/角柔顺系数为 `0.0012` / `0.02`，最大偏移为 `0.03 m` / `0.12 rad`，最大回中速度为 `0.06 m/s` / `0.3 rad/s`，`return_rate_s=4.0`；派生饱和阈值为 `25.45 N` / `6.04 Nm`，40 Hz 下从最大偏移开始的首步结果为 `0.0285 m` / `0.1125 rad`。

本次只调整已有测试覆盖，不把未执行的验证写成已完成。`lowpass_alpha=0.93`、最大导纳速度、`max_command_joint_step_deg=40`、direct joint hint `3 mm / 0.5°`、force guard 状态及其组合真机效果仍为**未测试/待受控真机验收**。profile 成为配置权威不代表这些行为已经获得实验数据证明。

### 已处理的 P1：恢复无硬件绿色测试基线

当前默认无硬件集合为 1,464 passed、34 skipped、0 failed。2 个 preflight 测试已对齐 angle trigger fallback、独立 impedance 外环和 arm-step safety floor；GUI 测试已区分输入 preview target 与实际 send target；6 个真实 UDP 用例已加 integration 标记和 UDP capability fixture。仍可继续细化其他测试的 marker，但不再阻断当前无硬件绿色基线。

### P1：完成位控链尚未认证的连续段

当前 75 Hz endpoint 通过不等价于 75→150 插值段全程无碰撞。至少需要：

1. 明确 continuous collision 或足够保守的 swept/intermediate check 策略；
2. 对 12 对场景做真实安装几何验收和可视化留档；
3. 将工具/TCP/profile 绑定到验收记录；
4. 保持任一检查不可用时 fail-closed，而不是退回单臂或理论轨迹。

### P1：收敛入口和生命周期

当前行数：

| 文件 | 行数 |
| --- | ---: |
| `teleop_runtime/debug/run_vive_rm65_real_teleop.py` | 18,772 |
| `teleop_runtime/collection/run_collect.py` | 6,189 |
| `teleop_runtime/deploy/run_policy.py` | 4,465 |
| `teleop_runtime/replay/executor.py` | 2,531 |

下一步应优先抽出稳定的 config/runtime builder、结构化状态 snapshot 和生命周期协调器，同时保持当前 CLI/import 兼容。不要把安全策略迁移和大规模文件搬迁混在同一批。

## 5. 中优先级问题

### P2：环境和依赖仍不可复现

`pytest.ini` 已解决测试收集，但目标 Python 3.10、本地 Python 3.12、OpenPI/GPU、MuJoCo、PyArrow、硬件 SDK 的安装关系仍靠文档和机器状态。建议增加唯一的项目元数据与 extras/lock，至少区分 core、test、dataset、sim、openpi、hardware。

### P2：机器私有配置与版本化 profile 混杂

RealSense 配置仍有 canonical/mirror 两条路径；RM65 IP `latest.json` 及大量带时间戳 assignment 位于工作树；主 README 直接列出实验台 IP、serial 和 by-id。应将 schema/安全默认值/批准 profile 与机器私有 latest/inventory 分开，启动日志必须显示实际来源。

### P2：主 README 同时承担操作手册和架构说明

README 已超过 560 行，并包含设备权限、固定设备表、频率、标定、采集、回放等大量现场命令；根目录还存在多个 `*_TMP.md` 和计划文档。建议将当前规范按 setup/teleop/collection/replay/deploy/dataset/sim 拆入 `docs/`，根 README 只保留导航和最小安全入口；临时文档标注日期、状态和替代文档。

### P2：运行产物与源码边界

仓库内仍可看到 `__pycache__`、日志、数据集和调试输出；部分大二进制/manifest 删除已进入 index。即使被 `.gitignore` 排除，它们也会影响同步、索引和误提交。建议默认 dataset/log/output root 放在仓库外，并在提交前检查 staged binary 与 generated output。

### P2：控制代码异常路径专项审计

对 `assert`、`except Exception`、best-effort close/hold 继续按行为审计：运动前置条件不得依赖可被 `python -O` 移除的 assert；宽泛捕获必须明确后续是继续发令、hold、fault、重连还是退出。优先覆盖 RM65 backend、impedance、position executor、replay/deploy 和真机入口。

## 6. 推荐整改顺序

### 阶段 A：冻结并确认控制语义

1. 保持 DMTAC impedance 当前左右 profile 与已覆盖的测试断言一致；未测试参数不得标成已验收；
2. 保持 12 对 collision、奇异慢速复核、fresh measured qpos 和 group hold 契约测试；
3. 设计并验证 75→150 连续碰撞策略及真实几何验收；
4. 保持 `v22_native` 在 native safety/backend 完成前 fail-closed。

### 阶段 B：建立绿色、分层的自动化基线

1. 保持 preflight、GUI 和 DMTAC 当前契约测试稳定；
2. 在允许 loopback UDP 的集成环境另行运行 6 项 VIVE integration 用例；
3. 用 marker 固定默认 unit/contract 集，独立运行 dataset/sim/openpi/hardware；
4. 增加 Python 3.10/3.12 与依赖 extras/lock/CI 声明。

### 阶段 C：按架构边界拆分提交

建议至少按以下主题独立 stage/提交并各自带测试证据：

1. dataset contracts/interface 与 v2.2/v2.3；
2. robot_motion action/runtime/position control；
3. sim/mujoco 与 contracts 迁移；
4. teleop collection/replay/deploy position 链；
5. VIVE clock/velocity/intent；
6. angle trigger/foot pad/device preflight；
7. real_teleop 组件抽离；
8. utils/legacy 迁移和 generated outputs 清理；
9. 文档、环境和测试配置。

## 7. 本轮未验证项

- 未验证真实 RM65 SDK、CANFD 150 Hz、impedance 40 Hz、双臂网络延迟或 fault/hold 的物理效果；
- 未验证 VIVE sender/receiver 的真实 UDP、clock offset/RTT 稳定性和 300 Hz 丢包；
- 未验证 RealSense、Q6、angle trigger、foot pad、DMTAC 或 InspireTac 的枚举与映射；
- 未在目标 `dexteleop310`、OpenPI/GPU 或完整 MuJoCo 环境执行可选测试；
- 未使用真实数据集验证 v2.1→v2.2 copy-on-write、v2.3 frame transform、训练增强或 replay/deploy 端到端数值一致性；
- 未执行任何 L4/L5 设备状态修改、homing、trajectory、replay、servo、gripper 或 force-control 操作。

本报告的用途是标明当前代码事实、尚未闭合的契约和下一步顺序；真机操作仍需独立的目标环境检查、现场安全确认和受控验收记录。
