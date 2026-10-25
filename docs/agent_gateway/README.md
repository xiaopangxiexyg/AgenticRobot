# External Agent Gateway

`agent_gateway` 是 DexTeleop 提供给上层 Agent/Skill 组织框架的稳定下层边界。它公开
版本化的 Skill、Route、observation、Artifact、operation 和 reconciliation 合同；具体
profile 负责装配 MuJoCo 或真机下层 owner。它不启动上层 Planner，也不把安全、IK、HOLD
证明或 actuator dispatch 交给远端调用方。

## 有界抓取搜索与观测诊断

- fast 抓取同时参考中央和有观测支持的较弱侧插入，沿用有限的局部轴向偏移、来源上限和
  完整路径检查。原中央浅源与新增弱侧源交错，避免新来源挤掉原有覆盖。父路径失败不证明
  轴向子路径失败；子候选仍须独立检查，不能继承父路径证明。
- 首次安全路径之后，接触更好的近邻检查最多两次，共享原 3 秒子预算及全请求截止时间。
  它不是无界优化；未知端点只跳过可选改善，不否定已验证路径。返回的 `contact_evidence`
  是局部观测评分，不是触觉反馈、力闭合或抓取成功判定。
- 文本定位返回 `semantic_identity_status=unverified_proposal`；标签和分数不能当身份真值。
  同图不同查询的重叠框不能作为独立参照。腕部相机随手运动，二维框变化不能单独证明世界
  坐标下的物体抬升；应使用对应 RGB-D、采集时 qpos 与既有标定转换，或如实报告未知。

可选改善沿用原绝对截止时间、发布保留量与pending-plan owner；最终场景读取后
仍须复查时间，过期候选不能替换原有效计划。该增量不改变当前夹爪几何默认，
也不代表已完成真机抓取验收。

本页是 Agent Gateway 的正式运行入口。协议与 package 所有权细节见
[`agent_gateway/README.md`](../../agent_gateway/README.md)；局域网开发者的 SSH 只读接入见
[`SSH_READ_ONLY.md`](./SSH_READ_ONLY.md)。
SAM2 + Grounding DINO 图片定位 Skill 的源码、权重和测试路径说明见
[`SAM2_GROUNDING_DINO.md`](./SAM2_GROUNDING_DINO.md)。
SAM2 + VGN/GraspNet 点击引导 DMTAC M 抓姿 Skill 的 RGB-D、固定相机标定、权重和只读边界见
[`POINT_GRASP.md`](./POINT_GRASP.md)。
MuJoCo 留存观测点移动和可选观测障碍规划的调用合同、启用方式与边界见
[`observed_point_motion.zh-CN.md`](./observed_point_motion.zh-CN.md)。
运行资产包的下载、解压和共享根登记统一见项目根 [`README.md`](../../README.md) 的“运行资产”章节；
本目录不重复资产包说明。

## 边界和运行等级

| 用途 | 接入方式 | 能力上限 | 等级 |
|---|---|---|---|
| 本机上层框架 | 私有 Unix socket | 由已审查 profile、client binding 和每次 Gateway admission 决定 | 取决于所调用 Skill；MuJoCo 为 R1，真机运动为 M |
| 远端开发者 | OpenSSH forced command | 只读 Skill、限定状态/发现、本人 operation 查询/取消和本人 Artifact 下载 | H1；若下层是 MuJoCo 则为 R1 |
| CLI/合同检查 | `--help`、离线测试 | 不启动 backend | R0 |

远端 SSH 接入不是第二套硬件协议，也不产生新的设备 lease。驱动类 Skill、通用提交、
lifecycle、HOLD、Artifact 上传和 materialization acknowledgement 对远端身份始终不可见；
这不会降低本机可信 Skill 调用方原有的权限。

## 环境前缀

先按 [`docs/README.md`](../README.md#命令约定) 激活 DexTeleop 环境，然后从仓库根目录运行：

```bash
export DEXTELEOP_ROOT="$PWD"
```

## 本机 Unix endpoint

为每次 Gateway session 分配一个独立、长度受控的 endpoint：

```bash
export DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint
)"
test -n "${DEXTELEOP_GATEWAY_SOCKET}"
```

分配命令只创建 mode `0700` 的 session 目录并返回尚不存在的 `g.sock` 路径，不会启动
仿真、设备或 Gateway。把该路径原样交给下层 launcher 和同机可信上层；不要把 history、
Artifact、Experience 或日志根目录放进 socket 路径，也不要自行迁移到 `/tmp`。

### MuJoCo profile（R1）

```bash
python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config "${DEXTELEOP_ROOT}/sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json" \
  --client-id rpent.dexteleop \
  --client-uid "$(id -u)" \
  --history-root "${DEXTELEOP_ROOT}/temp/runtime/agent_gateway/history"
```

这条命令只启动 DexTeleop 自有的 MuJoCo application、下层 motion owner 和 Gateway，
不会启动 RPent。需要 RGB/depth Skill 时额外提供一个私有、非 world-writable 的
`--artifact-root`；不提供时 Artifact-bearing perception Route 保持关闭。完整 MuJoCo
上下文和 safety-reference 语义见 [`docs/sim/README.md`](../sim/README.md#4-agent-tcp-pose-gatewayr1mujoco-only)。

### 真机 profile（M）

```bash
python -m agent_gateway real-rm65 --help
python -m agent_gateway real-perception --help
```

`real-rm65` 会装配独占的双 RM65 下层 owner，并要求 profile、设备、safety envelope 和
一次性运动确认。即使上层只打算读取感知，当前组合 launcher 也会启动相机和机械臂 owner；
不要把它当作 camera-only diagnostic。正式参数、启动门禁和关闭顺序以
[`agent_gateway/README.md`](../../agent_gateway/README.md#entrypoints) 为准，运行前仍需明确
真机授权和现场安全检查。

绝对抓姿使用现有 `motion.move_tcp_to_pose@1`，但 session 必须由操作者显式加入
`--authorized-action move_tcp_to_pose`；仅在 Route manifest 中看到它不代表已经取得运动
Grant。它的公共目标始终是 `canonical_station` 下的实际 DMTAC `tool_tcp`，本地一次请求
客户端提供 `--target-xyz-m`、`--target-quat-xyzw` 和可选 `--speed-scale`。客户端从同一份
实时 observation/schema 绑定 frame graph、Station frame、tool config 和 calibration hash，
不允许调用方用字符串把 arm-base 或 camera pose 冒充 Station pose。

一次请求客户端允许在 `RUNNING/READY` 或 `HOLD/HOLD` 状态提出新的独立请求，不再要求
普通 TCP 运动的 session 同时暴露夹爪 Route。`HOLD/HOLD` 不是客户端自行恢复控制的许可：
原有下层 owner 仍检查此前 HOLD 的成功回执、fresh measurement、新请求 admission 与 Grant，
再决定是否恢复。FAULT、混合／异常 execution gate 和已关闭的 owner 仍拒绝；失败 HOLD
不会被当作正常收尾。这使正常到达并 HOLD 后可以继续下一个 Skill，不削弱原有保护。

需要从当前姿态做较远但仍受原有 envelope 约束的诊断移动时，可用
`--relative-delta-xyz-m DX DY DZ` 替代绝对目标或相机 proposal。该偏移沿
`canonical_station` 的 XYZ 轴解释；客户端只从同一份提交所绑定的 fresh observation
snapshot 取一次实测 TCP，增加平移并保持实测姿态，随后提交原有绝对
`move_tcp_to_pose` 参数。它不新增 Route 或放宽下层 10 cm 相对范围、workspace、IK、碰撞、
dispatch、收敛与终态 HOLD 门禁。首次真机远移建议单轴 5--8 cm、
`--speed-scale 0.15`--`0.20`，并同时启用下述 motion trace。

若抓姿生成器使用同一 Gateway session 内 Station-aware Depth v2 的输出，可把直接数值参数
替换为 `--camera-grasp-proposal /absolute/path/grasp-proposal.json`。该文件使用
`dexteleop.real.camera_relative_grasp_proposal.v1`，必须逐项绑定当前 runtime session、Route
manifest、源 operation、Depth v2 Route/deployment、左右臂、Station revision、Station frame
spec、frame graph、相机在 Station 下的位姿，以及带真实采集 receipt 的 typed metric-depth
Artifact；可选 `capture_group_id` 只保留跨模态采集相关性，不参与控制授权。客户端只在取得
当前 RUNNING/READY snapshot 后执行：

```text
T_station_actual_tool_tcp
  = T_station_camera_optical @ T_camera_optical_actual_tool_tcp
```

进程内抓姿生成器应使用
`CameraRelativeGraspProposal.from_depth_operation_wire(...)` 从已完成的 Depth operation
构造，再序列化 `to_mapping()`，避免人工抄写 Route、operation、result 和 Artifact 身份。

session、Station、frame graph、相机/深度部署或机械臂任一身份不一致时，都会在提交运动请求
前关闭失败。proposal 不是 Grant，不会绕过一次性授权，也不会作为 motion Artifact 进入
下层；解析后的结果是所请求 `move_tcp_to_pose` 或 `approach_grasp_pose` Route 的 Station TCP
参数。抓姿执行应选后者，由下层在激活前完成 URDF/FCL、TCP envelope、非活动臂固定和整条
qpos 路径复核；Active 阶段只按控制时钟插值已复核的 qpos，不再动态重算 Cartesian IK，并在
规划时长结束、实测收敛且终态 HOLD 确认后才成功。激活和执行回执只保留固定轨迹身份、最新
实测/下层状态与终态 HOLD，不随轮询或控制帧数无限累积。直接 `--target-*` 模式保持不变，且
与 `--camera-grasp-proposal` 互斥。

MuJoCo 任意抓姿可视化的显式输入则是所选机械臂 base 下的实际 DMTAC TCP，二者数值不能
直接复用。相同物理目标应通过选定 collision-scene URDF/profile 的固定链转换：

```text
T_station_tcp = inverse(T_solver_world_station)
              @ T_solver_world_arm_base(side)
              @ T_arm_base_tcp
```

若抓姿来自相机，必须使用同一 Station revision 的
`T_canonical_station_camera_optical_frame` 先转换到 `canonical_station`；缺少或不匹配的
revision/frame graph/calibration 应关闭失败。DMTAC TCP 到 solver endpoint 的工具几何转换
仍只在下层 target owner 执行一次，Gateway/上层不得硬编码约 0.13–0.14 m 的偏置。
这些说明和离线测试不构成真机运行授权；任何真实调用仍需下层 envelope、IK、碰撞/跳变/
奇异性门禁、dispatch admission、收敛检查和终态 HOLD 证明。

#### 真机 Skill 运动 trace

需要定位抖动、超时或发送降频时，可在启动 `real-rm65` 时显式增加：

```bash
--motion-trace-root /absolute/private/motion_traces \
--motion-trace-max-duration-sec 120 \
--debug-motion-skill
```

trace 不改变目标、IK、admission、freshness、碰撞、轨迹或 HOLD。启动前一次性为 60 Hz
reference/IK 和左右各 120 Hz actuator command 分配固定内存；运行期间只复制固定字段，缓冲区
满时累计 drop 而不等待，不执行 JSON、FK 或磁盘写入。下层停止并解除 trace callback 后才生成：

`--debug-motion-skill` 默认关闭，并且缺少 `--motion-trace-root` 时拒绝启动。开启后只让既有
IK solver 将每帧 `solve/QP/integrate/set-configuration/min-distance` 分段耗时写入同一份
预分配 trace；`debug_profile_print_every=0`，因此不会在控制热路径逐帧打印。关闭该参数即可
回退到原有 target-builder 配置和 trace 字段缺省行为。

- `motion_trace.npz`：30 Hz 实测 qpos/TCP、60 Hz 请求目标/原始 IK/最终 qref/分支/碰撞余量/
  各阶段耗时，以及左右 120 Hz 的实际 SDK qcmd、发送时刻与结果；
- `summary.json`：branch switch、reference/qcmd/measured 的 step/velocity/acceleration/jerk、
  send gap、SDK latency、deadline、pose error 和 trace drop/error 汇总；
- `manifest.json`：schema、dtype、时钟、容量、运行身份和完整性状态。

进程退出后从 `gateway_stopped.motion_trace.output_dir` 取得实际目录，再检查：

```bash
python -m teleop_runtime.skill_deploy.motion_trace inspect \
  /absolute/private/motion_traces/<session-directory>
```

`gateway_stopped.motion_trace.closed=true`、`manifest.json.result.closed=true` 且
`summary.json` 存在，才表示 requested trace 已完整落盘；trace 成功不等价于运动成功，运动仍以
operation outcome、实测收敛和 lower-confirmed HOLD 为准。

### 真机只读 perception profile（H1）

```bash
python -m agent_gateway real-perception --help
```

`real-perception` 不创建 RM65 backend，只装配 RealSense、Artifact store 和只读 Skill。默认仅公开
RGB/depth capture；提供经过验证的模型 bundle 并显式启用后，可增加
`perception.localize_objects`。再显式传入 `--enable-point-grasp` 及 VGN（默认）或 GraspNet bundle，
才会增加 `perception.plan_point_grasp@2`。该 v2 Route 的调用方只提供已注册 depth role、点击像素和
候选数上限；同一应用从单个已发布 `VideoFrame` 取得相关的 RGB 与对齐深度，并绑定相机 generation、
frame index、内参、静态标定、Station revision 和 frame graph。它只返回未选择、无执行权限的
canonical-Station TCP 候选，不提供 RM65 backend、Grant 或运动，也不把单帧可见点云检查描述为完整
场景/机械臂/路径安全证明。调用方提供 RGB-D 和标定的 v1 合同仍只用于离线 CLI。外部源码由
`model/perception/` 的固定 Git submodule 提供；checkpoint 不进入 Git。模型权重和 submodule 见
[`SAM2_GROUNDING_DINO.md`](./SAM2_GROUNDING_DINO.md) 与
[`POINT_GRASP.md`](./POINT_GRASP.md)。本地静态图片应使用独立离线 CLI；生产 localization 只接受
Artifact，live point-grasp 则只接受上述三个语义参数并由部署侧采集 RGB-D。

## 状态、占用与并发

Gateway 可返回两个小型只读投影：

- `get_hardware_resource_views`：profile 绑定的稳定资源、`readable`/`drivable` 属性和
  语义 Skill 引用；
- `get_hardware_status_views`：`free`/`occupied`、独立 health，以及可选的 active
  operation 投影。

`occupied` 表示已有独占下层控制/资源 claim，或存在活跃独占 operation；它不等价于
“机械臂正在运动”。READY、idle 和 lower-confirmed HOLD 都不会自动释放 claim。读取状态
失败时按 unavailable + occupied 关闭失败路径。远端调用方不会排队、抢占或取得另一份
lease；资源被其他用户或 Skill 占用时，返回正常的下层 admission/status 结果。

## 关闭与故障处理

- 正常关闭 launcher，让 Gateway 先 fence 新请求，再由下层 owner 完成自己的 HOLD 和
  resource close；上层不得根据连接关闭自行声明 HOLD。
- transport 超时或连接中断不证明请求未被接收。可能已接收的提交进入既有
  reconciliation 流程，不能当作普通 rejection 重新提交。
- event replay 是提示而非真值；出现 gap 后通过 authoritative operation/discovery query
  重建状态。
- endpoint 只属于一个 session。不要复用旧 socket、猜测 socket 路径或删除仍在使用的
  session 目录。

## 离线验收（R0）

```bash
python -m agent_gateway --help
python -m agent_gateway allocate-unix-endpoint --help
python -m agent_gateway mujoco --help
python -m agent_gateway real-rm65 --help
python -m agent_gateway ssh-readonly-proxy --help

PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_agent_gateway*.py
```

这些检查只证明 CLI、合同、transport 和装配的离线行为；不能替代 MuJoCo、SSH 主机或
真机运行证据。


### MuJoCo 安全目标与初始几何复核

安全 TCP 的公开单目标路线先检查目标容差区域是否可能与原稳定包络相交；不将投影值
替换用户目标。候选、路径、实测输出和 HOLD 门禁仍分别执行。关节限位附近的普通迭代
全部未命中时，只用原 2 秒端点预算的剩余时间查询有界 IK；碰撞或路径质量拒绝不会
触发该查询。已有抓取链仍沿原分段绑定及 prepared resolver 执行。

[qualify_mujoco_scene.py](../../scripts/probes/qualify_mujoco_scene.py) 提供独立的零运动
初始几何资格检查。用 `python -m scripts.probes.qualify_mujoco_scene --help` 查看参数；
输出写入新文件，不覆盖已有报告。资格成立只说明该初始几何通过检查，不代表规划、
执行或抓取成功，也不启动 RGB-D、模型服务或真机。失败启动后的关闭会继续交给原
runtime 清理，并保留其未完成释放供同一 owner 重试。
