# AgenticRobot 抓姿生成与 RM65 Skill 全链路交接

> 文档日期：2026-09-08（Asia/Shanghai）  
> 文档目的：把抓姿生成、upward TCP 筛选、MuJoCo 可视化、左右臂、真机 Gateway、运动抖动修复、trace 证据、资产和验收边界集中记录，供后续开发者和远端 `user` 接手。  
> 当前文档属于独立 Git 候选分支；它不代表已经合入 `development`，也不代表真机验收已完成。

## 1. 先看结论

本次任务不是单独增加一个抓姿角度选项，而是建立一条可审计的链路：

```text
RGB-D/点击
  -> SAM2 + VGN/GraspNet
  -> DMTAC M tool_tcp 候选
  -> 平行夹爪 0/180° 等价姿态中选择 reference +Z 朝上的一个
  -> 每个 camera × candidate 独立线框渲染
  -> MuJoCo 中验证坐标、视角、左/右臂和状态不变性
  -> （经过独立运动授权后）Gateway snapshot-relative TCP move
  -> lower-owned IK/碰撞/dispatch/收敛/HOLD
  -> motion trace 诊断采样 cadence、SDK stall 和发送 gap
```

当前最重要的事实：

| 项目 | 当前状态 |
| --- | --- |
| 当前运动候选 worktree | `/home/zc/HandMotion/Developing/temp/worktrees/root-grasp-motion-trace-20260907` |
| 分支 | `task/root/grasp-motion-trace-20260907` |
| 运动候选固定提交 | `39ce36d352bf45bbae33cc1ea42c929dc654b3ae` |
| 候选固定基线 | `63e82b274eb548aef7fd6a9b114b2305bc9f7e36` |
| `development` 当时观测值 | `4743d20a97fa07acc47d4217c39be4040c22b62e` |
| 当前候选状态 | `ready-for-integration`（写入本交接文档后必须重新标记） |
| 真机动作 | 本轮没有由 Codex 执行；必须由远端 `user` 按授权命令执行 |
| 真机 trace 新 schema | `dexteleop.exclusive_rm65_skill_motion_trace.v2` |
| 远端主机 | `user@192.168.70.116`，工作根 `/home/user/zc` |
| 远端接入方式 | 直接在 `user` 本地 shell 运行；不使用 `codex_remote/remote_module.sh` |

已有离线/仿真实现包括：

- 平行夹爪等价腕部姿态的 upward 筛选，已同时用于 VGN 和 GraspNet。
- `front`、`left_wrist`、`right_wrist` 三视角投影；每个图像只绘制一个抓姿。
- MuJoCo `--arm-side left|right`，左臂严格复用右臂的 base/TCP/IK/接触/夹爪语义。
- Gateway 的 debug target-coordinate 叠加和 snapshot-relative TCP 移动输入。
- 30 Hz 硬件测量、60 Hz reference/IK、120 Hz CANFD command 路径的非阻塞改造和 v2 trace。

尚未完成的事情：

- `39ce36d3…` 尚未自动合入 `development`。
- 新修复后的真实 5--8 cm 远移尚未在远端硬件上执行，不能把它描述为真机通过。
- 真机相机持续接入、Artifact/存储策略、上层 Agent 消费和摄像头闭环不在本次交接范围。

## 2. 需求演进与边界

### 2.1 抓姿生成的原始问题

原始生成器只约束夹爪接近/开口几何。平行夹爪绕其接近轴旋转 180° 后，两个手指只是互换，位置和夹取方向等价，但 TCP 的“上方”轴可能朝下，导致：

- 同一个物理抓姿出现两个 TCP 姿态；
- 生成结果在图像中方向不稳定；
- 下游 IK 可能选择一个视觉上倒置、腕部不合理的分支；
- 仅检查夹爪宽度无法保证“手腕朝上”。

需求因此明确为：在候选生成阶段消除该对称性，只保留 reference frame 重力方向上更合理的一个 TCP 姿态；不能把这个偏好留给上层 Agent，也不能用经验性的 Z 补偿代替坐标合同。

### 2.2 可视化需求

用户后续明确了以下显示规则：

1. 需要把抓姿叠加到 RGB 图像，帮助后续调用 Skill 的模型/人理解 TCP、接近轴、开口轴和高度方向。
2. 同一抓姿要能投影到任意已知 camera，Sim 暂定 `front + 左腕 + 右腕`，real 侧沿用同一 `T_reference_from_tcp` / `T_reference_from_camera` 变换。
3. 腕部视角必须朝向方块，否则无法验证；但这种“朝向方块”只允许作为 MuJoCo 可视化 free-camera override，不能伪装成真实腕部相机标定。
4. 多个抓姿不得混画在一张检查图里：每个 `camera × candidate` artifact 只包含一个候选；montage 只展示 provider 选中的一个，其他候选保留为独立文件。
5. 可视化是几何叠加，不执行 IK、夹爪、机械臂移动，也不作深度遮挡或完整路径无碰撞证明。

### 2.3 左臂支持

左臂不能是复制一份只改字符串的旁路。必须与右臂使用同一：

- DMTAC M TCP 语义和工具 profile 校验；
- arm-base、camera、frame graph 和 `T_base_from_tcp` 转换；
- IK、硬限位、碰撞、轨迹、收敛和终态 HOLD；
- 夹爪接触体、执行器和状态证据；
- Gateway admission、`authorized-side` 和 component identity。

Sim 中通过 `--arm-side left|right` 选择受控侧，另一侧保持镜像 home。`right` 只是兼容默认值，不能当作 Skill 只支持右臂的证据。

### 2.4 真机移动与抖动问题

真机相对移动的目标不是“移动到固定世界点”，而是：

1. 在同一个 fresh `get_observation_snapshot` 中读取所选侧当前实测 DMTAC `tool_tcp` 位姿；
2. 沿 Gateway 公布的 `canonical_station` XYZ 轴加上有限的平移增量；
3. 保留该 snapshot 中实测的姿态四元数；
4. 将所得绝对 TCP 目标提交给既有 `move_tcp_to_pose` Route；
5. 继续经过 envelope、IK、碰撞/跳变/奇异性、dispatch、收敛和终态 HOLD。

它不是新增一个绕过安全检查的 primitive，也不是把 TCP 移到 arm-base、相机或任意用户自定义 frame。

真实运动中观察到 CANFD 降频、IK 超预算和小幅抖动后，诊断目标变为：硬件状态读取、FK、IK 和 120 Hz 发送互不以无界工作相互阻塞；慢 SDK 调用后不能追赶式连发；所有结论要由固定 trace 证据支持。

## 3. 当前代码/提交地图

以下是交接时需要重新核对的关键提交。提交 SHA 是事实 pin；集成前不能只引用分支名。

| SHA | 作用 | 交接说明 |
| --- | --- | --- |
| `dbd84dc09cdd83faa547504447022e9274049ea5` | `fix(perception): select upward grasp TCP symmetry` | VGN/GraspNet 使用统一 upward 筛选；包含 Sim/单元测试 |
| `ed3eb66b44d08fb01be7b795a8e0f0f8f48ac248` | multiview grasp overlay | 任意 reference/camera pose 的几何投影和状态分类 |
| `18c16a08b596ddeffea6839796368f299d8adf50` | focused single-view rendering | 每个候选/视角独立渲染，减少线框混合 |
| `8f0e8f372430b07349d4694fb06cfed82b05bb79` | Sim left-arm point grasps | `--arm-side left|right` 和左侧 CAD/TCP 合同 |
| `b426d08c0ffaffb4878b376c672e362f85c0a278` | Gateway debug target coordinates | real-rm65 叠加 target point/Station XYZ 调试线框 |
| `39970d524f1bfae126c86f528a371ffb4e9f29f0` | snapshot-relative TCP input | one-shot 从 fresh measured snapshot 计算相对目标 |
| `1484ff06`（已合入当前基线） | bounded motion trace v1 | 真机 trace 基础 schema；旧远端证据仍是 v1 |
| `ae3aeb8d90b1570040116beda6c671337531e987` | snapshot freshness gates | 保留 lower owner、source sequence 和 freshness/coherence 门禁 |
| `c40722a4b920c0955debbb9c6c9571c5d7de4591` | qpos sampling off FK path | authoritative hardware qpos 先过 freshness，FK 后置 |
| `71d1bb6bc500f29362914cec5145b28848b1c803` | decouple measurement sampling | lower-owned 采样与 60 Hz loop 解耦 |
| `39ce36d352bf45bbae33cc1ea42c929dc654b3ae` | post-stall burst fix + trace v2 | 当前运动候选，包含 cadence、持久 worker、发送 gap 和 v2 诊断字段 |

当前候选的主要文件：

- [抓姿合同与权重说明](./POINT_GRASP.md)
- [`deployments/perception/sam2_vgn_point_grasp.py`](../../deployments/perception/sam2_vgn_point_grasp.py)：VGN 生成、upward 筛选和候选去重
- [`deployments/perception/sam2_graspnet_point_grasp.py`](../../deployments/perception/sam2_graspnet_point_grasp.py)：GraspNet 轴映射和 upward 筛选
- [`deployments/perception/dmtac_grasp_overlay.py`](../../deployments/perception/dmtac_grasp_overlay.py)：共享线框投影器
- [`experiments/sim_point_grasp/visualization.py`](../../experiments/sim_point_grasp/visualization.py)：MuJoCo 三视角、wrist look-at 和单候选 artifact
- [`experiments/sim_point_grasp/evaluation.py`](../../experiments/sim_point_grasp/evaluation.py)：左右臂实际执行评估
- [`agent_gateway/profiles/real/point_grasp_render_gateway.py`](../../agent_gateway/profiles/real/point_grasp_render_gateway.py)：Gateway 单候选渲染边界
- [`agent_gateway/profiles/real/one_shot_client.py`](../../agent_gateway/profiles/real/one_shot_client.py)：相对 TCP 目标解析和 one-shot admission
- [`teleop_runtime/skill_deploy/rm65_ports.py`](../../teleop_runtime/skill_deploy/rm65_ports.py)：lower measurement sampler、HOLD 和资源生命周期
- [`teleop_runtime/deploy/arm_state_reader.py`](../../teleop_runtime/deploy/arm_state_reader.py)：双臂并行硬件读取、qpos/FK 证据绑定
- [`teleop_runtime/backends/arms/rm65.py`](../../teleop_runtime/backends/arms/rm65.py)：120 Hz background stream 调度和 CANFD 发送审计
- [`teleop_runtime/skill_deploy/motion_trace.py`](../../teleop_runtime/skill_deploy/motion_trace.py)：v2 固定内存 trace、summary、manifest 和 inspect

## 4. 抓姿生成合同

### 4.1 输入、输出和坐标

离线 CLI 接受同一帧 RGB-D、点击像素和显式本地标定；生产 point-grasp Route 只接受已经注册的 aligned depth role 和点击像素，由 deployment 侧取得同一个 `VideoFrame` 的 RGB/depth。两者都不自动连接机械臂。

输出最多 5 个排序的 DMTAC M `tool_tcp` 候选，每个候选至少保留：

- `T_camera_from_tcp`：TCP 坐标点转换到相机 optical frame；
- `T_base_from_tcp`：TCP 坐标点转换到所选 arm-base/reference frame；
- `gripper_width_m`、模型分数、点击距离和可见点云碰撞近似结果；
- `selected_ordinal=0`（仅表示 provider 排名，不是运动 Grant）；
- backend identity、model/profile hash、frame graph、Station/calibration identity（生产 v2）。

矩阵是 row-major 4×4 齐次变换，平移单位米。相机 optical frame 约定为 `+x` 右、`+y` 下、`+z` 前；不能把图像像素、arm-base 或法兰位姿直接当作 `canonical_station`。

DMTAC M 适配保持以下已审查语义：最大开口约 0.070 m，实际 `tool_tcp` 相对法兰 z=0.14 m；VGN/GraspNet 的掌/指包络、接近轴和渲染均采用同一工具几何，不在上层追加约 0.13--0.14 m 经验偏置。

### 4.2 Upward 选择算法

平行夹爪的对称成员通过 TCP 局部接近轴 `+Z` 旋转 180°得到：

```text
R_flip = diag(-1, -1, +1)
T_candidate_flipped = T_candidate @ block_diag(R_flip, 1)
```

算法用 reference frame 的重力向上 `+Z` 评价 TCP 局部 `+Y`（工具高度/腕部向上轴）：

```text
score_up = reference_from_tcp[2, 1]
if score_up < 0:
    选择翻转后的姿态
else:
    保留原姿态
```

该选择同时更新 `camera_from_tcp` 和 `reference/base_from_tcp`，因此不会破坏两个 frame 之间的刚体关系。位置和局部 `+Z` 接近轴保持不变；水平 tie 保留模型原姿态。选择策略常量是 `maximize_reference_plus_z_dot_tool_tcp_plus_y`，必须随 planner semantic content hash 一起记录。

VGN 在排序后还按位置小于 15 mm 且旋转小于 15° 去重，避免对称或近重复候选占满 5 个名额。GraspNet 的对称选择在轴映射、插入深度和可见点云过滤之后执行。该 upward 偏好已经实际应用到 VGN、GraspNet 和 MuJoCo 抓姿实验，不是只在可视化层旋转线条。

### 4.3 安全边界

抓姿生成/渲染结果是 `ranked_unprivileged_motion_proposals_only`：

- 不创建 RM65 backend；
- 不执行 IK、路径规划、夹爪动作、轨迹 dispatch 或机械臂运动；
- 可见点云碰撞近似不是完整场景、连杆或路径无碰撞证明；
- provider 选择 `ordinal=0` 不会自动获得 `motion.move_tcp_to_pose@1` Grant；
- 真机采用候选时，仍必须重新绑定 Session/Station/frame/tool identity，并经过所有下层门禁。

## 5. MuJoCo 可视化和完整流程

### 5.1 只读三视角渲染（推荐先运行）

这是“看清楚抓姿”的入口，不做 motion：

```bash
cd /absolute/path/to/AgenticRobot
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m experiments.sim_point_grasp.commands.visualize_candidates \
  --arm-side left \
  --seed 0 \
  --max-candidates 5
```

可将 `--arm-side left` 换为 `right`；输出目录必须是不存在的目录，默认位于 `runtime_artifacts/runs/experiments/sim_point_grasp_visualization/<timestamp>/`。输出应包括：

- `front`、`left_wrist`、`right_wrist` 原始 RGB；
- `front_depth_m.npy`、provider `skill_result.json` 和选中候选 annotation；
- 每个 `camera × candidate` 的独立 `<camera>_candidate_<ordinal>_grasp_overlay.png`；
- 仅显示 ordinal 0 的 `grasp_multiview_montage.png`；
- `visualization_report.json`，schema 为 `dexteleop.experiments.sim_point_grasp_visualization.v2`。

单候选约束由渲染函数强制：每个 artifact 的输入必须恰好一个候选；如果把 5 个候选传给单个 overlay，程序应拒绝而不是生成难以辨认的混合图。

### 5.2 腕部视角语义

- `front` 使用场景固定头部相机。
- `left_wrist`/`right_wrist` 保留模拟腕部相机的原点，但在本验证命令中用 MuJoCo free camera 朝向随机方块中心。
- 报告必须写明 `visualization_only_look_at_override_from_wrist_origin`。
- 投影外参从实际 GL camera 反算，保证线框和当前 RGB 视图一致。
- 该 look-at 不修改 URDF、camera registry 或生产腕部外参，也不证明真机腕部相机已标定。

生产 `real-rm65` 组合中，腕部 pose 只能由同一 lower owner 发布的新鲜双臂 measured qpos/coupled FK cut 提供；camera-only profile 目前只广告具有可信静态外参的 `main`。任何调用方不能靠提交字符串 `wrist_left`/`wrist_right` 越权启用角色。

### 5.3 物理抓取 Sim 全流程

需要验证 upward 姿态能否进入真实运动链时运行：

```bash
cd /absolute/path/to/AgenticRobot
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --arm-side left \
  --trials 10 \
  --seed-start 0
```

该入口会调用正式 provider、对候选做 IK/完整 MuJoCo 场景碰撞预检、选择 7/5/3.5 cm 预抓取偏移、闭合 DMTAC M 并尝试抬升；它不连接真实摄像头或机械臂。严格成功条件是方块上升至少 8 cm、脱离桌面、双指接触保持 20 个 physics ticks；“有候选”或“完成 IK”不等于抓取成功。

### 5.4 MuJoCo 结果如何呈现

交接报告至少同时展示：

1. 一张三视角 selected-only montage，用于快速判断同一候选在 front/left_wrist/right_wrist 的朝向是否一致；
2. 至少一组按 `camera × ordinal` 分开的原图，用于判断候选 1--4 是否被遮挡、出画面或方向不合理；
3. `visualization_report.json` 中的 `T_base_from_camera`、`T_camera_from_tcp`、visibility 分类和 `simulation_state_unchanged`；
4. 若运行物理流程，再附 `summary.json`、逐 trial video/montage 和失败阶段统计。

图例中的 `visible`、`partially_visible`、`behind_camera`、`out_of_frame` 必须保留。几何叠加没有做深度遮挡测试，不能使用“线条可见”声称夹爪在真实物体前方或一定不碰撞。

## 6. 左右臂对照表

| 层 | 右臂 | 左臂 | 验证要点 |
| --- | --- | --- | --- |
| Sim 参数 | `--arm-side right` | `--arm-side left` | 另一侧镜像 home；不能仍写死 right base |
| DMTAC TCP profile | `hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json` | `.../rm65_left_DMTAC_M.json` | TCP、工具 hash 和 side identity 一致 |
| 实验 tool contract | `rm65_right_DMTAC_M_4b2_cad_tool_tcp.json` | `rm65_left_DMTAC_M_4b2_cad_tool_tcp.json` | CAD 接近/张合轴和约 0.14 m 深度保持对应侧 |
| Sim camera/view | `right_wrist` | `left_wrist` | wrist view 只可使用 Sim look-at override |
| Gateway | `--authorized-side right` | `--authorized-side left` | component 必须分别为 `right_arm`/`left_arm` |
| relative move | `--side right` | `--side left` | snapshot 只能读取所选侧，姿态从同一 snapshot 保留 |
| envelope | `agent_primitive_right_relative_tight_v1.json` | `agent_primitive_left_relative_tight_v1.json` | 不能用另一侧 profile 冒充 |

## 7. 真机 Gateway 语义与命令

### 7.1 先确认命令是“授权执行”

以下命令只应由远端 `user@192.168.70.116` 在现场安全检查完成后执行。Codex 本轮不执行、不代替操作者确认机械臂空间，也不把命令输出当作验收证据。

Gateway 启动时必须：

- 使用独立 Unix socket，父目录私有且路径短于 Linux AF_UNIX 限制；
- 绑定左右 IP、左右 DMTAC M tool profile、选定侧 envelope；
- 明确 `--authorized-side left|right` 和 `--authorized-action move_tcp_to_pose`；
- 用 `AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES` 显式确认持久真机 session；
- startup HOLD、双臂资源、position force guard、fresh measured cut 全部通过后才发布 socket；
- 启用 trace 时提供非 world-writable 的私有 trace root。

### 7.2 远端 user 的启动模板（左臂、带 v2 trace）

远端仓库建议使用已同步的 tracked-only 目录：
`/home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked`。以下是运行模板，路径需要在远端实际核对；它不会自动下载模型，也不会自动修复资产链接。

```bash
cd /home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310

export AR_RUN_ROOT=/home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked
export AR_TRACE_ROOT="$AR_RUN_ROOT/runtime_private/motion_traces"
umask 077
mkdir -p "$AR_TRACE_ROOT" /home/user/zc/ag
chmod 700 "$AR_RUN_ROOT/runtime_private" "$AR_TRACE_ROOT" /home/user/zc/ag

export AR_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint --root /home/user/zc/ag
)"
printf '%s\n' "$AR_GATEWAY_SOCKET" \
  > "$AR_RUN_ROOT/runtime_private/current_gateway_socket.txt"
printf 'Gateway socket: %s\n' "$AR_GATEWAY_SOCKET"

python -m agent_gateway real-rm65 \
  --socket-path "$AR_GATEWAY_SOCKET" \
  --arm-ip-assignment "$AR_RUN_ROOT/teleop_runtime/configs/hardware/rm65_arm_ips/latest.json" \
  --left-tool-profile "$AR_RUN_ROOT/hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json" \
  --right-tool-profile "$AR_RUN_ROOT/hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json" \
  --envelope-profile "$AR_RUN_ROOT/teleop_runtime/configs/skill_deploy/agent_primitive_left_relative_tight_v1.json" \
  --authorized-side left \
  --authorized-action move_tcp_to_pose \
  --client-id dexteleop-local-primitive-operator \
  --client-uid "$(id -u)" \
  --motion-trace-root "$AR_TRACE_ROOT" \
  --motion-trace-max-duration-sec 120 \
  --confirm AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES
```

如果现场要切换右臂，必须成对切换 `--envelope-profile`、`--authorized-side` 和 one-shot `--side`；不能只改 one-shot 参数。

### 7.3 左臂 6 cm 相对移动模板

该命令表示：以同一 fresh snapshot 中左臂当前实际 DMTAC TCP 为基准，沿 `canonical_station +Z` 增加 0.06 m，姿态保持 snapshot 中实测姿态。它不是“移动到 z=0.06 m”，也不是相对于相机/arm-base。

```bash
cd /home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
export AR_GATEWAY_SOCKET="$(cat runtime_private/current_gateway_socket.txt)"

python -m agent_gateway.profiles.real.one_shot_client \
  --socket-path "$AR_GATEWAY_SOCKET" \
  --client-id dexteleop-local-primitive-operator \
  --side left \
  --action move_tcp_to_pose \
  --relative-delta-xyz-m 0 0 0.06 \
  --speed-scale 0.15 \
  --operation-timeout-sec 30 \
  --confirm EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE
```

建议先做单轴 5--8 cm、`speed-scale 0.15--0.20` 的受控验收；执行前现场必须确认急停、桌面/方块/线缆和另一臂均安全。若命令超时或 transport 断开，不能直接重复提交，因为请求可能已经被 lower owner 接收；应先查询 operation/reconciliation。

## 8. 运动并发与抖动修复要点

### 8.1 旧问题

旧采样器用“本次读取完成时间 + period”计算下一次 deadline，实际周期变成：

```text
硬件读取耗时 + 33.3 ms
```

远端旧 trace 约 20.2 Hz，而配置目标是 30 Hz。双臂刷新还会反复创建/销毁 `ThreadPoolExecutor`；慢的 FK/诊断工作可能老化 measurement cut；CANFD SDK 调用跨过 deadline 后又立即追赶发送，形成“长空档 + 亚毫秒连发”。

### 8.2 当前修复

1. **相位保持的 30 Hz measurement sampler**：以固定相位推进 deadline，错过周期时跳过而不 burst；SDK 读取慢不会在下一次立即连发。
2. **持久双 worker executor**：background measurement 模式创建一个双 worker executor，左右硬件读取并行；sampler 停止后再 shutdown。启动 rollback、重复 close 和异常关闭均保留所有权。
3. **qpos 先于 FK**：先取得 authoritative hardware qpos、source sequence、measured/available time，先完成 freshness/coherence；FK、姿态派生和诊断在 measurement cut 发布后进行，不能阻塞硬件 cut。
4. **CANFD no-catch-up**：exclusive real RM65 profile 设置 `background_streaming_no_catch_up_after_send=True`。SDK 跨 deadline、出现 skipped tick 或 send gap 至少 1.5 个周期后，下一次至少等完整周期，阻断 post-stall burst。
5. **诊断观察者隔离**：trace/position audit 不执行 FK、JSON、磁盘 I/O 或无界 queue；固定 NumPy buffer 满时只丢诊断样本，不阻塞控制；trace 不拥有安全、admission、dispatch 或 HOLD authority。

这些修复不改变目标姿态、envelope、IK 约束、碰撞模型、终止规则或 HOLD 语义。IK 超过 16.667 ms 当前仍是 `warning_only` 证据，不能未经新验证改成静默忽略或安全旁路。

## 9. Motion trace v2 记录和解读

### 9.1 文件和生命周期

Gateway 关闭 lower owner、解绑 trace callback 后才生成目录：

```text
<trace-root>/<timestamp>_exclusive_rm65_skill_<nonce>/
├── motion_trace.npz
├── summary.json
└── manifest.json
```

Schema：`dexteleop.exclusive_rm65_skill_motion_trace.v2`。v2 新增/强化：

- `measurement_span_ms`、`measurement_available_delay_ms`；
- 每侧硬件读取 total wall/CPU、lock wait、SDK call wall/CPU、parse/validation、retry sleep、attempt count；
- unique measurement count、achieved Hz、interval p50/p95/p99/max、stall count；
- 每侧 hardware-read timing 分布；
- send gap、long-gap threshold、post-stall minimum gap、post-stall burst count；
- `<4 ms` 且 qcmd 发生变化的 `moving_send_under_4ms_count`；
- trace drop/error、operation/branch/reference/qcmd/measured 的关联字段。

### 9.2 正确的验收命令

远端 Gateway 正常运行并在结束时确认 lower HOLD 后，才检查：

```bash
python -m teleop_runtime.skill_deploy.motion_trace inspect \
  /home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked/runtime_private/motion_traces/<trace-directory>
```

同时保存：

```bash
sha256sum \
  <trace-directory>/motion_trace.npz \
  <trace-directory>/summary.json \
  <trace-directory>/manifest.json
```

必须先看 `gateway_stopped.motion_trace.output_dir`、`closed=true` 和 `manifest.result.closed=true`，再使用 summary；只存在一个 `.npz` 或只有控制台输出不能算完整 trace。

### 9.3 新修复后的目标

远端 operator 运行新候选后，优先读取：

```text
measurement_timing.achieved_hz                 接近 30 Hz
actuator_timing.left.post_stall_burst_count   0
actuator_timing.right.post_stall_burst_count  0
moving_send_under_4ms_count                   0
post_stall_minimum_gap_ms                    不小于配置期望
schema                                       v2
```

此外必须继续确认：dispatch 无 drop/error、operation `completed`、最终 pose error 在合同内、shutdown 时 lower HOLD 已确认、trace closed。trace 绿色不等价于运动成功；运动成功以 operation outcome、实测收敛和 lower HOLD 为准。

## 10. 已有真机基线证据（旧 trace）

远端旧 trace 目录：

```text
user@192.168.70.116:
/home/user/zc/AgenticRobot-grasp-motion-trace-1484ff06/runtime_private/motion_traces/20260908_001034_exclusive_rm65_skill_20de8147
```

已返回的 SHA-256：

```text
bb2eba671b2140aa3c146eca46ba3782ceccb45fda8b329288fdd3fee20efa3e  motion_trace.npz
67cd43b95396f50233342b361f52069404ce5954dd453dc0d429d165d1ac407b  summary.json
32d24f2bb25fee49ac39f04d873a5af0bfd9409fc5f6d5a75024a935be7dbd84  manifest.json
```

旧 trace 的事实结论：

- 左臂目标为 `+Z 0.06 m`；dispatch 64/64 成功；expired reference、drop、error 为 0；lower HOLD 和 safe shutdown 已证明。
- 首帧平移误差约 59.995 mm；最终平移误差约 3.970 mm；最终旋转误差约 0.979°。
- IK 超预算记录约 17.540、18.532、39.146、17.451 ms；其中约 39.146 ms 后出现约 41--42 ms 双臂 CANFD gap。
- 测量采样约 20.2 Hz，而配置目标是 30 Hz。
- 单侧最大 SDK call 约 left 120.2 ms、right 91.1 ms。
- 该目录仍是 v1 schema，不能用来验证 v2 新字段或“post-stall burst=0”。

该基线用于解释为什么要修复采样 phase、FK 解耦和 no-catch-up，不能用来声称新候选已经改善真机抖动。

## 11. 模型、权重和资产规则

### 11.1 为什么有时会看到模型下载

模型下载/安装是显式的环境准备，不是 Gateway/Skill 每次启动的默认副作用：

- 外部源码由 `model/perception/` 固定 Git submodule 持有；
- checkpoint 权重不进入 Git，放在共享 `DEXTELEOP_ASSET_ROOT` 的 bundle；
- manifest 中必须记录来源、源码 revision、模型类型和真实 SHA-256；
- Gateway 启动只校验 bundle/manifest，缺失或 hash 不匹配必须 fail closed；
- 未下载 checkpoint 时只能做合同/单元测试，不能声称真实模型端到端通过。

因此“启动 Gateway 发现缺权重后自动下载”是不合规行为。先检查本机和远端是否已经存在可复用 bundle，再由 operator 按来源许可下载/解压/写 manifest。

### 11.2 资产根和缺失 RM65 模型错误

基础资产和可选感知资产都应使用 clone 共享的本地 asset store，不要为每个 linked worktree 复制，也不要把 `assets` 整体提交或通过普通源码同步覆盖：

```bash
export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
python scripts/dev_asset_store.py configure --asset-root "$DEXTELEOP_ASSET_ROOT"
python scripts/dev_asset_store.py status
python scripts/dev_local_resources.py audit
python scripts/dev_asset_store.py link-worktree --worktree /absolute/path/to/worktree
```

远端曾出现：

```text
FileNotFoundError: RM65 asset not found: /home/user/zc/AgenticRobot-grasp-motion-trace-1484ff06/assets/rm65_inspire_tac
```

这表示目标 worktree 的 `assets` 兼容链接/绑定缺失或指向错误，不表示应从网上临时下载 RM65 资产，也不应把另一 worktree 的 ignored `assets` 目录递归复制进 Git。先在远端检查：

```bash
cd /home/user/zc/AgenticRobot-grasp-motion-jitterfix-39ce36d3-tracked
readlink -f assets || true
python scripts/dev_asset_store.py status
python scripts/dev_local_resources.py audit
```

必要时由 `user` 明确把已有、可信的 `/home/user/zc/AgenticRobot/assets` 配置为共享根，再用 `link-worktree` 建立兼容映射。若目标已有指向其他位置的 `assets`，工具会拒绝覆盖；先做只读 inventory，不要 `rm`、`git clean` 或 `rsync --delete`。

### 11.3 权重来源

当前文档不重新发明下载地址；准确的官方链接、许可和 manifest 模板以 [POINT_GRASP.md](./POINT_GRASP.md) 为准：

- 基础/感知资产包：项目根 README 的“运行资产”章节；
- VGN 官方 `vgn_conv.pth`：Google Drive file id `1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ`；
- GraspNet 官方 `checkpoint-rs.tar`：Google Drive file id `1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk`；
- SAM2/Grounding DINO bundle：见 [SAM2_GROUNDING_DINO.md](./SAM2_GROUNDING_DINO.md)。

下载后必须把真实 `sha256sum` 写入 manifest，不能保留全零占位值；GraspNet 还必须保留项目授权 acknowledgement。`gdown` 只用于 operator 选择的下载步骤，不是运行依赖，也不能被启动脚本隐式安装。

## 12. 调试、测试和证据分层

### 12.1 开始写入前的 worktree 盘点

共享 worktree 不是普通单目录项目。任何写入、交接或集成前先：

```bash
git worktree list --porcelain
git -C /home/zc/HandMotion/Developing/temp/worktrees/root-grasp-motion-trace-20260907 status --short --branch
git -C /home/zc/HandMotion/Developing/temp/worktrees/root-grasp-motion-trace-20260907 rev-parse HEAD

python scripts/worktree_inventory.py /home/zc/HandMotion/Developing/AgenticRobot \
  --mainline development \
  --branch task/root/grasp-motion-trace-20260907 \
  --branch task/root/grasp-render-skill-20260907 \
  --branch task/root/grasp-render-motion-freshness-combined-20260907 \
  --branch task/root/real-grasp-motion-freshness-20260907 \
  --branch task/root/rm65-sampler-coherence-fix-20260907 \
  --strict
```

盘点只报告 owner、tip、changed paths 和 overlap，不授予并行修改或集成权限。候选新增本交接文档后，旧的 `ready-for-integration` 声明失效，必须写入新的 tip 和验证结果。

### 12.2 快速离线验证

当前运动候选已通过以下聚焦组合（环境为 `dexteleop310`）：

```bash
bash scripts/run_pytest.sh \
  tests/test_real_skill_deploy_rm65_ports.py \
  tests/test_real_skill_deploy_motion_trace.py \
  tests/test_real_skill_deploy_control_runtime.py \
  tests/test_real_skill_deploy_rm65_lower_assembly.py \
  tests/test_position_control_audit.py \
  tests/test_rm65_actuator_command_trace.py \
  tests/test_agent_gateway_real_rm65_cli.py \
  tests/test_agent_gateway_mujoco_long_move_trace.py
```

结果：`108 passed`。MuJoCo 8 cm Gateway full flow 已覆盖 admission、IK、dispatch、measured convergence、terminal rule、lower HOLD 和 close。

更广的 Real Skill/Gateway/RM65 回归为 `301 passed, 2 failures`；两项失败是当前沙箱禁止 AF_UNIX bind 的环境证据：

```text
PermissionError: [Errno 1] Operation not permitted
```

在 socket-capable 环境按同一 pin 重跑对应 6 项得到 `6 passed`。不能为了让受限沙箱变绿而改测试、绕过 socket 或放宽 admission。scoped Ruff 和 `git diff --check` 已通过；完整历史 `rm65.py` 仍有既有 lint debt。

### 12.3 证据如何呈现

每次交接/验收至少记录：

1. 完整 commit SHA、base SHA、worktree 路径和分支描述；
2. 命令、Python/conda 环境、资产根和关键配置 hash；
3. Sim 输出目录、`visualization_report.json`/`summary.json` 和失败阶段；
4. 真机 operation outcome、dispatch、pose error、HOLD receipt；
5. trace 三个文件的实际路径、schema、hash、closed 状态和关键 timing 字段；
6. 未执行、未验证和因环境阻塞的范围。

不要把以下几类证据混写：

| 证据 | 能证明 | 不能证明 |
| --- | --- | --- |
| 单元/合同测试 | schema、边界、纯函数和失败关闭 | 真机硬件行为 |
| MuJoCo | 坐标转换、IK/碰撞流程、render 视角和 Sim 左/右臂 | 真实 CANFD、相机外参、真实 SDK latency |
| 旧真机 trace v1 | 旧运行的 dispatch/HOLD/时序事实 | 新 v2 字段或修复后抖动已消除 |
| 新真机 trace v2 | 新候选一次实际运行的 timing 和控制结果 | 另一台机器、另一套资产或未运行的分支 |
| `ready-for-integration` 描述 | owner 声明的固定交接信息 | 集成授权或 development 已更新 |

## 13. 故障排查表

| 现象 | 先检查 | 正确处理 |
| --- | --- | --- |
| `RM65 asset not found .../assets/rm65_inspire_tac` | `readlink -f assets`、asset store status、bundle presence | 绑定已有可信共享资产；不要临时下载或覆盖 peer assets |
| 询问“为什么进行模型下载” | bundle、manifest、checkpoint hash、启动命令是否显式下载 | 启动只校验；下载由 operator 单独授权并记录来源/hash |
| `gateway_request_invalid: Lower owner state was captured after the measured cut became available.` | lower owner 的 measurement cut、available time、snapshot freshness/coherence | 这是时序一致性保护；在同一固定候选上重试/取新 snapshot，不要删门禁、放宽 age 或手工伪造状态 |
| `PermissionError: [Errno 1] Operation not permitted` on AF_UNIX | 当前执行环境是否允许 socket bind | 保留为环境证据，在 socket-capable shell 重跑同一 pin；不要改测试绕过 |
| 腕部图看不到方块 | 是否运行了 visualization-only look-at override、报告语义是否正确 | 仅在 Sim 调整 free camera；不要改生产 URDF/registry 来“修图” |
| 一张图里线框太多 | artifact 是否传入多个 candidate | 每个 `camera × candidate` 单独渲染；montage 只放 selected ordinal |
| 左臂运动却使用右侧 profile | `authorized-side`、envelope、tool profile、one-shot `--side` 是否成对为 left | 全部切换到左侧，并重新检查 frame/tool/calibration identity |
| CANFD low-rate warning / 小抖动 | trace 的 achieved Hz、SDK call、send gap、post-stall burst | 先用 v2 trace 复测；不在上层加 sleep、FK 或随意改速度/安全阈值 |
| trace 只有 npz 没 summary/manifest | Gateway 是否正常 shutdown、lower HOLD 是否确认 | 视为 incomplete trace；不能从单个 npz 宣布诊断完成 |
| motion timeout/transport 断开 | operation authoritative status 和 reconciliation | 不直接重复提交；先确认请求是否已接收及最终 HOLD |

## 14. 规则和限制（必须保留）

### 真机安全与权限

- Codex 不自动执行真实机械臂运动、homing、reset、replay、servo、force-control 或设备状态变更。
- 真机命令只作为 `user` 在 `192.168.70.116` 本地 shell 的操作手册；不能声称 Codex 已执行或 user 已验收，除非 user 返回完整证据。
- 不使用 `codex_remote/remote_module.sh`；不通过任意 SSH shell、socket 转发或共享 UID 绕过 Gateway。
- 远端 operator 必须显式确认现场安全、授权 token、selected side、envelope 和 operation outcome。

### Git/worktree 协作

- 不修改、reset、clean、stash、删除、移动或覆盖 peer worktree；不通过删除 ignored assets 制造“干净”。
- 不自动合入 `development`、不推送、不把分支名当作权限；集成前重新 inventory 并重新确认目标/输入 SHA。
- 交接必须绑定 clean tip、base、实现行为、受影响消费者、验证环境、未验证范围和重要评审问题。
- 新提交或新增未提交源码会使旧 `ready-for-integration` 失效；本交接文档提交后必须更新 branch description。

### 运行资产、schema 和诊断

- `assets/`、checkpoint、runtime_private、trace、cache、设备配置和模型源码不作为普通 tracked 源码同步；远端同步采用 tracked-only、no-delete，并保留远端已有 assets/runtime。
- 不覆盖远端已有 `assets` 符号链接；资产是本地资源，使用 asset-store 工具绑定。
- 任一公共 schema 变化必须同步 manifest、summary、测试和消费者；v1 trace 不能冒充 v2。
- motion trace 是旁观者，不得成为 control authority，不得改变 safety/admission/dispatch/HOLD。
- IK 超预算当前是 warning-only，但必须记录真实耗时；不能通过 silent ignore 或 safety bypass 隐藏。
- 渲染 overlay 不做深度遮挡，候选生成的可见点云检查不等于完整碰撞证明。

## 15. 后续工作建议

按以下顺序继续最稳妥：

1. 在文档候选提交后更新 branch description 和 worktree inventory，明确“文档 tip + 运动修复 tip”的组合输入。
2. 获得具体集成授权后，在独立 integration worktree 按固定 SHA 合并并复跑受影响测试；不直接前移 `development`。
3. 由远端 `user` 建立新 tracked-only 目录、复核 asset symlink 和环境，再启动带 v2 trace 的 Gateway。
4. 先执行左臂单轴 6 cm 相对移动；回传 operation JSON、HOLD、trace 目录和三文件 hash。
5. 比较新旧 trace：measurement cadence、SDK stall、post-stall gap/burst、`moving_send_under_4ms_count`、IK over-budget 和最终 pose error。
6. 若抖动仍存在，只基于 trace 定位具体阶段（SDK、planner、send lock、FK/诊断、CANFD）；不要在上层增加无界计算或临时 sleeps。
7. 真机稳定后再把抓姿候选通过受信 proposal 接到运动 Route；相机、Artifact 存储和上层 Agent 接入另开任务，不能借本次交接越过身份/授权边界。

## 16. 交接完成检查单

交接者逐项勾选并附证据：

- [ ] 当前 worktree、branch、base 和 tip 已重新 inventory。
- [ ] 文档/代码工作区 `git status` 干净，`git diff --check` 通过。
- [ ] upward 筛选在 VGN、GraspNet、Sim 单元测试和实际候选 metadata 中一致。
- [ ] 左右臂分别跑过 visualization-only Sim；报告包含 side、frame、camera 和状态不变性。
- [ ] 每个渲染 artifact 只包含一个抓姿；montage 没有混合候选。
- [ ] MuJoCo full-flow 测试通过；失败实验按 stage 分类，不把“有候选”写成“抓取成功”。
- [ ] 远端资产 root、RM65 bundle、tool profile 和 envelope 的实际路径/哈希已记录。
- [ ] 真机命令由 user 手动执行，operation outcome、lower HOLD、trace closed 和 hashes 已回传。
- [ ] v2 trace 的 achieved Hz、post-stall burst、moving send under 4 ms 和 SDK timing 已分析。
- [ ] 明确写出未验证范围；没有把离线、Sim 或旧 trace 描述成新真机通过。
