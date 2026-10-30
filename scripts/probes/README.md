# Shared probes and maintenance tools

This directory contains versioned, non-canonical operator tools migrated from
the former Git-tracked `todos/probes/` directory.  Keeping executable tools
here prevents personal planning state from becoming a runtime dependency.

These scripts have mixed risk levels.  Some are read-only audits, while others
can train models, connect to services, transform datasets, or execute an
explicit `--apply` path.  A filename containing `audit`, `probe`, `verify`, or
`smoke` is not proof that a command is read-only.  Inspect the exact script and
its `--help`, identify its source and target paths, and follow the owning
dataset, deployment, simulation, or hardware safety rules before execution.

They are not canonical teleoperation, collection, replay, or deployment
entrypoints.  Those remain documented under `docs/` and in the repository root
`README.md`.

Experiment-specific learner, loss, training, and evaluation implementations
belong under `experiments/`. A launcher kept here for compatibility must stay
thin and delegate to that owner.

## 冻结观测的双臂抓姿重放（纯离线）

`replay_bimanual_grasp_snapshot.py` 加载已留存 RGB-D、mask、qpos 及标定，
`--sides left right` 共用一次感知推理和物理候选池，各臂随后独立做局部链及路径准入。
`shared_candidates.json` 保留共享池与组件诊断；逐臂结果不代表执行或抓取成功。
原单臂入口保留，模型路径和输入要求见该工具 `--help`。它不创建真实硬件 backend；
本次集成验证使用已留存输入或推理替身，没有启动模型服务。

## DMTAC 连续开合包络对照（纯离线）

`compare_rm65_dmtac_sweep.py` 只读加载已审核 CAD 与 sparse proxy URDF，使用保存的
`observation.json` 双臂 qpos，以及可选的离线端点诊断结果。它不启动设备、不覆盖资产，
也不注册或切换生产碰撞 profile：

```bash
python -m scripts.probes.compare_rm65_dmtac_sweep \
  --cad-urdf assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf \
  --proxy-urdf assets/rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf \
  --observation /path/to/frozen/observation.json \
  --output /path/to/new-owned-output
```

依赖已有 SciPy、Pinocchio/FCL、trimesh 环境。工具验证两级反向转动的 mimic 关系、轴、
零偏和关节范围后，由圆弧弦高上界构造连续全开合保守凸包；密集采样仅作实现复核，
不是连续覆盖证明。另以独立 CAD FK 检查坐标，并确认传感器网格被已审核盒体覆盖。
生产碰撞模型仍使用原 box；诊断克隆只替换 terminal geometry，保留原有 20 对检查
及 20 mm 净空要求。这里不包含完整 planner 的附加 D405/点云障碍检查。

CAD 与 proxy 对照必须使用同一个固定场景。输出先列出桌面/相机杆的原始位置和尺寸
差异，然后仅在诊断克隆中把 CAD 固定场景对齐到 proxy；不能把不同桌面高度下的
距离差解释为 proxy 误差。CAD 距离的 101 个开合样本不构成连续距离认证；凸包覆盖
也不证明路径、抓取或真机可执行性。未经几何和物理场景复核，不将诊断凸包自动上线。


只读留存检查可使用 inspect_safe_move_snapshot.py，直接加载安全移动尝试的
snapshot.json 与同批 RGB-D/FK，不重新连接设备或重算采集位姿。输出目录必须不存在；
报告只提供观测表面、碰撞点投影与离线几何证据，不代表路径执行或抓取成功。


## 冻结局部关节路径的分段离线诊断

`python -m scripts.probes.validate_grasp_pipeline --help` 提供独立的离线验证工具。
它消费已经选定的两阶段关节路径，输入必须含 `pregrasp_target`、`constrained_approach`、
`stage_boundary_stops` 和两段动力学证书；不会重新搜索抓姿或 IK，也不会改变当前 whole-plan 默认。
调用方须保证点云、目标标签、夹爪几何与 MuJoCo contract 来自同一冻结 solver-world 场景。
工具会核对留存路径、原 waypoint 与当前碰撞 profile 身份并拒绝不匹配；外部点云、几何与
MuJoCo contract 之间没有自动交叉 hash 绑定，不能据此声称所有输入身份均已自动验证。

工具复用现有 motion port、C2 补接、碰撞与可见点云复核、不可变计划 store 及 force monitor，
检查前段执行、确认 HOLD、连续停稳、新后段授权 fixture 和一次性消费。worker 仅持有本地规划模型；
力输入是零力模拟源，MuJoCo 通过 qpos/forward 做运动学跟随。输出写入指定的新目录。
这里没有硬件 backend、SDK、相机或模型服务启动，结果不构成执行器跟踪、制动或物理抓取验收。

`--budget` 用于规划与后段采纳等待；它不是强制终止底层进程的硬时限。
`Future.cancel()` 无法取消已运行的原生工作，进程池关闭会等待这类工作结束；应在受控 CPU 资源中运行。
冻结状态的精确起点比较和模拟确认不提供真机恢复策略，也不新增生产 staged Skill 或真实 Route/grant。
失败退出时分别关闭 force monitor 与 plan store；优先保留首个中断，否则保留首个普通异常，
附带其他关闭错误的完整信息，不将关闭失败报告为成功。


## 显式 DMTAC 几何候选

`python -m scripts.probes.build_rm65_dmtac_stepped_proxy --help` 从只读 CAD 和 strict proxy 生成新的
stepped 自体包络与独立 rigid 近似，输出目录必须不存在。stepped 的连续扫掠覆盖证明与双臂
101 个原生 FK 开合样本会保留在报告中；rigid 采用 lower footprint 向上延长，**没有完整 CAD
扫掠覆盖证明**，不能把 stepped 的证明移用于 rigid。生成过程不修改共享资产或运行绑定。

纯 profile 工厂支持显式 `dmtac_proxy_urdf_path=Path(...)`，仅对 DMTAC 场景接受；可验证
原两盒或 core 加 upper/lower 的三盒候选，并构造与实际几何相符的完整 pair 集。默认仍是原
`rm65_dual_arm_hand_proxy_dmtac_strict.urdf` 和 20 对检查；三盒候选是 31 对，不自动被 CLI、
Skill 或执行链选中。这是离线候选加载能力，不能代替具体模型的几何覆盖和执行验收。
## 显式自观测过滤候选

`build_rm65_self_observation_urdf.py` 从已绑定的物理 URDF 和人工复核的胶囊参数生成新目录，
只追加固定在原 link 上的 visual mesh；不改源文件、碰撞模型或默认资产绑定。参数中的
端点和半径单位为米，端点位于指定 parent link 的坐标系。生成结果仍需几何与负对照复核。

快照检查、双臂冻结重放及真实 RM65 Gateway 共用以下显式选项：

- `--captured-robot-filter-config`：JSON 配置；省略时保留旧 15 mm 深度范围、旧匹配与证据哈希。
- `--captured-robot-observation-urdf`：物理 URDF 的 visual-only 克隆；原 link、joint、collision 必须一致。
- `--captured-robot-self-envelope-urdf`：显式的 stepped 三盒/手候选，仅将末端两盒用于捕获时自观测。

后两个资源选项必须同时选择扩展过滤模式，不能隐式改变默认行为。以下配置显式启用
源实现的 CAD 全图射线、工具椭球容差和 50 mm 手臂表面窗口：

```json
{"geometry_mode":"cad_surface","depth_tolerance_m":0.025,
 "tool_surface_bounds_m":[[-0.045,0.045],[-0.05,0.05],[-0.05,0.05]],
 "body_surface_tolerance_m":0.05}
```

`link_primitives` 需完整左右 TCP 标定、显式工具范围和 stepped 自观测盒；未识别的附件必须
选择 `cad_surface`。`Config.for_urdf(...)` 是显式候选选择器，不会被默认构造调用。
可复用的 DMTAC 显式配置在
`teleop_runtime/configs/skill_deploy/rm65_dmtac_capture_self_filter_cad_v1.json`；
它只改变自观测删点，不改变运动碰撞模型或默认 15 mm 策略。
两种扩展模式只删除冻结捕获中的背景点，始终保护目标标签；缓存仅保存模型局部几何和查询树，
每帧重新计算双臂与相机变换。输入哈希绑定 URDF 描述、TCP 和自观测盒参数，不是外部 mesh
内容的完整覆盖证明。运动碰撞仍使用原严格模型与检查对，未将 rigid 候选换为默认。

`inspect_safe_move_snapshot.py` 读取 v2 连续会话时，复用冻结重放入口的
`devices/lifecycle.json` 判定：会话必须 closed 且 cleanup_complete；v1 单次捕获及正常
Skill 记录仍按原格式读取。`--envelope-profile` 只为离线检查显式选择另一份已有包络配置，
默认仍用冻结 context，不替换真机运行绑定。

`benchmark_self_observation.py` 在已关闭会话上复用该读取与装配规则，对同一显式配置比较
CAD 和局部几何模式，输出逐视图 mask、原始 RGB-D、变换、关节值及耗时。必须提供
`--captured-robot-filter-config`（cad_surface 配置）和
`--captured-robot-self-envelope-urdf`；不会用默认配置悄悄放大删点范围。新输出目录不可已存在，
状态名必须唯一且不能逃逸路径，快照必须位于会话内；失败状态不跳过。CAD 是参考实现，
不作为逐像素真值，耗时对比也不代表真机安全或现场附件覆盖证明。

### Safe TCP 保姿与线缆残留验收

保持姿态的普通移动应显式请求 `pose_goal_policy=preserve_target`，并读取
`pose_goal_report` 的实测误差；不能把历史到位容差解释为允许修改目标的授权。
复用新目录发现结果建立 attachment，旧 Route/schema 哈希不可沿用。

`teleop_runtime/configs/skill_deploy/rm65_captured_self_cad_cables_v1.json` 是显式选择的
捕获时 CAD 表面策略：工具 TCP 局部 XYZ 的有符号范围分别为 ±55/±50/±50 mm，
归一化平方和上限 1.05，臂体表面窗口 50 mm。它不改变默认构造、不按物体或像素设例外，
也不把 AABB 角点当作表面；通过 `--captured-robot-filter-config` 选择，并配合既有
`--captured-robot-self-envelope-urdf`。所有目标标签仍保留。显式工具参数上限为 60 mm，
不是默认值；新增策略仅用于解释冻结采集中的自体，不跟随规划中的机器人删点。
靠近机械臂的真实外物也可能落入这层观测窗口，因此不能据此宣称完整外物/携物碰撞保证。
刚性 proxy、固定场景、20 mm 净空和最终路径检查不变。

`--safe-motion-observed-clearance-m` 单独选择 Safe TCP 的观测点云净空；省略仍继承
固定场景的 20 mm。它不调整固定物体/双臂的 CAD 净空，不用于抓姿规划，不接受 Agent
逐请求修改。显式 10 mm 保留体素半对角的保守扣除，近邻障碍与穿透仍被拒绝。
搜索、局部连续求解及最终检查使用同一个请求级 oracle；阈值绑定进入组合场景哈希、
部署配置及碰撞诊断。降低观测余量属于显式实验配置，不等于残留点已经认证为自体。

`--safe-motion-debug-dir` 的拒绝记录同时保存体素中心投影与
`nearest_voxel_sources`：后者按原始体素索引反查过滤后 RGB-D 的真实贡献像素，
每视图最多展示 16 个来源并标注总数与截断，完整数组仍在 `*-filtered.npz`。
投影落在无深度像素不代表该障碍没有观测来源。端点 IK 拒绝也保留最近观测体素，
便于区分腕部、指尖和外物，不能再用局部路径的最近点代替端点证据。
