# Real2Sim fidelity boundary

This package is the simulation-owned boundary for making MuJoCo closer to the
real RM65 + DM-Tac system. It does not introduce a new runtime entrypoint and
does not change shared control, collection, dataset, or real-device semantics.

## What "close to real" means

A useful Real2Sim is not merely visually similar and is not merely able to
replay the same qpos. For the same timestamped action history, it should
reproduce the distributions that affect policy decisions:

1. the same action meaning, frames, TCP, limits, and command lifecycle;
2. the same clocks, delay, hold, interpolation, and sampling phase;
3. plant response including accepted command, servo target, measured state,
   saturation, and asymmetric left/right behavior;
4. camera geometry and timing before photometric randomization;
5. contact, gripper, object, and tactile outcomes relevant to task success;
6. the same causal observation cut and missing/stale-data behavior.

Every fidelity statement must therefore be tied to content-addressed evidence,
an explicit status, limitations, and held-out validation. A runnable scene is
not evidence that a component has been identified.

The offline DM-Tac free-motion fitter is
`python -m sim.rm65.real2sim.gripper_position_fit`. It consumes the standalone
hardware identification traces, fits left/right and close/open independently,
and writes a content-addressed position-only artifact. It deliberately rejects
contact-force claims when no external force gauge exists and never initializes
real hardware. Collection conditions and commands are documented in
`hardware/DMTAC_gripper/GRIPPER_ACTUATOR_IDENTIFICATION.md`.

## Current baseline (2026-07)

| Component | Current sim truth | Status | Main gap |
| --- | --- | --- | --- |
| Shared action/control semantics | Uses the shared qref/qcmd planning path | strongest existing layer | Keep as read-only dependency |
| Arm transport/plant | All-valid-data fit of per-joint delay, high-speed J5-J6 target lag, and MuJoCo position actuators | provisional | New untouched held-out validation, force limits, and contact/load dependence |
| DM-Tac free motion | Bilateral, direction-specific delayed rate/acceleration-limited path | identified for step response | Fast preemption remains provisional; contact/load response is unmodeled |
| Contact/grasp | MuJoCo contacts; some task paths still use debug contact latch | debug/provisional | Force/contact coupling, slip, compliance, and real success labels |
| Timing | Independent clocks exist in `TimingSpec` | provisional | Some JSON inputs inherit code defaults; real jitter/phase/drop distributions are absent |
| Camera geometry | Sim camera streams exist | provisional | Real intrinsics/extrinsics, 15 Hz behavior, rolling/exposure latency, masks |
| Photometry | Rendered RGB | unmodeled | Exposure, white balance, noise, blur, lighting/material distributions |
| Tactile | No policy-grade tactile observation model | unavailable | Geometry-to-DM-Tac signal and latency model |
| Formal state trace | Accepted/target/measured state are now separate | contract-complete | Add real/sim paired reports and thresholds |

These labels are deliberately conservative. Promote a row to `identified`
only through a versioned manifest and held-out evidence.

### Arm free-motion baseline: `20260721.v4`

The v4 profile uses fourteen valid paired records: ten clean captures and four
guardian-restored captures. Each pair contains the 120 Hz SDK-accepted qcmd
trace and independent approximately 200 Hz UDP measured-qpos feedback, with
zero reported command errors or disabled feedback rows. Guardian-restored
records retain weight 0.5 because they lack an in-band UDP stop row; clean
records use weight 1.0.

The four evening captures `221544/221548`, `222640/222650`, `224159/224204`,
and `224526/224530` were first evaluated without tuning against frozen v3.
Their position-error thresholds passed, but the maximum effective-delay
mismatch was 8 ms, above the inherited 5 ms gate. That failed v3 held-out
result is permanently recorded before the captures were explicitly converted
to v4 fit data. The nominal `220204/220211` traces were rejected because they
have no common command/feedback window; orphan feedback `223212` was also
rejected. Neither contributes samples or weight to the fit.

J1-J4 remain frozen because their millisecond-scale delay estimates were not
stable across clean sessions. Exact 1000 Hz MuJoCo replay selected a bilateral
J5-J6 update with a 50 degree/second velocity hinge and a 75 degree/second fit
cap. The exact residual delays, slopes, and caps are versioned in
`sim/rm65/actuation/profile.py` and the v4 manifest.

Across all fourteen fit sessions, v4 changes mean per-joint qpos RMSE from
0.360008 to 0.358448 degrees. The weighted objective improves 0.712%, the ten
clean sessions improve 1.244%, and the four newly converted clean sessions
improve 1.819%. However, the guardian-restored subset regresses 0.463%, the
worst-joint RMSE changes from 1.168741 to 1.185046 degrees, and maximum absolute
error changes from 9.177512 to 9.206431 degrees. V4 is therefore a useful
stage result but remains `provisional`, not an identification claim.

The exact v4 source index and fit manifest are
`artifacts/rm65_arm_all_valid_fit_20260721_v4_sources.json` and
`artifacts/rm65_arm_free_motion_20260721_v4.json`. V3 preserves its failed
held-out result, while v2 and v1 remain historical. Promotion to `identified`
requires a new clean capture collected after v4 is frozen and never used for
tuning.

The v4 fit manifest remains immutable when runtime implementation or
configuration changes. In particular, its source hashes describe the exact
core and MuJoCo adapter used for the 20260721 fit; tests compare those hashes to
fixed historical values rather than silently reinterpreting the fit against
the current files.

真机操作者断开／重连双臂双夹爪与操作者开合夹爪使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_hardware_connection_v1.json`。它固定于合并提交
`f99ab7c8a`（基线 `d92d01f3e` 即真机持物闭合测量绑定提交，+ 任务分支 `0f20b76bc`），保留前一绑定的全部非运行时引用；
198 项已有运行时输入中 3 项改变（真机 Gateway CLI、组合、相机资源；Unix 传输和运行时服务也有改动，但不是绑定的运行时输入），新增 1 项
（`agent_gateway/profiles/real/hardware_connection.py`），合计 199 项。内容移植自 2026-10-24 现场标签
`agent-safe-move-hardware-connection-production-final-20261024`（`da450b2a6`、`998359d5c`、`36a39440e`），并新增操作者夹爪 RPC。
只有显式 `--enable-hardware-connection-control`（要求双夹爪 v2 配置）才启用：断开走原应用关闭事务，相机与感知模型常驻；重连新建双臂、
双夹爪、控制运行时和执行会话，只建立初始 HOLD，新 `runtime_session_id` 使旧请求、计划和批准失效；急停锁存不能经重连解除。
`request_operator_gripper` 只在已连接、代次一致、无急停锁存、无控制故障且没有 Skill 操作时执行，走夹爪 Skill 端口与收敛规则后 HOLD 夹爪。
离线证据按内容哈希记录：1353 项关联测试（当前绑定测试在本绑定前排除）与设计说明。绑定仍为 `provisional`，没有新增真机运动执行或 held-out 资格。

真机持物测量按 DMTAC 实测闭合放置手指，使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_held_closure_v1.json`。它固定于合并提交
`1664dba7f`（基线 `d95a13374` 即真机放置小支撑物窗口绑定提交，+ 任务分支 `c20f490d5`），保留前一绑定的全部非运行时引用；
198 项已有运行时输入中 2 项改变（`robot_motion/held_object_measurement.py` 新增 `tool_entity_at_opening`，
`agent_gateway/profiles/real/held_object.py` 先读夹爪样本），新增 0 项，合计 198 项。有夹爪读数时，手指、DMTAC 接触实体和闭合区域
按实测开口放置（开口 = 最大开口 7.05 cm ×（1 − 闭合度），最小 1 mm）后再取种子像素和计算包围盒，部分闭合的空夹爪不再把自己的手指当成物体；
没有夹爪读数时几何不变。离线证据按内容哈希记录：1412 项关联测试（当前绑定测试在本绑定前排除）、14 个 Dashboard 持物样本
（28 个手臂状态，视觉加闭合判定 TP/FN/FP/TN 6/0/2/20 → 6/0/0/22）和 34 个保存的 RM65-6F 场景回归（空手臂持物 0/59，目标 5/5）。
绑定仍为 `provisional`，没有新增真机运动执行或 held-out 资格。

真机放置的小支撑物窗口使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_small_support_v1.json`。它固定于合并提交
`43f134f95`（基线 `dfbd54b98` 即真机放置位姿查询绑定提交，+ 任务分支 `529a16a94`），保留前一绑定的全部非运行时引用；
198 项已有运行时输入中 2 项改变（真机物体运动的放置支撑面测量、真机放置位姿查询），新增 0 项，合计 198 项。
放置前先用足迹外扩 4 cm 的窗口测量支撑面（原规则），它不是一个平面时改用足迹外扩 1 cm 的窗口；一个窗口通过要求持物臂腕部相机的点
拟合成一个平面，并且没有点低于该平面 1.5 cm 以上（越过支撑物边缘时下方表面会出现在这里）。0–2 cm 间隙检查、规划器的支撑面许可和
抬起补全不变；放置位姿查询按同一检查预测，并让足迹离认证顶面边缘至少 1.5 cm。离线证据按内容哈希记录：1168 项关联测试
（当前绑定测试在本绑定前排除）、小支撑物说明与放置生命周期冒烟输出。绑定仍为 `provisional`，没有新增真机运动执行或 held-out 资格。

真机持物放置位姿查询 `perception.query_support_release_pose@2` 使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_support_release_v1.json`。它固定于合并提交
`b9091bf5e`（基线 `be42c02d9` 即真机 RM65-6F 绑定提交，+ 任务分支 `47cb84014`），保留前一绑定（RM65-6F）的全部非运行时引用；
196 项已有运行时输入中 7 项改变（真机 Gateway CLI、组合、会话、工厂、primitive profile、物体运动和工具实体的增量接线），
新增 2 项（真机查询模块和 v2 契约），合计 198 项。行为变化只在显式选择时生效：`--support-release-prompt SUPPORT_ID=PROMPT`
（可重复）并且部署了物体运动、RM65-6F 场景 profile 和 registry v5 时才发布这个只读查询。它要求一次抬起后保留的已验证抓握，
按真机深度容差（残差 6 mm、倾斜 5°、4 mm 网格，点数最多的 4 mm 高度带作平面种子）认证支撑面顶部矩形，返回 pre-place 位姿
（phase=carry）和放置位姿（phase=place，物体底部离支撑面 0.5–1.5 cm），两段都用部署的载荷规划器检查；先在每个矩形上检查持物
yaw，再检查转 yaw，最后检查倾斜。它不派发运动。MuJoCo 的 v1 查询不变。离线证据按内容哈希记录：1278 项关联测试（当前绑定测试在
本绑定前排除）、34 个保存的真机场景上的支撑面认证和合成持物查询报告，以及支撑面提示词调查。绑定仍为 `provisional`，没有新增真机运动执行或
held-out 资格。

真机 RM65-6F 运动与 Station 模型、相机 registry v5、D435 深度修正和可选的手部—桌面净空使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_real_rm65_6f_v1.json`。它固定于合并提交
`790182d3a`（基线 `eed84fb8f` 即真机持物绑定提交，+ 任务分支 `66d06523b`），保留前一绑定（真机持物本臂腕部测量）的全部非运行时引用；
192 项已有运行时输入中 3 项改变（真机 Gateway CLI、障碍碰撞 oracle、抓取轨迹规划），新增 4 项（RealSense 感知绑定、
场景 profile、障碍场景快照、registry v5），合计 196 项。行为变化只在显式选择时生效：碰撞场景
`dual_arm_hand_proxy_dmtac_{strict,grasp}_rm65_6f` 使用 joint 6 加长 17.2 mm 的 6F 运动模型（link6 即 6F 法兰），并默认使用 6F Station URDF
和 registry v5（主相机为 10-31 手眼经机器人连杆 ICP 修正，腕部为 10-31 手眼挂在 6F 法兰上）；registry 的 `depth_correction` 按相机序列号
修正米制深度（v5 只修 D435，×1.01406；D405 待标定板测量）；`--table-hand-clearance-m`（0–0.02 m，只能配 6F profile）只替换夹爪代理、
腕部相机代理和所持物体与固定 CAD 桌面之间的净空，接触仍然禁止，肘部、相机立柱和双臂之间保持 0.02 m。观测体素阻挡时，检查结果改为指名观测体素。
默认 profile、默认净空和标准 RM65 资产不变。6F 资产不在版本库内，由 `scripts/assets/make_rm65_6f_station_assets.py` 生成，其内容哈希记录在
metadata。离线证据按内容哈希记录：3683 项关联测试（当前绑定测试在本绑定前排除）、6F 模型与安装一致性检查、桌面净空检查、资产复现检查，
以及 34 个保存的真机场景的 6F 离线复测报告。绑定仍为 `provisional`，没有新增真机运动执行或 held-out 资格。

真机持物测量改用持物臂自己的腕部相机，并且 SAM3 点提示未命中按该视角无掩码处理，使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_real_held_own_wrist_v1.json`。它固定于合并提交
`b278f4532`（development `e00a9d9d3` + 任务分支 `68e69d748`），保留前一绑定（MuJoCo 持物放置第三批）的全部非运行时引用；
192 项已有运行时输入中 5 项改变（真机持物观测与运动生命周期、持物度量、真机相机几何、分割提供者、多视角抓取融合、SAM3 客户端），合计 192 项。
行为变化：真机持物测量以持物臂自己的腕部相机为主视角，只有它能判定两指之间有物体（种子像素避开深度断边和离夹爪实体 5 mm 以内的点，至少 15 个）；
其他相机只在主视角确认后、在它们看到主视角物体的位置用点提示分割，每边最多把盒子扩大 3 cm；首次抬起的支撑补全和放置的支撑面只用持物臂腕部相机拟合，
其他相机的高度作为一致性证据报告（真机持物运动仍默认关闭）。SAM3 点提示的最佳掩码不含所给像素时，该视角按无掩码处理：持物观测记录
`no_mask_at_seed` 并继续其他视角，分割 Skill 和表面锚点得到原有的无掩码结果，多视角抓取融合排除该辅助视角（`target_point_mask_unavailable`）；
文本提示的同类异常和传输、协议、掩码格式错误仍然报错，主抓取目标点提示未命中仍使该次请求失败。离线证据按内容哈希记录：
2262 项关联测试（当前绑定测试在本绑定前排除）、点提示未命中的单独检查，以及 33 个保存的真机场景离线复测报告（不该有物体的臂状态报 held 25/57 → 0/57）。
绑定仍为 `provisional`，没有新增标定、真机运动执行或 held-out 资格。

MuJoCo 持物放置第三批（批次 2d、四分之一速度、2e）使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_mujoco_agent_place_v3.json`。它固定于合并提交
`576aacb41`（development `6deeefe32` + 任务分支 `07e567b84`），保留前一绑定（持物放置第二批）的全部非运行时引用；
192 项已有运行时输入中 5 项改变（释放候选生成、释放查询与支撑面度量、MuJoCo 持物运动规划），合计 192 项。
行为变化均为默认关闭的 MuJoCo 放置链：持物关节速度降为四分之一（半速时持物沿规划路径的偏差最多吃掉 3.7 mm 规划间隙，
四分之一速度最多 0.85 mm）；释放查询在每个朝向下增加离持物最近的可行位置；支撑面顶高改为已认证平面矩形最高角点加 2 mm
认证残差（原先取掩膜内所有高点的最大值，掩膜边缘落在支撑面上的物体时会把释放高度抬到该物体顶面）；一块区域里保持持物姿态的
候选都没有路径时，同一区域的位置在当前偏航下再把持物姿态朝机械臂基座多转 6° 尝试一次（物体相对支撑面法线的倾角不超过 10°），
结果字段 `release_lean_rad` 报告附加的倾角；没有可用候选时以类型化拒绝返回，不再中断查询。
1 mm 硬穿透、3 mm/2° 滑移、50 ms 观测间隔、0.6 s 释放窗口和 2 mm 非接触间隙均未改变。离线证据按内容哈希记录：
875 项关联测试（当前绑定测试在本绑定前排除）、独立评审（结论为有条件通过，问题已在 `eb229e6af` 修正）与 DeepSeek Agent 放置套件运行台账
（queue-b-05 至 queue-b-08，以及 queue-b-08 三次倾斜释放运行在 `07e567b84` 上的重放：运动、夹爪和查询调用结果一致，
只有一次引用原运行图像 ID 的只读定位调用在重放中未提交）。
绑定仍为 `provisional`，没有新增标定、真机运动执行或 held-out 资格。

MuJoCo 持物放置第二批（批次 2/2b/2c）使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_mujoco_agent_place_v2.json`。它固定于合并提交
`eb44be38a`（development `a0121721f` + 任务分支 `bb8aacdff`），保留前一绑定（真机持物放置）的全部非运行时引用；
192 项运行时输入中 9 项改变（MuJoCo 持物运动规划、对象运动生命周期、持物观测、放置与抓取查询及其度量、
Gateway 装配），没有新绑定的源码。行为变化均为默认关闭的 MuJoCo 放置链：持物运动从机械臂当前保持的关节指令出发
（与测量值相差不超过 5 mrad 时），关节速度减半；目标保持 0.5 s 仿真时间；持物终点 IK 须在路由容差内再留出
1 mm/0.5° 的下垂余量，路径走完后 TCP 在 1.5 s 内未进入容差即带 HOLD 失败；释放查询依次尝试最多三块已认证的
空闲矩形；抓取查询只提供水平侧抓和俯抓（去掉 35° 俯斜侧抓），每个候选的路径搜索 12 s；持物观测对同一原生状态复用一次
采样。1 mm 硬穿透、3 mm/2° 滑移、50 ms 观测间隔和 0.6 s 释放窗口均未改变。离线证据按内容哈希记录：772 项关联测试
（当前绑定测试在本绑定前排除）与 DeepSeek Agent 放置套件运行台账（queue-b-03/04）。绑定仍为 `provisional`，没有新增
标定、真机运动执行或 held-out 资格。

RealMan 真机持物搬运发布默认关闭的 `motion.move_grasped_object_to_pose@1`（动作
`move_grasped_object_to_pose`，物体 `<side>_held_object`），使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_real_place_v1.json`。它固定于提交
`bba13f9d4`（合并 `56dd61af0` = MuJoCo 放置集成 r3 `81d1fe75f` + 真机分支 `4bc971814`，加悬停摘要长度修正，再合并评审修正 `67a90edf8`），保留前一绑定（MuJoCo 持物放置）的
全部非运行时引用；189 项已有运行时输入中 20 项改变（真机 Gateway 装配、会话、CLI、动作授权、夹爪端口、运动原语
生命周期与 Profile、带负载的观测安全移动规划、障碍碰撞与起点豁免、悬停查询及其注册，以及经同一分支带入的
MuJoCo 悬停预算选项），另绑定 3 项新运行时源码（持物包围盒测量、真机持物观测、真机持物运动生命周期），
合计 192 项；2 项离线查询输入改变（`tool_entity_goals.py` 的水平朝前/朝左/朝右姿态，
`observed_obstacles.py` 的起点接触豁免）。真机每次持物运动先用三路静止 RGB-D（以两指之间的像素为分割提示）
和夹爪闭合测量手中物体，不在两指间、或夹爪无采样/有故障/最后指令不是闭合/闭合度不在 0.05–0.97 之间即拒绝；首次 lift 才把包围盒补到其支撑面，同一抓取（夹爪无新指令且闭合度变化不超过 0.03）保留并合并包围盒；测得的工具系包围盒沿整条路径参与碰撞检查，起点时物体自身体素
豁免；放置只豁免目标足迹下实测支撑面 15 mm 带内的体素，包围盒底面须在支撑面上方 0–2 cm，放置按 `preserve_target` 精确到达目标；到位后再测量，
掉落或未落在支撑面上则失败并给出原因。离线证据按内容哈希记录：2166 项关联测试（当前绑定测试在本绑定前排除），
以及只读真机快照上的带负载规划与持物测量结果、持物运动门控的离线核对（21 项）。绑定仍为 `provisional`，没有新增真机运动执行、标定或
held-out 资格。

MuJoCo 持物放置链发布两个默认关闭的只读查询 Skill（`perception.query_support_release_pose@1`、
`perception.query_grasp_approach@1`），使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_mujoco_agent_place_v1.json`。
它固定于合并提交 `c8d22fd5109c`，保留前一绑定的全部非运行时引用；183 项已有运行时输入中
11 项改变（MuJoCo Gateway 装配、会话、CLI、夹爪命名释放观察、持物运动生命周期与观察器、
物体搬运位姿校验及运动 schema 常量），另绑定 6 项新运行时源码（深度支撑面测量、
两个查询的 Skill 定义与 MuJoCo 端口、运动 schema），合计 189 项。持物运动去掉竖直 0-8 cm 抬升走廊
（真机配置本无持物运动），抬升在握持未验证前须超过剩余验证高度（3 cm）加到达容差和 2 mm，否则在
激活时拒绝；TCP 走完检查路径而物体状态 1.5 s 仿真时间内仍不满足时 HOLD 并给出原因；物体目标经同一
观测切片中实测的手-物变换换算为 TCP 目标，冻结关系仍约束载荷几何及 3 mm / 2° 滑移门限。1 mm 硬穿透、
50 ms 观测间隔和 0.6 s 释放窗口不变。离线证据按内容哈希记录：659 项关联测试（运行时源码与本树一致），
以及 DeepSeek Agent 在 MuJoCo 放置套件上的运行台账（含掉落、抓歪、放置路径阻挡注入）。绑定仍为
`provisional`，没有新增真机动作、标定或 held-out 资格。

上一阶段夹爪—物体关系与夹爪悬停路径查询发布为 RealMan 与 MuJoCo Gateway 的默认关闭只读 Skill
（`perception.describe_tool_object_relations@1`、`perception.query_tool_hover_paths@1`），使用独立历史绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_gateway_unify_v1.json`。
它固定于合并提交 `9b5c06ce9523`，保留前一绑定的全部非运行时引用；177 项已有运行时输入中 12 项
改变（两侧 Gateway 装配、会话、CLI 及 MuJoCo 观测规划），另绑定 6 项新运行时源码（共享工具实体
场景、DMTAC 实体绑定、Skill 定义及通用/RealMan/MuJoCo 执行器），合计 183 项；6 项离线查询输入中
仅保存场景查询入口改变。悬停每个候选 5 s、整体 20 s，TCP 区域查询仍为 3 s；悬停注册绑定安全 TCP
Route 的公共坐标系并要求其 +Z 竖直。MuJoCo 观测场景安全移动在 CLI 配置观测场景时默认启用正距离
起点退出，配置数据类默认仍为 strict。离线证据按内容哈希记录：同一源码树 2492 项关联测试；21 个
保存真机样本经生产关系核心的结果与探针逐视角一致（354/354 视角、162/162 融合物体）；6 个样本
24 组悬停查询中探针与生产核心 24/24 一致，公共坐标系与世界坐标系 23/24 一致，唯一差异是首个候选
恰在 5 s 预算边界（5.001 s 对 4.949 s）。MuJoCo 孪生悬停诊断中保持姿态的提议被腕部与中央相机立柱
的真实碰撞拒绝，更换碰撞配置结果不变，工具轴向下找到路径并按规划终点无接触执行。绑定仍为
`provisional`，没有新增真机动作、标定或 held-out 资格。

上一阶段离线夹爪—物体关系及悬停倾角锥使用独立历史绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_relations_v1.json`。
它固定于合并提交 `d20dd86bb0e1`，保留前一绑定的全部非运行时引用，177 项运行时输入未变；
6 项离线查询输入中保存场景查询与工具实体几何两项改变。离线证据按内容哈希记录：同一源码树
1630 项关联测试、21 个保存操作员样本的关系结果（干净起点无两指间/接触带判定，预抓 5/5 两指
之间，接触 4 例中 3 例在接触带）和 MuJoCo 孪生 20/20 设计摆放。关系只读、不发布 Gateway
Route；URDF 传感器盒随各保存场景 manifest 的冻结文件绑定，因为资产库不入 Git。绑定仍为
`provisional`，没有新增真机动作、标定或 held-out 资格。

上一阶段观察点云距离缓存及离线端点排序的组合来源使用独立历史绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_observed_cache_order_v1.json`。
它固定于源码提交 `b2b77e61e09b`，保留前一绑定并记录 177 项运行时输入，其中观察点云
距离代码用于真实观测规划；6 项离线查询输入及同一源码树的 978 项关联测试单独记录。
该列表不是全部传递依赖或工作区漂移的实时检查，固定 commit/tree 与干净工作区门禁另行
核验。绑定仍为 `provisional`，没有新增真机动作、标定或 held-out 资格。

观察点云缓存对相同输入复用原距离结果，缓存键及计算使用稳定快照；调用方仍负责配对
输入的跨线程一致性。离线端点排序默认 `joint_distance`，其他已有模式需显式选择并进入
查询身份和审计；此配置不会改变实时设备执行权限。

上一阶段球体载荷、仿真对象接触及 Skill 悬停查询的组合来源使用独立历史绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_hover_visual_v1.json`。
它固定于源码提交 `1e2ab619c2f3`，保留前一绑定并记录 176 项运行时输入、6 项离线查询
输入，以及同一组合源码树上的 1297 项专项通过；完整组合门禁另行记录。该列表不是对
全部传递依赖或后续工作区漂移的实时检查，完整源码身份由固定 commit/tree 与干净工作区
门禁共同约束。该绑定仍为 `provisional`，没有新增真机动作、标定或 held-out 资格。

离线保存场景查询及 RealMan 配置观测场景后的安全移动默认启用正距离起点退出，硬接触、
穿透、新增近邻及距离恶化仍按原门禁拒绝，运动侧必须证明最终退出到普通 20 mm 净空。
`--strict-start-clearance` 可显式选择原严格起点准入；未配置观测场景时 RealMan 不启用
正距离退出，直接调用的下层规划器默认亦不变。先前 12 场景对照是旧的显式启用实验，
不能当作新默认的真机验收。

上一阶段区域查询与端点路径质量下界使用独立历史绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_skill_region_checked_v2.json`。
继承168项输入，更新两个已有规划源码，并绑定路径质量模块和起点检查采样器，合计170项
输入／166项Python源码；离线查询入口另作源码证据，固定于 `b3b03e8a6212`。
源码及消费者离线检查805项、相同源树的查询检查252项分别记录，完整组合验收另行执行。
早期 `skill_region_v1` 候选仍作为未完成完整验收的历史证据保留，不作为当前绑定。
该阶段离线区域查询默认 proximity、可显式选择 halton；当时严格20mm起点净空保持默认。
来源12场景对照显式启用了正距离起点退出，不能推断当时严格默认下的通过率；没有新增真机
标定、运动或held-out资格，既有fit、timing及所有历史绑定保持。

此前显式起点净空能力使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_explicit_clearance_v1.json`。
保留168项输入／164项Python源码，仅更新四个既有owner，固定于 `e9245d2fdd87`；
净空、目录保护及相关消费者977通过，9项subtest另列。严格默认不变，
`--enable-inherited-start-clearance` 显式启用并进入Real执行manifest；仅已有正距离近邻可继承
请求局部下界，运动侧必须通过最终密集采样的离开证明。查询故障保持原原因并拒绝。
完整组合门禁另行记录，不以规划结果替代真机动作资格；旧绑定、fit和timing不改。

前驱实测点与表面锚点的真机安全运动接入使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_real_observed_motion_v1.json`。
继承164项输入并更新6项，补入三个执行／装配源码和显式CAD过滤配置，
合计168项输入／164项Python源码，固定于 `c0334f16467f`；相关离线消费者632通过。
实测点动作、表面锚点v2采集及锚点v2运动均显式选择，复用原授权、安全规划和HOLD；
默认15mm自观测范围不变，CAD配置需显式指定。此绑定记录源码及合同资格，
不代表真机运动或上层Agent任务成功；原fit、timing、历史绑定和held-out资格保持。

前驱 MuJoCo 显式指定支撑体释放使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_named_release_v1.json`。
继承163项输入并更新11项，新增公开夹爪定义，合计164项输入／161项Python源码，
固定于 `88db4d1db3c3`；夹爪、事务回滚、观察器、CLI及合同消费者479通过。
`--enable-named-release-v3` 显式启用，v1/v2默认及真机／仿真15秒上限保留；
同一rig锁内排队、提交支撑身份及失败回滚复用原owner。完成开爪后仍需独立观察
原0.6秒释放窗口，源native场景真值诊断不等于Agent任务资格。完整组合门禁另行记录，
旧绑定、fit、timing和held-out资格不变。

前驱 Agent/MuJoCo 提案过期清理与 measured 私有 cut 复用使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_agent_proposal_cut_v1.json`。
它继承163项输入／160项Python源码，仅更新factory、target和payload三个既有owner，
固定于 `4de2591b3c48`；当前cut、过期/HOLD及checkpoint消费者验收373通过。
Gateway清理开关启用，下层target factory仍默认关闭以保留手动checkpoint子集；
原TTL、sequence高水位及HOLD不变。完整集成及包内离线门禁另行记录，fit与timing不变。

前驱 SAM3／MuJoCo 携物支撑组合使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_sam3_placement_v1.json`。
它继承154项输入并更新其中10项，补入九个已改动的夹爪、物体观测、支撑和合同owner，
共163项输入／160项Python源码，固定于 `b584813411cc`；组合消费者验收652通过。
真机／仿真夹爪公开操作上限均为15秒，原较短请求与下层保护仍有效；支撑网格严格不超过100000点。
本批公开新增为显式支撑体v2 Route；释放候选及深度支撑几何仍是下层API，无端到端放置成功声明。
完整非硬件组合资格另行记录，fit、timing、旧绑定及held-out资格不变。

前驱显式 SAM3 多实例定位整合使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_sam3_candidates_v1.json`。
它继承前驱全部154项输入／151项Python源码，仅更新三个SAM3客户端／CLI源码身份，
固定于 `66765c65e736`；定位、分割及三类CLI消费者验收674通过、零跳过／失败。
默认单实例binding哈希保留，多实例另行绑定，只读RPC依赖RPent `24eb2ff00cc7`。
完整组合门禁另行记录，历史40份artifact原字节保留，fit、timing和held-out资格不变。

历史锚点运动 v2–v4、预抓取连接比较与夹爪资源生命周期整合使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_motion_gripper_v1.json`。
它继承153项输入，更新其中9项源码身份，另纳入已改动的 Real gripper resource owner，
共154项输入／151项Python源码，固定于 `6c8f9289d4ad`。该源码的CLI、夹爪、手动维护及
领域边界消费者验收572通过、零跳过／失败；完整组合门禁另行记录。
历史39份绑定和标定原字节保留，v4 fit与timing不变，不新增物理响应或held-out资格。

历史 SAM3 默认与外部定位 owner v2 整合使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_sam3_default_v1.json`。
它继承下面只读默认绑定的全部153项输入／150项Python源码，更新其中7项源码身份，
固定于 `8bdf790dde02`。固定源码的离线 owner／消费者验收590通过、零跳过／失败；
完整组合门禁另行记录。SAM3服务和真实设备未启动；v4 fit、timing及历史绑定原字节保留，
不新增外部checkpoint身份认证、物理响应或held-out资格。

历史只读默认发布与可选 TCP arc 排序整合使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_readonly_defaults_v1.json`。
它继承 checkpoint integration 的152项输入，更新两端CLI默认发布的源码身份，
并加入共用抓取轨迹源码，共153项输入／150项Python源码，固定于 `b19fd84165d7`。
旧绑定和v4标定原字节保留；固定源码的直接消费者验收为601通过、零跳过／失败，
完整组合门禁另行记录。该绑定不重拟合，不新增物理响应或held-out资格。

历史 checkpoint 整合候选使用独立绑定
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_checkpoint_integration_v1.json`。
它合并主线 V25 的92项输入与历史 observer 候选的147项输入，再加入 TCP envelope
companion，共152项输入／149项 Python 源码，固定于 `a47d7e3922ef`。两份前驱
manifest 的原字节和资格各自保留，使用不同路径与引用身份；不继续争用数字版本名。
原生 checkpoint owner、inactive envelope target 和公共隐私出口的固定消费者验证为
218通过、零跳过／失败。完整非硬件门禁另行记录，不把前驱报告合计当作组合验收。
模型选择和默认控制路径继承主线；私有恢复数据不进入公共 Artifact，未重新拟合，
不声明 Gateway operation/READY、磁盘冷恢复或 RPent 上下文恢复已经完成。

主线历史 V25 为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_v25.json`。
V25 冻结 V24，登记显式模型选择、原生抓取观测的私有状态摘要及拒绝诊断的公开边界。
继承 88 项输入并更新其中 3 项，新增 SAM3 客户端和三个抓取叶子，共 92 项输入、
89 项 Python 源码；匹配固定提交 `83581f2ac`，直接消费者验收 766 项通过。
默认 SAM2、进程内 GraspGenX 与 GraspMoE、v5 几何、原规划/碰撞/取消/HOLD 门禁保留。
完整 oracle 端点检查留在私有诊断，公共查询只过滤这一新增字段，不更改原有公开字段。
离线几何 benchmark 由固定 Git 与报告绑定，不加入运行时输入，也不授予执行权。
本批未启动模型服务、设备或训练；不新增模型质量、真机、held-out 或六路并发资格。

本分支另保留历史 checkpoint observer 候选：
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_checkpoint_observer_v1.json`。
它完整保留原候选 V25 的字节和身份，固定源码为 `db09fe458444`、147项输入，
不改写主线 V25，也不将两份独立验收解释为组合验收。合流采用上面的独立
checkpoint integration 绑定，并保留这份历史原件。

历史 V24 为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v24.json`。
V24 冻结 V23，并补齐 ControlStation 兼容调用方的源码绑定。继承的 86 项输入
保持 V23 字节，新增兼容入口与 Real factory 两项，共 88 项输入、85 项源码，
匹配固定提交 `9156ae01d`；对应调用方验收为 125 项通过、零失败。
该默认关闭入口只装配既有绝对位姿 motion 与 measured-state Route，在资源创建前
拒绝已退休或未装配的动作。双臂安全观察、所选单臂授权、原数值包络、执行 owner
及 lower HOLD 关闭证据保留。本绑定不新增真机、模型或 held-out 认证。

历史 V23 为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v23.json`。
V23 冻结 V22，登记显式可选的 safe TCP v2、只读连续查询 v1–v5，以及已授权的
十个调试运动入口归档。86 项输入匹配固定源码提交 `6b36561dc`；直接调用方验收
为 1,744 通过、1 项原有可选依赖跳过。工具轴证据只覆盖采样参考与 float32
发布表示；查询不能激活计划。既有运动、观测障碍检查、授权及下层 HOLD owner
保留。新可选杂物任务显式使用其源合同的 1 kHz，既有 4 kHz 默认不变。
未重新拟合机械臂响应，不代表新留出数据、模型服务、真机或六路并发调度验收。

历史 V22 为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v22.json`。
V22 冻结 V21，登记共享 supplied-image localization、真机兼容入口及默认关闭的 MuJoCo 定位。
真机 v1 的既有契约身份和有界 CAS 读取保持一致；MuJoCo 保留原始图像坐标，
复用现有分割模型及关闭门禁。更新4项来源，新增4项，共64项源码、65项输入。
固定631项通过、1项原RPent配对轮子缺失跳过、零失败的消费者报告绑定
`ab5de19b3dcda8cc66bae9a85f4916d0d7f6a28a` / `a835306ec77f78d2d9b2d7f8b0864741f2cf1b81`。
该证据只覆盖本地离线集成，不新增模型、硬件或 held-out 响应认证。

历史V21为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v21.json`。
V21冻结V20，仅重绑motion_deployment一项源码，仍为60项源码、61项输入，无新增输入。
显式profile导出恢复原79b6默认字节；归一化模式仅启用时明确输出，原语义hash、IK、
控制及执行门禁不变。固定217项通过、零跳过的消费者报告绑定
`d1a54844dcc27a9f24f48e7dd77788bbb11c7319` / `306fdc47e383abe480f92b0403ee7f213e4b3152`。

历史V20为 `artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v20.json`。
V20在冻结V19之上登记默认关闭的对象/关系查询、认证表面锚点v1/v2/v3及显式safe profile，
复用原规划、留存操作、CAS和资源关闭owner；更新5项来源，新增15项，共60项源码、61项输入。
固定源码与567项通过、1项原RPent配对轮子缺失跳过、零失败的离线消费者报告绑定
`78b09d63ff9e1efc42b9ca7738a00d76107d8c68` / `b4c0572d4394ebbf385665875abd58379ff4dc51`。
冻结计划不能激活；查询保留原碰撞、质量、rig→planner锁序、绝对deadline和执行保护。
候选局部包络拒绝可继续比较后续候选；稠密查询只对逐种子诊断做内联摘要，保留实际残差、
覆盖、容差及拒因，不改变lower完整诊断、排名或认证。普通/观测点Route保留原范围。
V20、V19、V18及历史绑定逐字保留，已有segment v2/v3仍按显式配置启用。
本批不重拟合arm参数，`provisional`及新held-out `not_run`保留，不构成真机或模型能力验收。
历史V17、V16及更早绑定字节保持。

历史 V12 绑定为
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v12.json`。
V12直接以冻结V10为前驱，登记完整native chain、分阶段接触策略和prepared Skill消费口；
重绑planned TCP一项既有源码，新增显式motion_skill.py消费依赖，共21项源码、22项输入。
实现及304项消费者报告绑定 `73b40a4b177ffe59026b40ff193b8e4ef93b7b7d` /
`54b2d891fb66149da2453e727ef1d9ffe5e058d1`，不借用首次失败的完整验收或未提交工作树结果。
本批没有新公开fast Route，不重拟合参数，不增加物理抓取或arm-response声明；
`provisional` 与新held-out `not_run` 保持。独立B601候选V11未合入，不作为此前驱。

v10历史运行时源码绑定冻结于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v10.json`。
它在冻结 v9 的基础上登记原生物体跟随、携物几何及物理步后保护：重新绑定
executor、position rig 与 planned TCP 三项既有源码，并显式登记七项相关依赖。
实现来源是 `7d66cb491d44c891ef41678f066b80d072f87315`；本次夹具与元数据
迁移不改变所绑定的运行时代码。所引用的 115 项消费者回归确实运行于该提交及
`c5b5fa16e7cdd202e30990e4b9c3c6dd086d9166` 源码树，未借用 v9 的旧报告。
首轮完整验收的 11 项夹具／绑定失败以及旧课程正向实验的会话构造拒绝均另行保留，
不将其记为通过；修复后的完整验收也单独记录。该绑定不提供完整抓放或真机验收，
未重拟合参数，`provisional` 与新 held-out `not_run` 保持不变。
v9 及全部更早清单按原字节保留。

v9 历史运行时源码绑定冻结于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261022_v9.json`。
它在冻结 v8 的基础上，登记固定左右单臂 Skill 会话的 planned-motion 更新，
并补充 `agent_skill_target.py` 依赖；会话只选一臂，另一臂保留实测碰撞支持，
默认仍为右臂。左臂抓取、夹爪、历史、空间推理及模型组合仍拒绝启用。
源码修订为 `e4ab4e5d0580dd4cae3d2ce8aa783979025e5c0c`，这是添加本清单之前
的实现提交；聚焦测试实际运行于 HEAD `11c34162e` 加 MERGE_HEAD `71faf8b1d`，
其暂存树与 e4 的 `84cc94c8` 完全相同，并非提交 e4 后重跑。
本清单、说明和回归迁移不改动所绑定的运行时代码。
v9 按精确哈希引用 78 模块、1893 通过／6 跳过的聚焦离线报告及其源码树绑定，
不把可选依赖跳过计为通过，也不把 v8 的 401 项报告解释成新实现的证据。
首轮完整验收发现旧绑定未更新，此问题和修复后的完整重跑另行留存于阶段验收记录。
本次没有重拟合、没有新增真机或 held-out 证据；`provisional` 与 held-out
`not_run` 保持不变。v8、v7、v5 和 v4 拟合清单均按原字节保留。

v8 历史运行时源码绑定冻结于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261018_v8.json`。
它绑定当时 actuator、MuJoCo executor、最终命令准入、rig／assembly，以及显式
目标接触的 oracle、projection、result、binding 和 planned-motion 源码，同时保留
原 simulation contract、相机登记、冻结 v4 拟合及 v7／v5 运行时绑定的精确身份。
本次相对 v7 改动两个既有源码文件，并新增五项接触路径源码依赖；没有调整执行器
参数、重新拟合或改写历史清单。14 模块的离线报告按内容哈希登记，包含原生接触、
撤销临时 guard 后的实测复核及只读候选预览；它不证明真机抓取、制动或 swept-TCP
安全。`provisional` 及 held-out `not_run` 保持不变，完整组合验收单独记录。

v7 仍冻结保留于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261017_v7.json`。
它记录当时最终 TCP command admission 的源码身份；其哈希不随新接触 guard 更新。

The v5 runtime-source binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260922_v5.json`.
It records the executor change that
excludes the expected B601 gripper self-contact pair from collision admission,
without rewriting the old fit evidence. This binding explicitly records no
arm-profile refit and carries forward the `provisional` claim and held-out
`not_run` status; the collision exclusion is a simulation runtime behavior
change, not new physical-response evidence.

The 20260829.v4 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260829_v4.json`.
It records the exact execution-continuity and lifecycle sources before the
B601 self-contact exclusion and retains the preceding 20260822.v3 binding.

The 20260822.v3 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260822_v3.json`.
It records the corrected left/right wrist-camera serial assignment, the 4B2 CAD
scene contract, and the exact archived 20260724.v1 registry. It is a historical
runtime contract rebind, not a refit or a new fidelity result.

The historical v1 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260724_v1.json`.
Its exact camera-registry payload is retained at
`artifacts/rm65_camera_registry_runtime_20260724_v1.json`; consumers must not
compare that historical binding against the mutable current registry path.

### DM-Tac position baseline: `20260722.v1`

The bilateral suite `dmtac_bimanual_position_20260722_184856_698964` completed
all six runs with settings restored and no trace-queue drops. The fit uses 89
free-motion segments after two robust left-side RMSE outliers were removed;
44 separate held-out segments evaluate left/right close/open independently.
Held-out median normalized-position RMSE ranges from 0.0110 to 0.0189 and the
worst segment RMSE is 0.0313.

The frozen runtime artifact is
`artifacts/dmtac_gripper_position_20260722_v1.json`. It records the original
remote fit artifact and trace hashes, the 20 Hz / 0.002 software gate, all four
directional parameter sets, and the validation metrics. MuJoCo delegates the
normalized command path to `sim.rm65.actuation.DmtacPositionActuator`; only the
finger-servo/contact binding remains backend-specific.

The 0.5 s fast-preemption probe is retained as `evaluated_not_identified`:
endpoint median absolute error is small, but long reversing strokes reach
0.123--0.136 maximum normalized error under the zero-initial-rate model. Force,
contact compliance, friction, slip, and load dependence remain unmodeled because
the collection had no external force gauge.

## Package placement

RM65-specific fidelity code and evidence live at the RM65 embodiment level:

- `sim/rm65/actuation/profile.py`: versioned response parameters;
- `sim/rm65/actuation/core.py`: backend-independent accepted-qcmd, transport,
  raw-target, and filtered-target state machine;
- `sim/rm65/real2sim/`: evidence contracts, fit source indexes, and manifests;
- `sim/mujoco/actuation/executor.py`: thin MuJoCo actuator/measurement adapter.

The former `sim.real2sim` compatibility package has been removed. Import the
canonical `sim.rm65.real2sim` and `sim.rm65.actuation` packages directly.

## Sim-scoped execution plan

### P0: establish truth and auditability

- Use `Real2SimArtifactManifest` for every fidelity baseline. Bind the exact
  `SimulationContract`, task/profile inputs, source traces, calibration files,
  code revision, environment, resolved clocks, claims, and validations.
- Keep formal measurements layered as accepted qcmd -> actuator target ->
  measured qpos. Never compute tracking metrics from accepted qcmd alone.
- Freeze one deterministic probe suite under `sim/`: arm steps/chirps, gripper
  open/close, cube drop, contact onset, and camera calibration-board renders.
- Produce a machine-readable comparison report with per-side/per-joint metrics.

Exit criterion: a run can be reproduced from hashes, and every dashboard value
can be traced to either a real source artifact or a measured sim state.

### P1: identify the dynamics that dominate closed-loop error

- Fit left/right arm delay, gain/damping, force limits, rate limits, saturation,
  and noise from existing safe real traces. Prefer distributions over a single
  fixed delay when the data supports them.
- Fit grippers independently per side; separate free-motion response from
  contact response. Do not reuse the left trace as right-side evidence.
- Replace task-level contact latch results with physical contact/grasp metrics
  for formal evaluation. Retain latch only under an explicit `debug_only` mode.
- Identify object mass/inertia, friction, compliance, and reset distributions
  from measured task objects rather than reward tuning.
- Match the actual camera intrinsics/extrinsics, frame timestamps, sampling
  rate, latency, and dropout before tuning appearance randomization.

Exit criterion: held-out real traces meet declared thresholds for response,
contact, and camera geometry without fitting on the validation episode.

### P2: observation fidelity and robust policy training

- Add measured photometric distributions: lighting, exposure, white balance,
  depth holes/noise, motion blur, and occlusion.
- Add a DM-Tac observation adapter only after a defensible sensor model and
  time alignment exist; otherwise report tactile as unavailable.
- Derive domain-randomization ranges from residual real-vs-sim errors. Keep the
  identified nominal model fixed and version randomization separately.
- Run paired open-loop probes first, then closed-loop policy evaluation with
  identical action/observation contracts and causal cuts.

Exit criterion: robustness gains hold on held-out real episodes, and failures
can be assigned to a fidelity component rather than hidden by aggregate task
success.

## Minimum validation report

Each fit should report at least:

- timing: median/p95 command-to-motion delay, jitter, effective sample rate,
  dropout, and timestamp phase error;
- arm: per-joint RMSE/max error, settling time, overshoot, velocity/acceleration
  error, and saturation rate for left and right separately;
- gripper/contact: close-response error, contact-onset time, normal force/impulse
  proxy, slip distance, release behavior, and success confusion matrix;
- camera: reprojection error, pose error, frame-age distribution, crop/mask
  agreement, and depth validity statistics;
- task: state-transition and failure-mode confusion matrices in addition to
  reward/success rate.

Fit and held-out reports must reference different source artifact IDs. A failed
or not-yet-run validation remains part of the manifest; it must not be omitted.

## Artifact status rules

- `identified`: backed by referenced source artifacts; use only after a fit.
- `provisional`: physically plausible plumbing or partial evidence; limitations
  are mandatory.
- `debug_only`: useful for pipeline development but not a fidelity claim.
- `unmodeled`: the real phenomenon matters but the sim intentionally omits it.
- `unavailable`: the needed real observation/evidence is not currently present.

`Real2SimArtifactManifest.artifact_sha256` hashes canonical JSON excluding the
hash field itself. Loading a persisted manifest verifies that hash. Evidence
references must resolve to content-addressed `ArtifactRef` entries, while all
resolved timing rates are embedded so changes in code defaults cannot silently
change the interpretation of an old experiment.

## Ownership guardrails

RM65 profiles, reusable target dynamics, and evidence belong under `sim/rm65/`;
backend binding belongs under `sim/mujoco/`, and task selection belongs in sim
task configs. Tests may live under `tests/`, but should exercise only sim-owned
surfaces. Real logs and calibration outputs are read-only evidence. Changes to
`robot_motion/`, `teleop_runtime/`, `dataset_contracts/`, hardware behavior, or
canonical entrypoints require a separate mainline decision and are outside this
plan.

历史 V13 以冻结 V12 为前驱，仅重绑 canonical collision projector 的安全准入查询范围；
诊断与单调撤退保留原报告范围，原硬穿透、接触和最终输出门禁保持。22 项输入／21 项源码
绑定固定提交 `b74fd69c52ab4bd107fd3860dd8c253f68cfc3b3`；本地消费者报告为 285 passed，
不修改旧拟合参数或历史证据。新 held-out 仍为 not_run，fidelity 仍为 provisional。
70 mm CAD 的 4000 Hz 配置仅用于该仿真接触任务，不代表真实系统响应重新标定。

当前 V14 直接以前驱 V13 的冻结字节为基础，只重绑 oracle.py 中只读 forbidden geometry-pair
清单接口；22 输入／21 源码仅该项改变，没有新增输入。原碰撞准入、诊断、撤退与最终输出
门禁以及 RM65 响应配置保持；绑定固定提交 `b978c73d386bab608a0febb5ba1ba51e9cafded6`，原生离线消费者 77 passed。
V11 为独立 B601 候选的历史绑定，不作为当前绑定；V12/V13 及旧 fit 均保留原字节。
未重拟合响应参数，fidelity 仍 provisional，新 held-out 仍 not_run，不代表设备或抓取验收。
