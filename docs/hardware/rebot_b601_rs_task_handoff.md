# B601-RS 硬件开发交接说明

## 当前硬件开发线（2026-10-18）

- 后续候选以本地 `development@6e62032688b1c366ce87c83092758b0adfa41bcb` 为基线，
  工作区 `Developing/worktrees/b601-hardware-tuning-20261018`，分支
  `codex/b601-hardware-tuning-20261018`。下方旧版路径／状态是历史记录，不是新启动入口。
- 旧工作区 `Developing/worktrees/root-b601-staged-assist-20260923` 的 31 项未提交内容
  已完整保存在本地 `62959784f`，不删除旧分支或运行资料。新线仅移植实际缺失的硬件
  commissioning、GUI、标定 profile、轨迹 pacing 及只读碰撞审计访问器；不整支合并旧线。
- 最小共享接口为 `B601SharedControlRuntime.sample_policy(now_sec=...)`，独立提交
  `cf8dfd782`。它只读既有 command buffer 的有效性，不推进 servo 状态、不续期过期目标。
  迁移提交 `f9c98c9bb` 不包含 replay／RM65 应用改造。后续修复提交及验证结果随交付提供。
- 相对统一基线新增的 owner/probe 重力默认值沿用旧硬件线已确认组：全局 1，J2 scale
  0.96，bias `0 -0.22 -0.11 -0.08 0 0`，Coulomb 全零；不改共享控制器／仿真默认重力。
  本轮控制修复本身不调整这组数值、硬保护、碰撞模型或公开命令单位。
- 新基线只在本地完成合并；本硬件候选没有合入 `development` 或推送。
  基线中已接入的自观测过滤、夹爪等扩展保持原显式启用方式，默认 15 mm／v1／严格碰撞
  行为不随硬件迁移切换。物理验证仍由操作者运行，不把离线回归当成硬件有效性证明。

### 有运动边界但还没有稳定锚点时继续有界中间试验（2026-10-21）

基于 `38a6541575b2d2c4432f5cb9658b8d535dc71e9f`，远端原始
`near-zero-holding-margin-acceptance/b601-owner-01s9tb_9` 已完成 J2 12/12、J3 4/12，
共 55 条定力静止响应。J3 第一段实耗约 70.747 N·m·s，超过上轮旧 46.8 预算，
说明 120 的新预算确实生效且取得了新增数据，不是配置没有传入。

第 5 个 J3 点只有一次负向运动（约 −0.885539 N·m）和一次零附加命令的
`censored_settled_outside_point`，就在 2/27 次尝试、14.605/120 N·m·s 时结束。
原 `record_transition` 仅允许“已有稳定锚点”或“还没有任何有效响应”的续接，漏掉
“已有运动响应但还没有静止响应”的正常搜索阶段，错误提前给出
`holding_transition_without_stable_anchor`。

仅对显式 coordinated 的 interior-recheck 搜索，使用已有真实运动命令与当前
被删失命令之间的中点提出下一次试验。本记录的下一候选为约 −0.442770 N·m，
先重新到达冻结实测位置，再经过原精确派发平台、位置稳定观察及重复确认。
被删失的零命令不进入正向运动边界、不进入静止样本，也不成为可用保持力。
原始观测及删失记录全部保留；无有效稳定响应仍不能完成，分辨率收敛、试次、
时间、暴露和最终合成限幅仍可终止，拒绝过的稳定锚点与矛盾边界不被复活。
未显式启用的旧搜索行为保持不变，不自动应用重力或摩擦模型。

同一两条实测记录对照到首个决定：旧版 incomplete，新版 searching 并提出
−0.442770 N·m；不是已经测得该保持力，也不推演后续物理响应。两方向、默认行为、
新试验后的真实重复保持要求和不续期预算有独立离线覆盖。原始 16 段数据不因后续
采集或停机问题失效，运行摩擦/唯一重力残差的拟合就绪仍应分别判断。

### 名义到位与实测保持分离，并统一联合采集预算（2026-10-21）

本增量基于 `827b464dc042105fe5ca524c0e11fc45fee89cff`。原始远端
`near-zero-holding-dwell-acceptance/b601-owner-zvmy0ks0` 已完成 J2 全部 12 段局部
采集、44 条定力静止响应和一次联合位姿观察。它们是同一近零中心附近的重复局部
试验，不是 12 个全工作区位姿，也不是独立测力或唯一重力残差。J3 尚无完整保持点；
整轮仍是部分采集，已完成的数据保留，不因后续退出而丢弃。

首个 J3 点在重新到位阶段运行约 48.436 秒，其中 87.9% 的帧触及联合 L1 限幅。
停止前附加命令为 J2 +0.15295、J3 −0.84705 N·m，J3 目标误差约 0.00251056 rad，
比原入口 0.00225 rad 多约 0.00026056 rad，不到一个位置计数。六轴已静止，
但还未开始任何定力试次，命令暴露先到 46.7784/46.8 N·m·s。
这是入口精度与有限驱动能力/预算不协调，不是定力观察再次重置，也不是 CAN
掉线或 95 Hz 验收失败（三项频率约 99.06 Hz、均通过）。采集停止后的 J4 加速度
隔离事件不作为这一停止的起因。

- 仅显式 coordinated workspace 增加 `holding_arrival_tolerance_rad`，本 profile
  设为 0.003375 rad（原入口扩大 50%，约 0.193°）。未配置仍沿用 movement/2；
  必须有联合采集配置，且不超过原 0.0045 rad 点邻域和 native 到位包络。
  native 到位、采样入口、静态观察与 GUI 容差使用同一值，避免内部又有更紧的隐藏入口。
- 到位只决定允许在哪里取样。随后仍冻结完整实测位姿，并记录名义目标；定力平台
  的被测轴细稳定跨度仍是 0.0008 rad，原 1.5 秒观察、全轴 0.0045 rad 上下文、
  支撑命令 0.025 N·m 变化限制及完整回执全部不变，不用放宽入口冒充静止。
- 单点时限 114.075→180 秒，暴露 46.8→120 N·m·s；每关节整组时限
  760.5→1200 秒，暴露 156→480 N·m·s。整工作区仍为 3120 秒/1040 N·m·s，
  覆盖两组各 480 加定位/返回余量。支撑轴、被测轴与最终尾量仍按实际完整派发累加，
  预留最坏派发额度、不自动重置/续期；超限仍撤力停止，不自动重试。
- 重力参数、Kd、联合 L1 1.0 N·m、硬速度、位移、watchdog、TTL、碰撞、GUI
  撤销和最终人工确认失能均未修改。固定位置/完整启动模式及框架默认值不推广这组
  实验参数，replay/RM65 的接口和消费者不变。

同一实测 J3 输入、同一完整回执的首个入口判定对照：旧入口在实际停止前一直未接纳；
新入口约 20.982 秒即可冻结实测点并提出首条保持命令，此时暴露约 19.297 N·m·s。
对照到首个改变的决定即止，不宣称后续机械运动或整轮硬件必然通过。离线测试另外
覆盖两轴正负误差、默认不变、细稳定观察不变、旧预算边界可继续、新预算/预留仍停止。
代码与验证按固定提交部署到独立硬件 clone；未合并主线、未操作 CAN、未自动应用模型。

### 定力复测使用细位移稳定段，不被全轴短窗反复清零（2026-10-21）

本增量基于 `3cb85cf44f7f2e195424013f9556226a0fabdd00`。远端
`near-zero-static-observation-acceptance/b601-owner-jydcvv6_` 已完成约 5.011 秒的
联合位姿观察，并完成第一个 J2 保持点的 7 条有效响应及同命令复验，保持命令
约 −0.014486 N·m；尚未完成整条 J2 采样序列或进入 J3 序列。报告仍为
`ACQUISITION_PARTIAL / auto_apply=false`，不是重力或运行摩擦模型已拟合。

第二个 J2 点在约 +0.106205 N·m 的同一命令上，发生 18 次
`censored_context_not_stationary`，第 19 次被本段 114.075 秒时限终止。各次六轴
位置跨度最多约 0.000767 rad（两个编码器计数），支撑命令跨度均未超过 0.025 N·m。
原判定同时要求被测轴细位置稳定段和六轴短窗速度连续静止，后者反复清空前者的
1.5 秒计时。命令暴露约 32.052/46.8 N·m·s，频率三项约 99.052 Hz 全部通过，
设备 cleanup/host release 完整，既不是预算耗尽，也不是本轮 CAN 断联。

修改仅在显式 coordinated workspace 的固定命令观察内：

- 仍先确认被测轴实际派发的精确命令平台、前序完整回执和后续反馈；不是把请求
  当成已发送。使用原有 `2 * position_resolution_rad`（本候选 0.0008 rad，约
  0.046°）的被测轴细位置跨度累计稳定段，不因六轴短窗速度抖动反复清零。
- 被测轴在细位置带外持续爬动仍重启稳定段，原持续位移/方向确认仍可判定运动；
  全六轴原采样点/累计位置带和其他关节命令跨度 ≤0.025 N·m 的约束均保留。
  支撑轴位置虽然很小、但其闭环命令变化过大的用例仍不能算合格保持。
- 观察完成时仍要求原有一次全轴新鲜停稳检查，确保下一步变速/阻尼交接可用。
  未配置位置分辨率时保留原连续静止逻辑，不推定未知精度。
- 新合格保持记录注明 `fixed_command_fine_position_dwell`、实际细位置带和交接
  确认；不重写历史观测，不把此次旧运行回溯标为完成。

同一实测平台输入的判定重放，旧版 5.860 秒超时，新版约 1.510 秒接纳一条观测；
该窗口被测轴跨度约 0.000383 rad，支撑命令跨度约 0.0151 N·m。重放只到首个
改变的判定，不推演后续物理运动。十种 40/100 Hz 有界抖动、两种支撑命令变化
拒绝及原爬动/停稳/上下文改变反例均覆盖。本轮不改变重力/Kd、总 L1、速度、
位移、TTL/watchdog、时限/暴露、碰撞、发送回执、GUI 撤销及最终人工失能确认。

### 联合静态观察按完整位置窗口完成，不因短窗口抖动清零（2026-10-20）

本增量基于 `3e64db48a6927cb0aa5edf6eb6e1c7f97235deba`。远端首次近零验收
`near-zero-acceptance/b601-owner-tmon55az` 的两个定位任务已结束，J2/J3 超时前
均约 3.01034°，目标均 3°；联合观察最后约 64 秒的位置跨度分别仅约 0.022°、
0.044°，但六轴短窗口静止门反复重置 5 秒计时，最长连续合格仅 2.451 秒，
观察任务在 114.075 秒停止。观察阶段联合助力峰值约 0.460/1.0 N·m，未饱和；
反馈/计算/发送循环均约 99.05 Hz，三项频率验收通过。不是发送频率、力矩上限
或预算耗尽。原始报告保留 `leg_timeout / NO_COMPLETED_TRIALS`，不改历史结论。

修改仅作用于显式 coordinated workspace 的 `observe / return_observe`：

- 保持 J2/J3 位于原目标带内，累计整个六轴位置观察窗口；窗口内微小变化不再
  因单次短窗口速度或跨度判定而清零。沿用已配置的 5 秒时长、全六轴最大跨度
  0.0045 rad（约 0.258°）及 J2/J3 目标误差 ±0.00225 rad（约 ±0.129°）。
- 真实原生制动、目标带越界、整个窗口任何轴跨度超界仍重新计时；不延长原
  单段时限、暴露或整批预算。全窗口范围约束继续拒绝低速累计漂移。
- 观察时长满足后，交接仍要求一次原有新鲜停稳检查，保留窗口等待，而不是
  要求从头再连续静止 5 秒。这样不绕过下一阶段参考速度/阻尼变更的停稳边界。
- 完成只在原完整六轴回执消费后提交，记录完整位置范围、完成判据和交接确认。
  GUI 显示观察秒数、最大跨度、等待交接或目标带原因，不再把观察过程仅显示成
  “等待六轴停稳”。结果仍是带附加命令的联合保持观察，不是零助力重力测量。

定位/返回交接、固定命令二分、准备、停止/失能仍走原静止门。重力/Kd、联合
L1≤1.0 N·m、硬速度 0.195 rad/s、实际软制动、位移、碰撞、反馈时基/新鲜度、
40 ms watchdog、权限、六轴回执及 GUI 最终人工失能确认均不变；不修改共享
runtime、replay/RM65 或厂商库。扩展为观察阶段的有界位置容忍，不是放宽硬超速。

### 连续定位与测点搜索分离，cleanup 后先保存结果（2026-10-20）

本增量基于 `95838772fddb8e7692524647cdc3157d240ff4fa`。真实记录
`hardware-tuning/j2-j3-clean-stop-acceptance/b601-owner-nuhnenxy` 已完成两个关节
各 12 个维持点；失败发生在返回 J2，而不是维持点采集。随后设备清理确认，
第二次 Ctrl-C 在 cleanup 后反复解码历史样本的汇总中中断，原目录因此缺少 owner JSON。

控制修复仅作用于显式 coordinated workspace：

- `position / return` 使用连续跟踪，保留当前受限支撑偏置；不再因“刚确认运动”
  自动提出减半命令。`sweep` 的重新接近／固定点搜索仍保留原二分提案与重复验证。
- 运动证据仍独立更新，原有确认方向后的快速正功尾量移除仍可工作；不是关闭
  运动检测或制动。真实速度制动及零命令回执后的有界恢复仍由原生控制负责。
- `CoordinatedPoseHolding.step` 新增可选 `point_search` 布尔参数，默认 true 保持
  原直接消费者语义；workspace 显式传入阶段选择，审计新增 `point_search_enabled`。
  参考滞后/提前停住异常现在也保留实际轴的原生 `fault_frame`，不再只有显示任务标签。

旧记录 J2 返回途中，上一完整附加命令约 −0.379231 N·m，在运动确认帧提出
−0.1896155 N·m 中点；J2 随后回退。参考滞后触发量约 0.0196473 rad，门为
0.0195 rad，超出约 0.76%；暴露约 7.63/46.8 N·m·s，L1 峰值约 0.831/1.0 N·m。
所以不是预算或力矩上限耗尽；继续放宽门不能修正连续途中错误卸载。
同输入分支回归验证新路径保留偏置，旧测点路径仍提出中点；这不是物理反事实预测。

结果可靠性修复：

- `run_probe` 新增可选 `post_cleanup_checkpoint` 回调，不改变已有调用方要求。
  仅自动采集在设备清理/host release 尝试完成后、任何样本解码统计前原子保存最小结果，
  包括真实 stop、cleanup 回执/错误、保护保持故障、计数、终态采集快照、频率告警。
  acceptance 为 `NOT_EVALUATED`，不提前宣称通过。
- cleanup 后汇总被 Ctrl-C 中断时，保留该结果并记录 `postprocessing=INTERRUPTED`，
  正常关闭 GUI/日志并返回 130；控制期没有检查点时的中断不被该路径吞掉。
  真实 fault/ERROR、导出失败或不完整清理不提升为成功。
- workspace 统计一次解码必要字段，后续评估共用精简列表；丰富原样本仍走既有无损归档，
  不把整个嵌套 GUI/搜索历史整体展开，也不反复为各个统计反序列化。
- canonical `save_automatic_report` 支持显式独立输出目录，保持默认路径兼容与排他发布。
  缺少终态记录的恢复报告保留 `terminal_record_missing`、首个 guard 和 cleanup 边界，
  不补造失能回执或把完成的局部点等同于整批成功。

旧运行已离线恢复到
`hardware-tuning/j2-j3-clean-stop-recovered-b8g5h6vd/automatic_calibration_report.json`，
原始四个文件恢复前后大小/SHA-256 相同，完整指纹见同目录 `recovery_evidence.md`。
保留 24 个已核验维持点：J2 六个非零点约 −0.079～−0.124 N·m，另六点为零；
J3 12 点均为零附加命令。仍为软件命令/当前基线，不是实测输出力、唯一重力或已拟合摩擦。
`ACQUISITION_PARTIAL / auto_apply=false`，运行摩擦合格稳态窗口仍为零。

本轮不改任何 profile、重力/Kd、总 L1≤1.0 N·m、CLI 硬速度 0.195 rad/s、
软速度、时限/暴露、几何/碰撞、Type-24 反馈时钟/TTL、40 ms watchdog、最终限幅、
完整回执或 GUI 最终人工失能确认。未连接 CAN、执行真机、复位接口、应用模型、合并或推送。
共享 runtime、replay/RM65 生命周期不改；同伴 `edecc9cccd83b86390367c34ec76bb362d6672d7`
只读检查干净。主线现为 `ad89186db0bcbceb808c4746c3d5e3b9e60fd336` 且有既有修改，
本轮未迁移基线或触碰主线。历史 owner/probe 重叠仍须固定提交集成验证。

验证：联合控制 **138 passed**；owner/probe 与 workspace **359 passed**；底层运动、
维持力搜索、session、cycle、子模块边界 **646 passed**；automatic calibration 与
runtime 边界 **265 passed**，以上文件组不重叠，共 **1408 passed**。前两组包括四种原生终止
原因/关节诊断、40/100 Hz 正负向与完整 24 点耦合模型、cleanup 后/前中断、导出错误、
无损样本与一次解码指标一致性、排他恢复报告和原记录不改写。首轮组合的一个旧结构
断言要求统计仍引用完整样本变量；现已更新并加强为“cleanup→检查点→精简统计”顺序，
不改变任何控制验收断言。上述模型不是实机辨识模型。
原完整参数无 CAN 预检结果：`hardware-tuning/j2-j3-systemic-preflight-g8urd2vf/preflight.json`；
通过 canonical owner 构造的交付命令预检：
`hardware-tuning/j2-j3-systemic-command-preflight-yl055eju/b601_owner.json`。
后者 motion、automatic/cycle/workspace、支撑/保护阻尼、breakaway 包络和最终力矩限幅
与前者逐项一致；均为 `can_opened=false / motor_command_sent=false`，不代表物理验收。
Ruff、`git diff --check` 和本地资源审计通过。实际约 39.94 Hz 接收节奏、物理返回收敛、
USB 稳定性和更大位姿范围的重力/摩擦辨识仍未由本轮离线验证证明。

### 正常第二次停止不再误报为执行失败（2026-10-20）

本增量基于 `a37493bfc9ba6bba542e4ad212251c42d0059d3c`，只修改 cleanup 后的自动采集
结果分类，不改保护保持、失能、GUI、控制循环、profile 或任何运行阈值。真实记录
`j2-j3-brake-context-acceptance/b601-owner-583q9z4i` 在操作者关闭/刷新/跳转页面后
按约定 `page_unload` 取消采集，随后终端两次停止并完整清理。原生控制无故障帧、
降级进入次数为零，却因正常 `explicit second stop request` 写在 `escalation_reason`
中，被汇总层误判为 `FAILED / exit 1`。

现在仅这一条已知正常停止说明不作为故障升级；真实降级进入次数、其他或未知升级
原因仍阻止正常停止分类。上游已经判定的 ERROR/FAILED、清理失败和采集 error 不提升。
保留原始停止说明及频率告警，未完成采集仍为 cancelled/partial，严格 acceptance 不改。
新增回归先在旧代码复现四个误判；真实结果仅在内存重建原有 cleanup 后、finalizer 前
的 `STOPPED_BY_OPERATOR`，新分类为 `STOPPED_WITH_WARNINGS / exit 0`，同时保留六个
J2 维持点、未完成第七段、三个频率告警及 acceptance FAIL。原始运行文件未重写。
验证命令 `bash scripts/run_pytest.sh -q tests/test_rebot_b601_low_gain_probe.py
tests/test_b601_automatic_calibration.py tests/test_runtime_domain_boundaries.py` 为 **571 passed**；
Ruff 与 `git diff --check` 通过。框架同伴 `edecc9cccd83b86390367c34ec76bb362d6672d7`
只读检查保持干净，本批三个文件没有并发修改；共享 runtime 和命令接口不改。
本增量不提供新的力控或硬件验收结论，不连接 CAN、不执行真机、不合并或推送。

### 正常到位与实际制动分开计时、及时移除同向驱动尾量（2026-10-20）

本增量基于 `1a17696dcdbeab57a0a932ac613c708393ae0981`，修复同一显式联合采集路径，
不再增加预算或放宽保护。真实记录 `j2-j3-midpoint-acceptance/b601-owner-facl464p`
保留 14 个局部维持点（J2 12、J3 2）；最后停在 J3 第三段重新接近，未开始该点
维持二分。该段仅使用 4.746/46.8 N·m·s、约 8.61/114.075 秒，不是暴露预算耗尽。
设备失能、临时上限恢复、主动上报撤销和本地资源释放均确认；原始记录不改。

已定位两个不同层次的问题：

- J2 在正常 `target_reached_braking` 中等待 J3，已累计约 8.563 秒；下一帧 J3
  窗口速度约 −0.0830 rad/s，触发 J2 的 `support_velocity_braking`，却继承正常
  到位计时，立即撞上 8.5683 秒制动时限。现在仅显式标定的正常到位首次转换到
  真实速度制动时重设制动时钟，并清除旧停稳驻留。后续速度制动原因变化不续期；
  人工撤销、许可失效、参考偏差及默认非标定路径不借此重置时限。新增
  `brake_started_at_sec`，联合审计保留 `motion_fault` 的实际关节与原生故障帧，
  不再只能看到当前被测 J3 的标签。
- 第二次中点提案时，上一份实际附加命令约 −0.8625 N·m，目标约 −0.4312 N·m，
  但普通 0.5 N·m/s 撤力使约 −0.7997 N·m 尾量持续到目标穿越之后；此时原生 PD
  已请求正向减速。现在确认净位移的运动状态、窗口速度和最新差分方向均同意
  “旧命令仍沿运动方向做正功”时，只移除这部分旧命令，使用既有原生制动 slew
  （当前 profile 为 10 N·m/s）。这不是提升加力速度：跨符号目标先撤至零，后续
  反向加力仍按原 1 N·m/s；普通静止撤力仍为 0.5 N·m/s。另一支撑轴、单计数抖动、
  相反原始/窗口方向和固定命令观察不使用该快速卸载。

同输入软件分支中，0.025 秒原先只能卸载约 0.0125 N·m，现在最多卸载 0.25 N·m；
这证明命令变化率修复，不是未来机械臂运动轨迹的实测预测。完整回执仍决定提案提交，
联合最终 L1≤1.0 N·m，软速度 0.07605、CLI 硬速度 0.195 rad/s、位移/碰撞、
40 ms watchdog、反馈时钟/TTL、Kd/重力和最终 GUI 人工失能确认均不变。
维持力仍须同点重新接近、双向运动边界与静止重复核验；中点提案不是实测重力。

验证：原生 motion、联合 holding、holding search、workspace、session 及 runtime
边界组合 **762 passed**；automatic calibration、cycle、staged console、owner/probe
与 shared runtime 消费者组合 **692 passed**。最终变更分支切片 **26 passed**
与前述集合重叠，不相加。包括 40/100 Hz 正负向、旧到位时长大于制动时限、
真实速度制动不续期、人工/许可/参考偏差终止不重置、单计数/相反方向不快速卸载、
先零后原速反向、普通支撑轴缓释及完整 24 点耦合模型。该模型不是识别出的实机。
新增计时回归在旧代码复现立即超时；Ruff 与 `git diff --check` 通过。

原完整参数无 CAN 预检通过，输出
`hardware-tuning/j2-j3-brake-context-preflight-lei8ye9f/b601_owner.json`，
`status=PASS / can_opened=false / motor_command_sent=false`。本轮没有连接 CAN 或执行真机。
交付命令经 canonical owner `build_probe_command` 构造并无 CAN 预检通过，输出
`hardware-tuning/j2-j3-brake-command-preflight-cqezdpci/b601_owner.json`；motion、automatic
calibration/cycle/mode/workspace、阻尼与 breakaway 包络和原完整命令的有效配置逐项一致。
实际约 39.94 Hz、真实多轴耦合与卸载后的收敛仍需下一轮硬件证据；不能据离线通过
宣布已完成重力/摩擦辨识或丝滑力控验收。

### 联合接近时的中点卸载提案与制动后恢复（2026-10-20）

本增量基于 `2c464d2b32b97ff8fe17e6283138477774531234`，修复显式联合采集的接近
控制，而不是修改 shared-control／replay 框架或以中点试探替代保护。原记录
`j2-j3-margin30-acceptance/b601-owner-fqpl1w8y` 已保留 14 个已验证局部维持点
（J2 12、J3 2）；J3 第三段 `cw4:joint3:sweep-3` 在重新接近 0.04235988 rad 时，
约 69 秒内出现 10 次联合制动，J3 约 0.857°..3.164° 往返，单段暴露
31.175/31.2 N·m·s 后停止。该段未进入维持二分，不能仅归因为预算偏小。
后续 operator_interrupt 是人工退出；设备清理与本地资源释放均完成。原始记录不改。

当前控制变化仅由 `CoordinatedPoseHolding` 及其 workspace 调用方持有：

- 真正发生同向主动运动后，不再按原 0.02 rad/s 低速范围继续积累偏置。
  使用现有因果窗口速度、已有起动位移及两计数以上净进展识别一次运动过程；
  单计数往返不触发。向零减半的是上一份完整派发的**附加命令**，不含重力基线。
  中点只是新提案，不是已验证维持力、唯一平衡点或正负运动边界；实际输出继续
  受既有 slew 与最终 L1 限制。若原生 PD 已有更强减速响应，则保留该响应，
  不能为了达到中点抵消制动。运动中暂停偏置积累，重新静止后才继续细化。
- 真正原生制动仍按原路径撤掉主动输出并保留阻尼，不在超速中维持半力观察。
  首次制动记住上一份完整派发的局部提案；只有被测轴自身速度制动、有解析位移
  且附加命令正在同向做功，才减半该轴。其他轴不是因为被测轴变化就直接丢失
  恢复起点。现有完整零回执、原生停稳及有效运行许可之后，才按原 slew 使用
  提案重新接近同一目标。提案随完整匹配派发提交；不匹配回执不能提交。
  新目标、人工撤销、许可失效或终止路径不能借此恢复旧驱动力；撤销立即作废
  提案，即使最终条件化仍有非零尾量、当帧不能匹配请求零值，也不得保留恢复记忆。
- 正常单轴固定命令试探不经过新接近卸载，另一轴继续受限姿态支撑。
  真正的同点定力二分仍复用 `HoldingForceSearch`，其相反运动边界、静止重复
  核验及删失数据规则不变；制动过程不计作静止观测，不产生虚构的力边界。
- workspace 将原生到位校正容差对齐到既有测量入口 `movement_rad/2`，本配置
  为 0.00225 rad，而不是在相距约 0.01352 rad 就进入到位相关行为后又要求上层
  精细重新接近。未放宽测量入口；默认构造仍沿用原生容差。仅新增一个内部可选
  构造参数，现有调用方兼容；不改变 owner、MIT、反馈或共享 runtime 接口。
- 一并修正饱和处理：真正制动中即使限幅，也不能因普通 anti-windup 回退而重新
  提交制动前的非零偏置。制动输出、变化率和保护时序均继续由原生路径约束。

按本轮批准，只将同一个显式 profile 的四项预算增加 50%：单段
31.2→46.8 N·m·s、76.05→114.075 s；单关节局部批次 104→156 N·m·s、
507→760.5 s。workspace 的 1040 N·m·s／3120 s 不变。换点、换候选及软制动
不重置或续期预算。1.0 N·m 联合 L1 包络、0.195 rad/s CLI 硬速度、0.07605 rad/s
软速度、0.0045 rad 测量带、1.5 秒驻留、Kd、重力、碰撞、反馈时钟/TTL、
40 ms watchdog 和最终人工确认失能均未放宽；旧固定点和单轴完整流程不改。

离线证据包括：J2/J3 正负向、40/100 Hz、单计数与反向负载、实际零尾量回执后恢复、
撤销和换目标、拒绝回执、强原生减速优先，以及完整 24 点耦合模型和报告保留。
整批模型三项历史断言原来强制交接幅值 >0.01 N·m，新控制在该模型的首个交接
只需约 0.003 N·m；改为断言**实际派发组合连续且非零**，不制造最小支撑命令。
另有原有强非零支撑、普通换轴及终止全零测试继续覆盖，未删除保护断言。
原记录 12585.719749760 的同输入分支重放：旧公式仍增加负向偏置，新提案将
−0.41944927 向 −0.20972464 N·m 回退；只证明该帧决策，不声称后续真机已收敛。

核心四文件及 package/runtime 边界组合 **358 passed**；容差输入与换目标检查在
随后 **9 passed** 的重叠切片中验证。自动标定、command calibration、owner/probe、
原生 joint motion 消费者组合（排除 hardware/slow_sim）为 **1012 passed**；最终
撤销尾量、饱和制动、实际回执和权限相关切片 **29 passed**。这些集合有重叠，
不能相加当作独立测试数量；耦合模型并非识别出的真实机械臂模型。
Ruff/diff 和原 fqpl1w8y 完整参数无 CAN 预检通过，输出
`hardware-tuning/j2-j3-midpoint-approach-preflight-o_w2_k11`，
`status=PASS / can_opened=false / motor_command_sent=false`。实际约 39.94 Hz、USB
链路复发及真实 J3 是否收敛仍待物理验证；不自动执行真机、不应用模型、不合并或推送。
框架线只读元数据刷新仍为干净的 `edecc9cccd83b86390367c34ec76bb362d6672d7`；
本批五文件无其当前新增路径重叠，不修改对方工作区或同步其控制循环。

### 结束阶段的设备确认与本地通信资源释放（2026-10-20）

本增量基于 `f89da4ef9873e5d41d34711ebf2a59e38d9002c9`，仅修复终止阶段，
不调整采集 profile、重力、Kd、速度/位移/力矩保护、watchdog、TTL 或 GUI 失能确认。
`j2-j3-margin30-acceptance/b601-owner-9ld3h0ej` 首次 active-report 写入已报
`Network is down`，随后撤销上报失败，旧 `close()` 因保留的设备状态锁存而拒绝关闭。
这不是程序结束时主动将 can0 DOWN 的证据；之前 `tmnlqa05` 的六轴失能、临时力矩
上限恢复及主动上报关闭均完整，无 cleanup error。单独的 USB 消失事件不能由此归因。

正常结束仍先执行原有保护保持/人工失能协议，再恢复临时上限、撤销主动上报。
之后正常 `close()` 保留原检查；设备清理未确认时，owner 改用新增终止接口
`B601RSBackend.release_transport()`，只关闭本进程资源，不新增电机指令，也不执行
`ip link down`、USB 重置、重新连接或自动恢复驱动。新增接口不用于运行中换连接。

核对 MotorBridge 源码发现，Python `Controller.close()` 只调用原生 controller free，
而独立 motor handles 保留 native Arc 引用；清空 Python 容器不是释放这些句柄的证明。
后端现在显式按 `Controller.close_bus()` → 每个 `Motor.close()` → `Controller.close()`
释放。现有自包含 `close()` 的测试/注入适配器保持兼容。释放失败保留原对象或未释放
句柄供显式重试，显式 False 也算失败；不丢失错误，不自动无界重试。

设备状态锁存仍保留：尚未确认失能、主动上报撤销或临时上限恢复的 backend 实例
不能再次连接。只释放 host transport 不代表机械臂失能，也不会将失败提升为 PASS。
正常及异常结果都增加 `cleanup`，分别记录设备阶段确认、transport release 方法/
完成与错误；原 `cleanup_receipt`、`cleanup_errors` 等正常结果字段保持兼容。
终端仅新增一条清理摘要；设备未确认时仍需操作者根据真实支撑和动力状态处置。

故障注入覆盖首次上报写入/回滚失败、六轴 disable 失败、bus/motor/controller 释放
异常与显式 False、部分资源重试及错误 JSON 持久化；全部不连接真实 CAN。
`tests/test_rebot_b601_low_gain_probe.py`、`tests/test_rebot_b601_hardware_backend.py`、
`tests/test_b601_workspace_calibration.py` 合计 **471 passed**；三个 package/runtime
边界测试文件合计 **46 passed**。修改的 owner/tests Ruff 通过；backend 的 33 项
历史 Ruff 诊断未新增；diff 检查通过。沿用 tmnlqa05 完整 child argv，仅替换输出并追加
`--preflight-only`，预检为 `hardware-tuning/j2-j3-terminal-cleanup-preflight-aldiqef5`，
`status=PASS / can_opened=false / motor_command_sent=false`。
这些是离线/故障注入证据，不是真机结束资源释放或 USB 链路稳定性验收；原真机命令
无须改参数。无 CAN 操作、物理运动、接口/电源重启、模型应用、合并或推送。
框架线元数据只读刷新至 `edecc9cccd83b86390367c34ec76bb362d6672d7`，当时工作区
干净；其本轮六个应用/说明/测试路径与这批五文件无重叠。历史 backend/owner 接口
重叠仍需固定提交集成，未访问/修改对方实现内容或替对方合并。

### 联合采集历史限制的显式 30% 余量（2026-10-20）

基于 `3e9e7b8a35acd246afecd4ecadaae0eb43c2b519`，按操作者本轮进一步放宽 30%
的要求，只修改 `b601_j2_j3_coordinated_holding_v1.json` 的 workspace 实际执行配置：

| 限制 | 原值 | 新值 |
| --- | ---: | ---: |
| 准备超时 | 13 s | 16.9 s |
| 单局部段时限 | 58.5 s | 76.05 s |
| 单局部段命令暴露 | 24 N·m·s | 31.2 N·m·s |
| 单关节局部采集批次时限 | 390 s | 507 s |
| 单关节局部采集批次命令暴露 | 80 N·m·s | 104 N·m·s |
| 整个多位姿流程时限 | 2400 s | 3120 s |
| 整个多位姿流程命令暴露 | 800 N·m·s | 1040 N·m·s |
| 定力观察超时 | 4.5 s | 5.85 s |
| 原生制动超时 | 6.591 s | 8.5683 s |
| 原生软速度上限（含连续限速） | 0.0585 rad/s | 0.07605 rad/s |

这不是将所有数值乘 1.3。共享 1.0 N·m 最终包络、0.195 rad/s CLI 硬速度、
40 ms watchdog、反馈/派发时钟与新鲜度、TTL、完整回执、碰撞、关节/采集几何范围、
最终人工确认失能均保持原样；重力和所有 Kd、PI/速度反馈增益与命令变化率不变。
连续静止驻留仍为 1.5 秒、测量位移带仍为 0.0045 rad、最大尝试仍为 27 次，
不把慢漂当静止，不将“响应不一致”降格为合格保持点。已经修复的归零回执冲突及
过去取消的恢复次数门不再增加补丁；约 40 Hz 的真实 Type-24 反馈也不靠改验收线掩盖。

实现复用上一增量的 workspace 参数覆盖，补齐软速度/制动时限及批次预算字段，
仅显式联合模式可设置。构造 workspace 时复用原 `B601AutomaticCalibrationController`
的连接前验证与 settings 解析，使配置传入真正的 `CoordinatedPoseHolding` 和 trackers，
而非只显示在配置文件里；未新增运行状态、重试策略或另一套校验。
固定位置和 supported-cycle 保持原共享基础 profile，原 JSON 文件未更改。
非法值、超过硬速度的软门、单段超过批次预算等仍拒绝，计量不因换试探而续期。

原生离线对照以 0.064 rad/s 作为声明反馈：旧软门制动，新软门不因此制动；
六种旧预算边界均可越过、新边界均停止。完整耦合模型及消费者回归：
`bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_holding_calibration.py tests/test_b601_workspace_calibration.py tests/test_b601_calibration_session.py --tb=short`
为 **297 passed**；`tests/test_control_station_boundaries.py`、
`tests/test_hardware_boundaries.py`、`tests/test_runtime_domain_boundaries.py`
合计 **46 passed**；Ruff、diff 与无 CAN 原参数预检通过。
预检为 `hardware-tuning/j2-j3-margin30-preflight-Edak6o/b601_owner.json`，
`status=PASS / can_opened=false / motor_command_sent=false`。
离线结果不能证明扩大后的真实速度/驻留行为已通过；不自动执行真机、不应用模型、
不修改 retained 记录、不合并/推送。框架线同期推进至 `84b0773e4`，当前修改路径与
其新提交不重叠，保留对方未提交内容，历史共享路径仍待固定提交集成时检查。

### J2 局部维持采集完成、J3 软制动响应传递与单段余量（2026-10-20）

本增量基于 `7e45ee05e3fb5eb2f818fddacd1d17ed6a7ba956`。记录
`j2-j3-brake-zero-acceptance/b601-owner-tmnlqa05` 已越过之前的制动归零冲突：
先完成 J3/J2 的 3° 定位和联合姿态观察，再完成首个中心附近的 J2 **12 个局部段**
维持值验证。报告为 `VERIFIED_COMMAND_HOLDS_AVAILABLE / verified_points=12`，
并非 12 个全局位姿；非零值约 −0.0805..−0.1020 N·m，部分中心回测为零。
这些是软件附加命令与编码器响应，`auto_apply=false / identifies_unique_gravity=false`，
不能直接当作纯重力误差、测力或已拟合摩擦。

新停采发生在首个 J3 局部段约 36.09 秒，原因 `leg_command_exposure_budget`。
本段接近 16 N·m·s 后因派发保留额度停止，整批约 68.63/800；随后
`operator_interrupt` 是人工停机，不是首个采集故障。六轴最终 safe_stop 完整、
无 cleanup error；实际控制/反馈仍约 39.952 Hz，不因退出码或本增量变为 100 Hz。

该段第一次定力试探 +0.4662366 N·m 后已有同向进展并触发原生软速度制动。
旧外层记录 `censored_native_brake` 后清空观察，却未通知保持搜索器；搜索仍为
`verify_arrival_command`、0 次观测/0 次转移，下次仍准备相同命令。
修复仅对**被测轴自身软速度制动且已达到原有半测量带位移**的响应传递删失记录，
使用前一份完整回执的实际附加命令，而不是当前撤力帧或未发送的请求。
搜索复用已有有界提案逻辑，先返回同点后改试新的命令；不新增状态、无期限重试、
预算续期或自动应用模型。支撑轴单独触发制动、计数量级抖动不构成被测轴力边界。

按已有 50% 放宽授权，仅显式联合保持 profile 的 `workspace` 增加两项覆盖：
`max_sweep_leg_sec=58.5`（原 39），
`max_sweep_leg_command_exposure_nm_sec=24`（原 16）。复用原配置验证，在连接前完成
校验；未启用联合模式不能设置这些覆盖。固定位置/完整单轴流程仍使用原 39 秒/16；
两轴共享 1.0 N·m、总预算、速度、位移、watchdog、重力、阻尼与人工确认失能均不改。
GUI 展示有效单段时限及实际累计暴露，切换试探命令不重置计量。

原 trace 第 9138 行的软制动分支同输入回放：旧搜索仍 +0.4662366，新搜索改为
+0.2331183 N·m；新记录 1 次删失转移、0 次已确认静止/运动观测。
前一实际派发为 +0.4662366，当前制动帧已降到 +0.2197580，不能混用。
原观察内上一帧进展 0.00383495 rad，制动反馈帧实际净进展 0.00613592 rad；
重放只验证该分支及候选传递，不推演此后的真实机械臂运动。
脚本保存在 `hardware-tuning/j2-j3-probe-feedback-preflight-L7ItAH/replay_censored_branch.py`。

验证：

```bash
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_holding_calibration.py tests/test_b601_workspace_calibration.py tests/test_b601_calibration_session.py --tb=short
# 281 passed；覆盖原生软制动、另一轴制动、有限新旧预算边界、GUI 及 40/100Hz 完整耦合模型。
bash scripts/run_pytest.sh -q tests/test_control_station_boundaries.py tests/test_hardware_boundaries.py tests/test_runtime_domain_boundaries.py --tb=short
# 46 passed；Ruff、JS syntax、diff 检查通过。
```

原参数加 `--preflight-only` 并复用 owner 的 MotorBridge 环境，PASS：
`hardware-tuning/j2-j3-probe-feedback-preflight-L7ItAH/b601_owner.json`，
`can_opened=false / motor_command_sent=false`，workspace 中有效覆盖为 58.5/24。
完整回归曾有一个新测试输入提前触发腕部上下文变化而非目标制动路径；改为另一受控轴
的原生软制动后针对用例及上述完整回归通过，未因此修改生产判定或降低断言。

原 J3 返回同点过程中仍出现约 1.01°..3.43° 的往返及反复制动；L1 峰值
约 0.951/1.0，并非最终力矩饱和。不得声称只扩大预算就能解决平顺性，也不能把新的
离线提案当作真机成功。下一轮重点是 J3 同点返回和新的试探驻留；已得到的 J2 记录
保持有效性分层，不因 J3 未完成而丢弃。没有连接 CAN、执行真机、修改原始记录、
应用补偿、合并或推送。框架线当前固定提交 `0dfbcc9c1` 的新增功能路径与本增量
不重叠；双方历史相对统一基线仍共享 GUI、交接文档及部分 owner/backend 路径，
后续固定提交集成时分别检查，不以分支 ancestry 判断是否已吸收，也未修改其工作区。

### 协同制动与普通到位力的归零回执冲突（2026-10-20）

本增量基于 `8f4b98e8d970c69184e9c5bacaaab4e9f6edd862`。记录
`j2-j3-holding-adaptive-acceptance/b601-owner-50txgbn2` 在 J3 完成定位后、首段
J2 定位时停止，约 25.233 秒报 `braking_timeout_assist_withdrawn`。没有进入保持
搜索、没有完成位姿驻留或保持标定点。两轴 L1 命令峰值约 0.687/1.0 N·m，
协调及最终力矩／变化率限幅未触发；最终六轴 safe_stop 完整、无 cleanup error。

J2 触发正常软速度制动后约 0.45 秒已请求零命令，但 J3 的原生阶段仍是
`target_reached_braking`。该名字虽含 braking，采集模式下实际仍输出位置到位力，
约 −0.02057 N·m。旧协同合成保留所有 brake 阶段的原生需求，导致这项非零命令
一直存在；J2 却要求完整六轴零命令回执才能恢复，最终耗尽原 6.591 秒制动时限。
这不是把时限加长就能消除的问题，也不是硬速度或力矩上限触发。

修复仅在已经存在的 coordinated braking 分支内，把另一轴的普通
`target_reached_braking` 当作正常到位需求暂时退出，使用原制动阻尼／变化率撤掉
该位置项。真实制动器本身的需求照旧；完整零命令、六轴停稳和原生恢复门仍保留。
无协同制动时，正常到位非零维持力不变。重力、driver Kd、原目标、预算、profile、
watchdog、最终包络、共享控制和最终人工确认均不改。

同输入对照使用原 CLI/profile、截至原 guard 的反馈和合成完整回执：旧版本在原
第 1097 行、25.233 秒复现超时，J2 无恢复；候选约 19.199 秒恢复原 3° 目标，
回放至原故障行不再报错。此后物理运动没有被执行，不能把固定反馈当作真机反事实。
新增九个原生控制用例覆盖 J2/J3 互换、正负方向、最终调理尾量未归零时禁止恢复，
以及没有全局制动时仍保留普通到位力。
`bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py
tests/test_rebot_b601_joint_motion.py tests/test_b601_workspace_calibration.py --tb=short`
最终 **508 passed**；Ruff 和 diff 检查通过。原命令加 `--preflight-only`、使用 owner
依赖环境的预检 PASS，保存在
`hardware-tuning/j2-j3-brake-zero-preflight-DxrGY1/b601_owner.json`，
`can_opened=false / motor_command_sent=false`。
没有连接 CAN、启动真机、修改原始日志、应用模型、合并或推送；实际约 39.946 Hz
及既有几何／连续平滑性问题仍未解决。

### 同点保持复测、内部维持值与显式 50% 采集余量（2026-10-20）

本增量基于 `59dbd875e87aaa605f42d5072856efb3fb9bbb64`。记录
`j2-j3-support-recovery-acceptance/b601-owner-f0x4xla6` 已完成 J3 3°、J2 约 3.03°
定位及五秒联合观察；随后计划要求 J2 回移到约 2.427°，不是“始终上不去”。两轴
命令 L1 峰值约 0.656/1.0 N·m，协调和最终力矩／变化率限幅均未触发。
停采为 `holding_search_incomplete`：−0.14209 N·m 首次被判静止，复测正向移动。
首次约一秒内已净进展 0.00115 rad，位置跨度约 0.00153 rad；短时未超 0.003 rad
不能据此断定已可靠维持。同一点的历史、预载和支撑命令影响也不能被这次记录排除。

本轮仅显式 coordinated acquisition 启用以下行为，默认单轴搜索不变：

- 定力平台的观察时钟不重置；单独累计连续静止时间。超出正负一计数范围的进展或
  既有六轴 quiet 条件失效，仅重启静止驻留，不续期观察、段时限或命令暴露。
  相邻一计数跳动仍可通过；持续慢漂不能靠结束时一帧低速提前当作静止。
- 首次到点后冻结完整实测姿态，并把 J2/J3 保持参考一次性设置为该实测点。
  反复试探返回此点，不再在名义到位带的两侧来回比较。原局部计划另存
  `nominal_target_rad`，实际保持参考另存 `holding_reference_rad`；下一段仍按原计划。
  不改 J1/J4/J5/J6 的名义目标，不增加这些轴的位置控制；不在后续观察中重锚。
- 临界静止候选被实际运动否定时，排除该值及与已观察运动界矛盾的静止值，但保留
  全部原始记录。在同一尝试／段／命令预算内选择仍有观测依据的内部维持候选，
  必须完成新的精确重复检查；不能拿旧次数凑完成，也不把不一致的边界拟合成根。
  正负运动界交叉、没有剩余候选或预算用尽仍停止。
- 若首次到点驱动力尚未形成静止锚点，明确记录的删失位移只能引导新的有界提案：
  沿驱动力方向走过时试探其一半，反向回退时朝反方向有界扩展。仍先返回同点，
  不把变力过程或删失端点冒充固定命令的运动边界、保持值或真实力矩测量。
- GUI 显示当前局部计划角度、冻结实测点、内部重验及被排除的旧维持候选。
  正常采集回移不再仅显示“中心点 3°”，最终人工确认失能保持不变。

只修改 `b601_j2_j3_coordinated_holding_v1.json`：测量位移带 0.003→0.0045 rad，
因此初次定位到位带 0.0015→0.00225 rad；观察超时 3→4.5 秒，尝试数 18→27。
上述采集门均扩大 50%；名义观察时长 1→1.5 秒用于建立更可靠静止证据，不是放宽
数据有效性的要求。保留原 0..10° 采集范围、3° 中心、0.01 rad 局部半径、两轴共享
1.0 N·m、0.195 rad/s 硬速度、40 ms watchdog、TTL、碰撞及所有最终撤力边界。
不增加非约束性的力矩上限，也不因采集容差变宽而宣称位置／力测量精度提高。

原第三次 −0.14209 N·m 定力观测的同输入 41 帧复算，保留旧 1 秒／0.003 rad 参数
以隔离逻辑变化：旧记录为 stationary，候选仍在观察，连续静止仅约 0.801 秒。
原始六次响应镜像验证：矛盾 −0.14209 保留并排除，转向 −0.20665 的新重复检查；
新增的重复响应是声明输入，不是对未执行真机运动的预测。

扩大后的真实 profile 也纳入 40 Hz、腕轴偏移的完整 24 点耦合模型／报告回归。
中途反例暴露了首个维持锚点缺失及按名义点返回的基准问题，修复后保持原完成断言，
不删除失败记录或改成“失败也算通过”。模型不是已辨识 B601，不能代替物理验收。
最终验证（集合有重叠，不相加）：

```bash
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_holding_calibration.py tests/test_b601_workspace_calibration.py tests/test_b601_calibration_session.py --tb=short
# 260 passed，包含同点参考、慢漂/计数抖动、有限重验、GUI、完整 24 点及报告。
bash scripts/run_pytest.sh -q tests/test_control_station_boundaries.py tests/test_hardware_boundaries.py tests/test_runtime_domain_boundaries.py --tb=short
# 46 passed；Ruff、JS syntax、diff 检查也通过。
```

原命令加 `--preflight-only`，并复用 owner 的 MotorBridge 环境，结果为 PASS：
`hardware-tuning/j2-j3-holding-adaptive-preflight-fT5KnX/b601_owner.json`，
`can_opened=false / motor_command_sent=false`。不连接 CAN、不自动应用模型、不合并
或推送。实际约 39.95 Hz 与原几何警告仍未解决。

### 协同定位的支撑轴软制动配置传递（2026-10-20）

本增量基于 `38f80a52076260d4274c42c77d32b3f01cf3c512`。记录
`j2-j3-arrival-context-acceptance/b601-owner-mc44tra7` 在首段约 2.128 秒停止，
原因是 `coordinated_target_stopped_before_goal`，尚未进入保持搜索。J3 目标 3°，
实际停点约 0.989°；不能把本次记录当成完整标定点。六轴 safe_stop 回执完整。
两轴命令 L1 峰值约 0.523/1.0 N·m，协调及最终绝对／变化率限幅均未触发。

根因是配置遗漏，不是硬限幅：profile 已有
`workspace.positioning_support_soft_speed_recovery=true`，旧单轴 workspace 正确
传递，但协同路径的原生 tracker 使用了基础 motion 的 False。J3 先软制动，随后
J4 窗口速度约 0.0621 rad/s 超过既有软门 0.0585，停止原因为支撑轴制动；停稳后
tracker 未恢复原目标，捕获实际停点，上层正确拒绝这个未到目标的停点。

- 协同 `set_goal` 在既有已消费回执、六轴停稳的阶段交接处接收显式配置；定位及
  返回传入 workspace 的配置。扫描目标不传此参数，恢复基础 motion 原值 False。
  不改变原生 tracker、重力、硬保护、watchdog、范围、命令上限或任何数值预算。
- 复用原有 governor 的同目标恢复，仍须实际尾量归零、完整六轴发送确认和可信
  停稳。人工撤销、超时及真实故障不改成重试；保留错误停点的终止保护。
  audit 增加实际使用的 `support_soft_speed_recovery` 字段供核验。
- 固定本次反馈至第 180 行、使用原 CLI/profile 和合成完整回执：False 复现原错误；
  True 在同一帧进入 `speed_governor_resumed_original_target`，J3 目标仍是 3°。
  这只证明控制分支差异，不是恢复后的物理运动反事实或完整真机验收。

验证：`bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py
tests/test_rebot_b601_joint_motion.py tests/test_b601_workspace_calibration.py --tb=short`
最终 **496 passed**；边界三模块 **46 passed**。新增用例覆盖配置开关、阶段恢复／
扫描默认、尾量与回执门；完整耦合模型仍覆盖 40/100 Hz。首次回归有一处测试夹具
只改变 raw、未改变窗口速度，未能构造预期的“未停稳”；修正一致的移动输入后
重跑上述完整集合通过，未改停稳判定来迎合测试。Ruff 和 diff 检查通过。

连接前预检重用原始子命令及 owner 的 `_motorbridge_environment`，结果位于
`hardware-tuning/j2-j3-support-recovery-preflight/b601_owner-bound-environment.json`，
PASS，`can_opened=false / motor_command_sent=false`。首次直接重放子命令漏掉原 owner
依赖环境，报缺少 `state_observation_metadata_v1`；失败文件另存未覆盖，未改依赖或
绕过能力检查。实际约 39.94 Hz、近零几何警告及连续软制动的物理平滑性仍未解决。
没有执行 CAN／真机、自动应用模型、合并或推送；框架线文件保持不变。

### 协同到位、六轴停稳与实测采样基准分离（2026-10-20）

本轮基于 `c1966bad693ad23428cc7c2d84d7390f8836fffa`，保持硬件线独立工作区。
`j2-j3-coordinated-holding-acceptance/b601-owner-suwuriqm` 的首段 J3 已到约 3.01°，
但 J4 从约 0.747° 停到 1.121°，相对旧名义目标偏移约 0.006519 rad。旧 `_at_goal`
要求六轴都在名义目标 ±0.0015 rad 内，而实际位置反馈只控制 J2/J3；1,559 个定位
周期中 J2/J3 在带内且六轴停稳有 872 个，最长连续约 17 秒，六轴名义误差门却没有
一次通过，最终 39 秒 `leg_timeout`。本段两轴 L1 命令峰值约 0.610/1.0 N·m，
协调合成／owner 最终绝对与变化率限幅均未触发，不是靠增力或延长等待能解决的问题。

修复只作用于显式 coordinated workspace；不改原生 tracker、底层 owner、backend、
MotorBridge、profile、重力参数、共享控制或框架线：

- 到位仅比较 J2/J3 的受控目标，容差仍为既有 `movement_rad / 2`。六轴继续使用
  原因果窗口停稳条件；J1/J4/J5/J6 的原支撑漂移、位置、速度和反馈有效性保护不变。
  不增加这些轴的位置控制，也不把其名义目标改写成实测值来假装执行过目标。
- 定力搜索只在已消费完整发送回执、实际到位和六轴停稳后，冻结完整**实测**姿态
  `sample_pose_rad`；`nominal_target_rad` 另存。后续同点观测不重锚到新的被动轴姿态。
  受控的另一轴偏移仍按原路径有界重到位；未受控轴超出原测量带时保留已确认观测，
  本点按既有 `holding_search_incomplete` 告警撤力停采，不空耗尝试数或继续混合条件。
  这不是安全故障自动恢复；原段／整批时限、曝光和尝试预算均不续期。
- 普通驻留／位姿观察记录完整六轴窗口的起点、最小值和最大值。若慢漂在整段驻留
  中累计越过既有测量带，记录 `workspace_pose_observation_rejected`，不冒充静态样本；
  新驻留仍占同一段预算。联合维持观察仍不是纯重力观测或真实力传感器数据。
- GUI／结果新增有界 `workspace.arrival` 诊断，区分等待 J2/J3 到位与等待六轴停稳，
  显示受控误差和容差；名义被动轴偏差另存，不作到位门。保持上下文变化明确显示
  停采、保留数据、不自动重锚／重试。旧 profile／GUI 消费者无需新增输入。

固定原始反馈对照复用本轮 CLI/profile 与 J2/J3 原生控制器、六轴合成命令和合成完整
回执：旧实现消耗 1,596 帧后进入首段 `leg_timeout`；候选在 735 帧、约 17.472 秒
完成首段，转到原有 J2 3° 任务。比较止于首次任务交接；固定反馈与合成回执不构成
后续运动的反事实物理仿真或真机验收，原记录不修改、不重建成成功报告。

保留原 1.0 N·m **两轴共享**上限、0.195 rad/s 原始硬速度、40 ms watchdog、
0..10° 当前采集范围、支撑漂移、命令暴露、真实故障撤力和 GUI 最终人工确认失能。
约 39.947 Hz 实际反馈／派发以及上一轮偶发发送前 54.37 ms 延迟是独立未结项，
本修复不声称解决它们，不降频率验收线、不延长 watchdog，也不重复使用旧反馈。
正常软制动清除整组偏置的平滑性问题同样没有在本轮未经依据修改；当前修复先消除
定位已完成但上层永远不承认的合同错误，再以实际采集评估连续驱动效果。

定向验证（集合有重叠，不相加）：

```bash
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_holding_calibration.py tests/test_b601_calibration_session.py tests/test_b601_workspace_calibration.py tests/test_b601_calibration_cycle.py tests/test_b601_automatic_calibration.py --tb=short
# 547 passed；后补的受控支撑轴重到位用例在下面最终选集中执行。
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_workspace_calibration.py::test_workspace_gui_start_stop_and_background_semantics --tb=short
# 45 passed，包括最后新增的同点重到位与冻结基准不变用例。
bash scripts/run_pytest.sh -q tests/test_control_station_boundaries.py tests/test_hardware_boundaries.py tests/test_runtime_domain_boundaries.py --tb=short
# 46 passed
```

完整 24 点的声明式耦合模型覆盖 10/25 ms 周期及腕轴 0/0.006519 rad 偏移；原始
未受控轴漂移保护、六轴静止、完整回执、人工撤销、尾量、最终零命令和基础阻尼、
GUI 最终确认仍有正反例。四个被动轴采样中改变会保留先前观测后告警停采；受控的
另一轴则在同一冻结基准及原预算中重到位，不新增另一套复位或无限重试机制。
新完整启动参数的连接前预检通过，记录在
`hardware-tuning/j2-j3-arrival-contract-preflight/b601-owner-e9ex3cqa/b601_owner.json`，
`can_opened=false / motor_command_sent=false`。Ruff、JS syntax、diff 检查通过。
没有执行 CAN、真机运动、补偿模型自动应用、分支合并或推送；原近零几何间隙警告
仍未关闭，不将流程回归或连接前预检说成物理／碰撞验收。

### 协同保持首次真机记录后的衔接修复（2026-10-20）

本增量基于 `df5aa53d541a558b3b2b9605ef04690cce72d2ee`。首次记录
`j2-j3-coordinated-holding-acceptance/b601-owner-8cndxas0` 在首个 J3 3° 定位任务
停止：约 9.438 秒时参考偏差 0.019530 rad 越过 0.0195 rad 门，9.588 秒捕获约
1.406° 的实际停点，之后上层仍等 3°，直到 39 秒 `leg_timeout`。清理六轴 safe_stop
回执完整；随后离线报告因空身份／非运动拟合角色标记异常，将退出码升为 2。
没有合格保持采样点，不能将定位段当作已完成标定。

本轮不把真正的参考偏差故障改成自动重试，也不扩大任何阈值：

- 在显式协同模式中，已暂停参考的减速尾量若吃掉预留的一个编码器计数余量，提前
  请求现有 `reference_pacing_headroom_braking`。实际偏差仍在原门内时，完成原六轴
  零命令／静止确认后沿既有 governor 恢复同一目标；不调用 start 重置时限或计数。
  实际参考偏差已越界时仍是终止路径，不转成预测性恢复。
- 原生 tracker 新增默认 False 的 `calibration_target_hold`，只有协同采集显式传 True。
  合格软制动后，若原目标在原到位及参考偏差范围内，必须先有完整零命令与六轴
  停稳确认，才能保持原目标，而不是把当前停点冒充目标。进入后是有界主动
  `calibration_target_tracking`，不是六轴静止声明；允许另一轴在原软速度边界内受控
  移动，不再用静态停点的跨轴累计范围去反复撤力。真实到位、固定命令观察和最终
  撤力仍由原六轴窗口、驻留和回执判断；硬速度、反馈新鲜度、实际参考偏差始终检查。
- 完成上述停稳交接后清除已结束的制动锁存，保留目标与 governor 计数，下一次
  定命令探测可正常接入。人工撤销、许可撤回、硬保护和真实参考偏差停止不能恢复。
  真正参考偏差终止报 `coordinated_reference_lag_stop`；底层捕获非目标终止停点则报
  `coordinated_target_stopped_before_goal`，均进入原 owner 故障撤力路径，不再空等整段超时。
- 没有批次时 `calibration_trial=null`，有批次后试次／阶段身份非空。报告在校验运动
  拟合专用标记前，先执行既有保持采集排除规则；独立 admission 调用也保留该规则。
  这允许只读分析旧的空身份非拟合记录，不放宽正常运动试次身份或将保持数据纳入拟合。

对同一轮原始 1,594 个反馈样本做固定输入对照，基线出现 1,161 个 J3 错误停点保持
样本，候选为 0；候选仍保持原 3° 目标。此为输入重放／控制状态证据，不是反事实
物理轨迹预测，不能据此宣称真机已经到位。旧数据的报告可只读重建，仍显示
`leg_timeout`、`INSUFFICIENT_DATA`，不修改原 owner 结果、trace 或注释。

回归覆盖正负方向、10/25 ms 周期、量化静止、真实参考偏差不恢复、许可/人工/硬保护
停止、零回执前不得接管、近目标保持后可进入固定命令观察，以及非空真实 GUI 注释
下的完整 24 点报告。现有 profile、重力、总助力、速度、几何范围、段时限、预算、
watchdog 和最终 GUI 失能确认均不变；没有连接 CAN 或运行真机。框架默认调用不传
新增关键字，保留原行为；集成须同时带入 tracker 和唯一新调用方，不复制另一套控制环。

最终定向验证：

```bash
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_rebot_b601_joint_motion.py tests/test_b601_command_calibration.py tests/test_b601_holding_calibration.py tests/test_b601_calibration_session.py --tb=short
# 652 passed
bash scripts/run_pytest.sh -q tests/test_b601_workspace_calibration.py tests/test_b601_calibration_cycle.py tests/test_b601_automatic_calibration.py -k 'held_probe or result or gui or short_target or terminal or lifecycle or workspace' --tb=short
# 76 passed, 287 deselected
bash scripts/run_pytest.sh -q tests/test_control_station_boundaries.py tests/test_hardware_boundaries.py tests/test_runtime_domain_boundaries.py --tb=short
# 46 passed
```

开发中的一次扩展兼容回归为 503 passed / 1 failed：提前排除保持记录时，曾遗漏独立
`_admission_reason` 调用的保持拒绝合同。该实现遗漏已恢复，原失败用例在上述最终
652 项内复测通过；不修改该用例预期，也不将早期失败检查计为整套通过证据。

### J2/J3 协同保持、单轴定力观察候选（2026-10-19）

本增量基于 `82bead03e8dce586bb78524354fe117c78e75b5a`，只在显式
`b601_j2_j3_coordinated_holding_v1.json` 的 GUI「J2/J3 位姿采集」中启用。
旧 profile、固定位置模式、完整带启动模式、共享控制注入接口及 owner 执行循环不变。
新路径复用现有单关节参考/制动控制器、六轴因果窗口、HoldingForceSearch、六轴发送回执
和最终 owner；不是另一套 CAN 发送器，也不把标定流程扩展成 replay/RM65 框架。

需要修正的结构问题是：过去「另一个轴保持」只表示重力前馈和 Kd、Kp=0；重力模型
有残差时不能等同于位置不动。逐轴测试中强制归零又会移除刚找到的保持命令。本候选
把 J2/J3 的姿态目标、学习到的保持偏置和实际发送尾量作为同一组受限状态：

- 搬运和缓慢重新到点时，两轴均有低增益反馈与有界条件偏置；测某一轴时，该轴的
  **完整附加请求**被定力试探替代，另一轴继续保持姿态，不叠加第二份试探/积分。
- 原来的逐段快/慢参考速度进入实际 planner；速度切换只在已消费回执且六轴新鲜静止
  的交接处发生。正常点间、轴间交接保留实际保持命令，不先撤到零。
- 固定命令观察必须晚于实际完整派发，记录全六轴位置、命令范围和阻尼。其他轴
  位姿变化超出原 0.003 rad 测量带，或其保持命令仍明显变化，只能记作条件变化/未决，
  不能据此更新被测轴正负运动边界。实际平台可用于重复核验，但仍需原两次重复确认。
- 人工撤销、页面关闭、许可超时、故障或整批结束才撤掉全部自主保持/试探项。结束
  必须确认实际六轴零附加命令、基础 Kd 和新鲜静止；最终失能仍由 GUI 人工确认。
  人工介入若发生在计算与发送之间，未发送的非零帧同样被否决。

新 profile 保持中心 `J2=3° / J3=3°`、局部半径 `0.01 rad`、原 `0..10°` 采集范围；
不启用全角度网格。两轴最终软件附加命令满足同一个
`|u2|+|u3| <= 1.0 N·m` 上限，**不是两轴各自 1.0 N·m**；保持偏置也受该总上限
约束。正常增力 `1.0 N·m/s`、减力 `0.5 N·m/s`，实际发送后的尾量同样计入上限
和命令暴露。真正制动沿用现有制动控制，不以正常慢撤力替代故障处理。积分饱和或
下游实际命令不同于请求时不继续累积偏置。这里的上限约束的是软件前馈附加项，
不是包含驱动器内部阻尼的真实输出力矩，更不是人手受力测量。

速度、位移、支撑轴、反馈有效性、40 ms watchdog、原段/批时限及暴露预算均未放宽。
GUI 后台继续/关闭撤销合同不变；尚未开始的模式再次进入时只重新积累窗口，不将
离开期间的旧静止证据用于授权，运行中的 watchdog 仍严格执行。

报告新增的 `coordinated_pose_observations` 不是 `gravity_observations`。正常交接点
记录 `handoff_full_dispatch_confirmed=true`、`withdrawal_confirmed=false`，整批最终
撤力单独记录。采集值是**在记录姿态、进入方向和传动历史下可重复保持的命令区间**；
静摩擦/预载可能产生多个可保持值，不能证明其为唯一重力残差，也不自动更新模型。
后续需要相同完整姿态的双向到点、重复交叉验证，再分离可重复偏置和历史依赖区间。

离线证据覆盖 J2/J3 有姿态相关残差、静摩擦死区和非对角运动耦合的声明式简化模型，
在 40/100 Hz 均可完成 24 个保持采样段、非零轴间交接、回原位和最终撤力；该模型
没有用真机数据辨识。另覆盖缺失/重复/错误六轴回执、人工撤销、尾量、最终合成限幅、
模式切换和 watchdog。离线模型通过不等于已证明真机无回弹、保持稳定或完成标定。
既有 nominal 近零几何审计的约 0.773 mm 间隙警告仍未关闭，新保持修正造成的实际
双轴轨迹尚未验证；不能以旧的逐轴 nominal 审计宣称本候选碰撞验收通过。J1 Type-24
约 40 Hz 的既有实际节奏也未在本增量改变。没有连接 CAN、启动真机或应用补偿模型。

定向验证命令及结果（既有 dexteleop310 环境，不安装依赖）：

```bash
bash scripts/run_pytest.sh -q tests/test_b601_coordinated_holding.py tests/test_b601_holding_calibration.py tests/test_b601_calibration_session.py --tb=short
# 164 passed
bash scripts/run_pytest.sh -q tests/test_b601_workspace_calibration.py tests/test_b601_calibration_cycle.py tests/test_b601_automatic_calibration.py -k 'held_probe or result or gui or short_target or terminal or lifecycle or workspace' --tb=short
# 76 passed, 287 deselected
bash scripts/run_pytest.sh -q tests/test_control_station_boundaries.py tests/test_hardware_boundaries.py tests/test_runtime_domain_boundaries.py --tb=short
# 46 passed
```

Ruff、`node --check control_station/b601_torque_console/static/app.js`、`git diff --check`
均通过。新 profile 仍需绑定固定提交、按完整参数执行连接前预检；预检不是物理运动验收。

### Type-24 独立缓存修复（arrival-credit 真机记录之后）

`j2-j3-arrival-credit-acceptance/b601-owner-cr02hmu0` 的首次退出是 J2 原始速度
`+0.134238 rad/s > 0.13 rad/s`。此前 J3/J2 定位和约 5 秒零助力重力观察已完成；
第一段 J2 负向扫描在反向回弹中触发硬保护，不是到位、预算或日志持久化失败。
六轴最终 disable 回执完整，但该事实不能推广到其他未核实的运行。

该记录 1114 个 active 派发周期共有 14717 次反馈接纳拒绝；每周期首末原因均为
`joint1 latest status frame is communication type 2, not Type-24 active report`，
retry wait 中位数约 13.73 ms。MotorBridge 原先只有一个 Type-2/24 共用最新状态槽，
命令回复会覆盖已收到的主动报告。本轮为 MotorBridge 添加只读的
`active_report_observation_v1` / `robstride_get_active_report_observation()`，保留原
generic getter 的合同。Type-2 不得刷新独立 Type-24 槽的位置、时间或序号；停止上报
不能用新回复伪造新鲜度。

backend 的新增 `prefer_active_report=False` 保持普通 host-fresh 消费者不变；owner 的
active-report 路径显式选择 True，能力存在时读取独立槽，不存在时保留原 generic 路径和
原 Type-24 接纳门。已声明能力但 getter 缺失、异常、空缓存或过龄时不能 fallback。
六轴完整、Type-24、序号推进、接收年龄和原 watchdog 均不放宽。不修改上报寄存器、
不接受 Type-2 替代控制反馈，也不改变收到的速度字段仍仅用于诊断的约定。

这消除缓存覆盖问题，不等于已证明 100 Hz：该轮接纳的 J1 Type-24 时间间隔约 25 ms，
日志没有保存所有被覆盖的原帧；固件实际上报周期仍需下一轮证据确认。构建仅在本工作区
的 `hardware/MotorBridge/target/release` 生成，新真机命令必须绑定本工作区 MotorBridge，
不能继续绑定未修改的外部 checkout。父仓 gitlink 与子模块源码提交一起交付，不复制
二进制或改写 Python 环境；子模块提交未推送时，不能声称其他机器已经能够拉取。

### 连续模式反向回弹时的制动方向修复

同一记录在反弹初期出现 `raw_v=+0.058`、80 ms 窗口 `control_v=-0.023 rad/s`，原软件
制动仍输出约 `+0.046 N·m`，与最新位置进展同向。正常 TRACK 也曾在窗口尚为正、
原始进展已经为负时，将负向增推误判为耗散并使用快速制动 slew。

本轮只对显式连续模式修复两处：软件速度制动须窗口和最新配对位置差分同向，幅值受
两者较小绝对速度约束；不同向或最新速度为零则不产生该软件制动项，既有 motor-local
Kd 不变。TRACK/到位快速反向制动速率也要求两个方向证据一致，否则增推仅走正常
tracking slew。窗口仍用于起动/状态/停稳，raw 仍用于独立硬保护；不提高力矩、速度、
位移、预算或超时，不自动重试故障，不改变人工退出和 GUI 失能确认。离散时间上的
符号约束不等于独立测得的能量被动性证明；真机回弹是否消失仍待现场验证。

### 采集运行余量放宽 50%（type24-direction 真机记录之后）

`j2-j3-type24-direction-acceptance/b601-owner-89q710sv` 在前两段定位、首个 5 秒
重力观察完成后，仍在第一段 J2 扫描中因 `+0.134166 > 0.13 rad/s` 退出。原有反向
窗口错误助推已被消除；本轮反转时软件速度制动归零。但扫描中还出现两次
`reference_pacing_headroom_braking`，最后一次在到位带之前被速度软门切入撤力。
首个参考余量制动的参考偏差约 `-0.012527 rad`，计入现有反向进展预测后安全余量约
`-0.000866 rad`，因此属于有限运行包络触发，不是预算耗尽或零回执遗漏。

按本轮操作者授权，显式 local-identification / workspace / supported-cycle 候选做如下
50% 调整：六轴原始速度命令参数 `0.13 -> 0.195 rad/s`；局部连续采集软制动
`0.039 -> 0.0585 rad/s`；完整搬运软制动 `0.05 -> 0.075 rad/s`；局部参考跟踪余量
`0.013 -> 0.0195 rad`。新增可选 `motion_max_reference_lag_rad` 仅在显式连续采集
时接纳，且不得低于原 profile 下限或超出既有相对目标包络；通用 motion profile
保持 `0.013 rad`。它仅配置现有 planner/pacing/lag guard，不增加状态或恢复路径。
量化 direct 调试普通上限保持原值，只有 release/support-confirmed 采集允许显式请求
最高 `0.195 rad/s`。起动扫描自己独立的 `0.15 rad/s` 门不变。

扫描参考速度、到位容差、起动确认、停稳驻留、40 ms watchdog、反馈接纳、零派发回执、
1 N·m 总助力、命令暴露预算、硬位移及碰撞/工作区范围均不改。不扩大 J2/J3 当前
0..10 度采集网格，不自动应用重力/摩擦模型；故障仍无自动重试或回位，最终失能
仍需 GUI 人工确认支撑。放宽是显式配置变更，不是硬件力矩/能量安全证明。

Type-24 独立缓存的实测拒绝原因已经变为 J1 序号未推进，记录中首末拒绝均属此类，
实际派发约 `39.944 Hz`，J1 接纳的主动报告接收间隔中位数 `25.002 ms`。
本轮不再有 Type-2 覆盖拒绝，也不降低频率验收线或重复使用旧反馈冒充 100 Hz。
实际上报节奏是独立的待解决项，不能声称此次余量放宽修复了控制频率。

### 到位合成助力的快速恢复修复（margin50 抖动记录之后）

`j2-j3-margin50-acceptance/b601-owner-pj8ryb79` 已完成 J3/J2 定位及首个零助力
重力观察，但第一段 J2 扫描在到位稳定阶段因 `braking_timeout_assist_withdrawn`
退出，未完成 8 个关节采集任务。全段 J2 原始速度峰值约 `0.1023 rad/s`，没有触及
本轮 `0.195 rad/s` 硬速度门；到位阶段约 6.6 秒内存在往复进展，不应仅延长超时。
最终六轴 disable 回执完整；已完成的重力观察及加力记录保留，但没有合格的稳态运动
窗口，不能将整批数据称为已经足够拟合的重力／动摩擦模型。

到位阶段的 264 个 active 派发样本中有 42 次相邻周期附加助力变化超过 `0.1 N·m`。
例如同一方向请求从约 `-0.0076` 一周期恢复为 `-0.2167 N·m`，随后又受速度反馈
减力至零；下游 torque/slew limiter 没有造成这次跳变。该阶段的请求包含位置弹簧、
维持助力及速度反馈，旧实现却在请求暂时与运动反向时，允许整个合成请求使用
`10 N·m/s` 快速制动速率。合成请求反向不等于每个组成项都在耗散能量。

本轮仅修复显式连续模式的到位输出：合成助力幅值增加一律遵循原正常 tracking slew，
不再因暂时反向而快速恢复。局部采集在实测约 25 ms 周期下，原 `0.15 N·m/s`
限制每步增幅约 `0.00375 N·m`。同输入回归中上述恢复边界输出为 `-0.01135 N·m`，
旧方法在同一用例失败。此为软件输出边界验证，不是对真实机械轨迹的反事实预测。
快速撤力、真正速度制动、原有过零回执、最终撤力及 motor-local Kd 均保持原语义；
不增加状态、参数或重试，不再放宽门限、预算和时限，不改变重力／摩擦配置。

离线覆盖包含正负方向、原始与窗口方向一致／不一致、10/25 ms 周期、到位与真正
速度制动的区别，并增加现有量化／静摩擦合成模型的 40 Hz 控制周期验证。合成模型
并非辨识出的 B601，不能据此声称硬件抖动消失。实际派发仍约 `39.946 Hz` 而非
请求的 100 Hz，上报节奏和真机稳定性仍需分别验证；本轮没有连接 CAN 或运行真机。

### 40 Hz 的进一步定位与下一轮证据（2026-10-19）

同一 margin50 记录的 1401 个 active 周期中，反馈重试的首末拒绝原因全部为 J1
序号未推进。J1 已接纳帧的主机接收重建间隔中位数 `25.0013 ms`、P95
`25.2529 ms`，接纳时年龄中位数仅 `0.4302 ms`。循环的调度等待中位数
`7.7321 ms`、反馈读取和接纳 `14.8792 ms`；六轴派发仅 `0.9109 ms`，重力和
主动力矩计算 `0.1526 ms`。这些分项中位数不能直接相加当作某一个完整周期，
但表明主要等待发生在下一份六轴合格反馈，而不是六轴发送或重力计算。

当前 owner 只调用开启 Type-24 上报，不写 `EPScan_time (0x7026)`。不能从约 25 ms
间隔直接推断寄存器当前值，也不能未经核实写该寄存器。Type-24 独立缓存的
`status_sequence` 仍采用 Type-2/24 共用单调序号：Type-2 不刷新该缓存，但两次
Type-24 之间可能跨过 Type-2 的序号。因此序号差只能证明推进，不能直接换算成
Type-24 帧数或上报 Hz；保留既有接纳语义，不改成重复使用旧 J1 来凑 100 Hz。

下一轮在现有近零位姿候选运行时，复用 `rebot_b601_rs_frequency_probe.py --mode
read-only` 被动抓取 30 秒原始 CAN：分别比较 device 1..6 的 Type-24 间隔、Type-2
间隔及 owner 接纳时刻。被动探针不构造 MotorBridge、不启停上报、不发送电机命令，
也不能单独启动数据流；由原 owner 继续唯一拥有控制和上报生命周期。输出使用新的
运行目录，原始 JSONL 用独占创建，不覆盖旧记录。此步骤用于区分总线可见帧节奏和
软件接纳等待；尚不能证明电机内部采样时间。未拿到原始帧前不修改反馈门、调度规则、
固件参数或频率验收标准。本轮只读复核及离线探针测试不等于真机频率恢复。

### 正常到位减力与安全撤力分离（arrival-slew-rx 回弹记录之后）

`j2-j3-arrival-slew-rx-cEYvZl/b601-owner-6kxiophs` 正常退出且六轴最终停止回执完整，
但采集因 `leg_timeout` 停在第一段 J2 扫描，完成数仍为 0/8。目标始终约 `0.873°`，
五次到位制动发生于约 `1.54..1.58°`，之后约 0.65 秒回到 `2.81..2.88°`。没有新反向
目标，也没有启动正常最终卸力：五次回弹前的 `arrival_unloading` 均为 false。
第一处边界的附加助力约 `-0.35329 -> 0 N·m`，同期重力项只增加约 `0.0056 N·m`。
这支持正常到位撤空助力是直接触发因素，不证明重力模型准确，也不能将重力残差与
减速器／结构弹性分离。纯重力短时静止可能仍由静摩擦承载，不能当作独立测力。

旧逻辑把内到位带同时用作维持控制的零速度终点，位置误差尚在就产生整股减力；
速度修正又可触发 fast-tail removal，绕过慢恢复限制。上一轮只限制恢复速度，
没有解决撤空发生得过早，不能将其离线通过解读为回弹已修复。

本次在显式连续采集模式内统一 TRACK／到位的有界速度需求：减速距离相对于实际
目标，继续复用既有 Ki/Kv 位置到速度关系、参考速度和加速度上界，不在内到位带
突设零速度需求。到位阶段的合成助力增加仍走普通 tracking slew，减少复用现有
`arrival_unload_slew_nm_s=0.5`；不再使用正常行进的快速尾量撤除通道。真正速度制动、
人工撤销、许可丢失、硬速度或反馈故障已经离开正常到位，保持原有快速停止语义。
没有新增状态、重试或参数，非连续路径保持旧行为。最终卸力仍等待停稳驻留，并且
需要零派发后的新停稳证据；卸力后离开目标带不能报告成功，不重置原制动时限。

同输入软件边界测试（正负方向）中，原来约 25 ms 撤空的请求改为从 `0.35329` 减至
`0.34079 N·m`；PI 合成、正常到位与真正速度制动分别覆盖。另覆盖目标内速度需求
逐渐归零、下游限幅回执、最终卸力后漂离和权限撤销。该数值是输出边界结果，不是
实际物理运动预测；量化／摩擦及残差合成模型也不是辨识出的 B601。

本轮原始 CAN 抓包确认 device 1 的 Type-24 约 `39.948 Hz`、device 2..6 约
`99.747..99.812 Hz`，owner 实际约 `39.943 Hz`。device 253 的六帧是主机发送的开启
上报命令，不能当作另一关节反馈；J7 仍不加入六轴控制。上报寄存器当前值未知，
本次没有修改它。原始记录、部分加力窗口和首个重力观察保留，尚无合格稳态运动
窗口，不自动应用新模型。本次开发不连接 CAN；真机回弹及停止效果仍待现场验证。

固定候选的离线组合回归为 1188 passed、2 skipped、53 deselected，包含 staged/joint
motion、自动采集／workspace／session 及 40/100 Hz 和反向残差的三项原生发送路径
合成模型；Ruff、diff 和完整启动参数的 no-CAN preflight 通过。同输入 PI＋slew
边界在旧实现双向失败、候选双向通过。两个 skip 与未选择的其他模型不计作通过；
没有进行集成分支合并、设备寄存器操作或真机复测。

### 连续 TRACK 的同类合成助力修复（pufipubb 记录之后）

`j2-j3-continuous-arrival-acceptance/b601-owner-pufipubb` 的正常到位减力已符合上一轮
修复：例如约 `-0.402 -> -0.390 N·m`，没有在进入到位时直接撤空。但第一段 J2
扫描仍因 `leg_timeout` 未完成，发生八次同目标速度治理恢复。此前正常 TRACK
仍可在运动暂时反向时，以快速阻尼速率恢复整股 spring/bias/velocity 合成请求；
记录有约 26 ms 内 `-0.250 -> -0.494 N·m` 的增加，随后又快速撤小。短窗口确实
检测到多计数突发速度，不能称单计数误报：一处约 50 ms 进展 `0.00307 rad`、
短窗口速度 `-0.0611 rad/s`，越过未改的 `0.0585 rad/s` 软门，长窗口仍为
`-0.0290 rad/s`。不通过改时间窗、放宽门限或取消真正速度停止来处理。

本次把相同合成助力规则覆盖到正常连续 TRACK：增加走既有 tracking slew，减少
尊重显式 normal fall 配置，未配置时复用既有 `0.5 N·m/s` 卸力率。早期速度前沿
仍禁止加推进力或恢复下游已经切掉的尾量，但正常负载助力不被该前沿瞬间撤空。
复用已有 `protect_maintenance=False` 起步交接信息，将其传入私有输出函数，保留
初始高脉冲的快速撤尾；真正软／硬速度停止、许可撤销、人工介入、反馈故障及过零
回执保持原语义。没有新增模式、状态、参数、预算或自动重试，不改非连续路径及
共享执行接口，也不写设备寄存器。边界回归覆盖记录中的增力／撤力及真实软门停止。

本轮最终 `safe_stop` 六轴回执完整、无 cleanup error；随后 Ctrl-C 中断了离线报告，
主结果和 10322 帧样本归档已保留，报告状态为 `INTERRUPTED`。它不同于更早的采集
超时，也不是失能回执失败。实际反馈／派发仍约 `39.95 Hz`；后续真机效果和模型
拟合资格仍需验证，不能由本次软件边界／合成模型回归推出。

本次最终核心 staged/joint-motion 回归 848 passed、2 skipped、53 deselected；
采集及交接选择集 250 passed，完整 workspace 48 passed。另有 40/100 Hz 两个原生
派发合成模型通过，集合有重叠不相加。记录边界的四个增力／撤力方向用例在旧实现
失败、候选通过；新软速度门用例确认正常渐变不延迟真正软停止。Ruff、diff 及完整
启动参数 no-CAN preflight 通过；不视为仓库整体集成或硬件验收。

### 等效保持命令搜索候选（r9g83mx9 撤力回退之后，2026-10-19）

本轮真机 `r9g83mx9` 在第一段 J2 扫描前，零附加命令下约 `2.900°` 保持五秒；
单段超时撤力时则从 `2.241°` 在约 `0.33 s` 内返回 `2.900°`。该回退期间驱动和
软件位置 Kp 均为零，附加命令为零或负向制动，不能归为显式正向位置目标。
这支持“当前基础补偿不能在该位置保持”的判断，但不能单独分离重力残差、传动
弹性、预载和摩擦滞回，更不能把扫描起动力矩直接装成重力修正。

新增显式 profile `docs/hardware/profiles/b601_j2_j3_holding_search_v1.json`，
旧 workspace profile、定点模式及完整抬升流程不变。仍用原 owner 启动和 GUI 的
“J2/J3 多位姿自动采集”入口，不增加命令输入或确认口令。第一版只取 `(3°,3°)`
一个中心，两轴仍限制 `[0°,10°]`，先保留原换位和五秒零助力观察，再逐轴对原局部
端点进行保持命令搜索。首轮局部采样半径显式缩到 `0.01 rad`（约 `0.57°`），
优先在已能到达的位置附近采集，再扩大覆盖。不是增设硬运动门，也不扩大原包络。
**不是全角度采集已通过，也没有同时驱动两轴作补偿保持。**

搜索流程：同一冻结目标的有界 TRACK 到位 → 保留当前实际发送命令并暂停自动卸力 →
固定附加命令观测 → 向零试探／有边界的二分 → 慢速重到位 → 同条件复验。
固定命令观测不叠加位置 PD、速度调节或积分；基础重力仍随实际 q 更新，冻结的是
模型参数而非重力向量。重到位优先携带已观察静止的保持命令；尚无此类观测时才使用
最近实际派发命令，交接清掉旧积分，
防止从原偏差反复开始或把旧积分重复计入；该项只活在当前点，不安装成补偿模型。
撤力时出现与撤力同向的变化先固定实际命令；增加反向纠偏命令时仍遵守原有
普通 slew，不因这项变化把纠偏爬升永远卡在同一水平。

新的纯搜索器输出可复验的保持值、已观察静止范围、正／负运动边界和全部试次。
不假定零与到达命令天然成有效边界；必要时只在现有包络内有限扩展。
同一水平的重复结果矛盾、权限不足或尝试耗尽，记录未完成，不强行给出二分根。
同条件复验不代表已验证跨方向／跨预载重复性；完整六轴实际姿态、名义目标、
重到位起点、净进入方向、命令、重力命令、阻尼和时间分别保存。

profile 的观测范围 `0.002 rad`（约 `0.115°`，超过四个解析位置计数）是定点采样／
变化判断范围，不是新的硬位移门。命令搜索分辨率 `0.025 N·m`，固定观测一秒。
保持点采样邻域独立于旧“撤力后到位”容差，使用前述 `0.002 rad`；不再被旧的
“到位容差不得小于跟踪停止带”条件强迫生成大跨度目标。跟踪停止带及保护不变。
观测超时三秒、最多十二次尝试，最终保持值至少两次同条件观测。观察计时从实际
完整派发后的因果反馈开始；爬升未完成／下游尾量未归一不计成固定命令样本。
爬升时间不冒充固定观测时间，仍占原有单段时限及命令预算。爬升中移位后停住、
平台开始时已经离开同一点，保存诊断后在同预算内重新到位，并缩小下一试探跨度，
不当作持续运动边界。首次实际稳定的完整姿态冻结为同点采样基准，名义目标另存；
后续比较不再让首次到位偏差提前消耗实际采样邻域。既有 TRACK 的名义目标不改写。

仅显式新模式把正常到位的自动卸力／到位制动时限换成原有有限点位采集时限，
允许测量非零保持命令。真正速度制动、支持轴保护、取消、反馈异常和退出仍走
原有制动时限、最终零派发、新停稳证据、基础阻尼恢复及 GUI 支撑确认失能流程。
全程不重置单段／整批时钟或暴露预算；当前单段仍为 `39 s / 16 N·m·s`，
原 `0.195 rad/s` 硬速度、`40 ms` watchdog、力矩及几何包络保持不变。
采样点完成与撤力后的到位误差分开记录，后者不回溯作废前者；停止确认本身仍必需。

GUI 展示重到位／定命令观察、尝试数、当前命令和搜索原因；报告另存
`holding_calibration`。已经完成的保持观测即使随后撤力失败仍保留。
保持／重到位窗口明确排除原运行阻力拟合，原始命令与运动窗口仍作诊断保存。
这些均是软件命令／编码器响应，不是输出侧实测力矩、唯一重力参数或被动性证明，
`auto_apply=false`。后续先检查正反进入重复性，再拟合位置相关等效偏置并重新采集。

该候选未改变 `shared_control_runtime.py`、硬件 backend、设备寄存器或框架线，
没有连接 CAN、执行真机、合并或推送。J1 Type-24 导致的约 `39.95 Hz` 节奏问题仍未
解决；新增因果观测同时用 `40 Hz / 100 Hz` 软件回执路径检查，不用离线通过冒充
实际采集效果。第一版仍要求初始中心按旧流程换位成功，不能声称已解决任意两轴
都需要非零保持时的跨位姿交接；这是后续吸收验证后才扩展的范围。

验证：核心／命令报告组合 `1082 passed / 55 skipped`；采集 workspace／cycle／
几何组合 `127 passed / 1 skipped`，该 skip 是 worktree 默认资源路径缺少 pinned
reBotArm；指定现有只读来源后几何三项全部通过。最终新增功能和 session 为
`84 passed`（其中新增 27 项），包含 40/100 Hz、反向撤力回退、非零保持复验、
真实软／硬速度停止、下游尾量、因果计时、缓升时间、预算、GUI 实际 JS 函数及
后续退出失败的数据保留；集合有重叠，不相加。Ruff、JS syntax、diff 和完整
参数 no-CAN preflight 通过。名义几何报告 17 个独特配置无三角表面／地面相交，
但起点／返回近零段仍有既有凸包／2 mm 预防裕量警告，最小网格间距约 `0.773 mm`，
状态保留 `CLEARANCE_REVIEW_REQUIRED`，不称几何验收通过。没有做新模式真机测试。

#### s5d_69vz 撤力回退不更新搜索的修复（2026-10-19）

基线 `5d5e018a4654d67a6b18dbc9f0ad6d4f42c59d9b` 的真机记录在 J2 第一采样点
约 `2.417°` 得到一次 `-0.304858 N·m`、约一秒的静止观测。请求撤向零后，命令在
约 `-0.070971 N·m` 冻结，但因平台形成时已回退到约 `2.747°`，观测被排除。
后续十一轮都不更新候选，重复向零撤力，最终 `leg_timeout`；并非硬保护或命令暴露
预算耗尽。名义目标约 `2.306°` 与实际采样位置相差 `0.001947 rad`，旧逻辑还混用了
这两个位置的 `0.002 rad` 邻域。

修复后把定命令观测与变力过程分开保存：`observations` 仍只有已解析静止或持续运动；
`transitions` 保存变力范围、实际平台、派发回执、六轴位移和请求值，仅用于下一次试探。
以上述数值选择约 `-0.187915 N·m` 的中间命令，不再原样请求零；这不是已验证保持值，
也不把未实际到达的零或变力终点冒充正／负运动边界。中间命令静止后继续缩步；范围已
足够小则复验已知静止命令。该路径只保证寻找可复验的可用保持值，不保证唯一根或全局
最小绝对值。真实非零定命令运动边界存在时优先使用边界，不因零尚未实测又回去试零。

变力回退、以及短暂平台回退后停在点外，都在同一尝试和命令预算内返回采样邻域。
没有可用静止锚点、支撑姿态改变或预算耗尽时停止当前采集，不新增无限重试；变力记录
不进入固定力或运行阻力拟合。GUI 分别显示有效观测数、变力缩步数、下一候选命令。
完成点的记录也保存实际采样姿态和名义目标，后续撤力回退不作废保持观测。

未修改 profile、运动 tracker、staged 助力合成、硬件 backend、共享控制接口或厂商资源；
所有既有速度、位移、总力矩、变化率、watchdog、真实制动、撤销和最终失能确认保持不变。
新接口仅为纯搜索器内部增量，报告增加 `transition_count` 和详细 `transitions`。
旧实现对同输入的回退缩步与实际采样位置边缘两项回归失败，候选通过；40/100 Hz 原生
控制器＋脚本反馈覆盖返回、复验、撤力及短时回弹。脚本反馈不是实际传动模型或真机证据。
离线完整命令 preflight 通过（`can_opened=false / motor_command_sent=false`）；没有执行
新一轮真机、安装补偿、合并或 push。沿用上述几何警告及约 40 Hz 实际反馈节奏限制。

本轮验证命令：`bash scripts/run_pytest.sh -q tests/test_b601_holding_calibration.py
tests/test_b601_calibration_session.py tests/test_b601_workspace_calibration.py
tests/test_b601_calibration_cycle.py tests/test_b601_staged_torque_console.py
tests/test_b601_automatic_calibration.py tests/test_rebot_b601_staged_assist.py
tests/test_rebot_b601_joint_motion.py`，结果 `1375 passed / 55 skipped`；未设置
`B601_REBOT_SOURCE_ROOT` 的可选离线 plant 验证未计入通过。之后补充短时回弹、复验
状态和 J2 绑定用例，最终保持搜索与 session 两模块为 `102 passed`，与大集有重叠。
四个改动 Python 文件 Ruff、现有 GUI 的 Node syntax 和 `git diff --check` 通过。

#### k3b28lv8 已完成保持点与后续零误差目标（2026-10-19）

基线 `c1892c4f843916c05e9819d4c13a7108620cd89f` 的真机结果已经有一个完整 J2
保持点：六次定命令观测、一次变力退回，约 `13.62 s` 选出 `-0.150781687 N·m`；
最后两次均约一秒、J2 位置跨度约一个计数，并完成撤力和停稳确认。
`automatic_calibration_report.json` 为 `VERIFIED_COMMAND_HOLDS_AVAILABLE / verified_points=1`。
该值对应实际 J2 约 `2.461°`、J3 约 `2.263°` 及记录中的其余关节姿态；是软件
附加保持命令，不是唯一重力残差或实测力矩。起步／到达命令不等于最终保持命令。

之后撤力将 J2 带回第二目标约 `2.900477°`，接受目标时 offset 正好为零。
旧会话仍发布 CUE，但误差不足以授权起步；约 1546 个周期一直为零附加命令、
`awaiting_breakaway_authorization`，第 2 段耗尽 `39 s`。没有开始该点的保持搜索，
不是第一点搜索失败，也不是硬限幅或命令暴露预算不足。后续 `operator_interrupt`
是 owner 的停止原因，与先发生的采集 `leg_timeout` 分开；六轴 safe_stop 回执完整。

修复限定于显式 holding-search 会话：准备时若已在既有采样邻域，先等待原六轴
静止窗口、完整零附加命令和基础阻尼回执，进入内部 `observing` 状态。
不 arm、不发布位置目标、CUE 或起步授权；复用原因果定命令观察器，两次静止观测
确认零为该点可用保持值，再走原单独撤力／新停稳确认，继续后续真正需要移动的段。
基础阻尼与实际六轴回执逐样本保存；该观测不冒充运动数据或自动安装零重力修正。
GUI 明示“已在采样点，零附加命令观察（不驱动）”；完成段另存
`observation_mode=passive_zero_assist`，与主动定命令观测分开。

该被动路径不产生非零驱动。若发生真实漂移、命令变化、人工介入或反馈异常，
保留已确认观测并按原退出路径停采，不在被动观测中偷偷启用加力。
复用原段／整批时限和预算，不延长 39 秒；六轴原速度、位移、watchdog 和停止语义不变。
只新增采集内部状态与报告说明，不改硬件 owner、共享执行接口、profile 或厂商配置。

验证：`bash scripts/run_pytest.sh -q tests/test_b601_holding_calibration.py
tests/test_b601_calibration_session.py` 为 `115 passed`；workspace/cycle/automatic
三模块的 `-k 'held_probe or result or gui or short_target or terminal or lifecycle or workspace'`
为 `76 passed / 287 deselected`。J2/J3、40/100 Hz 的同输入用例覆盖零点先采和
非零点撤力后落到下一零点，并保留后续正常移动；旧会话方法在零点入场断言失败，
新方法通过。覆盖命令尾量、硬速度、人工介入／许可撤销、漂移、原段时限和 GUI。
Ruff、Node syntax、diff 检查通过；均为离线软件证据，没有新真机、CAN、合并或 push。

#### pk81t8jv 复核重复与采集邻域余量（2026-10-19）

基线 `fd0195ab1c852f62b18fabbcad3439660b4a2bec` 的这轮记录停在第一个 J2 保持点，
实际采样 J2 约 `2.3951°`、J3 约 `2.2633°`。四次定命令静止观察分别为
`-0.354424 / -0.212275 / -0.141201 / -0.105664 N·m`，另外八次为变力／回退的
截断观察；没有完成保持点或运动段。最初向零撤力偏离采样点约 `0.004985 rad`；
后六次重复 `-0.105664 N·m` 的复核均在因果定力观察开始前越出原 `0.002 rad`
邻域，偏移为 `0.002301～0.002685 rad`，约多一至两个编码器计数。
12 次尝试预算被耗尽，而本段命令暴露约 `3.34 N·m·s`，并未达到原 `16` 上限。
六轴最终 safe_stop 回执完整；退出码 0 不表示采集完成。

旧搜索在弱保持命令的返回复核失败后，仍把它当成最小幅度静止候选，反复选择
同一命令；重新到点偏置也独立选回这一历史值。修复后，返回复核的截断记录新增
`verification_command_nm`，该命令只在本次搜索的后续候选选择中排除，历史静止记录
不删除、不改写为正／负持续运动边界。按幅度依次选择其他已观察过的静止命令，
重新到点也复用同一个选择，并要求新的完整因果定力观察才能确认完成。
如果已知锚点均不能重复，就明确停采并保留数据；不增加无穷重试、不自动拟合安装。
尚无静止锚点时保留原“最近实际派发命令”偏置，不提前使用未经观察的下一候选。

本次按用户授权，仅将显式 `b601_j2_j3_holding_search_v1.json` 的采集邻域及静止
跨度门从 `0.002` 放宽 50% 至 `0.003 rad`（约 `0.172°`），单点尝试上限从 12
放宽 50% 至 18。逐次保存实际六轴姿态、命令和入场过程，不能把该邻域视为精确同一点。
通用默认仍为 `0.002 / 12`；观察时长 1 秒、两次核验、39 秒段时限、命令暴露预算、
局部行程、六轴硬速度／位移／支撑轴保护、40 ms watchdog 和停止语义均不变。
`movement_rad` 与 `return_command_nm` 加入搜索状态，明确有效采集半径和返回偏置；
原 `observed_stationary_range_nm` 仍仅是历史观察范围，不是已验证的完整保持区间。

离线验证：holding/session 两模块 `127 passed`；workspace/cycle/automatic 的
`-k 'held_probe or result or gui or short_target or terminal or lifecycle or workspace'`
为 `76 passed / 287 deselected`。新增 J2/J3、40/100 Hz 的真实控制器＋脚本反馈回放，
覆盖较弱保持复核失败后用另一已观察命令返回、取得新回执、最终撤力；以及七计数
短回退在旧／新采集邻域的不同分类。该脚本反馈不是已辨识物理模型。
按 pk81t8jv 前六次结果（正负镜像）回放，旧纯搜索继续选择 `0.105664` 幅值，
新搜索选择 `0.141201` 并仍等待新观察；新旧同输入断言分别通过／失败。
Ruff、diff 和完整参数的 `--preflight-only` 无 CAN 预检通过。
本轮不改硬件 owner、共享运行接口、重力参数、厂商模块或其他开发线；未执行真机、
CAN 操作、模型应用、merge 或 push。原约 39.95 Hz 实际节奏和几何净空告警仍未解决，
不因本次搜索回归通过而宣称全角度或力反馈硬件验收通过。

#### 4j8unjzh 实际平台复核与多轴采集边界（2026-10-19）

这轮仍在第一个 J2 搜索点，但不是 18 次尝试或 16 N·m·s 预算不足：3 次定力
观察、5 次截断观察，本段暴露约 `2.42 N·m·s`。内部停止原因是
`holding_verification_changed`。复核请求为 `-0.342390794 N·m`，返回加力时 J2
向原采样点回移，观察器按原位移门冻结在实际 `-0.326653378 N·m`，因果平台持续
约 1.003 秒且位置静止；搜索器错误地把“新的有效平台”解释成“原请求复核失败”。
报告已保留这个实际命令的单次观察，但不能称它已重复核验。

修复只改变定命令观察的交接语义：会话显式传入原请求值，纯搜索校验它属于当前
候选；若观察器已明确冻结斜坡，实际平台在原包络内、完整因果定力观察为静止，
就把实际值作为待复核的新候选。仍须在同一实际值再观察，不能把原请求与实际值
按“足够接近”算作两次。没有明确冻结／来源不匹配、真实持续运动或原预算耗尽时，
保留原拒绝／停止语义。本轮不改任何阈值、profile、驱动器或发送流程。

按六轴记录核对，约 11.3 秒 J2 搜索内，J2 总位置跨度约 `0.461°`，J3 跨度约
`0.022°`，J1/J4/J6 约一个计数、J5 约两个计数，非 J2 附加命令始终为零。
因此这次首次失败不能归为另一关节明显运动造成；J2 变化时 J3 重力模型命令仍有
约 `0.0182 N·m` 的变化，说明已有重力前馈使用完整姿态而非独立单轴常数。
较早的 J3 定位阶段同时观察到 J2 约 `0.110°` 的位置跨度；这是同步变化证据，
尚不能单独分离重力残差、动态耦合与预载释放。六轴未在轴切换时失能，最终 safe_stop
回执完整。

当前工作区结构是“单轴定位并完全撤力、另一轴定位并完全撤力、单轴局部搜索”，
驱动 Kp 为零，支撑轴只有重力与阻尼，不是位置保持。这是全姿态自动标定的未完成
边界，不能继续用单轴门限放宽代替。后续增量应在原工作区 GUI 模式内建立：

1. 同一个 owner 下的 J2/J3 全姿态搬运／稳定与分别试探；单轴调试、人工透明模式保留。
2. 分开保存两轴维持偏置和运动项，另一轴试探时维持支撑轴姿态；正常阶段交接与
   人工撤销／异常撤力分开，后者仍撤销全部自主附加项并遵循原失能协议。
3. 每个候选用完整 q、入场方向／历史、重力模型值、两轴实际派发附加命令和回执
   定义条件。支撑姿态改变后先重稳定，再采样，旧保持值只作为初值而非跨姿态真值。
4. 静态采集交付局部可重复保持范围；重力与摩擦分离另用同姿态双向慢速运动和
   方向／历史分组验证。没有独立测力时不能把某个静止偏置当成唯一真实重力。

以上双轴保持与连续交接尚未实现，本轮没有把单轴 1 N·m 配置直接扩成双轴权限，
也没有取消支撑轴命令为零的现有单轴观察合同。涉及该合同的后续改造须作为独立增量，
明确合成包络、回执消费者、撤销及预算，并先验证不改变硬保护。

验证：holding/session `141 passed`；相关 workspace/cycle/automatic 选择
`76 passed / 287 deselected`；Ruff、diff 通过。新增强返回途中回弹／冻结平台的
真实控制器脚本回放，旧实现的 J2/J3、100 Hz 两例在 `holding_verification_changed`
失败，修复后通过；40 Hz 与原稳定返回继续通过。增加另一校准轴漂移时不得形成
原姿态保持值／有符号力边界的用例，以及平台来源、命令包络和有限预算反例。
均为离线软件证据，未连接 CAN、未执行真机、未应用模型、未 merge／push。

## 历史交接记录

> 版本：2026-09-22。本文是 B601-RS 当前代码、真机 commissioning、重力采集和
> Harness/Git 交接的统一入口。它描述可复核的实现和流程，不授予新的使能、运动或
> 合并权限。任何真机动作仍须由现场操作者在确认机械支撑和物理急停后前台执行。
> B601 Real2Sim、双向 ReBot/RM65 遥操、policy 传导和人工实时干预的长期阶段目标见
> [`../architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md`](../architecture/plans/rebot_b601_rm65_real2sim_hitl_roadmap.md)；
> 路线图不替代本文的当前候选和真机证据。

## 0. 先看结论

- 当前候选位于
  `/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908`，
  分支为 `task/root/b601-gravity-calibration-runtime-20260908`。默认重力 profile 的代码
  checkpoint 为 `70fcb2a7ba690cbc733c586b11633738b9f1b8b1`；本文的最终交接提交和 tree 以分支
  description、交接消息及现场 `git rev-parse` 为准，不能在被提交的文档中自引用其 SHA。
- 候选尚未合并到 `AgenticRobot-development-mainline`，也未 push、未安装。截至本次交接，
  development 为 `028b6d28fb9056843959cceb074af8f0a6b3898a`，不包含上述候选；共同祖先为
  `ef54a901bb5df92d9926c7c47ef7def5c1ddfdf5`。development 在本候选六个声明路径上没有
  自共同祖先以来的改动，三方 merge preview 未见文本冲突；这只是当前只读检查，不是
  合并授权或最终语义评审。
- 外部 reBotArm 源码当前 pin 为 commit
  `5ba28acef46237eb6a7560658bbc43b06cf8a259`、tree
  `ddd34d2b400b3987d4875bd6a99bd29b74d39dac`；
  `/home/zc/HandMotion/Developing/reBotArm_control_py` 工作树干净，配置文件
  `config/rebotarm_rs.yaml` SHA-256 为
  `be51e877050cc4716188fca3b4f2a41109682513c7715c71594066a133ba5cf9`。
- 候选中的 `hardware/reBotArm_control_py` gitlink 指向上述 commit，但当前任务 checkout
  显示为未初始化（`git submodule status` 前缀为 `-`）。真机命令必须显式使用
  `--rebot-source-root /home/zc/HandMotion/Developing/reBotArm_control_py`，不能依赖一个
  未记录版本的本地模型副本。
- 已完成的代码重点是：六轴 gravity+damping protective hold、启动锁到最终控制器的
  连续 handoff、J2/J3/J4 编码器边界兼容、J4/J5/J6 反馈工作余量、qvel/qacc/非有限值
  guard、原始 RobStride 帧诊断、交互式多姿态残差采集、startup replay、离线 candidate
  fit，以及逐关节重力 scale/offset/bias/Coulomb 和 driver Kd 配置。
- 操作者已确认下列正常重力补偿 profile 的真机手感合理，因此它已成为 enabled gravity
  mode 的程序默认值：软件 `kp/kd=0/0`，driver Kd=`0 0 0 0 0 0`，全局
  scale=`1.0`，joint scale=`1 0.96 1 1 1 1`，offset 全零，bias=
  `0 -0.22 -0.11 -0.08 0 0 N.m`，Coulomb 全零，速度阈值为 `0.02 rad/s`。
- 当前状态是“独立候选已准备交接”，不是“已合并/已安装/最终硬件通过”。逐关节 offset
  和 Coulomb 仍为零；没有独立力矩传感器标定，也没有高带宽触觉或系统级 `HARDWARE_PASS`。

本次默认值变更限定在 B601 硬件 owner/probe 及其 protective/degraded hold；已有
MuJoCo `B601Real2SimControllerProfile` 保持独立的仿真契约，避免未经单独验收就改变
既有仿真动力学和回归结果。

## 1. 任务目标与明确非目标

### 1.1 目标

B601 的第一阶段是本地有线、六自由度 arm 的可拖动重力补偿和末端阻抗 commissioning：

1. 用同一份 pinned URDF/Pinocchio 模型计算 `g(q)`、FK 和 Jacobian。
2. 用完整 `joint1..joint6` 共同实现末端平移或 6D 位姿阻抗；不能只使能 J6，不能把
   其他关节留在去使能状态后声称完成末端阻抗。
3. 解决“使能 → 锁止 → 重力补偿/阻抗”的启动力矩空窗，避免启动时明显前倾或下坠。
4. 首次 Ctrl-C、普通异常、控制时长结束和可恢复反馈异常优先保持全臂重力补偿加阻尼；
   只有无法继续发出完整安全命令或操作者明确最终停止时才全臂失能。
5. 用可复核的 JSON/log、固定 commit/tree、配置和 vendor pin 形成独立证据。

### 1.2 不属于本任务的工作

- 不把 B601 backend 变成 policy、网络遥控、leader-follower blending 或 action-chunk
  调度器；这些属于 `control_composition`/`teleop_runtime` 等上层。
- 不把 J7 夹爪伺服强行塞进六自由度 arm 的 FK、Jacobian、重力或 protective hold。
- 不把 MuJoCo ideal-effort 结果当作 RobStride 的绝对力矩、摩擦、负载或人手触感标定。
- 不自动改写源 URDF、YAML、固件参数或把 candidate overlay 直接激活。
- 不用 Harness 的离线/集成证据代替现场真机观察，也不在没有授权时启动 CAN、使能或运动。

## 2. 代码所有权和依赖边界

| 路径 | 唯一职责 | 不应放入的内容 |
| --- | --- | --- |
| `hardware/rebot_b601_rs/` | MotorBridge/SocketCAN、ID 映射、反馈、mode、enable/disable、逐关节 receipt、临时 `0x700B` torque-limit 事务 | policy、HITL blending、末端数学 |
| `robot_motion/rebot_b601_rs/` | Pinocchio 模型、FK/Jacobian、gravity、MIT torque composition、末端 wrench/pose spring、torque/slew conditioning | CAN、vendor 初始化、设备状态 |
| `sim/mujoco/rebot_b601_rs/` | MuJoCo 场景和 ideal generalized-effort 后端 | 真机 I/O 和硬件结论 |
| `scripts/hardware/rebot_b601_rs_low_gain_probe.py` | 人工运行的低带宽 probe、启动状态机、保护保持、交互采集、离线 fit CLI | 第二套生产控制入口 |
| `tests/test_rebot_b601_low_gain_probe.py` | 纯 helper、状态转换、拟合和故障路径测试；不打开 CAN | 真机操作脚本 |
| `docs/hardware/rebot_b601_rs_hardware_commissioning.md` | 频率/阶段/命令级 commissioning 说明 | moving status 或未核实结论 |
| 本文 | 面向下一位开发者的范围、规则、事实和交接模板 | 替代代码或现场授权 |

`hardware/reBotArm_control_py` 是独立 Git submodule，不是 AgenticRobot 的普通 Python 包。
更新 vendor 时先固定 child commit/tree，再更新 parent gitlink；不得复制一份生成的 URDF/mesh
绕过 pin，也不得从 mutable `main` 分支猜版本。

系统级 HITL 仍采用两个独立通道：运动通道传递受限 human residual，触觉通道将末端
wrench 通过 `J.T @ wrench` 映射为 B601 力矩。B601 的低带宽 probe 只负责本地 arm 控制，
不成为第二个 policy 或远程 authority。

## 3. 坐标、关节和数据契约

- 使用右手系和 SI 单位：`X` 向前、`Y` 向左、`Z` 向上；角度/速度/加速度为
  rad/rad·s⁻¹/rad·s⁻²，力/力矩为 N/N·m。
- arm 顺序固定为 `joint1, joint2, joint3, joint4, joint5, joint6`；ID7 是夹爪，不在
  当前 arm backend 中。
- pose 是 `base_link_T_gripper_end`；Jacobian 是 `LOCAL_WORLD_ALIGNED`、在
  `gripper_end` 原点、行顺序为线速度 XYZ 再角速度 XYZ。
- 末端 wrench 必须先在正确 frame 和原点下表达。不同 frame 不能只重排 XYZ；需要完整
  的 adjoint wrench 变换。离线检查保持功率不变量
  `wrench.T @ twist == tau.T @ qvel`。
- `MotorBridge 0.4.9` 的 `MotorState` 没有 vendor 源时间戳或接收序号；代码报告
  `freshness_verified=false`，`mechPos (0x7019)` 是显式读取/host 接收观察，不得把
  缓存值伪装为已验证的源 freshness。
- 每个完整 MIT/POS_VEL 发送都返回逐关节 receipt；任一关节失败或部分发送不能报告为
  六轴成功。`safe_stop()` 也是逐关节 receipt，`close()` 不隐式失能。

## 4. 控制数学与 torque owner

正常软件请求力矩为：

```text
tau_request[i] = gravity_scale * gravity_joint_scales[i] * g_i(q + offset)
               + gravity_joint_bias_nm[i]
               + gravity_coulomb_friction_nm[i] * direction(dq_i)
               + Kp_i * (q_ref_i - q_i)
               + Kd_i * (dq_ref_i - dq_i)
               + (J(q).T @ wrench_overlay)_i
```

然后按以下顺序处理：

1. 每关节绝对上限取 pinned URDF effort ceiling 乘现场指定的
   `--torque-limit-scale`。
2. 再按上一帧和 `--torque-rise-time-sec` 形成的 torque slew envelope 限制变化率。
3. 最后调用 upstream 同序的
   `send_mit(position, velocity, kp, kd, feedforward_tau)`。

`--gravity-joint-scales J1 J2 J3 J4 J5 J6` 的操作者真机验证默认值为
`1 0.96 1 1 1 1`。它不是 URDF 修改，也没有隐式标定激活。该向量必须一致用于启动锁、
ACTIVE、protective hold 和缓存 gravity-only degraded hold，并记录到结果 profile。
`--gravity-joint-offsets-rad`、`--gravity-joint-bias-nm`、
`--gravity-coulomb-friction-nm` 的默认值分别为全零、`0 -0.22 -0.11 -0.08 0 0` 和
全零。Coulomb 方向只在有限差分速度绝对值超过默认的
`--gravity-coulomb-velocity-threshold-rad-s=0.02 rad/s` 时取正或负，否则为零；这避免静止量化噪声
驱动完整摩擦力矩反复换向。所有项贯穿启动、ACTIVE、protective hold 和缓存补偿，并记录
在结果 profile；它们组成默认 runtime profile，但不修改 URDF、不隐式改写模型文件。

canonical `rebot_b601_rs_owner.py` 默认启用 `--gravity-debug-mode`；直接调用
low-gain probe 或其他模式若要保持参数可比性，需显式使用该开关，owner 可用
`--no-gravity-debug-mode` 恢复普通保护保持语义。该模式在
使能前冻结本次 run 的 gravity scale、六轴 scale/offset/bias/friction、Coulomb
阈值和 driver Kd；ACTIVE、protective hold、恢复后的 live hold 及 degraded cache
均复用该不可变快照。它只固定参数来源，不改变任何反馈新鲜度、运动/力矩/加速度
门禁、发送重试、二次 Ctrl-C/SIGUSR1 全失能或物理急停规则。普通模式不构造该快照，
因此既有严格故障语义保持不变。结果 JSON 的 `gravity_debug` 和
`protective_hold.gravity_parameter_source` 是验收时的证据字段。

在该显式调试模式中，保护保持也不会为冻结快照中 `driver Kd=0` 的关节补入默认
`protective_hold_kd`；`protective_hold.software_kd_nm_s_per_rad` 应与本次调试的
单一阻尼 owner 一致。普通模式的默认阻尼 fallback 保持不变。

### 单关节连续助力调试界面（待真机验收）

`rebot_b601_rs_owner.py --torque-console --torque-console-confirmation
B601_RS_TORQUE_CONSOLE --allow-motion` 在原单一 CAN owner 进程内启动仅绑定
`127.0.0.1` 的浏览器界面；网页只提交短时意图，不能直接调用 MotorBridge。
独立开发 worktree 尚未编译 MotorBridge 时，用 `--motorbridge-root` 显式指定
已有 Python bindings 和已构建 ABI 的同版本 checkout，不把二进制复制进源码树。
界面默认选 J3、助力上限 `0.30 Nm`、上升斜率 `0.15 Nm/s`、撤力斜率
`2.0 Nm/s`、意图超时 `0.35 s`，这些值均可通过 `--torque-console-*` 显式调整。
所选关节助力期间的有限差分速度另设默认 `0.05 rad/s` 保护阈值；越界会在本
周期撤销助力并进入原保护保持，网页同时显示速度和阈值。它不作用于未加力的
手动摆位，也不是主动制动，零阻尼时仍可能发生惯性滑行。
该模式要求软件 Kp/Kd 为零、冻结本次重力参数、启用驱动临时力矩上限，且不与
既有单次关节/笛卡尔助力叠加。必须在启动交接完成、六轴速度都落入静止阈值后
才能从当前实测位置开始一轮；网页失联撤去助力，超出所选关节/其他关节位移
保护则退回原 protective hold，仍须第二次 Ctrl-C/SIGUSR1 才全臂失能。

此模式自动开启 `--debug-dynamics-trace`，逐周期记录反馈位置、有限差分速度、
重力、请求/限幅后的助力与最终软件力矩、完整发送回执；网页的曲线只是短时
滚动视图。所谓“已接受发送的助力”仅由完整发送回执确认，不代表独立测得的
电机实际输出力矩。该界面尚无真机验收，不能据离线测试宣称力矩标定或触觉
性能通过。

显式自动标定 profile 的两个 GUI 模式共用同一个阻尼发送适配口：固定位置采集、
完整搬运流程及其子阶段都必须通过该口解析当前控制帧的 driver Kd，再由原六轴
MIT dispatch 发送并校验回执，不能因 GUI 包装器类型不同而退回零阻尼。开始标定
时，子控制器首帧的阻尼渐升最多使用一个正常控制周期；搬运或等待期间未执行的
时间不计作渐升额度。后续继续从已确认发送的 Kd 渐升，起步扫描的 J3 Kd 仍为零，
纯重力阶段及故障保持仍使用冻结的基础配置。这里校验的是软件发送命令，不是
电机实际阻尼力矩或独立测力结果。

`b601_j3_local_identification_v1.json` 显式启用
`motion_continuous_speed_regulation`，两个 GUI 模式的局部采集阶段都复用已有连续调速。
该选项把显式配置的绝对软制动上限作为连续调速终点，不再叠加第二份速度余量：参考速度
`0.0075 / 0.015 rad/s` 对应的减力起点为 `0.01625 / 0.026 rad/s`，终点仍为
`0.039 rad/s`（本轮批准从 `0.03` 放宽 30%）。两者之间保持原 TRACK、目标和有界维持状态，
按完整 PD 加维持助力减小同向推进量；真正软超速仍制动。本轮局部 profile 的跟踪和采集
到位容差为 `0.0065 rad`，原为 `0.005 rad`；配套命令的六轴原始硬速度保护为 `0.13 rad/s`，
原为 `0.10 rad/s`。普通默认值和其他采集 profile 不变。完整搬运/回程仍使用独立的
J3 `0.05 rad/s` 软制动上限，继承本轮 `0.039 rad/s` 支撑轴软门和六轴硬门。
起步扫描期间 J3 原有的独立 `0.15 rad/s` 原始速度上限不变，其他轴使用上述硬门；
带撤力支撑确认的显式采集路径也同步接纳最高 `0.13 rad/s` 的配置，普通量化调试的接纳上限不变。
反馈、回执、人工撤销、力矩、位移绝对上限及预算保护不变。连续减力不等于
已辨识摩擦，也不证明真机能完成全部运动段，仍须检查实际采集报告和模型数据质量。

连续模式进入到位稳定阶段时，有界维持项先投影到最近完整派发的附加助力，预留当前
同方向 PD 份额，不能恢复 TRACK 调速时未实际派发的内部维持量。速度误差减力继续作用于
完整 PD 加维持输出；不再因阶段名变为 BRAKE 而失效。到位输出以实际派发量为渐变起点，
快速制动速率只允许撤掉推进量或产生有可靠运动证据的反向制动，同方向增推仍受原跟踪
渐升速率限制。到位后仍须完成停稳、渐退、零助力派发确认和撤力后停稳，不把位置进入
容差直接当作稳定；原生零回执前不得反向。所有启用 `arrival_stabilization` 的目标到位路径中，
内侧 `arrival_margin_rad` 仍用于提前进入制动/接近带；制动后停稳驻留按配置的最终位置容差判定，
不再把内侧门限重复用作停稳门。量化位置在最终容差内相邻计数间变化时，按既有位置分辨率和
停稳速度累计驻留；越出最终容差、持续超速或六轴运动仍会清零驻留。以上是软件命令链修复，
不是测力标定结果。

重要 owner 规则：

- 最终 active、protective hold 和 degraded hold 的 vendor MIT `kp` 始终为零。只有边界明确的
  `LOCKING/LOCKED/HANDOFF` 与 startup-replay `RAMP_IN/RELOCKED/HANDOFF` 临时锁允许
  使用 YAML MIT Kp；该 Kp 在电机本地闭环执行，退出临时状态时连续回到零。若接纳反馈
  位于 nominal URDF 命令包络外，对应关节的临时 Kp 必须置零，只保留重力和本地 Kd，避免
  被夹紧的编码参考成为主动回拉命令。
- `pose-hold` 的位置恢复只能来自末端 6D spring 经完整 Jacobian 转置映射；软件关节 Kp
  应为零。`cartesian-hold` 同理，只产生平移 overlay。
- 软件 Kd 和 driver Kd 只能有一个实际阻尼 owner。临时锁期间 YAML Kd 由电机本地拥有，
  主机临时 Kd 为零；交接时本地 Kd 连续回到 active driver Kd。`--driver-kd` 的已验证
  默认值为 `0 0 0 0 0 0`，也接受一个六轴共用值；任一项非零时必须
  `--kd 0`、不使用非零 `--target-kd`，并且先完成六轴 `0x700B` 临时 torque-limit
  round-trip；driver Kd 只提供电机局部速度耗散，不提供位置锁。正常重力补偿的软件
  `kp/kd` 默认均为零；需要固定参考采样的 `gravity-calibration` 必须显式选择非零 `--kp`。
- 未选择 driver Kd 时，软件 Kd 由 host 计算；20 Hz 反馈的有限差分量化可能产生阶梯感，
  这是连续性 commissioning 的观察项，不应通过叠加两个 Kd 解决。
- active path 的单帧有限差分 qvel/qacc 越限先隔离，不下发、不更新控制历史或安全缓存；
  下一帧相对最后可信样本恢复时继续运行，连续两帧越限才转入 protective hold。该策略用于
  拒绝已观测到的 J4 单帧伪零值，不提高用户配置的运动阈值，也不放宽位置/力矩边界。
- protective hold 去掉软件位置 Kp，以当前反馈作为每周期参考，保持可拖动；其重力在
  可信反馈下逐周期重算。
- degraded hold 使用“最后完整发送的 pose + gravity-only torque”缓存；反馈不可信时
  不伪造软件阻尼，已配置的 motor-local driver Kd 仍保持有效。这是当前明确的降级语义，
  因此要让故障保持仍有较大阻尼，现场档位应提前选择经过 round-trip 验证的 driver Kd。
- 这也是真实的当前限制：如果运行档位只有 host `--kd`、没有非零 `--driver-kd`，进入
  degraded cache 后不会继续用未经可信反馈支持的软件 Kd；此时能保证的是缓存重力，不是
  “重力+大阻尼”。下一轮若要补齐该目标，应单独设计可由缓存状态证明安全的降级阻尼并加
  针对性测试，不能在文档中把现有 gravity-only fallback 写成已经满足大阻尼。

## 5. 启动状态机和启动下坠诊断

```text
feedback warmup
  -> preflight safe_stop
  -> optional 0x700B apply/readback
  -> MIT mode prepare
  -> [DRIVE] 3/2/1/ENABLE
  -> ENABLED_UNARMED
  -> LOCKING
  -> LOCKED
  -> GRAVITY_RAMP
  -> HANDOFF
  -> ACTIVE
```

| 状态 | 实际动作 | 诊断重点 |
| --- | --- | --- |
| `ENABLED_UNARMED` | 已使能但尚未交给最终控制器 | post-enable q、qvel、ID 和 receipt |
| `LOCKING` | 用当前使能后 q 建立完整六轴临时参考；pinned YAML MIT Kp/Kd 由电机本地闭环执行，主机临时 Kp/Kd 为零 | 首锁帧是否完整、driver total-torque limit 是否已回读、是否出现方向反转 |
| `LOCKED` | 连续默认 4 个 feedback/control 样本重复发送当前锁参考 | qvel、dt、位置限位、driver Kp/Kd 与前馈力矩证据 |
| `GRAVITY_RAMP` | 语义上进入重力阶段；完整 `gravity_scale*g(q)` 从首个控制帧请求，不能再用第二个标量把重力削半 | 若仍下坠，先看模型重力、torque limit、slew、driver Kd 和 q 方向 |
| `HANDOFF` | 同一时长把临时 joint lock 平滑交给最终 controller；末端阻抗项与其他最终项一起交接 | startup lock 与 final controller 是否连续 |
| `ACTIVE` | 正常 `hold`/`slow-step`/Cartesian/pose 或 calibration 控制 | 逐周期 q、qvel、qacc、tau 和 receipt |

当前实现不把重力再独立衰减一次；`--torque-rise-time-sec` 是主机前馈 torque slew 和
driver-local lock-to-final handoff 的时间。若现场观察到刚使能轻微前倾，应从 JSON/log 对比
`gravity_target_tau_nm`、`vendor_kp_nm_per_rad`、`vendor_kd_nm_s_per_rad`、
`conditioned_tau_nm`、`slew_limited` 和 `gravity_scale`：

- `gravity_scale < 1` 或 URDF effort ceiling/scale 不足会产生真实重力残差；
- slew 在首帧限制的是总力矩，不是把某一关节单独压低；
- q 符号、base gravity 方向、payload/末端质量和关节零位误差会使某些姿态残差偏向同一侧；
- 软件/driver 双 Kd、静摩擦和 20 Hz 读数阶梯会把“能移动”误认为“已平衡”；
- 仅凭静摩擦把 arm 卡住不是重力补偿通过，必须在多个姿态、松手和小扰动下复核。

新增的 `startup_diagnostics` 直接保存使能前、使能未 armed、首个锁帧、各锁定 settle
以及首次 ACTIVE 的位置差；逐周期 `samples[].vendor_status` 保存既有 MotorBridge
operation-status 缓存速度/力矩。两者都不增加 CAN 请求、不参与控制，也没有位移 pass/fail
阈值。配套 MotorBridge `robstride-startup` 模式只被动追加 type-1/type-2 原始帧到文件；不要在
现有同步 `0x7019` 控制循环内再为六轴逐周期追加 `0x701A/0x302C`，否则会显著压低控制节拍。

## 6. 正常模式和适用范围

| 模式 | 全部参与关节 | 参考/效果 | 备注 |
| --- | --- | --- | --- |
| `hold` | J1–J6 | 当前 post-enable q 的关节位置保持 + gravity + 单一 damping | 用于基础启动/静态诊断，不等价于末端阻抗 |
| `slow-step` | J1–J6 | 只改变指定目标关节的参考，其余仍保持；完整六轴 dispatch | 仅低幅方向和限幅检查 |
| `cartesian-hold` | J1–J6 | 当前 FK 平移锚点，平移 spring 经 `J.T` 共同调节 | 不控制姿态 |
| `pose-hold` | J1–J6 | 当前 FK 的完整 SE(3) 锚点，平移+旋转 wrench 经 `J.T` 共同调节 | 正式末端阻抗首测；软件 joint Kp=0 |
| `gravity-calibration` | J1–J6 | 可拖动 gravity+damping；R 后暂时固定 q 记录静态位置残差 | 多窗口无需重启 |
| `startup-replay` | J1–J6 | 首次自然稳定后按 R；同一次使能中平滑加锁、保持并再次交接 | 区分边界释放与非边界重力偏差；不自动判稳 |
| `feedback-stability` | 不改变既有状态 | read-only 读取 `mechPos` 并统计有限差分 | 不切 mode、不使能、不失能、不发送控制帧 |
| `driver-torque-limit-roundtrip` | J1–J6 | 读原值→写 scaled effort→回读→恢复→回读 | 不使能、不发 MIT/POS_VEL；成功后才可在 active 命令使用 driver Kd |
| `gravity-calibration-fit` | 无设备 | 离线归约、姿态去重、回归和 holdout 报告 | 只输出 candidate correction |

`--joint` 只影响 `slow-step` 目标或单关节 gain override，不缩小 active enable/dispatch 集合。

## 7. Protective hold、degraded hold 和最终失能

### 7.1 设计原则

对承重机械臂，“停止程序”和“立即去使能”不是同义词。只要仍能发出完整六轴 MIT
gravity+damping 帧，就应尽力保持负载；不允许因普通超时、一次反馈异常或首个 Ctrl-C
把部分关节先失能而形成不对称负载路径。

```text
first Ctrl-C / ordinary handled fault / duration complete
    -> admit a verified current or last-safe snapshot
    -> all-six-joint gravity+damping protective hold (draggable)
    -> if live sample fails: cached gravity degraded hold + one retry
    -> only explicit second Ctrl-C/SIGUSR1, or live+cached full-frame failure
       -> full six-joint safe_stop
```

### 7.2 首次 Ctrl-C

- J1–J6 保持 enabled；J7 不在本 backend。
- protective hold 的软件 Kp=0，参考是每周期当前反馈，不是强行锁死在 Ctrl-C 瞬间位置；
  人可在大阻尼和重力补偿下拖动。
- 进入 hold 的首帧优先使用新鲜、合法、近似静止的 `last_safe`；若它过期但当前完整
  fault snapshot 合法，则使用当前 snapshot；若 live 读取已失效，则使用最后完整命令缓存。
- 第一次 Ctrl-C 的程序结果常为 `status=FAIL, stop_reason=operator_interrupt`，这表示
  试验被操作者中断，不等价于 hold 逻辑失败。必须同时查看
  `protective_hold.latched`、首帧 receipt、`degraded` 和 cleanup receipt。

### 7.3 普通异常和可恢复降级

以下情况只要已有合法 pose/gravity cache 且完整六轴发送仍可用，先进入或继续 degraded
protective hold：

- feedback 缺失、`valid=false`、stale、vendor status 异常或读数非有限；
- 当前位置超出模型边界但缓存仍为合法位置；
- feedback dt 不前进、`dt<=0`、qvel/qacc 非有限，或 qvel/qacc 超过当前 commissioning gate；
- FK、Jacobian、gravity、阻抗、功率一致性或总力矩计算异常；
- 单次 MIT frame 返回失败、部分关节发送或驱动器暂时拒绝命令；
- `duration-sec` 到期、校准窗口未完成、普通控制循环异常；
- 保护保持中的一次 live sample/compute 异常。

降级行为是：

1. 记录 `degraded.reason_counts`、`last_reason` 和进入次数；
2. 用最后完整发送的 q 和 gravity-only torque 组帧；不把错误样本刷新为 `last_safe`；
3. 完整六轴帧立即重试一次；成功后继续保持并等待 live feedback/control 恢复；
4. live 恢复后重新计算 gravity，记录 `recovery_count`，再回到正常 hold。

### 7.4 允许全臂失能的极端条件

全臂 `safe_stop()` 仅在以下事实成立时允许：

1. 操作者第二次 Ctrl-C 或发送 `SIGUSR1`，明确要求最终停止；
2. 没有任何通过位置、限位、年龄、速度和有限值检查的 pose/gravity cache；
3. live 完整六轴 dispatch 失败，且同一缓存完整六轴 dispatch 在一次立即重试后也失败；
4. protective-hold 首帧或后续实现本身抛出未能被 hold 内部处理的异常，无法证明下一帧
   仍可发送；
5. backend/驱动器拒绝或通信故障使“六轴完整发送”无法成立，而不是只坏一轴；
6. 现场物理急停、支撑或控制权状态已经不可确认，此时由人员按现场安全规程处置。

这些条件不是“更严格的常规测试阈值”，而是证明无法继续保持负载的最后分支。失能前后
必须记录 `full_disable_receipt.completed`、完成/失败关节列表和 `cleanup_errors`；若
receipt 不完整，不能把程序退出当作设备安全已确认。

### 7.5 保护保持中各故障的审查表

| 事件 | 首选处理 | 何时升级 |
| --- | --- | --- |
| feedback 缺失/无效/过期/vendor 异常 | degraded cached gravity + driver Kd（若已配置） | 无完整缓存或缓存帧重试仍失败 |
| 位置再次越过反馈工作包络 | 拒绝该样本，保留上一合法 cache | 无合法 cache 或无法完整发送 cache |
| timestamp 不前进、`dt<=0` | 不用该样本更新 q/qvel，进入降级 | 后续 cache dispatch 也失败 |
| qvel/qacc 非有限或超过 gate | 不接受为 last-safe，降级保持 | 无安全 cache 或发送失败 |
| FK/Jacobian/gravity/impedance/tau 异常 | 不发送异常计算结果，使用 gravity cache | cache 不存在/发送失败 |
| MIT 失败、部分发送、driver reject | 一次完整帧重试，再用 cache | live 与 cache 均不能完成六轴 |
| hold 首帧/后续未处理异常 | 视为 hold implementation failure | 立即 full safe_stop，并保留异常和 receipt |
| 第二次 Ctrl-C/SIGUSR1 | 明确最终停止 | 直接执行全六轴 `safe_stop` |

## 8. 反馈边界和数值异常处理

### 8.1 不改 URDF 的边界投影与反馈工作包络

URDF 继续定义 nominal 模型、规划和出站命令边界，但它不是这台实物在人工拖动下每个可达
角度的完整证明。反馈观测分为编码器边界兼容和真实工作余量两层：

- J2：`-0.010 <= raw_q < 0` 投影到 `0`；该范围覆盖两次零位只读捕获和
  2026-09-08 host-id 限定反馈下观察到的 `-0.007627 rad` 启动边界瞬态；
- J3：`-0.003 <= raw_q < 0` 投影到 `0`；
- J4：运行 nominal boundary 使用精确 `[-pi/2, +pi/2]`，对
  `-pi/2-0.001 <= raw_q < -pi/2` 投影到 `-pi/2`；
- J4/J5/J6 在各自 nominal URDF 上下界外另接纳 `0.1 rad` 的真实反馈工作余量；这些值
  不做投影，可进入 FK、Jacobian 和 `g(q)`；
- 超出上述窗口/余量仍 fail-closed；不把任意大值 clamp 成边界；不改写源 URDF。

每个证据 JSON 同时保留 `raw_qpos_rad`、归一化后的 `qpos_rad` 和
`boundary_normalized_joint_names`。出站 reference 和 torque envelope 仍按有效 nominal limit
约束；slow-step 不能借用读侧余量。临时锁之外的 active probe vendor Kp 固定为零，因此位于读侧余量
时，软件位置/重力计算保留真实姿态，而编码到 MIT 帧的位置 reference 夹回 nominal
范围。临时锁必须额外取消该关节的 vendor Kp，确保读侧余量不会变成 driver 位置命令；
实际取消项进入 startup diagnostics 和逐样本证据。J4 源 URDF 的 `-1.57/1.57` 是两位小数记录，
不能把它误读成比 `±90°` 更窄的真实机械边界。

### 8.2 运动 guard 的顺序

1. 先检查六轴反馈完整、状态有效、位置有限并通过边界投影/限位。
2. 用相邻 normalized q 和精确 host receive timestamp 计算 qvel。
3. 用连续 qvel 和同一 dt 计算 qacc；首个控制样本没有前一 qvel 时明确记录 unavailable。
4. 检查 `dt<=0`、非有限 qvel/qacc、qvel 越过 `--velocity-abort-rad-s`、qacc 越过
   `--acceleration-abort-rad-s2`。
5. 只有 guard 通过的样本才可计算 gravity/impedance、发送 MIT、更新 `last_safe`。

拒绝样本必须单独记录 offending joint、raw/normalized q、previous/current qvel、qacc、dt、
sequence/timestamp，并标记 `command_dispatched=false`、`eligible_for_last_safe=false`。
巨大的非有限/异常数值（例如曾经出现的 J4 超大读数）不能进入 FK、gravity 或 hold reference。

## 9. 参数政策：硬限制、协议域和现场配置要分开

这是本任务最容易被误解的地方。代码/测试不能把一个现场调参值伪装成“通用安全范围”。
当前只有一个 B601 开发者和一个现场控制 owner；不应为了假想的多开发者/多控制器场景
堆叠重复防御逻辑。唯一 CAN owner、完整六轴 dispatch、物理边界和失能收据是实际需要的
安全条件，其余调参保持在命令和现场记录中。

| 类别 | 来源/例子 | 处理规则 |
| --- | --- | --- |
| 真实物理/模型硬边界 | URDF joint position/effort；已确认的 vendor 电机额定边界；若审核确认末端速度 5 m/s、加速度 50 m/s²，则可在权威边界实现 | 必须有来源、单位和 owner；所有 profile 共享；越界 fail-closed |
| 数学/协议域 | 比例 scale 的有限性、`torque_limit_scale` 作为 `(0,1]` 的 effort fraction、非负 Kp/Kd、正持续时间 | 只防止无意义输入；不能借此偷偷收窄正常 commissioning |
| 现场 commissioning 配置 | `gravity-scale`、`torque-limit-scale`、Kp/Kd、driver Kd、slew 时间、force/torque/displacement、持续时间、qvel/qacc gate | 放在命令/profile/记录中，由硬件负责人按姿态、负载、支撑复核；不要写进常规测试断言 |
| protective-hold admission | last-safe age、stationarity velocity、fault-to-cache rebase | 是“能否用这帧作保护首帧”的门，不是 hold 激活后的拖动速度限制；异常不通过时优先尝试缓存 hold |
| 调试/证据阈值 | p50/p95/p99、重复窗口去重、拟合 pose tolerance | 只用于诊断或离线选择；不能改变出站控制边界 |

当前 `0.12 rad/s` 是根据已确认人工单轴扰动对正常 qvel gate 的 commissioning seed；
`4.0 rad/s²`、`0.02 rad/s` fault-hold stationarity、力矩/位姿/wrench 包络仍是独立门，
不能因一个门放宽而联动放宽。任何新数值先回答“它是物理上限、协议域还是本次现场配置”，
回答不了就不要进入代码或测试。

## 10. 正式重力补偿采集与拟合

### 10.1 为什么旧窗口不能直接拟合

此前一些窗口使用 `--kp 0`，或在窗口 5 尚未完成时被 `duration-sec` 结束。它们可以用于
反馈漂移、启动和故障诊断，但 `--kp 0` 时静态位置残差不可观测，也不能据此证明质量/质心。
不要把“JSON status=OK”当成拟合输入合格。

### 10.2 现场采集协议

先做六轴 `driver-torque-limit-roundtrip`，确认 `apply_receipt`/`restore_receipt` 完整且
原值逐轴一致。正式采集使用：

```text
LOCKING → GRAVITY_RAMP → HANDOFF → ACTIVE
  ↘ FREE_DRAG
      R 固定当前位置
      → SETTLING（默认 2 s，不计入拟合）
      → RECORDING（默认 20 s，完全不接触）
      → FREE_DRAG
```

每个窗口的现场动作：

1. 在 `FREE_DRAG` 由人拖到目标姿态；松手，确认没有接触、碰撞或持续下坠。
2. 按一次 `R`（无需 Enter）；程序立即以当前六关节位置作为固定参考。无需判断接近方向，
   也不需要为了配对而再次回到同一姿态。
3. 看到 `CALIBRATION RECORDING #N` 后手离开机械臂，但手和物理急停保持在旁边可立即
   介入；不得在记录期间用手“帮它稳住”。
4. 看到 `CALIBRATION COMPLETE #N: returned to FREE_DRAG` 后再移动到下一姿态。
5. Ctrl-C 第一次仍是六轴 gravity+damping protective hold；第二次 Ctrl-C/SIGUSR1 才是
   final safe-stop。窗口中断会被标成 incomplete，不能静默算作完整样本。

推荐数据量不是“重复 100 个同类窗口”：

- training：6–8 个信息不同的姿态，每个姿态一次 `R`；
- validation：3–4 个训练中未出现的新姿态，每个姿态一次 `R`；
- 覆盖肘部、腕部和 J2 约 1.9–2.1 rad 的代表区域，但绝不为覆盖区域主动越过物理边界；
- 若回归矩阵秩、奇异值和姿态多样性已经足够，应停止追加同类窗口。

### 10.3 离线 fit 语义

`gravity-calibration-fit` 不连接 CAN、不切 mode、不使能、不失能：

1. 只提取完整 `RECORDING` 区间；每窗用 q、qvel、gravity 项和 position 项的中位数/MAD
   归约为一个姿态观测，不把 20 Hz 每帧当独立姿态。
2. 每个完整的 `R` 窗口直接形成一个静态残差观测；姿态容差只用于识别并平均近似重复姿态，
   不再要求操作者提供方向标签。旧文件里的无方向标签窗口也按相同质量条件处理。
3. 第一轮正式拟合 `global-scale`，其 `effective_gravity_scale` 可直接用于后续显式
   `--gravity-scale` 复测；只有独立验证显示明显姿态相关残差且确认存在未建模末端负载，
   才评估 `global-scale-payload`（非负质量 + 末端一阶矩）。
4. 用 rank、奇异值/条件数和 log-volume 选择信息量足够的 pose；保留独立 holdout。
5. 输出绑定 calibration dataset SHA、URDF SHA、config SHA、模型、姿态数、rank、训练/验证
   残差，`activation` 必须为 `candidate_only_until_independent_validation`。

输出是有效重力修正，不是没有电流/力矩标定时的真实质量或质心测量。没有非零软件 Kp、
窗口不完整、秩不足或验证残差不下降，都应保持 `CANDIDATE/FAIL`。

## 11. 真机操作和终端行为规范

### 11.1 运行前硬性前置

- 有可靠机械承托；末端不压桌、不接触本体或其他物体；运动空间清空。
- 物理急停可立即触达，并已明确“触发/恢复”流程；观察员在场。
- 同一 `can0` 上没有第二个 CAN 控制 owner；关闭官方 demo、遥操、数采和其他 servo。
- 记录 AgenticRobot commit/tree、vendor commit/tree、YAML SHA、固件、电机型号/ID、CAN
  bit-rate 和 payload/支撑状态。
- 命令在 controlling terminal 前台运行；禁止 `&`、`nohup`、daemon、吞掉 controlling
  TTY 的管道或自动发送按键。任何不可预期运动都先按物理急停，再保留日志和进程证据。

### 11.2 shell 不崩和输出分层

探针的详细 stdout/stderr 由 `--log` 写入文件，终端只保留：

```text
[B601] mode=... pid=... log=...
[B601] STOP: Ctrl-C ...
[DRIVE] 3 / 2 / 1 / ENABLE
[B601] STARTUP ...
========================================================================
[B601] !!! PROTECTIVE HOLD LATCHED / DEGRADED / RECOVERED !!!
[B601] ARM STATE / CONTROL / DAMPING / FULL DISABLE ...
========================================================================
最终 JSON 摘要、结果路径、log 路径、shell remains active
```

包装命令不要使用 `set -e` 或 `set -euo pipefail`；所有失败都保存 `STATUS=$?`、打印结果
和日志路径，再让当前 shell 继续。不要为“看起来成功”吞掉非零退出码。

推荐的前台包装骨架：

以下代码是日志/TTY 包装骨架，不是可直接用于现场的控制参数；现场必须把经过评审的
`--mode` 和全部 profile 参数补齐，并再次核对 commit/tree。

```bash
WT=/absolute/path/to/pinned/AgenticRobot
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning/<run-id>
LOG="$OUT_DIR/command_and_output.log"
MODE=hold  # 现场替换为已评审模式，并补齐该模式所需的全部参数

mkdir -p "$OUT_DIR"
run_logged() {
  {
    printf '\n===== command start =====\n'
    date -Is
    printf 'cwd=%s\ncommand:' "$PWD"
    printf ' %q' "$@"
    printf '\n'
  } >>"$LOG"
  "$@"
  STATUS=$?
  printf 'exit_code=%s\n' "$STATUS" >>"$LOG"
  printf '[B601] probe status=%s\n' "$STATUS"
  return "$STATUS"
}

if cd "$WT"; then
  run_logged env PYTHONDONTWRITEBYTECODE=1 "$PY" \
    scripts/hardware/rebot_b601_rs_low_gain_probe.py \
    --mode "$MODE" \
    --config "$CFG" --rebot-source-root "$SRC" \
    --output "$OUT_DIR/result.json" --log "$OUT_DIR/result.log"
  STATUS=$?
else
  STATUS=86
  printf '[B601] ERROR: cannot enter worktree: %s\n' "$WT"
fi
printf '[B601] probe status=%s\n' "$STATUS"
printf '[B601] result=%s\n' "$OUT_DIR/result.json"
printf '[B601] log=%s\n' "$OUT_DIR/result.log"
printf '%s\n' '[B601] shell remains active'
```

`Ctrl-C` 第一次只请求 protective hold；第二次或 `SIGUSR1` 请求完整六轴失能。物理急停始终
是最高优先级；`safe_stop` 后仍需确认承重，不能用程序退出替代人员支撑。

## 12. Harness / Git 交接规则

### 12.1 共同原则

- 共享 worktree 先 inventory：分支、实际 HEAD/tree、dirty 状态、owner、声明 scope 和
  相关 peer；盘点不是锁，也不授予合并权。
- peer worktree 只读，禁止 reset、clean、stash、移动、删除或同步 peer 来制造干净状态。
- “ready” 必须绑定一个 clean、immutable 的完整 SHA、固定 base、实际验证命令、环境、
  已知 skips/未验证行为；任何新 commit、amend 或参与运行的未提交源码都会使旧 ready 失效。
- source code、offline/sim、remote、real hardware 证据分开呈现；沟通消息不等于 authority，
  旧 claim、旧 result、旧 hash 不能复用。

### 12.2 待合并（Harness/Task）

若本次工作由 Harness Task 管理，必须在 writer-capable canonical Harness 环境依次执行：

```text
inspect → conflicts → writer claim → renew while editing
    → scoped offline tests → diff-check → commit
    → recheck conflicts → release writer → handoff/quiesce
```

必须使用 Task 生成的 scope/lease/claim，不要再创建第二套 ownership JSON；不得直接改
`development`、不得自行 push/merge。人类真机验收必须绑定 exact claim、exact worktree、
exact commit/tree、observer、expiry、命令、输出根目录和通过条件。

### 12.3 合并后

合并完成后先重新核对 development 的实际 HEAD/tree、submodule gitlink、配置 SHA 和 clean
状态，再直接运行项目自己的 Python/pytest/field probe。Harness integration 只在仓库/发布
门禁明确要求时执行；不能因为历史上有 pre-merge Harness claim 就把同一段长 integration
test 当成每次合并后的真机必需步骤。合并后的直接 Python 运行仍须遵守第 11 节的现场安全
规则，且不能把 pre-merge claim 的证据路径当作 post-merge 证据。

### 12.4 当前分支元数据注意事项

当前独立分支 owner 为 `codex-root`，声明范围是 backend、probe、两份 focused test 和两份
B601 文档；排除 development merge、push、install 和 agent 真机动作。交接文档提交后，
branch description 必须更新为 exact tip、`ready-for-integration` 和本节验证结果。候选包含
此前多个 B601 安全修复 checkpoint，不能只摘取最后一个默认值提交而漏掉其依赖链。集成前
固定完整候选 tip、development tip 和 vendor/MotorBridge 输入；不得编辑、reset、clean、
stash、删除或移动任何同伴 worktree。

## 13. 可直接复用的验证命令

13.1/13.2 是离线检查；13.3 会只读连接 CAN 但不切 mode、不使能、不发控制帧；13.4 是
明确的真机运动采集，只有现场操作者批准后才能运行。

### 13.1 当前候选的 focused tests

```bash
cd /home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
/home/zc/anaconda3/envs/dexteleop310/bin/python3.10 -m pytest -q \
  tests/test_rebot_b601_hardware_backend.py \
  tests/test_rebot_b601_low_gain_probe.py
/home/zc/anaconda3/envs/dexteleop310/bin/python3.10 -m ruff check \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  tests/test_rebot_b601_low_gain_probe.py
git diff --check
git status --short --branch
```

当前核实结果：`167 passed`、Ruff PASS、diff-check PASS。测试不打开 CAN。

### 13.2 CLI/语法检查

```bash
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py --help
env PYTHONDONTWRITEBYTECODE=1 "$PY" -m py_compile \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py
```

全 B601 套件可按项目 wrapper 运行；若出现 MuJoCo URDF 缺失，先检查
`git submodule status hardware/reBotArm_control_py`。此前较宽候选运行结果是 `206 passed,
71 skipped, 3 failed`，3 个失败均源于任务 worktree 的 nested vendor submodule 未初始化，
不是本轮 exact tip 的全量 PASS；应记录为历史环境证据，不得改写成当前全绿。

### 13.3 只读 feedback stability

```bash
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-commissioning/feedback-stability-<run-id>
mkdir -p "$OUT_DIR"
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode feedback-stability \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --duration-sec 20 \
  --rate-hz 20 \
  --fault-hold-max-velocity-rad-s 0.02 \
  --confirmation B601_RS_FEEDBACK_STABILITY_READ_ONLY \
  --output "$OUT_DIR/feedback.json" \
  --log "$OUT_DIR/feedback.log"
STATUS=$?
printf '[B601] feedback status=%s\n' "$STATUS"
printf '%s\n' '[B601] shell remains active'
```

`status=OK`/`acceptance_status=CAPTURED` 只表示采样完成；还要看每关节 p95/p99、漂移和
`stationarity_exceed_count`。不要仅因采样噪声提高 active probe 的 fault-hold gate。

### 13.4 正式采集（真机，现场批准后执行）

先完成 round-trip，再运行下面的两批命令；训练和验证只改变输出文件，不改变控制参数。

```bash
WT=/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908
PY=/home/zc/anaconda3/envs/dexteleop310/bin/python3.10
CFG=/home/zc/HandMotion/Developing/reBotArm_control_py/config/rebotarm_rs.yaml
SRC=/home/zc/HandMotion/Developing/reBotArm_control_py
OUT_DIR=/tmp/b601-rs-gravity-calibration-<run-id>
mkdir -p "$OUT_DIR"
cd "$WT"

env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --duration-sec 900 \
  --warmup-sec 2 \
  --freeze-sec 1 \
  --rate-hz 20 \
  --calibration-settling-sec 2 \
  --calibration-duration-sec 20 \
  --kp 5.0 \
  --kd 0 \
  --driver-kd 1.0 \
  --gravity-scale 1.0 \
  --torque-limit-scale 0.40 \
  --apply-driver-torque-limits \
  --torque-rise-time-sec 0.5 \
  --velocity-abort-rad-s 5 \
  --acceleration-abort-rad-s2 40 \
  --max-error-rad 2 \
  --allow-motion \
  --auto-disable \
  --confirmation B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING \
  --fault-hold-max-age-sec 0.15 \
  --fault-hold-max-velocity-rad-s 0.02 \
  --fault-hold-max-rebase-rad 0.05 \
  --output "$OUT_DIR/train.json" \
  --log "$OUT_DIR/train.log"
STATUS=$?
printf '[B601] calibration status=%s\n' "$STATUS"
printf '[B601] shell remains active\n'
```

验证批次使用同一组参数并输出 `validation.json`/`validation.log`；不得复制训练姿态。
记录期间终端提示 `FREE_DRAG`、`CALIBRATION RECORDING`、`COMPLETE`，人手只在拖动阶段接触。

### 13.5 离线 candidate fit

```bash
cd "$WT"
env PYTHONDONTWRITEBYTECODE=1 "$PY" \
  scripts/hardware/rebot_b601_rs_low_gain_probe.py \
  --mode gravity-calibration-fit \
  --config "$CFG" \
  --rebot-source-root "$SRC" \
  --calibration-input "$OUT_DIR/train.json" \
  --calibration-validation-input "$OUT_DIR/validation.json" \
  --calibration-model global-scale \
  --calibration-pose-tolerance-rad 0.08 \
  --calibration-max-poses 12 \
  --calibration-max-velocity-rad-s 0.03 \
  --calibration-max-position-span-rad 0.02 \
  --calibration-overlay-output "$OUT_DIR/gravity_overlay_candidate.json" \
  --calibration-report-output "$OUT_DIR/gravity_fit_report.json" \
  --output "$OUT_DIR/gravity_fit.json" \
  --log "$OUT_DIR/gravity_fit.log"
STATUS=$?
printf '[B601] fit status=%s\n' "$STATUS"
printf '[B601] shell remains active\n'
```

fit 完成后必须检查：完整窗口数/不完整窗口数、独立姿态数、rank/奇异值、训练和验证残差、
饱和计数、dataset/URDF/config SHA 以及 `activation=candidate_only_until_independent_validation`。
未满足条件时保留报告，不改 URDF/YAML，不把 candidate overlay 作为运行时参数自动加载。

## 14. 历史真机证据如何解读

历史结果必须区分“现场观察”和“机器可重放验收”：

- `feedback-stability` 曾稳定在约 20 Hz，J2/J3 零位附近的微小负读数通过边界层投影；这
  不是高带宽触觉闭环证明。
- no-touch qpos/Cartesian/pose hold 曾完成完整六轴发送和 cleanup；它证明低带宽链路和
  终止路径可运行，不证明真实力矩/摩擦已标定。
- 2026-09-04/06 的单轴人工扰动在旧 `0.03 rad/s` gate 下出现 `0.098028 rad/s`，原始
  JSON 应保留 `FAIL`；操作者另行确认“人工单轴扰动现象已验收”。后续正常 qvel seed 使用
  `0.12 rad/s`，不联动放宽 qacc 或 fault-hold stationarity。
- J2/J3 边界观察和 J4 精确 `±pi/2`/0.001 rad 兼容窗口只作用于反馈接纳；J4/J5/J6
  另有各 `0.1 rad` 的真实反馈工作余量，但不放宽出站命令。曾出现的 J4 巨大非有限/异常
  数值必须 fail-closed，不能通过扩大 clamp 消除。
- 2026-09-08 type-17 调试运行完成 4 个有效窗口后，J6 从 `2.774` 连续移动到
  `3.232712 rad`，其中触发旧上界的 `3.142859 rad` 来自 status=0、device=6、host=0xFD、
  parameter=0x7019 的正常回复，不是单帧通信伪值；这组证据支持上述 J6 工作余量。
- 同一次 4 窗口数据每窗 399 帧、最大窗口速度 `0.0077 rad/s`、最大位置跨度
  `0.00058 rad`，且没有 torque/slew saturation，可保留为训练候选。`global-scale` 离线拟合
  只给出 `1.0 -> 0.9999686` 和约 `0.00052 N.m` 的原始 RMS 残差，且没有独立验证批次；
  因此它不支持现在修改运行时重力比例、joint bias 或 payload 参数。
- `R` 在操作者松手后把当时位置设为 reference；按键前已发生的轻微下坠，以及被静摩擦
  挡住而没有形成位置误差的重力残差，都不会出现在 `position_tau=Kp*(qref-q)` 中。因此这
  4 点不能解释启动/自由拖动阶段的轻微前倾，后者仍应通过启动轨迹或实际电流/力矩观测定位。
- 第二批 4 个全新姿态也各完成 20 秒有效窗口，作为独立验证集时与第一批最近姿态的 L2
  距离为 `1.09..1.95 rad`。`global-scale` 只得到 `1.0 -> 0.9999686459`：训练 RMS
  `0.5248 -> 0.5173 mNm`，验证 RMS `1.1275 -> 1.1020 mNm`，绝对改善仅
  `0.0255 mNm` 且方向是略微减小补偿。joint-bias 和 payload 模型在验证集上均变差，
  payload 质量落到 `0 kg` 边界。因此这些 overlay 只保留为离线诊断，不得接入真机；
  也不应继续采集大量同类静态窗口来试图解释启动前倾。
- 更早的交互采集曾完成 1–4 号窗口，后续窗口 5 在 duration 边界前未完成，或旧版本因 J4
  四舍五入边界中止；这类 Kp=0、窗口不完整或来源不清的旧数据只可诊断，不能与上面本次
  4 个有效 Kp=5 窗口混为同一正式输入。
- protective hold 的人工验收重点是：首次 Ctrl-C 后 J1–J6 仍 enabled、gravity+damping
  可拖动；再次 Ctrl-C/SIGUSR1 才完成六轴 disable。程序因为操作者中断而返回 FAIL 时，
  仍须阅读 hold/cleanup 字段再判定安全行为。

建议保留的原始输出根目录示例：

```text
/tmp/b601-rs-commissioning/
/tmp/b601-rs-gravity-calibration-<run-id>/
/tmp/b601-rs-degraded-protective-hold-r1/
```

不要把临时 JSON/log 写回源码 worktree，也不要删除失败证据来制造“全绿”。

## 15. 结果呈现模板

交接/验收消息按以下顺序呈现，便于下一位开发者复核：

```text
结论：OFFLINE_PASS / HARDWARE_PASS / FAIL / NOT_RUN
阶段：<stage>
操作者/观察员/时间：<...>
AgenticRobot：HEAD=<full sha> TREE=<full sha> clean=<yes/no>
vendor：commit/tree=<...> config_sha256=<...> firmware/model/IDs=<...>
命令：<完整可复制命令>
输出：json=<absolute path> log=<absolute path>
控制：mode=<...> active_joint_names=joint1..joint6 active_vendor_kp=0 temporary_lock_vendor_kp=<...>
启动：state_history=<...> arming_settle=<...> handoff=<...>
保护：first Ctrl-C=<hold/degraded> second Ctrl-C/SIGUSR1=<safe_stop>
频率：feedback/compute/tx 的 count、observed Hz、p95/p99、deadline miss
运动：q/qvel/qacc、gravity/damping/overlay/final tau、限幅计数
异常：stop_reason、rejected_sample、degraded.reason_counts、escalation_reason
清理：cleanup_receipt.completed、full_disable_receipt、restore receipt、cleanup_errors
人工观察：方向、下坠、前倾、振荡、J6 回中、阻尼手感
未验证/阻塞：<明确列出，不用“看起来正常”替代>
```

机器字段与人工观察必须分别写出：`hardware_acceptance=NOT_CLAIMED`、离线 PASS 不等于
真机 PASS、`status=OK` 不等于重力模型已标定。

## 16. 下一位接手者的最短路线

### 16.1 当前交接包

| 项目 | 固定内容 |
| --- | --- |
| AgenticRobot worktree | `/home/zc/HandMotion/Developing/temp/worktrees/root-b601-gravity-calibration-runtime-20260908` |
| branch | `task/root/b601-gravity-calibration-runtime-20260908` |
| 默认 profile 代码 checkpoint | `70fcb2a7ba690cbc733c586b11633738b9f1b8b1` |
| development（本次只读快照） | `028b6d28fb9056843959cceb074af8f0a6b3898a` |
| merge base | `ef54a901bb5df92d9926c7c47ef7def5c1ddfdf5` |
| MotorBridge candidate | `0c8980afeeaf74bbd6d7329027d0af4a998b77e9`，tree `b34e138748ba4d21113a2653ed53fcce6a602571` |
| MotorBridge ABI SHA-256 | `85466a8e78532fa8194e01e59d056b36eeaea1e68c8c290cdfcb78c24dc1497f` |
| reBotArm vendor | `5ba28acef46237eb6a7560658bbc43b06cf8a259`，tree `ddd34d2b400b3987d4875bd6a99bd29b74d39dac` |
| YAML SHA-256 | `be51e877050cc4716188fca3b4f2a41109682513c7715c71594066a133ba5cf9` |

当前 worktree 除历史保留的
`.playwright-cli/page-2026-09-08T10-30-40-740Z.yml` 外没有源码修改；该文件不属于 B601，
不得删除或提交来制造“全净”。交接文档提交后应以最新 `git rev-parse HEAD`、tree 和 branch
description 替换上表的代码 checkpoint 进行最终集成 pin。

### 16.2 本轮实现完成项

1. 六轴同步 enable/dispatch/receipt，J7 明确排除；部分发送不冒充完整成功。
2. 驱动器本地临时锁完成 `ENABLED_UNARMED → LOCKING → LOCKED → GRAVITY_RAMP →
   HANDOFF → ACTIVE`，完整重力从首个锁帧参与，总力矩只经过一层 slew。
3. 第一次 Ctrl-C、普通异常和 duration 完成优先进入 J1--J6 gravity+damping protective
   hold；反馈/模型暂时不可用时使用最后完整补偿缓存和既有 motor-local Kd。只有第二次
   Ctrl-C/SIGUSR1，或 live 与 cached 完整六轴发送均失败，才允许最终全失能。
4. J2/J3 零位负零窗口、J4 精确 `+/-pi/2` 与 J4--J6 `0.1 rad` 只读工作余量已经分离；
   出站命令仍使用 nominal URDF 包络。
5. 单帧有限但不可信的 qpos 跳变先隔离，下一帧恢复则继续；非有限值、连续异常和无法
   建立可信控制状态进入 protective/degraded 路径，不能靠放宽运动阈值掩盖坏帧。
6. MotorBridge `robstride-startup`/type-17 被动原始帧记录不增加 CAN 请求；host-id 相关
   ABI、参数 ID、raw float bits 和关节诊断用于定位偶发 J4 参数响应异常。
7. `gravity-calibration`、`startup-replay` 和离线 fit 均保留；静态 R 窗口不能分离重力
   bias 与静摩擦的限制已经记录。
8. 重力运行时支持六轴 `scale/offset/bias/Coulomb`，Coulomb 只在有限差分速度超过显式
   threshold 后按方向生效；所有项贯穿 startup、ACTIVE、protective hold 和缓存补偿。
9. `--driver-kd` 兼容一个共享值或六个独立 J1--J6 值；软件 Kd 与 driver Kd 不允许同时
   成为 damping owner。
10. 操作者确认的默认正常重力 profile 已写入程序；省略相应 CLI 时使用第 0 节列出的
    数值，结果 JSON 的 `profile` 记录最终生效向量。

相关 checkpoint 顺序：

```text
6dd00466 passive startup diagnostics
55b76fe6 continuous startup replay
97473f0e stable driver-local startup lock
f803708f reversible per-joint gravity scale
ffc2bbd2 gravity offset/bias/Coulomb runtime terms
77ab9834 independent per-joint driver Kd
70fcb2a7 operator-validated gravity defaults
```

### 16.3 验证与证据边界

- 当前候选的 B601 相关离线回归得到 `476 passed`（1 个既有 hppfcl deprecation
  warning）；Ruff、`py_compile` 和 `git diff --check` 通过。该结果不包含真机动作。
- `/tmp/b601-rs-j2-gravity-scale-095-20260909-173751/`：J2 scale `0.95` 的完整 startup
  replay 状态机通过，约 20 Hz，无 torque/slew limit；最终顶层 `FAIL/operator_interrupt`
  是第二次 Ctrl-C 的记账，不抹去 replay 子结果和完整 cleanup。
  JSON/runtime/raw SHA-256 分别为 `381f3be1829ae72a07982a1611762d77a653d97c2c67470a55604263edbb4d68`、
  `7e70afccaaaf3e5ca6da8bd1977f70848429bda230ae99eeb9693dedcfd21ba4`、
  `dc8027306559cfec7b91a5d266a0bf5bd3cf5db467873fcaa72554583a3d5224`。
- `/tmp/b601-rs-j2-bias-minus010-r1/result.json`：历史机器证据确认当时实际下发的旧 profile 为
  driver Kd `0.2 0.3 0.2 0.15 0.1 0.05`、joint scale `1 0.95 1 1 1 1`、bias
  `0 -0.105 -0.1 -0.05 0 0`，protective hold 和最终六轴 cleanup 完整。该次没有按 R，
  所以 `startup_replay.status=FAIL`；它不能冒充 replay 验收，也不能作为当前默认 profile
  的验收证据。当前默认 profile 是操作者在 2026-09-22 调整后确认的配置；其 exact
  handoff commit 的真机重放仍需单独取得证据。
  JSON/runtime/raw SHA-256 分别为 `55f2d0575350e9233ff062dddf4b256b082f8005e7429b355c04b7f0c4cfd582`、
  `92a190c930de0f596d80e0e3ad28d795719f82bc182664d0cc03606c6b587f9b`、
  `4d2802bcc33b15fdedaee88c2cc985a941ad35fdafc373ba4ba8d26e3146a796`。
- 默认值提交本身只改变“省略参数时取什么值”，尚未在 exact handoff commit 上做一次
  省略所有重力 profile 参数的等价真机重放。若集成前要求严格机器可重放证据，应补这一
  次短 no-touch 运行，而不是重新标定全部参数。
- 旧静态 calibration 窗口和官方 PD sweep 可作为诊断/先验，但不能证明本机 offset、
  Coulomb、MIT Kp 力矩比例或 payload 已被独立辨识。

### 16.4 仍未完成或不得扩张的结论

- 未实现自动双向低速 sweep 来分离 `k/bias/Coulomb`；当前 Coulomb 默认全零。
- 未对新增默认 profile 做多负载、多温度、末端工具变化和全工作空间定量验收。
- 未完成高带宽末端触觉、passivity、力传感器闭环或系统级 teleoperation 验收。
- 未修改 URDF/YAML/固件持久参数；不得把 runtime bias 反写成虚假的 link mass 或 joint
  zero。
- 未合并 development、未 push、未安装，也没有授权 agent 运行真机。

### 16.5 下一位接手者的最短路线

```text
1. 读取本文、branch description 和当前 HEAD/tree；保留无关 untracked 文件
2. 复跑 167 项 focused tests、Ruff、py_compile 和 diff-check
3. 只读确认 development 与候选的共同祖先、声明路径差异和 merge preview
4. 若用户要求严格 premerge 真机证据，在承托/急停/唯一 CAN owner 下用默认参数省略方式
   做一次短 startup-replay/no-touch 等价验证；不要重复大规模静态窗口采集
5. 获得明确合并授权后才创建隔离 integration composition；合并固定提交并复跑验证
6. 合并后可直接使用目标 Python 环境运行，不再套 Harness runtime；Harness 只用于尚未合并
   composition 的受控验收
7. 新 payload、摩擦辨识或 protective-hold 独立大阻尼属于下一任务，不能夹带进本次合并
```

关联的细节文档：

- [`rebot_b601_rs_hardware_commissioning.md`](rebot_b601_rs_hardware_commissioning.md)：阶段化
  commissioning、完整命令、频率字段和现场输出规范；
- [`rebot_b601_rs_haptic_acceptance.md`](rebot_b601_rs_haptic_acceptance.md)：离线/MuJoCo 与
  H1–H8 真机验收边界；
- [`rebot_b601_hitl_topology.md`](../architecture/rebot_b601_hitl_topology.md)：vendor gitlink、
  Pinocchio 模型和上层 HITL ownership 边界。

任何阶段遇到不可解释运动、方向反转、持续下坠、振荡、第二个 CAN owner、日志/版本身份
缺失或 cleanup 不完整，立即停止该阶段并保留原始证据；不要靠重复运行或临时放宽阈值掩盖
根因。

## 17. 启动异常恢复候选（2026-09-10，离线验证，真机暂停）

本增量基于 `1de3579c457758d14c6a9b2b7dab3362a53232f4`，分支
`codex/b601-startup-hold-recovery`；原 gravity 标定和 real2sim 候选保持不变。
具体候选 tip 与交接状态以该分支描述为准。本节不是合并或真机通过声明。

使能完成但尚无完整重力发送缓存、也没有新鲜已接纳状态时，原清理逻辑可能直接
走失能。新增恢复步骤最多读取两次反馈：整个读取窗口不得超过现有
`fault-hold-max-age-sec`；六轴必须完整、有限、在反馈限位内，有限差分速度不得超过
`fault-hold-max-velocity-rad-s`，相对使能前位置的每轴偏移不得超过
`fault-hold-max-rebase-rad`。旧位置仅用于偏移比较，不被冒充为当前位置或已发送缓存。
通过后复用原六轴零位置刚度的重力/阻尼 protective hold。已有新鲜状态或完整发送缓存
不经过这次额外采样；原默认重力参数、阻尼参数、力矩上限及出站限位不变。

明确全失能请求在恢复读取前后检查，进入 hold 时再次检查，不因恢复而额外发送首个
保持帧。正常 duration 完成和第一次停止事件仍进入保持；**如果已经因 duration 或
异常进入保持，此后的第一次 Ctrl-C 就是最终全失能请求**，不是重新开始计数。

离线验证：`tests/test_rebot_b601_low_gain_probe.py` 与
`tests/test_rebot_b601_hardware_backend.py` 共 **228 passed**，Ruff、diff-check 和
local-resource audit 通过。故障注入覆盖使能返回时的第一次 Ctrl-C、首个 RX 失败后恢复、
持续 RX 失败、坏数/越界/不完整/过快/过大偏移/读取超时、显式最终停止。
调用链测试使用 fake backend，并替换无期限 hold 循环检查入口和清理顺序；已有 hold
循环测试独立覆盖 live/cache、完整发送失败和释放语义。不能据此宣称硬件托举效果通过。

### 仍阻止无支撑使能实验的边界

- 使能前预检仍有显式六轴 `safe_stop`；不得用新进程接管正在保持的机械臂。
- 如启动后持续无法得到可信反馈且从未成功发送重力帧，本增量仍不能构造可信的
  重力保持，保留现有失能兜底；没有偷偷取消该保护，也不会盲发旧 q 或假重力。
- 部分使能失败、模型持续失败、总线发送失败和进程/供电丢失不在“始终托住”的保证内。
- MotorBridge 0.4.9 缺少设备源时间戳：本地读取时限不是硬件 freshness_verified。

因此当前修复只消除了**反馈仍可恢复时不必要的启动失能**，没有兑现“任意启动故障
绝不失能”的绝对承诺。下一次使能实验必须先确认可靠机械支撑、现场急停和唯一 CAN
控制进程；若要求无支撑并覆盖持续失联，需要另行确定驱动器本地保持/制动方案及其
硬件证据，不能仅靠放宽软件阈值实现。真机仍由操作者前台执行，详细输出写 log/JSON，
不得用 timeout、后台运行或关闭终端代替显式最终停止。

## 18. 末端六轴人工采集引导候选（2026-09-10，离线验证）

`pose-hold` 新增可选的 `--guided-pose-axis-protocol`。它只在 startup handoff 完成、状态
进入 `ACTIVE` 后开始，在前台终端依次提示：静置基线、`+/-X`、`+/-Y`、`+/-Z`、
`+/-Rx`、`+/-Ry`、`+/-Rz`，且每个动作后都有明确的松手回中采集阶段。坐标约定固定为
base/world-aligned 的 X 前、Y 左、Z 上，旋转遵循右手定则。

该协议是**操作员驱动的采集标记层**：不会生成末端增量、关节轨迹或新的 `qref`，不会
改变现有 `J.T @ wrench`、重力、阻尼、力矩调理、启动倒计时和 protective hold 路径。
每个已发送控制样本增加 guide phase、phase index 和 phase elapsed；结果 JSON 另记录完整
schedule、transition events、开始/完成时间以及 `operator_only_no_automatic_reference`
语义。引导目标必须严格小于既有平移/旋转 abort envelope，运行时长必须覆盖 startup
handoff 与完整引导，否则使能前 fail closed。

默认操作目标为每个平移方向 `0.03 m`、每个旋转方向 `8 deg`；静置 `5 s`，每个动作
`6 s`，每次回中 `4 s`，完整采集 `125 s`。这些值只用于终端提示和证据标签，并不是
位置命令或对实际位移达成的声明。当前候选没有运行真机；真机采集仍须操作者前台执行，
观察异常立即第一次 Ctrl-C 进入重力/阻尼保持，确认安全后再次 Ctrl-C 全失能。

## 19. 交互式末端六轴人工采集候选（2026-09-10，离线验证）

`pose-hold` 的 `--interactive-pose-axis-protocol` 用操作员按键取代固定时间切换。startup
handoff 进入 `ACTIVE` 后，程序先保持六轴重力补偿和 driver-local 阻尼，但不启用末端
spring，也不保存启动位置为任务空间锚点。操作员可用任意合理时长把机械臂拖到适合测试
的位置；完全松手并静止后按 `A`，程序才用该周期的新鲜六轴反馈做 FK，并把当前位置保存
为阻抗锚点。

锚点采集要求 `max(abs(dq)) <= fault-hold-max-velocity-rad-s`；不满足时只记录拒绝事件并
停留在 `POSITIONING`，不会使用旧位置或错误 q。锚点成功后，十二个方向仍按 `+/-XYZ`、
`+/-RxRyRz` 顺序进行，每项使用不同按键隔离状态：`N` 开始本项、人工缓慢移动并短暂停留、
`D` 结束动作、松手等待回到锚点附近并静止后按 `R` 进入下一项。`R` 同样经过上述六轴
静止门禁；同一批次收到多个键最多只产生一个状态转换。

协议本身不生成自动位移、旋转或关节参考。交互期间 nominal `qref` 每周期与当前新鲜 qpos
共置，且 handoff 后 vendor/software joint Kp 均为零；锚点前的 task-space overlay 为零，
锚点后唯一恢复项仍是既有有界 `J.T @ wrench`。结果 JSON 保存锚点 qpos/SE(3)、每次状态
转换和拒绝、每样本状态及相位时间。完成最后一次静止 `R` 后正常控制自动退出到既有六轴
重力/阻尼 protective hold；此时下一次 Ctrl-C 或 SIGUSR1 才请求全失能。运行总时长只是
交互超时，不驱动阶段切换。

本增量不改变重力 profile、driver torque limit、速度/加速度 abort、位姿 displacement
envelope、启动锁、异常降级或二段式退出。离线 fake-backend/状态机测试不能代替真机手感、
方向或力标定证据；真机仍只允许操作者在前台执行。
