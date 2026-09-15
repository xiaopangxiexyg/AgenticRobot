# ReBot → RM65 MuJoCo 执行接入进度

更新日期：2026-09-15。本文记录阶段事实，不修改总路线图，也不授权真机操作。
入口由 [硬件文档](../../hardware/README.md) 索引，避免与上一批待集成 contact 候选的
报告索引路径并发写入。最终候选身份、正式 Run 和集成状态以 Harness receipt 为准。

## 1. 目标与本批边界

主线仍是：已验证的 ReBot 重力补偿/阻尼与 protective hold → ReBot 真机遥操 RM65 sim
→ RM65 policy 传导到 ReBot sim → 当前 execution tick 的 human residual → 双方真机闭环。
MuJoCo 只做合理的大致表达，不以精细辨识拖延真机主线。

本批 Task 为 `AR-20260915-b601-rm65-mujoco-execution-r0`，基线为
`e6b8265221d5df56e9c937d5edfd1beef691297b`。代码检查点为
`ee9991a234537f5135dd7a333b44bb44c079eff4`（tree
`deaa474038577c414ddb9dd6fcb33610b53ca987`）。该检查点不是本报告提交后的最终 HEAD。

本批新增 `control_station/sources/rebot_b601_rm65_motion.py` 和对应测试；未改变既有
硬件后端、URDF、重力 profile、启动状态机、protective hold、shadow 或 policy 逻辑。
既有 composition 的相对 pose 函数只改为复用同一 `robot_motion` 下层运算，数学行为不变。

## 2. 已实现的单源执行接缝

```text
B601ReadOnlyPoseSource → B601RM65ResidualSource → bounded human pose proposal
→ HumanProposalAdapter → MujocoDeployRuntime authoritative admission
→ canonical coupled IK/collision safety → qref/qcmd planner → dispatch → measurement
```

- `B601RM65MotionBinding` 默认关闭，通过 backend-neutral typed target port 接入显式 single-human target。
- 反馈、FK、frame mapping、anchor-relative residual 与 proposal 均复用现有实现。
- 显式要求既有 component execution handoff；不借用 policy/agent-skill producer 身份。
- binding 不启动设备，不拥有 runtime lifecycle，不跑独立 IK、插值、仿真或 dispatch。
- 调用方拥有采样、activation/clutch/recenter；目标 runtime 是唯一 execution owner。
- RM65 两臂 coupled schema 使用完整测量 anchor，非目标臂保持其测量参考。
- 平移在 reference frame 中相加；旋转在 baseline-local 中右乘。运算由 `robot_motion.action.relative`
  唯一实现，既有 composition 只作转发，单源入口不为此加载 composition runtime。
- anchor 绑定 gate epoch；目标 baseline 必须与测量值经原生 schema dtype 转换后精确一致。
- 校验源时间、receive freshness、序号、source binding、schema、anchor lineage、范围和 TTL。
- 错误输入使 RM65 sim 进入 hold 并撤销旧 proposal/segment；后续合法样本不能自动恢复。
- 显式 ready 后仍须重新捕获 master 与 target anchor，不能复用旧 gate epoch。

这些职责来自 `debug-teleop-ik-motion` 的逐层证据与单 owner 契约；不是 shadow → Active
的模式开关。Admission receipt 不是 dispatch receipt，排队 qref 也不是测量运动。

## 3. 已观察的离线证据

- 新文件 focused：18 passed，包含错误参考/emission 拒绝、recenter、非交换旋转与测量更新。
- 源/residual/shadow/session/边界相关回归：73 passed。
- 架构修复后，新执行接缝、source/target contracts、station/composition/runtime 边界：108 passed。
  首轮正式 sim/station 曾因 source 直接导入 sim 与数值库而失败；已修复代码，未放宽边界断言。
- 下层 relative action、既有 formal/component composition 与新接缝回归：93 passed。
- Ruff check 与 `git diff --check` 通过。
- local-resource audit 通过。标准 pytest wrapper 临时绑定 reviewed RM65 assets，退出后清理链接；
  没有复制、修改共享资产或添加依赖。
- 真实 RM65 MuJoCo model、canonical collision IK、原生 target handoff/dispatch 检查通过，
  测量关节确有有限、小幅变化。该检查使用 direct-qpos diagnostic execution，
  **不证明 torque/contact dynamics、ReBot 硬件响应或真实 operator acceptance**。
- B601 侧使用合成六轴 feedback/FK；不能称为真机 → sim 遥操 PASS。
- 有既有 OSQP pending-deprecation warning，未为消除告警改变要求或关闭保护。
- 正式 motion/sim/station Profiles 应对最终候选重新运行；结果以 exact snapshot Run evidence 为准。

可从该候选 source 根目录使用项目 Python 与标准 wrapper 重跑：

```bash
DEXTELEOP_PYTHON=/absolute/path/to/project/python3.10 \
DEXTELEOP_ASSET_ROOT=/absolute/path/to/reviewed/assets \
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
bash scripts/run_pytest.sh -p no:cacheprovider -q \
  tests/test_control_station_rebot_b601_rm65_motion.py
```

未绑定资产时真实模型测试明确 skip；合成契约测试仍可运行。该命令不连接硬件。

## 4. 后续落地与真机暂停点

1. 在现有应用 lifecycle 中装配本 binding、显式 ReBot/RM65 frame/TCP 配置与 target measurement。
2. ReBot ingress 默认只读；使能/重力补偿仍由既有已验收硬件 lifecycle 拥有。
3. 接入前台 activation/clutch/recenter、简短状态提示、统一 command/output log 与 JSON。
4. 提供现场操作者运行的 exact-source 命令：写清预期行为、失败判据、结果位置与停止方法。
5. 从当前合理位置开始，先无触碰/零 residual，再小范围平移、旋转与 recenter。
   核对方向、跟随测量、延迟、越界拒绝、旧参考不恢复；J7 独立，不混入六轴 arm。
6. 保留原回退：普通异常或首次 Ctrl-C 保持六轴 gravity+damping；再次显式请求才全掉能。
   本 binding 的 RM65 sim hold 不等于 ReBot 全臂失能。

现阶段没有已装配的完整真机 → sim CLI，不能把单元测试命令当成硬件验收命令。
本批没有自行进行真机操作、GUI 观察、合并、push 或安装。

## 5. 前台执行会话增量

2026-09-15：操作者明确同意恢复同一 Task。原 `0aa867e4` 候选的交接资格已撤回，
本轮最终提交、tree 和重新 handoff 状态以新的 Harness receipt 为准。不得沿用旧 Run
冒充本轮最终 source 的证据。

在同一 source 模块内补齐 `B601RM65MotionSession`，没有新增平行入口或配置文件：

- 默认 inactive；`activate()` 显式重捕 master 与完整 RM65 测量 anchor。
- 前台调用 `step()` 时只 poll 一次；adapt 时间在 poll 完成后取得，不提前盖时间戳。
- 调用方明确提供当前执行 tick 的生效时间/期限，并继续推进已有 target runtime。
  session 只经 binding 提交 human proposal，不执行 IK、插值或 dispatch。
- `engage_clutch()` 将 RM65 置 hold 并清空 anchor；clutched/inactive/held 时不采样或发布。
- `release_clutch()` 从两侧当前位置重捕。移动 master 的绝对距离不进入释放后的 residual；
  静止释放后的第一帧 residual 为零，非目标臂仍来自完整测量参考。
- freshness、domain/frame、schema 值与范围验证仍只有 binding 负责。全部 anchor 验证完成后，
  仅显式 operator recenter 能把 hold 改成 ready；不能降低 lower-owner fault/transitioning。
- 错误输入清除会话参考，合法输入不会自动恢复。采样/测量 provider 异常及 Ctrl-C 先清理
  target，再重新抛出原错误，不吞掉故障。下层取消失败也不能伪装为 hold 成功。
- 不创建、关闭、使能或失能 B601；RM65 hold 与独立 ReBot gravity/damping lifecycle 分离。
  既有六轴重力 profile、启动锁、protective hold 与 J7 边界完全未改。

专项检查为 28 passed，包含原真实 RM65 model/canonical IK 测量检查，以及 clutch 取消、
双侧 recenter、拒绝后显式恢复、坏/过期测量不重开 gate、异常/Ctrl-C 清理、disabled 零副作用、
fault/transitioning 不降级。首轮合成 fixture 因同一时间重复采样被正确拒绝，已让测试时间
真正推进；没有降低生产端的 freshness/单调性要求。

仍未装配完整真实 CAN → GUI 的前台 CLI；上述会话是其可独立离线验收的执行生命周期接缝。
下一步在既有应用 leaf 内绑定真实只读 source、明确 frame/TCP profile、target measurement 与
前台按键/日志。master 重力补偿保持独立、已验收的 owner，不通过普通 `finally` 自动失能。
真机命令必须在该装配及失败清理验证完成后另行提供；本轮专项结果不是真机遥操 PASS。
