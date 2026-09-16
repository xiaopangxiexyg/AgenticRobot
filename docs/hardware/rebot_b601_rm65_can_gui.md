# ReBot CAN → RM65 MuJoCo GUI 装配进度

本页记录应用接入进度，不修改总路线图。当前是 **离线验证通过、不可真机验收、未 handoff**。
它不是新的重力补偿程序，也不证明硬件遥操已经可用。

## 实现与所有权

统一入口为 `python -m control_station b601-rm65`，应用叶子是
`control_station/sim_hil_profile/rebot_b601_rm65_application.py`。

- `assemble_b601_rm65_session()` 只绑定已存在的 backend/model/profile/target；不启动、关闭或驱动注入的资源。
- 复用 `B601ReadOnlyPoseSource`、canonical frame-mapping profile、residual adapter、motion binding/session。
  所有映射必须来自显式 profile；缺少 canonical 模块时在打开 CAN 之前失败，禁止复制实现或默认为 identity。
- 仿真准备复用 `prepare_mujoco_sim_hil_target()`，禁止 task-specific scripted reset；不启动 policy。
  复用 actuated position rig、canonical coupled IK 和原有速度/加速度/关节/碰撞保护，不使用 direct-qpos debug。
- 两臂测量使用 rig 的一次 lock-consistent execution cut 和 private-MjData FK projector。
  保留原测量时间及来源 hash；不是从旧目标或 GUI 缓存反推当前姿态。
- GUI 复用 `MujocoPassiveViewerSession`，由唯一 rig 的 `sync_viewer()` 同步；GUI 不推进物理或重建 IK。
- 前台交互复用已有 terminal loop：`a` 激活、`c` clutch、`r` 重新锚定、`s` 撤销跟随、`q`/EOF 退出。
  初始 inactive，操作者先把 master 放到合适位置、静置，再激活；当前没有反向 haptic channel 或 policy overlay。

CAN backend 以默认 `allow_motion=False`、`allow_active_reporting=False` 创建，只接收**独立 owner 已开启**的
完整新鲜六轴反馈。它不使能、不切模式、不配置 report、不发送 MIT/position/torque、不操作 J7。
现有 master 重力补偿/阻尼 owner 必须独立运行；应用退出仅撤销仿真跟随并关闭自己的只读连接。
绝不能把 master `safe_stop()` 加入应用的普通 `finally`。

`--validate-only` 不打开 CAN/GUI，不启动 worker，只构造、校验契约和模型；
`--connect-read-only` 必须提供 `--confirmation B601_RM65_READ_ONLY_CAN_SIM_TARGET`。
这只是连接确认，不是 motor/report/motion 授权。
日志显式用 `--log` 指定，可追加实际 argv、状态和错误 traceback；结果 `--output` 必须是尚不存在的新文件。
JSON 的 `target_cleanup_completed` 只有 lower runtime 的 `close()` 成功返回才设为 true。

## 已修复的时序缺陷（2026-09-16）

初次标准离线回归为 **50 passed、2 failed**；失败没有被忽略或改成 skip。

1. 激活/重新锚定发生在已到期 tick 时，现有前台循环立刻再次 `session.step()`。
   一次缓存尚未更新就被二次读取，新鲜度推进检查正确拒绝，session 进入 held。
   原前台 owner 现在将激活/recenter 后的首个发布安排到下一个周期；不绕过重复帧检查。
2. `B601RM65MotionSession.step()` 在 residual adaptation **之前**取得默认 target time；
   binding 随后再取真实 monotonic clock，严格的 `now <= target_effective_at` 正确拒绝过去时间。
   synthetic 固定时钟隐藏了此问题。本轮真实 MuJoCo rig/时钟测试已复现。
   现在由原 admission owner 在最终校验时选择默认生效时间；显式时间验证、原 sample TTL 和单次 source poll 均保留。

修复需要原文件：

```text
control_station/sources/rebot_b601_rm65_motion.py
control_station/sim_hil_profile/rebot_b601_rm65_foreground.py
tests/test_control_station_rebot_b601_rm65_motion.py
```

上述路径经用户明确授权后由原生 `scope-extend` 纳入当前 Task；未手改 Task/coordination 元数据，
也没有在 GUI 层增加第二套 session 来绕过门禁。修复后联合回归为 **54 passed**。

当前基线 `37077731` 同时缺少 development 已有的 canonical frame-mapping profile；
最终组合需包含 predecessor 和 profile 两项，不能从其他 worktree 的绝对路径动态导入 tracked 源码。
predecessor 未集成只阻塞最终 integration，不影响当前 scope 内开发。

## 当前离线复核命令

在当前 Task 源码根，使用标准项目 Python 和 approved asset binding：

```bash
bash scripts/run_pytest.sh -q \
  tests/test_control_station_rebot_b601_rm65_application.py \
  tests/test_control_station_rebot_b601_rm65_motion.py
```

当前预期为 **54 passed**。
测试覆盖 injected owner 生命周期、完整当前姿态锚点、GUI close/fault、lazy CLI help、
参数/confirmation 拒绝、离线模式不连接 CAN、日志 argv、Ctrl-C、target cleanup，以及真实模型 actuated rig 的 atomic FK measurement。
外部构造在生命周期测试中 mock；坐标 profile 用 synthetic port，不代表真实安装姿态标定。

## 后续验收顺序

1. 在完整 clean commit 上运行 motion/sim 及 impact-added Profiles；全部通过后才可 handoff。
2. 由 integrator 组合 predecessor、canonical mapping profile 和应用候选，重跑 exact combined-source 门禁。
3. 提供现场操作者的完整 foreground 命令：先离线 validation，再独立 master gravity/damping 与新鲜 report 预检。
4. GUI 首先 inactive，确认 RM65 两臂模型及 TCP；`a` 后小幅平移、逐轴旋转，确认只有选中右臂按相对运动跟随。
5. `c` 后移动 master 不应改变 follower；`r` 从双方当前实测姿态重新锚定，不得跳回旧目标。
6. stale/坏帧/越界/GUI fault 撤销仿真跟随并要求重新预检；`s`、`q`、Ctrl-C 不得影响 master 的重力补偿。
7. master 的首次 Ctrl-C protective hold 和再次 Ctrl-C/SIGUSR1 全臂掉能仍只由其原硬件控制 owner 处理。

真机和 GUI 观察均为 **NOT_RUN**。组合依赖和 Profiles 完成前不提供或执行 live CAN 验收命令；不合并、不 push、不安装。
