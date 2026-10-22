# RTC 集成前置阶段：后台资源所有权与退出屏障

本阶段基线为 `development` 的 `e8a73bbb083269436aa1e7fcd7dd85ce88dd30b8`。
RTC 核对输入为 `task/root/rtc-feasibility-recovery-20261015` 的
`4335a2b7ea643a46c71f7a7d19f4c47cb162c5bf`。这是后续 RTC 接入所需的
资源生命周期修复，不表示 RTC 整条功能分支已经合并或完成真机验收。

## 资源获取与保留

- `RM65OfficialAllIKGovernorPlanner` 的构造只创建本地状态，不启动线程或进程。
  需要异步 C2 预计算的调用者必须先保存 planner，再调用 `start_background_workers()`。
  普通同步规划不需要后台工作。部署 executor 在登记 factory 返回值之后统一调用启动端口；
  启动端口必须有对应的 `shutdown_background_workers(timeout_sec=...)`。
- planner factory 必须返回资源所有者之后才启动后台工作。factory 抛出异常而没有返回所有者时，
  executor 无法证明其是否偷偷获取了资源，准备失败保持 sticky，停止证明保持未知。
  自定义 factory 不得在这种状态下重建或重新接入同一 backend；必须查明并回收原有资源。
- `OwnedThread` 在 launch 前发布句柄；启动中断或 join 中断都保留原 owner。
  退出确认需要 target 的 finally 完成记录以及正常 join，不能只看 `is_alive()`。
  从未尝试 launch 的线程只有在原子禁止未来 launch 后才能给出无获取证明。
- executor 的 planner 获取与关闭通过同一获取门串行。关闭先永久拒绝新提交，再有界等待获取完成。
  即使 factory 在关闭期间返回，返回值也先登记到原 owner，再拒绝启动。
  启动承诺与关闭 fence 在同一锁内排序；先前已承诺的获取仍由原 owner 排空，
  启动完成后再次检查 fence，不得在关闭后发布 ready。已登记 owner 与已准备好是两个不同事实，
  并发首帧准备和提交必须等待同一获取门，不能只凭 planner 非空跳过准备。

## 关闭与重试

- 同一次 executor 关闭的并发等待者持有同一个 attempt，得到相同的结果及根因。
  后来的调用可重试未完成的 Hold、线程退出、planner drain 与 GC 恢复；旧等待者的结果不变。
  重试不能重新开放提交，也不能恢复已经失效的 generation。
- Hold 在进入回调前登记 pending；包括 `KeyboardInterrupt` 在内的失败保留该义务。
  只有成功返回才清除 pending。已经成功的 Hold 不因单纯重复 close 再次执行。
- C2、阻抗 guard 与 hand state history poller 保留未确认退出的句柄。
  关闭失败之后继续清理同一 owner，不应将“没有活线程报告”当作退出证明。
- `dispatch_workers_stopped` 是 executor 输出的严格布尔肯定证明，覆盖已保留的 dispatch 和
  planner 后台资源以及待完成的停止义务。应用层无论 close 成功还是异常，都读取同一证明；
  缺失、错型、snapshot 失败均视为未知。
  证明通过 `shutdown_snapshot()` 窄端口读取，不能调用完整 snapshot 的 projector、诊断或状态锁。
- executor 退出未确认时保留 arm/hand backend；arm guard 关闭失败时同时保留 arm backend
  和 restore-speed 操作；hand poller 失败时保留 hand backend。相机、预览和诊断等独立清理继续执行。
  日志、snapshot 和 debug flush 失败会记录为独立错误，不得中断后面的清理。
  ResourceLifecycle 直接接收到的 `KeyboardInterrupt` / `SystemExit` 在完成独立清理后重抛原对象，
  不能被错误列表静默吞掉。内部 owner 的聚合关闭错误继续使用 RuntimeError，保留 cause 和重试义务。

## 并行开发约定

1. 后续 RTC native preparation worker、线程池和新后台执行端口必须接入同一 owner 及停止证明。
   不能只增加一个 `*_alive=False` 字段来绕过退出屏障。
2. 重放或吸收 RTC/shared 后续提交时，保留这些生命周期修复、现有 background-streaming
   状态读取约定、generation/epoch 门禁与主线近来的 B601 修复。
3. RTC native 调度、OpenPI 新预算与应用接线、驱动/SDK 组合属于后续阶段。
   本阶段不变更 SDK、设备配置、运行频率或真机动作入口，也不将离线验证作为真机证据。
4. 继续复用现有集成通道，按固定 commit、差异、独立审查、故障复现、聚焦回归、全量验收、
   主线前移逐阶段收束。不得覆盖其他工作区的未提交改动。

## 验证与交接

本阶段使用 `scripts/run_pytest.sh`，覆盖部分获取、中断 join、关闭重试、并发 waiter、
factory/close 竞态、严格停止证明、诊断失败和 backend 依赖；随后运行根全量 gate 与
`sim/sim_real_tests/tests` 离线 gate。已有用例身份、结果、重数及完整 skip 原因与上阶段
逐项比较，不能仅凭总数增长宣告覆盖保留。

现有测试的有意迁移仅包括：生命周期 fake 提供新增的严格停止证明及更新未知退出错误文本；
关闭失败用例继续保留同次并发异常 identity，后次调用改为验证实际重试；官方 C2 测试 helper
先登记 owner，再显式启动，并通过 teardown 排空所有 owner。不得删除原有行为断言。

原始报告、绑定的文件 hash、进程与超时记录和独立审查收束记录保存在机器上的
`AgenticRobotEvidence/defect-repair-20261015T074029Z/rtc-worker-ownership-20261021-r3/`。
具体通过数量与主线提交以该阶段最终验收记录为准，未完成的 gate 不得标为通过。

首轮全量验收的已完成分片暴露了本文未接入导航以及两个 writer 导入回归。后两项来自本批
hand poller 新增的顶层 utils 引用，在 SDK 同名模块或预加载 utils 环境中会遮蔽线程工具。
这些是真实集成缺陷，不能归因于取消。r2 补齐了导航；发现 writer 回归后也中止并排空，
r1/r2 均不作为完整通过证据。原始失败、取消收据和部分报告分别保留在相邻证据目录。

r3 将唯一线程 owner 实现放到纯标准库的 robot_motion.runtime.owned_thread，utils.owned_thread
只重导出同一类；本批四个生产消费者直接使用 canonical 路径。collection 原本已依赖同层
runtime，不调整 sys.path、不替换 sys.modules、不复制 owner。新增类 identity 检查，并保留
原有两个 writer shadowed-utils 测试。以新的固定候选和全新完整验收收束本阶段。


## 发布前二次复核：真实 stale 恢复

旧版恢复入口只清故障，但 reference worker 已由外层异常处理退出；旧用例只手工
注入 fault，未验证重新派发。修复仅在真实 source-stale 分支保留原线程等待，
完成原 generation 失效和 Hold 后发布同一 owner/fault 的 parked 事实；不重建线程、
不重绑 affinity、不自动恢复目标，也不导入未合并 native 调度栈。

显式恢复与 stale handler 使用同一 dispatch 序列，成功 Hold 后再次核对 fault identity、
parked owner、pending/fatal blocker 和永久 close fence。新的 secondary worker failure
始终阻断恢复，clear、成功 Hold 和流重置不能清除它。worker 内中断也记录为 fatal 后
退出；同步 start/clear/close 的中断传播规则保持。非 stale 故障仍退出原线程。
原有窄分类用例改为真实线程触发，新增恢复后再次派发、关闭与双故障竞争回归。

手动安全复位同样在观测校验前后核对同一 fault，并通过短 close gate 与关闭建立
先后顺序；pending、secondary fatal、启动失败或 Hold 未完成时拒绝清故障。故障与
generation 的变更在 close gate 内提交，观测校验、诊断和流重置不跨该 gate。
secondary fatal 的有界诊断在 pending-first 与 primary-first 顺序下均进入公开快照。
