# Data Collection Blueprint P2

## 0. Status

本文档只保留 `Phase 2` 的**工程任务清单**，不展开技术细节。

`Phase 2` 的目标是把 `Phase 1` 的右臂采集能力扩展到：

- 双臂
- 双手
- 双侧 tactile
- 单相机起步，保留多相机扩展位
- 可选双臂 collision-aware retargeting

建议 `Phase 2` 再拆成：

- `P2A`：双臂独立采集
- `P2B`：双臂 collision-aware 采集

---

## 1. 前置条件

进入 `Phase 2` 前，`Phase 1` 至少需要完成：

- 单臂采集 runner 可用
- 单臂 writer 可用
- 单臂 replay 可用
- `schema_v21.py` 已稳定
- RGB / RM65 state / Inspire tactile source 已稳定

---

## 2. Phase 2 任务清单

| ID | 任务 | 目标产物 | 完成判据 |
| --- | --- | --- | --- |
| P2-T0 | 冻结双臂 schema 约定 | 双臂 `observation.state` / `action` / tactile 列定义 | 双臂 schema 不再反复调整 |
| P2-T1 | 把 P1 模块改成 side-agnostic | 可复用的 left/right 通用 source / assembler / writer 入口 | 右臂实现不再硬编码 `right` |
| P2-T2 | 左臂 RM65 状态采集接入 | `left` RM65 state source | 左右臂 state 可并行稳定采集 |
| P2-T3 | 左手 Inspire 状态与 tactile 接入 | `left` hand state / tactile source | 左右手 telemetry 可并行稳定采集 |
| P2-T4 | 双臂 builder | `build_bimanual_collection_session(...)` | 能统一创建双臂采集运行时 |
| P2-T5 | 双臂 sample assembler | 双臂 row 装配逻辑 | 左右 state/action/tactile 被正确拼接 |
| P2-T6 | 双臂独立采集 runner | `run_bimanual_collect.py` 独立模式 | 不开 collision-aware 也能稳定录双臂 episode |
| P2-T7 | 双臂独立 replay | 双臂独立 replay 工具 | 左右臂独立 action 可近似复现 |
| P2-T8 | collision-aware metadata 补强 | `DualArmCollisionAwareRetargeter` metadata 扩展 | 双臂联合 IK 的中间 pose 与输出 joint target 都可追踪 |
| P2-T9 | collision-aware 双臂采集 runner | `run_bimanual_collect.py` 联合模式 | 开启 collision-aware 后可稳定录制双臂 episode |
| P2-T10 | collision-aware replay 策略 | 双臂联合 replay 逻辑 | 优先 joint replay，必要时 fallback 到 cartesian |
| P2-T11 | 多相机接口预留 | camera config 可扩展结构 | 不改 writer 核心即可增加 camera key |
| P2-T12 | Phase 2 验收 | 结构、时序、性能、回放、模式隔离检查 | 双臂独立模式与联合模式均通过验收 |

---

## 3. 分阶段顺序

推荐顺序：

1. `P2-T0`
2. `P2-T1`
3. `P2-T2 + P2-T3`
4. `P2-T4 + P2-T5`
5. `P2-T6 + P2-T7`
6. `P2-T8`
7. `P2-T9 + P2-T10`
8. `P2-T11`
9. `P2-T12`

不要一开始就直接进入 collision-aware 采集。

---

## 4. P2A：双臂独立采集任务

这一阶段不启用联合 collision-aware。

### 任务

- 完成左右臂 / 左右手的 source 通用化
- 扩展 `observation.state`
- 扩展 `action`
- 扩展双侧 tactile 列
- 扩展双臂 episode metadata
- 做双臂独立采集 runner
- 做双臂独立 replay

### 目标

- 先把双臂数据结构跑通
- 保持每侧 action 仍是各自独立 cartesian target
- 不引入双臂联合 IK 的额外语义复杂度

---

## 5. P2B：collision-aware 双臂采集任务

这一阶段启用 `DualArmCollisionAwareRetargeter`。

### 任务

- 补齐 `DualArmCollisionAwareRetargeter` 中间 metadata
- 显式记录左右输入 cartesian target
- 显式记录联合 IK 输出 joint target
- 显式记录 collision status / fallback / jump gate / hold-last-safe
- 定义双臂联合模式下的主 `action`
- 定义 replay 优先级
- 将联合模式写入 dataset metadata

### 目标

- 让双臂联合采集的 action 既有“任务语义”，又有 replay-safe 辅助信息
- 避免把不同 teleop 模式混成语义不清的数据集

---

## 6. 必须保留的模式标记

进入 `Phase 2` 后，dataset metadata 必须显式保留：

- `teleop_mode`
- `control_mode`
- `tcp_mode`
- `collision_aware_enabled`
- `collision_avoidance_mode`
- `retarget_pipeline`

否则后续双臂独立模式与双臂联合模式混训时，action 语义会污染。

---

## 7. 验收任务

`Phase 2` 验收必须覆盖两类模式：

### 双臂独立模式

- 数据结构正确
- 左右 state/action 对应正确
- replay 正常

### 双臂联合模式

- metadata 完整
- collision-aware 输出可追踪
- replay 优先使用 joint target
- 不会把联合模式误写成普通独立模式的数据语义

---

## 8. 当前未决问题

`Phase 2` 仍需后续确认：

- 双臂联合模式下 top-level `action` 是否继续以 cartesian target 为主
- 是否允许同一个数据集混合保存独立模式与联合模式
- 多相机扩展时是否每个 camera 都要求独立时间戳与 age
- 双臂 tactile stale 的策略是分侧标记还是整条样本降级
