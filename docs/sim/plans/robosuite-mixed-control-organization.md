# robosuite mixed-control 实验组织说明

状态：实验线结构整理（A3，2026-08-24）。本说明约束仿真 benchmark 的代码和数据
边界，不授予真机、GUI、策略部署或 canonical dataset 写入权限。

## 1. 两条线，两个 owner

| 责任 | 生产线 owner | robosuite 实验线 owner |
| --- | --- | --- |
| 后端和 reset/replay | `sim/`、`robot_motion/` | `sim/robosuite/` A0/A1 adapter/backend |
| policy/RTC 推理协议 | `robot_motion/deploy/`、`deployments/` | `experiments/sim_learning/streaming_intervention/` A2 mock/source seam |
| 控制 composition | `control_composition/`（backend-neutral、非权威）及下层执行 owner | A2 experiment-local composer/evidence |
| 采集生命周期和持久化 | `collection_runtime/`、`dataset_contracts/` | A3 内存对象和 `to_payload()`，暂不写 canonical dataset |
| GUI、设备、安全、IK、dispatch | `control_station/`、`teleop_runtime/`、`robot_motion/` | 明确禁止依赖 |

实验线可以消费公共的纯数据契约，但不能把实验对象反向导入生产包。以后若要
晋级，必须先在生产 owner 处提出新的兼容契约和调用边界，而不是把 A3 runner
直接当作部署入口。

## 2. A3 包内的责任分层

```text
sim/robosuite/                         A0/A1 backend + reset/replay
  -> streaming_intervention/           A2 policy/RTC request + composition/evidence
    -> robosuite_mixed_control_a3/
       adapters.py                     callable -> A2 source（不导入 deployment）
       contracts.py                    mode/source admission matrix（本次整理）
       capture.py                      passive observation trace + row materialization
       __init__.py                     public experiment API / compatibility exports
```

`contracts.py` 是 A3 模式和 policy variant 的唯一 owner。`capture.py` 只负责执行
A2 runner、记录 reset/next observation、绑定 row lineage 和构造内存 dataset；它
不再自行决定 mode/source 组合。`__init__.py` 从 contracts 导出矩阵，并继续从
`capture.py` 保留旧常量的兼容可见性，避免已有实验脚本因模块整理立即失效。

## 3. 三种采集模式

每一帧仍只有一个执行来源；这里的 “mixed” 指实验中的 component-level
composition，不是多个来源同时拥有同一组件，也不是安全层的最终裁决。

| `collection_mode` | policy variant | frame mode | 人类来源 | policy 来源 |
| --- | --- | --- | --- | --- |
| `ordinary_teleop` | `none` | `base_only` | 全量 human action | 无 |
| `takeover_teleop` | `standard` | `absolute_overlay_owner` | 全组件绝对接管 | nominal standard policy |
| `streaming_mixed_control` | `standard` | `relative_overlay` | human residual（至少一个组件） | nominal standard policy |
| `streaming_mixed_control` | `rtc` | `relative_overlay` | 不允许 human 同帧介入 | RTC intervention + nominal base |

`resolve_collection_profile()` 先解析这一矩阵，`validate_collection_frames()` 在
创建 backend 之前拒绝空帧、错误 mode、错误 policy source、RTC+human 同帧和缺失
RTC nominal base。矩阵是 admission contract；它不负责限幅、IK、dispatch 或物理
安全。

## 4. 当前证据边界

- A0/A1 提供可重放的仿真后端和 reset manifest；A2 提供标准/RTC mock inference
  以及 composition/evidence；A3 提供三种模式的 content-addressed row/dataset
  内存对象。
- 当前 A3 没有 canonical writer、episode owner、命令行入口、训练器或 benchmark
  metric evaluator。`to_payload()` 不是持久化承诺。
- callable policy adapter 只约束 A2 所需的 7-D action/chunk 形状，不表示已经
  接通 OpenPI、真实策略服务、归一化、帧定义、延迟或异步 worker。
- runner 目前按帧同步调用 policy source；它是可复现的离线采集 seam，不是已经
  证明实时调度性质的 streaming runtime。
- 未启动真实设备、GUI、网络策略服务或任何机械运动；仿真 smoke/contract 结果
  也不等价于真机 readiness。

## 5. 后续晋级顺序

1. 在 A3 实验线增加独立的 episode artifact/manifest writer，并固定 reset manifest、
   runtime fingerprint、source revision 和 observation/action hash 的持久化语义。
2. 将 canonical `PolicyActionEmission` 接到一个显式的 robosuite experiment
   adapter，保留标准 nominal 与 RTC intervention 的来源标签；仍不把 adapter
   变成生产 authority。
3. 增加离线 evaluator，分别报告 ordinary、takeover、streaming-standard 和
   streaming-RTC 的 task success、human-off 区间和 intervention cost；不要用
   replay self-consistency 代替因果/物理验证。
4. 只有实验 artifact、指标和数据质量契约稳定后，才评估是否需要一个
   `control_station` sim profile。该 profile 必须重新声明 writer、safety、IK、
   dispatch 和 episode lifecycle owner，不能直接复用 A3 的内存 runner。

## 6. 组织变更的验证

本次结构整理的最低验证是：A3 既有测试、独立的
`tests/test_robosuite_mixed_control_a3_contracts.py`、导入/编译检查和文档导航检查。
这些检查只证明 experiment contract 和现有 mock backend 的边界，不提升部署证据
等级。
