# MuJoCo checkpoint 前置组件：暂停候选

2026-10-24：按用户要求暂停 Gateway checkpoint 接入。本文件记录可继续开发的
局部成果，不是已开放的 Gateway API 或完整恢复能力。候选基于 `d4285acc1d3c`；
没有修改现有 Gateway、控制 runtime、RPent 或真实设备入口。

## 已保存的实现

`sim/mujoco/native_checkpoint.py` 是仅支持已验证 MuJoCo 3.10.0 的原生物理组件。
调用方必须已持有物理状态的排他边界；组件自身不暂停线程、不取得 Gateway admission
或控制 authority。恢复构造独立 `MjData`/model 候选，不修改运行中的 owner，
也不调用 reset、forward 或 step。

- 复用官方 native payload，保存编译模型、积分器和求解缓存，不另写全数组序列化器。
- 补充官方 3.10 codec 漏存的 9 个 scalar 和 3 个固定 vector：约束数量、awake
  有效长度、lazy-stage 标志及 solver 诊断。原生载荷与辅助状态共同参与校验。
- 检查版本、ABI、显式载荷大小限制、有限积分状态、模型身份与原生即时字节往返。
- 拒绝 process callback、plugin、活跃 stack/threadpool；这些检查不能代替调用方互斥。
- 记录源 compilation signatures。官方反序列化后的 signatures 为零，
  **不支持恢复 `MjSpec.bind/edit/recompile` 关系**。
- 仅处理可信来源的 native bytes；SHA 不提供来源认证，原生解析器也不是安全沙箱。
  `max_native_bytes` 限制序列化载荷，不等于反序列化总内存上限。

字段依据为 [MuJoCo 3.10 官方 codec](https://github.com/google-deepmind/mujoco/blob/3.10.0/python/mujoco/structs_wrappers.cc#L816-L975)
和 [原生字段定义](https://github.com/google-deepmind/mujoco/blob/3.10.0/include/mujoco/mjxmacro.h#L998-L1040)。

## 当前验证及其含义

在 Python 3.10.20 / MuJoCo 3.10.0 上执行：

```bash
bash scripts/run_pytest.sh tests/test_mujoco_native_checkpoint.py -q
```

收尾结果：**50 passed**；Ruff check/format 通过。这是小型真实 MuJoCo 物理模型的
组件测试，不是完整机器人 profile、渲染、远端、模型或任务成功验收。

立即恢复要求未改写的 native bytes 完全一致；后续相同动作续跑比较所有已分配、
公开的语义字段及模型 hash，包含辅助状态和 contact/solver/warning/timer 成员。
墙钟 `timer.duration` 不作确定性判据；稠密求解下未使用的四个 sparse-J 索引不比较；
dual/island arena 按官方 codec 的实际启用条件选择，不能读取未分配指针。
按字段比较 contact 避免把 C 结构尾部 padding 当成物理差异，不改保存物或 live state。
原先失败及有限重复测试的证据保留，没有覆盖成最终通过结果。

## 尚未完成，不能据此放开上层

1. Gateway admission、物理线程和控制 owner 的联合静止边界。
2. 双臂 planner/actuator/gripper、命令序号、时钟、authority、执行 gate 及保留引用的完整恢复。
3. 新实例中 FK、collision、payload、observer、scratch 等消费者的统一绑定。
4. profile 身份、持久化/原子提交、历史产物与恢复分支的来源关系。
5. sleep 混合状态、完整 no-model profile、渲染和远端组合验收。
6. RPent SDK/runtime/记录前缀与上述下层合同的联合恢复。

本模块尚无生产消费者。Gateway native checkpoint 和未完成后台执行的既有拒绝不变。
后续先检查最新 development/correction 与并行改动，再继续 owner 合同；不得将当前
50 项通过视为完整 profile 已实现，也不得用 fresh start/reset 冒充精确冷恢复。
