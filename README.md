# DexTeleop

DexTeleop 汇集共享 robot motion、真实机械臂遥操与数采、policy deploy、replay、simulation
和 dataset contracts。运行命令按领域统一维护在 [`docs/`](./docs/README.md)，不要从历史
实验报告或日志中拼接参数。

## 从这里开始

| 目标 | 当前入口 |
|---|---|
| 安装环境、入口检查 | [`docs/setup/`](./docs/setup/README.md) |
| 真机遥操、正式数采、replay、deploy | [`docs/real/`](./docs/real/README.md) |
| MuJoCo、VIVE sim teleop、dataset replay | [`docs/sim/`](./docs/sim/README.md) |
| 设备枚举、USB、标定 | [`docs/hardware/`](./docs/hardware/README.md) |
| 数据后处理、分析、测试、验收 | [`docs/data/`](./docs/data/README.md) |
| 被替代命令与历史报告 | [`docs/legacy/`](./docs/legacy/README.md) |

环境依赖和锁文件的完整说明仍以 [`README_ENV.md`](./README_ENV.md) 为准。
所有长期命令都通过环境名激活 Conda，不依赖 `miniconda3` / `anaconda3` 的安装路径；统一
前缀和常用命令速查见 [`docs/README.md`](./docs/README.md#命令约定)。

第一次使用建议按顺序完成：安装环境 → 运行 R0 入口检查 → 阅读目标领域的简洁入口 →
做仿真或只读设备预检 → 最后才进入 H2/M 操作。R0/R1/H1/H2/M 的含义见
[`docs/README.md`](./docs/README.md#运行等级)。

## Canonical entrypoints

- GUI human teleop / formal collection：
  `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui`
- Headless/compatibility collection：`python -m teleop_runtime.collection`
- Policy deployment：`python -m teleop_runtime.deploy`

这里只声明入口身份；可直接运行的完整参数、前置条件和风险等级见对应 `docs/` 页面。

## 目录边界

- `control_composition/`：与本体、sim、遥操设备和模型实现解耦的多源控制组合、session/handoff 与差异观测；只产生 proposal candidate 和 lineage，不拥有 authority、安全或执行。
- `robot_motion/`：共享动作语义、Rig、Safety、Planner、position control 和 runtime contracts。
- `hardware/`：厂商 SDK 与设备 backend。
- `sim/`：仿真 backend、task 和 adapter；不得依赖 `teleop_runtime` 或真实硬件。
- `collection_runtime/`：与设备、控制和 GUI 解耦的实时采集生命周期与持久化机制。
- `teleop_runtime/`：真实遥操、GUI/headless orchestration、来源绑定的采集适配和 policy deploy frontend。
- `dataset_contracts/`、`dataset_interface/`：纯数据合约与数据适配。
- `deployments/`：模型 client、协议和 artifact 集成。
- `control_station/`：组合 human/replay/policy/dataset/sim/real target 的应用层。
- `legacy/`、`docs/legacy/`：已确认归档的代码或说明，不是生产入口。

更严格的所有权、验证和安全规则见 [`AGENTS.md`](./AGENTS.md)。

## 安全

真实机械臂、手、夹爪、servo、replay、homing 和 force-control 都可能改变设备状态。运行前先
确认设备 side/IP/串口、tool/TCP、工作空间和急停；只读检查、仿真和 dry-run 不能替代现场
放行。文档中的 H2/M 命令需要相应硬件范围的明确授权。
