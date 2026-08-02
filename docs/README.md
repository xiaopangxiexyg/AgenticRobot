# DexTeleop 文档

文档按运行领域组织。每个领域目录的 `README.md` 是该领域**当前主命令的唯一汇总入口**；
同目录其他文件解释参数、数据语义、安全边界、验收和特殊变体，不再复制同一条主命令。

| 领域 | 简洁运行入口 | 详细文档 |
|---|---|---|
| 环境安装与基础检查 | [setup](./setup/README.md) | [环境与复现权威说明](../README_ENV.md) |
| 真机遥操、数采、Replay、Deploy | [real](./real/README.md) | [参数数组](./real/teleop_args.sh)、[遥操数采与长程 Progress Profile](./real/TELEOP_COLLECTION.md)、[Replay](./real/REPLAY.md)、[Deploy](./real/DEPLOY.md)、[UMI](./real/UMI.md)、[OpenPI](./real/OPENPI.md) |
| MuJoCo 与仿真 | [sim](./sim/README.md) | [MuJoCo/仿真说明](./sim/MUJOCO.md) |
| Sim human-in-the-loop | [HIL](./HIL/README.md) | [CS-6b1f 使用说明、capture 与策略兼容矩阵](./HIL/README.md) |
| 设备、USB、标定 | [hardware](./hardware/README.md) | [设备与标定](./hardware/DEVICES_AND_CALIBRATION.md)、[DMTAC/Q6](./hardware/DMTAC.md) |
| 数据后处理、分析、测试、验收 | [data](./data/README.md) | [长程任务细粒度阶段标注](./real/LONG_HORIZON_STAGE_ANNOTATION.md)、[后处理与测试](./data/POSTPROCESS_AND_TESTING.md)、[v2.3 验收](./data/ACCEPTANCE_V23.md) |
| 被替代命令和历史报告 | [legacy](./legacy/README.md) | 仅用于追溯和复现实验 |

## 使用方式

1. 先进入对应领域的 `README.md` 复制当前命令。
2. 需要理解参数或排查问题时，打开同目录专题文件。
3. Package 内 README 仍是模块接口与局部示例的技术权威，但不是另一套正式运行配方；
   涉及多个模块组合时以这里的领域入口为准。
4. 历史命令只进入 `docs/legacy/`，不与当前入口并列。

## 命令约定

- 除非页面另有说明，命令都从仓库根目录、在 Bash 中执行，并假定已经激活
  `dexteleop310`。
- `/path/to/...` 表示必须替换的本机路径；`${NAME:?set NAME}` 表示必须先设置的环境变量，
  未设置时 Bash 会在连接设备前停止。
- `latest` 是当前站点保存的 assignment，不是跨机器通用配置。换机器、换 USB 口或换设备后
  必须重新做对应的只读检查/assignment。
- `--help`、dry-run 和离线命令也可能加载较重的软件依赖，但不应连接真实执行器；若实际行为
  与风险等级不符，立即停止并修正文档。

## 现有 README 的归属

仓库里不会为了“集中命令”而清空模块自己的 README。它们按下表归属，避免把局部示例误当作
正式整机命令：

| 原始说明位置 | 负责内容 | 当前跨模块入口 |
|---|---|---|
| `teleop_runtime/**/README.md` | 单个 teleop/collection/replay/deploy 模块接口 | [real](./real/README.md) |
| `sim/**/README.md` | 仿真 backend、任务和调试参数 | [sim](./sim/README.md) |
| `hardware/**/README.md` | 单设备枚举、配置和厂商工具 | [hardware](./hardware/README.md) |
| `utils/dataset_postprocess/**`、`utils/statistic/**` | 单个数据工具参数 | [data](./data/README.md) |
| `README_ENV.md` | 环境依赖与安装细节 | [setup](./setup/README.md) |

判断标准很简单：能够单独验证某个模块的命令可以留在模块 README；用于正式遥操、数采、
回放或部署的组合命令只在对应领域 `README.md` 维护。

## 运行等级

| 等级 | 含义 |
|---|---|
| R0 | 纯解析、import、dry-run 或静态检查 |
| R1 | 本地计算、离线数据、仿真或 GUI，不连接真实执行器 |
| H1 | 只读连接真实设备，可能占用端口但不改变配置/目标 |
| H2 | 修改设备 assignment、标定、权限或持久配置 |
| M | 可能向机械臂、手或夹爪发送运动/力控制命令 |

同一命令同时满足多个等级时按副作用最高者标注。H2/M 命令不能因为出现在文档中就直接运行，
仍需现场安全检查和明确授权。

## 文档维护边界

- 新增当前运行方式：更新领域 `README.md`。
- 参数、验收或安全语义变化：更新同领域专题文件。
- 实验命令通过验收后再提升；旧说明迁入 `legacy/`，保留替代关系。
- 真机遥操参数数组只在 [`docs/real/teleop_args.sh`](./real/teleop_args.sh)维护。
