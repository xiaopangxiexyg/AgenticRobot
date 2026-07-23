# Deploy 参数说明

Deploy 消费 policy 输出并组装真实 observation/action/backend；运行命令见
[real README](./README.md)。它与 replay 的动作来源、contract 和人工放行条件不同，不能互换。

## 1. Canonical 入口

先按 [real README 的第 3 节](./README.md) 检查 canonical CLI 参数（R0）。

`python -m teleop_runtime.deploy` 是 shell canonical wrapper。
`python -m teleop_runtime.deploy.run_policy` 仍是受支持的直接/程序化入口，OpenPI launcher
当前依赖其 `main`，但人工命令不应无理由绕过 wrapper。

部署 profile、单臂 dry-run、双臂/双手、dataset replay debug 和运行按键见
[`teleop_runtime/deploy/README.md`](../../teleop_runtime/deploy/README.md)。

## 2. OpenPI 两进程部署

OpenPI server 使用 OpenPI 环境，DexTeleop client 使用 `dexteleop310`。Checkpoint 目录必须
包含有效 `policy_artifact.json`。

无硬件 contract 检查和“只观测/推理、不发送动作”的可执行命令统一在
[real README 的第 3 节](./README.md) 维护。

只有在完整 preflight、qpos-init 和人工 ready gate 都确认后，才可增加 `--send-actions`
进入 M 级。不要同时加入 `--auto-start` 或 `--start-init_qpos`。

Server 启动、checkpoint 选择和现场按键以
[OpenPI 部署](./OPENPI.md)为准。

## 3. 验收工具（R1）

Absolute/relative 真机 A/B 完成后可离线比较：

```bash
python -m teleop_runtime.deploy.acceptance \
  "${ABSOLUTE_LOG_DIR:?set ABSOLUTE_LOG_DIR}" \
  "${RELATIVE_LOG_DIR:?set RELATIVE_LOG_DIR}" \
  --output "${ACCEPTANCE_JSON:?set ACCEPTANCE_JSON}"
```

该命令只读取已完成的两组日志并写一份离线报告，不连接机械臂。

参数和通过条件见
[`teleop_runtime/deploy/README.md`](../../teleop_runtime/deploy/README.md#v22-rawv23-relative-真机-ab-闭环验收)
与 [v2.3 验收](../data/ACCEPTANCE_V23.md)。
