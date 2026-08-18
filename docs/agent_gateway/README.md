# External Agent Gateway

`agent_gateway` 是 DexTeleop 提供给上层 Agent/Skill 组织框架的稳定下层边界。它公开
版本化的 Skill、Route、observation、Artifact、operation 和 reconciliation 合同；具体
profile 负责装配 MuJoCo 或真机下层 owner。它不启动上层 Planner，也不把安全、IK、HOLD
证明或 actuator dispatch 交给远端调用方。

本页是 Agent Gateway 的正式运行入口。协议与 package 所有权细节见
[`agent_gateway/README.md`](../../agent_gateway/README.md)；局域网开发者的 SSH 只读接入见
[`SSH_READ_ONLY.md`](./SSH_READ_ONLY.md)。

## 边界和运行等级

| 用途 | 接入方式 | 能力上限 | 等级 |
|---|---|---|---|
| 本机上层框架 | 私有 Unix socket | 由已审查 profile、client binding 和每次 Gateway admission 决定 | 取决于所调用 Skill；MuJoCo 为 R1，真机运动为 M |
| 远端开发者 | OpenSSH forced command | 只读 Skill、限定状态/发现、本人 operation 查询/取消和本人 Artifact 下载 | H1；若下层是 MuJoCo 则为 R1 |
| CLI/合同检查 | `--help`、离线测试 | 不启动 backend | R0 |

远端 SSH 接入不是第二套硬件协议，也不产生新的设备 lease。驱动类 Skill、通用提交、
lifecycle、HOLD、Artifact 上传和 materialization acknowledgement 对远端身份始终不可见；
这不会降低本机可信 Skill 调用方原有的权限。

## 环境前缀

先按 [`docs/README.md`](../README.md#命令约定) 激活 DexTeleop 环境，然后从仓库根目录运行：

```bash
export DEXTELEOP_ROOT="$PWD"
```

## 本机 Unix endpoint

为每次 Gateway session 分配一个独立、长度受控的 endpoint：

```bash
export DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint
)"
test -n "${DEXTELEOP_GATEWAY_SOCKET}"
```

分配命令只创建 mode `0700` 的 session 目录并返回尚不存在的 `g.sock` 路径，不会启动
仿真、设备或 Gateway。把该路径原样交给下层 launcher 和同机可信上层；不要把 history、
Artifact、Experience 或日志根目录放进 socket 路径，也不要自行迁移到 `/tmp`。

### MuJoCo profile（R1）

```bash
python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config "${DEXTELEOP_ROOT}/sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json" \
  --client-id rpent.dexteleop \
  --client-uid "$(id -u)" \
  --history-root "${DEXTELEOP_ROOT}/temp/runtime/agent_gateway/history"
```

这条命令只启动 DexTeleop 自有的 MuJoCo application、下层 motion owner 和 Gateway，
不会启动 RPent。需要 RGB/depth Skill 时额外提供一个私有、非 world-writable 的
`--artifact-root`；不提供时 Artifact-bearing perception Route 保持关闭。完整 MuJoCo
上下文和 safety-reference 语义见 [`docs/sim/README.md`](../sim/README.md#4-agent-tcp-pose-gatewayr1mujoco-only)。

### 真机 profile（M）

```bash
python -m agent_gateway real-rm65 --help
```

`real-rm65` 会装配独占的双 RM65 下层 owner，并要求 profile、设备、safety envelope 和
一次性运动确认。即使上层只打算读取感知，当前组合 launcher 也会启动相机和机械臂 owner；
不要把它当作 camera-only diagnostic。正式参数、启动门禁和关闭顺序以
[`agent_gateway/README.md`](../../agent_gateway/README.md#entrypoints) 为准，运行前仍需明确
真机授权和现场安全检查。

## 状态、占用与并发

Gateway 可返回两个小型只读投影：

- `get_hardware_resource_views`：profile 绑定的稳定资源、`readable`/`drivable` 属性和
  语义 Skill 引用；
- `get_hardware_status_views`：`free`/`occupied`、独立 health，以及可选的 active
  operation 投影。

`occupied` 表示已有独占下层控制/资源 claim，或存在活跃独占 operation；它不等价于
“机械臂正在运动”。READY、idle 和 lower-confirmed HOLD 都不会自动释放 claim。读取状态
失败时按 unavailable + occupied 关闭失败路径。远端调用方不会排队、抢占或取得另一份
lease；资源被其他用户或 Skill 占用时，返回正常的下层 admission/status 结果。

## 关闭与故障处理

- 正常关闭 launcher，让 Gateway 先 fence 新请求，再由下层 owner 完成自己的 HOLD 和
  resource close；上层不得根据连接关闭自行声明 HOLD。
- transport 超时或连接中断不证明请求未被接收。可能已接收的提交进入既有
  reconciliation 流程，不能当作普通 rejection 重新提交。
- event replay 是提示而非真值；出现 gap 后通过 authoritative operation/discovery query
  重建状态。
- endpoint 只属于一个 session。不要复用旧 socket、猜测 socket 路径或删除仍在使用的
  session 目录。

## 离线验收（R0）

```bash
python -m agent_gateway --help
python -m agent_gateway allocate-unix-endpoint --help
python -m agent_gateway mujoco --help
python -m agent_gateway real-rm65 --help
python -m agent_gateway ssh-readonly-proxy --help

PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_agent_gateway*.py
```

这些检查只证明 CLI、合同、transport 和装配的离线行为；不能替代 MuJoCo、SSH 主机或
真机运行证据。
