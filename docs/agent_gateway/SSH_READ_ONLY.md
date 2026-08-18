# Agent Gateway SSH 只读接入

本页说明如何把**已经运行的** DexTeleop Agent Gateway 以最小 forced-command SSH 通道
开放给局域网开发者。该通道不启动硬件、不创建第二个 Gateway、不允许远端运动，并且不
改变可信本机 Skill 调用方的权限。

## 1. 信任模型

一个远端开发者对应：

1. 一个独立 SSH key；
2. 一个独立、无交互 shell 用途的本机 Unix UID；
3. 一个独立 Gateway `client_id`；
4. Gateway 启动时的一条 UID 到 `client_id` 显式绑定。

不能把多个开发者复用为同一个本机账号或同一个 Gateway client identity。SSH 只负责认证
并启动 stdio proxy；Gateway 仍会以 Unix peer credential、client binding、Route effect
和每次 admission 独立校验请求。proxy 和 Gateway 是两层同时生效的上限，而不是二选一。

## 2. 主机管理员准备（H2）

下面的名称和 UID 只是占位符，必须由主机管理员按本机账号策略替换。此步骤会修改账号、
group 或 `/run` 权限，不应由普通远端开发者执行。

```bash
install -d -o gateway-user -g dexteleop-agent-ro -m 2750 \
  /run/dexteleop-agent-gateway
```

Gateway process 和所有只读 no-shell 账号必须共享专用 group；运行时不会自行 `chgrp`、
修改账号或扩张 group membership。为 key 配置 forced command 时，固定到当前主机上的
DexTeleop 环境和 socket 路径，例如 `authorized_keys` 中的一行：

```text
restrict,command="/absolute/conda/envs/dexteleop310/bin/python -m agent_gateway ssh-readonly-proxy --socket-path /run/dexteleop-agent-gateway/SESSION/g.sock" ssh-ed25519 AAAA... developer-name
```

`restrict` 应禁止 PTY、port forwarding、agent forwarding、X11 forwarding 和用户提供的
任意命令。更新 Gateway session 后，管理员需要把 forced command 中的 endpoint 更新为
本次实际 socket；远端客户端不应知道或提交该主机内路径。

## 3. 启动共享 endpoint

在专用 setgid root 下分配 group-access session：

```bash
export DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint \
    --root /run/dexteleop-agent-gateway \
    --group-access
)"
```

session 目录为 mode `2750`，Gateway socket 在绑定后为 `0660`。group 用户可以 traverse
目录和连接 socket，但不能删除 socket。没有任何 `--ssh-read-only-client` 时仍保持私有
endpoint mode `0600`。

启动既有 MuJoCo 或 `real-rm65` profile 时，为每个开发者重复声明绑定：

```bash
python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config /absolute/task_config.json \
  --client-id rpent.dexteleop \
  --client-uid "$(id -u)" \
  --ssh-read-only-client rpent.remote.alice 1101 \
  --ssh-read-only-client rpent.remote.bob 1102
```

`real-rm65` 接受同一个可重复参数。primary `--client-id` 仍是唯一可能取得真机 reviewed
motion grant 的身份；附加身份只读。远端访问不会削弱或替换原 Skill 权限模型。

## 4. 远端开发者连接（H1/R1）

客户端应使用支持 `dexteleop.agent_gateway.unix_json.v2` framed stdio 的上层 adapter，并让
OpenSSH 执行服务端 forced command；客户端不得传远端命令、请求 TTY 或开启 forwarding。
以 RPent 为例，先做连接和发现检查：

```bash
python -m rpent_dexteleop inspect-gateway \
  --gateway-transport ssh-readonly \
  --ssh-target dex-readonly@robot-host \
  --ssh-identity-file /absolute/path/to/developer-key \
  --deployment-profiles /absolute/path/to/reviewed-template.json
```

远端开发者必须保留正常 OpenSSH host-key verification。`--ssh-config` 和 `--ssh-port` 可选；
identity/config path 必须是本地绝对路径。RPent 的远端 adapter 不发送
`ssh-readonly-proxy` 或 socket path，而是依赖主机 key/account 的 forced command。

## 5. 允许和拒绝的请求

允许面严格限定为：

- session health、五份互相闭合的 discovery 文档；
- bounded observation、hardware resource/status views；
- `submit_read_only_skill_request`，且最终 Route 必须同时满足 `read_only`、
  `t0_read_only`、non-motion、non-publication、`cancel_only`；
- 本身份 operation 的 query/cancel；
- 仅由本连接 own-operation 暴露的 Artifact，通过 `chunked_rpc` 下载。

明确拒绝：generic Skill submission、任何 motion Route、lifecycle、HOLD、event replay、
Artifact upload、host-local Artifact path 和 materialization acknowledgement。payload 中的
`client_id` 不是认证来源，任意缓存 ID 或其他人的 operation ID 也不能越权读取。

## 6. 运行检查与故障定位

主机侧 R0 检查：

```bash
python -m agent_gateway ssh-readonly-proxy --help
python -m agent_gateway mujoco --help
python -m agent_gateway real-rm65 --help
```

远端连接成功后，先检查 discovery 和 hardware status，再提交只读 Skill。常见失败按以下
边界定位：

| 现象 | 应检查的 owner |
|---|---|
| SSH authentication/host key 失败 | OpenSSH 账号、key 和客户端 config |
| proxy 立即拒绝 method | forced-command allowlist；远端请求不属于只读面 |
| Gateway 拒绝 identity | 启动参数中的 UID/client binding 与实际 SSH Unix UID |
| Route 被拒绝 | Gateway read-only effect/admission 校验或 reviewed deployment exposure |
| resource `occupied` | 既有下层 control/resource claim 或 active exclusive operation；不排队、不抢占 |
| status unavailable + occupied | 下层 health/status read 失败；按 fail-closed 处理 |
| Artifact 下载被拒绝 | 该 Artifact 未由本连接查询到的 own-operation 暴露，或机制不是 `chunked_rpc` |

不要通过开放 shell、转发 Unix socket、共享 primary UID/client ID 或放宽 Gateway Route 来
绕过故障。真机 H1 只读运行证据需要单独授权和记录；离线测试只证明协议策略。
