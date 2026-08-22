# Quest Dev Core Protocol v2

核心通信使用页面同源 WebSocket `/ws`，消息为 UTF-8 JSON text frame，最大 1 MB。
v2 的首要变化是：任何生命周期或 feature 消息前都必须完成配对认证。

## 连接与配对

WebSocket upgrade 默认要求 `Origin` 的 scheme 为 HTTP(S)，且 `Origin.netloc` 与请求
`Host` 完全一致。只接受浏览器规范要求的 masked、单帧、无 RSV 数据；不支持 fragmented
或 extension frame。server 最多接纳 4 个连接，未认证连接默认只有 5 秒绝对总期限；
ping、非法 JSON 或其他 frame 不会延长该期限。

连接建立后 server 先发送：

```json
{
  "type": "server_hello",
  "protocol_version": 2,
  "server_timestamp": 1710000000.0,
  "mode": "quest_dev",
  "server_instance_id": "...",
  "authentication_required": true
}
```

浏览器的第一条 text 消息必须是：

```json
{
  "type": "client_auth",
  "protocol_version": 2,
  "client_instance_id": "browser-...",
  "pairing_token": "..."
}
```

server 使用恒定时间比较 token。类型、版本、实例 ID 或 token 任一不合法时返回失败的
`client_auth_result` 并关闭连接；成功时返回 `authenticated: true` 和连接局部
`client_id`。后续带 `client_instance_id` 的消息必须与认证身份一致，否则关闭连接。
未认证连接不会收到 feature 广播。

配对不可在配置中关闭。默认 pairing code 每次进程启动随机生成，为恰好 6 位 ASCII
数字；也可由 `QUEST_DEV_PAIRING_TOKEN` 注入固定 6 位数字。启动 URL 使用 `#pair=`
fragment，fragment 不进入 HTTP 请求；它仍是局域网内的短期 bearer secret，只适用于
明确受信的开发局域网，不应暴露到互联网。明文 HTTP/WS 被强制限制在 loopback。

## XR 生命周期

认证后，浏览器核心发送：

- `xr_session_state`：`started`/`ended`、页面实例、XR session、实际 reference-space
  type 与 generation；
- `reference_space_reset`：同一 session/type 下 `previous_generation` 到
  `new_generation=previous+1`；
- feature 自己注册的上行消息。

所有消息使用 `protocol_version: 2`。controller-pose runtime 会拒绝非零初始 generation、
重叠 session、错误 generation、乱序样本和身份不一致；end/reset/disconnect 都会让旧
latest pose 失效。核心协议不定义机器人动作或控制消息。

## Feature 消息

Feature 可通过 `server.publish_feature_message(feature_name, payload)` 向所有已认证
客户端广播自己的版本化消息。core version 与 `feature_protocol_version` 分开演进。

## HTTP 接口与状态最小化

- `/api/config`：只返回 browser-safe 配置；不含 token、token env 名、Redis 地址或
  host admission 配置；
- `/api/status`：要求 `Authorization: Bearer <pairing-token>`，只保存连接认证状态、
  消息类型、身份/generation/sequence/count 等摘要；不保存 token、controller matrix
  或 robot pose body。

这些开发接口不应暴露到不可信网络。
