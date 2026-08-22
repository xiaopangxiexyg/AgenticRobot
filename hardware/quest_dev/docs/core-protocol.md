# Quest Dev Core Protocol v1

核心通信使用同源 WebSocket `/ws`，消息为 UTF-8 JSON text frame，最大 1 MB。

## `server_hello`

连接建立后主机发送：

```json
{
  "type": "server_hello",
  "protocol_version": 1,
  "server_timestamp": 1710000000.0,
  "mode": "quest_dev"
}
```

## Quest 生命周期消息

浏览器核心目前发送：

- `client_hello`
- `xr_session_state`
- `reference_space_reset`

核心 server 会记录最后一条客户端消息，并将已注册的消息类型分发给 feature handler。核心协议不定义机器人或控制消息。

## Feature 消息

Feature 可以通过 `server.publish_feature_message(feature_name, payload)` 广播自己的版本化消息。最后一条 feature 消息会出现在 `/api/status` 的 `last_feature_messages` 中。

## HTTP 接口

- `/api/status`：核心连接和各 feature 最后一条状态。
- `/api/config`：只返回浏览器需要的安全配置，不暴露 Redis 地址等 host-only 配置。
