# Quest Dev Architecture

```text
Quest Browser
  web/app.js                    pairing + WebXR/WSS lifecycle
  web/feature_registry.js       browser feature hooks
  web/features/visualization/   line cube, axes, future render leaves
  web/features/controller_pose/ read-only grip-pose publisher
             ⇅ HTTPS / authenticated WSS
server.py                       static host, auth hub, safe config/status
  features/controller_pose/     strict latest-only telemetry
  features/visualization/       optional read-only robot feedback + fit tools
```

## Core ownership

Core owns：

- 开发证书、HTTPS 静态文件与 WebSocket framing/size/connection limit；
- 同源检查、配对认证、连接身份和 authenticated-only broadcast；
- WebXR session/reference-space lifecycle 与 feature hook 调度；
- browser-safe config 投影和 pose/token-redacted authenticated status。

Core 不拥有 Redis、机器人 schema、标定 acceptance、轨迹、控制、safety 或渲染语义。

## 当前 feature ownership

- `controller_pose`：只解析当前认证客户端、XR session 和 reference generation 的
  `gripSpace`；每客户端仅保留 latest sample，不持久化或重建轨迹。
- `visualization`：拥有 Quest 端基础几何渲染、只读 `robot_tcp` 下行和 feature-local
  离线标定拟合；未验收标定不会变成可渲染 robot transform。

`hardware/quest_dev` 不得成为 future/selection 的权威拥有者。后续接入
`control_composition` 时，应由其提供有因果身份、frame 与时间语义的只读选中 future
快照；Quest adapter 只能投影和显示，不能推进 cursor、重新选择 future 或形成执行指令。

## Browser hooks

- `configure`
- `onXRReady`
- `onXRFrame`：每个 XR frame 调用一次，供有界遥测与非逐眼状态更新使用；
- `handleServerMessage`
- `renderView`：每个 eye/view 调用；不得在此重复发布 once-per-frame 遥测；
- `onReferenceSpaceReset`
- `onSessionEnd`

## Adding a feature

Host side：

1. 新增 `features/<name>/runtime.py`；
2. 导出 `create_runtime(server, config, stop_event)`；
3. runtime 实现 `start()`/`stop()`，通过 server 注册消息、disconnect 或 status handler；
4. 在 `config.yaml` 的 `features.<name>` 下显式启用。

Browser side：

1. 新增 `web/features/<name>/feature.js`；
2. 调用 `QuestDevFeatures.register(name, hooks)`；
3. 在 `web/index.html` 加载脚本；
4. 只有 browser-safe 选项可放入 `features.<name>.client`。

新 feature 必须保持单一资源拥有者：被动 observer/latest cache 可以存在，但不能复制
source advancement、authority、canonical safety、dispatch 或 episode commitment。
