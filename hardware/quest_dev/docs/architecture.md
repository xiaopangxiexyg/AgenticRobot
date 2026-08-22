# Quest Dev Architecture

```text
Quest Browser
  web/app.js                 core WebXR/WSS lifecycle
  web/feature_registry.js    browser feature hooks
  web/features/<name>/       feature UI/rendering
           ⇅ HTTPS / WSS
server.py                    generic static host, WS hub, config, lifecycle
  features/<name>/runtime.py optional host-side feature runtime
```

## Core ownership

Core owns:

- HTTPS certificate and static file service;
- WebSocket framing, connection registry and size limit;
- WebXR session and reference-space lifecycle;
- safe browser config projection;
- feature discovery, start/stop and message dispatch.

Core does not own Redis, robot schemas, calibration math or rendering semantics.

## Adding a feature

Host side:

1. Add `features/<name>/runtime.py`.
2. Export `create_runtime(server, config, stop_event)`.
3. Runtime implements `start()` and `stop()`.
4. Enable it under `features.<name>` in `config.yaml`.

Browser side:

1. Add `web/features/<name>/feature.js`.
2. Call `QuestDevFeatures.register(name, hooks)`.
3. Add its script to `web/index.html`.
4. Put browser-safe options under `features.<name>.client`.

Available browser hooks:

- `configure`
- `onXRReady`
- `handleServerMessage`
- `renderView`
- `onReferenceSpaceReset`
- `onSessionEnd`
