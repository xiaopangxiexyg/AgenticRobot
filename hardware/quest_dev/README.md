# Quest Dev

`quest_dev` 是从现有 `quest_pose/` 中抽取的通用 Quest Browser/WebXR 开发基座。它不以机械臂末端显示为模块身份；visualization 是第一个可插拔 feature，后续可以并列增加交互、采集、诊断或其他 Quest 功能。

## 目录结构

```text
hardware/quest_dev/
├── server.py                       # 通用 HTTPS/WSS host 与 feature 生命周期
├── protocol.py                     # core protocol
├── config.yaml
├── features/
│   └── visualization/
│       ├── runtime.py              # host-side feature runtime
│       ├── protocol.py             # robot_tcp feature protocol
│       ├── robot_feedback.py       # 可选 Redis GET adapter
│       ├── transforms.py           # pose/SE(3)/WebGL matrix
│       └── alignment.py            # Kabsch 空间标定
├── web/
│   ├── app.js                      # 通用 WebXR/WSS lifecycle
│   ├── feature_registry.js         # browser feature hooks
│   ├── xr_math.js
│   └── features/visualization/     # visualization renderer 与 browser hooks
├── docs/
├── examples/
├── tools/
└── tests/
```

架构和新增 feature 的方法见 [docs/architecture.md](docs/architecture.md)。

## Core 提供什么

- Quest Browser HTTPS secure context；
- WebSocket/WSS 连接、重连和 1 MB 消息上限；
- `immersive-ar` 优先、`immersive-vr` 回退；
- WebXR reference-space 生命周期和 `reset` 通知；
- 逐眼 framebuffer、view/projection context；
- host/browser feature registry；
- `/api/status` 与安全的 `/api/config`；
- 脚本入口和 Python package 入口。

Core 不 import Redis、AR5、Pinocchio、IK 或真机 SDK，也不定义机器人控制协议。

## 当前 visualization feature

visualization feature 包含：

- RGB 坐标轴 WebGL renderer；
- 可选 AR5 `tcp_pose` 只读反馈；
- 新鲜度、机器人身份和 pose 校验；
- quaternion/刚体矩阵/WebGL column-major 转换；
- 三点以上 Kabsch 标定和残差门限；
- `robot_tcp` 下行协议。

未经 `T_quest_from_robot` 标定的 robot pose 不会被直接当成 Quest pose 绘制。协议见 [docs/features/visualization-protocol.md](docs/features/visualization-protocol.md)。

## 安装与启动

```bash
python -m pip install -r hardware/quest_dev/requirements.txt
python hardware/quest_dev/server.py
```

需要 visualization 的离线标定或 Redis 反馈时，再安装：

```bash
python -m pip install -r hardware/quest_dev/requirements-visualization.txt
```

也可以使用 package 入口：

```bash
python -m hardware.quest_dev.server
```

默认监听 `8444`，避免与现有 `quest_pose` 的 `8443` 冲突。首次 HTTPS 启动会生成开发用自签名证书。Quest 与主机需要在同一局域网，并在 Quest Browser 中手动接受证书风险。

桌面 HTTP 测试：

```bash
python hardware/quest_dev/server.py --http
```

## Feature 配置

Feature 位于：

```yaml
features:
  visualization:
    enabled: true
    client:
      axis_length_m: 0.18
      show_controller_axes: true
```

只有 `client` 子项会通过 `/api/config` 发给浏览器。Redis 等 host-only 配置不会暴露。

AR5 反馈默认关闭。需要时启用：

```yaml
features:
  visualization:
    robot_feedback:
      enabled: true
```

该 adapter 只调用 Redis `GET`，模块没有 Redis 写入接口。

## 离线标定工具

```bash
python -m hardware.quest_dev.tools.fit_alignment \
  hardware/quest_dev/examples/alignment_points.example.json
```

## 测试

```bash
bash scripts/run_pytest.sh -q hardware/quest_dev/tests
```

## 来源与边界

- `server.py` 来自 `quest_pose/server.py` 的 HTTPS、证书和 WebSocket 部分；遥操作发布、标定和 Redis 写入已移除。
- WebXR core 和 renderer 来自 `quest_pose/web/app.js`，现已拆成 core 与 browser feature。
- visualization 的反馈解析来自 `AR5_real/tools/target_pose_viewer.py`。
- pose 变换来自 `AR5_real/tools/jog.py`。
- Kabsch 拟合来自 `AR5_real/tools/estimate_workstation_y_rotation.py`。

本目录不跨模块 import `quest_pose` 或 `AR5_real`，原遥操作入口保持不变。
模块不会由 AgenticRobot 自动启动，也不拥有机器人控制或运动权限。首次 HTTPS 启动生成的
`assets/certs/*.crt` 与 `*.key` 是本机开发材料，已被模块 `.gitignore` 排除；只提交用于保留
目录边界的 `.gitkeep`。
