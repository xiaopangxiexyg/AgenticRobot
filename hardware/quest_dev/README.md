# Quest Dev

`quest_dev` 是 AgenticRobot 内独立的 Quest Browser/WebXR 开发基座。当前 R0 只建立
安全连接、最小几何可视化、只读手柄位姿遥测和可重放标定拟合；它尚未接入
`control_composition` 的 future visualization，也不会启动机械臂、产生控制目标或执行运动。

## R0 已有边界

- core protocol v2：同源 WSS、页面会话配对、认证客户端广播；
- WebXR `immersive-ar` 优先、`immersive-vr` 回退，并报告实际 reference space；
- 每个 reference-space generation 独立锚定的黄色线框立方体；
- 手柄坐标轴和最多 20 Hz 的 `gripSpace` 只读遥测；
- host 侧按认证客户端保留 latest-only 样本，reset/end/disconnect 立即失效；
- 固定在 TCP 上的 Quest 手柄与机器人 TCP 成对位姿离线拟合；
- 可选 AR5 Redis `GET` 反馈；默认关闭，且当前不会在 Quest 中绘制机器人 TCP。

## 目录结构

```text
hardware/quest_dev/
├── server.py                          # HTTPS/WSS、配对与 feature 生命周期
├── protocol.py                        # core protocol v2
├── config.yaml
├── features/
│   ├── controller_pose/
│   │   ├── protocol.py                # 严格的只读 grip pose wire parser
│   │   └── runtime.py                 # session/generation 绑定的 latest-only 状态
│   └── visualization/
│       ├── runtime.py                 # host-side visualization runtime
│       ├── protocol.py                # robot_tcp 下行协议
│       ├── paired_pose_calibration.py # 两个固定 SE(3) 的离线拟合
│       ├── alignment.py               # 已有的对应点 Kabsch 工具
│       └── transforms.py
├── web/
│   ├── app.js                         # 认证后的 WebXR/WSS lifecycle
│   ├── feature_registry.js            # browser feature hooks
│   └── features/                      # visualization 与 controller_pose
├── docs/
├── tools/
└── tests/
```

架构与 feature 规则见 [docs/architecture.md](docs/architecture.md)。

## 安装与启动

```bash
python -m pip install -r hardware/quest_dev/requirements.txt
python -m hardware.quest_dev.server
```

默认监听 `8444`。首次 HTTPS 启动会生成本机开发用自签名证书；Quest 与主机需在
同一受信局域网，且需在 Quest Browser 中手动接受证书风险。

每次启动默认生成新的随机 6 位数字配对码。终端会分别打印基础网址、6 位码和含
`#pair=...` fragment 的一键网址：可以只在 Quest 输入一次基础网址，然后在页面的数字
键盘中输入 6 位码；也可以直接输入一键网址。URL fragment 不随 HTTP 请求发送，浏览器
读取后立即从地址栏移除，但配对码会在当前页面内存中保留以支持 WSS 重连。需要固定码
时由调用环境提供恰好 6 位 ASCII 数字组成的 `QUEST_DEV_PAIRING_TOKEN`，不要把它写入
仓库或共享日志。

若 Quest 显示网页不可访问，依次确认：输入的是终端打印的 `https://LAN_IP:8444/`
（不是 `http://` 或 `localhost`）；已在 Quest Browser 高级选项中接受自签名证书；主机
防火墙已允许 Quest 所在局域网访问 TCP `8444`；Wi-Fi/热点没有开启 AP/client isolation。
主机可先用下列命令确认服务和证书监听正常：

```bash
curl --noproxy '*' -k https://LAN_IP:8444/api/config
```

若 UFW 已启用，按实际网段最小范围放行，例如：

```bash
sudo ufw allow from 192.168.70.0/24 to any port 8444 proto tcp
```

桌面传输测试可使用：

```bash
python -m hardware.quest_dev.server --http
```

明文 HTTP/WS 模式会无条件绑定 `127.0.0.1`，不能从 Quest 或 LAN 访问，也不能代替
Quest WebXR 的 HTTPS secure-context 检查。

`/api/status` 需要同一 6 位 pairing code：

```bash
curl -H "Authorization: Bearer $QUEST_DEV_PAIRING_TOKEN" \
  https://HOST:8444/api/status
```

使用随机 token 启动时，请从终端 pairing URL 取得 token；避免让包含 header 的命令进入
共享 shell history。

## 最小几何体与手柄遥测

进入 MR/VR 后，visualization feature 在首次有效 viewer pose 前方放置一个黄色线框
立方体，并绘制手柄 RGB 坐标轴。reference space 发生 `reset` 时旧锚点立即失效，下一
帧在新 generation 中重新锚定。参数位于 `features.visualization.client`。

`controller_pose` feature 上传 `T_reference_from_grip`，但不保存轨迹，也不提供动作、
目标或控制字段。协议见
[docs/features/controller-pose-protocol.md](docs/features/controller-pose-protocol.md)。

## 固定手柄的成对位姿标定

将 Quest 手柄刚性固定在机器人 TCP 后，单纯对 TCP 与手柄原点做 Kabsch 会把未知安装
偏置误当成空间变换。R0 因此拟合：

```text
T_questReference_from_controllerGrip(i)
  = T_questReference_from_robotReference
  × T_robotReference_from_tcp(i)
  × T_tcp_from_controllerGrip
```

离线工具同时求 `T_questReference_from_robotReference` 和
`T_tcp_from_controllerGrip`：

```bash
python -m hardware.quest_dev.tools.fit_paired_pose_calibration paired_poses.json
```

数据格式、静止窗口证据和退化判据见
[docs/features/paired-pose-calibration.md](docs/features/paired-pose-calibration.md)。工具输出
固定为 `accepted: null`：拟合结果只是可重放诊断证据，不会写入 frame graph，也不能
自行授权机器人绘制或运动。R0 尚未实现真实机器人/Quest 同步采样器与人工验收 profile。

已有的 `fit_alignment` 只适合确实共点、且不存在未知旋转安装偏置的对应点场景：

```bash
python -m hardware.quest_dev.tools.fit_alignment \
  hardware/quest_dev/examples/alignment_points.example.json
```

## 测试

```bash
bash scripts/run_pytest.sh -q hardware/quest_dev/tests
node --test hardware/quest_dev/tests/*.test.js
```

测试覆盖协议、认证、位姿严格校验、生命周期失效、变换、只读反馈与合成成对位姿拟合；
它们不构成 Quest 真机显示、网络时延、真实静止检测或机械臂运动证据。

## 来源与所有权

- HTTPS/WSS 与 WebXR 基础来自既有 `quest_pose`，遥操作发布和 Redis 写入已移除；
- visualization 反馈解析来自既有 AR5 只读反馈契约；
- `hardware/quest_dev` 只拥有 Quest 接入与 feature-local 诊断，不拥有控制 authority、
  canonical safety、轨迹生成、future selection 或执行生命周期；
- 模块不会被 AgenticRobot 自动启动，也不跨模块 import `quest_pose`、AR5 runtime、
  `control_composition`、`control_station` 或 RPent。

证书私钥和生成材料位于忽略的 `assets/certs/`；只提交 `.gitkeep` 以保留目录边界。
