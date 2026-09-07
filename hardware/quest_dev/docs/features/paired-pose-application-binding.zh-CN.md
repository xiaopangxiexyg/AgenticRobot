# Quest–RM65 成对位姿操作说明

本文是 Quest 手柄固定在 RM65 末端后，进行只读成对位姿采集的现场操作指南。
它对应 `quest-rm65-operator-trial-v1` 候选采集 Profile，面向第一次联调和问题定位。

当前入口只做以下事情：

- 启动 Quest HTTPS/WebXR 服务和桌面操作页面；
- 读取选定 RM65 的被动姿态观测；
- 接收 Quest 手柄姿态；
- 把人工划定的静止窗口整理为内存中的候选成对位姿数据。

当前入口不会发送机械臂动作，不会执行 Home、Servo、Replay 或标定运动，也不会自动
验收标定、安装坐标系或渲染 policy future。`seal` 先封存当前进程内存中的候选；只有
显式加上 `--publish-sealed-evidence`（或 `--publish-diagnostic-fit`）时，干净退出后才
会把候选证据/诊断拟合写入 content-addressed runtime artifact。它们仍然不是标定验收
结果，也不会自动安装坐标系。

## 一、开始前检查

1. 确认电脑和 Quest 位于同一个可信局域网。`--trusted-lan` 会使用固定配对码
   `111111`，只适用于已经确认隔离和可信的局域网；协议认证仍然开启。
2. 确认你使用的是已审核的 RM65 IP。文档中的 IP 只是示例，不能根据网段猜测设备。
3. 将 Quest 手柄刚性固定在 RM65 末端，并确认不会松动、碰撞或遮挡急停。
4. 第一次只做连接和静止窗口测试，保持机械臂不动。需要移动机械臂获得多姿态数据时，
   必须另行执行已经审核的机械臂运动安全流程。
5. 确认急停可用，工作区无人员和障碍物；真实启动仍会进入现有 RM65 后端初始化流程，
   所以“未发送动作”不等于“启动过程绝对零写入”。

## 二、先做 Dry-run

在 AgenticRobot 项目根目录中执行。下面的 `read` 只要求你在桌面终端输入 IP，不会把 IP
写入脚本：

```bash
read -rp "请输入确认过的 RM65 IP: " QUEST_ARM_IP
QUEST_ARM_URDF=/home/user/zc/HarnessInputs/contract-assets-v1/rm65_inspire_tac/rm65_arm.urdf

bash scripts/run_quest_calibration.sh \
  --side left \
  --arm-ip "$QUEST_ARM_IP" \
  --arm-urdf-path "$QUEST_ARM_URDF" \
  --trusted-lan \
  --dry-run
```

检查 JSON 输出至少包含：

```json
{
  "safety": {
    "policy": "hold",
    "send_actions": false,
    "end_effectors": false,
    "cameras": false
  },
  "quest_pairing": {
    "fixed_code": "111111"
  }
}
```

如果使用左臂，将 `--side right` 改为 `--side left`。Dry-run 不打开 Quest 端口，也不
创建 RM65 后端。

## 三、启动真实连接测试

确认 Dry-run 正确后，重复同一条命令并删除最后的 `--dry-run`：

```bash
bash scripts/run_quest_calibration.sh \
  --side left \
  --arm-ip "$QUEST_ARM_IP" \
  --arm-urdf-path "$QUEST_ARM_URDF" \
  --trusted-lan
```

也可以直接使用 Python 入口：

```bash
python -m control_station.quest_calibration.launch \
  --side left \
  --arm-ip "$QUEST_ARM_IP" \
  --arm-urdf-path "$QUEST_ARM_URDF" \
  --trusted-lan
```

启动后：

- 桌面操作页面会自动打开；
- 终端会打印 Quest 基础 URL、配对码和一步访问 URL；
- RM65 姿态观测和 Quest 接收器会在同一个应用生命周期内运行。

不要把 `--operator-host 127.0.0.1` 复制到 Quest 浏览器中。Quest 必须访问终端打印的
电脑局域网地址；`127.0.0.1` 只代表 Quest 自己。

## 四、Quest 端连接

1. 在 Quest 浏览器中重新打开之前成功使用的
   `https://<电脑局域网IP>:8444/`。
2. 如果出现本地开发证书警告，确认继续访问。
3. 在配对页面输入 `111111`。
4. 启动 MR/VR，允许浏览器需要的 XR 权限。
5. 保持固定手柄静止，等待桌面页面显示候选控制器和有效姿态。

若页面提示无法访问网页，先确认使用的是电脑的局域网 IP、不是 `127.0.0.1`，再确认
电脑与 Quest 在同一局域网、终端进程仍在运行、端口没有被防火墙拦截。不要把完整的
带认证 header 的 URL 粘贴到共享聊天、命令历史或截图中。

## 五、终端采集一个静止窗口

1. 确认 Quest 页面已配对并进入 MR/VR，终端已经选中正确的 Quest 控制器和 RM65 侧别。
2. 在启动终端按一次 `r`；保持机械臂和手柄静止，窗口会自动记录固定 3 秒并排空两侧事件。
3. 等待终端显示 `window completed`，改变到下一个姿态，再按一次 `r` 进入下一窗口。
4. 先完成 1–2 个窗口验证链路，再决定是否继续采集；完成至少 3 个窗口后按 `c` 封存并运行 AX=XB 诊断解算。

`operator-trial-v1` 的主要门槛是：窗口 1–5 秒（当前终端固定记录 3 秒）、至少 5 个机器人样本、至少 10 个
Quest 样本，以及有限的平移和旋转变化。单个静止窗口只产生一个代表性成对位姿，不会
自动拆成多条标定姿态。若 Quest 短暂报告 `emulated_position=true`，该事件会保留在
有序观测证据中但不会进入标定数据集；只有真实跟踪的 Quest 样本参与计数、配对和解算。
如果真实样本不足，窗口仍会被拒绝，而不会用模拟位姿补足。

Quest 手柄坐标轴的显示使用独立的仅渲染末帧缓存：WebXR 短暂没有返回
`gripSpace` 位姿时，画面可继续显示最近一次真实跟踪位姿，但该缓存不会发送给
`controller_pose` 采集通道，也不会进入标定数据集。画面中的轴因此可能是冻结的诊断
显示；采集仍以原始真实 Quest 样本为准。

按 `c` 后，Quest 会先显示诊断得到的机械臂根坐标系，再显示最后一个采集姿态的 TCP
末端标记；后续新鲜的 RealMan 只读反馈会更新末端位置。该预览不等于正式验收，也不会
安装坐标系或驱动机械臂。

真正的标定数据还需要覆盖工作空间中的多个位置，并包含绕两个以上非平行轴的姿态变化。
本入口不移动机械臂，也不替操作者决定这些运动姿态。

## 六、工具 Profile 和停止条件

位置观测后端会读取 RM65 当前激活的 RealMan 工具名，并尝试加载同侧的本地 TCP Profile。
正常情况下终端会出现类似：

```text
auto-loaded TCP profile ... configured_tcp_enabled=...
```

出现以下任一情况时，停止采集并先核对工具安装和 Profile，不要继续拟合：

- `active_tool_profile_unusable`；
- `active_tool_profile_tcp_missing`；
- `pose_source...failed`；
- 工具名、机械臂侧别或 TCP Profile 不匹配；
- 姿态观测被标记为 stale、invalid 或 identity drift。

## 七、正常关闭

所有需要保留的状态或截图先记录，再回到启动终端按一次 `Ctrl+C`。应用会按顺序停止
Quest 接收器、桌面页面和 RM65 观测资源。不要直接关闭终端窗口，也不要在应用仍运行时
重复启动第二个相同端口的实例。

## 八、常见故障

| 现象 | 先检查 | 不要做的事 |
| --- | --- | --- |
| Quest 无法访问网页 | 使用电脑局域网 IP、`https://`、`:8444`；确认终端仍运行 | 不要在 Quest 中使用 `127.0.0.1` |
| 配对码无效 | 当前实例是否使用 `--trusted-lan`；输入是否为 `111111` | 不要把旧实例的配对页面当成新实例 |
| 桌面没有控制器 | Quest 是否已进入 MR/VR，浏览器 XR 权限是否允许 | 不要按数组顺序猜控制器身份 |
| RM65 没有姿态 | IP 是否经过确认，机械臂是否可读，工具 Profile 是否匹配 | 不要反复尝试未经确认的 IP |
| `seal` 后找不到文件 | 当前版本只在内存中保留候选 | 不要把内存候选称为已发布标定 |

## 九、当前边界和下一步

本说明对应的是“连接 + 最简单几何体/手柄姿态 + 静止窗口候选”的操作阶段。若在
干净退出前显式加入 `--publish-sealed-evidence --publish-diagnostic-fit`，可得到候选
证据和离线 AX=XB 诊断报告；下一阶段仍需人工验收、安装 `T_quest_from_robot`，最后才
讨论把 policy 预期轨迹投影到 Quest。任何前一阶段的成功都不能自动证明后一阶段已经
完成。
