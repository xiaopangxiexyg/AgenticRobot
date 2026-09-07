# Quest 被动标定与 RM65 可视化任务交接文档

## 1. 文档目的与当前状态

本文档用于交接 AgenticRobot 项目中的 Quest 被动标定、RealMan RM65 只读状态
观测、坐标系解算和 Quest 端机械臂可视化任务。它记录已经确认的需求、实现边界、
数据链路、操作流程、调试命令、Git 规则、验收证据和未完成事项。

当前实现是“候选标定与诊断预览”模式，不是正式标定安装，也不是机械臂控制模式。
`c` 解算得到的结果只发送到 Quest 做预览；不会把变换写入正式配置，也不会改变
机械臂控制器状态。

当前代码交付信息：

| 项目 | 当前值 |
| --- | --- |
| 开发分支 | `task/codex/quest-passive-left-b426-all-20260907` |
| 当前提交 | `ff41f224b612ccf01bad25ea70ea3a2604f3b64f` |
| 基线提交 | `b426d08c0ffaffb4878b376c672e362f85c0a278` |
| 远端主机 | `user@192.168.70.116` |
| 远端 worktree | `/home/user/zc/AgenticRobot-quest-passive-20260906` |
| 远端状态 | `HEAD == origin/task/codex/quest-passive-left-b426-all-20260907`，工作区干净 |
| Python | `/home/user/miniconda3/envs/dexteleop310/bin/python` |
| 左臂 IP | `192.168.101.21` |
| 右臂 IP | `192.168.1.20`，本任务不参与标定 |
| 左臂 URDF | `/home/user/zc/HarnessInputs/contract-assets-v1/rm65_inspire_tac/rm65_arm.urdf` |
| Quest reference space | 默认 `local-floor` |
| 配对码模式 | `--trusted-lan`，默认配对码 `111111` |

已验证的软件证据：

- 远端 Python Quest/RM65 相关测试：`205 passed, 4 subtests passed`；
- 本地浏览器 JS 契约测试：`5 passed`；
- 远端 Node 版本不支持 `node --test`，但所有 Quest 浏览器脚本均通过 `node --check`；
- 最终 dry-run 已确认 `send_actions=false`、`cameras=false`、
  `end_effectors=false`；
- 尚未把“真实 Quest + 真实左臂完成三组静止采样并得到合格拟合”的结果写成硬件验收证据。

## 2. 原始任务要求

### 2.1 功能目标

任务的目标链路是：

1. 在 Quest 中获取左手柄的 WebXR `gripSpace` 姿态；
2. 读取左侧 RealMan RM65 的当前关节状态，并通过既有 FK 得到机器人 TCP 姿态；
3. 操作者将左手柄与机械臂末端刚性固定，采集多个静止点位的成对姿态；
4. 解算 Quest reference space、机械臂基座坐标系和 TCP/手柄安装关系；
5. 将机械臂基座坐标系和末端 TCP 在 Quest 中渲染出来，用于诊断和后续可视化；
6. 后续才考虑正式标定接受、安装和动作轨迹可视化，本阶段不执行这些写入或控制动作。

### 2.2 设备与左右侧定义

- 本次标定固定使用“左手柄 + 左臂”；
- 左臂地址为 `192.168.101.21`；
- 右臂地址为 `192.168.1.20`，仅作为项目既有左右臂定义的参考，不应被本次命令连接；
- 左右侧的语义必须以 AgenticRobot 项目现有 URDF、配置和 side binding 为准，不能仅凭
  文件名或终端显示猜测；
- 当前命令必须使用 `--side left`、`--active-sides left`（由 launcher 注入）和
  `--controller-handedness left`。

### 2.3 明确允许与禁止的行为

允许：

- 接入 Quest WebXR 页面；
- 读取 Quest 当前手柄姿态；
- 读取 RealMan 当前状态；
- 使用既有 FK 计算 TCP 观测姿态；
- 记录静止窗口、执行离线/诊断 AX=XB 拟合；
- 在 Quest 中显示诊断得到的机器人基座和 TCP；
- 将明确请求的候选证据和诊断报告发布到项目的 artifact layout。

禁止：

- 通过 Redis 作为本任务的数据源或控制通道；
- 发送机械臂动作、目标、复位、夹爪命令或控制指令；
- 使用 `--send-actions`；
- 把被动读取路径误称为“机械臂已经被标定并可控制”；
- 自动接受、安装或覆盖正式 calibration frame；
- 在 Quest 页面或 feature 中创建动作控制字段；
- 为了让测试通过而 reset、clean、stash、移动或删除其他 worktree；
- 通过历史缓存姿态代替当前有效的 WebXR `getPose()`；
- 在没有重新检查 branch、HEAD、工作区和远端输入 SHA 的情况下同步或推广代码。

## 3. 系统链路与所有权

### 3.1 总体数据流

```text
Quest WebXR frame
  ├─ frame.getPose(gripSpace, referenceSpace)
  │    ├─ 当前真实 pose -> raw controller pose
  │    ├─ emulated pose -> raw diagnostic + emulated pose
  │    └─ null/异常/缺少 gripSpace -> tracking diagnostic，无 raw pose
  │
  ├─ controller_pose feature
  │    └─ WSS authenticated sample
  │         └─ ControllerPoseTelemetryRuntime latest-only admission
  │              └─ ordered accepted/invalidation events
  │                   └─ paired-pose capture candidate/window reducer
  │
  └─ visualization feature
       ├─ 仅用于显示的短时旧姿态缓存（最多 250 ms）
       └─ robot-base/TCP preview renderer

RealMan RM65 observation-only path
  ├─ 既有 RM65 backend 读取当前硬件状态
  ├─ canonical FK -> T_robot_base_from_tcp
  ├─ application-owned passive robot pose journal
  └─ 与 Quest accepted events 在静止窗口内配对

三个或更多静止窗口
  └─ AX=XB candidate fit
       └─ diagnostic preview message
            └─ Quest 绘制左臂 base axes + TCP axes/marker
```

### 3.2 模块职责

| 模块 | 职责 | 不应做的事 |
| --- | --- | --- |
| `control_station/quest_calibration/launch.py` | 组合左臂只读观测、Quest host、终端 `r/c` 控制和发布选项 | 不提供任意 `run_policy` 参数透传，不发送动作 |
| `hardware/quest_dev/web/app.js` | 创建 XR session、reference space、逐帧采样和显示调用 | 不把显示缓存送入采集 |
| `hardware/quest_dev/web/controller_pose_sampling.js` | 调用 `getPose()`，区分真实、emulated、不可用和异常 | 不把 `null` 当成有效姿态 |
| `hardware/quest_dev/web/controller_pose_display.js` | 给渲染器提供最多 250 ms 的显示平滑缓存 | 不作为标定采样源 |
| `hardware/quest_dev/web/features/controller_pose/feature.js` | 上传 raw pose 与每个 input source 的 tracking diagnostics | 不发送 action/target/control 字段 |
| `hardware/quest_dev/features/controller_pose/protocol.py` | 校验矩阵、session、source、诊断与 pose 的一致性 | 不保存轨迹，不调用机械臂 |
| `hardware/quest_dev/features/controller_pose/runtime.py` | 做认证、序号、频率、session/generation admission，只保留 latest | 不自己刷新 RealMan |
| `teleop_runtime/deploy/quest_paired_pose_capture.py` | 选择候选、固定窗口、配对 Quest/机器人事件、封存候选证据 | 不接受正式标定，不安装 frame |
| `hardware/quest_dev/features/visualization/paired_pose_capture.py` | 被动静止窗口 reducer、站定性检查、R0 数据集投影 | 不拥有设备生命周期或后台设备线程 |
| `hardware/quest_dev/web/features/visualization/feature.js` | 接收 preview/只读 robot feedback 并绘制 base/TCP | 不把 preview 变成 accepted binding |
| `RealMan backend / ArmStateReader` | 读取并发布既有左臂状态和 FK 姿态 | 不由 Quest feature 发起第二套读取或控制 |

一个 Quest host 只创建一个 controller-pose runtime；一个应用只借用这个已启动的
runtime。禁止为了绕过候选或状态问题而创建第二个 host、第二个 controller runtime、
第二个 RealMan reader 或全局隐式 registry。

## 4. 坐标系与解算含义

固定手柄安装到机械臂 TCP 后，采样点 `i` 满足：

```text
T_questReference_from_controllerGrip(i)
  = T_questReference_from_robotReference
  × T_robotReference_from_tcp(i)
  × T_tcp_from_controllerGrip
```

其中：

- `T_questReference_from_controllerGrip(i)`：Quest 当前 reference space 到手柄 grip 的姿态，
  来自当前 WebXR frame；
- `T_robotReference_from_tcp(i)`：左臂基座到 TCP 的姿态，来自 RealMan 只读状态和 FK；
- `T_questReference_from_robotReference`：要显示的 Quest reference space 到左臂基座的变换；
- `T_tcp_from_controllerGrip`：手柄刚性安装偏置；
- 采样中只允许使用同一 Quest session、同一 reference-space generation、同一
  controller source identity 和同一机器人 source binding。

当前 `c` 计算的是候选诊断 fit。它发布的 preview 至少包含：

- Quest session/reference-space epoch；
- `matrix_quest_from_robot_base`；
- 机器人 frame 和 TCP frame；
- 最后一个静止样本中的 `matrix_robot_from_tcp`，以便在下一包实时反馈到达前也能显示 TCP；
- preview/candidate/accepted 状态明确标记为“diagnostic preview”，不能冒充正式标定。

Quest 中默认绘制：

- 左臂基座 RGB 坐标轴；
- TCP RGB 坐标轴；
- TCP 线框/末端 marker；
- controller axes 和 reference marker（用于诊断）。

reference space reset 会使旧 preview、旧 controller display cache 和旧 generation 数据
失效。必须等待新 generation 的真实 pose 后重新采集，不能跨 generation 拼接。

## 5. 手柄追踪语义：必须区分“可见”和“可采集”

这是本任务最关键的故障点。

### 5.1 两套列表

每个 XR frame 同时产生两套不同语义的数据：

1. `controllerPoses`：原始采集列表，只包含当前 frame 有可发送 pose 的 controller；
2. `displayControllerPoses`：渲染列表，可以在短时间内保留上一帧真实 pose，但会标记
   `displayOnly=true`、`trackingState=untracked/emulated`。

因此，“在 VR 中看到了左手柄/手柄轴”不能证明当前 `getPose()` 有效。Quest 系统自身
的手柄模型也可能继续显示；这同样不是标定证据。

### 5.2 当前诊断原因

`controller_diagnostics` 为当前 `XRInputSource` 上报 pose-redacted 状态：

| `reason` | 含义 | 能否进入标定 |
| --- | --- | --- |
| `tracked` | 当前真实 grip pose 有效 | 可以 |
| `emulated_position` | WebXR 给出的是估计/模拟位置 | 不可以 |
| `missing_grip_space` | `XRInputSource.gripSpace` 不存在 | 不可以 |
| `pose_unavailable` | `frame.getPose(gripSpace, referenceSpace)` 返回 `null` | 不可以 |
| `pose_transform_unavailable` | pose 没有有效 4x4 transform | 不可以 |
| `pose_matrix_invalid` | 矩阵含非有限值或不满足刚体约束 | 不可以 |
| `pose_error` | `getPose()` 抛出异常 | 不可以 |

host parser 会校验诊断和 pose 的一致性，例如 `untracked` 不能同时携带真实 pose，
`tracked/emulated` 必须有 pose，避免浏览器错误数据伪装成可采集样本。

### 5.3 当前选择门槛

选择历史候选进入 capture 前，必须同时满足：

- latest sample 存在；
- latest telemetry 年龄不超过 0.5 秒；
- client instance、XR session、reference space type/generation 均与候选一致；
- source session ID、handedness、target ray mode 均一致；
- `emulated_position` 为 `false`；
- 当前 diagnostic 不是 `pose_unavailable`、`missing_grip_space` 等失败状态。

失败不会把 session 锁死为 terminal，而是给出可重试原因。

## 6. 采集状态机与现场操作

### 6.1 启动

启动后终端会输出 Quest URL、配对码、operator 状态和安全边界。使用 Quest 打开
输出的 URL，输入配对码 `111111`，进入 WebXR VR/MR。

确认以下状态后再按键：

- 页面已 authenticated；
- XR session 已 started；
- 左手柄为 `tracked-pointer`；
- 左手柄当前有真实 grip pose；
- 页面 controller status 不再持续显示 `pose_unavailable`；
- 机械臂处于静止状态，且本次只读取左臂。

### 6.2 第一次按 `r`

第一次 `r` 的语义是“请求下一次静止采集”，不是手动开始/结束两个阶段。

- 如果左手柄当前有效：自动选择唯一的左手柄候选并开始 `3s` 窗口；
- 如果候选存在但 `getPose()` 暂时为空：终端输出 `r armed; waiting for a current real grip pose`，
  保持等待；tracking 恢复后自动选择并开始 3 秒窗口；
- 在等待期间不要连续按 `r`，否则只会收到“previous window operation is still draining”；
- 如果没有任何候选，先检查 Quest 是否已配对并进入 XR。

### 6.3 静止窗口

窗口开始后：

- 保持左臂和左手柄刚性关系不变；
- 保持该点位静止至少 3 秒；
- 不需要再按一次 `r` 结束；
- 程序到达 3 秒边界后自动 pin end cut、排空 Quest/RealMan 事件并完成窗口；
- 机器人动作发送始终关闭。

当前 `operator-trial-v1` profile 的关键门槛：

| 项目 | 要求 |
| --- | ---: |
| 窗口最短/最长 | 1 s / 5 s |
| 当前终端窗口 | 3 s |
| source span 最小值 | 0.75 s |
| 每窗最少机器人样本 | 5 |
| 每窗最少 Quest 样本 | 10 |
| 最大 Quest/机器人位置 excursion | 0.008 m / 0.003 m |
| 最大 Quest/机器人旋转 excursion | 1.5° / 0.75° |
| 最大 Quest-机器人配对 skew | 100 ms |
| `c` 前最少已接受窗口 | 3 |

### 6.4 后续 `r` 和 `c`

- 一个窗口自动完成并看到 `window completed` 后，再按 `r` 进入下一个窗口；
- 至少完成 3 个被接受的静止窗口后，按 `c`；
- `c` 会封存候选证据、运行诊断 AX=XB fit，并发布 Quest preview；
- `c` 不接受正式标定、不安装 frame、不发送机器人动作；
- 如果窗口因 tracking loss、duration、sample count、stationarity 或 pair skew 被拒绝，
  保留已有证据状态，重新等手柄 tracking 稳定后再按 `r`；
- 完成诊断后按 `Ctrl+C`，让 launcher 正常关闭 host、observer、RealMan observation
  和终端循环。

### 6.5 期望终端输出

正常开始窗口：

```text
[Quest] selected controller candidate input-1 (left); robot_side=left
[Quest] window started: ...; hold the arm/controller still for 3s; it will finish automatically
...
[Quest] 3s static window elapsed; completing automatically and draining Quest/RealMan samples...
[Quest] window completed: ...; samples=robot=...,quest=...
```

当前问题的可诊断输出应类似：

```text
[Quest] r armed; waiting for a current real grip pose: ... pose_unavailable (WebXR getPose(gripSpace, referenceSpace) returned null) ...
```

这表示可重试，不表示 capture session 已 terminal。若输出 `passive capture became terminal`，
则必须重新启动整个 observation-only session，不能继续在同一 terminal 状态中按键。

## 7. 正确启动命令

### 7.1 远端真实只读运行命令

以下命令应直接粘贴到 `user@192.168.70.116` 终端。变量必须先单独赋值；不要把变量赋值
和 `"$AR_PYTHON"` 写在同一条赋值命令中，也不要使用会触发 zsh `read: -p: no coprocess`
的 `read -p`。

```bash
cd /home/user/zc/AgenticRobot-quest-passive-20260906

AR_PYTHON=/home/user/miniconda3/envs/dexteleop310/bin/python
AR_RM65_URDF=/home/user/zc/HarnessInputs/contract-assets-v1/rm65_inspire_tac/rm65_arm.urdf
AR_RM65_IP=192.168.101.21

"$AR_PYTHON" -m control_station.quest_calibration.launch \
  --side left \
  --arm-ip "$AR_RM65_IP" \
  --arm-urdf-path "$AR_RM65_URDF" \
  --controller-handedness left \
  --trusted-lan \
  --no-open-browser \
  --publish-sealed-evidence \
  --publish-diagnostic-fit
```

其中：

- `--publish-sealed-evidence` 发布候选采集证据；
- `--publish-diagnostic-fit` 发布候选诊断拟合报告；
- 两者都不代表正式 calibration acceptance 或 frame installation；
- 若只做现场验证、不希望写 artifact，可暂时去掉这两个 `--publish-*` 选项；
- 无论是否发布 artifact，都不能添加 `--send-actions`。

### 7.2 运行前 dry-run

正式接入 Quest/RealMan 前，先在远端执行：

```bash
cd /home/user/zc/AgenticRobot-quest-passive-20260906

/home/user/miniconda3/envs/dexteleop310/bin/python \
  -m control_station.quest_calibration.launch \
  --side left \
  --arm-ip 192.168.101.21 \
  --arm-urdf-path /home/user/zc/HarnessInputs/contract-assets-v1/rm65_inspire_tac/rm65_arm.urdf \
  --controller-handedness left \
  --trusted-lan \
  --no-open-browser \
  --dry-run
```

输出中必须确认：

```text
"side": "left"
"ip": "192.168.101.21"
"send_actions": false
"cameras": false
"end_effectors": false
"controller_handedness": "left"
```

### 7.3 第二个终端查看 Quest host 状态

真实 launcher 运行期间，可从同一远端主机的第二个终端查看 pose-redacted 状态：

```bash
curl -ksS \
  -H 'Authorization: Bearer 111111' \
  https://127.0.0.1:8444/api/status
```

重点观察：

- controller runtime 的 `clients`；
- `controller_tracking[].tracking_state`；
- `controller_tracking[].reason`；
- `latest_age_s`；
- `reference_space_generation` 和 `last_sequence`；
- Quest host 是否仍在监听。

如果得到 `connection refused`，优先检查 launcher 是否已经退出、是否按了 `Ctrl+C`、
是否端口被其他进程占用；不能据此推断 Quest tracking 正常或失败。

## 8. 调试与故障处理矩阵

| 现象 | 真实含义 | 处理 |
| --- | --- | --- |
| VR 中看到手柄，但终端 `pose_unavailable` | 显示的是系统模型或短时旧缓存，当前 `getPose()` 为 null | 保持真实 grip tracking，等待状态恢复；第一次 `r` 已 armed 时不要重复按 |
| `missing_grip_space` | 当前 input source 没有 gripSpace | 检查是否使用真实 tracked controller，而不是 gaze/screen input |
| `emulated_position` | WebXR 位置被估计，不能用于标定 | 等待 Quest 获得真实空间追踪 |
| `latest telemetry is stale` | host 最近收到的样本超过 0.5 s | 检查 WSS、Quest 页面、网络和 `/api/status` |
| `selected source is absent` | session.inputSources 中已没有历史 source | 重新握持/重启 XR session，必要时重启 launcher |
| `static window rejected: duration is outside profile` | 窗口边界或事件排空不满足 profile | 不手动提前结束；重新按 `r`，完整保持 3 s |
| `quest sample count` 不足 | 真实 pose 帧不足或 tracking 中断 | 重新采集，确保 tracking 持续且不遮挡控制器 |
| `robot sample count` 不足 | RealMan 只读观测未达到窗口样本门槛 | 检查左臂 IP、SDK 观测链路和 arm state refresh |
| `candidate selection ambiguous` | 存在多个符合条件的手柄候选 | 只保留左手柄/关闭其他 XR input source，保持 handedness=left |
| `passive capture became terminal` | session/reference/source/reducer 事实不可再信任 | Ctrl+C 后重新启动 observation-only session |
| `unrecognized arguments` | 实际运行的目录/分支不是同步后的版本 | 先检查 `pwd`、`git status`、`git rev-parse HEAD` |
| `cd: no such file or directory` | 使用了旧项目目录名 | 使用 `/home/user/zc/AgenticRobot-quest-passive-20260906` |
| `read: -p: no coprocess` | zsh 中 `read -p` 不是 bash prompt 语义 | 删除交互确认行，直接使用固定已复核 IP |

遇到错误时必须保留：完整终端日志、capture ID、window ID、Quest session ID、
reference generation、`/api/status` 输出和 Git HEAD。不要只截取最后一行。

## 9. 协议、时间和一致性规则

### 9.1 Quest source identity

WebXR 没有跨 session 可依赖的硬件序列号。当前浏览器使用：

```text
handedness + targetRayMode + session-local occurrence -> input-N
```

`profiles` 是能力诊断字段，不是稳定身份。浏览器可以从 generic profile 更新为具体
controller profile，但不能因此生成第二个候选或使已选 source 失效。以下变化则必须
视为身份边界：

- XR session 变化；
- reference-space generation 变化；
- handedness 变化；
- target ray mode 变化；
- source 从当前 `inputSources` 消失。

### 9.2 时间域

- Quest/机器人窗口配对只使用接收端同一 monotonic clock domain 的时间；
- 不能把 WebXR `xr_frame_time_ms` 与机器人 native timestamp 直接相减；
- 每个 source 的 native time、receiver time、clock descriptor 和 source epoch 必须保留；
- event sequence 是 ordering/fence 依据，不等价于 wall-clock 时间；
- Quest callback 可能在 HTTP 线程外序到达，必须先按全局 event sequence 排序再筛选 source；
- `unsubscribe_with_cut()` 是 capture close fence，不能用普通 `unsubscribe()` 加另一个
  event cut 代替；
- robot journal 和 Quest runtime 的 event gap、binding drift、generation drift 都必须
  fail closed。

### 9.3 静止采样

每一个 window 使用半开事件区间：

```text
[start_next_event_sequence, end_next_event_sequence)
```

窗口结束不是立即同步 callback 完成，而是 pin end watermark 后排空直到 watermark。
必须同时满足机器人和 Quest 的样本数、source span、静止 excursion、配对 skew 和
identity 约束。不能拿下一窗口或历史缓存补齐当前窗口。

## 10. Git、远端同步与交接规则

### 10.1 开发规则

- 一个任务使用一个隔离 worktree 和一个明确 owner；
- 写入前检查 `git worktree list --porcelain`、branch description、工作区脏状态和语义范围；
- 不覆盖其他 owner 的 worktree；
- 新提交会使旧的 ready 说明失效，必须重新写入完整 tip、验证命令和已知未验证范围；
- 同步前重新确认目标 branch 和输入 SHA；
- 远端只允许 `fetch` 后 `merge --ff-only`；
- fast-forward 失败时停止并报告，不能 reset、rebase、clean 或强推来制造同步；
- 当前分支已标记 `ready-for-integration`，当前 tip 为 `ff41f224`。

### 10.2 远端复核命令

```bash
ssh user@192.168.70.116 \
  'git -C /home/user/zc/AgenticRobot-quest-passive-20260906 status --short --branch'

ssh user@192.168.70.116 \
  'git -C /home/user/zc/AgenticRobot-quest-passive-20260906 rev-parse HEAD'
```

预期 HEAD：

```text
ff41f224b612ccf01bad25ea70ea3a2604f3b64f
```

同步流程必须是：

```bash
git push origin task/codex/quest-passive-left-b426-all-20260907

ssh user@192.168.70.116 \
  'git -C /home/user/zc/AgenticRobot-quest-passive-20260906 fetch origin task/codex/quest-passive-left-b426-all-20260907'

ssh user@192.168.70.116 \
  'git -C /home/user/zc/AgenticRobot-quest-passive-20260906 merge --ff-only origin/task/codex/quest-passive-left-b426-all-20260907'
```

不要使用 `git reset --hard`、`git clean`、强制 push 或全目录复制。

## 11. 测试与验证命令

### 11.1 远端 Python 回归测试

```bash
ssh user@192.168.70.116 \
  'cd /home/user/zc/AgenticRobot-quest-passive-20260906 && \
  /home/user/miniconda3/envs/dexteleop310/bin/python -m pytest -q \
    hardware/quest_dev/tests/test_controller_pose.py \
    hardware/quest_dev/tests/test_paired_pose_capture.py \
    hardware/quest_dev/tests/test_realman_feedback.py \
    hardware/quest_dev/tests/test_robot_feedback_canonical.py \
    hardware/quest_dev/tests/test_server.py \
    tests/test_control_station_quest_calibration_launch.py \
    tests/test_control_station_quest_calibration_web.py \
    tests/test_deploy_quest_paired_pose_capture.py \
    tests/test_quest_terminal_capture.py \
    tests/test_rm65_pose_observation_binding.py'
```

预期：`205 passed, 4 subtests passed`。

### 11.2 浏览器脚本语法检查

远端 Node 12 不支持 `node --test`，使用：

```bash
ssh user@192.168.70.116 \
  'cd /home/user/zc/AgenticRobot-quest-passive-20260906 && \
  for f in hardware/quest_dev/web/*.js \
           hardware/quest_dev/web/features/controller_pose/feature.js \
           hardware/quest_dev/web/features/visualization/*.js; do \
    node --check "$f" || exit 1; \
  done'
```

本地若 Node 支持内建 test runner，可额外执行：

```bash
node --test \
  hardware/quest_dev/tests/controller_pose_sampling.test.js \
  hardware/quest_dev/tests/controller_pose_display.test.js \
  hardware/quest_dev/tests/controller_pose_feature.test.js \
  hardware/quest_dev/tests/controller_source_identity.test.js \
  hardware/quest_dev/tests/visualization_feature.test.js
```

### 11.3 代码卫生

提交前至少执行：

```bash
git diff --check
git status --short --branch
git rev-parse HEAD
```

Python 代码变更执行对应环境的 `ruff check` 和 `py_compile`；浏览器代码执行
`node --check`。socket 测试若在受限 sandbox 出现：
`PermissionError: [Errno 1] Operation not permitted`，这是环境能力证据，必须在允许
创建本地测试 socket 的环境中重跑同一固定测试集合，不能修改代码绕过。

## 12. 结果呈现与交接格式

每次交接或现场测试报告必须按以下顺序呈现：

1. **代码身份**：仓库、branch、完整 HEAD、base、远端 worktree、是否 clean；
2. **变更范围**：文件、行为变化、是否修改公共协议或数据 schema；
3. **安全边界**：`send_actions`、camera、end-effector、Redis、calibration acceptance、
   frame installation 的实际状态；
4. **软件验证**：精确命令、Python/Node 版本、通过数量、失败/跳过及原因；
5. **硬件验证**：Quest 是否真实进入 XR、左手柄 tracking 是否连续、RealMan 是否只读、
   是否完成静止窗口、每窗样本数和 fit residual；
6. **可视化结果**：Quest session/reference generation、左臂 base 是否显示、TCP axes/marker
   是否显示、是否为 preview-only；
7. **证据位置**：capture ID、evidence hash、diagnostic fit hash、artifact path；
8. **未验证项和下一步**：不能把 dry-run、contract test 或 preview 当成物理验收。

推荐使用如下简短结论模板：

```text
代码：<branch>@<full SHA>，远端 worktree=<clean/dirty>
安全：send_actions=false；camera=off；end_effectors=none；正式标定未安装
软件：<精确命令>；<通过数>/<失败数>
硬件：Quest=<未运行/已进入 XR>；左手柄=<tracked/丢失>；左 RM65=<只读/未接入>
采集：windows=<数量>；每窗 robot/quest samples=<...>；fit RMS=<...>
显示：base=<显示/未显示>；TCP=<显示/未显示>；preview-only=<true>
未验证：<物理验收、网络、拟合质量或其他明确缺口>
```

## 13. 当前已实现的关键修复

本任务相关的主要提交含义如下：

- `6a90f22`：稳定浏览器 session-local controller identity；
- `bf58da4d`：把 tracking loss 与 passive capture 分离；
- `590bdc01`：tracking loss fail-safe；
- `eabf86c6`：选择候选前拒绝 stale controller candidate；
- `ff41f224`：上报 `getPose()` 细分原因、限制 display cache 250 ms、增加 0.5 s
  latest freshness gate，并使一次 `r` 在 tracking 恢复后自动开始 3 秒窗口。

修复后的核心原则是：

```text
VR 中可见 != 当前 raw grip pose 有效
历史候选存在 != 当前可以选择
diagnostic preview != 正式 calibration accepted
read-only RealMan feedback != 机械臂已被控制
```

## 14. 未完成事项与后续工作

以下事项不能在本交接中被宣称为已完成：

- 在真实现场完成左手柄/左臂刚性安装、至少三个静止窗口并记录合格 fit residual；
- 评估 Quest tracking 丢失频率、网络延迟和实际 `latest_age_s`；
- 评审 diagnostic fit 的几何误差、姿态覆盖和点位独立性；
- 由授权流程审核并安装正式 calibration frame；
- 在独立安全评审后实现或接入机械臂动作可视化轨迹；
- 为未来 trajectory renderer 增加独立的 accepted frame binding 和过期策略；
- 决定 250 ms display cache 是否需要变成配置项，并以现场证据调参；
- 为正式 calibration 设计持久化、版本化、回滚和接受门禁。

任何后续实现都必须继续保持本阶段的 read-only boundary，直到用户明确授权并完成
独立的动作控制安全评审。
