# Connection remote wrapper design

本目录用于放置局域网远程连接、目录同步、远程运行和真机设备访问相关的工具脚本。核心思路是：**Codex 只在本地运行，远程电脑只作为 SSH 可达的执行环境和硬件主机**。

本文是一份通用实现说明，用来指导后续增加 wrapper 包、脚本和安全规则。它不绑定某一个具体项目，也不要求远程电脑运行 Codex。

## 出发点

在真机开发中，常见环境通常不是单机闭环：

- 本地电脑上运行 IDE、Codex、代码编辑器和主要交互界面。
- 远程局域网电脑连接真实设备，例如 USB 设备、摄像头、VIVE、网口机械臂、采集卡、GPU 等。
- Python 程序必须在远程电脑上运行，才能访问远程电脑的 `/dev/*`、网卡、驱动、相机和机械臂控制网络。
- 本地 Codex 需要帮助我们修改代码、同步代码、启动远程测试、查看日志、抓取相机画面、检查设备状态，但不应该直接变成远程电脑上的常驻智能体。

因此我们采用：

```text
本地电脑
  Codex / IDE / CLI
  Connection wrapper scripts
  SSH client
  rsync / sshfs / port forward

远程电脑
  Python runtime / conda env
  project checkout or synced project directory
  USB devices / cameras / robot network / GPU
  logs / recorded data / runtime services
```

Codex 的所有决策、文件编辑和命令发起都发生在本地；远程电脑只执行被 wrapper 明确表达出来的动作。

## 目标

本工具集的目标如下：

1. 让 Codex 像调用本地命令一样调用远程 Python。
2. 让远程电脑上的 USB、摄像头、网口机械臂等真机设备可被受控访问。
3. 让本地代码可以可靠同步到远程运行环境。
4. 让远程日志、录制数据、截图、视频、调试包可以拉回本地。
5. 给危险动作建立清晰的安全边界，例如机械臂使能、回零、运动、USB reset、reboot 等。
6. 把复杂 SSH 命令封装为稳定 wrapper，减少人工和 Codex 拼接长命令时出错。
7. 保证 Codex 不需要安装到远程电脑，也不需要用 Codex remote project 模式。

## 非目标

这些事情不属于本工具集的默认目标：

- 不让远程电脑运行 Codex agent。
- 不默认开放远程机器的完整 shell 给任意自动化调用。
- 不绕过设备厂商 SDK、驱动权限、udev 规则或机械臂安全机制。
- 不把真实运动命令做成无确认、无状态检查的一键动作。
- 不把 `.codex/`、私钥、token、浏览器 profile 等敏感目录同步到远程。
- 不让 wrapper 隐式执行 `sudo`、`reboot`、`shutdown`、`rm -rf` 等高风险动作。

## 推荐工作流

### 1. 配置 SSH 免密

使用已有工具：

```bash
./setup_ssh_passwordless.sh -H REMOTE_IP -u user
```

成功后应能直接登录：

```bash
ssh user@REMOTE_IP
```

普通 `codex_remote` wrapper 只连接已经记录在本机 `known_hosts` 中的目标，
未知或变化的 host key 会失败。首次接入应通过上面的 SSH 初始化流程核对并建立
信任；不要在普通执行、同步或设备 wrapper 中临时放宽 host-key 校验。

建议在 `~/.ssh/config` 中配置固定别名：

```text
Host dex-remote
  HostName 192.168.20.170
  User user
  IdentityFile ~/.ssh/id_ed25519
```

当前工作站中，`dex-remote` 是 `192.168.20.170` 的默认 SSH alias。Codex
调用 wrapper 时仍应始终显式传入 `-H/--host`：用户未指定目标时使用
`-H dex-remote`；用户指定 IP、`user@host[:port]` 或其他 SSH alias 时使用
该目标。显式目标连接失败后不得回退到 `dex-remote`。

```bash
# 默认目标：当前 SSH 配置中的 192.168.20.170
./remote_whoami.sh -H dex-remote

# 用户指定的其他远端
./remote_whoami.sh -H user@192.168.20.171
```

裸 IP 只表达 SSH 主机，不能自动确定远端用户名、项目目录、Python 环境或
设备布局。`-H/--host` 表示执行 wrapper 的远端电脑；`--arm-ip` 等参数表示
由该电脑访问的机器人或控制器，两者不得混用。

### 2. 选择代码访问方式

常见有三种方式：

```text
方式 A：rsync 单向同步
  本地编辑 -> 同步到远程 -> 远程运行
  适合大多数开发和真机调试。

方式 B：NFS/SSHFS 挂载
  远程直接看到本地目录，或本地直接看到远程目录。
  适合共享数据，但要注意权限和敏感目录。

方式 C：远程独立 git checkout
  本地提交/推送，远程拉取。
  适合更正式的多机协作。
```

本目录已有：

```bash
./realtime_sync_ssh.sh -H REMOTE_IP -u user -l ~/Release -r '~/Release'
./lan_share_mount.sh -H user@REMOTE_IP -r /mnt/main-share
```

默认推荐优先使用 `realtime_sync_ssh.sh`，因为它边界清楚：本地是源，远程是运行副本。

### 3. 远程运行 Python

wrapper 不应该要求 Codex 每次手写复杂命令：

```bash
ssh dex-remote 'cd ~/DexTeleop && source ~/anaconda3/etc/profile.d/conda.sh && conda activate dexteleop310 && python xxx.py'
```

而应该封装成：

```bash
./remote_python.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- tools/inspect_dataset.py
./remote_module.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- tools.report_environment
./remote_pytest.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- tests/test_xxx.py -q
```

远程 Python 进程运行在远程电脑上，因此技术上可能访问真实硬件。通用 Python、
module 和 pytest wrapper 只允许 L3 非控制执行；关键词检查只是误调用防线，不能证明
任意代码没有设备副作用。设备或运动动作必须进入限制具体动作的专用 wrapper。

### B601 单轮采集管理

`remote_b601_managed_cycle.sh` 是显式真机授权下的专用入口，复用现有
`rebot_b601_rs_owner.py`，不接受任意脚本、运动 profile、力矩覆盖或命令尾参。
`--operation preflight` 只运行原 owner 的无 CAN 参数／资源检查；`start` 固定执行
所选固定 profile 一轮：`--recipe near-zero` 为默认近零 28 段，`--recipe four-centres`
为既有 3°／6° 四中心 136 段；不接受任意 profile 路径。启动声明本次无人介入、原位有
可靠托架。该声明不是传感器测量。
保留原始 1 N·m 两轴组合包络、0.195 rad/s 原始速度保护、40 ms watchdog 和全部回位门。
可选 `--field-report` 只启用原影子查询；本入口不启用前馈、不安装模型、不重置 CAN。

每次创建新的运行目录，记录完整 Git SHA、实际命令、配置摘要及进程启动标识。
先检查源码一致且干净、没有已有 owner，再执行原参数预检和 CAN 链路状态检查；管理
启动持有进程继承锁。仍须串行使用这台机械臂，不同时手动启动另一条开发线的 owner。
SSH 返回后子进程继续执行，连接超时不会以 SIGTERM 杀死控制进程。

`status --run-directory <returned-directory>` 是只读观察，分别报告进程状态、原 owner 的
实时阶段、采集完成度、回位失能回执和模型准备情况。`stop` 只通过同一 owner 的本地
控制端口请求保护保持，不发送重复 Ctrl-C，不自动升级为全失能。人工介入、页面关闭、
硬件故障后不自动重试或重新加力；需要人工处理时，保持进程和原有重力／阻尼协议。
通用 Python/module/pytest wrapper 继续拒绝这些控制入口；不使用新的授权口令或令牌。

### 冻结抓姿数据的限定探针

`remote_readonly_agent_acceptance.sh` 是另一个显式授权的只读验收入口：仅运行已有
`real-perception --camera-single-start` Gateway 和 RPent 正式 CLI；另可显式选择
`--acceptance-scope tcp`，仅启动 `real-tcp-observe` 和 RPent 的 state 附着。
TCP 分支要求显式的双臂 IP assignment、左右 tool profile 和 envelope 配置路径，
不接受相机配置；相机分支不接受机械臂配置。两者均不接受任意命令、motion profile
或夹爪动作；TCP 分支只读 SDK/FK，不构造控制器，不发 HOLD/STOP。
模型密钥仅经 SSH stdin 进入上层子进程环境，
不放入 argv、源码或输出；新输出目录保存 Gateway、配置准备、Agent 日志和收尾结果。
相机 Gateway 的子进程环境移除宿主继承的 `OPENAI_API_KEY`，包括 `--check-only`
路径；上层 Agent 使用本次 stdin 提供的密钥，不改变宿主环境。
已有 Grounding DINO 的 BERT 缓存可通过 `--hf-home PATH` 显式绑定；缓存根包含
`hub/`。子进程仍强制离线，不下载权重，也不改共享缓存中的模型文件。
相机专用验收可复用已有 `dexteleop310` 的 SAM2/Grounding DINO CUDA 栈；不要仅因
GraspGenX 环境能加载权重，就假定该环境也有 Grounding DINO 的可用 CUDA 扩展。
使用保存图片完成文本定位后再安排现场开流，不将模型加载通过当作推理通过。
`--camera-scope all` 默认要求三视角齐全。操作者明确选择 `--camera-scope main` 时，
验收在新的运行目录生成仅含原 `front` 项的配置副本，不启动腕部相机；不改原配置、
主相机流或标定，新鲜度准入保持原逻辑。报告记录所选范围，不能将主相机验收解释为
三相机验收。缺少必需相机时仍失败，不自动切换范围或降成 RGB-only。
Agent 检查实际公开的全部 RGB 角色，并绑定所选图像 Artifact 做文本定位；主视角深度
不用于其他视角像素的反投影。随后只讨论“右臂上移 3 cm 再返回”的状态来源和执行
前提，缺少 TCP 状态 Skill 时必须明确报告，不猜数值、不调用格式适配器或运动工具。
该相机专用 profile 不构造机械臂控制 owner，不能证明真机运动链已接通。
可选 `--camera-sdk-debug-log` 在新运行目录保存 `librealsense.debug.log`，并在
Gateway 日志记录 SDK 包路径/版本、实际相机配置、启动线程和阶段。它只透传给
正式 `real-perception` 入口，不改变流、重试、reset 或新鲜度准入。启用原生日志
会提前导入 SDK，可能影响初始化时序；单次成功不证明根因或长期修复。
固定 300 秒预算，TERM 后预留 90 秒收尾；不自动重试或 reset。`--check-only` 只检查
路径和两个正式入口的参数，不开相机或请求模型。只读报告通过已有
`finish(status="completed")` 收尾，不声明物理任务成功。验收入口读取正式 CLI 的
transcript，保留 `agent_finish` 和 `environment_success`；即使 Agent 退出码为零，
stuck/failure 或缺少 completed 收尾仍算运行失败。报告完成后仍须复核实际工具
结果、图像来源和清理证据，不能当成定位成功或物理抓取成功。

只读抓姿场景实验另有两个限定入口，不能用普通 Python wrapper 启动真实 Gateway 来替代：

- `remote_rm65_grasp_snapshot.sh`：明确设备只读授权后，提供 `--repo`、`--python`、
  `--camera-config`、`--arm-ip-assignment` 和新的 `--output` 目录，采集双臂 qpos 与可用相机 RGB-D。
  先检查既有连接/相机占用，不发送运动、HOLD、STOP、夹爪命令或相机 reset；固定 60 秒上限。
  `--dry-run` 仅显示参数，不连接设备。相机缺失和稳定性验证失败保留原因，不冒充完整采集成功。
  `--camera-startup-check [streams|sensors]` 复用同一探针，仅诊断相机，不导入或连接机械臂 SDK；
  streams 保留完整 RGB-D、新鲜帧及清理门禁，并保存同帧 native depth；sensors 仅枚举目标设备的
  当前选项和流能力，不开启流或设置参数。两者均写入新证据目录，与 `--check-only`、`--interactive`
  互斥；`observed` 只表示完成枚举，不能解释为相机采集或物理任务成功。
  相机复用 rule-base / real-teleop 的 `RealSenseRGBSource`：按配置顺序错峰启动、持续采集完整
  RGB-D、单次启动、启动超时终止批次、按实际启动顺序反向关闭；没有独立的原生 pipeline
  预热/轮询路径。探针始终禁用三种 hardware-reset 配置和自动曝光参数写入，保留分辨率、帧率、
  对齐与标定角色。连续新鲜帧检查通过后才连接双臂只读采样，RGB/Depth 使用同一个 `VideoFrame`
  的原始时间戳，不以读取缓存的时间替换。`capture_stages.jsonl` 实时保存阶段，`snapshot.json`
  及帧数据先于设备关闭保存；`cleanup_complete=false` 不能解释为设备已释放。
  用户态超时不能解除内核 `D` 状态；发现原生启动阻塞后不继续开启其他相机或连接机械臂。
  `--check-only` 仅校验源码/文件/依赖，不打开设备；可额外传相机 registry、CAD URDF 和左右工具
  profile 检查交互采集所需的上下文。操作者在设备主机直接运行同一探针的 `--interactive` 模式，
  启动时复用相同 owner 连接设备一次，回车保存新状态，退出时统一关闭；启动、单次采样和关闭分别
  有 60 s 看门狗，不限制人工等待时间。会话级清理证据与逐状态快照分开保存；SSH wrapper
  不提供持续交互或任意命令入口。详见 POINT_GRASP 的补充样例采集说明。
- `remote_grasp_snapshot_inference.sh`：`--python` 后用 `--` 传固定探针的 `--mode segment|infer`、
  `--repo`、`--input`、`--output`、`--bundle`，以及分割用 `--pixel u v` 或推理用 `--source`。
  只处理保存的数组、写入新目录；离线加载已有模型，固定 120 秒上限，不连接相机或机械臂。

两个入口均须显式 `-H`，不接受任意远端程序或 shell 命令，不覆盖生产源码。
抓姿实验与离线分臂回放见 [POINT_GRASP](../docs/agent_gateway/POINT_GRASP.md)。

### 3.1 远端受保护 Harness

Harness preflight 和 portable exchange 不属于 Python 入口，不能通过
`remote_python.sh` 绕过解释器限制。使用专用入口，并显式指定 SSH target 与远端
host-private wrapper：

```bash
./remote_harness.sh \
  -H user@192.168.70.116 \
  --harness /home/user/zc/HarnessConfig/bin/harness \
  runner preflight agentic-source-runner \
  --profile agenticrobot-integration \
  --expected-control-commit FULL_HARNESS_COMMIT \
  --json

./remote_harness.sh \
  -H user@192.168.70.116 \
  --harness /home/user/zc/HarnessConfig/bin/harness \
  transfer exchange-run /absolute/path/to/exchange \
  --runner-id agentic-source-runner \
  --slot-id source-slot-01 \
  --sender-id local-controller \
  --recipient-id agentic-source-node \
  --key-id KEY_ID \
  --key-file /private/path/to/transfer.key \
  --json
```

该 wrapper 只接受 `runner preflight` 和 `transfer exchange-run`，并要求远端入口为
普通、可执行、非软链接的 `*/HarnessConfig/bin/harness`。它不开放通用 shell，也不
替代 exchange 的 HMAC、exact protected commit、Queue Profile、R1/设备或运动授权。
等待远端 evidence 时本地 Task writer 应释放；wrapper 本身不 claim writer。

这个入口只解决“如何受控调用远端 protected Harness”，不负责生成 transfer 身份或
密钥。本地 controller 在提交 exchange 前仍必须从仓库外的 host-private Harness
配置解析 `HARNESS_ACTOR`、`HARNESS_TRANSFER_{SENDER_ID,RECIPIENT_ID,KEY_ID,KEY_FILE}`、
`HARNESS_INTEGRATION_EXCHANGE_ROOT` 和 `HARNESS_TRANSFER_RECEIVER_BASES_FILE`；可选的
回收目录使用 `HARNESS_TRANSFER_RESULT_ROOT`。receiver-bases 文件必须保存各 repository
经远端 preflight 核验的精确 commit，不能保存移动 branch 名。任一字段缺失时应在创建
request/exchange 前停止，由 operator provision；不得把站点地址、身份或 key 写入源码。

受保护 Harness 的版本部署使用独立的 `remote_harness_deploy.sh`，不扩大上述运行入口：

```bash
./remote_harness_deploy.sh \
  -H user@192.168.70.116 \
  --root /home/user/zc \
  --expected-commit FULL_HARNESS_COMMIT \
  plan

./remote_harness_deploy.sh \
  -H user@192.168.70.116 \
  --root /home/user/zc \
  --expected-commit FULL_HARNESS_COMMIT \
  --confirm-plan-sha256 FULL_PLAN_SHA256 \
  install
```

该入口必须由用户或 deployment owner 对精确 commit 明确授权。它只会从固定的
`HarnessCodingTool` checkout 获取 `origin/agentic-robot`，验证 canonical identity 和
精确 commit，然后调用 host-private Harness 完成 plan/install。远端目录布局固定由
`--root` 推导；不接受任意仓库、branch、命令或安装路径。`install` 会重新 fetch、重新
验证 commit 和 plan SHA，并核对 `HarnessControl/current` 后输出 control status。
部署完成后仍应通过 `remote_harness.sh runner preflight --expected-control-commit ...`
独立确认 runner 实际加载了新版本。

### 4. 远程访问真机设备

设备访问原则：

```text
USB / serial / camera / robot network
  必须由远程电脑上的进程访问。

Codex / wrapper
  只负责通过 SSH 发起受控命令、读取状态、收集日志、拉回结果。
```

例如：

```bash
./remote_lsusb.sh -H dex-remote
./remote_camera_snapshot.sh -H dex-remote --device /dev/video0 --out ./debug/latest.jpg
./remote_arm_ping.sh -H dex-remote --ip 192.168.1.18
./remote_arm_dry_run.sh -H dex-remote --project '~/DexTeleop' --script tools/arm_dry_run.py -- --config configs/arm.yaml
```

## 安全限制

### 总原则

所有 wrapper 都应遵守：

1. 默认只读优先。
2. 默认诊断优先。
3. 默认 dry-run 优先。
4. 真机运动必须显式确认。
5. 高风险动作必须拆分为“检查”和“执行”两步。
6. 不让 Codex 直接拼接危险运动命令。
7. 不在脚本中保存明文密码、token、私钥。
8. 不同步 `.codex/`、`.ssh/`、`.gnupg/`、浏览器 profile 和密钥文件。
9. 不把 `sudo` 包在无确认 wrapper 里。
10. 所有远程命令都应打印目标 host、远程目录、conda/env、即将执行的命令。

### 动作安全等级

建议每个 wrapper 标注一个安全等级：

```text
L0: 本地只读
  例如打印帮助、检查本地配置。

L1: 远程只读
  例如 hostname、lsusb、camera list、ip addr、tail log。

L2: 远程非运动写入
  例如 rsync 代码、创建日志目录、保存截图、启动 preview server。

L3: 远程进程控制
  例如启动/停止 Python 任务、kill 白名单进程、tmux 管理。

L4: 设备状态改变
  例如 USB reset、udev reload、机械臂 enable/disable。

L5: 真实运动或系统级动作
  例如机械臂 home、初始化、轨迹执行、reboot、shutdown。
```

默认策略：

```text
L0-L1: 可以自动运行。
L2: 可以运行，但应打印影响范围。
L3: 只允许白名单进程，禁止模糊 kill。
L4: 需要显式参数，例如 --i-understand 或 --confirm DEVICE_ID。
L5: 需要人工确认，并且必须先通过 safety check。
```

### 真机运动保护

所有机械臂、夹爪、移动平台、力控、关节运动类 wrapper，都应先执行统一检查：

```text
1. 确认连接的是预期远程主机。
2. 确认机械臂 IP 可达。
3. 确认急停状态正常。
4. 确认当前模式允许外部控制。
5. 确认速度、加速度、力阈值在安全范围。
6. 确认工作空间限制已加载。
7. 确认不是 replay 全速运行。
8. 确认用户显式传入 --confirm-motion。
```

危险动作不要只做成：

```bash
./remote_arm_home.sh
```

更推荐：

```bash
./remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18
./remote_arm_home.sh -H dex-remote --arm-ip 192.168.1.18 --speed-scale 0.1 --confirm-motion
```

### SSH 命令约束

wrapper 可以支持两类模式：

```text
受控模式
  只允许 wrapper 内部定义好的子命令和参数。
  推荐给 Codex 使用。

专家模式
  允许传入任意远程命令。
  只给人工使用，默认不推荐 Codex 自动调用。
```

例如 `remote_shell.sh` 可以存在，但应明确标注为专家模式；Codex 常规工作应优先使用 `remote_python.sh`、`remote_camera_snapshot.sh`、`remote_arm_ping.sh` 等受控 wrapper。

## 推荐目录结构

可以按包分目录，也可以先平铺脚本。长期推荐：

```text
Connection/
  README.md

  setup_ssh_passwordless.sh
  setup_ssh_passwordless.README.md
  realtime_sync_ssh.sh
  realtime_sync_ssh.README.md
  lan_share_mount.sh
  lan_share_mount.README.md

  remote_core/
    remote_whoami.sh
    remote_run.sh
    remote_python.sh
    remote_module.sh
    remote_env.sh

  remote_sync/
    remote_push.sh
    remote_pull.sh
    remote_rsync_project.sh
    remote_diff.sh
    remote_clean_cache.sh

  remote_hw/
    remote_lsusb.sh
    remote_lspci.sh
    remote_dev_nodes.sh
    remote_groups.sh
    remote_dmesg_usb.sh

  remote_camera/
    remote_camera_list.sh
    remote_camera_info.sh
    remote_camera_snapshot.sh
    remote_camera_record_test.sh
    remote_camera_preview_server.sh

  remote_robot_net/
    remote_arm_ping.sh
    remote_arm_port_check.sh
    remote_arm_state.sh
    remote_arm_dry_run.sh
    remote_arm_init_safe.sh
    remote_arm_estop.sh

  remote_safety/
    remote_safety_check.sh
    remote_motion_guard.sh
    remote_confirmed_action.sh
    remote_zero_velocity.sh

  remote_logs/
    remote_tail_log.sh
    remote_journal.sh
    remote_dmesg.sh
    remote_collect_debug_bundle.sh

  remote_proc/
    remote_ps.sh
    remote_kill_safe.sh
    remote_tmux_start.sh
    remote_tmux_log.sh
    remote_service_start.sh
    remote_service_stop.sh
```

如果脚本数量还少，可以先保持平铺，但命名要带包前缀，例如 `remote_camera_snapshot.sh`。

## Wrapper 包详细列表

### remote_core

远程基础执行层。其他包应尽量复用这里的逻辑。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_whoami.sh` | L1 | 打印远程 host、user、pwd、hostname、时间 |
| `remote_run.sh` | L2/L3 | 在远程项目目录执行受控命令 |
| `remote_shell.sh` | L3+ | 专家模式，执行任意远程 shell 命令 |
| `remote_python.sh` | L3 | 激活远程环境并运行 `.py` 文件 |
| `remote_module.sh` | L3 | 激活远程环境并运行 `python -m module` |
| `remote_pytest.sh` | L3 | 远程运行测试 |
| `remote_env.sh` | L1 | 打印 Python、conda、CUDA、PATH、项目路径 |

建议优先实现 `remote_whoami.sh`、`remote_python.sh`、`remote_pytest.sh`。

### remote_sync

本地和远程之间的代码、数据、日志同步。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_push.sh` | L2 | 本地项目同步到远程 |
| `remote_pull.sh` | L2 | 从远程拉回结果、日志、图片、视频 |
| `remote_rsync_project.sh` | L2 | 项目级 rsync，内置排除规则 |
| `remote_diff.sh` | L1 | 对比本地/远程文件差异 |
| `remote_clean_cache.sh` | L2 | 清理远程 `__pycache__`、`.pytest_cache` 等 |
| `remote_fetch_debug_bundle.sh` | L2 | 拉取远程调试包 |

默认排除：

```text
.git/
.codex/
.ssh/
.gnupg/
__pycache__/
.pytest_cache/
.mypy_cache/
.venv/
node_modules/
*.pyc
*.log
```

是否排除数据集、录制结果、模型权重应由项目配置决定。

### remote_python_runtime

Python 运行和环境管理。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_script.sh` | L3 | 远程运行指定脚本 |
| `remote_module.sh` | L3 | 远程运行 Python module |
| `remote_replay.sh` | L3/L5 | 远程运行 replay；真机 replay 属于 L5 |
| `remote_dry_run.sh` | L3 | 远程 dry-run，不发真机运动 |
| `remote_notebook.sh` | L3 | 启动 Jupyter 并做端口转发 |
| `remote_pip_check.sh` | L1 | 检查 pip 包版本 |
| `remote_conda_list.sh` | L1 | 导出远程 conda 环境 |
| `remote_python_import_check.sh` | L1 | 检查关键 Python 模块能否 import |

建议所有 Python wrapper 都支持：

```text
-H / --host
--project
--env
--python
--dry-run
--timeout
--log-dir
--
```

`--` 后面的参数原样传给 Python 程序。

### remote_proc

远程进程管理。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_ps.sh` | L1 | 查看远程相关进程 |
| `remote_kill_safe.sh` | L3 | kill 白名单进程 |
| `remote_tmux_start.sh` | L3 | 在 tmux 中启动长期任务 |
| `remote_tmux_log.sh` | L1 | 查看 tmux 输出 |
| `remote_tmux_stop.sh` | L3 | 停止指定 tmux session |
| `remote_service_start.sh` | L3 | 启动 systemd user service |
| `remote_service_stop.sh` | L3 | 停止受控 service |
| `remote_watch.sh` | L1 | 周期监控远程命令 |

`remote_kill_safe.sh` 不应接受任意 PID 默认 kill；应使用白名单进程名、session 名或 wrapper 生成的 pidfile。

### remote_logs

远程日志和诊断包。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_tail_log.sh` | L1 | tail 指定日志 |
| `remote_journal.sh` | L1 | 查看 systemd/journalctl |
| `remote_dmesg.sh` | L1 | 查看内核日志 |
| `remote_dmesg_usb.sh` | L1 | 过滤 USB/相机/串口相关内核日志 |
| `remote_last_error.sh` | L1 | 提取最近错误 |
| `remote_collect_debug_bundle.sh` | L2 | 打包日志、设备状态、版本信息 |

`remote_dmesg_usb.sh` 在内核日志读取失败（例如权限不足）时保留诊断并返回退出码 4；读取成功但没有匹配记录时正常返回。

调试包建议包含：

```text
hostname
date
uname -a
ip addr
ip route
lsusb
lspci
groups
python --version
conda env list
pip freeze or conda list
/dev/video*
/dev/ttyUSB*
dmesg tail
project git status
selected app logs
```

### remote_hw

远程硬件发现。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_lsusb.sh` | L1 | 列 USB 设备 |
| `remote_lspci.sh` | L1 | 列 PCIe/GPU/采集卡 |
| `remote_dmesg_usb.sh` | L1 | 查看 USB 插拔和错误 |
| `remote_groups.sh` | L1 | 检查远程用户组 |
| `remote_dev_nodes.sh` | L1 | 列 `/dev/video*`、`/dev/ttyUSB*`、`/dev/input*` |
| `remote_udev_check.sh` | L1 | 检查 udev rules 是否生效 |
| `remote_gpu.sh` | L1 | 运行 `nvidia-smi` |
| `remote_temp.sh` | L1 | 查看温度、功耗、风扇等 |

常见权限检查：

```text
camera: video group
serial: dialout group
USB HID/input: input or plugdev group, or dedicated udev rule
robot network: route and firewall
GPU: video/render group, NVIDIA driver, CUDA runtime
```

### remote_camera

远程摄像头、深度相机和视频流。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_camera_list.sh` | L1 | 列摄像头 |
| `remote_camera_info.sh` | L1 | 查看格式、分辨率、fps |
| `remote_camera_snapshot.sh` | L2 | 抓图并拉回本地 |
| `remote_camera_record_test.sh` | L2 | 录制短视频并拉回本地 |
| `remote_camera_preview_server.sh` | L3 | 启动 MJPEG/RTSP/WebRTC 预览 |
| `remote_camera_stop_preview.sh` | L3 | 停止预览服务 |
| `remote_realsense_check.sh` | L1 | RealSense 检测 |
| `remote_opencv_camera_check.sh` | L1 | OpenCV 打开测试 |

推荐 preview 方式：

```text
1. 远程启动本地监听服务，例如 127.0.0.1:8080。
2. 本地用 SSH 端口转发访问。
3. 不在局域网裸露无认证视频端口，除非明确需要。
```

### remote_usb_serial

USB、串口、HID 和低层设备控制。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_serial_list.sh` | L1 | 列 `/dev/ttyUSB*`、`/dev/ttyACM*` |
| `remote_serial_probe.sh` | L1 | 测试串口打开 |
| `remote_serial_monitor.sh` | L1/L2 | 读取串口输出，可保存日志 |
| `remote_usb_tree.sh` | L1 | 查看 USB 拓扑 |
| `remote_usb_reset.sh` | L4 | 受控重置 USB 设备 |
| `remote_udev_reload.sh` | L4 | 重载 udev rules |

`remote_usb_reset.sh` 必须要求明确设备 ID，例如：

```bash
./remote_usb_reset.sh -H dex-remote --vendor 1234 --product abcd --confirm-reset
```

不要做成“重置所有 USB 设备”。

### remote_robot_net

网口机械臂、控制器、夹爪、力控设备。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_arm_ping.sh` | L1 | ping 机械臂 IP |
| `remote_arm_port_check.sh` | L1 | 检查控制端口可达 |
| `remote_arm_state.sh` | L1 | 读取机械臂状态 |
| `remote_arm_enable.sh` | L4 | 使能 |
| `remote_arm_disable.sh` | L4 | 下使能或停止 |
| `remote_arm_home.sh` | L5 | 回零/初始化 |
| `remote_arm_dry_run.sh` | L3 | 只打印指令，不发运动 |
| `remote_arm_init_safe.sh` | L5 | 低速、限幅初始化 |
| `remote_arm_estop.sh` | L4/L5 | 紧急停止 |
| `remote_arm_replay.sh` | L5 | 执行 replay 或轨迹 |

机械臂 wrapper 必须支持：

```text
--arm-ip
--speed-scale
--workspace
--dry-run
--confirm-motion
```

默认 `speed-scale` 应很低，例如 `0.1` 或更低。

### remote_safety

所有真机动作的安全门。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_safety_check.sh` | L1/L2 | 检查急停、限位、网络、权限、设备状态 |
| `remote_motion_guard.sh` | L4/L5 | 运动前统一守卫 |
| `remote_confirmed_action.sh` | L4/L5 | 显式确认动作模板 |
| `remote_zero_velocity.sh` | L4 | 发送零速度或停止命令 |
| `remote_power_state.sh` | L1 | 检查电源/使能状态 |
| `remote_workspace_check.sh` | L1 | 检查工作空间配置 |

边界：

```text
remote_safety_check.sh 只输出当次只读预检证据，不生成动作许可。
高风险 wrapper 必须自己限制动作类型、目标和参数，并重新检查当前安全状态。
高风险 wrapper 仍然要求与精确动作绑定的显式确认；普通预检不能替代它。
```

### remote_vive

VIVE、SteamVR、动作捕捉和姿态数据流。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_vive_check.sh` | L1 | 检测 VIVE/SteamVR/驱动状态 |
| `remote_vive_sender_start.sh` | L3 | 启动 VIVE sender |
| `remote_vive_sender_stop.sh` | L3 | 停止 VIVE sender |
| `remote_vive_stream_probe.sh` | L1 | 检查 UDP/TCP 数据流 |
| `remote_vive_record_sample.sh` | L2 | 采样并保存一小段数据 |
| `remote_vive_log.sh` | L1 | 查看 sender 日志 |

VIVE wrapper 应清晰区分：

```text
只检测设备
启动数据发送
记录数据
把姿态数据接入机器人控制
```

最后一类如果会影响真机运动，应进入 `remote_safety` 和 `remote_robot_net` 的安全门。

### remote_record_replay

录制、回放、数据集管理。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_record_start.sh` | L3/L5 | 开始录制多路数据；若联动真机为 L5 |
| `remote_record_stop.sh` | L3 | 停止录制 |
| `remote_record_status.sh` | L1 | 查看录制状态 |
| `remote_replay_run.sh` | L3/L5 | 运行 replay |
| `remote_replay_validate.sh` | L1/L2 | 校验 replay 文件 |
| `remote_fetch_episode.sh` | L2 | 拉回远程 episode |
| `remote_dataset_index.sh` | L1 | 列远程数据集 |

replay 的默认行为必须是 dry-run；真实运动应要求：

```text
--confirm-motion
--speed-scale
```

### remote_network

局域网、端口、路由和隧道。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_ip.sh` | L1 | 显示远程网卡/IP/路由 |
| `remote_ping_device.sh` | L1 | ping 指定设备 |
| `remote_port_check.sh` | L1 | 检查端口占用和可达性 |
| `remote_port_forward.sh` | L3 | 建立 SSH 端口转发 |
| `remote_latency_test.sh` | L1 | 测局域网延迟 |
| `remote_bandwidth_test.sh` | L1/L2 | iperf 测试 |
| `remote_firewall_check.sh` | L1 | 检查防火墙状态 |

端口转发建议默认绑定本地 `127.0.0.1`：

```bash
ssh -L 127.0.0.1:8080:127.0.0.1:8080 dex-remote
```

避免默认暴露到 `0.0.0.0`。

### remote_vis

可视化和调试界面。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_tensorboard.sh` | L3 | 启动 TensorBoard 并转发端口 |
| `remote_mjpeg_preview.sh` | L3 | 摄像头网页预览 |
| `remote_plot_fetch.sh` | L2 | 远程生成图并拉回本地 |
| `remote_open3d_server.sh` | L3 | 点云/3D 可视化服务 |
| `remote_video_player_data_pull.sh` | L2 | 拉取多视角视频到本地 |

可视化 wrapper 应尽量让数据留在远程，只把需要查看的低风险结果拉回本地。

### remote_admin

远程机器状态和维护。

| wrapper | 安全等级 | 用途 |
| --- | --- | --- |
| `remote_disk.sh` | L1 | 磁盘空间 |
| `remote_mem.sh` | L1 | 内存 |
| `remote_gpu.sh` | L1 | GPU 状态 |
| `remote_temp.sh` | L1 | 温度/功耗 |
| `remote_time_sync.sh` | L1 | 检查时间同步 |
| `remote_reboot.sh` | L5 | 重启，必须人工确认 |
| `remote_shutdown.sh` | L5 | 关机，必须人工确认 |

`remote_reboot.sh` 和 `remote_shutdown.sh` 不建议给 Codex 自动调用。

## 通用脚本接口规范

所有 wrapper 尽量使用一致参数：

```text
-H, --host          SSH host、IP、user@host[:port] 或 alias；Codex 每次显式传入
-u, --user          SSH user，优先使用 ~/.ssh/config 时可省略
-p, --port          SSH port
--project           远程项目目录
--env               conda/env 名称
--python            Python 可执行文件，默认 python
--timeout           超时时间
--log-dir           远程日志目录
--local-out         本地输出目录
--dry-run           只打印动作，不执行危险动作
--yes               跳过低风险确认；不适用于 L4/L5
--confirm-reset     确认 USB reset
--confirm-motion    确认真机运动
--                 后续参数透传给远程程序
```

返回码建议：

```text
0: 成功
1: 参数错误
2: SSH 连接失败
3: 远程环境错误
4: 设备不存在或权限不足
5: 安全检查失败
6: 远程程序失败
7: 超时
8: 用户未确认高风险动作
```

输出约定：

```text
[local] wrapper name/version
[target] host/user/project/env
[check] preflight result
[run] remote command summary
[result] output path / log path / exit code
```

`[target]` 表示请求的 SSH 目标和配置参数，不单独证明已连接的远端身份。
连接失败时也可能已经打印该行；需要结合身份检查、实际输出和退出状态判断。

只需核对远端工作目录和 Python 时，可用 `remote_env.sh -H HOST --summary`
（可结合 `--project`、`--env`、`--python`）。它保留完整 wrapper/target header、
远端 hostname、实际工作目录、Python 可执行路径和版本；省略 shell/PATH 展示，
不查询全部 conda 环境、GPU 或 CUDA，并输出明确的检查范围说明。
激活环境、路径检查和 Python 版本查询失败仍返回失败；摘要不是 GPU/CUDA 或
整体环境就绪的证据。需要这些诊断时去掉 `--summary`，默认完整输出保持不变。
该选项不缓存身份、不改变安全检查或授权，也不创建另一份审计日志。

## SSH 实现建议

建议集中实现公共 SSH 参数：

```bash
ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=5 \
  -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 \
  "$HOST" \
  'remote command'
```

`BatchMode=yes` 可以避免脚本卡在密码输入；免密未配置时应直接失败并提示运行 `setup_ssh_passwordless.sh`。

如果远程命令需要复杂参数，优先使用：

```text
1. wrapper 明确定义参数。
2. 使用 `--` 后透传给 Python 程序。
3. 避免手写多层 shell quote。
```

## Conda/Python 环境建议

远程 Python wrapper 应兼容：

```text
conda activate ENV
micromamba activate ENV
source venv/bin/activate
直接指定 /path/to/python
```

推荐优先支持：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310
python -u script.py
```

`python -u` 可以保证日志实时输出。

## 设备权限建议

远程用户通常需要加入相关用户组：

```bash
sudo usermod -aG video,dialout,plugdev,input,render user
```

修改用户组后需要重新登录远程 session。

常见设备：

```text
/dev/video*: 摄像头，通常需要 video 组
/dev/ttyUSB*: USB 串口，通常需要 dialout 组
/dev/ttyACM*: USB ACM 串口，通常需要 dialout 组
/dev/input*: HID/input，可能需要 input 组或 udev rule
RealSense: librealsense udev rules
GPU: NVIDIA driver/CUDA，可能需要 video/render 组
```

设备权限问题应优先通过 udev rules 和用户组解决，不建议在 wrapper 中默认使用 `sudo`。

## 最小优先实现集合

建议第一批先实现这些：

```text
remote_core/remote_whoami.sh
remote_core/remote_env.sh
remote_core/remote_python.sh
remote_core/remote_pytest.sh

remote_sync/remote_rsync_project.sh
remote_sync/remote_pull.sh

remote_hw/remote_lsusb.sh
remote_hw/remote_dev_nodes.sh
remote_hw/remote_groups.sh
remote_hw/remote_dmesg_usb.sh

remote_camera/remote_camera_list.sh
remote_camera/remote_camera_snapshot.sh
remote_camera/remote_opencv_camera_check.sh

remote_robot_net/remote_arm_ping.sh
remote_robot_net/remote_arm_port_check.sh
remote_robot_net/remote_arm_state.sh
remote_robot_net/remote_arm_dry_run.sh

remote_safety/remote_safety_check.sh
remote_safety/remote_zero_velocity.sh

remote_logs/remote_collect_debug_bundle.sh
remote_logs/remote_tail_log.sh

remote_proc/remote_ps.sh
remote_proc/remote_tmux_start.sh
remote_proc/remote_tmux_log.sh
remote_proc/remote_kill_safe.sh
```

有了这些，Codex 就能完成多数安全任务：

```text
同步代码
检查远程环境
远程跑 Python/pytest
检查 USB/摄像头/串口/机械臂网络
抓图或录制短视频
查看日志
启动和停止受控长期任务
生成调试包
做 dry-run
```

真实运动类 wrapper 应第二阶段再实现，并且必须接入 `remote_safety`。

## 示例：一次典型真机调试

```bash
# 1. 确认 SSH 和远程身份
./remote_core/remote_whoami.sh -H dex-remote

# 2. 同步代码
./remote_sync/remote_rsync_project.sh -H dex-remote --local ~/DexTeleop --remote '~/DexTeleop'

# 3. 检查远程 Python 环境
./remote_core/remote_env.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310

# 4. 检查硬件
./remote_hw/remote_lsusb.sh -H dex-remote
./remote_hw/remote_dev_nodes.sh -H dex-remote
./remote_robot_net/remote_arm_ping.sh -H dex-remote --arm-ip 192.168.1.18

# 5. 摄像头抓图
./remote_camera/remote_camera_snapshot.sh -H dex-remote --device /dev/video0 --local-out ./debug/camera

# 6. 远程 dry-run
./remote_robot_net/remote_arm_dry_run.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310

# 7. 真机运动前安全检查
./remote_safety/remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18

# 8. 低速受控初始化
./remote_robot_net/remote_arm_init_safe.sh \
  -H dex-remote \
  --arm-ip 192.168.1.18 \
  --speed-scale 0.1 \
  --confirm-motion
```

## 示例：远程 Python wrapper 行为

理想调用：

```bash
./remote_core/remote_python.sh \
  -H dex-remote \
  --project '~/DexTeleop' \
  --env dexteleop310 \
  -- \
  teleop_runtime/replay/arm_init/dry_run.py --config configs/arm.yaml
```

wrapper 内部大致执行：

```bash
cd ~/DexTeleop
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310
python -u teleop_runtime/replay/arm_init/dry_run.py --config configs/arm.yaml
```

注意：这只是说明行为，不建议每次让 Codex 直接拼这段命令。应由 wrapper 统一处理 quote、日志、返回码和错误提示。

## Codex 使用建议

给 Codex 的默认规则应是：

```text
1. 优先调用 Connection wrapper，不直接手写复杂 ssh 命令。
2. L0-L1 可以直接运行。
3. L2-L3 运行前说明影响范围。
4. L4-L5 必须请求人工确认。
5. 真实运动必须使用专用 wrapper，并在同一动作边界内重新检查安全状态；
   remote_safety_check 的只读结果本身不构成许可。
6. 不自动执行 reboot/shutdown/udev reload/USB reset/机械臂运动。
7. 不同步密钥、token、.codex、.ssh。
8. 远程失败时先收集 debug bundle，再修改代码。
```

可以在项目 `AGENTS.md` 或相关说明中要求 Codex 遵守这些 wrapper 边界。

## 与现有脚本的关系

当前已有脚本：

```text
setup_ssh_passwordless.sh
  建立 SSH 免密。

realtime_sync_ssh.sh
  本地到远程的实时同步。

lan_share_mount.sh
  局域网共享和挂载。
```

后续 wrapper 应建立在这些基础能力上：

```text
SSH 免密 -> 远程命令可运行
同步/挂载 -> 远程能看到代码或数据
remote_core -> 统一远程 Python 和基础执行
remote_hw/camera/robot -> 统一设备访问
remote_safety -> 统一真机安全门
```

## 实施顺序建议

第一阶段：连接和诊断

```text
remote_whoami
remote_env
remote_lsusb
remote_dev_nodes
remote_groups
remote_dmesg_usb
```

第二阶段：代码和 Python

```text
remote_rsync_project
remote_python
remote_pytest
remote_collect_debug_bundle
```

第三阶段：摄像头和日志

```text
remote_camera_list
remote_camera_snapshot
remote_camera_record_test
remote_tail_log
remote_tmux_start/log/stop
```

第四阶段：机械臂只读和 dry-run

```text
remote_arm_ping
remote_arm_port_check
remote_arm_state
remote_arm_dry_run
remote_safety_check
```

第五阶段：真机低速动作

```text
remote_zero_velocity
remote_arm_disable
remote_arm_init_safe
remote_arm_home
remote_arm_replay
```

第五阶段必须保守推进，每个动作先人工 review，再逐步开放给 Codex 请求执行。

## 检查清单

新 wrapper 合并前检查：

```text
[ ] 有清晰用途和安全等级。
[ ] 支持 -H/--host。
[ ] 打印目标 host 和远程项目目录。
[ ] 失败时返回非零状态。
[ ] 不静默吞掉远程错误。
[ ] 不默认 sudo。
[ ] 不默认执行真实运动。
[ ] 不默认删除远程数据。
[ ] 不同步敏感目录。
[ ] 参数 quote 可处理空格和 ~。
[ ] 有 --dry-run 或等价检查模式。
[ ] L4/L5 有显式确认参数。
[ ] 真机运动由专用 wrapper 限制精确动作，并在执行边界重新检查安全状态。
```

补充开发要求：

```text
[ ] 使用 remote_common.sh 中的共享 SSH 参数解析，除非有明确理由。
[ ] 输出包含 [local]、[target]、[run] 或 [check]、[result] 等可审计行。
[ ] 对 ambiguous input、未知操作等级、缺失远程路径和控制风险通用执行采用 fail closed。
[ ] 不隐藏关键命令输出；长输出应给出日志路径、有限输出或 debug bundle。
[ ] 修改 wrapper 行为或安全策略前，先说明安全影响并保持窄范围改动。
```

`AGENTS.md` 只保留 Codex 每次需要执行的硬规则。背景说明、VSCode 使用提示、wrapper API 规范和示例流程以 README 为准。

## 结论

这套 wrapper 的核心价值不是“让 Codex 获得无限远程权限”，而是把远程电脑上的运行环境和真机设备整理成一组**可审计、可复用、可限权、可逐步开放**的本地命令。

最终效果应是：

```text
Codex 留在本地。
Python 跑在远程。
真机设备接在远程。
每个动作都有清晰 wrapper。
危险动作有安全门和人工确认。
```

## 当前平铺实现

本目录当前先采用平铺脚本，而不是完全照搬上面的长期推荐目录结构。公共 SSH、rsync、quote 和目标打印逻辑放在 `remote_common.sh`；实际入口都放在 `codex` 目录根部，便于 Codex 和人工直接调用。

已实现的第一批 wrapper：

```text
remote_whoami.sh
remote_env.sh
remote_harness.sh
remote_harness_deploy.sh
remote_python.sh
remote_module.sh
remote_pytest.sh
remote_rsync_project.sh
remote_pull.sh
remote_lsusb.sh
remote_dev_nodes.sh
remote_groups.sh
remote_dmesg_usb.sh
remote_camera_list.sh
remote_camera_snapshot.sh
remote_opencv_camera_check.sh
remote_arm_ping.sh
remote_arm_port_check.sh
remote_arm_dry_run.sh
remote_safety_check.sh
remote_collect_debug_bundle.sh
remote_tail_log.sh
remote_ps.sh
remote_view.sh
remote_sync_check.sh
remote_dataset_camera_semantics.sh
remote_dataset_camera_quarantine.sh
remote_dataset_camera_promotion.sh
remote_dataset_camera_training_validity.sh
remote_dataset_tactile_storage.sh
```

只读文本查看可加 `remote_view.sh --compact`：保留完整 `[target]`、解析后的路径、
类型、文件字节数与权限、正文及范围说明，省略重复请求路径和详细元数据；默认仍展示
详细元数据。目录列表和非普通文件信息不精简，敏感路径及非文本文件的拒绝规则不变。

文本先按 `--head` / `--tail` 选择行，再用 `--max-bytes` 限制正文展示字节数。
末尾 `selected_bytes_omitted` 只计算所选行中因字节上限未展示的部分；为 0 不表示
已读完整文件。目录末尾报告列表行数上限（包含 `ls` 的汇总行），不承诺完整列举。
正常达到展示上限不再造成 broken pipe 失败，真实读取或展示错误仍返回非零。
字节限制可能切断多字节字符；读取增长中的日志也不是一致性快照。

示例：

```bash
./remote_whoami.sh -H dex-remote
./remote_whoami.sh -H user@192.168.20.171
./remote_rsync_project.sh -H dex-remote --local ~/Release --remote '~/Release' --dry-run
./remote_python.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 -- script.py --arg value
./remote_camera_snapshot.sh -H dex-remote --device /dev/video0 --local-out ./debug/camera
./remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18 --require-device /dev/video0
./remote_view.sh -H dex-remote --path '~/DexTeleop/logs/latest.log' --tail 200
./remote_sync_check.sh -H dex-remote --local ~/DexTeleop --remote '~/DexTeleop' --path teleop_runtime/foo.py
./remote_dataset_camera_semantics.sh -H dex-remote \
  --project '~/DexTeleop' --env dexteleop310 \
  --source-root /data/source_dataset \
  --output-root /data/camera_semantic_prepared_views/source_dataset \
  --registry /data/frozen_camera_role_registry.json --dry-run
./remote_dataset_camera_quarantine.sh -H dex-remote \
  --project '~/DexTeleop' --env dexteleop310 \
  --view-root /data/camera_semantic_prepared_views/bad_trial \
  --audit-root /data/camera_semantic_prepared_views/_invalid_audit
./remote_dataset_camera_promotion.sh -H dex-remote \
  --project '~/DexTeleop' --env dexteleop310 --promote \
  --dataset-root /data/canonical_swapped \
  --corrected-root /data/corrected_full \
  --retirement-root /data/retired_swapped \
  --registry /data/frozen_camera_role_registry.json --dry-run
./remote_dataset_camera_training_validity.sh -H dex-remote \
  --project '~/DexTeleop' --env dexteleop310 \
  --dataset-root /data/source_dataset \
  --prepared-view-root /data/camera_semantic_prepared_views/source_dataset \
  --dry-run
./remote_dataset_tactile_storage.sh -H dex-remote \
  --project '~/DexTeleop' --env dexteleop310 \
  --detach-whole-prefix --dataset-root /data/source_dataset \
  --cold-root /independent/cold/root --dry-run
```

相机 promotion 默认只接受原生 v2.3。历史旧格式只能在 dry-run 和 apply 两次命令中都显式
加入 `--allow-legacy-format-preserving`；该选项只允许经过逐列、逐清单和完整视频验证的
格式保持型左右腕纠正，不执行 schema 迁移，也不会把旧数据声明为 v2.3。

### 实际同步和执行模型

常见真机开发时，本地工作目录可能已通过确认过的
`connection/realtime_sync_ssh.sh` 实例，以 no-delete 模式实时同步到远程
指定工作目录。同步脚本必须加载项目根目录的
`.realtime_sync_ssh.exclude`；设备发现/分配、标定结果、相机注册表、工作站几何、
数据集和本机路径配置均由远端机器持有，不得由本地同路径文件实时覆盖。
修改排除清单后需要重启既有实时同步进程，lsyncd 才会重新生成配置并生效。
此时应按下面的边界工作：

```text
代码修改类任务
  只在本地工作目录修改。
  不通过 SSH 直接改远程源代码。
  不为常规编辑再运行 remote_rsync_project.sh，除非用户明确要求一次性同步。

运行/测试类任务
  在远端工作目录执行，因为远端才有硬件、驱动、相机、GPU、机械臂网络和 Python 环境。

远端运行结果
  不会自动同步回本地。
  需要用 remote_view.sh、remote_tail_log.sh、remote_collect_debug_bundle.sh 或 remote_pull.sh 显式查看或拉回。

一致性检查
  当远端运行结果疑似使用旧代码，或运行前需要确认关键改动已经同步时，
  用 remote_sync_check.sh 对显式文件/目录做只读 hash/size 对比。
```

示例：

```bash
./remote_pytest.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 -- tests/test_x.py -q
./remote_view.sh -H dex-remote --path '~/DexTeleop/logs/latest.log' --tail 200
./remote_pull.sh -H dex-remote --remote '~/DexTeleop/debug/latest.jpg' --local ./debug
./remote_sync_check.sh -H dex-remote --local ~/DexTeleop --remote '~/DexTeleop' --path tests/test_x.py
```

### 当前实现中的硬安全门

安全等级不是只写在帮助文本里的标签。当前平铺实现对 Codex 误调用做了几层默认保护：

```text
remote_python.sh / remote_module.sh / remote_pytest.sh
  会扫描脚本名、模块名和参数。
  如果出现 replay、trajectory、motion、arm、robot、gripper、servo、confirm-motion 等控制风险词，默认拒绝运行。
  只打印命令的 wrapper --dry-run 不受影响。
  --python 只能指向 basename 类似 python、python3、pypy 的解释器，避免 wrapper 被误用成任意命令启动器。
  关键词检查只能发现部分明显风险；调用方仍必须根据代码真实副作用分类，未知情况按较高风险处理。

真实控制风险调用
  不能通过上述通用 L3 wrapper 执行，也不存在命令行绕过参数。
  必须使用限制了具体动作、目标和参数的专用 wrapper，并取得当前请求对精确动作的授权。

dry-run 控制调用
  应优先使用 remote_arm_dry_run.sh。
  该 wrapper 只接受 basename 明确包含 dry/check/validate/simulate/noop 的脚本。
  dry-run 标志只能是白名单参数，例如 --dry-run、--no-motion、--simulate。
  如果额外参数包含 --confirm-motion、--no-dry-run、--real*、--live*、--execute* 等，会直接拒绝。
  它只预览最终远程命令，不执行目标 Python；不能用脚本名或 dry-run 参数证明脚本无运动副作用。

remote_safety_check.sh
  只运行身份、路径、设备存在性与网络可达性的只读预检。
  不生成 token，也不能授权后续设备或运动动作。

remote_rsync_project.sh --delete
  即使传了 --yes，真实删除模式也必须额外传入 --confirm-delete RESOLVED_REMOTE_PATH。

remote_dataset_camera_semantics.sh
  是相机语义 prepared view 的专用 L3 非运动维护入口。它固定执行
  `utils.dataset_postprocess camera-semantics-prepare-view`，不提供 module、script
  或任意参数透传；默认只做完整 dry-run。`--apply` 必须携带前一次计划给出的
  64 位 `--expected-plan-hash`，并在远端 Python 启动前拒绝已存在的输出目录。
  它不改变通用 Python wrapper 的控制风险检测，也不能调用设备或机器人入口。

remote_dataset_camera_quarantine.sh
  是错误 prepared view 的可恢复隔离入口。它先用严格 loader 证明目标无效，
  再生成包含目录清单的计划；只有携带完全一致的 `--expected-plan-hash`
  才会在同一文件系统内原子改名到审计目录。有效视图、符号链接、跨文件系统、
  已存在目标均会被拒绝；该入口不删除或覆盖任何数据。

remote_dataset_camera_promotion.sh
  是完整相机纠正数据接替 canonical 路径的固定 L3 事务入口。它只接受经过严格
  v2.3/core-v2.3、全视频解码、相机 swapped→identity、同触觉 inventory、独立 inode 和同文件
  系统证明的晋升，旧目录会保留在 retirement root。旧源可为 `DETACHED_VERIFIED`，但纠正输出
  必须为 `SELF_CONTAINED`。apply 必须绑定 plan hash 并确认无 reader；此模式回滚前还必须先把
  当前 canonical 严格拆离。回滚只接受已发布事务文件和新的冻结 plan hash。它不提供删除、覆盖
  或任意参数透传。

remote_dataset_camera_training_validity.sh
  是损坏相机 episode 训练排除的固定 L3 metadata 入口。它只接受严格 loader 已验证的
  camera prepared view exclusion receipt，首次写入前重新绑定完整源 camera/numeric inventory，
  再在 `.modality_validity.lock` 下原子更新 `meta/modality_validity.json`。apply 必须绑定冻结的
  plan hash 并确认无 reader；它不改视频、Parquet、触觉 payload，也不接受任意参数透传。

remote_dataset_tactile_storage.sh
  是触觉模块可恢复拆装的专用 L3 非运动入口。它只允许只读 status、固定
  `whole-prefix` detach 和按精确 inventory rehydrate，不开放 legacy hardlink
  破窗参数或任意透传。apply 必须绑定同一次 proposal 的 `plan_hash`、
  `dataset_fingerprint`，并显式确认没有采集、训练或其他 reader；rehydrate 还
  必须绑定 64 位 `inventory_hash`。detach 先在独立文件系统完成冷副本验证和
  restore drill，cold 副本在 rehydrate 后也不会被自动删除。
```

示例安全流程：

```bash
./remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18 --require-device /dev/video0
./remote_arm_dry_run.sh \
  -H dex-remote \
  --project '~/DexTeleop' \
  --env dexteleop310 \
  --script tools/arm_dry_run.py \
  -- --config configs/arm.yaml
```

第二步只显示命令，不执行目标脚本。真实动作必须另行使用与该动作语义匹配的专用
wrapper；本示例不授予也不演示真机运动。

### RM65 遥测查询与定向配置工具

`scripts/probes/query_rm65_realtime_push.py` 可查询已有推送配置，或通过
`--listen-port` 被动接收十秒；它不配置控制器，不复用已占用的 UDP 端口。
`--check-only` 只检查显式输入，不连接 SDK。设备读取仍需当前目标的明确授权。

`remote_rm65_left_udp_destination.sh` 和 `remote_rm65_udp_force_coordinate.sh`
分别提供左臂接收地址、双臂工具坐标系的定向配置入口。默认仅预检，
`--dry-run` 只显示命令且不 SSH；真正配置必须另行获得该次操作授权并显式
传入 `--apply`、控制器地址、配置文件和新的仓库外证据目录。前者只改左臂
目的 IP，后者只将 force_coordinate 改为 2；其余配置字段保持原值。
两者复用现有接收器、原始配置日志和分离回滚守护进程，检查读回及十秒
被动反馈，失败时恢复原配置；不接入网关默认启动，不发送运动命令。
恢复尝试不依赖错误记录或恢复开始日志写盘成功；若磁盘写满，仍按原顺序
尝试设置并读回原配置，未能保存恢复结果时继续报错并保留守护恢复交接，
不标记恢复已完成。本次只验证模拟 SDK、故障注入和命令预览，未执行设备配置。

### D405 独立现场诊断

`remote_camera_v4l2_pair.sh` 在显式相机启动授权下，只对给定 USB serial 的
一台 D405 同时采集固定 640×480@15 的 Z16/YUYV，各 30 帧、限时 12 秒。
它用于区分内核传输与 SDK 启动故障，报告不能证明生产 RGB-D 已可用；
不重试、重开、重置或触碰机器人，既有证据目录会被拒绝。

`remote_camera_driver_bind.sh` 默认只检查明确端口和 serial 对应的 D405；
只有本次驱动恢复获得授权并显式加 `--bind` 后，才通过 `sudo -n` 绑定
已确认空闲且未绑定的唯一 uvcvideo 控制接口。不 unbind、重置 USB、
加载模块或启动流。两工具对占用检查失败或 stderr 不为空均停止；
`--dry-run` 不 SSH。raw 采集清理期间记录重复取消请求，完成所属子进程的
TERM、等待、必要的 KILL 和最终退出确认后才结束；单个进程 API 或日志关闭
错误不会跳过其它资源，未确认退出不标记清理完成，取消仍返回失败。
本次合并只做临时 sysfs/进程模拟与命令预览。

### VSCode Codex 插件使用

本目录带有一个子目录级 `AGENTS.md`，用于把远程 wrapper 和真机安全规则限制在本目录内。它不是全局规则。

从仓库根目录调用 `./codex_remote/remote_*.sh` 时，`remote-access` skill 仍应
生效，不需要为远端任务单独切换到旧的绝对路径。若要修改 wrapper 本身，
可以把会话根目录设为本目录，使子目录级 `AGENTS.md` 自动参与约束。

建议：

```text
远程 wrapper / 真机调试任务
  可以从项目根目录调用 codex_remote wrapper；每条命令都显式携带 -H。

普通本地开发任务
  不要让远端 wrapper 规则扩展到无关的本地任务。
```
