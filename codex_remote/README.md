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

建议在 `~/.ssh/config` 中配置固定别名：

```text
Host dex-remote
  HostName 192.168.20.170
  User user
  IdentityFile ~/.ssh/id_ed25519
```

后续 wrapper 统一使用 `dex-remote`，避免在脚本和文档中散落 IP 地址。

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
./remote_python.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- teleop_runtime/replay/arm_init/dry_run.py
./remote_module.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- sim.mujoco.replay.runner
./remote_pytest.sh -H dex-remote --project ~/DexTeleop --env dexteleop310 -- tests/test_xxx.py -q
```

远程 Python 进程运行在远程电脑上，因此可以访问远程电脑的真实硬件。

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
./remote_arm_dry_run.sh -H dex-remote --config configs/arm.yaml
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
--require-safety-ok
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

推荐机制：

```text
remote_safety_check.sh 生成 safety token 或 safety-ok 文件。
高风险 wrapper 要求该文件在短时间内存在，例如 30 秒内。
高风险 wrapper 仍然要求 --confirm-motion。
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
--require-safety-ok
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
-H, --host          SSH host，例如 dex-remote
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
  --require-safety-ok \
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
5. 真实运动必须先跑 remote_safety_check。
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
[ ] 真机运动接入 remote_safety_check。
```

补充开发要求：

```text
[ ] 使用 remote_common.sh 中的共享 SSH 参数解析，除非有明确理由。
[ ] 输出包含 [local]、[target]、[run] 或 [check]、[result] 等可审计行。
[ ] 对 ambiguous input、未知操作等级、缺失远程路径、过期 safety token 采用 fail closed。
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
```

示例：

```bash
./remote_whoami.sh -H dex-remote
./remote_rsync_project.sh -H dex-remote --local ~/Release --remote '~/Release' --dry-run
./remote_python.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 -- script.py --arg value
./remote_camera_snapshot.sh -H dex-remote --device /dev/video0 --local-out ./debug/camera
./remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18 --require-device /dev/video0
./remote_view.sh -H dex-remote --path '~/DexTeleop/logs/latest.log' --tail 200
./remote_sync_check.sh -H dex-remote --local ~/DexTeleop --remote '~/DexTeleop' --path teleop_runtime/foo.py
```

### 实际同步和执行模型

常见真机开发时，本地工作目录已经通过上层脚本
`/home/user/Release/util_scripts/Connection/realtime_sync_ssh.sh` 以 no-delete
模式实时同步到远程指定工作目录。同步脚本必须加载项目根目录的
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

真实控制风险调用
  必须显式传入 --allow-control-risk。
  同时必须传入 --require-safety-ok TOKEN_PATH。
  远程执行前会检查 token 是否存在、是否新鲜、是否来自带硬件/控制检查的 remote_safety_check。

dry-run 控制调用
  应优先使用 remote_arm_dry_run.sh。
  该 wrapper 只接受 basename 明确包含 dry/check/validate/simulate/noop 的脚本。
  dry-run 标志只能是白名单参数，例如 --dry-run、--no-motion、--simulate。
  如果额外参数包含 --confirm-motion、--no-dry-run、--real*、--live*、--execute* 等，会直接拒绝。

remote_safety_check.sh
  默认 token 只能写到 /tmp/codex_* 或 ~/.cache/codex_remote_safety/。
  不允许空检查生成 token；必须至少检查 --arm-ip 或 --require-device。

remote_rsync_project.sh --delete
  即使传了 --yes，真实删除模式也必须额外传入 --confirm-delete RESOLVED_REMOTE_PATH。
```

示例安全流程：

```bash
./remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18 --require-device /dev/video0
./remote_module.sh \
  -H dex-remote \
  --project '~/DexTeleop' \
  --env dexteleop310 \
  --allow-control-risk \
  --require-safety-ok /tmp/codex_remote_safety_ok \
  -- sim.mujoco.replay.runner --dry-run
```

### VSCode Codex 插件使用

本目录带有一个子目录级 `AGENTS.md`，用于把远程 wrapper 和真机安全规则限制在本目录内。它不是全局规则。

建议：

```text
远程 wrapper / 真机调试任务
  从 /home/user/Release/util_scripts/Connection/codex 启动单独的 Codex 会话。

普通本地开发任务
  从对应项目目录启动 Codex，不要复用本目录的远程 wrapper 会话。
```

注意：在 VSCode 中仅仅打开本目录里的某个文件，不一定等同于让 Codex 以本目录作为当前工作目录。若需要这些规则稳定生效，应让 Codex 会话根目录/当前目录指向本目录。
