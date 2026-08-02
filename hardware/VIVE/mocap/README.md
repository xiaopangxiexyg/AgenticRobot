# VIVE UDP tools

本目录包含用途明确分开的程序：

- `vive_data_sender.py`：本地默认的新 sender。从 OpenVR 取 pose/速度并发送
  `teleop_runtime.vive/v1` UDP JSON（300 Hz），并自动尝试独立 UDP 校时。默认行为仍
  严格要求恰好两个 GenericTracker，并按设备名字典序将较小者分配为 `left_wrist`、
  较大者分配为 `right_wrist`；少于或多于两个都会直接报错。需要发送更多 tracker
  时必须显式传入 `--tracker-count N`（`N >= 3`）；第 3 个角色为 `auxiliary`，后续按
  顺序为 `auxiliary_2`、`auxiliary_3` 等，UDP v1 根字段和单 tracker 字段均不改变。
  显式 `--single` 时，唯一物理 tracker 保持兼容角色 `left_wrist`，并附带一个
  invalid `right_wrist` 记录。接收端应用负责将这个 source role 映射到实际逻辑臂；
  不应仅根据逻辑右臂就假定 sender 会把单 tracker 改名为 `right_wrist`。
- `vive_data_sender_legacy.py`：稳定 v1 sender。强制关闭校时控制面，并冻结
  发布版的字段形状/60 Hz 默认值，用于 `DexTeleop_StableCollect_0705` 等稳定
  receiver。
- `vive_diagnostic_receiver.py`：只用于测试性数据记录，不连接机械臂，
  不参与遥操作控制。

## 单文件发送端迁移

`vive_data_sender.py` 本身是可独立复制的发送端文件。把这一个文件复制到 Windows
SteamVR 电脑即可；发送端不需要同时复制 `clock_sync.py`、`teleop_runtime/` 或其他
仓库模块。内嵌实现只要求目标 Python 环境能够执行 `import openvr`，不要求
`numpy` 或 `transforms3d`：

```powershell
python -c "import openvr; print(openvr.__file__)"
python .\vive_data_sender.py --host <receiver-ip> --port 9999 --hz 200
```

排查 `stale` 时先看 sender 每秒的 `[Status]` 行：`rate`/`loop` 反映实际采样
循环，`valid` 反映 OpenVR 当前 tracker 是否有效，`send_errors` 反映 UDP 是否真正
发送失败；`get_pose`、`registry` 和 `trackers_max` 会在慢循环时打印。`rate` 持续接近
目标且 `send_errors=0` 时，发送端通常仍在发送，即使某个 tracker 的 `valid=false`
也只是一个带错误原因的有效 UDP 帧，不等于链路断开。

接收端再按层判断：`packet_count`/`datagram_count`/`packet_age_monotonic_sec` 是 UDP
到达证据；`processed_packet_count`、`process_error_count` 和
`last_process_duration_sec` 是解析/发布证据，
`upstream_role_update_counts` 是 sender role 解析证据，`channel_update_counts` 是
当前逻辑映射证据，最后才看 Sim-HIL 的 `last_accepted_emission` 和
`last_rejection_reason`。因此“包新鲜但 human emission stale”应先查 role/valid、
session hold/fault 或 action contract，而不是盲目重启 sender。接收端
`worker_alive=false` 且 `process_error_count` 增长则是 receiver worker 异常，应重建
receiver；不要同时启动诊断 receiver 和遥操作 receiver 竞争 9999 端口。

若完整 `triad_openvr.py` 与 sender 一起复制，启动时仍会按确定顺序搜索 sender
同目录、其子目录以及父目录下的子目录，并优先使用找到的完整驱动。没有找到，或完整
驱动缺少 `numpy`/`transforms3d` 等非必要依赖时，会自动回退到内嵌 OpenVR 适配器；
控制台会打印实际选择。搜索会跳过 `.git`、虚拟环境和缓存目录，也不会退回到环境中
同名的第三方 `triad_openvr` 包。

仓库内两个 sender 入口仍共用同一份 OpenVR pose/v1 数据面实现。不要运行旧
`VIVE_sender/` 或旧 `data_sender*.py` sender 副本；这些入口不属于当前链路。
校时端口、启动命令和 compatibility matrix 见
[`CLOCK_SYNC.md`](CLOCK_SYNC.md)。

例如，严格等待并发送 3 个 tracker：

```bash
python -m hardware.VIVE.mocap.vive_data_sender \
  --host <receiver-ip> \
  --tracker-count 3 \
  --wait-for-trackers 10
```

诊断 receiver 会逐包接收，刻意不采用实时控制 receiver 的“清空旧包、只保留
最新包”策略。每次运行在 `teleop_records/vive_diagnostics/` 下创建独立目录：

- `vive_packets.jsonl`：收到的每个 UDP 包、接收时间、seq 缺包信息及原始 JSON；
- `vive_trackers.jsonl`：逐 tracker 的 pose、OpenVR 原生速度、相邻 pose 差分速度、
  单步位移/转角、重复 pose 标记；
- `summary.json`：接收频率、缺包率、无效帧、重复 pose 比例及各指标分位数；
- `metadata.json`：运行参数、主机和输出文件说明。

测试 receiver 与遥操作 receiver 默认都监听 UDP 9999，因此不能同时运行；实时
`VivePoseSource` 默认不启用 `SO_REUSEADDR`，第二个 receiver 会直接 bind 失败，
避免两个进程竞争同一数据流。只有明确的 legacy/诊断场景才应通过
`allow_socket_reuse=True` 选择复用；这不是 Sim-HIL 的默认配置。

## 两组建议录制

静止 60 秒：

```bash
python -m hardware.VIVE.mocap.vive_diagnostic_receiver \
  --label static \
  --duration-sec 60
```

运动 60 秒：

```bash
python -m hardware.VIVE.mocap.vive_diagnostic_receiver \
  --label motion \
  --duration-sec 60
```

`--duration-sec` 从首个 UDP 包到达时开始计时。设置为 `0` 时持续记录，按
`Ctrl-C` 会正常落盘并生成汇总。运动组建议包含慢速、快速、启停、反向以及转动，
并让左右 tracker 都有运动段；这样可以区分静止噪声、实际更新频率和运动速度分布。

## 独立链路 benchmark

`vive_tracker_benchmark_sender.py` 与
`teleop_runtime.debug.vive_tracker_benchmark_receiver` 保留为当前 JSON sender 的
独立链路基准工具。它们复用 `vive_data_sender.py` 的 OpenVR 采样和 tracker 映射，
但使用独立的 `dexteleop.vive_tracker_benchmark/v1` 协议，默认监听 UDP `10002`，
不会连接机械臂、发布控制指令或竞争实时遥操作的 UDP `9999` 数据端口和 `10001`
clock-sync 控制端口。

receiver 端先启动：

```bash
python -m teleop_runtime.debug.vive_tracker_benchmark_receiver --port 10002
```

再在有 SteamVR/OpenVR 的 VIVE 主机上启动 sender：

```bash
python -m hardware.VIVE.mocap.vive_tracker_benchmark_sender \
  --host <receiver-ip> \
  --port 10002
```

它按多个目标频率分阶段测量 sender 采样/读取耗时、发送与接收频率、丢包/重复包、
ACK RTT、tracker 有效率和直接链路校时后的一程延迟，并给出最高稳定频率。该工具
只用于链路容量和诊断，不是 teleop runtime 的数据源；纯逻辑回归见
`tests/test_vive_tracker_benchmark.py`。
