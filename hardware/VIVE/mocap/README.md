# VIVE UDP tools

本目录包含用途明确分开的程序：

- `vive_data_sender.py`：本地默认的新 sender。从 OpenVR 取 pose/速度并发送
  `teleop_runtime.vive/v1` UDP JSON（300 Hz），并自动尝试独立 UDP 校时。它只接受
  两个 GenericTracker，并按设备名字典序将较小者分配为 `left_wrist`、较大者分配为
  `right_wrist`；多于两个时直接报错。
- `vive_data_sender_legacy.py`：稳定 v1 sender。强制关闭校时控制面，并冻结
  发布版的字段形状/60 Hz 默认值，用于 `DexTeleop_StableCollect_0705` 等稳定
  receiver。
- `vive_diagnostic_receiver.py`：只用于测试性数据记录，不连接机械臂，
  不参与遥操作控制。

两个 sender 入口共用同一份 OpenVR pose/v1 数据面实现，避免新旧链路漂移；其
OpenVR 驱动固定为 `hardware/VIVE/triad_openvr/triad_openvr.py`。不要运行旧的
`VIVE_sender/`、`hardware/VIVE/triad_openvr.py` 或旧 `data_sender*.py` 副本；这些
路径不属于当前链路。校时端口、启动命令和 compatibility matrix 见
[`CLOCK_SYNC.md`](CLOCK_SYNC.md)。

诊断 receiver 会逐包接收，刻意不采用实时控制 receiver 的“清空旧包、只保留
最新包”策略。每次运行在 `teleop_records/vive_diagnostics/` 下创建独立目录：

- `vive_packets.jsonl`：收到的每个 UDP 包、接收时间、seq 缺包信息及原始 JSON；
- `vive_trackers.jsonl`：逐 tracker 的 pose、OpenVR 原生速度、相邻 pose 差分速度、
  单步位移/转角、重复 pose 标记；
- `summary.json`：接收频率、缺包率、无效帧、重复 pose 比例及各指标分位数；
- `metadata.json`：运行参数、主机和输出文件说明。

测试 receiver 与遥操作 receiver 默认都监听 UDP 9999，因此不能同时运行；否则
同一数据流会被竞争消费，统计结果无效。

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
