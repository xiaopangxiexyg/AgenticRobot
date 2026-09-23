# 纯遥操与 OpenPI 同机争用测试（现场执行草案）

目标是测量双臂 VIVE/RM65 纯遥操在真实 OpenPI 推理/训练负载下的控制周期与 IK 抖动。合成推理客户端只生成黑色图像与零状态，绝不把模型动作接到机器人；不启动相机、触觉、夹爪或数采。真机遥操仅由现场人员启动、监护和停止。

## 已核对的远端资源

- 主机：`user@192.168.70.116`；遥操工作树：`/home/user/zc/AgenticRobot-ik-realtime-isolation-20260922`。
- 推理替代检查点：`/home/user/zc/checkpoints/openpi/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-prepared/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-b1-nw4/10000`。
- 匹配的训练 profile：`/home/user/zc/AgenticRobot-deployment-20260919-writing-stage1-bydate-h32-200k/deployments/openpi/configs/training/writing_stage1_all_by_date_front_right_wrist_h32_absolute_standard_200k_v1.json`；AgenticRobot 服务代码：`/home/user/zc/AgenticRobot-deployment-20260919-writing-stage1-bydate-h32-200k`；OpenPI 代码：`/home/user/zc/openpi_stage4_model_rtc_runs/20260913_by_date_v1/code/openpi-359f1ced`。不要用当前遥操工作树的服务入口装载该历史检查点。
- 2026-09-23 离线验证：匹配的历史服务 `--dry-run` 通过，真实 GPU 服务成功加载，合成客户端返回 `[32,16]` 动作并丢弃。首次请求约 9.0 秒（含 JAX 编译）；随后一次 10 秒窗口 23 请求，吞吐约 2.29 Hz，延迟 p50 0.436 秒、p95 0.442 秒。**这是无真机、无遥操的推理通路验证，不是控制实时性验收。**
- 原 Stage1 model-RTC v2 检查点虽有权重，实际服务因其不可变 prepared view 缺少 `meta/info.json` 而无法启动；不再作为 A/B/C 组负载，也不绕过其绑定校验。训练 D 组另行选型和验证，见下文。

## 测量矩阵

每组至少 120 秒，保留原始 trace，不在测量期间切换 debug 档位。现场动作幅度、节奏、设备状态尽量一致；先做 A→B→C，再按需要做 D。每组独立记录操作者、开始/结束时间、遥操 PID、模型 PID、Git HEAD、CPU 亲和性与 GPU 占用。

| 组 | 遥操 | OpenPI | 用途 |
| --- | --- | --- | --- |
| A | 同一纯遥操配置 | 未启动 | 基线 |
| B | 同上 | 模型服务已加载、无请求 | 常驻线程/显存效应 |
| C1 | 同上 | 本地合成请求，目标 2 Hz | 接近可持续速率的真实推理争用 |
| C2（可选） | 同上 | 本地合成请求，目标 5 Hz | 饱和推理争用；实测单客户端上限约 2.3 Hz |
| D（暂缓） | 同上 | 独立输出目录的有界训练 | 数据加载、JAX 编译与优化器争用；先完成离线 smoke |

另做 C 的受限 CPU 版本以评估隔离：先记录 `lscpu -e=CPU,CORE,SOCKET,NODE`，把物理核心及其 SMT sibling 一起划分；历史模型服务不支持当前工作树新增的 `--native-thread-count`/`--exclude-cpu`，因此用相同的原生命令与 `taskset -c <模型专用CPU集合>` 包裹模型进程，并设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`。亲和性和线程上限分别测量，**不改变**遥操配置和安全阈值。遥操端的 `control_debug`、timing 和 action trace 会引入观测开销，所有组必须一致。

## 运行骨架（由现场人员执行）

终端 1，在远端遥操工作树运行纯遥操 GUI。下面使用同一低开销预分配 trace 档位做所有组；不加入 `DATASET_COMMON_ARGS`、相机或触觉参数。真机运动须遵守现场安全流程，操作者负责检查启动前置条件、状态与急停。

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
source /home/user/zc/openpi/.venv/bin/activate
source docs/real/teleop_args.sh
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS[@]}" \
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}" \
  --no-arm-ip-save-latest
```

终端 2，启动**已实际加载和推理验证**的替代模型。服务只绑定 `127.0.0.1`；它不连接遥操，推理结果不会下发给机器人。先确认端口 8078 未被其他服务占用、GPU 有足够余量，再启动；保持这个终端打开。

```bash
cd /home/user/zc/AgenticRobot-deployment-20260919-writing-stage1-bydate-h32-200k
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
CKPT=/home/user/zc/checkpoints/openpi/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-prepared/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-b1-nw4/10000
PROFILE=/home/user/zc/AgenticRobot-deployment-20260919-writing-stage1-bydate-h32-200k/deployments/openpi/configs/training/writing_stage1_all_by_date_front_right_wrist_h32_absolute_standard_200k_v1.json
OPENPI=/home/user/zc/openpi_stage4_model_rtc_runs/20260913_by_date_v1/code/openpi-359f1ced
export PYTHONPATH="$OPENPI/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m deployments.openpi.server --checkpoint-dir "$CKPT" \
  --prepared-training-profile "$PROFILE" \
  --openpi-root "$OPENPI" --host 127.0.0.1 --port 8078
```

终端 3，**仅在终端 2 报告服务监听后运行**。使用遥操隔离工作树中的客户端与同一 Python 环境；先在不运行遥操时做小窗口预热，冷编译可能超过 9 秒。动作仅被校验形状后丢弃。预热结束后先测空闲 B，再与终端 4 的采集窗口对齐运行 C1。C2 是另一次独立测量，不要与 C1 并行。

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
source /home/user/zc/openpi/.venv/bin/activate
CKPT=/home/user/zc/checkpoints/openpi/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-prepared/writing-stage1-all-by-date-front-right-wrist-h32-absolute-standard-200k-v1-b1-nw4/10000
OPENPI=/home/user/zc/openpi_stage4_model_rtc_runs/20260913_by_date_v1/code/openpi-359f1ced
export PYTHONPATH="$OPENPI/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probes/openpi_synthetic_inference_load.py --checkpoint-dir "$CKPT" --duration-sec 12 --target-hz 2
```

B 组采集结束后，在终端 3 运行 C1：

```bash
mkdir -p /home/user/zc/teleop_cpu_contention
set -o pipefail
python scripts/probes/openpi_synthetic_inference_load.py --checkpoint-dir "$CKPT" --duration-sec 120 --target-hz 2 \
  | tee /home/user/zc/teleop_cpu_contention/openpi_inference_C1_120s.jsonl
```

可选 C2：仅在 C1 已结束、另行启动 120 秒采集窗口后，在同一终端运行：

```bash
python scripts/probes/openpi_synthetic_inference_load.py --checkpoint-dir "$CKPT" --duration-sec 120 --target-hz 5 \
  | tee /home/user/zc/teleop_cpu_contention/openpi_inference_C2_120s.jsonl
```

终端 4，确认 PID 指向正确的遥操和模型进程后，按组采集。采样器只读 `/proc`/GPU 指标；不要把旧 PID 或 `pgrep` 的多匹配结果直接传入。

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
ps -e -o pid=,ppid=,comm=,args= | rg 'run_vive_rm65_real_teleop|deployments.openpi.server|prepared_server_runtime|serve_dexteleop_policy.py|prepared_skill_launch'
# 人工确认后填入：
read -rp '遥操进程 PID: ' TELEOP_PID
read -rp '服务或训练进程 PID: ' MODEL_PID
scripts/probes/capture_cpu_contention.sh --phase C1_inference --duration-sec 120 \
  --pid "$TELEOP_PID" --pid "$MODEL_PID"
```

基线 A 仅传 `--pid "$TELEOP_PID"`。B 的模型服务保持空闲。将 120 秒的采集窗口与负载窗口对齐；采集进程不能自动判断 GUI 动作是否一致。`capture_cpu_contention.sh` 会输出采集目录，其中含 procfs 每秒线程运行/等待、CPU pressure、GPU、亲和性和进程快照。

遥操正常退出后，`actuator-trace` 才落盘为 `command_trace.npz`。记录启动日志给出的 trace 目录及上述 `capture_dir`，对每组分别运行：

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
read -rp '本次 trace 目录完整路径: ' TRACE_DIR
read -rp '本组 capture 目录完整路径: ' CAPTURE_DIR
python scripts/probes/summarize_teleop_contention.py \
  --trace-dir "$TRACE_DIR" --capture-dir "$CAPTURE_DIR"
```

摘要报告给出每侧 120 Hz 命令 wakeup gap/迟到、IK shadow 60 Hz gap 与超周期计数。它**不包含单次 IK 求解耗时**；若需区分“求解自身变慢”和“线程被抢占”，另做同参数 A/C 短程 `CONTROL_DEBUG_FULL_ARGS` 对照，明确记录该档诊断开销，不能与低开销 trace 的数值混为一组。

## 训练 D 组的停止条件

**D 组尚无可直接与真机并行的命令。**另一个训练候选 profile 为 `/home/user/zc/AgenticRobot-training-20260920-writing-stage1-bydate-h32-200k/deployments/openpi/configs/training/writing_stage1_all_by_date_front_right_wrist_h32_absolute_standard_200k_v2_20260920_local.json`：18 个源数据集的 `meta/info.json` 均存在，组织后的数据根存在，并有一次历史 10 步 smoke checkpoint。它证明有人曾跑通短程训练，但不证明当前 checkout、资源余量或与遥操并行的安全性。

标准训练 profile 的 `prepared_skill_launch train` 不接受直接用 `--num-train-steps` 缩短任务；照抄 200000 步原始配置会启动长训练，不能作为争用实验。后续须建立**独立且唯一输出目录**的有界测试 profile，在无真机时通过配置绑定检查、数据加载和短程 GPU smoke，并确认磁盘、显存、既有训练进程及停止方式，然后再给出 D 组命令。不得 resume/overwrite 历史训练或冒充缺失数据。若任何组出现安全告警、控制失频或异常运动，现场人员立即按既定安全流程结束遥操；模型测试不负责真机急停。

## 结果判读

报告每组遥操 IK shadow 周期/漏周期、120 Hz 命令 gap/迟到、Hold/安全事件、线程 CPU 运行和排队等待、CPU pressure、模型请求 p50/p95/p99 与吞吐；需要时用独立 full-debug A/C 组补充 IK solve p50/p95/p99。与 A 组比较，同时区分启动/编译瞬态和稳态。不把“推理客户端响应正常”当作“真机实时性通过”。只有这些时间序列对齐、现场动作可比且无保护降级，才讨论隔离方案是否有效。
