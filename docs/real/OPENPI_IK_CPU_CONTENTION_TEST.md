# 纯遥操与 OpenPI 同机争用测试（现场执行草案）

目标是测量双臂 VIVE/RM65 纯遥操在真实 OpenPI 推理/训练负载下的控制周期与 IK 抖动。合成推理客户端只生成黑色图像与零状态，绝不把模型动作接到机器人；不启动相机、触觉、夹爪或数采。真机遥操仅由现场人员启动、监护和停止。

## 已核对的远端资源

- 主机：`user@192.168.70.116`；遥操工作树：`/home/user/zc/AgenticRobot-ik-realtime-isolation-20260922`。
- Stage1 model-RTC 已训练检查点：`/home/user/zc/checkpoints/openpi/writing-stage1-all-valid-all-view-h32-absolute-prepared-model-rtc-v2/writing-stage1-all-valid-all-view-h32-absolute-model-rtc-systemic-v3-359f1ced-r1-20260906/100000`，含 `params/`、`assets/`、`policy_artifact.json`。
- 历史训练 profile：`/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/profiles/writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_systemic_v3.json`，其 `openpi_checkout` 指向 `/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/openpi`；组织后的数据根存在。
- 对应 model-RTC profile：`/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/AgenticRobot/deployments/openpi/configs/model_rtc/writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json`。
- 历史文档中的 `/home/user/zc/openpi/checkpoints/...` 在当前主机不存在；不可照抄。当前遥操工作树的服务入口只支持 model-RTC profile v1，不能装载此 v2 检查点；须使用同次训练的 AgenticRobot 历史代码快照启动模型服务和训练。

历史代码快照的 `--dry-run` 已通过检查点/profile/资产哈希绑定，但一次实际 GPU 服务启动在训练配置解析阶段失败：`/home/user/zc/datasets/RealmanDMTAC/DexTeleop/camera_semantic_prepared_views/DexTeleop_0803__2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage.camera_only_v1/meta/info.json` 不存在。服务已退出，**C/D 组当前不可运行**。先由数据资产负责人确认该不可变 prepared view 的真实归档位置、来源和恢复方式；不得通过跳过 profile/哈希/数据校验或把另一数据集冒充该路径来启动。A/B 组也应在服务能够实际加载之后再做正式对比。

## 测量矩阵

每组至少 120 秒，保留原始 trace，不在测量期间切换 debug 档位。现场动作幅度、节奏、设备状态尽量一致；先做 A→B→C，再按需要做 D。每组独立记录操作者、开始/结束时间、遥操 PID、模型 PID、Git HEAD、CPU 亲和性与 GPU 占用。

| 组 | 遥操 | OpenPI | 用途 |
| --- | --- | --- | --- |
| A | 同一纯遥操配置 | 未启动 | 基线 |
| B | 同上 | 模型服务已加载、无请求 | 常驻线程/显存效应 |
| C | 同上 | 本地合成请求，目标 5 Hz | 真实推理计算争用 |
| D | 同上 | 独立输出目录的有界训练 | 数据加载、JAX 编译与优化器争用 |

另做 C/D 的受限 CPU 版本以评估隔离：先记录 `lscpu -e=CPU,CORE,SOCKET,NODE`，把物理核心及其 SMT sibling 一起划分；历史模型服务不支持当前工作树新增的 `--native-thread-count`/`--exclude-cpu`，因此用相同的原生命令与 `taskset -c <模型专用CPU集合>` 包裹模型进程，并设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`。这两类隔离手段应分别测量，**不改变**遥操配置和安全阈值。C/D 不要同时运行。遥操端的 `control_debug`、timing 和 action trace 会引入观测开销，所有组必须一致。

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

终端 2，**仅在缺失的 prepared view 经验证恢复后运行**。以下路径只针对上面核对的 Stage1 检查点；模型侧使用历史代码快照，服务绑定 `127.0.0.1`，模型结果不接入遥操。该组合的 `--dry-run` 绑定预检已通过，但实际加载此前失败：

```bash
cd /home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/AgenticRobot
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
CKPT=/home/user/zc/checkpoints/openpi/writing-stage1-all-valid-all-view-h32-absolute-prepared-model-rtc-v2/writing-stage1-all-valid-all-view-h32-absolute-model-rtc-systemic-v3-359f1ced-r1-20260906/100000
PROFILE=/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/profiles/writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_systemic_v3.json
RTC_PROFILE=/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/AgenticRobot/deployments/openpi/configs/model_rtc/writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json
OPENPI=/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/openpi
export PYTHONPATH="$OPENPI/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m deployments.openpi.server --checkpoint-dir "$CKPT" \
  --prepared-training-profile "$PROFILE" --model-rtc-profile "$RTC_PROFILE" \
  --openpi-root "$OPENPI" --host 127.0.0.1 --port 8078
```

终端 3，**仅在终端 2 的服务实际就绪后运行**。使用遥操隔离工作树中的客户端与同一 Python 环境；先用 1 个请求附近的小窗口预检，再做 120 秒 C 组。动作仅被校验形状后丢弃。

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
source /home/user/zc/openpi/.venv/bin/activate
CKPT=/home/user/zc/checkpoints/openpi/writing-stage1-all-valid-all-view-h32-absolute-prepared-model-rtc-v2/writing-stage1-all-valid-all-view-h32-absolute-model-rtc-systemic-v3-359f1ced-r1-20260906/100000
OPENPI=/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/openpi
export PYTHONPATH="$OPENPI/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probes/openpi_synthetic_inference_load.py --checkpoint-dir "$CKPT" --duration-sec 2 --target-hz 1
mkdir -p /home/user/zc/teleop_cpu_contention
python scripts/probes/openpi_synthetic_inference_load.py --checkpoint-dir "$CKPT" --duration-sec 120 --target-hz 5 \
  | tee /home/user/zc/teleop_cpu_contention/openpi_inference_120s.jsonl
```

终端 4，确认 PID 指向正确的遥操和模型进程后，按组采集。采样器只读 `/proc`/GPU 指标；不要把旧 PID 或 `pgrep` 的多匹配结果直接传入。

```bash
cd /home/user/zc/AgenticRobot-ik-realtime-isolation-20260922
ps -e -o pid=,ppid=,comm=,args= | rg 'run_vive_rm65_real_teleop|prepared_server_runtime|serve_dexteleop_policy.py|prepared_skill_launch'
# 人工确认后填入：
read -rp '遥操进程 PID: ' TELEOP_PID
read -rp '服务或训练进程 PID: ' MODEL_PID
scripts/probes/capture_cpu_contention.sh --phase C_inference --duration-sec 120 \
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

历史 100000 步训练不能直接重启或 resume。现有 `prepared_skill_launch` 支持 model-RTC profile 上的 `--num-train-steps`、`--save-interval` 和独立 `--exp-name`；可用其创建一次新的短程训练，但即使 20 步，JAX 编译、数据预检及最终 checkpoint 仍可能消耗数分钟和大量磁盘。**当前 prepared view 缺失，不能执行本节候选命令。**先确认不可变数据资产恢复、磁盘余量、GPU 空闲、snapshot/profile 兼容，并在无真机状态完成一次短程 smoke；未通过之前不执行 D 组。不得使用 `--resume` 或 `--overwrite` 指向历史训练目录。

候选命令（尚未实机验证，暂不作为直接真机并行命令；模型侧仍使用历史代码快照）：

```bash
cd /home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/AgenticRobot
source /home/user/zc/openpi/.venv/bin/activate
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
OPENPI=/home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/openpi
export PYTHONPATH="$OPENPI/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m deployments.openpi.training.prepared_skill_launch train \
  --profile /home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/profiles/writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_systemic_v3.json \
  --model-rtc-profile /home/user/zc/openpi_stage1_model_rtc_runs/20260906_systemic_v3/code/AgenticRobot/deployments/openpi/configs/model_rtc/writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json \
  --exp-name cpu-contention-smoke-YYYYMMDD-HHMMSS --num-train-steps 20 --save-interval 20
```

替换 `YYYYMMDD-HHMMSS` 为本次唯一标识；检查启动摘要中的 `checkpoint_launch_mode=fresh`，且 `checkpoint_dir` 是这个新实验名，然后才允许继续。无需为了停止训练而操作已有 checkpoint。若任何组出现安全告警、控制失频或异常运动，现场人员立即按既定安全流程结束遥操；模型测试不负责真机急停。

## 结果判读

报告每组遥操 IK shadow 周期/漏周期、120 Hz 命令 gap/迟到、Hold/安全事件、线程 CPU 运行和排队等待、CPU pressure、模型请求 p50/p95/p99 与吞吐；需要时用独立 full-debug A/C 组补充 IK solve p50/p95/p99。与 A 组比较，同时区分启动/编译瞬态和稳态。不把“推理客户端响应正常”当作“真机实时性通过”。只有这些时间序列对齐、现场动作可比且无保护降级，才讨论隔离方案是否有效。
