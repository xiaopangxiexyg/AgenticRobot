# UMI 数采

UMI 是 `teleop_runtime.collection` 的显式子命令。完整配置和状态机说明见
[`teleop_runtime/collection/umi/README.md`](../../teleop_runtime/collection/umi/README.md)。
顶层 `teleop_runtime.collection --help` 不枚举该兼容分派；参数帮助使用
`python -m teleop_runtime.collection umi --help`。

## 1. 生成两侧 UMI→desk 标定（H2）

```bash
python -m utils.umi_hand_eye --side left
python -m utils.umi_hand_eye --side right
```

配置默认引用 `teleop_records/umi_hand_eye/latest_left.json` 和 `latest_right.json`；缺失、过期
或与当前相机/tracker/tool 几何不一致时，下一步验证会失败。

## 2. 只验证配置

当前验证命令见 [real README 的第 1.7 节](./README.md)。它不初始化设备，但会
读取并验证配置、标定有效期和 v2.3 layout。

## 3. 正式启动（M）

以下命令会连接配置中的输入、相机和夹爪设备：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json
```

调试时保持状态采集但禁止真实夹爪指令：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --no-send-gripper
```

四踏板默认启用；以下写法只用于明确记录本次选择：

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --foot-pad
```

`--foot-pad` 当前为默认启用并带 manual fallback；需要完全禁用时使用 `--no-foot-pad`。

UMI 不构造 RM65 arm backend，但正式启动会连接相机、tracker、AngleTrigger、DMTAC，并可能
向夹爪发送命令，因此仍按 M 级处理。`--no-send-gripper` 只关闭夹爪命令，不会把相机和其他
设备连接变成 R0。
