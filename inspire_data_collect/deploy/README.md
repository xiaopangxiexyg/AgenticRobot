# inspire_data_collect.deploy

策略部署模块，复用现有 InspireTac 手、FR3、RealSense 和触觉可视化代码，不修改采集主循环。

## 数据流规划

```text
RealSenseRGBSource ---- images --------\
FR3ArmBackend --------- arm state ------> DeploymentObservation -> Policy -> ActionChunkBuffer
InspireTacHandBackend - hand/tactile ---/                                      |
                                                                             v
                                                                  ActionSafetyFilter
                                                                             |
                                                                  EffectorCommand
                                                                             |
                                                           FR3ArmBackend / InspireTacHandBackend
```

`DeploymentObservation.model_input()` 给普通模型适配器的默认输入是：

- `image`: 单相机时为一张图，多相机时为 `{camera_key: image}`，无相机时为 `None`
- `images`: `{camera_key: HxWx3 uint8}`
- `state`: 19 维 `[arm(7), right_hand(6), left_hand(6)]`
- `state_dict`: `{"arm": ..., "right_hand": ..., "left_hand": ...}`
- `state_valid`: 19 维 bool mask
- `tactile`: 按 `right,left` 拼接的触觉向量
- `tactile_by_side`: `{side: tactile_vector}`
- `last_action`: 上一次经过安全过滤后的 19 维动作

策略输出支持：

- `np.ndarray` shape `(19,)` 或 `(chunk, 19)`
- hand-only shape `(12,)` 或 `(chunk, 12)`，按 `right,left`
- 单手 active 时 shape `(6,)` 或 `(chunk, 6)`
- dict: `{"action": ...}` 或 `{"action_chunk": ...}`
- dict: `{"arm": q7, "right_hand": q6, "left_hand": q6}`，各字段可为单步或 chunk

所有动作都解释为绝对关节目标，布局与采集 writer 的 `action` 一致。

## Dataset Replay Debug 模式

部署器可以不加载模型，而是从采集好的 dataset 里读取某个 episode 的 `action` 轨迹并按部署控制链路 replay。Replay 仍然经过 `ActionSafetyFilter`，默认仍是 dry-run；真实下发硬件仍需显式加 `--send-actions`，并且未设置 `--auto-start` 时仍需按 `g` 开始。
`--action-queue-policy auto` 会在 debug replay 模式自动 append 预取的 action chunk，避免按 `--min-action-queue` 补队列时跳过未执行轨迹；普通模型 policy 仍默认 replace pending queue。旧的 `--append-action-chunks` 仍可用，并等价于 `--action-queue-policy append`。

直接从 dataset root 读取第 0 个 episode：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --debug-replay-dataset /abs/path/to/dataset \
  --debug-replay-episode 0 \
  --active-sides both \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hz 30 \
  --auto-start
```

也可以直接指定 episode parquet：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --debug-replay-parquet /abs/path/to/dataset/data/chunk-000/episode_000000.parquet \
  --active-sides left \
  --left-hand-ip "$LEFT_HAND_IP" \
  --hz 30 \
  --auto-start
```

常用 replay 参数：

- `--debug-replay-start-index N` / `--debug-replay-end-index M`：只 replay `[N, M)` 帧。
- `--debug-replay-loop`：轨迹结束后循环；默认结束后保持最后一个 action。
- `--debug-replay-chunk-size 32`：每次策略查询预取的 action 数。
- `--debug-replay-action-column action`：默认读取 19 维 `action` 列；若该列不存在，会自动尝试用 `action.arm.joint_target`、`action.right_hand.joint_target`、`action.left_hand.joint_target` 拼回完整 action。
- `--action-queue-policy auto|replace|append`：默认 `auto`；replay 自动 append，普通 policy 自动 replace。

## 策略加载约定

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/path/to/ckpt.pt"}'
```

可加载形式：

- `hold`: 内置保持当前状态策略
- `package.module:PolicyClass`
- `package.module:create_policy`
- `/path/policy.py:PolicyClass`
- 模块内默认暴露 `create_policy`、`Policy`、`policy` 或 `predict`

类策略建议实现：

```python
class Policy:
    def __init__(self, checkpoint: str = ""):
        ...

    def reset(self):
        ...

    def predict(self, obs):
        model_input = obs.model_input()
        return {"action_chunk": ...}
```

模块级 `predict(input_dict)` 也可用，此时部署器会传入 `obs.model_input()`。

## 运行示例

先 dry-run，只看观测、策略输出解析和安全过滤：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy hold \
  --active-sides both \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --auto-start
```

真实下发动作需要显式加 `--send-actions`。未设置 `--auto-start` 时，启动后按 `g` 才进入策略执行，按 `h` 暂停并清空 action chunk。

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy /abs/path/my_policy.py:Policy \
  --policy-kwargs-json '{"checkpoint": "/abs/path/model.pt"}' \
  --active-sides both \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --camera-mode single \
  --camera-config inspire_data_collect/realsense.json \
  --send-actions
```

单臂 FR3：

```bash
python -m inspire_data_collect.deploy.run_policy \
  --policy /abs/path/my_policy.py:Policy \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --arm-side right \
  --fr3-host "$FR3_HOST" \
  --send-actions
```

## 安全边界

- 默认 dry-run，不会向硬件发动作。
- `--send-actions` 只打开下发能力；默认仍需按 `g`，除非传 `--auto-start`。
- 手部动作会 clip 到 `[0, 1]`。
- InspireTac 启动 force 通过 `--inspire-hand-max-force N` 配置，默认 `800`；兼容 `--max-fingertip-force`，也可用简写 `--max-force`。
- FR3 动作会按采集代码相同的安全关节范围和 `--arm-shrink-deg` 裁剪。
- `--max-arm-step-rad` 和 `--max-hand-step` 默认启用逐 tick 限速。
- FR3 backend 默认保留 startup alignment，从当前 qpos 缓慢对齐到第一个策略目标。
