# UMI v2.3 collection

This package is the formal arm-free UMI collection frontend. It reuses the
existing DexTeleop Vive source, RealSense source and causal frame buffer,
AngleTrigger source, scalar-to-gripper retargeter, DMTac backend,
`HandStateHistoryPoller`, LeRobot-style writer, arm-mode label columns,
`CalibrationWebUi`, and four-pedal adapter.

It never constructs an arm backend. The legacy v2.3 `left.arm.pose` and
`right.arm.pose` layout components carry measured UMI tool-TCP poses, with
dataset metadata declaring `embodiment_has_arm=false`.

## Action temporal semantics

For an accepted source-sample sequence, a stored row uses the current sample as
the observation and the next accepted sample's UMI tool-TCP poses as its arm
action:

```text
observation_arm[t] = umi_pose[t]
action_arm[t]      = umi_pose[t+1]
action_gripper[t]  = AngleTrigger_command[t]
```

Only the 14 arm-pose dimensions use the next observation. The two gripper action
dimensions remain the real same-sample AngleTrigger commands; measured gripper
feedback remains in `observation.state`. Therefore an episode with `N` accepted
source samples stores `N-1` rows, videos and corresponding terminal-sample media.
The pending terminal source sample is deliberately dropped because it has no
known next arm pose. `observation.last_action` is rebuilt from the preceding
aligned action; the first row is causally initialized from its current arm pose
and current gripper command. Dataset and row metadata record this alignment and
the next source frame index/timestamp.

## Pose convention

The runtime uses the project convention `p_B = T_A_to_B @ p_A`:

```text
T_tcp_to_vive_world = T_tracker_to_vive_world @ T_tcp_to_tracker
T_tcp_to_desk       = T_vive_world_to_desk @ T_tcp_to_vive_world
```

For formal collection, left and right each load their own fresh hand-eye result.
Each result records `T_camera_to_tracker` and a derived `T_tcp_to_tracker` using
the named `dmgripper` profile, which resolves to that side's existing RM65
`DMTAC_M` tool profile. The default config selects the left result as the shared
desk transform source, while still requiring both side-specific TCP extrinsics.
Identity remains available only for explicit compatibility configs with
`allow_identity_tcp_placeholder=true`.

## Configuration validation

The default profile contains the current real-collection wrist camera serials:

- left: `261122273924`
- right: `260522278055`

Validate config, policy layout, calibration expiry, and device bindings without
initializing any device:

```bash
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --validate-only
```

Both `latest_left.json` and `latest_right.json` must exist, match the configured
camera/tracker and `dmgripper` bindings, contain `T_tcp_to_tracker`, and still be
within the configured maximum lifetime (3 hours by default). A changed or renewed desk
calibration cannot be appended to an
existing populated dataset because its pose-frame contract/hash differs.
The GUI shows a yellow calibration banner for the final 10 minutes and a red
banner for the final 5 minutes; both thresholds are configurable through
`calibration_warning_sec` and `calibration_critical_sec`. At expiry, an active
episode is finalized, new samples and commands stop, and the GUI remains open
with a red recalibration instruction instead of disappearing immediately.
Set `calibration_validity_hours` in the UMI JSON config, or pass
`--calibration-validity-hours HOURS` to override the lifetime for one run. The
setting applies to newly generated results only; existing results retain their
stored `expires_at_utc`.
Click **重新标定左右 UMI** in the Web GUI at any idle point (including after
expiry). The runtime closes the collection devices, launches the existing
dual-side `utils.umi_hand_eye` OpenCV window, and uses its existing automatic
sampling/coverage/solve path. When both results are saved and validated, the
runtime rebuilds the desk/TCP transforms, reconnects collection devices, and
returns to the ready state without restarting the command. Press `c` in the
calibration window after both sides report sufficient coverage; `q` cancels and
returns a retry prompt to the Web GUI.

The calibration tool binding and output directory are configured by
`hand_eye_config` and `hand_eye_output_dir`. Their camera, tracker, Vive port,
channel, and TCP-profile bindings are checked against the collection config
before any device is opened. If the configured dataset root already contains
episodes from a different calibration, collection resumes in a deterministic
same-parent directory named
`<dataset>__calib_<UTC>_<fingerprint>`; the actual path is shown in the GUI.
This prevents old and new immutable v2.3 pose contracts from being mixed.
Tactile data uses the existing external-stream writer; the default compression
is `none`, while `lz4` and `zstd` remain available through the config when the
corresponding existing optional package is installed.
UMI reads DMTac tactile snapshots at 15 Hz by default through
`dmtac_tactile_read_hz`. Its read period must fit within
`timing.tactile_max_age_sec`; otherwise required tactile freshness would
periodically reject otherwise valid rows. The GUI reports attempted, accepted,
and rejected sample counts, and each episode summary stores rejection counts by
reason. A temporarily rejected row is shown as **等待有效快照**; the episode
stays open and automatically resumes accepting rows when all required sensors
are fresh.

Gripper commands and position telemetry default to 20 Hz, with a 20 ms
post-write read guard (`dmtac_gripper_command_hz`,
`dmtac_gripper_telemetry_hz`, and
`dmtac_gripper_telemetry_command_grace_s`). Position freshness is determined
only from `joint_timestamp`, not from the newer telemetry-loop timestamp. Rows
also store the joint age, command begin/completion timestamps, read/send
elapsed time, lock wait, and skip status so actuator motion can be separated
from software scheduling delay.

## Formal run

After filling the DMTac assignment and producing a fresh UMI calibration:

```bash
# Collect a new UMI v2.3 dataset through the formal Web GUI.
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json
```

The provisional canonical-component v2.5-p shadow is wired but remains off by
default. Add `--v25p-native-shadow` only for dual-write validation; add
`--v25p-native-storage-layout parquet_stream_chunks_v1` as well to select the
chunked backend. The v2.3 episode remains authoritative and UMI's per-side
effective/availability timestamps are preserved.

AngleTrigger 不再从 UMI 启动流程调用终端 `input()`。若一个已保存侧仍能唯一匹配，且
恰好只剩一个未分配传感器，GUI 会补全另一侧并显示完整的 left/right 设备标签；操作员可
点击 **确认 Angle 左右自动判断**，也可拒绝并选择 **重新标定 Angle**。如果两侧都无法
唯一匹配，则不会按 ttyUSB 顺序猜测，只提供重新标定选项。重新标定步骤为：移动左侧并
确认、移动右侧并确认、依次采集左 0/1 与右 0/1 端点。每一步都通过
**确认 / 采集当前 Angle 步骤** 完成，成功保存后会自动继续设备初始化。

AngleTrigger 配置与 DMTac USB assignment 是两套独立配置。如果日志先显示
`Saved side assignment`，随后出现 `DMTAC CONFIG ERROR`，说明 AngleTrigger 已成功，
真正失败的是 DMTac 配置。例如旧 `/dev/serial/by-path/...` 不存在而当前只出现一个
未分配 `/dev/ttyUSB*` 时，程序不能安全猜测它属于左侧还是右侧，因此仍会拒绝输出。
UMI 会在启动 Web GUI **之前**检查 DMTac assignment。若恰好是一侧已匹配、另一侧缺失，
并且只剩一个可达候选，终端会显示补全结果并要求输入 `YES`；确认后保存新 assignment。
如果不是唯一补全，终端提供 `RECONFIGURE` 选项进入现有交互式重新分配。所有必需的终端
输入结束后才会启动 Web GUI，避免浏览器已显示但程序仍在终端等待输入。运行期的非交互
设备连接错误仍会留在 GUI，并提供 **重试设备初始化**。

Useful staging options:

```bash
# Collect input/state while preventing physical gripper command transmission.
python -m teleop_runtime.collection umi --no-send-gripper

# Foot-pad input is enabled by default.  A missing/unsupported device keeps
# the UMI GUI usable and falls back to the manual fix/init label controls.
python -m teleop_runtime.collection umi

# Explicitly disable foot-pad input for this run.
python -m teleop_runtime.collection umi --no-foot-pad
```

### Optional Plus episode curation

To add the same lightweight episode review used by the RM65 GUI, enable it
explicitly and keep the overlay outside the raw dataset:

```bash
python -m teleop_runtime.collection umi \
  --plus-episode-curation \
  --plus-annotation-root /path/to/plus_annotations
```

The GUI then offers an optional pre-start finite `variant` choice (only when a
task profile is supplied) and one post-save `disposition/reason` submission.
The sidecar writer runs after the existing episode has been saved and never
changes UMI rows, media, rejection rules, or episode success. Without the
flag, the Plus panel and sidecar writer are inactive; no Plus files are
created. Plus saves carry an episode-scoped token, and episode deletion is
temporarily blocked while the sidecar write is in flight. For a v2.4 compatibility view, pass
`--plus-source-binding v24_compatibility_content_locator_v1` explicitly; no
native v2.4 identity is inferred from paths.

No command starts an episode automatically. Recording begins and ends through
the reused formal Web GUI. Device startup and real gripper output are hardware
operations and must follow the project safety procedure.

空闲状态下 **开始数采** 和快捷键 `A` 始终可以提交一次开始请求。若 AngleTrigger、
相机、Vive、夹爪状态或首帧 causal snapshot 尚未 ready，后端不会创建 episode，但会在
GUI 的 episode message/control hint 中显示具体 `cannot start: ...` 原因；按钮不再因为
预检未通过而静默禁用。条件恢复后再次点击或按 `A` 即可开始。

空闲时还可在 GUI 输入 episode 序号删除错误数据。删除会连续重编号剩余 episode，
并刷新 writer 的下一 episode/全局 sample 索引，因此可直接继续采集。操作前版本保留
在数据集相邻的 `*.before_delete_episode_*` 恢复目录中。

## Dataset replay

Open the shared replay GUI on the collected UMI dataset. This command is an
offline dataset/media replay: `--no-connect-hardware` explicitly prevents RM65
hardware connection and action output.

```bash
# Replay episode 0 with wrist video, TCP trajectory, gripper state and tactile data.
python -m teleop_runtime.replay \
  --dataset-root datasets/umi_dual_wrist \
  --episode 0 \
  --no-connect-hardware \
  --host 127.0.0.1 \
  --port 8765
```

Omit `--episode 0` to select an episode in the GUI. Real-RM65 execution is not
enabled by this command; it requires the separate hardware replay preflight,
tool-profile checks, local output authorization, and an available start-qpos
initialization path.

The dataset root and task shown by the UMI GUI come from this validated config
and are intentionally read-only in the browser. Starting an episode performs a
fresh two-side pose/camera/gripper/tactile preflight. A stop with zero accepted
rows discards the empty episode; an explicit cancel discards it regardless of
row count.

## Operator labels

The existing GUI fix/init controls, keys (`c`/`g`), or foot pedals generate
latched per-side annotations. They do not control a robot:

- normal: `teleop`
- fix: `fixed_hold`
- init: `init_pose`

The labels remain in the existing `teleop.arm_mode.*` columns for v2.3 reader
compatibility. JSON detail records `annotation_only=true`,
`control_effect=false`, and `embodiment_has_arm=false`. Init arrival audit is
invalid with reason `annotation_only_no_arm`.

## Collection rejection policy

A row is written only when both sides have causal, fresh Vive pose, wrist image,
measured gripper state, and—when configured as required—tactile data. Invalid or
stale AngleTrigger input freezes the last physical target by sending no new
command and pauses row submission. Calibration expiry stops the active episode
and disables gripper command updates until the GUI calibration flow succeeds.
