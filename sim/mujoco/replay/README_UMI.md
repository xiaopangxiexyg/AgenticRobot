# Canonical desk-TCP action MuJoCo replay

This is the default MuJoCo dataset replay path. It consumes canonical Cartesian
actions: compact UMI 16D or real-teleop 26D. In both layouts the first 14 values
are left/right desk-frame tool-TCP poses; the remaining values are end-effector
commands. Arm observation qpos is never used as a fallback.

`python -m sim.mujoco.replay.runner` remains available only for explicit
joint-space/kinematic debugging. It is not the default or production replay
semantics.

## Control path

```text
canonical 16D/26D action at recorded ~15 Hz
  -> validate desk_absolute + tool_tcp contracts
  -> shared timestamp xyz-linear/quaternion-SLERP resampling at 60 Hz
  -> canonical desk to simulation world and left/right RM65 base frames
  -> robot_motion DualArmCollisionAwareRetargeter at 60 Hz
  -> MujocoPositionExecutionRig acceleration/step-limited qcmd at 120 Hz
  -> RM65 position actuators + identified bimanual DMTAC actuators
  -> MuJoCo physics at 1000 Hz
```

The reusable simulation control layer is
`sim.mujoco.control.desk_tcp_position`. It consumes the same
`robot_motion.position_control.ResampledMotionReference` contract used by the
real position stack. Online deployment can feed it from `CausalMotionResampler`;
dataset replay uses the shared offline resampler because all recorded
keyframes are already available and smooth trajectory inspection is desired.
The retargeter itself is imported from `robot_motion.rm65`; simulation does not
reverse-depend on the operator adapters in `teleop_runtime`.

The default replay profile is production `rm65_bimanual_position_v1`: adaptive
`min=2/max=4`, orientation cost `0.35`, damping `0.03`, and the
`rm65_bimanual_position_v1_qref` joint-velocity profile. Dataset main actions
are already the FK pose of accepted qref, so replay uses qref smoothing alpha
`1.0` and does not repeat collection-side inactive or adaptive `0.14 -> 0.45`
shaping. Use `--position-control-profile legacy` only for rollback comparison.

The physical MuJoCo scene remains the complete additive CAD-gripper model
`rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf`.
Coupled IK directly reuses the real-position-teleoperation default
`dual_arm_hand_proxy_open` collision scene from `robot_motion`, including its
checked 12 collision pairs. The proxy URDF is only the online collision/IK
model; it does not replace the Inspire-v3 MuJoCo scene.

The dataset action is already in canonical desk coordinates. The UMI
`raw_to_desk_transform` is provenance and is never applied again. The target
RM65 station contract supplies desk-to-simulation-world/base extrinsics. Tool
TCP offsets come from the checked-in left/right `DMTAC_M` calibration profiles.

Tactile manifests are validated and tracked by frame, with each manifest—not
top-level `info.json`—treated as storage truth. Tactile is currently only a
synchronized observation/visualization reference; it is not converted into a
MuJoCo contact force.

## Commands

Fast headless validation:

```bash
conda run --name "${DEXTELEOP_ENV_NAME:-dexteleop310}" python -m sim.mujoco.replay \
  --dataset-root remote_data/umi/2026-07-19_umi_dual_wrist_debug_remote_raw \
  --episode 0 \
  --max-frames 30 \
  --no-sleep
```

Realtime viewer replay:

```bash
conda run --name "${DEXTELEOP_ENV_NAME:-dexteleop310}" python -m sim.mujoco.replay \
  --dataset-root remote_data/umi/2026-07-19_umi_dual_wrist_debug_remote_raw \
  --episode 0 \
  --viewer
```

Viewer replay shows the exact left/right TCP targets supplied to IK by default:
the origin sphere marks target position and the red/green/blue arrows show the
target TCP X/Y/Z directions. This is a passive `user_scn` overlay and does not
change MuJoCo state, IK targets, commands, contacts, or physics. Disable it with
`--no-tcp-target-visualization`.

Useful range/output flags are `--start-frame`, `--max-frames`,
`--settle-seconds`, and `--summary-json`. Episode 1 is intentionally available;
the runner does not silently remove the 40-frame episode.

The historical `python -m sim.mujoco.replay.umi_runner` command is retained as
an alias for existing scripts.
