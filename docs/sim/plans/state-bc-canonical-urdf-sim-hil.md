# State-BC Sim-HIL with the canonical RM65 CAD scene

## Scope

This profile makes the current RM65 4B2 CAD scene the default task for the
MuJoCo Sim-HIL path used to exercise a state-based policy with human streaming
input. It is a task/configuration wiring change; it does not start a device,
open a viewer, or authorize real-robot motion.

The canonical scene contract is
`sim/configs/rm65_bimanual_physics_v1.json`, whose URDF is
`assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf`.
The task overlay keeps the existing v5 pre-grasp/lift curriculum and replaces
only the physics contract, CAD gripper profile, and the right-hand contact/body
names required by the scene.

## State-BC and human streaming path

The CLI still constructs the existing MuJoCo Sim-HIL application:

1. `CubeLiftTaskConfig` loads the canonical overlay and validates the contract
   and CAD gripper metadata hash.
2. The lower Sim-HIL target preparation builds the MuJoCo scene and the state
   policy source. State-BC remains an explicit `--policy-backend state_bc`
   selection with an explicit artifact; the default policy backend is not
   changed.
3. The Station control plane composes the policy source with the existing
   loopback human source. Human input remains a source for simulation testing;
   no VIVE or robot device is initialized by this profile.

The control GUI is opt-in and remains separate from the read-only status page:

```bash
PYTHONPATH=. python -m control_station.sim_hil \
  --policy-backend state_bc \
  --state-bc-artifact /absolute/path/state_bc.npz \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose_canonical.json \
  --enable --control-gui --mujoco-viewer
```

For a no-GUI wiring check, use the same explicit task and artifact with
`--dry-run`:

```bash
PYTHONPATH=. python -m control_station.sim_hil \
  --dry-run --policy-backend state_bc \
  --state-bc-artifact /absolute/path/state_bc.npz \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose_canonical.json
```

The canonical task is now the default task path. Historical checkpoints that
were trained against the legacy URDF must select their matching task explicitly
so task/contract compatibility remains fail-closed:

```bash
--task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json
```

`DEXTELEOP_SIM_HIL_TASK_CONFIG` remains the absolute-path override for either
profile.

## Verification boundary

The focused tests load the overlay, validate the canonical contract and CAD
profile, and instantiate the scene headlessly when MuJoCo is available. This
proves task/scene/source wiring and body-name compatibility. It does not prove
successful grasp/lift, state-BC artifact quality, GUI usability on a particular
desktop, real-robot readiness, or sim-to-real performance.
