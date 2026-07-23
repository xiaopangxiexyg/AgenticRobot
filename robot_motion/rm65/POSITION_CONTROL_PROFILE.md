# RM65 bimanual position profile

`rm65_bimanual_position_v1` is the production position-control profile shared
by live real/sim teleoperation and the real/sim action-execution adapters.
`legacy` remains the explicit rollback/debug selector.

The main arm action contract is `accepted_qref_fk_pose`: collection stores the
FK pose of the last accepted qref. Therefore the action already contains all
live-human source shaping that happened before publication, including tracker
inactive handling and adaptive post-IK qref smoothing.

The stage boundary is strict:

- Live human teleop applies source shaping (`inactive scale=0.5`, adaptive
  qref alpha `0.14 -> 0.45`) and then materializes the main action.
- Dataset replay and policy deployment consume that main action. They must not
  apply those source stages again.
- Replay/deploy retain only downstream action execution: action-time
  resampling, coupled IK/collision and joint envelopes, the 60 -> 120 Hz
  acceleration-limited command planner, 30 ms target-velocity low-pass,
  `0.675 deg` command-step cap, final backend follow alpha `0.9`, qref stale
  timeout `0.2 s`, and CANFD trajectory-mode/radio `0/0`.

The formal rates are `15 Hz action / 60 Hz reference / 60 Hz IK / 120 Hz
command`. The IK objective is position/orientation `1.0/0.35`, adaptive
`min=2/max=4`, damping `0.03`. The 5 ms non-iteration reserve is split into
3.5 ms for base-candidate step/collision certification and 1.5 ms for the
remaining post-selection stages; setting
`--position-ik-candidate-certification-reserve-ms 0` restores the old split.
The production qref profile is
`rm65_bimanual_position_v1_qref`; it intentionally freezes the validated
numeric envelope previously named `rm65_position_2x`.

The position solver prefers collision-safe wrist, elbow, and shoulder-normal
branches. Shoulder classification is injected by the runtime through the same
RM65 SDK analyzer used by the final backend gate. If no clean feasible branch
exists within the frame budget, the selected safe singular candidate carries
an explicit certificate bound to the exact final qref fingerprint and is
allowed through instead of alternating with a hard hold. The reference
publisher later binds that certificate and transport target to the formal qref
epoch/sequence without changing dispatch order.
`--no-position-singularity-candidate-certificate` restores the previous
metadata-only fallback gate. `--no-position-prefer-non-shoulder-singular-branch` atomically
restores the previous shoulder hold-entry behavior without changing force
control.

Use `--position-control-profile legacy` to restore the adapter's previous
behavior. Explicit older `--arm-execution-profile` values are also treated as
compatibility selections and are not overwritten by the formal profile.
