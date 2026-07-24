# Temporary MuJoCo PPO pipeline

This directory is an intentionally small end-to-end test of the current
Cube-Lift fast/vector environment. It is not the long-term model architecture.
Execution backends, actuator profiles, timing and task physics live under
`sim/mujoco`; this package only supplies temporary PPO/checkpoint tooling.

The task already controls only the right arm and right gripper: physical-policy
configs use a 7-D pose action containing three normalized right-TCP translation
increments, three base-frame rotation-vector increments, and one right-gripper
close increment.  The pose target is resolved through the shared RM65
pose-to-qref IK stage; the fitted RM65 and DMTAC actuators remain the only
simulated execution path.  The left arm is held at home.  The retained v0
direct-qpos debug config still exposes six joint deltas for compatibility.
Two explicit policy classes are available:

- the default `normal_v2` task consumes a 28-D measured motion/tracking state
  with no wrist RGB and no runtime episode clock. It uses physical MuJoCo
  contact, requires bilateral finger contact for lift reward/success, and
  enables pose-target anti-windup. Historical RGB configs remain available
  only for checkpoint reproduction while camera training is paused;
- `privileged_expert` consumes the 31-D simulator state and may use the debug
  contact latch. It is a teacher/debug artifact and is never a real deployment
  checkpoint.

Both classes retain the same identified RM65 and bilateral DM-Tac free-motion
response. Normal v2 selects a separately hashed Cube-Lift contact variant that
changes only simulator friction after the settled reset; the tuple is not
hardware force identification. Simulator-only dense-reward features are passed
through a separate training channel and never enter the normal observation.

## Environment

Use the project environment. PyTorch is locked to the official 2.10.0 CUDA
12.6 wheel; it runs on CPU when this debug machine has no working NVIDIA driver.

```bash
conda activate dexteleop310
cd /home/zc/HandMotion/Developing/DexTeleop
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

## Short smoke training

```bash
python -m sim.mujoco_temp_pipeline.train \
  --num-envs 2 \
  --rollout-steps 8 \
  --minibatch-size 16 \
  --update-epochs 1 \
  --total-timesteps 16 \
  --device cpu \
  --checkpoint /tmp/cube_lift_ppo_smoke.pt
```

## Normal training

```bash
python -m sim.mujoco_temp_pipeline.train \
  --num-envs 16 \
  --rollout-steps 128 \
  --total-timesteps 200000 \
  --run-name cube_lift_normal_v2 \
  --device auto
```

The default `rm65_right_cube_lift_rl_normal_v2.json` temporarily removes wrist
RGB. It retains only measured motion/tracking state, reduces pose/gripper action
increments, removes the sustained close-hover reward, rewards bilateral physical
contact, gates lift reward and success on that contact, penalizes failed and
unsafe terminations without penalizing a full-length time limit, and enables
pose-target anti-windup. The closed-grasp target is 22 mm above cube center and
the first physical milestone is a 70 mm bilateral-contact lift. For a full run:

```bash
python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json \
  --num-envs 16 \
  --rollout-steps 128 \
  --total-timesteps 2000000 \
  --run-name cube_lift_normal_v2 \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_v2.pt \
  --device auto
```

Before training, the simulator-only scripted gate should prove that the current
physics, safety projection, and task threshold admit a physical grasp:

```bash
python - <<'PY'
from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
    run_scripted_physical_grasp_gate,
)
print(run_scripted_physical_grasp_gate(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json"
))
PY
```

The expected reason is `bilateral_contact_lift_success`; this oracle reads
privileged cube state and is diagnostics-only. It is not a policy input.

If a home-reset probe still reports zero contact, first train the lift/contact
stage. Its simulator-only reset controller establishes bilateral physical
contact, then hands the same 28-D normal observation and 7-D action contract to
PPO. Bilateral contact is the primary reset condition. The v2 reset follows
the validated float32 gate trajectory, requires a 0.59 measured-close floor,
executes one centered stabilization step, and has bounded retries.
The retained v1 run learned to leave the cube to reduce its dense
safety-projection penalty, so its checkpoint is not a valid warm-start source.

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v2.json \
  --run-name cube_lift_contact_curriculum_v2 \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_contact_curriculum_v2.pt \
  --total-timesteps 200000 \
  --num-envs 8 \
  --rollout-steps 128 \
  --device auto
```

Only after deterministic v2 evaluation shows real bilateral lift success,
warm-start a fresh full home-reset run. `--init-checkpoint` transfers only
compatible policy weights and observation normalization; optimizer state,
training counters, and the task hash start fresh:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json \
  --init-checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_contact_curriculum_v2.pt \
  --run-name cube_lift_normal_v2_from_contact \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_v2_from_contact.pt \
  --total-timesteps 2000000 \
  --num-envs 8 \
  --rollout-steps 128 \
  --device auto
```

The old `normal_rgb_v4` config and checkpoint remain loadable only when selected
explicitly. For those compatibility runs, `--require-rgb-health` is an explicit
guard. It aborts at reset when the
configured RGB frames are flat, dark, or saturated; without it, the same
health statistics are recorded for diagnosis. It does not prove that the
cube is visible, so inspect home, pre-grasp, and contact-pose frames before a
long run.

Always pass the same `--task-config` to `evaluate` and `play`; checkpoint/task
hash mismatch remains a hard error.

Train the privileged expert only by selecting it explicitly:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v2.json \
  --run-name cube_lift_privileged_expert_rgb_v2 \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_privileged_expert_rgb_v2.pt \
  --total-timesteps 2000000 \
  --num-envs 16 \
  --device auto
```

The default checkpoint is
`sim/mujoco_temp_pipeline/checkpoints/latest.pt`. The temporary shaping reward
is computed outside the policy from a simulator-only reward state. Pass
`--no-reward-shaping` to train on the original task reward only. Console JSON
reports both original and shaped reward.
The environment applies a hard collision-safety projection before prescribed
robot motion. Reward shaping additionally penalizes the rejected fraction of
an action; it is not the mechanism that prevents penetration.

Because Cube-Lift does not yet claim identified grasp physics, the privileged
expert task may use `contact_latch_debug` semantics. The normal task never does.
In actuated v1 the
arms are physical but the gripper remains prescribed; default v2 uses the
identified free-motion gripper actuator. Bilateral cube contact plus sufficient
measured gripper closure acquires a latch; the cube then follows the right
grasp frame until the gripper opens. The latch makes the RL pipeline
testable without claiming realistic grasp physics. It exists only in this
simulation task and must not be reused as a real-robot deployment contract.

The legacy dense reward used by v1/v2 is deliberately conservative. The v3
profile adds an explicit close phase:

- a smooth reach score and reach-progress term guide the grasp frame to a
  pre-grasp point 30 mm above the cube center, rather than into the tabletop;
- an independent closed-grasp target gates closure reward, so closing the
  gripper never moves and invalidates the 30 mm reach target; the current
  state-only v2 physical profile uses the separately validated 22 mm target;
- v3 exposes an earlier close gate (12 cm with a 3 cm smooth width), while the
  strict 5 cm gate remains the lift-confidence signal;
- v3 rewards closure/progress through the wide gate, grants one acquisition
  bonus when closure crosses 0.45 in the strict region, then lets sustained
  cube lift and positive lift progress dominate;
- success has an explicit terminal bonus, while unsafe projected motion and
  unnecessarily large accepted actions are penalized.

When dense shaping is enabled, training does not directly add the task's raw
absolute-height reward. This prevents tiny table/contact height jitter from
being collected repeatedly without a grasp. The raw reward is still logged for
diagnostics. `--no-reward-shaping` restores raw-task-reward training (plus the
collision projection penalty).

## Training records

Every training command creates an independent directory below
`sim/mujoco_temp_pipeline/records/` by default. Override it with
`--record-root PATH`; use `--run-name NAME` to give the timestamped run a
recognizable suffix.

```text
records/<timestamp>_<run-name>/
├── run_config.json       # PPO/reward/task/runtime/CUDA parameters
├── metrics.jsonl         # complete per-update records
├── metrics.csv           # spreadsheet-friendly progress table
├── episodes.jsonl        # every completed episode and termination reason
├── summary.json          # final throughput, success and checkpoint paths
├── report.html           # self-contained curves and recent-update table
└── checkpoints/
    ├── step_000050000.pt
    └── step_000200704_final.pt
```

Open `report.html` directly in a browser; it has no web service or JavaScript
package dependency. The trainer updates it after every PPO update. By default
the terminal prints one compact progress line per update, including step
percentage, elapsed time, ETA, throughput, completed episodes and success rate,
raw/shaped reward, recent return, reach/closure/lift signals, safety
projection/rejection, clearance/penetration counts, actuator tracking error,
joint-limit/time-limit episode counts, PPO losses and the next checkpoint.
Complete per-update records are still
written to `metrics.jsonl` and `metrics.csv`. Use `--log-interval-updates N` to
print every Nth update (checkpoint and final updates are still shown), or
`--print-full-metrics` to restore the previous full-JSON terminal stream.
Safety projection/rejection rate, projection alpha, minimum robot-environment
distance and hard-penetration count are recorded beside the PPO metrics.
Rare states that enter the clearance envelope without hard penetration terminate
and auto-reset only that vector worker; their count is recorded separately as
`clearance_violation_count` instead of aborting the full training run.
MuJoCo joint ranges are soft constraints, so a position servo can leave a tiny
floating-point residual outside a canonical RM65 limit. The pose adapter clips
only residuals up to `1e-4` rad when re-anchoring the shared planner and reports
`pose_ik_limit_recovered`; a larger excursion terminates and auto-resets that
worker as `sim_joint_position_limit_violation`. The shared RM65 limits and the
real-robot path are not widened. The sim pose planner also takes its collision
clearance from the MuJoCo collision policy (currently 2 mm); the real-runtime
planner's 3 cm clearance default is not injected into this sim instance. If a
measured state still falls below the sim clearance, it terminates and
auto-resets as `sim_collision_margin_violation` instead of aborting PPO.
The report also records mean reach distance, both grasp gates, gripper closure,
bilateral finger contact/acquisition, lift fraction, and every individual
dense-reward term. RGB diagnostics remain present only for explicitly selected
camera configs.

For example:

```bash
xdg-open sim/mujoco_temp_pipeline/records/<run>/report.html
```

On a headless server, inspect `metrics.csv`, stream `metrics.jsonl`, or copy the
single HTML file back to the debug machine.

## Headless evaluation

```bash
python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/latest.pt \
  --episodes 32 \
  --num-envs 8
```

For an explicit normal checkpoint:

```bash
python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_v2.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json \
  --episodes 32 \
  --num-envs 8 \
  --device cpu
```

Historical RGB checkpoints must still be paired with their matching task
configs. These commands are compatibility examples, not current training
recommendations:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v2.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json \
  --episodes 32 \
  --num-envs 8 \
  --device auto
```

For the historical v3 input/reward contract:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v3.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json \
  --episodes 32 \
  --num-envs 8 \
  --device auto
```

For the historical v4 contract, pair the checkpoint with the same v4 task file:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v4.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json \
  --episodes 32 \
  --num-envs 8 \
  --device auto
```

## Viewer replay

Run this only in a graphical session. The current state-only normal-policy
viewer command is:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_v2.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json \
  --episodes 3 \
  --device cpu \
  --print-safety-events
```

The RGB policy is replayed with the same active-side wrist camera input:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v2.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json \
  --episodes 3 \
  --device auto \
  --print-safety-events
```

The v3 viewer command is:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v3.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json \
  --episodes 3 \
  --device auto \
  --print-safety-events
```

The historical v4 viewer command is:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v4.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json \
  --episodes 3 \
  --device auto \
  --print-safety-events
```

`play` loads the same policy and observation normalization as headless
evaluation, drives the reference `CubeLiftEnv` at 15 Hz, and displays the
result through the existing passive MuJoCo viewer. Historical RGB task configs
also render their configured active-side 48x48 camera input. This is still a
MuJoCo deployment check, not a real-camera or real-robot deployment contract.

## Legacy checkpoint collision regression

After changing task physics or collision configuration, the normal loaders
reject an older checkpoint because its task-config hash no longer matches. For
the narrow purpose of making an old policy stress the repaired collision layer,
run a quantitative headless check first:

```bash
python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/latest.pt \
  --episodes 8 \
  --num-envs 8 \
  --device cpu \
  --allow-task-config-mismatch
```

Then inspect the same policy in a graphical session:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/latest.pt \
  --episodes 1 \
  --device cpu \
  --allow-task-config-mismatch \
  --print-safety-events
```

The terminal reports projection/rejection rates, minimum accepted
robot-environment distance and hard-penetration count. In the viewer, projected
motion should stop at the safety boundary instead of moving through the table.
Treat reward and success metrics from this mismatched run as invalid: the
override is a collision regression tool, not a general checkpoint compatibility
mechanism. Without the explicit override, hash mismatch remains a hard error.
