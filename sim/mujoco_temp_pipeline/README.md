# Temporary MuJoCo PPO pipeline

This directory is an intentionally small end-to-end test of the current
Cube-Lift fast/vector environment. It is not the long-term model architecture.

The task already controls only the right arm and right gripper: the 7-D action
contains six normalized right-joint target deltas and one right-gripper close
delta. The left arm is held at home. The policy consumes the existing 31-D
privileged state, so the checkpoint is simulation-only and is not a real-robot
deployment artifact.

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
  --run-name cube_lift_debug \
  --device auto
```

The default checkpoint is
`sim/mujoco_temp_pipeline/checkpoints/latest.pt`. The temporary shaping reward
is computed outside the environment from the existing observation. Pass
`--no-reward-shaping` to train on the original task reward only. Console JSON
reports both original and shaped reward.
The environment applies a hard collision-safety projection before prescribed
robot motion. Reward shaping additionally penalizes the rejected fraction of
an action; it is not the mechanism that prevents penetration.

Because the current robot is driven by prescribed qpos rather than a physical
actuator/controller model, this temporary task uses
`contact_latch_debug` grasp semantics. Bilateral cube contact plus sufficient
gripper closure acquires a latch; the cube then follows the right grasp frame
until the gripper opens. The latch makes the RL pipeline testable without
claiming realistic grasp physics. It exists only in this simulation task and
must not be reused as a real-robot deployment contract.

The enabled dense reward is deliberately staged for this temporary task:

- a smooth reach score and reach-progress term guide the grasp frame to a
  pre-grasp point 30 mm above the cube center, rather than into the tabletop;
- an independent 15 mm closed-grasp target gates closure reward, so closing the
  gripper never moves and invalidates the 30 mm reach target;
- gripper closure is rewarded only inside that grasp region and penalized far
  away, with an immediate closure-progress term;
- latch acquisition receives a one-time stage transition signal, then sustained
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
package dependency. The trainer updates it after every PPO update. The terminal
continues to print the same metrics as JSON, including reward, recent episode
return, success rate, policy/value loss, entropy, KL, clip fraction, policy
standard deviation, learning rate and transitions per second.
Safety projection/rejection rate, projection alpha, minimum robot-environment
distance and hard-penetration count are recorded beside the PPO metrics.
Rare states that enter the clearance envelope without hard penetration terminate
and auto-reset only that vector worker; their count is recorded separately as
`clearance_violation_count` instead of aborting the full training run.
The report also records mean reach distance, grasp gate, gripper closure, grasp
latch state, lift fraction, and every individual dense-reward term so a stalled
reach/grasp/lift stage can be identified without viewer replay.

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

## Viewer replay

Run this only in a graphical session:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/latest.pt \
  --episodes 3
```

`play` loads the same policy and observation normalization as headless
evaluation, drives the reference `CubeLiftEnv` at 15 Hz, and displays the
result through the existing passive MuJoCo viewer.

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
