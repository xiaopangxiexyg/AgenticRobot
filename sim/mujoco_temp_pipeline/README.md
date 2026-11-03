# Temporary MuJoCo PPO pipeline

This directory is an intentionally small end-to-end test of the current
Cube-Lift fast/vector environment. It is not the long-term model architecture.
Execution backends, actuator profiles, timing and task physics live under
`sim/mujoco`; this package only supplies temporary PPO/checkpoint tooling.

The policy and simulator action spaces are deliberately separate:

```text
PPO latent action (1D / 3D / 4D / 7D)
  -> task-owned deterministic component mapping
  -> complete normalized pose action [xyz, rotvec, gripper] (7D)
  -> task phase conditioning
  -> pose IK and collision-safety admission
  -> actuator/contact response
```

PPO always stores the sampled latent value and its original log probability.
Conditioned, admitted, and measured actions are diagnostics/reward inputs only;
`--effective-action-for-ppo` is retained as a rejected migration flag because
rewriting an on-policy sample after safety projection is not valid PPO.

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
Task observations and actor inputs are also separate contracts. An explicit
`policy_observation_fields` allowlist may omit object pose or RGB even when the
environment carries those fields for diagnostics and ablation comparisons.

The default critic consumes the same deployable observation and shares the
actor representation. Matched Reach probes learned materially faster with this
sample-efficient topology than with two independent encoders. The established
unclipped value update remains the default; optional
`--value-clip-coef 0.2` experiments are recorded with their actual clip rate
and explained variance, but did not improve consistently across matched CPU
Reach probes. An independent same-observation critic remains available as an
isolation ablation via `--critic-backbone-mode separate`; its actor and critic
gradient norms are clipped independently.

The optional `--critic-observation-mode privileged_training_state` instead
feeds an automatically separate critic a 31-D simulator-state input. It never
changes the actor input, actor backbone, or deployment inference path;
checkpoints record both normalizers independently. This is asymmetric training
assistance, not a runtime requirement.

## Task suite

New profiles compose from versioned physical baselines while retaining a
dependency-aware hash. They are intended to be learned and evaluated in this
order:

| Profile | Objective | PPO latent | Public task output | Reset |
| --- | --- | --- | --- | --- |
| `rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json` | fixed-spawn reach baseline | xyz (3D) | pose+gripper (7D) | home; proprio actor |
| `rm65_right_cube_reach_pose_xyz_v1.json` | randomized reach | xyz (3D) | pose+gripper (7D) | home; proprio+object-estimator actor |
| `rm65_right_cube_reach_goal_delta_xyz_v2.json` | randomized reach, deployable goal-relative ablation | xyz (3D) | pose+gripper (7D) | home; proprio+base-frame target delta actor |
| `rm65_right_cube_reach_goal_delta_progress_xyz_v3.json` | randomized reach, progress-only reward | xyz (3D) | pose+gripper (7D) | home; proprio+base-frame target delta actor |
| `rm65_right_cube_reach_goal_delta_overhead_progress_xyz_v4.json` | experimental overhead-guided reach with coarse-to-fine TCP steps | xyz (3D) | pose+gripper (7D), full xyz far / 0.35 scale at target | home; proprio+smooth overhead target delta actor |
| `rm65_right_cube_reach_safe_hover_goal_delta_xyz_v1.json` | collision-aware 10 cm hover reach | xyz (3D) | pose+gripper (7D) | home; five clear steps required for success |
| `rm65_right_cube_reach_safe_pregrasp_goal_delta_xyz_v1.json` | collision-aware 4 cm pre-grasp reach | xyz (3D) | pose+gripper (7D) | home; five clear steps required for success |
| `rm65_right_cube_contact_acquire_fixed_pose_xyz_gripper_proprio_v1.json` | fixed-spawn contact baseline | xyz+gripper (4D) | pose+gripper (7D) | open pre-grasp; proprio actor |
| `rm65_right_cube_contact_acquire_pose_xyz_gripper_v1.json` | randomized contact acquisition | xyz+gripper (4D) | pose+gripper (7D) | open pre-grasp; proprio+object-estimator actor |
| `rm65_right_cube_contact_acquire_goal_delta_overhead_xyz_gripper_v2.json` | Reach-v4-compatible randomized contact bridge | xyz+gripper (4D) | pose+gripper (7D), coarse-to-fine xyz / independent gripper | open pre-grasp; proprio+smooth target-delta actor |
| `rm65_right_cube_contact_acquire_goal_delta_overhead_xyz_gripper_v3.json` | lower-projection randomized contact bridge | xyz+gripper (4D) | pose+gripper (7D), close-phase xy=0.1 / z=0.025 | open pre-grasp; proprio+smooth target-delta actor |
| `rm65_right_cube_contact_quality_goal_delta_xyz_gripper_v1.json` | inner-surface bilateral-contact acquisition | xyz+gripper (4D) | pose+gripper (7D) | pre-grasp; outer contact cannot substitute for grasp |
| `rm65_right_cube_contact_descend_close_pose_z_gripper_v1.json` | centered descend and close | z+gripper (2D) | pose+gripper (7D) | open pre-grasp; proprio actor |
| `rm65_right_cube_lift_short_pose_z_proprio_v1.json` | short 2 cm lift | z (1D) | pose+gripper (7D) | physical contact; proprio actor |
| `rm65_right_cube_lift_short_positive_z_proprio_v2.json` | short 2 cm lift without a zero-action shortcut | z (1D) | pose+gripper (7D), negative z admitted as hold | physical contact; proprio actor |
| `rm65_right_cube_lift_hold_pose_z_v1.json` | lift and hold an established grasp | z (1D) | pose+gripper (7D) | physical contact; proprio actor |
| `rm65_right_cube_pick_pose_xyz_gripper_v1.json` | fixed-spawn full-pick integration | xyz+gripper (4D) | pose+gripper (7D) | open pre-grasp; proprio actor |
| `rm65_right_cube_pick_pose_xyz_gripper_estimator_v1.json` | fixed-spawn estimator-input ablation | xyz+gripper (4D) | pose+gripper (7D) | open pre-grasp; proprio+object-estimator actor |
| `rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json` | randomized end-to-end 3 cm pick from home | xyz+gripper (4D) | direct pose+gripper (7D) | home, about 38 cm from cube; proprio+target-delta actor |
| `rm65_right_cube_pick_goal_delta_xyz_gripper_v3.json` | collision/contact-quality constrained 3 cm pick | xyz+gripper (4D) | direct pose+gripper (7D) | home; three clear steps plus inner-contact quality |
| `rm65_right_cube_pick_collision_uniform_light_v1.json` | matched-cost uniform collision-penalty ablation | xyz+gripper (4D) | direct pose+gripper (7D) | Pick v3 physics/success unchanged |
| `rm65_right_cube_pick_collision_category_light_v1.json` | semantic collision-category penalty ablation | xyz+gripper (4D) | direct pose+gripper (7D) | Pick v3 physics/success unchanged |
| `rm65_right_cube_pick_collision_category_balanced_v2.json` | stronger table/environment category ablation | xyz+gripper (4D) | direct pose+gripper (7D) | Pick v3 physics/success unchanged |
| `rm65_right_tall_block_pick_goal_delta_xyz_gripper_v1.json` | 30x30x50 mm clearance curriculum | xyz+gripper (4D) | direct pose+gripper (7D) | home; separate hashed physical contract |

The estimator profiles receive a simulator ground-truth proxy under an
explicit source contract. This represents the external estimator interface,
not privileged-policy permission or deployability proof. Every stage has a
proprio-only fixed-scene baseline; randomized object placement is rejected at
config load time unless the actor explicitly binds RGB, object pose, or a
derived target delta. The goal-delta Reach profile converts estimated object
pose plus measured-qpos RM65 forward kinematics into the same right-base frame
used by xyz pose actions. Its 3-D feature is real-computable and non-privileged,
and the Sim-HIL adapter reconstructs the same feature from the MuJoCo object
pose and measured TCP pose before emitting the public absolute-pose action.
The real-robot external estimator/FK source and its calibration are not wired
or proven by this simulation adapter.
The progress-only v3 Reach profile removes the old per-step absolute proximity
reward. A stationary policy therefore receives no dense reach return; only
measured reduction in target distance and terminal success are positive. This
avoids making a long hover at 7--8 cm competitive with reaching the target.
The experimental v4 profile retains that reward and final success target, but
changes the actor-visible delta to a smooth geometric approach target. Far from
the cube it aims 12 cm above the cube; inside an 8 cm horizontal radius it
smoothly descends to the final 3 cm target. The feature uses only the external
object position and measured TCP/base pose, so it is available to the normal
actor and Sim-HIL without a collision oracle or simulator reward state. Its
task-owned latent decoder keeps the full normalized xyz range outside an 8 cm
target-delta radius, then smoothly reduces the multiplier to 0.35 at zero
distance. At the inherited 12 mm TCP scale this means up to 12 mm per axis far
from the target and 4.2 mm per axis at zero distance. Rotation and gripper
remain zero. The decoder is observation-conditioned and checkpoint-hashed; it
replaces the rejected fixed global action cap so long-range motion is not
slowed unconditionally. The bootstrap teacher inverts the same decoder before
supervising the PPO latent action. This profile is a bounded collision-avoidance
candidate, not evidence that Reach or grasping is solved.
Training records and evaluation JSON report latent-action absolute mean and
saturation rate (`|a| >= 0.95`) separately from the public 7-D task-action
absolute mean. They
also report the distance-conditioned scale mean/minimum and the fraction of
steps where it is below one. These are diagnostics only: PPO log-probabilities
remain attached to the sampled latent action.
The Contact v2/v3 bridge preserves Reach v4's actor observation and dynamic xyz
decoder contracts, then adds one independently decoded gripper latent. This
allows a named-component warm start to reuse the deployable backbone and xyz
actor-head rows while leaving the new gripper row fresh and resetting the
critic. Contact success, reward, and the open-pregrasp reset remain task-owned;
no contact or collision signal is added to the actor input.
Inside 3.5 cm of the actor-visible target (or after measured close reaches
0.1), its task-owned decoder applies an additional 0.1 multiplier to xyz while
leaving gripper unchanged. A frozen 16-seed feasibility sweep selected this
point: 0.0 correction lost most grasps, while full correction succeeded only
by repeatedly asking the table-collision projector to truncate motion. The
0.1 setting retained 16/16 scripted success with no hard penetration and the
lowest projection/rejection rates among the successful isotropic settings.
Contact v3 keeps that 0.1 lateral correction but reduces close-phase z to
0.025. In the frozen 16-seed axis sweep it retained 16/16 success while
reducing projection/rejection from 38.4%/16.9% to 24.4%/9.6%; setting z to
zero failed 14/16, so v3 preserves a small vertical correction instead of
masking the axis entirely. The actor still outputs the same xyz-plus-gripper
latent and the public action remains the real-aligned 7-D pose delta.
A frozen seed-101 v3 adaptation (4096 PPO steps, actor-only warm start from the
v2 checkpoint, separate non-privileged critic, learning rate 1e-4) retained
32/32 deterministic successes on evaluation seeds starting at 20000, with no
hard penetration or clearance failure. Projection/rejection were 25.1%/10.7%
versus v2's 37.0%/15.5%; mean episode length increased from 37.4 to 68.4 steps.
The 128-step v3 smoke had lower 21.7%/8.4% rates but took 71.7 steps, so the
longer PPO adaptation traded some safety margin for speed. It is reported as a
training-objective limitation, not handled by post-hoc checkpoint selection.
LiftShort v1 maps latent z from `[-1, 1]` to `[0, 1]`; consequently, a zero
actor output commands a 0.5 upward step and makes the curriculum artificially
easy. LiftShort v2 keeps the public 7-D pose contract but maps zero to hold and
clips only negative executed z at the task boundary. Its physical gate proves
that zero fails while positive z succeeds. A fixed grid exposed a sharp
feasibility threshold: z<=0.2 was 0/4 and z>=0.3 was 4/4. With the generic std
anneal from 0.607 to 0.082, a frozen 4096-step run collapsed to 2/53 training
successes and 0/8 deterministic evaluation successes. Holding std at 0.607
instead produced 15/54 training successes and 8/8 deterministic evaluation
successes in 21 steps, with no hard penetration. Projection/rejection remained
42.9%/19.0%, so this is a fixed-spawn post-contact curriculum rather than a
solved lift task. Randomized estimator-input variants were not added: even a
constant z=1 policy failed some 2--4 mm jitter cases, and a 6 mm case failed
the scripted-contact reset itself.
The goal-delta Pick v2 profile closes the previous randomized end-to-end task
defect. The old actor-visible target was 30 mm above cube center while the
physically validated closed-grasp target was 22 mm, so PPO was asked to imitate
an 8 mm-inconsistent goal. The task now uses the 22 mm target everywhere,
removes observation-conditioned action scaling, starts from home about 38 cm
from the cube, randomizes cube xy by 6 mm and yaw by 0.15 rad, and requires a
3 cm lift with bilateral contact. A single actor-visible settling rule holds
one cycle when the previous public gripper action was close, then lifts by
4 mm while retaining xy centering. The vector teacher and independent physical
oracle both passed the frozen eight-seed feasibility sweep. This is MuJoCo
task evidence, not real grasp-physics or estimator calibration evidence.

### Collision-aware success and curriculum

Object contact is not treated as a generic collision. The safety policy admits
the intended finger/object and object/table contacts, while arm self-collision
and robot/table contact remain forbidden and are handled by the canonical pose
admission projector. The new success constraints are task-owned and default
off, preserving existing profiles and checkpoints:

- `success_min_safety_clear_policy_steps` requires a consecutive tail of public
  actions that needed no collision modification. An accepted monotonic recovery
  from an already marginal state is recorded separately and does not count as a
  fresh intervention;
- `success_min_bilateral_inner_contact_score` requires contact on both named
  inner finger surfaces, and `success_max_outer_contact_score` prevents an
  outside/side collision from satisfying the grasp predicate;
- `safety_intervention_penalty` is an optional fixed training cost per action
  modified by admission, in addition to the existing projection-fraction cost.
  It does not replace hard collision projection.

The intended non-privileged curriculum is safe hover at 10 cm, safe pre-grasp
at 4 cm, bilateral inner-contact acquisition, constrained Pick v3, then the
same Pick objective on the taller block. All stages keep the actor on deployable
proprioception plus target delta and keep PPO latent xyz or xyz+gripper separate
from the real-aligned public 7-D pose action. Simulator contact labels are used
only for reward, success, diagnostics, and an optional teacher; they are not
required actor observations.

The frozen v2 Pick checkpoint exposes why success alone is insufficient. On 16
seed-20000 cube episodes it reached 15/16, but 55.2% of actions were modified,
23.9% were fully rejected, and one hard penetration occurred. Blocking pairs
were dominated by link2/link4 self-collision and multiple arm, flange, base,
and finger contacts with the table. A 4096-step reward-only continuation under
the stricter v3 task regressed to 11/16 with three hard penetrations, so it was
rejected and did not replace the validated checkpoint.

The taller-block profile is a curriculum rather than a claim that collision is
solved. The unchanged cube checkpoint achieved 16/16 with no hard penetration
on its 16-episode seed-20000 evaluation, but still required projection on 52.1%
of actions. Across four scripted-oracle seeds, the taller block kept 4/4 grasp
success and reduced mean intervention rate from 56.4% for the cube to 43.7%;
the remaining self/table interventions must still be learned away.

Collision diagnostics now preserve the canonical forbidden group-pair meaning
instead of reducing every intervention to one bit: `robot_environment`,
`gripper_environment`, `robot_object`, `robot_self`,
`robot_gripper_self`, and `gripper_self`. Evaluation reports category action
rates, rejection rates, mean projection fraction, phase/category counts, and
concrete body pairs. PPO records the same per-category rates. These remain
training/evaluation diagnostics and are not actor inputs.

An optional `safety_collision_category_penalties` mapping applies a fixed
reward cost only when admission modifies an action in the named category. It
is additive to the projection-fraction cost, default-off, and cannot weaken or
replace hard projection. A frozen seed-20000 baseline showed that the validated
Pick actor's 55.0% intervention rate consisted of 27.5% gripper/environment,
14.8% robot/environment, 12.5% robot self-collision, and 0.2% gripper
self-collision per evaluated action. Self-collision occurred almost entirely
before contact; gripper/table collision persisted before contact, during
bilateral contact, and after contact.

A seed-411 matched-cost 4096-step ablation used the same checkpoint, rollout,
critic, and actor-reference coefficient. Uniform light penalty collapsed to
5/16 deterministic successes, 64.1% projection, 31.4% rejection, and seven
hard penetrations. Category-light improved that to 13/16, 58.2%, 28.5%, and
five, demonstrating that collision type affects PPO credit assignment. Raising
the non-privileged actor reference coefficient from 0.5 to 2.0 retained 15/16
and one hard penetration, reduced robot-self requests from 12.5% to 9.2%, but
shifted requests to gripper/environment and left total projection at 56.6%.
Increasing table/environment weights (`category_balanced_v2`) produced 14/16
and 59.6% projection. None beats the uncontinued checkpoint's 15/16, 55.0%,
23.8% rejection, and one hard penetration, so all trained candidates remain in
`/tmp` and the project checkpoint is unchanged. This evidence rejects a larger
scalar penalty as the next main lever. Follow-up work should target repeated
infeasible action requests through task-independent admission/event evidence;
a constrained optimizer needs an explicit cost budget before it is allowed to
influence PPO. PPO likelihood remains attached to its sampled latent action.

Collision accounting is now separated into three task-independent facts:

- a preventive admission event records one onset for a continuous projected
  streak, its bounded blocked fraction, category, and maximum streak length;
- an actual forbidden contact is read from MuJoCo's active-contact oracle over
  the complete policy interval, not inferred from projection;
- a hard violation remains the terminal penetration/safety fact.

These values are written as independent `collision_*` metrics and per-episode
fields. They are not actor observations, do not alter task reward, and do not
change the admitted public pose action. The legacy projection/category reward
terms remain compatibility options for earlier ablations.

Inspect the exact compiled collision geometry, material profiles, solver and
MultiCCD state without executing a policy action:

```bash
python -m sim.mujoco.diagnostics.collision_model \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json
```

The audit is task-reward agnostic: the task config selects the exact compiled
runtime scene and post-settle contact profile, while the report operates only
on MuJoCo model/contact facts.

Probe the local executable xyz set at an exact deterministic projector onset:

```bash
python -m sim.mujoco_temp_pipeline.probe_executability \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json \
  --episodes 1 \
  --seed 20000 \
  --max-onsets 1 \
  --device cpu \
  --output /tmp/cube_pick_local_executability.json
```

The probe uses one primary environment and one scratch environment. For every
candidate the scratch environment resets with the same seed and replays the
exact public-action prefix, then runs the normal task conditioning, pose IK,
planner, MuJoCo projector and physics step. This keeps memory constant and
fails if replay does not reproduce the onset observation. Candidate xyz
perturbations preserve the policy's gripper component. The report separates IK
feasibility, projector identity, full rejection, actual forbidden contact,
hard violation, nearest feasible correction and task/goal progress. Optional
`--onset-category` and `--onset-phase` filters select later table or post-contact
events without creating task-specific training curricula.

`measured_pose_reanchor` entries are counterfactual diagnostic controls only;
they are excluded from the policy-space feasible ratio. The probe never changes
PPO sample identity, actor observation, task reward, IK defaults, or hard safety.
Pose-target lag is also logged independently for clear, projected and onset
actions. Use the pre-action-aligned lag fields for causal comparisons because
anti-windup may reanchor the target after an intervention.

When `--init-checkpoint` crosses two of these profiles, only actor-head rows
with the same named latent component are transferred; new rows stay freshly
initialized and the critic head is reset. Shared components retain the source
checkpoint's exploration standard deviation instead of being widened back to
the fresh-run default. A new component starts at no more than
`--warm-start-new-action-log-std-cap` (default log std -1.5) and anneals
independently; this prevents a mature xyz policy and a new gripper dimension
from being forced through the same broad first-rollout noise. Observation and
public 7-D action contracts must still match exactly.

### Optional non-privileged Reach/Pick initialization

Goal-relative Reach can be seed-sensitive when a randomly initialized PPO
actor never discovers the correct near-target direction. An optional,
default-off actor bootstrap converts the actor-visible base-frame target delta
to the same xyz latent action for a bounded number of simulated transitions,
fits only the actor, resets the critic, and then starts ordinary PPO. It does
not read `training_reward_state`, does not initialize the critic, and does not
replace the sampled latent actions used by PPO likelihood accounting. Pick may
use the simulator post-contact phase only inside this optional teacher to label
lift samples. That phase is never appended to the 31-D actor observation, and
both evaluation and deployment run without it.

For a Pick contact that loses stability under continued target centering,
`--goal-delta-bootstrap-lock-lift-xy` optionally labels zero requested base-frame
XY increments during the reported lift phase, including its first settling
cycle. The default retains target centering. This changes teacher labels only;
the actor keeps the same observations and stationary action mapping. Zero XY
requests do not guarantee zero actual TCP drift. Record the explicit lift step
and assess collected teacher success and fitted actor success separately.

```bash
python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_reach_goal_delta_overhead_progress_xyz_v4.json \
  --goal-delta-bootstrap-transitions 4096 \
  --goal-delta-bootstrap-epochs 80 \
  --critic-backbone-mode separate \
  --num-envs 2 \
  --rollout-steps 128 \
  --minibatch-size 256 \
  --total-timesteps 25088 \
  --run-name reach_goal_delta_overhead_progress_bootstrap \
  --checkpoint /tmp/reach_goal_delta_overhead_progress_bootstrap.pt \
  --device cpu
```

For the randomized Pick profile, the validated bounded CPU command is:

```bash
python -m sim.mujoco_temp_pipeline.train \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json \
  --init-checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_contact_goal_delta_overhead_xyz_gripper_v3_s101.pt \
  --goal-delta-bootstrap-transitions 4096 \
  --goal-delta-bootstrap-epochs 120 \
  --goal-delta-bootstrap-lift-step-m 0.004 \
  --critic-observation-mode actor \
  --critic-backbone-mode separate \
  --num-envs 4 \
  --rollout-steps 128 \
  --minibatch-size 256 \
  --total-timesteps 512 \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt \
  --device cpu
```

The frozen seed-20000 evaluation produced 29/32 deterministic successes, no
hard penetration, and no clearance violation. An unanchored 8192-step PPO
continuation regressed to 20/32 with five hard penetrations. The optional
`--actor-reference-mse-coef` freezes the post-initialization actor and penalizes
deterministic latent-action drift using the same deployable observations. It is
default-off and does not make the actor or critic privileged. Capturing the
reference also freezes the actor observation normalizer after initialization,
so its action for the same raw observation does not drift as PPO visits new
states. An independent privileged critic normalizer continues to update. The
uncaptured default path retains adaptive actor normalization. The earlier
coefficient-2 8192-step ablation, collected before normalizer freezing, retained 28/32 but still did not beat the bootstrap
checkpoint, so the bootstrap checkpoint remains the current validated model.

Bootstrap collection records its own success, safety-projection, rejection,
and penetration rates in both `run_config.json` and the checkpoint. A
successful teacher is only an initialization feasibility check: high safety
projection still means the PPO/task path needs further collision-aware
optimization. A Contact curriculum may explicitly combine `--init-checkpoint`
with goal-delta bootstrap. The recorded initialization receipt then names the
ordered protocol (checkpoint transfer first, actor-only bootstrap second), so
the deployable actor provenance remains unambiguous; the critic is reset and
never initialized from teacher or privileged state.
For this initialized actor, a separate non-privileged critic is recommended:
it prevents the initially large value loss from rewriting the actor features.
This does not change the shared-actor default for fresh PPO runs.

## Environment

Use the project environment. PyTorch is locked to the official 2.10.0 CUDA
12.6 wheel; it runs on CPU when this debug machine has no working NVIDIA driver.

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
cd /home/zc/HandMotion/Developing/DexTeleop
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

MuJoCo vector workers keep process-local models, so host memory grows with
`--num-envs`; GPU availability does not remove that host-RAM cost. Training and
evaluation now default to four workers. On a memory-pressured workstation use
two, run only one train/evaluate command at a time, and check `free -h` before
launching. Do not parallelize multiple evaluation seeds on the same host when
swap is already active.

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
  --num-envs 4 \
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
  --num-envs 4 \
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
  --num-envs 4 \
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
  --num-envs 4 \
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
  --num-envs 4 \
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
  --num-envs 4
```

For an explicit normal checkpoint:

```bash
python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_v2.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json \
  --episodes 32 \
  --num-envs 4 \
  --device cpu
```

For the current randomized Pick checkpoint:

```bash
python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json \
  --episodes 32 \
  --num-envs 4 \
  --seed 20000 \
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
  --num-envs 4 \
  --device auto
```

For the historical v3 input/reward contract:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v3.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json \
  --episodes 32 \
  --num-envs 4 \
  --device auto
```

For the historical v4 contract, pair the checkpoint with the same v4 task file:

```bash
MUJOCO_GL=egl python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_lift_normal_rgb_v4.pt \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json \
  --episodes 32 \
  --num-envs 4 \
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

To inspect the current randomized Pick checkpoint visually:

```bash
python -m sim.mujoco_temp_pipeline.play \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json \
  --seed 20000 \
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
  --num-envs 4 \
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
