# Simulation Point-Grasp Evaluation

Status: `incubating` experiment. This is an offline MuJoCo evaluation command,
not a production Skill route and not a real-robot entrypoint.

Both physical arm choices are available through `--arm-side left|right`.
`right` remains the compatibility default. The selected arm supplies the
`base_frame_id`, tool TCP, IK target, grasp body, finger contacts, and gripper
actuation; the other arm is held at its mirrored home pose.

## Real-like one-shot terminal

`run_terminal` is the smallest end-to-end operator entrypoint. It starts from
a collision-checked, side-specific observation home whose real MuJoCo wrist
camera points at a red 40×40×40 mm cube. It then renders aligned RGB-D from
either the fixed head camera or that physical wrist camera, invokes the same
point-grasp provider implementations offered by the real Gateway, saves the
candidate annotation and selected-grasp overlay, and executes pregrasp, grasp,
close, and lift through the existing CubeLift motion owner. It never connects
to real cameras or robot hardware.

Pass `--cube-color blue` to use the versioned blue-cube scene contract. The
blue and red contracts have identical 40 mm geometry, mass, inertia, contact,
and spawn pose; only their visual identity and evidence hash differ.

The default matches the real backend direction: GraspGenX over a loopback
service, using the selected wrist view. Start the reviewed service separately,
then run:

```bash
bash scripts/run_graspgenx_server.sh \
  --host 127.0.0.1 --port 5556

MUJOCO_GL=egl python -m experiments.sim_point_grasp.commands.run_terminal

MUJOCO_GL=egl python -m experiments.sim_point_grasp.commands.run_terminal \
  --cube-color blue
```

GraspNet and VGN are selectable without changing the terminal or motion path:

```bash
MUJOCO_GL=egl python -m experiments.sim_point_grasp.commands.run_terminal \
  --backend graspnet --view selected_wrist --arm-side right

MUJOCO_GL=egl python -m experiments.sim_point_grasp.commands.run_terminal \
  --backend vgn --view front --arm-side left
```

Use `--graspgenx-mode in-process` only in the separately validated compatible
GraspGenX Python environment. Every invocation owns one new output directory
under `runtime_artifacts/runs/experiments/sim_point_grasp_terminal/` and prints
one compact JSON outcome. The directory retains `initial_rgb.png`, aligned
metric depth, `skill_annotated.jpg`, `grasp_pose_overlay.png`, the complete
Skill/preflight JSON, execution montage, and optional MP4. A nonzero exit after
candidate generation is a motion/contact/lift failure, not loss of the grasp
rendering evidence.

Before motion, the terminal requires the SAM2 target center to lie inside the
candidate's physical DMTAC M opening, pad height, and finger-length interval.
It also rejects tabletop candidates whose tool +Z approach has less than the
default `0.5` downward cosine (`--minimum-downward-approach-cosine`). Accepted
poses move through an 80 mm gravity-up transit: translate with the current
wrist orientation, orient at clearance, then move to the declared pregrasp.
Each waypoint stops early when normalized pose error makes no progress for 20
steps, so an IK branch that is stuck at a limit does not consume the full phase
budget repeatedly.

## Visualization-only three-camera flow

Use the visualization command when the goal is to inspect generated poses, not
to test grasp execution. It resets and settles the scene once, runs the `front`
RGB-D frame through the formal SAM2 + GraspNet provider, and projects every
ranked, upward-normalized DMTAC M pose into separate `front`, `left_wrist`, and
`right_wrist` artifacts. Each overlay contains exactly one grasp. The summary
montage contains only the provider-selected grasp across all three views, while
the other ranked candidates remain available as individual PNGs. It never calls
IK, candidate reachability preflight, `env.step`, arm control, or gripper control
after the reset. The report hashes MuJoCo time, qpos, qvel, and ctrl before and
after inference/rendering and fails if that state changes.

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m experiments.sim_point_grasp.commands.visualize_candidates \
  --arm-side left \
  --seed 0 \
  --max-candidates 5
```

The new output directory contains the three raw RGB images, `front_depth_m.npy`,
the provider's selected-only `front_skill_annotation.jpg`, `skill_result.json`,
one `<camera>_candidate_<ordinal>_grasp_overlay.png` per camera/candidate pair,
the selected-only `grasp_multiview_montage.png`, and
`visualization_report.json` (schema
`dexteleop.experiments.sim_point_grasp_visualization.v2`). Each camera record contains its actual rendered
optical-frame intrinsics, `T_base_from_camera`, per-candidate artifact name,
derived `T_camera_from_tcp`, and an explicit visibility classification. This is
a simple overlay: geometry hidden by scene depth is intentionally still drawn.

`front` uses the scene's fixed head-camera pose. For this validation command
only, each wrist view keeps the current simulated wrist-camera origin but uses a
MuJoCo free camera aimed at the randomized cube center. The report labels these
views `visualization_only_look_at_override_from_wrist_origin`; they are not wrist
camera calibration and do not change the URDF, camera registry, or production
extrinsics. The pose used for overlay projection is recovered from MuJoCo's
actual rendered GL camera, so the wireframe and RGB view share the same camera.

The model-backed command requires the pinned assets and CUDA described in
`docs/agent_gateway/POINT_GRASP.md`. A custom nonexistent output directory can
be supplied with `--output-dir`; existing directories are never overwritten.

The command composes the existing physical Cube-Lift scene with the existing
read-only `perception.plan_point_grasp.v1` provider:

1. randomize and settle a free 40×40×100 mm box (body name `debug_cube`) on
   the table; the versioned experiment contract always loads the detailed
   EG2-4B2 CAD mechanism from
   `rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf`,
   together with the DMTAC M TCP and six-joint mimic actuation profile;
2. render aligned 640×480 RGB and absolute metric depth from fixed camera
   `front` (`head_camera_frame` optical axes: x right, y down, z forward);
3. project the simulator object center only to choose the click pixel; no
   ground-truth object pose is passed to SAM2 or GraspNet;
4. call the formal `PointGraspProvider`, validate its public request/result,
   apply GraspNet's insertion depth and SAM2 target-cloud width centering, then
   preflight candidates through the same IK and full-scene collision chain;
5. retain the generator's upward-selected parallel-jaw TCP and try 7/5/3.5 cm
   pregrasp offsets; move by translating above the grasp, orienting at 8 cm
   clearance, and then entering the declared pregrasp before executing the
   selected side-specific CAD-aligned DMTAC M `tool_tcp` goal, closing the
   physical gripper, and lifting 10 cm;
6. let `CubeLiftKernel` own canonical RM65 IK, collision projection, fitted
   actuators, contacts, termination, and success evaluation.

Success is `bilateral_contact_lift_success`: the cube center rises at least
8 cm above its settled height, is not touching the table, both fingers remain
in contact, and the predicate holds for 20 physics ticks.

## Run

Install the pinned SAM2/Grounding DINO/GraspNet sources and provide the assets
described in `docs/agent_gateway/POINT_GRASP.md`. GraspNet remains restricted to
the previously accepted noncommercial-research scope.

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --arm-side left \
  --trials 10 \
  --seed-start 0
```

The default output is a new timestamped directory under
`runtime_artifacts/runs/experiments/sim_point_grasp/`. Each trial contains the
initial RGB, float32 meter-depth NPY, depth visualization, Skill JSON and
annotations, a grasp-axis overlay, execution video, montage, and trial result.
`summary.json` reports the total success rate and failures split into perception,
pregrasp motion, grasp motion, contact, and lift stages. `summary_montage.jpg`
provides a two-column visual index of every trial. Per-trial timing includes
SAM2, GraspNet network inference, candidate ranking/visible-collision filtering,
motion phases, and every canonical IK solve; the aggregate reports count, total,
mean, P50, P95, minimum, and maximum. Successful trajectory paths are listed
separately, while all trials retain their execution video.

Useful options:

```bash
python -m experiments.sim_point_grasp.commands.run_evaluation --help
python -m experiments.sim_point_grasp.commands.run_evaluation \
  --arm-side left --trials 3 --seed-start 20 --no-video \
  --output-dir /tmp/sim-point-grasp-smoke
```

The command never connects to a camera or robot. It refuses to overwrite an
existing output directory. Run left and right evidence into separate output
directories; every run and trial record identifies its selected arm.

The default selection policy is `reachability_aware`. Use
`--candidate-selection-policy provider_selected_ordinal` only for a baseline
ablation. Reachability preflight uses IK, hard limits, and the complete MuJoCo
collision model, but never contact, lift, or the trial success predicate.

## 2026-08-31 intervention ablation

The reviewed follow-up is five independent 20-seed groups. Every group uses
seeds 0--19, the near-camera center `(-0.272, 0.0) m`, the 400-point visible
cloud gate, zero preventive clearance, 5 mm forbidden-penetration tolerance,
and the unchanged strict 80 mm bilateral-contact lift predicate. Each of the
first four groups changes exactly one intervention from the 90-step, one-point,
80 g, standard-gripper baseline; the fifth combines all four:

| Group | Changed parameters |
| --- | --- |
| 200-step motion | `--phase-step-limit 200` |
| eight SAM2-mask points | `--target-point-count 8` |
| slow persistent strong close | `--gripper-force-profile strong --close-step-limit 30 --close-action-scale 0.25` |
| 200 g object | `--object-mass-kg 0.20` |
| combined | all parameters above |

The eight-point policy runs the first prompt at the projected object-center
pixel, obtains the SAM2 mask, erodes its valid-depth interior, and chooses the
remaining prompts by deterministic farthest-point sampling. Each prompt invokes
the same public provider. Poses within 4 mm and 5 degrees are deduplicated and
at most ten combined candidates enter the existing IK/full-scene-collision
preflight. Candidate selection does not inspect future contact or lift outcome.

The `strong` profile is a simulation-only variant of the detailed CAD profile:
it raises each coupled rotational joint force limit from 0.50 to 0.75 N m. The
slower 0.25 close increments over 30 steps limit impact while continuing to
command closure. It is not a real-hardware force recommendation. The 200 g
scene uses a separately hash-locked contract and scales all three object inertia
components by the same 2.5 mass ratio.

Example commands (choose a distinct nonexistent output directory for each):

```bash
python -m experiments.sim_point_grasp.commands.run_evaluation \
  --trials 20 --seed-start 0 --phase-step-limit 200 \
  --output-dir runtime_artifacts/runs/experiments/sim_point_grasp/ablation_200step

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --trials 20 --seed-start 0 --target-point-count 8 \
  --output-dir runtime_artifacts/runs/experiments/sim_point_grasp/ablation_8point

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --trials 20 --seed-start 0 --gripper-force-profile strong \
  --close-step-limit 30 --close-action-scale 0.25 \
  --output-dir runtime_artifacts/runs/experiments/sim_point_grasp/ablation_strong_close

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --trials 20 --seed-start 0 --object-mass-kg 0.20 \
  --output-dir runtime_artifacts/runs/experiments/sim_point_grasp/ablation_heavy

python -m experiments.sim_point_grasp.commands.run_evaluation \
  --trials 20 --seed-start 0 --phase-step-limit 200 \
  --target-point-count 8 --object-mass-kg 0.20 \
  --gripper-force-profile strong --close-step-limit 30 \
  --close-action-scale 0.25 \
  --output-dir runtime_artifacts/runs/experiments/sim_point_grasp/ablation_combined
```

### 5×20 measured result

The completed seeds 0--19 runs used the exact matrix above. No trial met the
unchanged strict predicate, so there is no successful-trajectory video to list;
all 100 failure trajectories, per-trial montages, and five summary montages were
retained.

| Group | Strict success | Preflight failure | Contact-stage failure | Lift-stage failure | Bilateral contact observed | Maximum lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200-step motion | 0/20 | 8 | 11 | 1 | 1 | 7.096 mm |
| eight SAM2-mask points | 0/20 | 3 | 16 | 1 | 1 | 7.096 mm |
| slow persistent strong close | 0/20 | 8 | 9 | 3 | 3 | 8.674 mm |
| 200 g object | 0/20 | 8 | 11 | 1 | 1 | 5.784 mm |
| combined | 0/20 | 4 | 15 | 1 | 1 | 6.128 mm |

Eight mask points reduced no-executable-candidate outcomes from 8 to 3; the
combined group retained most of that gain at 4. The strong-close intervention
increased trials that reached a bilateral-contact/lift boundary from one to
three and produced the largest lift, but remained far below 80 mm. Adding mass
alone did not improve the failure-stage distribution. The remaining dominant
boundary is therefore grasp/contact geometry and contact-wrench stability, not
only motion duration, click location, close force, or object impulse response.

| Group | SAM2 mean per trial | GraspNet mean per trial | Provider mean | Candidate preflight mean | IK solve mean per call | Whole-trial mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200-step motion | 103.947 ms | 53.402 ms | 223.734 ms | 50.057 s | 1.851 ms | 52.189 s |
| eight SAM2-mask points | 710.215 ms | 403.496 ms | 1.732 s | 41.345 s | 1.835 ms | 45.450 s |
| slow persistent strong close | 103.565 ms | 52.968 ms | 222.797 ms | 32.899 s | 1.757 ms | 35.263 s |
| 200 g object | 104.697 ms | 53.679 ms | 226.642 ms | 35.077 s | 1.917 ms | 37.360 s |
| combined | 703.237 ms | 400.137 ms | 1.729 s | 67.041 s | 1.608 ms | 71.208 s |

The eight-point rows report the sum of eight model calls per trial. Their
per-prompt warm means excluding the first trial were 87.383/50.285 ms for
SAM2/GraspNet in the eight-point group and 86.560/49.777 ms in the combined
group. The five run directories are:

```text
runtime_artifacts/runs/experiments/sim_point_grasp/ablation_200step_20seeds_20260831/
runtime_artifacts/runs/experiments/sim_point_grasp/ablation_8point_20seeds_20260831/
runtime_artifacts/runs/experiments/sim_point_grasp/ablation_strong_close_20seeds_20260831/
runtime_artifacts/runs/experiments/sim_point_grasp/ablation_heavy_20seeds_20260831/
runtime_artifacts/runs/experiments/sim_point_grasp/ablation_combined_20seeds_20260831/
```

## Current relaxed-collision ablation

The active research-ablation profile accepts at most 400 visible points inside
the DMTAC M approximate finger/palm envelope, requests zero preventive scene
clearance during IK/motion projection, and terminates forbidden physical
contact only when penetration exceeds 5 mm. Collision checking itself remains
enabled. Hard joint limits and the strict 80 mm bilateral-contact lift success
predicate are unchanged. These settings are intentionally more permissive than
the historical 200-point, 2 mm clearance, and 1 mm hard-penetration baseline
reported below and must not be interpreted as a real-robot safety profile.

The active object center is `(-0.272, 0.0) m`, 20 cm closer to the fixed head
camera along world +X than the historical `(-0.472, 0.0) m` center. The default
X/Y jitter remains 2.5/8 cm. Use `--object-center-x-m` and
`--object-center-y-m` to make any future distance ablation explicit in the run
configuration and summary instead of silently changing the scene.

The 2026-08-31 near-camera run used seeds 0--19 at this center. Click-pixel
depth was 0.572--0.601 m (mean 0.585 m), versus 0.686--0.719 m in the earlier
five-seed far-center run. All 20 trials returned a Skill candidate, but the
strict result remained 0/20:

| Furthest stage | Trials | Result |
| --- | ---: | --- |
| pregrasp motion | 4 | pose timeout |
| exact grasp motion | 4 | two pose timeouts, two zero-clearance collision-margin failures |
| gripper contact | 11 | stable bilateral contact not acquired |
| lift execution | 1 | bilateral contact observed, but only 5.1 mm maximum lift |

Moving the object closer therefore removed the perception-stage rejection seen
in the earlier seed-0 regression and let 12/20 trials reach contact or lift,
but did not fix grasp/contact quality or the unchanged 80 mm lift requirement.
The complete evidence is under
`runtime_artifacts/runs/experiments/sim_point_grasp/near20cm_relaxed400_clearance0_penetration5mm_20seeds_20260831/`.

The 2026-08-31 fixed-seed regression (seeds 0--4) used the historical
`(-0.472, 0.0) m` object center under the relaxed collision profile and
remained strictly 0/5. Seed 3 improved from an 86.4 mm pregrasp timeout to a
16.0 mm exact-grasp timeout, but no episode met the unchanged success rule:

| Seed | Furthest stage | Result |
| ---: | --- | --- |
| 0 | perception | no candidate passed the 400-point visible-cloud gate |
| 1 | gripper contact | stable bilateral contact not acquired; max lift 5.1 mm |
| 2 | exact grasp motion | forbidden penetration still exceeded the 5 mm hard limit |
| 3 | exact grasp motion | pose timeout, 16.0 mm position residual |
| 4 | gripper contact | zero-clearance collision projection rejected overlap; max lift 3.0 mm |

The videos, montage, machine-readable timings, and per-trial evidence are under
`runtime_artifacts/runs/experiments/sim_point_grasp/relaxed400_clearance0_penetration5mm_5seeds_v2_20260831/`.
This five-seed regression is a boundary comparison, not a statistically stable
grasp-rate estimate.

## Current validation evidence

The 2026-08-29 run used seeds 0–99, the pinned tall-box detailed-CAD scene, GraspNet
`selected_ordinal=0`, and a maximum of 200 visible collision points. The strict
result was 0/100:

| Furthest completed boundary | Trials | Result |
| --- | ---: | --- |
| candidate returned, pregrasp rejected/timed out | 95 | motion-planning failure |
| pregrasp reached, exact grasp motion timed out | 2 | grasp-motion failure |
| exact grasp goal reached, bilateral contact not acquired | 3 | contact failure |

All 100 trials returned at least one Skill candidate, so the relaxed visible
collision gate removed the earlier perception rejection but did not yield a
strict physical grasp success. Across all stages, failures break down into 59
pose timeouts, 33 joint-limit violations, 4 robot-environment penetrations, 3
bilateral-contact failures, and 1 collision-margin violation. A diagnostic rerun
of seed 0 found that the right-arm q3 measured state exceeded the hard limit by
only 0.000115228 rad after physics execution; IK itself had zero rejected steps.

| Measured boundary | Count | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| SAM2 inference, all trials | 100 | 87.525 | 85.995 | 87.837 |
| GraspNet inference, all trials | 100 | 51.369 | 50.994 | 53.850 |
| candidate ranking + visible collision | 100 | 66.044 | 61.751 | 117.095 |
| complete point-grasp provider | 100 | 216.910 | 210.233 | 272.252 |
| canonical IK numerical solve, per call | 4845 | 2.093 | 2.064 | 2.269 |
| complete IK/planning frame, per call | 4845 | 2.942 | 2.901 | 3.165 |
| whole trial wall clock | 100 | 1840.382 | 1992.981 | 2071.501 |

Model startup took 2964.9 ms for SAM2 and 26.2 ms for GraspNet after provider
construction. Excluding the first cold trial, mean SAM2 and GraspNet inference
were 86.050 ms and 51.073 ms. No successful trajectory video exists because no
trial satisfied the strict success predicate; all 100 failure videos and JSON
remain machine-local mutable evidence under
`runtime_artifacts/runs/experiments/sim_point_grasp/collision200_4b2cad_100trials_20260829/`.

### Post-baseline remediation

Additional defects found by that baseline have since been fixed without changing the
200-point visible-collision gate, hard joint limits, or strict success rule:

- pose qrefs are inset by 0.001 rad at the final post-collision command
  boundary, and pose-target anti-windup is enabled;
- the detailed 4B2 CAD uses a hashed simulation TCP profile whose +X axis and
  lateral center match the actual finger mechanism;
- GraspNet prediction column 3 now uses its official 60 mm finger geometry to
  map the model grasp seed to the DMTAC palm/finger TCP boundary;
- preflight covers returned candidates, adaptive pregrasp offsets, and the
  180-degree parallel-jaw wrist symmetry, while SAM2 target points robustly
  center each grasp along both the opening and finger-height axes;
- the CAD actuator profile converts the legacy translational servo to bounded
  revolute-joint gains (`kp=5`, `kv=0.1`, force limit `0.5` per coupled joint)
  instead of applying `1200 N m/rad` and `20 N m` independently to all six
  mimic joints;
- real bilateral contact locks the accepted gripper force target during arm
  projection, lift translation commands are limited to 25% increments, and
  the episode budget now covers the evaluator's complete legal phase budget.

The first 20-seed reachability ablation, run before the CAD TCP/depth fixes,
still failed 20/20 at pregrasp and established that candidate order alone was
not the root cause. After the CAD TCP fix, seed 1 reproducibly reaches both
pregrasp and exact grasp; its strict failure moves to contact acquisition.
Contact calibration found real bilateral CAD contact at +10/+15 mm insertion,
but the old unconverted actuator launched the 80 g object. A fixed-seed
contact-only sweep bounded the replacement actuator: lower settings did not
close under load, while higher settings again displaced or ejected the object.
Neither unsafe insertion nor higher-force settings are defaults.

The post-remediation seeds 0--4 regression remains strictly 0/5; infrastructure
OOM from an unrelated VLLM process required seeds 2--4 to be completed as
separate one-trial runs, so OOM itself is excluded from the results:

| Seed | Furthest stage | Evidence |
| ---: | --- | --- |
| 0 | perception | no visible collision-safe clicked-region candidate |
| 1 | lift | bilateral contact; maximum lift 9.7 mm |
| 2 | exact grasp motion | authoritative robot-environment penetration |
| 3 | pregrasp | pose timeout, 86.4 mm residual |
| 4 | lift | bilateral contact; maximum lift 14.3 mm, no projectile or time limit |

This is a real boundary improvement over the original five-seed check (which
included two pregrasp failures and an unstable projectile lift), but it is not
a successful grasp-rate claim. The remaining dominant issue is GraspNet grasp
quality/contact wrench stability for the detailed CAD mechanism. The evaluator
does not relabel transient contact or sub-80-mm lift as success.
