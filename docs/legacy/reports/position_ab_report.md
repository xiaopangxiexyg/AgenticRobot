# Position-control offline A/B report（历史）

> 这是已有 capture 的离线分析证据，不是运行入口。当前命令见
> [真实系统入口](../../real/README.md)。下文绝对路径只标识原始 capture，不要求其他机器存在。

Capture: `/home/zc/HandMotion/Developing/DexTeleop/logs/20260715_123512_283265_vive_rm65_real_teleop`

## Layer 1: recorded trace attribution

### left
- target_to_qref: translation p95=0.15594979364172767m, rotation p95=9.37732968277749deg, n=2750
- qref_to_qcmd: translation p95=0.0042107147097640366m, rotation p95=1.0592531504542493deg, n=5756
- target_to_qcmd: translation p95=0.15785637090413057m, rotation p95=10.061699250362082deg, n=5756
- raw q_ik→qref max joint delta p95=3.6065556328086643deg
- branches: {'current_pose_hold': 5, 'joint_limit_escape_left_j3_neg': 9, 'joint_limit_escape_left_j2_pos': 1, 'original': 2269, 'joint_limit_escape_left_j6_pos': 2, 'right_flip': 177, 'both_flip': 49, 'left_flip': 218, 'joint_limit_escape_right_j1_pos': 1, 'joint_limit_escape_right_j2_neg': 1, 'joint_limit_escape_left_j5_neg': 4, 'joint_limit_escape_right_j5_pos': 1, 'joint_limit_escape_left_j5_pos': 2, 'joint_limit_escape_left_j2_neg': 11}
- position-worsen/rotation-improve count: 465
### right
- target_to_qref: translation p95=0.1831932450673053m, rotation p95=10.892382575451453deg, n=2750
- qref_to_qcmd: translation p95=0.004678897583494098m, rotation p95=0.9688010936278975deg, n=5771
- target_to_qcmd: translation p95=0.1806672569416512m, rotation p95=11.489060011411084deg, n=5771
- raw q_ik→qref max joint delta p95=3.5629562977370783deg
- branches: {'current_pose_hold': 5, 'joint_limit_escape_left_j3_neg': 9, 'joint_limit_escape_left_j2_pos': 1, 'original': 2269, 'joint_limit_escape_left_j6_pos': 2, 'right_flip': 177, 'both_flip': 49, 'left_flip': 218, 'joint_limit_escape_right_j1_pos': 1, 'joint_limit_escape_right_j2_neg': 1, 'joint_limit_escape_left_j5_neg': 4, 'joint_limit_escape_right_j5_pos': 1, 'joint_limit_escape_left_j5_pos': 2, 'joint_limit_escape_left_j2_neg': 11}
- position-worsen/rotation-improve count: 408

## Layer 2: offline dual-IK replay

### baseline_ik_only
- replayed=2750 failures=0
- left: target→raw p95=0.14675912039150635m, target→qref p95=0.15121011924278208m
- right: target→raw p95=0.17873344809429043m, target→qref p95=0.18171763761787788m
### no_execution_limit_ik_only
- replayed=2750 failures=0
- left: target→raw p95=0.14658163112064357m, target→qref p95=0.14658163112064357m
- right: target→raw p95=0.17898185818603898m, target→qref p95=0.17898185818603898m
### candidate_selection_disabled_ik_only
- replayed=2750 failures=0
- left: target→raw p95=0.1467628667390773m, target→qref p95=0.15386372182419308m
- right: target→raw p95=0.17905095006170085m, target→qref p95=0.18397685122498533m
### position_priority_weights_ik_only
- replayed=2750 failures=0
- left: target→raw p95=0.11299016866085765m, target→qref p95=0.14676967637504798m
- right: target→raw p95=0.14434561287667402m, target→qref p95=0.17574622651491129m

## Layer 3: unilateral and speed-conditioned analysis

- state-pair counts: {'QUIETING+QUIETING': 50, 'INACTIVE+INACTIVE': 149, 'ACTIVE+INACTIVE': 90, 'ACTIVE+ACTIVE': 2242, 'QUIETING+ACTIVE': 73, 'ACTIVE+QUIETING': 111, 'INACTIVE+QUIETING': 8, 'QUIETING+INACTIVE': 27}
### left target linear speed
- 0.0-0.02m/s: n=540, translation p95=0.08277498806938147m, rotation p95=8.840709367658535deg
- 0.02-0.05m/s: n=957, translation p95=0.14576912786534868m, rotation p95=9.476718249648595deg
- 0.05-0.1m/s: n=856, translation p95=0.15618081267350037m, rotation p95=9.717542362519401deg
- 0.4+m/s: n=16, translation p95=0.17310764853419086m, rotation p95=10.460933818840973deg
- 0.2-0.4m/s: n=32, translation p95=0.22531416538817356m, rotation p95=5.731767021473319deg
- 0.1-0.2m/s: n=349, translation p95=0.2301692015517386m, rotation p95=9.312540222159287deg
### right target linear speed
- 0.0-0.02m/s: n=681, translation p95=0.10489354064952816m, rotation p95=10.700039024197501deg
- 0.02-0.05m/s: n=859, translation p95=0.15996393813906307m, rotation p95=10.545476817580052deg
- 0.05-0.1m/s: n=864, translation p95=0.24446258403304874m, rotation p95=12.183073860883907deg
- 0.4+m/s: n=13, translation p95=0.15772142280975998m, rotation p95=9.274619989427876deg
- 0.2-0.4m/s: n=46, translation p95=0.1506427746699588m, rotation p95=9.302039066048454deg
- 0.1-0.2m/s: n=287, translation p95=0.22032538915687175m, rotation p95=9.608040247226384deg
### unilateral replay: both
- failures=0
- left: target→raw p95=0.14678453201049463m, rotation p95=9.539230671804248deg
- right: target→raw p95=0.17970326857454344m, rotation p95=9.441751975882747deg
### unilateral replay: left_only
- failures=0
- left: target→raw p95=0.14434603890151615m, rotation p95=9.515921213694353deg
### unilateral replay: right_only
- failures=0
- right: target→raw p95=0.17962987791086746m, rotation p95=9.469942854941436deg

## Layer 4: solver convergence and collision-barrier A/B

### iterations_1
- iterations=1, barrier_gain=None, min_distance=default, replayed=2736 failures=14
- left: target→raw p95=0.14661486254763964m, rotation p95=9.548751243751006deg
- right: target→raw p95=0.179776483539354m, rotation p95=9.48694219622632deg
### iterations_2
- iterations=2, barrier_gain=None, min_distance=default, replayed=2736 failures=14
- left: target→raw p95=0.13222404841613922m, rotation p95=1.7316150437259445deg
- right: target→raw p95=0.17573609316081265m, rotation p95=3.2985263429142515deg
### iterations_4
- iterations=4, barrier_gain=None, min_distance=default, replayed=2736 failures=14
- left: target→raw p95=0.1143929672422974m, rotation p95=1.2299758425242682deg
- right: target→raw p95=0.1502382693125022m, rotation p95=1.8876453927248615deg
### collision_barrier_disabled
- iterations=1, barrier_gain=0.0, min_distance=0.0m, replayed=2736 failures=14
- left: target→raw p95=0.15497239468793564m, rotation p95=6.35593819568876deg
- right: target→raw p95=0.18369662788826632m, rotation p95=8.064002375207057deg
