# RealmanDMTAC 数据集盘点与整理清单

状态：待用户逐项补充语义备注；存储操作必须以执行当时的重新审计和回执为准。

盘点基线：2026-08-12，远端主机 `dex-remote`。

## 1. 范围与安全边界

本文是 RealmanDMTAC 数据集的唯一整理清单，记录当前目录、协议、Episode/帧数、
OpenPI 引用关系、hardlink 风险和触觉存储状态。它不是数据完整性回执，也不授权根据旧快照删除数据。

任何触觉本地退役必须在执行当时重新证明：

1. 数据集已封存且没有 writer、reader 或 maintenance lease；
2. NAS 挂载身份与预期 CIFS 设备一致；
3. cold inventory、export receipt、文件数、总字节和逐文件 hash 与当前 hot inventory 一致；
4. cold copy 位于独立设备且不是 source hardlink；
5. restore drill 已通过；
6. 只退役 receipt 明确覆盖的本地路径，不根据目录名或大小猜测；
7. `FormalCollection` 作为活跃采集根单独保护。

旧兼容路径大多是指向 canonical 根的软链接，不是第二份数据或独立备份。hardlink 目录组也不能按
路径大小推导可回收空间。

路径缩写：

- `D` = `/home/user/zc/datasets/RealmanDMTAC/DexTeleop/datasets`
- `D8` = `/home/user/zc/datasets/RealmanDMTAC/DexTeleop_0803/datasets`
- `S` = `/home/user/zc/datasets/RealmanDMTAC/DexTeleop_StableCollect_0705/datasets`
- `F` = `/home/user/zc/datasets/RealmanDMTAC/FormalCollection`
- `L` = `/home/user/zc/DexTeleop/datasets`

状态缩写：

- `C`：触觉已无损压缩；
- `DET`：本地 tactile 已按有回执流程拆出；
- `RAW`：触觉仍是未压缩原始数据；
- `ERR`：触觉审计异常；
- `Prog`：有 Progress sidecar；
- `Art`：有 modality artifact 元数据；
- `P0`：当前 OpenPI profile/Prepared binding 直接引用；
- `HL=n`：迁移快照中有 n 条 hardlink 路径，不代表 n 个独立 inode。

## 2. 当前总体状态

- canonical 根下当前有 52 个带 `meta/info.json` 的数据集；
- canonical 根外原有 14 个旧目录数据集；本轮已安全删除 8 个真空壳，当前剩 6 个真实目录；
- 迁移计划中有 2 个目标当前缺失；
- 另有 1 个不含 `meta/info.json` 的不完整目录；
- 当前 OpenPI 训练直接引用 7 个数据集，共选择 1217 个 Episode、393,540 个 front-only 窗口；
- Prepared binding 中未发现跨 source 的 selected identity 重复。

## 3. DexTeleop 历史数据

| ID | 相对路径 | 协议 | Episode / 帧 | 关系和当前状态 | 用户备注/决策 |
|---|---|---:|---:|---|---|
| C01 | `D/2026-07-23_dmtac_rm65_bimanual_position_v1` | v2.3 | 2 / 326 | C，独立小样本 | |
| C02 | `D/2026-07-23_vive_rm65_real_teleop_old_version` | legacy/待确认 | 515 / 84,650 | C、DET、Art；HL=6695，0723 转换族，hot payload 已由 NAS 独立副本验证后退役 | |
| C03 | `D/2026-07-23_vive_rm65_real_teleop_old_version_v22` | v2.2 | 515 / 84,650 | C、DET、Art；HL=6695，C02 派生，hot payload 已由 NAS 独立副本验证后退役 | |
| C04 | `D/2026-07-23_vive_rm65_real_teleop_old_version_v22_measured_state` | v2.2 | 515 / 84,650 | C、DET、Art；HL=6695，C02 派生，hot payload 已由 NAS 独立副本验证后退役 | |
| C05 | `D/2026-07-23_vive_rm65_real_teleop_old_version_v23` | v2.3 | 515 / 84,650 | C，HL=7198，0723 最新协议版本；NAS 事务链处理中 | |
| C06 | `D/2026-07-24_dmtac_rm65_bimanual_position_v1` | v2.3 | 1 / 55 | C，独立小样本 | |
| C07 | `D/2026-07-24_vive_rm65_real_teleop` | legacy/待确认 | 242 / 35,832 | C，HL=3374，0724 转换族 | |
| C08 | `D/2026-07-24_vive_rm65_real_teleop_filtered_no184` | legacy/待确认 | 241 / 35,667 | C，HL=3374 | |
| C09 | `D/2026-07-24_vive_rm65_real_teleop_v22` | v2.2 | 241 / 35,667 | C，HL=3133 | |
| C10 | `D/2026-07-24_vive_rm65_real_teleop_v22_pose_consistent` | v2.2 | 241 / 35,667 | C，HL=3133 | |
| C11 | `D/2026-07-24_vive_rm65_real_teleop_v23` | v2.3 | 241 / 35,667 | C，HL=3133，0724 最新协议版本 | |
| C12 | `D/2026-07-25_vive_rm65_real_teleop` | legacy/待确认 | 311 / 68,471 | C、DET、Art；已由 NAS 独立副本完成 restore drill 并退役 hot payload，0725 转换族 | |
| C13 | `D/2026-07-25_vive_rm65_real_teleop_filtered_no283` | legacy/待确认 | 310 / 68,408 | C、Art，HL=4340 | |
| C14 | `D/2026-07-25_vive_rm65_real_teleop_v22` | v2.2 | 310 / 68,408 | C，HL=3410 | |
| C15 | `D/2026-07-27_dmtac_rm65_bimanual_position_v1_old` | v2.3 | 34 / 16,873 | DET、Art；迁移计划原为 35，修复后为 34 | |
| C16 | `D/2026-07-27_dmtac_rm65_bimanual_position_v2_responsive` | v2.3 | 115 / 28,987 | C，独立数据 | |
| C17 | `D/2026-07-30_dmtac_rm65_bimanual_position_v1` | v2.3 | 2 / 440 | C、Prog，小样本 | |
| C18 | `D/2026-07-31_dmtac_rm65_bimanual_position_v2` | v2.3 | 22 / 5,884 | C、Prog | |
| C19 | `D/2026-07-31_dmtac_rm65_bimanual_position_writing_four_stage` | v2.3 | 3 / 2,147 | C、Prog | |
| C20 | `D/2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 5 / 2,065 | C、Prog；不是 D8 下同名 202 集数据 | |

转换族应作为整体管理：`C02-C05`、`C07-C11`、`C12-C14`。

## 4. task-organized、mask 与 writing 视图

| ID | 相对路径 | 协议 | Episode / 帧 | 关系和当前状态 | 用户备注/决策 |
|---|---|---:|---:|---|---|
| C21 | `D/masks/table_keep_mask/bimanual-cube-new-failure-case` | v2.2 | 4 / 1,044 | C，HL=24，派生 mask | |
| C22 | `D/masks/table_keep_mask/bimanual-cube-old-failure-case` | v2.2 | 13 / 2,302 | C，HL=123 | |
| C23 | `D/sorted_v22/bimanual-soft-object-old-success` | v2.2 | 201 / 39,048 | C，独立整理视图 | |
| C24 | `D/task_organized_v1/sources/cube/excluded/no_valid_task_view/bimanual-cube-old-success` | v2.2 | 150 / 31,887 | C，HL=44，无有效 task view | |
| C25 | `D/task_organized_v1/sources/cube/front_only/suboptimal/active/new_cube_front_ready_suboptimal` | v2.2 | 9 / 2,386 | C，suboptimal | |
| C26 | `D/task_organized_v1/sources/cube/front_only/success/active/new_cube_front_ready` | v2.2 | 726 / 199,935 | C，HL=10165，cube front active | |
| C27 | `D/task_organized_v1/sources/cube/front_only/success/archives/new_cube_front_ready_before_0725_backup_20260725` | v2.2 | 416 / 131,527 | C，HL=3744，历史备份 | |
| C28 | `D/task_organized_v1/sources/cube/wrist_only/suboptimal/legacy/bimanual-cube-old-failure-case` | v2.2 | 13 / 2,302 | C，HL=123 | |
| C29 | `D/task_organized_v1/sources/cube/wrist_only/suboptimal/new/bimanual-cube-new-failure-case` | legacy/待确认 | 4 / 1,044 | C，HL=24 | |
| C30 | `D/task_organized_v1/sources/cube/wrist_only/success/active/bimanual-cube-new-success` | v2.2 | 503 / 132,718 | C，cube wrist-only active | |
| C31 | `D/task_organized_v1/sources/writing/composite_sources/base_plus_0712/writing_phase1_grap_pen_before_20260724_append_backup` | v2.3 | 586 / 99,329 | C，HL=4527，writing 组合源 | |
| C32 | `D/task_organized_v1/sources/writing/composite_sources/base_plus_0724/writing_phase1_grap_pen_with_0724_backup_20260725` | v2.3 | 744 / 117,933 | C，HL=8248，writing 组合源 | |
| C33 | `D/task_organized_v1/sources/writing/front_only/active/writing_phase1_grap_pen` | v2.3 | 503 / 82,266 | C，HL=7043，active view | |
| C34 | `D/task_organized_v1/sources/writing/front_only/archives/writing_phase1_grap_pen_pre_merge_backup_20260724` | v2.3 | 503 / 82,266 | C，HL=6036，历史视图 | |

`C21-C34` 必须作为 hardlink 组审计，不能把单个路径名当作独立备份或独立占用。

## 5. DexTeleop_0803 近期数据

| ID | 相对路径 | 协议 | Episode / 帧 | 关系和当前状态 | 用户备注/决策 |
|---|---|---:|---:|---|---|
| C35 | `RealmanDMTAC/DexTeleop/remote_data/umi/2026-07-19_umi_dual_wrist_debug_remote_raw` | v2.3 | 当前缺失；计划 13 集 | canonical 和旧源均缺失 | |
| C36 | `D8/2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 202 / 86,667 | C、Prog；不同于 C20 | |
| C37 | `D8/2026-08-03_dmtac_rm65_bimanual_position_v2_writing_5_stage` | v2.3 | 108 / 57,838 | C、Prog | |
| C38 | `D8/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 355 / 142,576 | C、Prog、P0；选中 344 集 | |
| C39 | `D8/2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 261 / 102,887 | ERR、Prog、P0；选中 256 集 | |
| C40 | `D8/2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage_failure` | v2.3 | 当前缺失；计划 10 集 | canonical 缺失，旧路径为断裂软链接 | |
| C41 | `D8/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 35 / 13,057 | C、Prog、P0；选中 35 集 | |
| C42 | `D8/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_1453` | v2.3 | 206 / 80,069 | C、Prog、P0；选中 204 集 | |
| C43 | `D8/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_stable` | v2.3 | 53 / 19,691 | C、Prog、P0；选中 51 集 | |

`C39` 当前审计异常：Episode 239 左侧 `M26190034` 实际 stream 为 166,195,200 字节，
声明预期为 114,585,600 字节。当前 OpenPI 是 front-only，触觉异常不进入当前训练输入；仍必须冻结
源数据，待训练停止后独立修复。

## 6. StableCollect 历史数据

| ID | 相对路径 | 协议 | Episode / 帧 | 当前状态 | 用户备注/决策 |
|---|---|---:|---:|---|---|
| C44 | `S/2026-07-21_vive_rm65_real_teleop` | legacy/待确认 | 1 / 317 | C，小样本 | |
| C45 | `S/2026-07-22_vive_rm65_real_teleop` | legacy/待确认 | 9 / 9,319 | C | |
| C46 | `S/2026-07-31_vive_rm65_real_teleop` | legacy/待确认 | 125 / 36,833 | C | |

## 7. FormalCollection 数据

| ID | 相对路径 | 协议 | Episode / 帧 | 触觉/语义状态 | 当前关系 | 用户备注/决策 |
|---|---|---:|---:|---|---|---|
| F01 | `F/2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 129 / 46,153 | C、Prog、Art、integrity=true | P0，选中 128 集 | |
| F02 | `F/2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage__part02` | v2.3 | 202 / 67,364 | C、Prog、integrity=false | P0，选中 199 集 | |
| F03 | `F/2026-08-10_2026-08-10_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | v2.3 | 113 / 13,794 | C、Prog、Art、integrity=true | 目录名有重复日期 | |
| F04 | `F/2026-08-10_dmtac_rm65_bimanual_position_v2_writing_4_stage` | v2.3 | 53 / 7,743 | C、Prog、Art、integrity=false | 未进入当前 OpenPI profile | |
| F05 | `F/2026-08-10_dmtac_rm65_bimanual_position_v2_writing_4_stage_stable` | v2.3 | 296 / 37,657 | C、Prog、Art、integrity=true | 未进入当前 OpenPI profile | |
| F06 | `F/2026-08-10_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | v2.3 | 153 / 16,321 | RAW 约 17.91 GiB、Prog、Art、integrity=true | 原子压缩候选 | |
| F07 | `F/2026-08-11_dmtac_rm65_bimanual_position_v2_writing_interrupt_choice_stage1` | v2.3 | 201 / 37,069 | C、Prog、Art、integrity=true | 新长程任务 | |
| F08 | `F/2026-08-11_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | v2.3 | 251 / 29,620 | C、Prog、Art、integrity=true | 新长程任务 | |

## 8. canonical 根外残留

| ID | 路径 | Episode | 当前判断 | 用户备注/决策 |
|---|---|---:|---|---|
| R01 | `L/2026-07-23_debug_dmtac_rm65_position_state_only` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R02 | `L/2026-07-25_dmtac_rm65_bimanual_position_v1` | catalog 0 | 非空：残留未提交的 `episode_000000`，含触觉/视频/深度约 824 MB；保留待审计 | `REPAIR/QUAR 待定` |
| R03 | `L/2026-07-26_dmtac_rm65_bimanual_position_v1` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R04 | `L/2026-07-31_dmtac_impedance_15_40hz` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R05 | `L/2026-07-31_dmtac_rm65_bimanual_position_v1` | 77 | Episode 6 缺 front 视频；需修复或隔离 | |
| R06 | `L/2026-08-03_dmtac_rm65_bimanual_position_v2_balanced` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R07 | `L/2026-08-03_dmtac_rm65_bimanual_position_v2_boundary_hold` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R08 | `L/2026-08-03_dmtac_rm65_bimanual_position_v3_vive_velocity_debug` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R09 | `L/2026-08-03_rm65_v2_boundary_hold_motion_audit` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R10 | `L/2026-08-04_dmtac_rm65_bimanual_position_v2_balanced` | 0 | 已验证为空并于 2026-08-12 删除 | `REMOVED` |
| R11 | `L/masks/debug/writing_sample_deskfix` | 503 | debug/派生数据，未通过迁移完整性检查 | |
| R12 | `L/masks/table_keep_mask/bimanual-cube-new-success` | 503 | 派生 mask，未通过迁移完整性检查 | |
| R13 | `L/task_organized_v1/.../new_cube_front_ready.invalid-hardlink-write-through-20260730` | 726 | 明确 invalid，必须隔离 | |
| R14 | `L/task_organized_v1/.../writing_phase1_grap_pen.invalid-hardlink-write-through-20260730` | 503 | 明确 invalid，必须隔离 | |

不完整目录：

```text
/home/user/zc/DexTeleop_0803/datasets/
2026-08-03_dmtac_rm65_bimanual_position_v2_writing_5_stage_new
```

该目录没有 `meta/info.json`，不能视为有效数据集，也不能与 C37 自动合并。

## 9. 当前 OpenPI 保护集

当前训练 profile：

```text
writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2
```

保护集：

```text
C38 C39 C41 C42 C43 F01 F02
```

当前训练结束并封存 workspace 前，不得移动、重命名、修复或重新生成这些 source 的 identity、catalog、
numeric/video 文件或 Prepared binding。触觉不在当前 front-only 输入中，但源根仍保持冻结。

## 10. 用户整理选项

| 代码 | 含义 |
|---|---|
| `FZ` | 当前训练/部署使用，冻结 |
| `KEEP` | 作为独立 canonical 数据集长期保留 |
| `SHARD` | 同任务独立分片，由 profile 逻辑组合，不物理合并 |
| `GROUP` | 属于同一 hardlink/转换族，整体管理 |
| `ARCH` | 退出日常训练，保留 canonical 与独立备份 |
| `REPAIR` | 修复或补全后再决定 |
| `QUAR` | 隔离，不进入默认训练 |
| `EMPTY` | 零 Episode 空壳 |
| `DROP?` | 候选移除，仍需 profile、inode、link count 与备份证明 |
| `NAME=...` | 正式任务名 |
| `NOTE=...` | 用途、质量、失败原因或数据关系 |

推荐使用 `SHARD` 组织同任务多日期/多 part 数据，不做物理合并，以保持 Episode identity 和回退能力。

## 11. 本轮操作回执

本节只记录已经完成并重新验证的操作；不能提前填写计划结果。

| 操作 | 当前状态 | 结果/回执 |
|---|---|---|
| 建立清单 | 完成 | 本文件 |
| 删除 R01、R03-R04、R06-R10 零 Episode 空壳 | 完成 | 8 个目录均在删除前复核 metadata/catalog/payload/link/open-fd，删除后全部不存在；R02 非空未动 |
| C12 NAS 触觉拆出 | 完成 | `DETACHED_VERIFIED`；inventory `2be6d257ea5e4f26ec9d668896f8efbb67fee167187b4b918833f83ed6b1f9d1`；export、restore-drill、retire 三类 receipt 完整；释放 64,230,998,021 字节（约 59.82 GiB）；hot 仅保留 manifest，NAS payload 可恢复 |
| C02-C04 NAS 触觉拆出 | 完成 | 三者均为 `DETACHED_VERIFIED`；inventory 依次为 `8f91a55cfede3a8268f5cfca7de61daf86c7c29c923bf0db0aacce88eca31025`、`23cdd03f0796b3766d4519a548efbb881b87363b34d597995f9810d96c8778b1`、`3d19d2c15c59f3fbdf60f4f5fb10252c250cdeff29d8174b04d243ab18e049ed`；每个均有 export、restore-drill、retire receipt，无 pending journal |
| 继续 NAS 触觉同步 | 进行中 | 自动续传链正在处理 C05；已进入 payload 导出阶段，hot 数据仍完整且尚未越过退役边界 |
| 退役已由 NAS 独立恢复凭据覆盖的其余本地触觉 | 进行中 | 仅允许自动链在逐 inventory/hash/reader/restore-drill 全部通过后，按 receipt 覆盖范围原子退役；FormalCollection 与 OpenPI 保护集保持隔离 |
