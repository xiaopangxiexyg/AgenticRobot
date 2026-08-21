# DexTeleop 数据集系统盘点（2026-08-20 只读快照）

> 本文是一次只读盘点结果，用于后续审阅、备注和合并规划。
> 本次没有移动、改名、合并、删除数据，也没有修改历史采集配置。
> 远端证据来自明确主机 `user@192.168.70.116`，远端项目根为
> `/home/user/zc/DexTeleop`，数据根为 `/home/user/zc/datasets/RealmanDMTAC`。

## 1. 盘点范围与判定原则

本次将路径分成四类，避免把容器、派生视图、审计目录误当成可合并的原始
dataset episode：

1. **Native/source roots（55 个）**：包含 `meta/info.json`、episode/task
   数据，是后续按 profile、质量和 OpenPI 使用关系整理的主要对象。
2. **Valid camera-only views（17 个）**：相机语义纠正产生的新 inode 视图，
   明确不含 tactile；只能和同等 camera-only 语义的视图组成逻辑训练输入，不能
   直接覆盖或混入 native/tactile source。
3. **Invalid audit outputs（2 个）**：保留作审计，严格 loader 必须拒绝，不能
   进入训练或合并候选。
4. **Repair projection（1 个）**：独立的相机修复投影；不是 native source，
   需要通过 verifier 后才能作为派生输入。

物理存储格式与语义版本必须分开看：远端这些目录的 `meta/info.json` 均报告
`lerobot_v2.1_style`。目录名中的 `_v22`、`_v23`、`v2_writing_4_stage`
是 loader/profile/语义层名称，不能直接当作磁盘格式版本。

标记说明：

- `P1-1`：一阶段线性 profile，hash `9af03398...f361`。
- `P1-4`：四阶段 profile v1，hash `301b9734...e51db5`。
- `P2-4`：四阶段 profile v2，progress hash `0874f6b9...f3a438`，stage
  hash `b3e3d19d...c3934b`。
- `P2-5`：五阶段 profile v2，progress hash `3222716d...c5da9`。
- `无进度`：没有可验证的长程 progress profile；不应强行补成某个 profile。
- `进度不完整`：存在 progress 目录或标记，但缺少完整 profile/manifest。
- `T-local`：当前目录可见本地 tactile payload。
- `T-ext`：tactile 通过 external/native bundle 引用；本地没有 payload 不等于
  没有 tactile，需要以 attachment/retirement receipt 复核。
- `T-none?`：盘点标记中未看到本地 payload 或完整 external receipt，合并前必须
  重新核实。
- 所有大小是只读盘点时的近似 `du` 值，不能作为迁移或删除的唯一依据。

## 2. 顶层容器和别名关系

### 2.1 Canonical 容器

```text
/home/user/zc/datasets/RealmanDMTAC/
├── DexTeleop/
├── DexTeleop_0803/
├── DexTeleop_StableCollect_0705/
├── FormalCollection/
├── meta/                         # 容器级元数据，不是 episode dataset
├── .migration/                   # 迁移审计/事务状态，不是 dataset
├── .tactile_export_runs/         # tactile 导出运行记录，不是 dataset
└── README_HARDLINKS.md           # link/迁移约束说明
```

`FormalCollection/meta/progress` 是容器级 progress 元数据；
`FormalCollection/tactile/datasets` 是外置 native tactile 容器，当前可见
UUID 为 `e25d726e-bef0-4414-ba14-a3a2b6d7b14e` 和
`ff178d7a-d91c-4a20-83ba-2e61ea5865e6`。`68d9f362-a864-4970-aec5-56ce882050ca`
（2026-08-19 数据）此前已备份到 NAS 并从本地移除，不能因本地缺少
`tactile/` 就当作无 tactile。

以下目录是派生/审计容器，不是 native 合并输入：

- `DexTeleop/camera_semantic_prepared_views/`
- `DexTeleop/camera_semantic_repairs/`
- `DexTeleop/camera_semantic_retired_sources/`
- `DexTeleop_0803/camera_semantic_full_materialized/`
- `DexTeleop_0803/camera_semantic_retired_sources/`

根目录 `/home/user/zc/datasets` 还有一个兼容别名：

```text
/home/user/zc/datasets/2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage
  -> /home/user/zc/datasets/RealmanDMTAC/FormalCollection/
     2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage__part02
```

### 2.2 hardlink/旧路径风险

`RealmanDMTAC/README_HARDLINKS.md` 记录了 2026-08-06 迁移：46 个 dataset、
10,135 个 episode、156,017 个文件路径，其中 95,621 个路径
`st_nlink > 1`。README 将旧路径描述为兼容软链接，但本次只读 `lstat` 看到：

```text
/home/user/zc/DexTeleop                 mode=directory inode=40766467 nlink=44
/home/user/zc/DexTeleop_0803            mode=directory inode=20987115 nlink=41
/home/user/zc/DexTeleop_StableCollect_0705 mode=directory inode=28707154 nlink=21
/home/user/zc/datasets/RealmanDMTAC      mode=directory inode=20987264 nlink=9
```

代表性 `meta/info.json` 采样显示部分旧路径和 canonical 路径的 device/inode
相同，但单文件 `nlink=1`；这与“目录是软链接”的文档描述并不完全一致，可能
涉及 bind/overlay/迁移后的文件系统行为。**在全文件级 inode/link-count 审计
完成前，不把旧路径当作独立副本，也不把它们当作可以直接删除的目录。**任何
后续合并都必须使用新 inode、独立 backup receipt、逐文件 manifest 和原子发布。

## 3. Native/source roots（55 个）

### 3.1 `RealmanDMTAC/DexTeleop/datasets`（34 个）

| # | 相对路径 | 规模（约） | episode/帧 | 物理/触觉 | progress/profile | 当前整理备注 |
|---:|---|---:|---:|---|---|---|
| 1 | `2026-07-23_dmtac_rm65_bimanual_position_v1` | 220M | 2 / 326 | external, `T-local`, none | `P1-1` | 小型一阶段源；有 `meta/tactile_recompression`；可作为协议/loader fixture，不与生产大集合直接混合 |
| 2 | `2026-07-23_vive_rm65_real_teleop_old_version` | 7.4G | 515 / 84650 | external, lz4, `T-local` | 无进度 | C05 原始源；515 中仅 13 条相机完整，不能作为完整视觉源 |
| 3 | `2026-07-23_vive_rm65_real_teleop_old_version_v22` | 7.4G | 515 / 84650 | external, lz4, `T-local` | 无进度 | C05 语义 v22 变体；不要因名称自动合并 |
| 4 | `2026-07-23_vive_rm65_real_teleop_old_version_v22_measured_state` | 7.4G | 515 / 84650 | external, lz4, `T-local` | 无进度 | C05 measured-state 变体 |
| 5 | `2026-07-23_vive_rm65_real_teleop_old_version_v23` | 7.5G | 515 / 84650 | external, lz4, `T-local` | 无进度 | C05 v23 变体；仅保留 13 episode 的 camera-only 视图有效 |
| 6 | `2026-07-24_dmtac_rm65_bimanual_position_v1` | 37M | 1 / 55 | external, `T-local`, none | 无进度 | 单 episode 小源；有 recompression 标记 |
| 7 | `2026-07-24_vive_rm65_real_teleop` | 36G | 242 / 35832 | external, lz4, `T-local` | 无进度 | 主源；另有 filtered/pose-consistent 变体，需按 episode membership 区分 |
| 8 | `2026-07-24_vive_rm65_real_teleop_filtered_no184` | 36G | 241 / 35667 | external, lz4, `T-local` | 无进度 | 从 #7 排除 episode 184 的派生/筛选源 |
| 9 | `2026-07-24_vive_rm65_real_teleop_v22` | 36G | 241 / 35667 | external, lz4, `T-local` | 无进度 | v22 筛选变体 |
| 10 | `2026-07-24_vive_rm65_real_teleop_v22_pose_consistent` | 36G | 241 / 35667 | external, lz4, `T-local` | 无进度 | pose-consistent 筛选变体 |
| 11 | `2026-07-24_vive_rm65_real_teleop_v23` | 36G | 241 / 35667 | external, lz4, `T-local` | 无进度 | v23 筛选变体；有对应有效 camera-only 视图 |
| 12 | `2026-07-25_vive_rm65_real_teleop` | 5.9G | 311 / 68471 | external, lz4, `T-local` | 无进度 | C12 原始源；仅 episode 283 相机完整 |
| 13 | `2026-07-25_vive_rm65_real_teleop_filtered_no283` | 5.9G | 310 / 68408 | external, lz4, `T-local` | 无进度 | C12 筛选源；camera-only 输出为 0 episode，不可视为有效视觉集合 |
| 14 | `2026-07-25_vive_rm65_real_teleop_v22` | 7.1G | 310 / 68408 | external, lz4, `T-local` | 无进度 | C12 v22 变体 |
| 15 | `2026-07-27_dmtac_rm65_bimanual_position_v1_old` | 1.9G | 34 / 16873 | external, `T-none?` | 无进度 | C15 历史源；左右腕 68 段、33,746 帧已验证，可用 camera repair 输出 |
| 16 | `2026-07-27_dmtac_rm65_bimanual_position_v2_responsive` | 19G | 115 / 28987 | external, `T-local`, none | 无进度 | responsive 源；有 recompression 标记 |
| 17 | `2026-07-30_dmtac_rm65_bimanual_position_v1` | 285M | 2 / 440 | external, `T-local`, none | `P1-1` | 与本地 fixture 同类；有完整一阶段 profile |
| 18 | `2026-07-31_dmtac_rm65_bimanual_position_v2` | 3.8G | 22 / 5884 | external, `T-local`, none | 进度不完整 | 有 progress 目录但缺 `profile.cjson`；禁止自动归入 P2 |
| 19 | `2026-07-31_dmtac_rm65_bimanual_position_writing_four_stage` | 1.5G | 3 / 2147 | external, `T-local`, none | `P1-4` | 四阶段 profile v1；与 P2-4 隔离 |
| 20 | `2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 1.4G | 5 / 2065 | external, `T-local`, none | `P2-4` | P2 四阶段源；progress/stage hash 均完整 |
| 21 | `masks/table_keep_mask/bimanual-cube-new-failure-case` | 111M | 4 / 1044 | external, lz4, `T-local` | 无进度 | failure mask 集；不与成功集合合并 |
| 22 | `masks/table_keep_mask/bimanual-cube-old-failure-case` | 810M | 13 / 2302 | external, unknown, `T-local` | 无进度 | old failure mask 集 |
| 23 | `sorted_v22/bimanual-soft-object-old-success` | 4.9G | 201 / 39048 | external, lz4, `T-local` | 无进度 | old-success；需保持 v22 归一化语义 |
| 24 | `task_organized_v1/sources/cube/excluded/no_valid_task_view/bimanual-cube-old-success` | 14G | 150 / 31887 | external, unknown, `T-local` | 无进度 | 明确 excluded/no-valid-task-view；默认训练排除 |
| 25 | `task_organized_v1/sources/cube/front_only/suboptimal/active/new_cube_front_ready_suboptimal` | 307M | 9 / 2386 | external, lz4, `T-local` | 无进度 | active 但 suboptimal；不与 success 直接混合 |
| 26 | `task_organized_v1/sources/cube/front_only/success/active/new_cube_front_ready` | 21G | 726 / 199935 | external, lz4, `T-local` | 无进度 | OpenPI 0715 主要 active 源；保持现有 profile/index 不变 |
| 27 | `task_organized_v1/sources/cube/front_only/success/archives/new_cube_front_ready_before_0725_backup_20260725` | 14G | 416 / 131527 | external, lz4, `T-local` | 无进度 | 0715 archive/backup；不能和 active 源按文件夹名覆盖 |
| 28 | `task_organized_v1/sources/cube/wrist_only/suboptimal/legacy/bimanual-cube-old-failure-case` | 857M | 13 / 2302 | external, unknown, `T-local` | 无进度 | legacy failure；保留历史语义 |
| 29 | `task_organized_v1/sources/cube/wrist_only/suboptimal/new/bimanual-cube-new-failure-case` | 135M | 4 / 1044 | external, lz4, `T-local` | 无进度 | new failure；需和 #21 对照 episode identity |
| 30 | `task_organized_v1/sources/cube/wrist_only/success/active/bimanual-cube-new-success` | 18G | 503 / 132718 | external, lz4, `T-local` | 无进度 | OpenPI 0712 wrist-cube 主要源；保持现有 profile/index |
| 31 | `task_organized_v1/sources/writing/composite_sources/base_plus_0712/writing_phase1_grap_pen_before_20260724_append_backup` | 11G | 586 / 99329 | external, lz4, `T-local` | 无进度 | writing phase1 composite/archive |
| 32 | `task_organized_v1/sources/writing/composite_sources/base_plus_0724/writing_phase1_grap_pen_with_0724_backup_20260725` | 14G | 744 / 117933 | external, lz4, `T-local` | 无进度 | writing phase1 composite/archive |
| 33 | `task_organized_v1/sources/writing/front_only/active/writing_phase1_grap_pen` | 8.6G | 503 / 82266 | external, lz4, `T-local` | 无进度 | OpenPI writing phase1 source；保持 active 路径和 profile |
| 34 | `task_organized_v1/sources/writing/front_only/archives/writing_phase1_grap_pen_pre_merge_backup_20260724` | 8.6G | 503 / 82266 | external, lz4, `T-local` | 无进度 | writing phase1 archive；不和 active 物理合并 |

### 3.2 `RealmanDMTAC/DexTeleop_0803/datasets`（7 个）

| # | 相对路径 | 规模（约） | episode/帧 | 物理/触觉 | progress/profile | 当前整理备注 |
|---:|---|---:|---:|---|---|---|
| 1 | `2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 9.9G | 202 / 86667 | external, `T-none?` | `P2-4` | 有 progress/recompression/videos/data 标记；需按 attachment receipt 复核 tactile |
| 2 | `2026-08-03_dmtac_rm65_bimanual_position_v2_writing_5_stage` | 6.7G | 108 / 57838 | external, `T-local`, none | `P2-5` | 五阶段，不能与四阶段合并 |
| 3 | `2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 92G | 355 / 142576 | external, `T-local`, none | `P2-4` | OpenPI writing stage profile 保护源 |
| 4 | `2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 128G | 261 / 102887 | external, `T-none?` | `P2-4` | OpenPI writing stage profile 保护源；需复核外置 tactile |
| 5 | `2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 8.6G | 35 / 13057 | external, `T-local`, none | `P2-4` | OpenPI 保护源；不要改变现有 eligibility |
| 6 | `2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_1453` | 51G | 206 / 80069 | external, `T-local`, none | `P2-4` | OpenPI 保护源；不要改变现有 eligibility |
| 7 | `2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_stable` | 13G | 53 / 19691 | external, `T-local`, none | `P2-4` | OpenPI 保护源；不要改变现有 eligibility |

### 3.3 `RealmanDMTAC/DexTeleop_StableCollect_0705/datasets`（3 个）

| # | 相对路径 | 规模（约） | episode/帧 | 物理/触觉 | progress/profile | 当前整理备注 |
|---:|---|---:|---:|---|---|---|
| 1 | `2026-07-21_vive_rm65_real_teleop` | 321M | 1 / 317 | external, lz4, `T-local` | 无进度 | StableCollect 历史源 |
| 2 | `2026-07-22_vive_rm65_real_teleop` | 9.2G | 9 / 9319 | external, lz4, `T-local` | 无进度 | StableCollect 历史源 |
| 3 | `2026-07-31_vive_rm65_real_teleop` | 37G | 125 / 36833 | external, lz4, `T-local` | 无进度 | StableCollect 历史源；相机视图已另行生成 |

### 3.4 `RealmanDMTAC/FormalCollection`（11 个）

| # | 相对路径 | 规模（约） | episode/帧 | 物理/触觉 | progress/profile | 当前整理备注 |
|---:|---|---:|---:|---|---|---|
| 1 | `2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 5.4G | 129 / 46153 | external, `T-local`, none | `P2-4` | FormalCollection 第一部分 |
| 2 | `2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage__part02` | 8.1G | 202 / 67364 | external, `T-local`, none | `P2-4` | 根目录兼容 symlink 指向此目录；不能把 alias 计为第二份 |
| 3 | `2026-08-10_2026-08-10_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | 1.7G | 113 / 13794 | external, `T-local`, none | `P2-4` | 名称有重复日期/拼写；合并前必须确认 task/profile identity |
| 4 | `2026-08-10_dmtac_rm65_bimanual_position_v2_writing_4_stage` | 974M | 53 / 7743 | external, `T-local`, none | `P2-4` | FormalCollection 四阶段 |
| 5 | `2026-08-10_dmtac_rm65_bimanual_position_v2_writing_4_stage_stable` | 4.7G | 296 / 37657 | external, `T-local`, none | `P2-4` | task id 含 `_stable`，不能只按日期合并 |
| 6 | `2026-08-10_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | 2.1G | 153 / 16321 | external, `T-local`, none | `P2-4` | 与 #3 同名但不同目录/规模，必须依赖 dataset identity 区分 |
| 7 | `2026-08-11_dmtac_rm65_bimanual_position_v2_writing_interrupt_choice_stage1` | 4.5G | 201 / 37069 | external, `T-local`, none | `P2-4` | interrupt-choice 场景；后续需按语义标签决定是否进入主集合 |
| 8 | `2026-08-11_dmtac_rm65_bimanual_position_v2_writing_multi_choise_stage1` | 3.7G | 251 / 29620 | external, `T-local`, none | `P2-4` | multi-choice 场景；需和同名 8/10 数据按 identity 分开 |
| 9 | `2026-08-12_dmtac_rm65_bimanual_60hz_admittance_left_gripper_offline_test` | 56K | 0 / 0 | 无 tactile | 无 profile | 空 scratch/test；不属于可合并训练数据 |
| 10 | `2026-08-19_4_stage_writing_detail` | 3.9G | 159 / 32042 | external, `T-ext` | `P2-4 hash only / task mismatch` | `info.task_id=4_stage_writing_detail`；tactile UUID `68d9f362-a864-4970-aec5-56ce882050ca` 已在 NAS；本地只留 attachment metadata |
| 11 | `2026-08-20_4_stage_writing_detail` | 约 928M（需重新 `du`） | 168 / 26249（早期摘要曾记录 46/7662，已过时） | external, `T-ext` | `P2-4 hash only / task mismatch` | `info.task_id=dmtac_rm65_bimanual_position_v1`；tactile UUID `e25d726e-bef0-4414-ba14-a3a2b6d7b14e` 当前在 FormalCollection/tactile |

## 4. Valid camera-only prepared views（17 个）

这些目录是新 inode 的相机语义投影，触觉 payload、触觉字段和 sidecar metadata
已经剔除。它们的用途是视觉训练/纠正后的 camera-only 读取，不能冒充保留
tactile 的 native source。

| # | 相对路径 | 规模（约） | episode/帧 | profile/状态 | 备注 |
|---:|---|---:|---:|---|---|
| 1 | `DexTeleop/camera_semantic_prepared_views/2026-07-23_dmtac_rm65_bimanual_position_v1_camera_only_v3` | 38M | 2 / 326 | `P1-1`, valid | 最终 v3；旧 v2 审计目录不合格 |
| 2 | `DexTeleop/camera_semantic_prepared_views/2026-07-23_vive_rm65_real_teleop_old_version_v23.camera_only_v1` | 307M | 13 / 2581 | valid | C05 仅完整 13 episode |
| 3 | `DexTeleop/camera_semantic_prepared_views/2026-07-24_dmtac_rm65_bimanual_position_v1_camera_only_v1` | 5.8M | 1 / 55 | 无 profile, valid | 单 episode camera-only |
| 4 | `DexTeleop/camera_semantic_prepared_views/2026-07-24_vive_rm65_real_teleop_v23_camera_only_v1` | 4.2G | 241 / 35667 | valid | C07/C08 语义视图 |
| 5 | `DexTeleop/camera_semantic_prepared_views/2026-07-25_vive_rm65_real_teleop_v22.camera_only_v1` | 328K | 0 / 0 | unusable | 空输出，不能合并 |
| 6 | `DexTeleop/camera_semantic_prepared_views/2026-07-27_dmtac_rm65_bimanual_position_v2_responsive_camera_only_v1` | 3.1G | 115 / 28987 | valid | responsive camera-only |
| 7 | `DexTeleop/camera_semantic_prepared_views/2026-07-30_dmtac_rm65_bimanual_position_v1_camera_only_v1` | 44M | 2 / 440 | `P1-1`, valid | native v1 的视觉投影 |
| 8 | `DexTeleop/camera_semantic_prepared_views/2026-07-31_dmtac_rm65_bimanual_position_v2_camera_only_v1` | 665M | 22 / 5884 | 进度不完整 | 不自动并入 P2 |
| 9 | `DexTeleop/camera_semantic_prepared_views/2026-07-31_dmtac_rm65_bimanual_position_writing_four_stage_camera_only_v1` | 269M | 3 / 2147 | `P1-4`, valid | P1 四阶段视觉投影 |
| 10 | `DexTeleop/camera_semantic_prepared_views/2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage_camera_only_v1` | 241M | 5 / 2065 | `P2-4`, valid | DexTeleop 小源视觉投影 |
| 11 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_0803__2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage.camera_only_v1` | 9.9G | 202 / 86667 | `P2-4`, valid | 不含 tactile |
| 12 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_0803__2026-08-03_dmtac_rm65_bimanual_position_v2_writing_5_stage.camera_only_v1` | 6.6G | 108 / 57838 | `P2-5`, valid | 五阶段视觉投影 |
| 13 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_0803__2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage.camera_only_v1` | 17G | 355 / 142576 | `P2-4`, valid | OpenPI 保护源的视觉视图 |
| 14 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_0803__2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage.camera_only_v1` | 12G | 261 / 102887 | `P2-4`, valid | OpenPI 保护源的视觉视图 |
| 15 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_StableCollect_0705__2026-07-21_vive_rm65_real_teleop.camera_only_v1` | 37M | 1 / 317 | valid | StableCollect 视觉视图 |
| 16 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_StableCollect_0705__2026-07-22_vive_rm65_real_teleop.camera_only_v1` | 1.1G | 9 / 9319 | valid | StableCollect 视觉视图 |
| 17 | `DexTeleop/camera_semantic_prepared_views/DexTeleop_StableCollect_0705__2026-07-31_vive_rm65_real_teleop.camera_only_v1` | 4.6G | 125 / 36833 | valid | StableCollect 视觉视图 |

## 5. Invalid audit outputs 与 repair projection

### 5.1 必须隔离、严格 loader 拒绝的目录

| 相对路径 | 规模 | 原因 |
|---|---:|---|
| `DexTeleop/camera_semantic_prepared_views/_invalid_audit/2026-07-23_dmtac_rm65_bimanual_position_v1.invalid-a9654b5e5ce9` | 38M | 残留 `meta/tactile_recompression`，不是纯 camera-only 结果 |
| `DexTeleop/camera_semantic_prepared_views/_invalid_audit/2026-07-23_dmtac_rm65_bimanual_position_v1_camera_only_v2.invalid-2e97ae6ead69` | 38M | 首次不合格投影，保留作审计，严格 loader 拒绝 |

这两个目录不能出现在任何默认训练 glob、OpenPI profile 或“可合并 native
源”列表中。后续应继续保持 quarantine/invalid 命名和 loader fail-closed。

### 5.2 独立修复投影

`DexTeleop/camera_semantic_repairs/2026-07-27_dmtac_rm65_bimanual_position_v1_old`
约 1.8G，34 episode / 16,873 帧。它是 C15 的相机语义修复投影，经过 fresh
projection、完整解码和新 inode 验证后才可使用；它不是 native source，也不应
通过文件夹同名覆盖原始目录。

### 5.3 已知相机事故

- C05（2026-07-23 vive）：515 episode 中只有 13 条完整，502 条腕部视频受
  2026-07-30 hardlink write-through 事故影响；已知备份共享损坏 inode，不能当作
  独立 RGB 恢复副本。
- C12（2026-07-25 vive）：311 episode 中只有 episode 283 完整；
  `filtered_no283` 没有可用 camera-only episode。
- C15（2026-07-27 old）：34 episode、68 段腕部视频、33,746 解码帧已验证，
  repair projection 可作为独立相机输入。

详细记录：
`deployments/openpi/DATASET_MEDIA_INCIDENT_20260730.md` 和
`utils/dataset_postprocess/README.md`。

## 6. 本地仓库 fixtures（不属于远端生产集）

| 路径 | 规模 | 内容 | 用途 |
|---|---:|---|---|
| `datasets/2026-07-30_dmtac_rm65_bimanual_position_v1` | 285M | 2 episode / 440 帧，`lerobot_v2.1_style`，`P1-1`，本地 tactile | regression/协议 fixture；不要与远端同名目录直接合并 |
| `datasets/mock_2026-07-31_refinement_annotation_v1` | 1.1M | 4 个 mock episode，refinement annotation v1 | synthetic contract 测试；禁止用于机器人策略训练 |
| `datasets/mock_2026-07-31_refinement_annotation_v2` | 1.1M | 同上，增加 Hold wall-time 语义说明 | synthetic contract 测试；禁止用于机器人策略训练 |

本地环境没有 `/home/user/zc` 的生产目录；本清单中的生产证据均来自明确的
远端只读盘点。

## 7. Profile 与 OpenPI 保护关系

### 7.1 已发现的 profile

| 标签 | profile id | 版本/拓扑 | progress hash | stage hash |
|---|---|---|---|---|
| `P1-1` | `dmtac_rm65_bimanual_position_v1_progress` | v1 / LINEAR | `9af03398ac96f1c0f2a5627ac9d68683c4f0c6192442dedfebe4b0c0f779f361` | — |
| `P1-4` | `dmtac_rm65_bimanual_position_writing_four_stage_progress` | v1 / LINEAR | `301b9734d88e88ed4e09511a24d338a2074aa509683ac76f3e0ff40341e51db5` | — |
| `P2-4` | `dmtac_rm65_bimanual_position_writing_four_stage_progress` | v2 / LINEAR | `0874f6b97b3b5255240dd34e6cfac01c76a95c1077df8a20af6b9b3119f3a438` | `b3e3d19daefa9dbf9fe870c769c062c72b24e1442cfb400d52fae9664dc3934b` |
| `P2-5` | `dmtac_rm65_bimanual_position_writing_five_stage_progress` | v2 / LINEAR | `3222716daa948a0939ba735b4d2d1d3ef2946ef6fe0837dedf90c00c438c5da9` | 对应 five-stage stages profile |

P2-4 的四个 coarse step 为：
`right_acquire_brush`、`left_remove_brush_cap`、`right_write`、
`right_return_brush`。其 stage profile 包含 remove/return、approach/grasp/
lift_and_present、align/release、write 等阶段。P2-5 额外包含
`right_dip_ink`，不能按四阶段语义处理。

### 7.2 OpenPI 当前保护的写字四阶段来源

远端 OpenPI 配置：
`/home/user/zc/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_history_state_v1.json`

当前保护的 5 个 source root：

```text
DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage
DexTeleop_0803/datasets/2026-08-05_dmtac_rm65_bimanual_position_v2_writing_4_stage
DexTeleop_0803/datasets/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage
DexTeleop_0803/datasets/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_1453
DexTeleop_0803/datasets/2026-08-06_dmtac_rm65_bimanual_position_v2_writing_4_stage_stable
```

配置期望 `P2-4` progress/stage hash、task
`dmtac_rm65_bimanual_position_v2_writing_4_stage`、front camera、h16、
absolute、32D history-state；配置内的 eligible episode 数为 344、246、35、
203、51（合计 879），与当前目录的物理 episode 计数不同，这是训练筛选而非
目录损坏。**任何未来合并必须先保留这些 source path 或以原子方式更新 profile
和 path manifest；不能直接重写目录。**

OpenPI 其它主要关系：

- `0715-cube-front-ready-mix-4to1-relative` 使用
  `new_cube_front_ready` 与 `bimanual-cube-new-success`，不能用 writing 源替换。
- `0712-cube_11768` / `0712-wrist-cube` 使用 cube/wrist-cube 源。
- `dual_rm65_dmgripper_latest`、`dual_rm65_dmgripper_v22_masked_150k` 是策略/动作
  注册项，不等于某一个物理 dataset 根。

## 8. 后续合并建议（本轮不执行）

### 8.1 可以形成的逻辑合并组

**M1：P2-4 四阶段主组（候选，不是立即合并）**

候选包括 DexTeleop/0803/FormalCollection 中 progress hash、stage hash、task
和 action schema 完全一致的目录。合并前还必须逐 episode 验证：

1. dataset identity、episode identity 和时间范围不重复；
2. profile hash、stage hash、task/action schema 完全一致；
3. tactile 是 `T-local`、`T-ext` 还是已退休必须有完整 receipt；
4. camera role/normalization 与目标训练 profile 一致；
5. failure/annotation/quality exclusion 已解析；
6. 不改变 OpenPI 当前 8/4–8/6 保护集合的实际 episode 选择。

优先生成**逻辑 manifest/index**，而不是物理拼接。只有在完整 inode、hash、
episode identity 和独立备份审查完成后，才考虑 fresh-inode derived dataset。

**M2：P2-5 五阶段组**

单独维护；不能与 M1 四阶段混合。若训练只需要视觉，可使用对应 valid
camera-only view，但 profile 仍必须是 P2-5。

**M3：P1-4 四阶段 v1 组**

保留 v1 hash；不能因为显示名称相同就加入 P2-4。

**M4：P1-1 一阶段组**

用于早期一阶段/协议 fixture；不能和长程四/五阶段组合成一个 profile 混装集。

**M5：cube/writing legacy 与 active/archive 组**

当前 `active`、`archives`、`failure`、`suboptimal` 已承载训练选择语义。保持
现有 OpenPI profile/index，优先通过 logical selector 合并，不对文件夹做覆盖式
重命名。

**M6：camera-only 组**

只能在 camera-only profile、camera role registry、frame validity 和 episode
membership 完全一致时建立视觉逻辑视图；不和 native/tactile source 混装。

FormalCollection 的 8/19、8/20 都携带 P2-4 的 progress hash，但 task identity
并不等于当前 full-task v2：8/19 是 `4_stage_writing_detail`，8/20 是
`dmtac_rm65_bimanual_position_v1`。因此它们目前只能作为 detail 独立 selector；
8/19 tactile 已在 NAS、8/20 tactile 仍在 FormalCollection 外置容器。不能仅凭
progress hash 将其加入 P2-4 full-task，也不能在没有 task correction、attachment
receipt、annotation 和 identity 去重前物理合并。

### 8.2 “合并”的四种不同含义

后续执行时不能把下面四种操作都叫“merge”，否则很容易把派生副本重复计入
训练，或误删仍被 OpenPI 引用的 source：

| 类型 | 含义 | 是否增加 episode | 是否改变原目录 | 当前建议 |
|---|---|---:|---:|---|
| `D0_ALIAS_DEDUP` | 两个路径只是同一 symlink/alias/同一 dataset identity | 否 | 否 | 只在 index 中保留一个 canonical path；2026-08-07 根目录 alias 属于此类 |
| `D1_LINEAGE_DEDUP` | 同一采集源的 `_v22`、`_v23`、filtered、pose-consistent 或 camera projection | 否 | 否 | 保留 parent + derived lineage；训练 selector 选择一个视图，不把变体相加 |
| `D2_LOGICAL_APPEND` | 不同采集 session、episode identity 不重叠，但 profile/schema/modality 兼容 | 是 | 否 | 首选方式；生成只读 logical manifest/index，源目录保持原样 |
| `D3_MATERIALIZED_MERGE` | 将 D2 结果写入一个新物理目录 | 是 | 不覆盖源 | 只有 D2 manifest、inode、backup、视频/Parquet/tactile 校验全部通过后才考虑 |

因此，“同一日期的相同采集数据合并”在本计划中默认指 **D2 logical append**，
不是把几个目录直接 `cp` 到同一个目录，也不是删除原始来源。对于同一 session
的变体，优先判为 D1；对于同一天分多次正式采集、profile 和 schema 完全一致的
session，才判为 D2。

### 8.3 同日期/同采集主题的逐组合并建议

下表是当前最具体的执行建议。`候选 episode 数` 是物理计数的算术和，只能作为
预估；最终 manifest 必须按 dataset UUID、episode identity、时间戳和 content
hash 去重后再确定实际数量。

| 日期/组 | 涉及目录 | 关系判定 | 建议的合并目标 | 当前动作 |
|---|---|---|---|---|
| 2026-07-23 C05 vive | `2026-07-23_vive_rm65_real_teleop_old_version`、`_v22`、`_v22_measured_state`、`_v23` | 同一 C05 采集的语义/状态变体，D1；不是四份独立数据 | `lineage/C05`，只保留一个 parent identity；视觉训练使用已验证的 13-episode `old_version_v23.camera_only_v1` | 不做 D2；502 条腕视频损坏，禁止把四个 515-episode 目录相加 |
| 2026-07-23 DMTAC | `2026-07-23_dmtac_rm65_bimanual_position_v1` | 与 C05 不同采集协议，虽同日但不是同一主题 | 独立 `P1-1` 小集合 | 不与 vive 合并 |
| 2026-07-24 vive | `2026-07-24_vive_rm65_real_teleop`、`_filtered_no184`、`_v22`、`_v22_pose_consistent`、`_v23` | base 与筛选/语义变体，D1；241-episode 版本很可能是 base 去掉 episode 184 的派生 | parent + `filtered_no184`/`v23` camera selector | 不把 242 和 241 相加；使用 `v23_camera_only_v1` 时只读其 241/35667 视图 |
| 2026-07-24 DMTAC | `2026-07-24_dmtac_rm65_bimanual_position_v1` | 与 vive 不同 protocol/task，D0/D1 均不成立 | 独立无 profile 小集合 | 不合并 |
| 2026-07-25 C12 | `2026-07-25_vive_rm65_real_teleop`、`_filtered_no283`、`_v22` | 同一 C12 parent/filtered/v22，D1 | 仅以 episode 283 的完整相机证据建立小型 camera selector | 不把 311、310、310 相加；`filtered_no283` 的 camera-only 结果为空 |
| 2026-07-27 | `2026-07-27_dmtac_rm65_bimanual_position_v1_old`、`...v2_responsive` | task/profile/采集目的不同，D2 条件不满足 | C15 repair projection 单独；responsive 单独 | 不合并；旧源只作为相机修复 lineage |
| 2026-07-30 | `2026-07-30_dmtac_rm65_bimanual_position_v1` | 单个 `P1-1` session | `P1-1` logical group | 无同日候选，保持独立 |
| 2026-07-31 | `...bimanual_position_v2`、`...writing_four_stage`、`DexTeleop_StableCollect_0705/...vive...` | 语义/profile/设备来源不同；前者 progress 不完整 | `P1-4` 只和自己的 camera-only view 配对 | 不跨 profile/设备合并；v2 缺 profile 先冻结 |
| 2026-08-03 P2-4 | `DexTeleop/datasets/2026-08-03...4_stage`（5 ep）与 `DexTeleop_0803/datasets/2026-08-03...4_stage`（202 ep） | 同日、同 task、同 progress/stage hash 的两次采集，D2 候选 | `logical/p2-4/2026-08-03`，预估 207 ep；保留两个 source path 和来源字段 | 先做 identity/overlap 校验；不覆盖 OpenPI profile |
| 2026-08-03 P2-5 | `DexTeleop_0803/datasets/2026-08-03...5_stage` | 与 P2-4 多一个 `right_dip_ink`，D2 不兼容 | 独立 `P2-5` logical group | 不加入 P2-4 |
| 2026-08-04 | `DexTeleop_0803/datasets/2026-08-04...4_stage` | 单一 P2-4 session，当前为 OpenPI 保护源 | 保持原 source；可加入受保护 P2-4 selector | 不物理移动、不改配置 |
| 2026-08-05 | `DexTeleop_0803/datasets/2026-08-05...4_stage` | 单一 P2-4 session，当前为 OpenPI 保护源 | 保持原 source；可加入受保护 P2-4 selector | 不物理移动、不改配置 |
| 2026-08-06 P2-4 | `...2026-08-06...4_stage`（35 ep）、`..._1453`（206 ep）、`..._stable`（53 ep） | 同日、同 task、同 P2-4 hashes，三次 session，D2 强候选 | `logical/p2-4/2026-08-06`，预估 294 ep；保留 session/source 字段 | OpenPI 当前期望 eligible 为 35+203+51=289，不能用物理总数替换 selector |
| 2026-08-07 FormalCollection | `2026-08-07...4_stage`（129 ep）与 `...4_stage__part02`（202 ep） | 同日、同 profile；`/home/user/zc/datasets/2026-08-07...` 是指向 part02 的 alias，D0 + D2 组合 | `logical/formal/2026-08-07/p2-4`，预估 331 ep（不重复计算 alias） | 先校验两目录 identity/时间不重叠；alias 不能计为第三份 |
| 2026-08-10 P2-4 | `...multi_choise_stage1`（113 ep）、`...4_stage`（53 ep）、`...4_stage_stable`（296 ep）、另一个同名 `...multi_choise_stage1`（153 ep） | 同 profile，但 normal/stable/multi-choice 是场景标签；两个同名目录必须按 UUID 区分 | 建立 `logical/formal/2026-08-10/p2-4/all`，并派生 `normal`、`stable`、`multi_choice` 子 selector；预估 615 ep | 可以 D2 logical append，但不能抹掉场景标签或直接物理改名 |
| 2026-08-11 P2-4 | `...interrupt_choice_stage1`（201 ep）、`...multi_choise_stage1`（251 ep） | 同 profile、不同 scenario，D2 候选 | `logical/formal/2026-08-11/p2-4/all`，保留 `interrupt_choice`/`multi_choice` 标签；预估 452 ep | 待确认训练是否允许两种场景进入同一 selector |
| 2026-08-12 | `...60hz_admittance_left_gripper_offline_test` | 0 episode/0 frame scratch，非正式采集 | 不进入任何训练集合 | 永久标记 scratch，不合并 |
| 2026-08-19/20 | `2026-08-19_4_stage_writing_detail`（159 ep）、`2026-08-20_4_stage_writing_detail`（当前 info 为 168 ep） | progress hash 虽为 P2-4，但 `info.task_id` 分别为 `4_stage_writing_detail` 与 `dmtac_rm65_bimanual_position_v1`，不是当前 full-task 期望的 v2 task；D2 不成立 | 暂存 `detail` 独立 selector，不能直接放入 P2-4 full-task；先做 task identity 纠正/adapter 决策 | 先补齐 external graph、task correction 和 annotation/quality；不能按“同 profile hash”直接合并 |

#### 同日合并的实际优先级

若目标是先整理“同一日期的相同采集”，建议按以下顺序推进：

1. **第一优先：2026-08-06 P2-4**。三目录 profile/hash 完全一致，且 OpenPI
   已有明确的 35/203/51 eligible selector；只需生成不改变旧配置的 logical
   append index。
2. **第二优先：2026-08-07 FormalCollection P2-4**。先处理 part02 alias 去重，
   再确认 129 与 202 episode 没有重复 identity。
3. **第三优先：2026-08-03 P2-4**。5 episode 与 202 episode 可同组，但五阶段
   108 episode 必须隔离。
4. **第四优先：2026-08-10/11 P2-4 场景组**。可以同 profile 追加，但保留
   `normal/stable/multi_choice/interrupt_choice` 场景标签，先建立总 selector
   和子 selector，不急于物理合并。
5. **第五优先：2026-08-19/20 detail**。先建立独立 detail selector；两者 task
   identity 需要 correction，且 tactile attachment 状态不同，完成修正前不加入
   P2-4 full-task。

### 8.4 同日合并必须满足的“相同采集”定义

“同一天、目录名相似”不足以判定相同采集。一个目录只有同时满足以下条件，
才能进入同日 D2 logical append：

```text
same normalized task_id
same action/observation schema hash
same progress hash and stage hash (or both explicitly have no progress)
same physical embodiment/end-effector semantics
same camera role/normalization and declared color/depth contract
episode identity and source time range are disjoint
no unresolved failure/annotation/quality conflict
tactile graph is either equally local, or each external reference has a receipt
no source is an invalid audit, camera-only projection, or known damaged lineage
```

只要其中一项不满足：

- 仍可保留为同一日期的**目录族/lineage group**；
- 不能进入同一个默认训练 selector；
- 不能以 episode 数相加作为合并成功证据；
- 不能覆盖任何 OpenPI 现有 source path。

### 8.5 camera-only 视图与 native source 的对应关系

为避免同一 episode 被 native 和 camera-only 重复计入，派生视图应记录明确的
`source_dataset_identity` 与 `source_episode_id`。当前已知映射如下：

| camera-only/repair 视图 | 对应 source | 允许用途 |
|---|---|---|
| 2026-07-23 DMTAC `camera_only_v3` | 2026-07-23 DMTAC P1-1 | camera-only P1-1 |
| C05 `old_version_v23.camera_only_v1` | C05 四个 vive lineage 目录 | 仅 13 个完整 episode 的视觉 selector |
| 2026-07-24 DMTAC `camera_only_v1` | 2026-07-24 DMTAC | camera-only 单 episode |
| 2026-07-24 vive `v23_camera_only_v1` | 2026-07-24 vive v23/filtered lineage | 241 episode 视觉 selector |
| 2026-07-25 `v22.camera_only_v1` | C12 lineage | 当前为空，不可用 |
| 2026-07-27 responsive camera-only | 2026-07-27 responsive source | camera-only responsive |
| 2026-07-30 DMTAC camera-only | 2026-07-30 P1-1 | camera-only P1-1 |
| 2026-07-31 v2 / writing four-stage camera-only | 对应同名 native source | 各自 profile；不能互相替代 |
| 2026-08-03/04/05/StableCollect views | 对应 source path 前缀 | camera-only selector；不含 tactile |
| C15 `camera_semantic_repairs/...` | 2026-07-27 old source | 修复后的独立 camera projection |

camera-only 视图最多只能作为同 source 的一个 modality projection。若训练需要
action/qpos/tactile，必须回到 native/external graph；不能因为 camera-only 目录
可正常加载就把它当成完整 dataset。

### 8.6 明确不能自动合并的对象

- C05/C12 损坏源及其 full-size 变体；
- `camera_semantic_prepared_views/_invalid_audit/*`；
- 空的 2026-08-12 scratch/test；
- camera repair projection 与 native source；
- `_v22`、`_v23`、`measured_state` 只凭目录名推断的变体；
- 同名但 episode 数、dataset UUID 或 task identity 不同的目录；
- 旧路径命名空间与 canonical 路径，在 full inode audit 前不能去重；
- tactile 状态为 `T-none?`、但没有 external/retirement receipt 的目录。

## 9. 任何实际整理前的只读 gate

在进行合并、迁移、软链接重建或删除之前，必须针对**每个候选组**重新生成并
保存：

1. 完整文件级 `device/inode/nlink` 报告，明确 hardlink 是否跨候选组共享；
2. source/target 独立 backup receipt，证明备份不是同 inode 的别名；
3. episode identity、frame count、Parquet schema、视频 tail decode、camera
   validity 的 manifest；
4. profile/progress/stage/action-schema hash 报告；
5. tactile hot/external/retirement graph 及 NAS 可恢复 receipt；
6. OpenPI profile、训练 selector 和所有历史命令对旧路径的引用清单；
7. 以新 inode 写入临时目录、逐项校验、最后原子发布的执行计划；
8. rollback 路径和失败时保留 staging/audit 的策略。

任何一项缺失，候选组保持“只读、不可合并”状态。特别是不能以目录大小、
目录名称或 `st_nlink` 单点证据代替完整 manifest。

## 10. 待后续备注的最小决策项

为了让下一步合并不靠猜测，后续只需为每个 native root 补充以下少量标签：

- `保留 / archive / failure / suboptimal / scratch`；
- 是否进入哪个 OpenPI profile（若已进入，保持现有 profile 不变）；
- tactile 训练是否需要、是否允许只保留 external receipt；
- P2-4 场景数据（normal / interrupt / multi-choice / stable）是否进入同一
  逻辑训练视图；
- 8/19、8/20 FormalCollection 是否作为 P2-4 候选；
- C05/C12 是否仅保留可用 camera-only 子集。

这些标签应写入后续的逻辑 selector/manifest，而不是通过重命名原始文件夹
表达。当前本文只做盘点，不代替用户对训练集合的最终取舍。

## 11. 本次结论

- 远端 canonical root 下共发现 **75 个带 dataset marker 的目录**：55 个
  native/source、17 个 valid camera-only、2 个 invalid audit、1 个 repair
  projection。
- 另有 1 个根目录外的 native tactile bundle probe，只有 1 episode/30 帧，
  是测试材料，不是训练数据。
- 当前最安全的整理顺序是：先按 profile/hash 和质量语义建立 logical index，
  再处理 tactile attachment，最后才考虑 fresh-inode 的物理合并。
- 同日期同采集的首批 D2 logical append 目标已经明确为：2026-08-06 P2-4
  （35+206+53）、2026-08-07 FormalCollection P2-4（129+202，排除 alias
  重复）、2026-08-03 P2-4（5+202）、2026-08-10/11 P2-4 场景组，以及
 2026-08-10/11 P2-4 场景组。8/19、8/20 detail 暂不属于 full-task 合并组，
 直到 task identity correction 完成。这里的数量只是待校验上限，不能替代
  identity 去重和 OpenPI eligibility。
- 同日不同 task/protocol、同一 source 的 v22/v23/filtered 变体、camera-only
  projection 和 P2-4/P2-5 profile 差异，明确保持 D0/D1/隔离，不进行 episode
  相加。
- OpenPI 现有 profile 和 8/4–8/6 保护源必须保持不变；任何合并计划都只能
  通过新的 selector/manifest 或原子配置迁移实现。
- 旧路径与 canonical 路径的 link 语义存在文档与 `lstat` 证据不一致，必须
  先做完整 inode 审计，不能把“看起来同名/同日期”当作独立数据或可删除备份。

## 12. 训练暂停期间的 OpenPI profile 合并方案

本节是基于 2026-08-21 的只读复核给出的可执行方案；没有修改远端 OpenPI 文件。
训练暂停使得切换 profile 成为可行选项，但不应直接覆盖当前 profile 或复用旧
normalization/checkpoint。

### 12.1 先保留不可变的 879-episode baseline

远端当前 profile：

```text
/home/user/zc/openpi/configs/training/
  writing_stage3_0804_0806_front_h16_absolute_history_state_v1.json
```

它固定使用 2026-08-04、08-05、08-06 三天的五个 source root，expected eligible
为 `344 + 246 + 35 + 203 + 51 = 879`，并绑定：

```text
task_id                 = dmtac_rm65_bimanual_position_v2_writing_4_stage
progress_profile_hash   = 0874f6b9...
stage_profile_hash      = b3e3d19d...
camera                  = front only
action_horizon          = 16
action_pose_mode        = absolute
history-state contract  = db215d...
```

该文件、对应 norm-stats、checkpoint、organized workspace 都应作为回滚锚点，
不原地修改。即使新的合并 profile 失败，也可以直接恢复原训练命令。

### 12.2 推荐的新 full-task profile：先纳入 8/7，不立即纳入 detail/场景集

仓库中已经存在一个可作为候选模板的本地 profile：

```text
deployments/openpi/configs/training/
  writing_stage3_0804_0807_front_h16_absolute_v1.json
```

它在 879 基础上追加 2026-08-07 FormalCollection 两个 source：

```text
2026-08-07 part1: raw 129, eligible 126
2026-08-07 part2: raw 202, eligible 195
```

因此新的 full-task P2-4 候选总 eligibility 是 **1200**，而不是简单把两个
目录的物理 episode 数相加。建议实际落地时生成一个新的 profile id，例如：

```text
writing_fulltask_p2_4_0804_0807_front_h16_absolute_v2
```

并执行以下策略：

1. 复制旧 profile 的模型输入契约，不改变 front-only、h16、absolute、history
   state、policy IO layout 和 prompt/stage projection 语义。
2. source list 保留 7 个独立 source id，不先物理拼接目录；每条 source 记录
   raw/eligible/frame/window 计数、日期、session、quality 和 tactile receipt。
3. 使用新的 `organized_dataset_root`、`openpi_asset_id`、norm-stats asset id
   和 experiment name；不能复用 879 集合的 norm stats。
4. 新 profile 从基础权重或明确指定的旧 checkpoint warm-start，但不使用旧
   experiment 的 `--resume` 语义；数据分布已经改变，必须产生独立 checkpoint
   lineage。
5. `writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json`
   已经提供了 4 个阶段的 prompt-family 结构，可作为 stage-conditioned 训练
   的上层组合；若训练目标仍是完整任务策略，则使用新的 full-task profile，
   不把 stage1-only profile 当作完整任务数据。

这一步满足“同一日期相同采集可合并”的要求：8/6 的三个 session 已经在旧
profile 中作为独立 source 逻辑追加，8/7 的两个 session 追加到新的 v2 profile；
不会破坏任何原始 episode 或 OpenPI 旧路径。

### 12.3 2026-08-03 是否纳入：作为第二个扩展 profile，不直接混入首版

2026-08-03 有两个 P2-4 目录（5 episode 和 202 episode），profile/stage hash
与 P2-4 一致，理论上可以追加到 8/4–8/7 full-task 集合，物理上限为 207
episode。但它不在当前 879/1200 的既有 OpenPI eligibility 统计里，建议先生成
独立候选：

```text
writing_fulltask_p2_4_0803_0807_front_h16_absolute_candidate
```

只有完成以下检查后，才将它升级为正式 profile v3：

- 两个 8/3 source 的 task id、profile hash、stage hash 和 action schema 与
  8/4–8/7 完全一致；
- 8/3 两目录之间没有重复 episode identity；
- failure/abnormal interval、camera front、Parquet/video tail decode 通过；
- tactile local/external graph 有明确状态，不因缺少本地 payload 而误判为纯视觉；
- 重新计算 eligibility 后，训练窗口数量和 norm stats 可复现。

这样可以把“同日 8/3 的 5+202”合并为 logical source group，同时不让未完成
审查的早期采集直接改变主训练 profile。

### 12.4 8/10–8/11 必须继续使用 stage1 scenario profile

以下现有 profile 已经正确表达了这些数据的实际采集范围：

```text
deployments/openpi/configs/training/
  writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json
```

它的 source mixing 明确使用：

```text
fixed_middle       repeat_weight=1
initial_multi_choice repeat_weight=4
midcourse_interrupt repeat_weight=6
target_step_id     = right_acquire_brush
task_scope         = full_writing_sequence 或 stage1_only（逐 source 声明）
expected eligible  = 2283
```

因此建议：

- 8/10 两个同名 `multi_choise` 目录和 8/11 `multi_choise` 继续在该 profile
  中逻辑追加，使用不同的 source id 和 dataset identity 区分，不能按文件夹名
  去重；
- 8/10 `stable`、8/10 `fixed_middle`、8/11 `interrupt_choice` 保留
  `scenario_class`，不要因为 progress hash 相同就加入 full-task `P2-4`；
- repeat weight 是为 stage1 场景补偿采样设计的，不能原样搬到 full-task 4-stage
  profile，否则会改变阶段分布；
- 如果未来要训练完整任务而不是 stage1，必须把这些数据先通过明确的窗口/阶段
  projection 转为完整 task 语义，不能只改 `expected_task_id`。

### 12.5 8/19、8/20 detail 暂时隔离

只看 progress hash 会错误地把这两个目录归入 P2-4。实际只读证据是：

| 数据集 | `meta/info.json task_id` | `meta/progress/profile.cjson task_id` | 结论 |
|---|---|---|---|
| 2026-08-19 detail | `4_stage_writing_detail` | `dmtac_rm65_bimanual_position_v1` | 不是当前 full-task v2 task，暂不合并 |
| 2026-08-20 detail | `dmtac_rm65_bimanual_position_v1` | `dmtac_rm65_bimanual_position_v1` | 不是当前 full-task v2 task，暂不合并 |

它们可以先建立独立的：

```text
logical/formal/detail_20260819_20260820
```

但只能用于 detail/质量分析或专用 selector。要进入 full-task profile，必须先
由数据接口层完成显式的 task identity correction/adapter，并重新确认：

1. 四个 progress step 的 coverage 是完整任务还是局部 detail overlay；
2. action/state/camera schema 是否与 P2-4 full-task 完全一致；
3. 60 Hz force-admittance 或 post-capture detail 是否改变了训练域；
4. tactile external receipt 和 source episode identity 是否完整；
5. correction 产生新 profile hash/manifest，且旧事实仍可回读。

在这些条件完成前，不修改它们的 `task_id`，也不把它们加入 1200 或 879 的
norm-stats 统计。

### 12.6 推荐的 profile 层次

最终不要制造一个“所有 writing 目录的超级 profile”，而是保留三个可独立回滚
的训练层：

| profile 层 | 内容 | 预期用途 |
|---|---|---|
| `fulltask_p2_4_baseline_v1` | 8/4–8/6，879 eligible | 当前稳定基线，永久保留 |
| `fulltask_p2_4_extended_v2` | 8/4–8/7，1200 eligible；8/6/8/7 按 source 逻辑追加 | 完整四阶段策略训练 |
| `stage1_scenarios_0804_0811_v1` | 8/4–8/11 的 stage1/scenario projection，2283 eligible，带 repeat weights | 首阶段、异常/中断恢复专项训练 |

8/3 作为待验证 v3 candidate，8/19/20 作为 detail candidate，P2-5 五阶段永远
单独维护。每个 profile 使用独立：

```text
source manifest
eligibility/annotation receipt
norm-stats asset
organized workspace
OpenPI config name
checkpoint lineage
```

### 12.7 Profile 改动的回滚与验证顺序

如果决定实际切换 OpenPI，建议只新增文件/配置，不覆盖远端旧 profile：

1. 在本地生成新 profile，并验证 JSON、source path、hash、计数和 schema。
2. 只读扫描所有 source，生成 episode identity overlap 报告。
3. 在新 workspace 生成 prepared manifest；旧 workspace 保持不动。
4. 重新计算新 profile 的 norm stats，记录 stats hash。
5. 先运行 1–10 step smoke train，验证 loader、camera、action/state layout、
   history-state 和 loss 没有异常。
6. 再用独立 experiment name 开始正式训练；不要给旧 experiment 加 `--resume`。
7. 若失败，停用新 profile/experiment 即可回到 879 baseline；不需要恢复数据
   目录或撤销历史配置。

目前最合理的切换结论是：**先把 8/7 纳入新的 1200-eligible full-task profile，
保留 879 baseline；8/10–8/11 继续使用已有 stage1 scenario profile；8/3 等验证
通过后再扩展；8/19/20 detail 暂不进入 full-task。**
