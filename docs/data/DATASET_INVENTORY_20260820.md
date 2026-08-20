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
| 10 | `2026-08-19_4_stage_writing_detail` | 3.9G | 159 / 32042 | external, `T-ext` | `P2-4` | tactile UUID `68d9f362-a864-4970-aec5-56ce882050ca` 已在 NAS；本地只留 attachment metadata |
| 11 | `2026-08-20_4_stage_writing_detail` | 928M | 46 / 7662 | external, `T-ext` | `P2-4` | tactile UUID `e25d726e-bef0-4414-ba14-a3a2b6d7b14e` 当前在 FormalCollection/tactile |

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

FormalCollection 的 8/19、8/20 都是 P2-4，但 8/19 tactile 已在 NAS、8/20
tactile 仍在 FormalCollection 外置容器；二者可以作为同 profile 的逻辑候选，
但不能在没有 attachment receipt、annotation 和 identity 去重前物理合并。

### 8.2 明确不能自动合并的对象

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
- OpenPI 现有 profile 和 8/4–8/6 保护源必须保持不变；任何合并计划都只能
  通过新的 selector/manifest 或原子配置迁移实现。
- 旧路径与 canonical 路径的 link 语义存在文档与 `lstat` 证据不一致，必须
  先做完整 inode 审计，不能把“看起来同名/同日期”当作独立数据或可删除备份。
