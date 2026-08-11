# RealmanDMTAC 数据根：Hardlink 与兼容软链接警告

> **警告：这里是数据集的 canonical 存储位置。旧路径只是兼容软链接，hardlink 不是备份。**

## Canonical 根目录

```text
/home/user/zc/datasets/RealmanDMTAC/
```

当前目录按历史来源命名空间组织：

```text
DexTeleop/
DexTeleop_0803/
DexTeleop_StableCollect_0705/
```

从 2026-08-06 起，所有**新建**的正式采集统一使用：

```text
/home/user/zc/datasets/RealmanDMTAC/FormalCollection/
```

RealTeleop/headless collector 在该目录下创建
`<YYYY-MM-DD>_<collection-task-name>`；UMI 使用其独立的
`FormalCollection/umi_dual_wrist` 数据集目录。旧命名空间、历史配置、既有数据集及兼容
软链接保持原样，不能批量改写为新路径。命令行或本机 `local_paths.json` 的显式覆盖继续
用于旧数据读取、离线测试和受控回退，不代表新的正式采集默认位置。

原 `/home/user/zc/...` 数据集路径保留为指向这里的**目录软链接**。旧路径和这里
看到的是同一份实际数据，不得作为两份数据重复统计，也不能互为备份。

## Hardlink 安全规则

- 部分历史数据集通过 hardlink 共享文件 inode；不同文件名或不同数据集路径可能
  指向同一份底层内容。
- 对任一 hardlink 路径原地写入，都会同时改变所有共享该 inode 的数据集。
- 禁止使用会直接截断或覆盖原文件的原地处理方式，包括直接打开目标文件写回。
- 媒体、触觉或其他数据变换必须写入新的临时 inode，完整校验内容、声明帧数、
  解码帧数和尾部可解码性后，再原子替换明确批准的目标。
- 路径副本、软链接和 hardlink 都不是独立备份。独立备份必须位于经核验的独立
  存储，并具有单独的完整性回执。
- 移动、复制、删除、压缩、拆出或装回模态前，必须先检查源/目标设备、inode、
  `st_nlink`、共享链接集合、可用空间和恢复路径；发现不一致立即停止。

## 本次统一迁移

2026-08-06 的统一迁移采用同一文件系统内的目录重命名，因此保留了原 inode 和
跨数据集 hardlink 关系。迁移计划、逐步 journal 和完成 receipt 位于：

```text
/home/user/zc/datasets/RealmanDMTAC/.migration/20260806_full_relocation/
```

该事务覆盖 46 个数据集、10,135 个 Episode、156,017 个文件路径，其中 95,621 个
路径对应 `st_nlink > 1` 的文件。原兼容路径和 canonical 目标均已通过事务验证。

## 操作前必读

项目中的详细规则和数据交接说明位于：

```text
docs/data/v25p2/DATASET_CONTENT_AND_HANDOFF.md
```

如果不确定某个操作是否会写穿 hardlink、破坏模态引用或影响旧路径兼容性，停止操作，
先完成只读审计并取得明确确认。
