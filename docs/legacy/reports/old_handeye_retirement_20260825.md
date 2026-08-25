# `hardware/old_handeye/` 退役记录

日期：2026-08-25

## 结论

`hardware/old_handeye/` 是早期眼在手外手眼标定实验目录，已从版本化源码树删除。此次退役只删除该目录的 10 个 tracked 文件，不向根 `legacy/` 写入或跟踪代码。

当前手眼标定功能由以下版本化入口承接：

- 腕部相机（眼在手上）：[`utils/hand_eye/README.md`](../../../utils/hand_eye/README.md)
- 工位相机（眼在手外）：[`utils/head_eye/README.md`](../../../utils/head_eye/README.md)

## 审核依据

- `git grep` 在当前源码、测试、活跃文档和配置中未发现对 `hardware/old_handeye`、其模块名或 `_old_handeye` 的外部调用。
- 旧目录没有 `__init__.py`，不是当前可安装 Python 包；脚本依赖早期的本地 `PathConfig`、`teleop.mocap.vive_data_receiver` 和手工 `handeye/` 输出路径。
- 旧目录内的 README、脚本、测试、JSON 样例和 PNG 仅互相引用，没有被当前入口导入。
- 新入口已有 parser/help、功能边界和运行迁移测试：
  - `tests/test_hand_eye_utils.py`
  - `tests/test_handeye_utils.py`
  - `tests/test_runtime_artifact_migration_boundaries.py`

## 保留边界

本次不删除 `utils/hand_eye/`、`utils/head_eye/`、当前 `robot_motion/configs/rm65_camera_registry.json` 或现有 `teleop_records/headeye` 资源。它们仍是当前手眼标定和相机注册契约的一部分。

根 `legacy/` 继续保持严格本地、完全忽略；版本化历史说明只保留在本报告和 `docs/legacy/README.md` 导航中。

## 验证清单

退役候选应在提交前确认：

```bash
git grep -n -E 'hardware/old_handeye|old_handeye' -- . ':!docs/legacy/**'
git ls-tree -r --name-only HEAD hardware/old_handeye
bash scripts/run_pytest.sh tests/test_hand_eye_utils.py tests/test_handeye_utils.py \
  tests/test_runtime_artifact_migration_boundaries.py -q
python scripts/check_docs_navigation.py
```

第一条和第二条分别应无外部命中、无 tracked path；测试只证明当前替代入口和边界仍可用，不等同于真实相机或机械臂验收。
