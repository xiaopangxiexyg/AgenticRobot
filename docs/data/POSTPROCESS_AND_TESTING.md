# 数据后处理、分析与测试说明

简洁命令见 [data README](./README.md)。本页解释各数据、分析和测试入口的参数与输出边界。

## 1. 数据集后处理

统一入口的完整参数：

```bash
python -m utils.dataset_postprocess --help
```

当前 dry-run、apply 和 v2.3 验证命令只在 [data README §1–2](./README.md) 维护。

版本转换、episode 整理、mask metadata、force 计算和 wrist depth 工具见
[`utils/dataset_postprocess/README.md`](../../utils/dataset_postprocess/README.md)。不同子工具是否
默认 dry-run 不一致，必须先看对应 `--help`。

## 2. 数据集统计

```bash
python utils/statistic/pose_action_lag_scan.py --help
python utils/statistic/pose_action_next_state.py --help
```

这些是离线分析入口。输出报告应放在数据集自己的 analysis 目录或仓库 `logs/`，不要在根目录
继续新增 `*_report.md`。

Tracker replay 分析工具：

```bash
python -m robot_motion.tests.compare_tracker_replay --help
python -m robot_motion.tests.view_tracker_replay --help
python -m robot_motion.tests.write_tracker_replay_report --help
```

工具虽然位于 `robot_motion/tests`，但属于离线诊断，不是真机测试。

## 3. 可视化工具

各工具对输入格式的要求不同，先运行 `--help`：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py --help
python utils/portable_vis/multi_view/multi_view_video_player.py --help
python utils/portable_vis/multi_view/real_sim_video_compare.py --help
python utils/portable_vis/eef_vis/visualize_eef_trajectory.py --help
python utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py --help
```

说明分别位于 `utils/portable_vis/*/README.md`。带 GUI 的工具属于 R1，并需要桌面环境。

## 4. 测试

数据后处理、schema、外置 tactile 和 wrist depth 的最小相关集合：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  tests/test_dataset_organize.py \
  tests/test_dataset_v22_deploy.py \
  tests/test_dm_force_calculate.py \
  tests/test_tactile_external_storage.py \
  tests/test_wrist_depth_storage.py
```

通用单元测试和修改后的基础检查命令见 [setup 入口](../setup/README.md)。完整测试层级见
[`README_ENV.md`](../../README_ENV.md)。真机、相机、USB、GUI 和 motion 测试不能因为
文件名含 `test` 就自动运行，必须按硬件安全流程单独授权。

## 5. 验收文档

- [v2.3 验收](./ACCEPTANCE_V23.md)：contract、virtual equivalence、real A/B 和 OpenPI。
- [历史 `test-7.11.md`](../legacy/position_control/test-7.11.md)：位控/力控链路专项 A/B；
  当前正式 position 配方已经进入 [真实系统入口](../real/README.md)，旧文档仅保留演进与诊断证据。
- [历史 position A/B report](../legacy/reports/position_ab_report.md)：离线分析结果，
  不是运行入口。
