# DMTAC 旧夹爪驱动退役记录

日期：2026-08-25

## 结论

`hardware/DMTAC_gripper/portable_dmtac_gripper/gripper_driver_legacy.py` 是
`gripper_driver.py` 的早期重复实现，已从版本化源码树删除。此次退役只删除该
文件，不保留同名 shim，也不修改 `docs/legacy/README.md`、当前 DMTAC CLI 或
其他硬件入口。

当前 Inspire 夹爪串口/Modbus 实现仍由
`hardware/DMTAC_gripper/portable_dmtac_gripper/gripper_driver.py` 提供，并由
`portable_dmtac_gripper.__init__`、`cli.py`、`device.py` 和 `usb_assignment.py`
使用。

## 审核依据

- 删除前在活动源码、测试、配置和文档中搜索
  `gripper_driver_legacy` 及其模块导入路径，没有发现旧驱动的外部调用；命中
  仅来自待删除文件自身。
- 旧文件与当前驱动共享 `GripperError`、`GripperState`、
  `InspireGripperModbus`、`InspireGripper` 和串口 CLI 的职责。当前驱动是实际
  活跃导入路径，并包含旧实现没有的并发、串口恢复、寄存器解码和参数兼容增强。
- DMTAC 的 `legacy` 语义仍仅指明确保留的触觉计算/产物兼容 profile，不指向本
  文件，也不因此把退役代码移入根 `legacy/`。

## 当前驱动 API 与保留增强

当前 `gripper_driver.py` 保留以下公开 API：

- `GripperError`、`GripperState`；
- `InspireGripperModbus` 的 CRC、holding-register 读取、单寄存器和多寄存器写入、
  重连及上下文管理；
- `InspireGripper` 的故障清除、急停、参数保存/恢复、设备 ID/波特率、连续抓取、
  速度/力矩、位置、开/关/半闭、状态读取、等待空闲和开度限制操作；
- `list_serial_devices()` 与模块 CLI 的 state/open/close/half-close 等入口。

相对于旧实现，当前驱动还保留并继续作为唯一实现的增强包括：

- 按真实串口路径共享 `RLock`，将端口级并发和实例级 I/O/reconnect 锁定在同一
  事务边界；
- 使用单调时钟 deadline、动态串口读取超时、写超时、输出缓冲清理和短写检测；
- 支持 `retries` 兼容参数、可调 `inter_command_delay`，以及默认关闭的
  `use_write_multiple_registers` 优化路径；
- 在实际开度寄存器边界按 signed int16 解码，缓存连续抓取模式，并让显式开/关/
  半闭命令强制发送，保留 transient Modbus 错误的受控重连；
- CLI 提供新的 `--inter-command-delay` 与
  `--use-write-multiple-registers` 参数，同时保留现有默认值和状态输出边界。

## 仍保留的 DMTAC 兼容语义

本次删除不改变 `hardware/DMTAC_gripper/portable_dmtac_gripper/device.py` 中的
触觉 `legacy` compute profile。`old`、`compat`、`compatible` 等 profile 名仍
映射到 `legacy`，旧的全量 deformation/shear/depth/force 计算选择仍可显式使用。

CLI 的兼容别名也继续保留，包括 `--id`、`--baudrate`、`--timeout`、
`--max_open_mm`、`--tactile_ids`、`--backend`、`--mode`、`--max_fps`、
`--tactile_fields`，以及各历史子命令仍声明的隐藏参数。它们复用当前实现，
不再依赖已删除的旧驱动文件。

## 验证边界

退役提交前应确认：

```bash
git grep -n -E 'gripper_driver_legacy|portable_dmtac_gripper\.gripper_driver_legacy' \
  -- . ':!docs/legacy/**'
git ls-files hardware/DMTAC_gripper/portable_dmtac_gripper/gripper_driver_legacy.py
git ls-tree -r --name-only HEAD legacy
bash scripts/run_pytest.sh \
  tests/test_dmtac_gripper_driver.py \
  tests/test_dmtac_gripper_cli_config.py \
  tests/test_dmtac_gripper_only_debug.py \
  tests/test_dmtac_usb_assignment.py -q
python scripts/check_docs_navigation.py
python scripts/dev_local_resources.py audit
git diff --check
```

本 checkout 不包含历史命令中曾使用的 `tests/test_dmtac_gripper.py`；当前等价的
驱动契约测试是 `tests/test_dmtac_gripper_driver.py`。本次聚焦验证使用上述四个
实际存在的测试文件。新报告没有加入 `docs/README.md` 或
`docs/legacy/README.md` 导航，因此文档导航检查会报告本报告为
`unreachable_docs_file`；这是本 Task 的明确 scope 限制，未以越界修改消除该提示。

聚焦测试覆盖的是 mock 串口、CLI 配置、USB 分配和 no-motion 调试边界；文档及
资源检查只证明版本化结构和本地资源策略。此次退役没有启动串口、夹爪、触觉
设备或任何真机运动，因此不构成硬件连接、标定、力控或生产验收证据。

根 `legacy/` 继续保持严格本地、完全忽略且无 tracked path；版本化历史说明只
保留在本报告和其他 `docs/legacy/` 文档中。
