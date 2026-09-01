# SAPIEN virtual teleop 退役记录（2026-08-31）

## 退役对象

`teleop_runtime/debug/run_vive_sapien_virtual.py` 是历史 VIVE + SAPIEN operator-preview
可执行程序。在退役前它已被标记为冻结调试路径，但正式 real-teleop
入口仍从它导入 Home 事务、恢复、episode frame、终端输入和诊断函数。这使一个
历史可执行程序成为正式运行时的隐式 library/owner，并将两条生命周期耦合在一个
3319 行文件中。

## 替代所有者

- 正式组合入口：`control_station.real_teleop_entrypoint`
- Home 事务和 operator diagnostics：`teleop_runtime.real_teleop.runtime_home`
- 终端 clutch/quit 输入：`teleop_runtime.real_teleop.terminal_input`
- 绝对位姿恢复计划：`teleop_runtime.recovery`
- episode frame 构造：`teleop_runtime.real_teleop.episode_runtime`
- 虚拟/离线验证：canonical MuJoCo、contract 或 backend-neutral 路径

迁移保留了 real-teleop 的 Home 原子快照、新鲜度/稳定性门禁、恢复插值、预览同步、
碰撞诊断、episode schema 和 terminal input 行为。本次不改变真机 dispatch 所有权或安全边界。

## 删除条件

- 正式 Python 源码和测试已无该历史模块的 import/caller。
- 跟踪文档不再提供该退役入口的可执行命令或源码链接。
- 正式 real-teleop 测试改为针对新所有者，而不再通过历史程序间接测试。

## 证据边界

本次验证为静态、单元和无设备证据；没有启动仿真器、连接硬件或下发任何真实运动。
它不构成真机可用性或科学结论。
