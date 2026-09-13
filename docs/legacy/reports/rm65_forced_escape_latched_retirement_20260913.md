# RM65 `forced_escape_latched` 退役说明

`forced_escape_latched` 是 2026-08-02 引入的双臂关节限位逃逸实验。它在普通
限位候选之上维护方向锁存和 release guard，并在候选选择阶段建立第二条状态化
逃逸路径。当前正式位控配置从未启用该模式，V1/V2/V3 均使用 `legacy`；仓库中
也没有生产脚本或配置选择它。

从 2026-09-13 起，`joint_limit_guard_mode` 只接受：

- `legacy`：保持当前正式基线和普通限位候选行为；
- `boundary_hold`：在窄限位带内仅投影掉向内关节分量。

本次删除 forced 模式的方向锁存、release guard、二次候选调度及专属测试，不删除
通用 `local_step_limit_escape` 候选。原常量暂时仅作为 import 兼容符号保留，不再
属于可接受 choices，也不能激活运行时行为；待现有 `robot_motion/rm65/__init__.py`
并行候选闭环且完成外部导入清点后，可单独删除该符号。离线 counterfactual replay
仍按旧名称复制两个空状态容器，因此 solver 暂留不参与控制决策的空占位属性。
