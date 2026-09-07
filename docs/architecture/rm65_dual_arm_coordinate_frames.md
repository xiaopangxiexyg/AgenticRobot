# RM65 双臂统一坐标系规范

本文定义 RM65 双臂工作单元的逻辑 **Cell 坐标系 v1**。这是一层独立、未激活的规范与换算契约：本次变更不把它注册到任何现有 real、sim、motion、teleop 或 Agent Gateway 运行时图中，也不改变动作、IK、安全、碰撞、工具偏置或下发逻辑。

## 1. 规范坐标系

Cell 坐标系标识为 `rm65_dual_arm_cell`，轴语义契约为 `rm65_dual_arm_cell_x_table_y_left_arm_z_up_v1`。

- 原点：参考 URDF 根 link `world` 的原点，即标称安装状态下两臂基座之间的中心。
- `+X`：从两臂中心指向桌面。
- `+Y`：从两臂中心指向左臂。
- `+Z`：竖直向上。
- 手性：右手系，满足 `+X × +Y = +Z`。

这里仅使用 URDF 根原点与场景中的三个物理方向来锚定 Cell；Cell 轴名和正方向采用人类操作语义，不继承 URDF 原生轴的语义命名。

当前参考资产是：

`assets/rm65_inspire_gripper_v4/rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf`

其 SHA-256 为：

`0a91ac161dd3aa3ca5bbf3eb2fa13e647b9eb559dfb54ce115c21bb0b3dcbb34`

该资产在本规范中是几何依据，不代表本次将仓库中仍使用 v3 或其他资产的默认项迁移到 v4。默认资产迁移必须作为独立变更评审和验证。

## 2. 变换约定

统一采用 `T_target_source`：矩阵把 source 中表达的点或绝对位姿转换为 target 中的表达。

令：

- `U`：参考 URDF 的原生根坐标系 `world`；
- `H`：Cell 人类语义坐标系 `rm65_dual_arm_cell`；
- `C`：现有运行时使用的 `canonical_station`。

Cell 与 URDF 根原点重合，Cell 的 `+X/+Y` 分别对应 URDF 原生 `-X/-Y`，`+Z` 相同，因此：

```text
T_U_H =
[[-1,  0, 0, 0],
 [ 0, -1, 0, 0],
 [ 0,  0, 1, 0],
 [ 0,  0, 0, 1]]
```

这是绕 `+Z` 的精确 180 度旋转，行列式为 `+1`，不会引入镜像或左手系。

现有 `canonical_station` 的轴定义已经是“`+X` 向桌面、`+Y` 向左、`+Z` 向上”，所以 `C` 与 `H` 的方向完全一致，仅原点不同。`C` 是参考 URDF 中桌面近机器人边界、左右居中且位于桌面表面的 landmark：

```text
T_H_C =
[[1, 0, 0, 0.472],
 [0, 1, 0, 0.000],
 [0, 0, 1, 0.010],
 [0, 0, 0, 1.000]]

T_C_H =
[[1, 0, 0, -0.472],
 [0, 1, 0,  0.000],
 [0, 0, 1, -0.010],
 [0, 0, 0,  1.000]]
```

参考 landmark 在 Cell 中应满足：

- 左臂基座原点：`(0, +0.2625, 0)`；
- 右臂基座原点：`(0, -0.2625, 0)`；
- `canonical_station` 原点：`(+0.472, 0, +0.010)`。

这三个断言同时约束了桌面方向、左右臂方向和竖直方向，任何 X/Y 符号翻转或 source/target 顺序错误都会被测试发现。

## 3. 表达类型边界

本次提供的纯函数只转换齐次矩阵形式的绝对位姿：

```text
T_H_tool = T_H_C @ T_C_tool
T_C_tool = T_C_H @ T_H_tool
```

同一参考变换左乘到 anchor 与 target 后，相对位姿保持不变：

```text
inv(T_H_anchor) @ T_H_target
== inv(T_C_anchor) @ T_C_target
```

方向向量只使用旋转子矩阵，不叠加原点平移。Twist、wrench、协方差、速度或力控制量不能直接套用绝对位姿公式；它们分别需要 adjoint、dual-adjoint 或各自表示层定义的变换。本规范当前不提供也不暗示这些换算。

## 4. 所有权与零影响约束

代码入口 `robot_motion.rm65.cell_frame` 构造独立的 `rm65-dual-arm-cell-frame-overlay-v1` 图。它不在 import 时执行 I/O，不访问相机、机械臂、仿真器或网络，也不修改全局注册表。

为避免已有行为发生任何隐式变化，本阶段保持以下不变量：

- 不向现有 real、sim、teleop、motion 或 Gateway 坐标图新增 frame/edge；
- 不修改 `canonical_station` 的 ID、版本、轴契约或语义哈希；
- 不修改现有动作 schema、action chunk、TCP displacement、IK、碰撞、安全 envelope、工具偏置或 dispatch；
- 不修改现有数据集 desk/raw pose contract；
- 不切换默认 URDF，不执行仿真动作或真机动作；
- 只有调用方显式调用 Cell 适配函数时才进行换算，并且在进入旧执行链前显式转换回 `canonical_station`。

## 5. 复核与后续激活门槛

本阶段的自动化复核包括：精确矩阵和右手性、URDF landmark、Station/Cell 互逆、随机 SE(3) 往返、相对位姿不变、非交换旋转下的最终执行目标等价，以及构建 overlay 前后现有 real frame graph/Station frame 的 payload 与哈希不变。

若后续要让 real、sim、motion、teleop 或 Gateway 直接接受 Cell 坐标，必须逐个建立表示类型清单和唯一转换所有者，并满足：

1. 明确每个入口的 source/reference/controlled frame，禁止依赖无 frame ID 的裸数组猜测；
2. 以当前动作链输出作为 golden baseline，做同物理目标的前后执行目标逐元素对比；
3. 分别验证 absolute pose、relative pose、position/vector、orientation；twist/wrench 必须另立契约；
4. 对 X/Y/Z 符号、左右臂互换、矩阵逆向、乘法顺序和米/毫米单位做 mutation test；
5. 先离线与仿真 shadow，再做无运动 dry-run；任何真机动作验证均需单独明确授权；
6. 现有所有图、schema 与部署 binding 的内容哈希必须保持不变，除非该消费者进入一个单独、显式版本化的迁移。

在上述门槛逐域完成前，Cell 坐标系保持“已定义、可显式换算、未激活”状态。
