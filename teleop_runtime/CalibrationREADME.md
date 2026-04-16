# Calibration README

这份文档只说明**当前已经落地的** VIVE -> RM65 标定调试流程。  
它对应的入口是：

- [teleop_runtime/debug/run_vive_rm65_calibration_debug.py](/home/zc/HandMotion/RealRobotExperiments/dex_inspire_teleop/teleop_runtime/debug/run_vive_rm65_calibration_debug.py)

当前这套流程的核心目标是：

1. 用 `auxiliary tracker` 标定一个**虚拟 desk** 坐标系 `V`
2. 用左右 wrist tracker 标定 `tracker -> hand control frame` 外参
3. 采集本次 session 的 `home anchor`
4. 直接在 SAPIEN 里验证 VIVE 到双臂虚拟 RM65 的映射效果

这套流程**不会**重标 `desk -> base`。  
机器人侧继续沿用当前 preset 里的 `T_D_B_left/right`。

## 1. 坐标系定义

当前工具里涉及的主要坐标系：

- `W`: `vive_world`
- `V`: `virtual desk`
- `D`: 机器人侧当前 preset 使用的 desk frame
- `B_left / B_right`: 左右机械臂 base
- `T_left / T_right`: 左右 wrist tracker
- `H_left / H_right`: 左右人手控制参考系
- `E_left / E_right`: 左右机器人 TCP

当前流程里，真正会求的是：

- `T_W_V`
- `T_T_H_left`
- `T_T_H_right`
- 每次启动的 `home anchor`

不会求的是：

- `T_D_B_left/right`

## 2. 这套流程为什么叫“虚拟 desk”

因为操作者不是固定在真实桌面边缘工作的。  
如果强依赖“把 tracker 插到桌面固定槽位”，会与实际 teleop 使用方式冲突。

因此这里的思路是：

- 让操作者通过一组**动作**定义“前/左/上”三个方向
- 用这些动作拟合出一个虚拟工作平面 `V`
- 再把人手运动先映射到 `V`
- 最后再映射到机器人侧当前使用的 `D`

所以：

- `V` 决定人侧运动方向解释
- `D` 决定机器人侧固定 desk/base 关系

## 3. 工具入口

推荐命令：

```bash
conda activate grasp_py310_copy

python hardware/VIVE/mocap/vive_data_sender.py \
  --host 127.0.0.1 \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist \
  --tracker tracker_3:auxiliary
```

```bash
python -m teleop_runtime.debug.run_vive_rm65_calibration_debug \
  --host 0.0.0.0 \
  --port 9999 \
  --tcp-mode rm_eef
```

如果本机没有可用渲染设备，可以先加：

```bash
--skip-renderer
```

## 4. 当前状态机

这个 app 当前实现的顺序是：

1. `LINK_CHECK`
2. `VIRTUAL_DESK_X`
3. `VIRTUAL_DESK_Y`
4. `VIRTUAL_DESK_Z`
5. `VIRTUAL_DESK_NEUTRAL`
6. `LEFT_WRIST_COARSE`
7. `RIGHT_WRIST_COARSE`
8. `HOME_CAPTURE`
9. `VALIDATE`
10. `SAVE`

## 5. 按键说明

### 5.1 通用按键

- `q`: 退出
- `c`: 开始/结束当前 capture
- `r`: 重置当前 capture

### 5.2 链路检查阶段

- `n`: 在 `left/right/aux` 三个 tracker 都在线后继续

### 5.3 验证阶段

- `1`: 选择左手做细调
- `2`: 选择右手做细调
- `j/l`: 当前选中侧 `yaw -/+`
- `i/k`: 当前选中侧 `pitch +/-`
- `u/o`: 当前选中侧 `roll -/+`
- `x`: 当前选中侧细调角度清零
- `h`: 重新采 home
- `s`: 保存 calibration profile 并退出

注意：

- 一旦你调整了 wrist 外参，旧的 `home anchor` 会被清掉
- 必须重新按 `h` 采 home，之后再继续验证

## 6. 详细动作要求

### 6.1 LINK_CHECK

目标：

- 确认 `left_wrist_pose`
- 确认 `right_wrist_pose`
- 确认 `auxiliary_tracker_pose`

通过条件：

- 三个 channel 都在线且 `valid=True`

### 6.2 Virtual Desk X

动作：

- 手持 auxiliary tracker
- 沿你想映射成机器人“前方”的方向来回扫动
- 建议从负方向开始，最后停在正方向结束
- 高度尽量保持不变
- 振幅建议 `20-40 cm`

为什么要“负到正”：

- 当前工具会用轨迹的首尾趋势给主轴定正负号

### 6.3 Virtual Desk Y

动作：

- 沿你想映射成机器人“左方”的方向扫动
- 同样建议从负方向开始，正方向结束
- 保持高度尽量不变

### 6.4 Virtual Desk Z

动作：

- 做上下抬升
- 建议从低处开始，高处结束
- 不需要特别大幅，但要清楚

### 6.5 Virtual Desk Neutral

动作：

- 把 auxiliary tracker 放到你觉得舒服的 teleop 中心位置
- 尽量稳定停住

这一步定义的是：

- 本次 session 的 `origin_V`

它不是物理桌面原点，只是虚拟 desk 的中心参考。

### 6.6 Left/Right Wrist Coarse

每只手都要做一次。

建议 neutral 姿态：

- 前臂自然前伸
- 手背朝上
- 手指大致朝 `+X_V`
- 姿态尽量稳定

工具会根据这个 pose 计算当前侧的粗外参：

- `T_T_H_side`

默认只靠这一步做旋转粗对齐。  
平移偏置先用参数输入，默认是 `0,0,0`。

### 6.7 Home Capture

动作：

1. 人工把双臂 move 到本次要用的 teleop home
2. 双手回到自然操作起始姿态
3. 开始采样

这一步只负责：

- 本次 session 的起始锚点

不负责：

- desk/world 外参
- wrist mount 外参

## 7. 虚拟 desk 拟合方法

当前实现里：

- `X/Y/Z` 三段 sweep 都只用**位置轨迹**
- 对每段轨迹做 PCA / SVD
- 取第一主方向作为该动作的主轴

然后：

```text
x_V = principal_axis(X-sweep)
y_raw = principal_axis(Y-sweep)
z_hint = principal_axis(Z-sweep)
```

再做正交化：

```text
y_V = normalize(y_raw - dot(y_raw, x_V) * x_V)
z_V = normalize(cross(x_V, y_V))
```

最后用 `z_hint` 决定 `z_V` 的正负方向。

## 8. 默认质量门槛

当前默认要求：

- 每段 sweep 至少 `8` 帧
- 每段 sweep 的 `extent >= 0.15 m`
- 每段 sweep 的 PCA 主方向 `variance_ratio >= 0.85`
- 每段 sweep 的 `start_end_distance >= 0.08 m`
- 最终虚拟 desk 三轴的最大非正交误差 `<= 20 deg`
- hold / wrist / home 默认至少 `20` 帧

这些值都可以通过 CLI 改。

## 9. 保存内容

默认保存到：

- `teleop_records/calibration_profiles/<timestamp>_vive_rm65_virtual_desk.json`

其中会包含：

- `T_W_V`
- 左右 wrist 的 coarse/final tracker offsets
- 虚拟 desk 拟合质量信息
- 本次保存时间
- home capture 结果

## 10. 典型验证动作

保存前建议至少做这些动作：

1. 左手单独向前推一点，左臂应主要向机器人前方运动
2. 左手单独向左移一点，左臂应主要向机器人左方运动
3. 左手向上抬一点，左臂应向上
4. 右手重复一次
5. 左右手同时对称前推，双臂应整体对称前伸

如果方向不对，优先怀疑顺序：

1. `VIRTUAL_DESK_X/Y/Z`
2. 当前侧 wrist coarse/fine tune
3. 旧 home 没重采

## 11. 常见失败现象

### 11.1 X/Y/Z 标定一直被拒绝

常见原因：

- sweep 幅度太小
- 高度变化太大
- 开始结束方向不清楚
- 中途停顿太多

### 11.2 左右手方向镜像错了

优先排查：

- 左右 wrist tracker role 是否写反
- 左右手 neutral pose 是否一致
- fine tune 后是否重新采了 home

### 11.3 一调 wrist offset，机器人马上乱跳

这是正常保护行为的副作用：

- 改 wrist offset 后，旧 home anchor 已失效
- 应先重新 `h` 采 home，再继续验证

### 11.4 看起来“能动”，但姿态总是别扭

通常说明：

- 虚拟 desk 方向大致对了
- 但 wrist tracker 的旋转外参还不够准

这时应在验证阶段用 `j/l/i/k/u/o` 做小步微调。

## 12. 当前限制

- 现在是“虚拟 desk”标定，不恢复物理桌面的绝对世界位置
- 现在 wrist 外参主要靠 neutral pose + 细调，不是完全自动辨识
- SAPIEN 里目前还没有把 `W / V / D / target axes` 画成显式叠加
- `desk -> base` 外参仍沿用 preset，不在这套流程里重标

## 13. 建议工作流

1. 跑 sender
2. 跑 calibration debug app
3. 做 `LINK_CHECK`
4. 做 `VIRTUAL_DESK_X/Y/Z/NEUTRAL`
5. 做左右 wrist coarse
6. 做 `HOME_CAPTURE`
7. 在 `VALIDATE` 里小步微调 wrist 外参
8. 每次微调后重新 `h`
9. 满意后 `s` 保存

保存好的 profile 后续可直接用：

```python
from teleop_runtime import TeleopCalibration

calibration = TeleopCalibration.load_json("path/to/profile.json")
```
