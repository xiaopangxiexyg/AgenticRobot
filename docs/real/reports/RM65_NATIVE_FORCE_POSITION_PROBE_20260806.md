# RM65 控制器原生力位透传验证结论（2026-08-06）

## 决策结论

RM65 第三代控制器、六维力传感器和 DMTAC_M 工具组合已经证明支持
`Force_Position_Move_New` 的 mode-1 `float` 单轴透传。该能力不是空接口：外力能够让
末端沿指定工具轴移动，方向正确，停止请求也得到控制器确认。

但是，当前控制器 M/B/K 配置下的响应不适合直接接入 60 Hz 正式遥操作：低力区域
接近刚性，约 7--9 N 后速度又快速接近上限，形成明显死区和陡峭增益。当前决定是：

- 不把 controller-native float 接入正式遥操、数采或部署；
- 不再通过继续增加人工施力、`limit_vel`、位移或关节漂移上限来调试；
- 正式研究主线回到有界、default-off 的主机侧 Cartesian admittance；
- 原生模式只保留为隔离能力探针和未来 M/B/K 单参数 A/B 的可回退证据。

## 验证边界

- 左臂 RM65-6FB，第三代控制器，API `v1.1.5`；
- 活动工具 `DMTAC_M`，controller payload `0.395 kg`；
- 工具坐标系 Y 轴，`control_mode=[3,1,3,3,3,3]`；
- joint target 固定为启用模式前的当前 qpos；
- 20 Hz command 和 20 Hz state sampling；
- 独立守护进程与冗余进程均可请求 `Arm_Stop`；
- 该探针不属于正式遥操入口，不改变正式 position profile。

API2 的 `rm_force_position_move_t.limit_vel` 是浮点物理量，线速度值直接使用 m/s。
JSON 整数协议的千分之一缩放不能套用到 API2 浮点结构。早期把 `0.008 m/s`
错误放大为 `8.0` 的结果不参与能力判断；修正后测试记录均明确保存
`limit_velocity_sdk_value` 和单位合同。

## 关键结果

修正单位后，`limit_vel=0.008 m/s` 的三次短时测试在约 6 N 相对力保护前产生：

- 工具 Y 位移 `0.39--0.56 mm`；
- 峰值测得速度约 `2.6--4.7 mm/s`；
- 力与工具 Y 位移相关系数约 `0.896--0.920`；
- 横向位移仅 `0.065--0.113 mm`；
- 最大关节漂移 `0.043--0.061 deg`。

`rm65_left_native_force_position_passthrough_20260806_122934.json` 在持续施力下得到：

- 工具 Y 位移 `5.90 mm`；
- 约 `3.5 N` 后开始明显响应；
- `4.2--4.6 N` 时约 `3.5--4.0 mm/s`；
- 5 N 以上快速增强，短间隔测得峰值约 `13.9 mm/s`；
- J1/J4 最大变化约 `1.62/1.68 deg`；
- 原 `1.5 deg` 关节漂移保护触发，Arm_Stop 被确认。

新增实测末端速度保护后，放宽参数的两次压力测试进一步确认了非线性：

| 结果文件时间 | 工具 Y 位移 | 相对力峰值 | 关节漂移 | 实测速率停止 |
| --- | ---: | ---: | ---: | ---: |
| `20260806_124300` | `5.61 mm` | `9.29 N` | `1.45 deg` | `20.40 mm/s` |
| `20260806_124312` | `10.26 mm` | `8.97 N` | `2.72 deg` | `20.19 mm/s` |

两次运动的横向串扰分别约 `0.34/0.50 mm`，姿态漂移约
`0.13/0.14 deg`，未发现错误方向或宏观姿态失稳。两次均由实测速率保护停止，独立和
冗余 Arm_Stop 至少一条返回 code `0`。

按采样区间统计的力--速度关系为：

- `<3 N`：中位速度约 `0.4--0.6 mm/s`；
- `3--5 N`：约 `0.6--1.1 mm/s`；
- `5--7 N`：约 `2.6--3.7 mm/s`；
- `7--9 N`：约 `10.7--13.6 mm/s`，并出现接近 `20 mm/s` 的短时峰值。

20 Hz 状态读取会放大单步速度量化，因此短时峰值不能等同于控制器内部连续速度；但
累计位移、关节变化和保护触发都是真实的。`limit_vel` 是控制器参数，不应被当作独立
硬安全证书。隔离探针必须同时保留实测速度、力、力矩、位移、关节漂移和 Arm_Stop
保护。

## 已排除和未解决事项

已排除：

- float 入口完全无效；
- 工具 Y 轴方向配置错误；
- 观察到的全部位移只是传感器噪声；
- Ctrl+C 或故障后无法让控制器确认停止。

尚未证明：

- 当前死区主要来自 M、B、K 中哪一项；
- 20 Hz low-follow 固定 qpos 透传对死区和批次速度的贡献；
- teach pendant 调整后的原生模式能否在更低接触力下提供平顺、可重复的柔顺；
- 控制器补偿后的 realized qpos、上游 qref、碰撞预测和 action 数据合同能否闭合。

## 防止重复试错的约束

1. 不再复用 `limit_vel=0.016 m/s`、相对力 `10.5 N`、位移 `50 mm` 的放宽命令。
2. 不把增大 `limit_vel` 当作增加柔顺度；它是速度上限，不是 compliance gain。
3. 不在没有实测速度保护时运行 float/spring/force-track。
4. 连续测试之间必须等待力传感器基线稳定；基线标准差超限时保持拒绝启动。
5. 若未来重启原生路线，只允许在 teach pendant 中一次调整一个 M/B/K 参数，并建立
   新的参数快照、结果文件和明确回退；不得同时放宽人工力与运动保护。
6. 在 reference、碰撞、action 和恢复合同未完成前，原生模式不得进入正式遥操。

## 后续推荐路线

保留现有 position profile 作为唯一 motion/reference/IK/admission/dispatch owner，在
IK 前加入 translation-only、低带宽、有界的 Cartesian admittance modifier：

```text
VIVE pose intent
  + fresh tared wrench
  -> bounded admittance state
  -> Cartesian offset / contact rebase
  -> existing IK + collision + dynamics + admission
  -> existing 120 Hz qcmd transport
```

该功能必须 default-off，保留当前 position profile 的原子回退，并继续使用独立的
canonical force emergency-stop guard。详细所有权、数据和分阶段合同见
`todos/rm65_60hz_teleop_force_admittance_feasibility.md`。
