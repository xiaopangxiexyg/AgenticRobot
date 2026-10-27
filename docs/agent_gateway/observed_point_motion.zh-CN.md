# 保存观测点的关系动作：motion.move_to_observed_point@1

这是显式启用的 Agent 抽象入口，不是新的规划器。它把“已保存的实测表面点＋偏移”交给原 safe 执行链；原始状态读取、关系计算和绝对 safe 移动仍可独立调用。

## 输入与职责

```json
{
  "component_id": "right_arm",
  "observation_operation_id": "<本客户端已完成的 perception.back_project operation_id>",
  "point_index": 0,
  "offset_xyz_m": [0, 0, 0.08],
  "offset_frame": "reference"
}
```

- v1 反投影只有一个点，索引为 0；共享解析器也支持 v2 点数组，当前 MuJoCo 原生接线验收使用 v1。
- `reference` 使用部署公开运动参考系；`tool` 使用激活时实测 TCP 的轴。保持激活时的 TCP 朝向，不控制夹爪。
- 引用由原 Gateway operation store 持有，核验 client/session/manifest/Route/binding/profile 和结果 Schema。缺失、淘汰、失败或跨身份引用明确拒绝；不新建世界状态缓存。
- 保存的点是历史表面位置，不是物体中心、6D 位姿或实时跟踪。机器人运动后可复用；物体移动后应重新观测。v1 未提供采集时间，不伪造 timestamp；v2 保留原 RGB-D 采集时间与 revision。
- 校准图负责坐标变换；共享 `robot_motion.spatial_goals.compute_spatial_goal` 计算关系目标；原 safe lifecycle 负责最新 TCP、工作空间/相对幅度、安全包络、RGB-D 障碍场景、IK、路径、执行、收敛和 HOLD。保存目标不意味着保存的障碍图可以无限复用。

## 显式装配与兼容性

`build_mujoco_agent_tcp_pose_gateway_session(..., enable_observed_point_motion=True, enable_safe_tcp_pose=True, artifact_storage=..., safe_motion_observation=...)`。
必须配置已有 sensor-shaped safe planner；未配置时构造失败。新开关默认关闭，不改变原 safe@1 参数、保护条件或旧部署的目录哈希。

RPent 通过原 `build_motion_attachment(..., motion_skill_ids=("motion.move_to_observed_point",), include_execution=True, ...)` 显式选择；仍须发现并选择该 Route 的确切安全配置。新入口没有自己的 Agent 调度器或执行环。RPC、GUI 批准、取消、HOLD 和关闭沿用通用扩展。

## 结果与边界

`observed_point_report` 包含源 operation/request hash、点索引、公开参考系、派生目标、实测终点和位姿误差；原执行/评估/HOLD 回执仍完整保留。到达不是抓取或物理任务成功。

当前实现为 MuJoCo 显式 Route。目标解析器、关系数学和语义 Schema 不依赖仿真物体真值；RealMan 可接入同一输入合同和现有 safe owner，但**尚未发布/验收该真机 Route**。不以 native body pose 或物体标签作为目标输入，不授权真机运动。

验证入口：`bash scripts/run_pytest.sh -q tests/test_observed_point_motion.py`。其中离线部分覆盖身份隔离、淘汰、v1/v2 revision 和非法参数；原生部分覆盖实际 RGB-D 引用、两次移动复用、观测后新增障碍拒绝、无效引用零运动和正常 HOLD/close。模型能力及性能提升以另行冻结的多场景 Agent 对照为准。
