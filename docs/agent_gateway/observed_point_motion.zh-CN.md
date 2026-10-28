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

RealMan 在原 `real-rm65` 入口显式选择 `--authorized-action move_to_observed_point`，同时要求
`--enable-back-projection`、`--safe-motion-observation-roles` 和完整相机装配。
Route 为 `real.rm65.agent_skill_observed_point@1`；未选择该 action 时不发布。
仍须原会话 operator confirmation、确切 Route/profile/envelope 和每次请求的新鲜、单次使用 grant；
原 `move_tcp_to_pose_safe` 的授权不能授权这个 action。`component_id` 只选择已授权的一侧，
双侧部署也每次仅操作一个 arm。调用者只提交上面的五个字段，不能覆盖速度、目标位姿、参考系标识或安全策略。

Real profile 保留激活时实测 TCP 朝向，固定 profile 速度为 `0.2`，使用 `preserve_target`，
复用已有 `ObservedSafeMovePlanner`、lower dispatch 与 HOLD；缺少 RGB-D safe planner 时装配失败。
对于 DMTAC 装配，可显式传入
`--captured-robot-filter-config teleop_runtime/configs/skill_deploy/rm65_dmtac_capture_self_filter_cad_v1.json`
选择已版本化的 CAD 自观测过滤候选。默认 15 mm 过滤保持不变；该选项只处理捕获点云，
不替换原严格碰撞模型、起点净空或运动门禁。实际部署须用现场装配和标定复核过滤身份。
保存点的年龄上限直接沿用该 planner 的 `SafeMotionObservationConfig.maximum_scene_age_sec`，
默认仍为 30 秒，不增加 TTL 开关。`back_project@2` 使用原 `acquired_at`；`back_project@1`
没有传感器时间戳，以当前时间减去其 live operation 的 `started_at`，再加上部署相机的
`max_frame_age_ms`，作为保守年龄上界，不伪造采集时间或忽略相机缓存年龄。
缺失、失败、异客户端、异会话、异 manifest/Route/binding/profile、错误 frame graph、越界索引、
过期或未来引用均拒绝。规划结束、lower activation 前再次检查保留身份与年龄，失败进入原 HOLD。
每次移动的碰撞观测仍由 safe planner 重新采集、校验 TTL、静止性和几何，不复用保存点作为障碍场景。

## 可选的正距离起点退出

RealMan 继续默认拒绝不足原 20 mm 净空的起点。部署者可显式选择
`--enable-inherited-start-clearance`，同时配置 `--safe-motion-observation-roles`；
直接调用原 RM65 规划器时使用
`RM65GraspTrajectoryPlanningConfig(enable_inherited_start_clearance=True)`。
该选项只适用于含实测障碍云的普通 safe TCP 移动，不作用于抓姿接触链或无观测云的规划。
它进入执行 manifest 身份，旧 manifest/grant 不能直接复用；默认关闭时身份保持原值。

仅本次实测起点已有且严格正距离的近邻对继承原距离下界；硬接触、穿透和新出现的近邻仍拒绝。
非运动臂必须保持固定，运动侧近邻须沿最终稠密执行采样不退化、最终进入原净空，并禁止重新进入。
许可只绑定一次请求，不修改场景余量或污染下一次规划。查询故障单独报告不可用，不落回另一条规划路径。
该模式提供规划接纳能力，仍受原执行、deadline、取消和HOLD约束；离线保存场景结果不证明真机跟踪误差或实际运动资格。
MuJoCo Gateway 的独立 sensor-shaped planner 不通过此 RealMan 选项改变默认接纳。

## 结果与边界

`observed_point_report` 包含源 operation/request hash、点索引、公开参考系、派生目标、实测终点和位姿误差；原执行/评估/HOLD 回执仍完整保留。到达不是抓取或物理任务成功。

MuJoCo 与 RealMan 共享目标解析器、关系数学和语义 Schema；不以 native body pose 或物体标签作为目标输入。
RealMan 的协议和路由接线以离线 mock 验证；不代表实际 CAD/FCL 碰撞规划、部署或真机运动验收，也不授权真机运动。

Real focused 验证：`bash scripts/run_pytest.sh -q tests/test_agent_gateway_real_observed_point.py tests/test_agent_gateway_real_motion_primitives.py tests/test_agent_gateway_real_motion_lifecycle.py tests/test_agent_gateway_real_session_authorization.py tests/test_agent_gateway_real_rm65_cli.py`。
新增 Real 测试替换 native planner 构造并禁止几何 prepare，不启动设备；覆盖源引用、v1/v2、TTL、grant 隔离、
坐标关系、mock lower arrival/HOLD，以及无路径、取消、deadline、规划后状态未刷新、引用过期/淘汰等失败路径。

验证入口：`bash scripts/run_pytest.sh -q tests/test_observed_point_motion.py`。其中离线部分覆盖身份隔离、淘汰、v1/v2 revision 和非法参数；原生部分覆盖实际 RGB-D 引用、两次移动复用、观测后新增障碍拒绝、无效引用零运动和正常 HOLD/close。模型能力及性能提升以另行冻结的多场景 Agent 对照为准。
