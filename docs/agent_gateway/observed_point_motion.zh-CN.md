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

## 正距离起点退出（观测场景默认）

自 2026-10-29 起（用户决定：起点尽量不受自体观测残留干扰），RealMan 只要配置了
`--safe-motion-observation-roles`，普通 observed safe 移动默认启用正距离起点退出；
需要原 20 mm 严格起点准入时显式传 `--strict-start-clearance`。`--enable-inherited-start-clearance`
仍可显式使用，但没有观测场景时照旧在设备前拒绝。没有观测云时该模式本就不存在，起点准入与
manifest 身份不变。直接调用原 RM65 规划器时数据类默认仍为严格，需要
`RM65GraspTrajectoryPlanningConfig(enable_inherited_start_clearance=True)`。
该选项只适用于含实测障碍云的普通 safe TCP 移动，不作用于抓姿接触链或无观测云的规划。
它进入执行 manifest 身份：含观测场景的默认部署身份因此改变，旧 manifest/grant 不能直接复用；
`--strict-start-clearance` 部署保持原严格身份。

仅本次实测起点已有且严格正距离的近邻对继承原距离下界；硬接触、穿透和新出现的近邻仍拒绝。
非运动臂必须保持固定，运动侧近邻须沿最终稠密执行采样不退化、最终进入原净空，并禁止重新进入。
许可只绑定一次请求，不修改场景余量或污染下一次规划。查询故障单独报告不可用，不落回另一条规划路径。
该模式提供规划接纳能力，仍受原执行、deadline、取消和HOLD约束；离线保存场景结果不证明真机跟踪误差或实际运动资格。

MuJoCo Gateway 的 sensor-shaped planner 镜像同一语义（`start_clearance="inherited-positive-v1"`）：
配置观测场景（`--enable-observed-point-motion` 与 `--safe-motion-observation-*`）时默认启用，
`--strict-start-clearance` 恢复严格起点准入；没有观测场景时 `--enable-inherited-start-clearance`
为用法错误（退出码 2）。直接构造 `MujocoObservedSafeMotionConfig` 时默认仍为 `strict`，身份不变；
只有继承模式进入观测 safe profile 身份，`mujoco.agent_safe_tcp_pose.rgbd.<hash>` 随之改变。
仅原生 RM65 观测 oracle 在正距离下的净空不足可被接纳；接触、穿透、查询不可用和包络拒绝仍返回
`safe_tcp_start_blocked`，详情含 `planning_phase=inherited_start_clearance_admission`。包络约束仍叠加在
继承 oracle 上；稠密参考采样回退返回 `safe_tcp_inherited_start_clearance_regressed`，未离开净空返回
`safe_tcp_inherited_start_clearance_not_exited`，成功路径 metadata 记录 `inherited_start_clearance` 计数。
不使用观测场景的原生 MuJoCo 场景规划不变。

保存场景的只读查询与显式模式用法见
[保存场景 Skill 查询](../development/saved_scene_skill_query.md)。

## 结果与边界

`observed_point_report` 包含源 operation/request hash、点索引、公开参考系、派生目标、实测终点和位姿误差；原执行/评估/HOLD 回执仍完整保留。到达不是抓取或物理任务成功。

MuJoCo 与 RealMan 共享目标解析器、关系数学和语义 Schema；不以 native body pose 或物体标签作为目标输入。
RealMan 的协议和路由接线以离线 mock 验证；不代表实际 CAD/FCL 碰撞规划、部署或真机运动验收，也不授权真机运动。

Real focused 验证：`bash scripts/run_pytest.sh -q tests/test_agent_gateway_real_observed_point.py tests/test_agent_gateway_real_motion_primitives.py tests/test_agent_gateway_real_motion_lifecycle.py tests/test_agent_gateway_real_session_authorization.py tests/test_agent_gateway_real_rm65_cli.py`。
新增 Real 测试替换 native planner 构造并禁止几何 prepare，不启动设备；覆盖源引用、v1/v2、TTL、grant 隔离、
坐标关系、mock lower arrival/HOLD，以及无路径、取消、deadline、规划后状态未刷新、引用过期/淘汰等失败路径。

验证入口：`bash scripts/run_pytest.sh -q tests/test_observed_point_motion.py`。其中离线部分覆盖身份隔离、淘汰、v1/v2 revision 和非法参数；原生部分覆盖实际 RGB-D 引用、两次移动复用、观测后新增障碍拒绝、无效引用零运动和正常 HOLD/close。模型能力及性能提升以另行冻结的多场景 Agent 对照为准。

## 关节限位附近的端点查询

MuJoCo 原 safe planner 仍先尝试原有8个迭代种子。只有全部种子都不满足调用方的位姿容差、
且原2秒端点预算仍有剩余时，才由共享单臂数值层补做一次有界最小二乘查询（最多100次函数评估）。
残差以实际TCP及其装配偏移计算，关节上下限不变；取消和期限在数值回调中检查。
这一增量不改变实时`solve_step`/`solve_iterative`的默认实现，也不构造第二个规划或执行所有者。

新候选必须重新通过原FK容差、碰撞、路径质量、轨迹复验及执行准入；已经得到合格位姿、
但被碰撞或路径质量条件拒绝的请求，不启动这次补搜。新入口和原绝对safe入口共用该修复。
`endpoint_search_diagnostics`保留未收敛种子的残差、求解状态及预算；历史`endpoint_attempts`
仍表示达到位姿容差后进行碰撞／路径质量筛选的候选，不能把空列表解释为没有尝试。

修复依据是保存的准备动作目标和实测qpos：原迭代在第三关节上限停滞，而有界查询能找到
满足该动作原容差的数值候选。数值可达不等于路径可执行或完整场景成功，后两者另做原生验收。

## 包络预检与失败归因

标准 MuJoCo safe 配置同时约束单次激活位移和相对会话初始化 SafetyReference 的每轴±10cm区域。
分成多个短步只满足前一种限制；正常HOLD不重置后者。实验准备和Agent动作都必须服从同一参考。

safe lifecycle在采集障碍场景、IK和路径搜索前复用下层投影，判断目标允许的位姿容差区域
是否与包络相交。完全无交集返回`safe_tcp_goal_outside_safety_envelope`并沿原HOLD链收尾；
名义目标越界但容差仍相交时继续原规划，绝不把投影后的坐标替换成新目标。
`preserve_target`使用原3mm/1度搜索容差，默认使用原profile到达容差，均未放宽。
SO(3)相对旋转接近π、名义投影不唯一时，预检不作无解判断，仍由原候选及全路径安全检查决定。

IK达到容差后，如果所有候选均被包络拒绝，返回`safe_tcp_target_envelope_blocked`；
`safe_tcp_target_blocked`仍用于实体/观测障碍或混合拒绝，`safe_tcp_endpoint_search_exhausted`
仍表示未找到位姿合格候选。详细诊断保留在下层错误中，不改变轨迹hash、Skill参数或注册目录。
新增测试覆盖容差相交、非有限参数、局部旋转轴、π附近投影不唯一，以及左右臂实际分段运动后累计越界
的零新增dispatch拒绝与正常HOLD/close；不等于复杂场景或真机验收。

## 场景资格与启动失败收尾

实验布局必须明确使用的坐标系，通过已装配的`TargetFrameProfile.frame_graph`转换至
场景world；不能把公开station坐标直接写入场景。首次运动前应复用原碰撞oracle检查完整机器人
与场景，不能只检查物体之间的AABB。零运动几何资格不代表IK、路径或任务成功。

Gateway启动失败时，即使application尚未标为started，关闭仍会调用原runtime的幂等close，
让启动回滚遗留的资源能够再次释放。已经启动的application仍必须先取得HOLD。
`runtime.capture_lifecycle_diagnostics()`只在串行生命周期边界读取原所有者的pending/terminal状态，
不执行I/O、不授予运动权限，也不替代HOLD或正常关闭回执。测试记录应分别保留启动错误、
关闭错误、资源释放状态与HOLD证明；例如启动几何拒绝后正常释放，不应记作运动验收通过。

零运动检查入口为`python -m scripts.probes.qualify_mujoco_scene TASK_CONFIG --output NEW_REPORT.json`。
它使用原prepared target与整机collision oracle，保留原预防余量和1mm硬穿透界限，另外报告
物体/支撑接触的初始深穿透。结果中的距离可能是裁剪后的下界，不冒称精确余量。
此工具不创建执行所有者或渲染器，不前进physics，检查前后time/qpos/qvel必须一致；
报告不覆盖，构造异常保留为失败并重新抛出。对应原生检查：
`bash scripts/run_pytest.sh -q tests/test_mujoco_scene_qualification.py`。
