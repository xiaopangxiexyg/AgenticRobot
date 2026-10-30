# 夹爪-物体关系与夹爪悬停路径查询 Skill

两个只读 Skill 在 RealMan 真机和 MuJoCo Gateway 上使用同一份实现
（`robot_motion/tool_entity_scene.py`、`agent_gateway/profiles/tool_entity.py`）。
目标平台只提供采集、分割、运动学和规划端口。两者默认关闭，路由均为
`read_only` / `t0_read_only` / `cancel_only`，不移动机器人，也不授予运动权限。

| Skill | 语义 | 输入 | 输出 |
|---|---|---|---|
| `perception.describe_tool_object_relations@1` | `agenticrobot.relation.tool_object.v1` | 1–4 个物体（`label` + `language_prompt`）、相机 role（`null` 为全部）、`min_score` | 每个视角和按 label 融合后的关系：`between_fingers[_in_contact_band]`、`in_contact_band`、`near`、`far`、`insufficient_surface`，以及稳健距离、未测到的视角及原因 |
| `perception.query_tool_hover_paths@1` | `agenticrobot.relation.tool_entity_hover.v1` | 一个 `perception.capture_surface_anchor@2` Artifact、夹爪盒尖端相对锚点的水平偏移区间、高度间隙区间、`preserve` 或 `tool_axis_down`（可带倾斜锥） | 第一个通过检查的 `target_tcp_pose_xyz_quat_xyzw`、规划终点、实测（重新计算）的偏移/间隙/倾角，以及每个候选的尝试记录 |

## 语义要点

- 关系：每个视角的物体 mask 先腐蚀 2 px，只取 1.5 m 内的有效深度，体素化后保留最大的连通
  表面簇；夹爪放在该帧采集时关节角对应的 TCP 处。夹爪几何是 DMTAC GraspGenX v5 筛选盒
  （最大张开）加 URDF 张开状态下的 DMTAC 传感器碰撞盒（仅用于接触判断）。没有审查过
  DMTAC_M 工具的臂列在 `unbound_sides`，不会被静默省略。接触带只是接近证据，不是接触、
  力闭合或抓取成功的判定。
- 悬停：一次请求采集一次观测场景，所有候选从同一测得状态出发。候选顺序为：离当前工具状态
  最近的点、中心、面中心、Halton 填充；候选按到达容差内缩，规划终点按**原始**区间重新测量。
  搜索总预算 20 s，每个候选 5 s（2026-10-29 起；倾斜锥悬停在 3 s 时经常来不及完成）。路由
  上限 40 s。`query_incomplete` 表示预算内没有完成，不能证明不可行。
  部署可用 `--tool-hover-candidate-budget-sec`/`--tool-hover-region-budget-sec` 放宽
  （真机和 MuJoCo CLI 相同；候选不超过总预算，上限 120 s/240 s，需同时启用悬停查询）。
  预算进入悬停 Profile 哈希，路由上限为总预算 + 20 s 采集/点云余量，结果校验要求使用本部署预算。
- 结果中的位姿使用与 `motion.move_tcp_to_pose_safe@1` 相同的公共参考系（注册时强制检查）。
  上层把 `target_tcp_pose_xyz_quat_xyzw` 原样提交给该运动 Skill；运动 Skill 会在新的场景上
  重新规划和检查，悬停查询本身不保留计划。
- 区间只对悬停规划器实际到达的位姿 `planned_endpoint_pose_xyz_quat_xyzw` 成立。运动 Skill
  默认的到达容差（审查过的配置为 10 mm、5°）可能让夹爪尖端或最低点偏出约 2 cm。需要执行后
  仍在区间内时，提交 `planned_endpoint_pose_xyz_quat_xyzw` 并加
  `pose_goal_policy=preserve_target`（按 3 mm、1° 的精确规划界限瞄准这个已到达过的位姿）。
  不要对 `target_tcp_pose_xyz_quat_xyzw` 使用 `preserve_target`：精确朝下等提议位姿可能超出
  1° 内的 IK 范围（MuJoCo 孪生场景实测 `safe_tcp_endpoint_search_exhausted`）。实测到达容差
  仍由部署决定。
- 公共参考系的 +Z 必须是竖直方向（注册时检查）；间隙、足迹和 `tool_axis_down` 都按它计算。

## 启用

真机（`agent_gateway/profiles/real/gateway_cli.py`）：

- `--enable-tool-object-relations`：需要 `--enable-back-projection` 和 `sam3-service` 分割
  （`--sam3-timeout-sec` 不超过 45 s），并且至少一只臂装有审查过的 DMTAC_M 工具。
- `--enable-tool-hover-queries`：另外需要 `--enable-surface-anchor-v2`、
  `--safe-motion-observation-roles` 和已授权的 `move_tcp_to_pose_safe` 动作。
- 采集使用 `capture_aligned_rgbd(require_stationary=True)`，臂在运动时会拒绝。悬停规划使用
  安全移动规划器配置的独立实例（与运动 Skill 的起点间隙模式和端点排序相同），不占用运动
  规划器的锁，也不进入运动读取窗口。

MuJoCo（`agent_gateway/profiles/mujoco/gateway_cli.py`）：

- `--enable-tool-object-relations`：需要 `--artifact-root`、分割后端和 `--enable-safe-tcp-pose`
  （冻结多视角采集属于 safe TCP 规划 owner）。
- `--enable-tool-hover-queries`：需要 `--enable-surface-anchor-v2` 和
  `--enable-observed-point-motion`（观测 RGB-D 安全场景）。
- 在 HOLD、无活动 operation 时冻结多视角采集。悬停使用私有、从不激活的 safe TCP 规划器，
  其观测场景、碰撞 oracle、起点间隙模式和安全包络与 `move_tcp_to_pose_safe@1` 相同。
- `--motion-side left` 仍只支持基础 TCP/状态 Route。

## MuJoCo 孪生场景中的悬停碰撞（2026-10-29 排查）

默认任务 `rm65_right_dmtacm_blue_cube_70mm_agent_v1` 中，右臂初始工具 +Z 水平指向 −x。
用 `preserve` 在立方体（x=−0.272, y=0）上方悬停时，8 个候选全部被拒（`safe_tcp_target_blocked`）。

1. 固定碰撞模型：最近碰撞对是 `right_hand_proxy_0 / camera_pole_0`（穿透 2.7–4.8 cm）。
   在相同 IK 端点上，用孪生场景自身的碰撞几何（`mj_geomDistance`）测得 `right_arm_link5`
   穿入中央相机立柱 2.4–3.4 cm。立柱和桌面的位置、尺寸与站点 URDF 一致，所以这是真实碰撞：
   水平工具位于立方体上方时，腕部正好落在立柱处。
2. 碰撞配置：把场景 profile 从 `dual_arm_hand_proxy_dmtac_grasp` 换成 `..._dmtac_strict`，
   或把起点间隙改为 `strict`，结果不变。`tool_axis_down`（精确向下）在第 2 个候选找到路径；
   45° 倾斜锥会向立柱一侧倾斜，仍然全部被拒。因此不需要修改碰撞配置。

原生 MuJoCo 端到端测试（`tests/test_agent_gateway_mujoco_tool_entity.py`）覆盖：关系查询 →
表面锚点 → `preserve` 被拒 → `tool_axis_down` 找到路径 → 把 `planned_endpoint_pose_xyz_quat_xyzw`
以 `preserve_target` 交给 `motion.move_tcp_to_pose_safe@1` 执行，并在原生几何上检查夹爪盒最低点
高于立方体顶面的间隙（实测 6.2 cm，请求 3–10 cm）、尖端在立方体上方、无接触。
