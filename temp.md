我建议把它做成一个 pose-aware reachability atlas，不是简单的 3D 位置盒子。当前代码里已有很好的基础：双臂基座间距约 0.525m，左右基座在 desk/world 下偏 ±0.2625m，见 [rm65_dual_arm_hand.urdf (line 8)](/home/zc/HandMotion/Developing/DexTeleop/assets/rm65_inspire_tac/rm65_dual_arm_hand.urdf:8)；当前运行时双臂碰撞 IK 已经用 dual_arm_hand_proxy_open 场景、手部 proxy、桌面/相机杆碰撞，见 [pipeline.py (line 287)](/home/zc/HandMotion/Developing/DexTeleop/rm65_inspire_tac/pipeline.py:287) 和 [dual_arm_collision_aware.py (line 209)](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/retargeters/dual_arm_collision_aware.py:209)。所以可达空间最好直接复用这套 URDF + IK + collision profile。
核心定义
输入不要只设为 xyz，而应是：
query(left_pose, right_pose, frame="tool_tcp/arm_endpoint", current_q optional)
=> reachability_status + score + metrics + nearest_safe_pose
其中每个 pose 是 6D/7D 位姿，统一转换到 desk/world 坐标；内部评估用当前 solver frame，例如 arm_link6 或 tool_tcp 经过 TCP 标定后的 frame。现有 arm_link6 -> hand_base 有 +0.009m 偏移，见 [arm_frames.py (line 14)](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/arm_frames.py:14)，这个必须固定，否则 UMI 采集区会和机器人真实控制点错位。
三色分区建议
绿色：高质量可达区。IK 能稳定收敛，末端误差建议 <5mm / <2deg，关节余量 >10deg，最小碰撞距离 >5cm，附近网格的 IK 分支连续，没有明显奇异或大关节跳变。UMI 采集主工作区只允许绿色。
黄色：可过渡区。目标可以通过小偏差投影到安全 pose，例如平移修正 <3cm、姿态修正 <10deg；或关节余量 3-10deg、碰撞距离 3-5cm。黄色可允许短时经过，但数据要打标，训练集里可降权或剔除。
红色：不可采集区。IK 无解、投影距离过大、碰撞距离 <3cm、接近奇异、需要跨 IK 分支跳变、或者左右臂互相遮挡/碰撞。在线采集时应阻止、冻结、reanchor 或把 UMI 映射 scale 降低。
离线构建流程
固定模型版本：使用 assets/rm65_inspire_tac/rm65_dual_arm_hand_proxy_open.urdf 作为采集 atlas 的默认场景，记录 URDF hash、scene profile、TCP profile、base distance、joint limit、碰撞阈值。

单臂 atlas：对左右臂分别采样 xyz + orientation_bin。位置网格可先 2-3cm 粗采样，边界处细化到 1cm；姿态不要全 SO(3) 暴力均匀采样，优先采 UMI 常见手腕朝向、抓取 approach cone、roll 离散角。

每个 pose 多种 seed 解 IK：home seed、邻近 voxel 最优 q、随机低差异 seed、wrist flip/branch seed。保存最优解，而不是只保存 bool。

双臂 atlas 分两层：先查左右单臂是否可达，再用现有 dual-arm collision IK 对 (left_pose, right_pose) 做 coupled 评估。完整 12D pose 空间不能密铺，建议对常见 UMI 任务区、双手相对距离、跨中线动作、handover/双手抓取姿态做重点采样，并在线缓存真实查询结果。

每个采样点保存指标：status, score, q_best, pose_error, joint_margin, collision_min_distance, manipulability, branch_id, nearest_safe_pose, failure_reason。格式可用 zarr/hdf5/npz + metadata.json，在线查询用 KD-tree 或规则网格插值。

在线使用
现有 workspace_clip 只是 AABB，默认大约 x[-0.60,0.60], y[-0.80,0.80], z[0,0.90]，见 [presets.py (line 34)](/home/zc/HandMotion/Developing/DexTeleop/teleop_runtime/configs/presets.py:34)。新的 atlas 应替代或补强它：
UMI pose -> desk/tool_tcp pose
        -> atlas 快速查色
        -> 黄色/边界时跑精确 dual IK
        -> 绿色通过，黄色投影/警告，红色阻止或 reanchor
        -> 写入 dataset metadata
数据集里建议记录 reachability_status、reachability_score、projection_delta_m/deg、min_collision_distance_m、nearest_joint_margin_deg。这样后处理时可以直接过滤“红色”和长时间“黄色”的片段。
我会优先落地的版本
先做一个 workspace_atlas 离线生成器，只支持 dual_arm_hand_proxy_open + arm_link6/tool_tcp，输出左右单臂 atlas 和在线双臂 evaluator。第一版不追求把全空间采满，而是覆盖 UMI 采集盒、桌面上方主操作区、左右手交互区。等采集几批数据后，再用真实 UMI 轨迹反向加密采样边界区。这样最实用，也最容易和现有双臂 IK/回放诊断对齐。