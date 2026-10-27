# Contact-GraspNet 保存真机观测的抓姿路径筛选（2026-10-27）

本批只回答一个问题：Contact-GraspNet 从实测 RGB-D 生成的目标抓姿，经过与
GraspMoE 相同的 DMTAC v5 几何、RM65 IK、完整场景避障和路径质量检查后，能否
提高完整路径覆盖。**本批没有仿真运动、Agent 调用或真机动作。**

## 固定条件与来源

- 三个保存状态：`state_0001-r0`、`state_0002-r0`、`state_0006-r0`。输入为同一采集
  状态的 SAM3 mask、对齐 RGB-D、内参、光学坐标到 `canonical_station` 的变换与双臂
  qpos。目标来自保存的传感器观测，不从仿真物体真值构造。
- 共同运动源码 `193a4a493a3dd4bde6c7af66a846179b59edcecf`，基线为
  `development@da50e15846663989dbdb925855ce6e38f274f6ab`。每个臂、状态和
  接触参考面单独使用原 20 秒抓姿候选与完整路径搜索预算；不同接触参考面不合并
  成一次更宽预算。
- 模型输入、源码、脚本、检查点和资产绑定在各批 `freeze.json` 中固定。所有正式
  批次按声明分母执行，来源哈希匹配，worker 正常退出、回收。原失败批次保留，
  没有覆盖或自动重跑。

本机原始证据位于
`/home/zc/HandMotion/Developing/temp/skill-grasp-benchmark-20261027/`。
该目录是独立运行产物，不属于本提交；若要在其他主机重现，须另行提供保存的
观测和模型资产，不能只使用本文计数。

## 模型运行与输入对照

原始隔离运行首先暴露了 vendor 导入路径缺依赖以及 TensorFlow 2.20/Keras 3
无法提供旧版 `tf.compat.v1.layers.batch_normalization` 的问题。隔离 worker 在
导入 TensorFlow 前设置 `TF_USE_LEGACY_KERAS=1` 后，官方模型与原检查点完成
3/3 次正式前向，模型会话关闭，进程退出并回收。该兼容修复尚未进入生产服务。

| 输入给生成器的点云 | 三个状态的精确目标接触候选 | 阶段 |
| --- | --- | --- |
| 保存的广域场景 | 0、0、0 | `contact-graspnet-fullpath-prep-03` |
| 目标加局部支撑背景 | 0、0、0 | `contact-graspnet-fullpath-prep-04` |
| 保存的 SAM3 目标点 | 20、43、32 | `contact-graspnet-fullpath-prep-05` |

目标接触使用模型原有的 0.1 mm 筛选阈值，并在路径重放时与保存的目标点再次
核对。候选数是模型输出证据，不等于几何可用、完整路径或抓取成功。

## 共同路径链结果

目标点输入的候选使用两个既有且已审的 Contact-GraspNet 接触参考面：
`finger_base`（58.4 mm）与 `fingertip`（103.4 mm）。变换为
`T_station_from_optical × T_optical_from_grasp × T_source_to_DMTAC`，随后由原
v5 候选过滤、场景碰撞、臂侧 IK、完整路径与 C2 运动质量检查负责准入。

| 原始 CGN 条件 | 完整路径 | 最终状态 |
| --- | ---: | --- |
| `finger_base` | **0/6** | 无可接受路径 |
| `fingertip` | **0/6** | 无可接受路径 |

正式批次 `contact-graspnet-v5-fullpath-01` 的冻结哈希为
`300f815c2d887d65c78ecd4c7693eedf65ac329df329076923664c90a3321eef`；
资格 4/4，正式 12/12，未启动 0，worker 退出码 0、正常回收。资格重复首个状态，
不能算作新的独立样本。

独立诊断 `contact-graspnet-v5-diagnostic-02/03/04` 各有 4/4 个首状态资格
记录和正常清理。`finger_base` 的候选全部未通过夹指接触高度要求；`fingertip`
分别有左 61、右 66 个候选通过高度，但没有候选通过 native 腕部粗可达性检查。
最近的必要腕部半径超额仍为左 **142.0 mm**、右 **76.6 mm**，不是微小阈值误差。
`state_0002` 的右臂另有两次端点和完整路径查询，两次均被原运动质量门拒绝。

还测试了一个单独标记的**派生候选**条件：由官方接触点、预测张口量和源码中
的 5 mm 额外开口约定，绕抓姿局部 Y 轴旋转 180°，重新计算位姿原点以保持同一
实测接触点。它不是模型原始预测。接触公式在重放前后均通过数值检查，但派生
候选在三个状态都未通过原目标接触几何：`finger_base` **0/6**、`fingertip`
**0/6** 条完整路径。批次 `contact-graspnet-opposite-approach-01` 的冻结哈希为
`8c77747ca0d668bb8d7d493427c83a5dab9c296c18f531954e7da99cdd2902da`；
资格 4/4、正式 12/12，来源与清理检查通过。没有放宽碰撞、可达性或路径质量门。

同三个状态的已固定控制对照：GraspMoE 完整路径 **5/6**，M2T2 广域输入
**0/6**，M2T2 目标加局部背景 **3/6**。这是离线路径覆盖，不能写成 MuJoCo
抓取、Agent 任务或真机物理成功率。

## 处理结论

Contact-GraspNet 的可重现前向问题已在隔离环境修好；目前没有通过完整路径
证据，**不切换默认生成器，不新增公共 Skill 或 Route**。若再研究该模型，下一批
应以独立固定的生成器覆盖假设为起点，再使用同一 v5/IK/路径链验证。官方
`forward_passes>1` 先预处理一次点云，再复制相同点集并以
`is_training=False` 推理，不能把简单增加该参数当成候选多样性改进。

近期主线仍是让下层 Skill 复用保存的真实可获取观测，完成目标引用、关系求解、
安全执行和阶段反馈，减少 Agent 对像素、长串坐标及重复观测细节的处理量。
