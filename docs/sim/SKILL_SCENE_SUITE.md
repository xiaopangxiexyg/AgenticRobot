# Skill 场景与任务扩展

这是一组显式启用的场景配置与任务材料。它复用 SceneSpec v2、已有 DMTAC
装配、任务配置加载器、三相机与原场景资格检查，不增加执行 owner、观测 owner、
控制循环或新的 Agent Skill。旧默认场景、接触参数与碰撞保护保持原值。

## 内容与用途

配置入口：`sim/configs/scene_suites/rm65_dmtac_skill_v1.json`。
生成器：`sim/mujoco/scene_suite.py`。

13 类布局各有 3 个固定 XY 位置变体，共 39 个场景实例。
每类布局有 3 个任务，共 39 个任务定义、117 个场景/任务实例。
变体移动整组布局，保持物体间的语义关系；没有每次启动随机换图或修改旧结果。

| 布局 | 主要变化 | 用途 |
|---|---|---|
| shape_palette | 方块、圆柱、球、椭球、长条与扁盒 | 形状辨认、表面定位 |
| same_color_shapes | 同为蓝色的球、方块和圆柱 | 避免仅按颜色选择目标 |
| same_color_sizes | 三种大小的蓝色方块 | 尺寸比较、指代消歧 |
| rotated_neighbors | 斜放长条与盒子 | 朝向理解、近邻避障 |
| dense_clutter | 多个形状密集摆放 | 局部自由空间、引用点复用 |
| partial_occlusion | 小目标与灰色高盒 | 多视角可见性比较 |
| narrow_corridor | 两块长障碍与中间目标 | 通道理解、规划或合理拒绝 |
| raised_support | 小方块放在独立台阶上 | 支撑关系与实测高度 |
| placement_regions | 绿色和黄色平面区域 | 目标区域辨认、后续放置 |
| long_object | 长条、邻近圆柱与目标区 | 后续整体占用与放置 |
| bilateral_targets | 两侧不同颜色方块 | 左右手依次执行 |
| ycb_proxy_palette | 已审尺寸/质量的四种简化代理 | 几何跨度与证据边界 |
| wrist_supports | 保持 home 与相机装配，远侧高台承托两个目标 | 真实腕视角与主视角的目标身份比较 |

本批有 23 个原型，覆盖 box、cylinder、sphere、ellipsoid。尺寸是完整 XYZ
范围，质量为仿真候选值，惯量由原有形状实现计算。YCB 代理复用此前审阅的
尺寸/质量数值；没有自动绑定扫描视觉网格，因此不得把彩色几何代理说成逼真的
草莓、柠檬等扫描外观。本批不下载资源、不修改共享网格或真实材料参数。

自由放置的椭球以最短轴竖直摆放。初次冷启动物理检查发现另一轴向会使椭球在
1 秒内滚动约 57 mm；本批通过修改合成几何的轴向尺寸修正，并用原 Kernel 的
冷启动稳定性回归验证，不通过增大摩擦、固定物体或改变控制器保持它的位置。

遮挡和通道是布局意图，不是验收结论。实际相机可见性、规划可达性及任务完成
必须分别测量；不能由场景名称推断。障碍、台阶和区域标记都是普通物理自由体，
没有 weld、隐式锚定或无限质量。初始几何通过不等于动力学稳定或抓取成功。

## 复用原入口

在项目根目录、已配置的原 MuJoCo Python 环境中运行：

```bash
python -m sim.mujoco.scene_suite --list
python -m sim.mujoco.scene_suite \
  --case same_color_shapes --variant 0 --output /tmp/skill-scene-example
python -m scripts.probes.qualify_mujoco_scene \
  /tmp/skill-scene-example/task.json \
  --output /tmp/skill-scene-example/initial-qualification.json
```

输出目录必须不存在。生成器不覆盖保留材料；每次独立运行使用新目录。
`task.json` 可传给现有 `MujocoSimHILTargetConfig.task_config_path` 和 Gateway
工厂，不需要新的设备/仿真启动器。生成配置只沿用现有 CubeLift 配置作为装配与
初始化入口，不会把 CubeLift 的 lift 成功条件当作这些语言任务的评分器。

输出文件：

- `scene.json`：完整标准 SimulationContract / SceneSpec v2。
- `task.json`：继承原任务装配，精确绑定该 scene 文件 SHA-256。
- `public_tasks.json`：只含任务 ID、阶段、语言目标，可供 Agent 选择任务。
- `evaluation_only.json`：测试端的对象名和判定要求，不发送给 Agent。

场景编辑使用原 SceneSpec 的 `mujoco_world` 坐标，且在 metadata 中显式标明。
这是环境构造输入，不是运动目标。运行时目标仍应来自相机 RGB-D、分割与保存的
实测引用，并经过公共坐标转换；不能读取原生物体 pose 来替代 Agent 定位。

原任务 reset 会重置 focus 物体的姿态。本生成器因此明确拒绝旋转 focus，避免
配置宣称有旋转而初始化时静默归零；本批旋转仅施加于其它物体。台阶目标的
初始支撑高度显式传给原 reset，原 URDF 桌面本身不变。

focus 的语义名称仅用于编写布局，生成后内部 body 名沿用原任务的 `debug_cube`，
从而保留既有夹爪接触 profile 的精确 geom 绑定。球、圆柱等可以是这个 body 的
实际形状；其它物体各自有独立 body。测试端引用会同步映射成实际 body 名，
Agent 仍只用自然语言与 RGB-D 证据选择目标，不消费这些内部标识。

## 任务阶段与证据

- `observation`：识别、尺寸/关系和多视角可见性。需用真实渲染 RGB-D 验证，
  静态 JSON 或无运动几何检查不代表 Agent 已理解。
- `safe_motion`：表面点上方悬停、同引用不同偏移、双手依次悬停。
  仍需原 safe 规划、执行、实测稳定与 HOLD/close 证明。超出当前包络或受障碍
  阻挡时保留原拒绝，不能移动 home 或扩大包络来制造通过。
- `payload_pending`：抓取后搬运与稳定放置，仅为待验证任务材料，不能作为当前
  可执行成功能力发布。当前非 box 载荷支持、携物关系和接触失败仍需各自验收；
  也不能因空手悬停通过而认为放置已通过。

测试端可以使用 simulator truth 做评分与诊断，但不得把私有对象位姿、segmentation
ID 图、目标 body 名或答案位置作为 Agent 工具输入。实际 RGB-D、关节/TCP 观测和
通过既有工具保存的引用可复用。比较不同 Skill 时使用相同布局、变体和任务文本。

## 验证入口

```bash
bash scripts/run_pytest.sh -q tests/test_mujoco_skill_scene_suite.py -m 'not integration'
flock -n /path/to/existing/native-resource.lock \
  bash scripts/run_pytest.sh -q tests/test_mujoco_skill_scene_suite.py -m integration
```

第二条使用当前开发/验收约定的同一 native 资源锁，不能另开一把锁绕过串行。
原生资格用例禁止 `mj_step` 与 renderer 构造，经原 prepared-target 与原 collision
oracle 检查全部 39 个场景，并验证 time/qpos/qvel 不变。它们不创建执行 owner，
也没有运动 HOLD 或抓放成功可供计分。实际结果与固定源码 SHA 由该批验证报告记录。

第一轮三视角渲染已经显示：原 home 下桌面中心目标主要由 front 看到，不能据三台
相机存在就宣称腕部视角有用。`wrist_supports` 是独立增加的高台布局；其腕部可见性
也必须根据原生渲染和有效深度像素验收，不改变相机外参、home 或构造假图像。
