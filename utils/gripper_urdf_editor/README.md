# Gripper URDF Editor

这是一个离线、面向双指平行夹爪的 URDF GUI 编辑器。它不会连接机械臂、夹爪或任何控制运行时，也不会原位覆盖输入 URDF。

它是针对本仓库现有 EG2-4B2 资产的结构开发的，但识别逻辑不依赖固定前缀：

- 独立 proxy URDF 会识别 `left_finger_joint` / `right_finger_joint`；
- 双臂整机 URDF 会分别识别左右两套 gripper，并在 GUI 中切换；
- translated-end 模板会选择 `*_end_component`，而不是较长的 linkage；
- mesh-backed proxy 会默认选择真正可调的 pad collision box，而不是缩放整个 moving-jaw mesh。

## 启动

在仓库根目录运行：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python -m utils.gripper_urdf_editor assets/inspire_gripper/urdf/eg2_4b2_parallel_proxy.urdf
```

也可打开整机 URDF；视图默认只显示当前选中的夹爪，避免机械臂尺度掩盖末端细节：

```bash
python -m utils.gripper_urdf_editor assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf
```

无图形界面的结构检查：

```bash
python -m utils.gripper_urdf_editor assets/inspire_gripper/templates/parallel_gripper_translated_end_component_template.urdf --inspect
```

GUI 只使用 Python 标准库 Tk；STL/OBJ 预览器也由本工具内置，不需要 Matplotlib、Open3D、ROS 或网络资源。

## 推荐工作流

1. 打开独立夹爪 URDF 或包含夹爪的整机 URDF。
2. 在“自动识别平行夹爪”中确认父 link 和两个 prismatic joint。双臂模型会有两项。
3. 检查 A/B 自动选择的末端几何。必要时可从下拉框人工改为同一 jaw 固定子树中的其它 visual/collision。
4. 用开合滑条或“闭合/半开/张开”快捷按钮检查几何、contact frame 和实时内侧开口。负开口表示两个内侧表面已穿透。
5. 直接拖动末端 `X/Y/Z` 尺寸、位置和 `RPY` 滑条，3D 结果会实时变化。默认线性与旋转步长都是 `0.01`；也可切换 `0.05 / 0.10 / 0.50 / 1.00`，或在右侧键入任意精确数值后按回车。
6. 对称夹爪通常保持尺寸、位置同值联动；如果两侧局部坐标定义不同，应关闭位置联动并分别编辑。
7. 用闭合/张开 `q` 修改 master joint 的范围。mimic jaw 的 limit 会按 multiplier/offset 同步更新。
8. 旋转、平移、缩放 3D 视图，并切换 visual、collision、contact frame；正视/侧视/俯视适合检查对称性和间隙。
9. 使用“导出新 URDF”。输出先写同目录临时文件，重新解析与校验后再原子替换目标导出文件。

连续拖动产生的高频更新会自动合并为一次撤销记录，而不是让每一个像素位移占用一步撤销。滑条采用以当前值为中心的局部窗口：选 `0.01` 时用于精调，选 `0.50 / 1.00` 时窗口会同步扩大用于快速移动；在数值框中输入窗口外的值后，滑条会自动重定位到新值附近。

位置和旋转各有两种复位：

- `恢复输入位置/旋转`：恢复当前输入 URDF 中该末端自己的初始值。这适合模板中原始 `Z=71.164 mm` 等非零安装位姿；
- `位置/旋转归零`：明确写入三个零值；
- 开启 A/B 位姿联动时，两侧分别恢复各自的输入值，不会把 A 的初始值复制给 B；所有复位操作均可撤销。

## 编辑语义

### 末端识别

候选 pair 必须由两个平行的 prismatic joints 构成，并优先考虑：

- 共同 parent link；
- `mimic` 耦合；
- `gripper`、`finger`、`jaw` 命名提示；
- 两个 joint origin 有可观测间距。

每个 jaw 只沿 prismatic child 后的 fixed-joint 子树寻找末端，避免越过另一个主动关节。`end_component`、`fingertip`、`terminal`、`pad` 的优先级高于 linkage/carriage。

### 尺寸与位置

- box：直接更新 `size`；
- cylinder/sphere：更新半径与长度，并检查轴对称约束；
- STL/OBJ mesh：根据读取到的原始包围盒换算并更新非均匀 `scale`，同时补偿 geometry origin，使缩放围绕原包围盒中心进行；
- “位置来源”会明确显示当前修改的是末端 fixed mount origin，还是末端 geometry origin；
- 若存在 direct fixed `inner_contact` frame，默认在尺寸/位置变化后把它重新贴到朝向另一 jaw 的内侧主表面；
- 同 link 内相同几何类型的 visual/collision 会自动联动；跨类型时，仅当原始包围盒三轴尺寸在 5% 内一致才联动。因此现有 proxy 的 mesh jaw 与 box pad 会共同变化，而不匹配的完整机构 mesh 不会被误缩放。

### 开合位置

GUI 中的闭合和张开按实际内侧开口大小推断，而不是假定数值较大的一端一定张开。因此当前资产中的 `q=-35 mm` 会显示为闭合，`q=0 mm` 显示为张开。滑条始终以 `0% = 闭合`、`100% = 张开` 表示。

## 导出与校验

导出会拒绝输入文件本身作为目标。相对 mesh 路径会从输入 URDF 目录解析，再相对于新 URDF 的目录重写，所以将导出文件保存到其它目录时不会静默破坏 mesh 引用。`package://`、`model://` 和绝对路径保持原样。

导出前后检查：

- XML 可解析且根节点为 `robot`；
- link/joint 名称存在且唯一；
- joint parent/child 存在，运动树无多父级或环；
- prismatic/revolute joint 轴和 limits 有效；
- mimic target 存在。

## 边界与注意事项

- 当前工具专用于两个平移 jaw 的平行夹爪，不把多指灵巧手或转动式夹爪误装成这一语义。
- 3D mesh 预览支持 STL 和 OBJ；其它 mesh 格式仍会保留在 URDF 中，但 GUI 会提示并跳过其表面。
- `package://` / `model://` mesh 因缺少包解析上下文而不做本地预览。
- 修改几何尺寸不会自动重估 mass/inertia。若导出的 URDF 用于动力学仿真，应在确定材料、质量和转动惯量后单独更新 inertial；本工具不会用未经授权的密度假设改写它们。
- 自动贴合 contact frame 选择朝向另一 jaw 的主表面中心。非盒状或斜面接触件应在导出前结合 CAD/碰撞模型复核。
- 这里的 3D 检查和 XML 校验是离线资产证据，不代表真实夹爪行程、碰撞安全或硬件标定已经验证。
