## 核心结论

对于你已有的真机位控机械臂和固定桌面场景，一个好的 Real2Sim 不是“把 URDF、桌子和物体放进 MuJoCo，再把颜色调得像真机”，也不是寻找一组所谓绝对真实的质量、摩擦和刚度参数。

更合理的定义是：

> **建立一个与真实部署链路在控制接口、时序、运动响应、接触事件、观测误差和任务结果上等价，并且能表达不确定性的仿真系统。**

它最终应包含四部分：

[
\mathcal S =
\left(
M_{\text{nominal}},
p(\theta),
R_{\text{residual}},
V_{\text{validation}}
\right)
]

其中：

* (M_{\text{nominal}})：最可能的名义数字孪生；
* (p(\theta))：由真机数据估计出的参数分布，而不是手工拍脑袋的随机范围；
* (R_{\text{residual}})：MuJoCo 物理模型无法表达的残差，例如内部位控滤波、通信抖动、背隙；
* (V_{\text{validation}})：持续验证 sim 和 real 是否仍然一致的测试体系。

对你的系统，优先级应是：

```text
控制和时间语义
    >
运动学和坐标系
    >
位控闭环响应
    >
夹爪和接触几何
    >
物体摩擦、质量和接触
    >
相机和视觉外观
    >
更复杂的材料真实性
```

你当前 MuJoCo 路径仍然采用逐 tick 直接写入 `qpos/qvel`，并明确把执行模式标记为 `kinematic_direct_qpos_debug`、`robot_dynamics=not_identified`。这适合验证控制协议和任务逻辑，但不能作为真正的 Real2Sim 训练模型。 

---

# 一、首先确定 Real2Sim 的边界

## 1. Real2Sim 应匹配哪一层接口

你的真实部署链路大致是：

```text
15 Hz observation
    ↓
policy / human / replay action
    ↓
action continuity / safety
    ↓
约 45 Hz pose interpolation + IK + safety projection
    ↓
150 Hz deploy_reference_qpos
    ↓
SDK / network / robot controller
    ↓
机器人内部位置环
    ↓
measured qpos / qvel / TCP / gripper state
```

因此，最合理的 sim-real 对齐边界不是：

```text
policy action → 直接写 MuJoCo qpos
```

而是：

```text
真实与仿真共享：
policy action
→ interpolation
→ IK
→ safety
→ deploy_reference_qpos

之后分叉：
real backend → SDK → robot
sim backend  → position-servo model → MuJoCo
```

这样 sim 和 real 使用完全相同的：

* action representation；
* action continuity；
* 插值；
* IK；
* 安全投影；
* 控制频率；
* 指令饱和逻辑；
* 终止语义；
* observation assembly。

只替换“关节位置指令送到哪里”。

这是整个 Real2Sim 中最重要的架构原则。

## 2. 不要追求一个无条件“全真实”模型

Real2Sim 的精度应该由任务定义。

例如：

| 任务          | 必须准确的部分             | 可以较粗略的部分    |
| ----------- | ------------------- | ----------- |
| 空中 reaching | 运动学、延迟、位置跟踪         | 摩擦、物体质量     |
| 抓取抬升        | TCP、夹爪几何、摩擦、位控响应    | 桌面外围视觉      |
| 桌面推动        | 指尖几何、桌面摩擦、物体惯量、接触软硬 | 机械臂内部惯量的一部分 |
| 精密插入        | 坐标标定、碰撞几何、控制延迟、接触刚度 | 大范围视觉随机化    |
| 视觉 VLA      | 相机外参、曝光、遮挡、颜色、时序    | 精确电机参数      |
| 高速动态操作      | 延迟、饱和、速度/加速度限制、物体惯量 | 低速静摩擦的细节    |

因此应在项目开始时建立一份 `Real2SimFidelityContract`，明确：

```text
目标任务
action boundary
observation boundary
控制频率
需要匹配的状态和事件
允许的误差
不纳入建模的现象
```

---

# 二、Real2Sim 第一层：数据和时间基础设施

很多所谓“动力学误差”，实际是时间戳错位、通信延迟或者指令语义不一致。

## 1. 真机必须记录的时间戳

每个控制周期至少记录：

```text
policy_observation_timestamp
policy_inference_start/end
raw_action_timestamp
planning/interpolation_timestamp
IK_output_timestamp
safety_output_timestamp
command_enqueue_timestamp
SDK_dispatch_timestamp
SDK_receipt_timestamp
robot_state_source_timestamp
robot_state_receive_timestamp
camera_capture_timestamp
camera_receive_timestamp
```

还应保存：

```text
raw policy/human/replay proposal
selected action
safety-modified action
deploy_reference_qpos
actual transmitted qpos
measured qpos
measured qvel
measured TCP
gripper command/state
task object pose
contact/event labels
```

不能只保存：

```text
observation_t
action_t
observation_t+1
```

否则无法区分：

* 策略推理延迟；
* planner 延迟；
* 网络延迟；
* 机器人内部位置环延迟；
* 状态上报延迟；
* 相机延迟；
* observation assembly stale data。

## 2. 必须识别四类时间行为

### 固定延迟

例如：

[
q^{cmd}*{t}
\rightarrow
q^{meas}*{t+d}
]

需要估计每个关节的有效延迟 (d)。

### 延迟抖动

真实系统可能不是固定 20 ms，而是：

```text
大部分 15–20 ms
少量 25–35 ms
偶发更高延迟
```

这应该成为一个分布，而不是单个常数。

### 指令保持与重采样

你的系统具有 15 Hz、约 45 Hz 和 150 Hz 三层时钟。仿真必须复现：

* policy action 持续多久；
* planning target 如何插值；
* 150 Hz command 是否零阶保持；
* 不整除频率时 scheduler 如何分配 tick；
* 是否会跳过或重复 command；
* reset 后第一个 command 的语义。

### 异步观测

RGB、深度、机械臂状态和手部状态往往不是同一个采样时刻。

Real2Sim 不应总是给策略一个完美同步的 observation，而应该复现：

```text
RGB age
depth age
joint-state age
object-pose age
各通道 timestamp difference
```

否则策略会在 sim 中利用真实系统不存在的“时间一致性”。

---

# 三、运动学和坐标系标定

在位控机械臂桌面任务中，运动学误差通常比刚体惯量误差更先暴露。

需要标定的内容包括：

```text
每个关节零位 offset
关节方向和比例
link geometry correction
robot base 相对 table/world 的位姿
flange 到 tool TCP 的位姿
夹爪中心和指尖接触点
固定相机外参
腕部相机外参
桌面平面位置和法向
围挡、托盘、目标区位姿
```

## 1. 建立唯一坐标系真值

建议明确以下 frame：

```text
world
table
left_arm_base
right_arm_base
left_flange
right_flange
left_tool_tcp
right_tool_tcp
fixed_camera
left_wrist_camera
right_wrist_camera
object frames
target frames
```

尤其需要避免：

* 仿真使用 robot base；
* 真机 action 使用 table frame；
* 相机检测输出 camera frame；
* 数据集又存了另一个 world frame。

应形成一张静态和动态 TF contract。

## 2. 运动学标定数据

建议在工作空间内采集约 50–100 个静态姿态：

* 覆盖中心、边缘、低位、高位；
* 覆盖不同腕部姿态；
* 不只采集舒适区；
* 至少保留 20% 完全不参与拟合。

外部测量可以使用：

* 标定板；
* AprilTag/Charuco；
* 高精度外部相机；
* 激光跟踪仪；
* 机械测量治具。

优化变量可以先限制为：

```text
joint zero offsets
base transform
TCP transform
camera extrinsics
table transform
```

只有残差仍有明显随关节配置变化的系统性模式时，才进一步修改 link length、joint axis 等结构参数。

## 3. 运动学验收不应只看平均误差

至少报告：

```text
TCP position median / p95 / max
TCP rotation median / p95 / max
按 workspace 区域分组误差
按关节配置分组误差
重复到达误差
绝对误差与重复性误差分离
```

一个常见问题是：

```text
重复性很好，但绝对位置有固定偏差
```

这通常是 TCP、base 或 camera 外参问题，而不是机器人动力学问题。

---

# 四、位控机械臂最关键的部分：闭环执行器模型

## 1. 不要把位控机械臂当成理想位置源

当前 direct-qpos 模式本质是：

```text
q_sim = q_command
```

真实机器人实际是：

```text
q_command
→ SDK 缓存
→ 通信
→ 内部轨迹生成
→ 内部位置/速度/电流环
→ 电机和减速器
→ q_measured
```

因此 Real2Sim 要建模的是：

[
q^{cmd}
\rightarrow
q^{meas}
]

而不是必须完整恢复：

[
\tau_\text{motor}
\rightarrow
q
]

对一个无法访问底层电流环、力矩环的商用位控机械臂，所谓“真实电机参数”通常不可辨识。更现实的是建立一个**有效闭环模型**。

## 2. 推荐的第一版模型

建议包含四个串联部分。

### A. command delay buffer

[
\bar q^{cmd}*k=q^{cmd}*{k-d}
]

其中 (d) 可以是：

* 固定整数 command steps；
* 固定延迟加随机 jitter；
* 每个关节共享；
* 或少量关节存在不同相位。

### B. 内部 reference filter

[
r_{k+1}
=======

\alpha r_k+(1-\alpha)\bar q^{cmd}_k
]

用于表达：

* SDK 平滑；
* 控制器插值；
* 低通滤波；
* command rate 与内部 servo rate 差异。

### C. 有限能力的位置伺服

[
\tau_k=
\operatorname{clip}
\left[
K_p(r_k-q_k)-K_d\dot q_k+\tau_{ff},
-\tau_{\max},
\tau_{\max}
\right]
]

MuJoCo 的 actuator 模型支持 transmission、activation dynamics、gain 和 bias；position actuator 可以配置位置增益、速度阻尼以及一阶 filter time constant，也可以配置力限制。这比直接覆写 `qpos` 更接近真实位置伺服。([MuJoCo Documentation][1])

### D. command constraints

真实系统通常还存在：

```text
max joint velocity
max joint acceleration
可能的 jerk limit
joint limit slowdown
per-cycle delta-q limit
command deadband
command quantization
safety projector clipping
```

这些必须和真机使用相同语义。

## 3. 推荐的 MuJoCo 实现层级

建议保留三个 profile：

```text
kinematic_direct_qpos_debug
    直接写 qpos
    用于协议、replay、termination 调试
    不用于最终 sim2real 训练

identified_position_servo_v1
    delay + command hold + filtered position actuator
    用于第一版 Real2Sim

identified_position_servo_residual_v2
    物理 servo + 小型受限残差模型
    仅在 v1 无法覆盖系统性误差时加入
```

不要直接从 direct-qpos 跳到复杂神经动力学模型。

## 4. 位控闭环的辨识实验

### 单关节阶跃

每个关节分别测试：

```text
小、中、大三种幅值
正、负两个方向
至少三个起始关节配置
空载和典型负载
每种重复 5 次以上
```

用于识别：

```text
dead time
rise time
overshoot
settling time
steady-state error
direction asymmetry
速度饱和
```

### 梯形轨迹

用于识别：

```text
最大速度
最大加速度
制动行为
target 更新时的平滑方式
```

### chirp / multisine

在安全频率范围内逐渐改变频率，用于估计：

```text
闭环带宽
相位滞后
不同频率下的幅值衰减
```

### PRBS 或随机小扰动

用于拟合简单的 ARX/state-space 模型：

[
q_{t+1}
=======

\sum_i a_iq_{t-i}
+
\sum_j b_jq^{cmd}_{t-j}
+
\epsilon_t
]

### 多关节耦合轨迹

单关节拟合完成后再运行：

* 多关节同步；
* 不同方向同时运动；
* 快速腕部动作；
* 接近工作空间边界；
* 不同 payload。

用于检测：

```text
控制器内部耦合
电源或速度共享限制
结构柔性
配置相关误差
```

## 5. 位控模型的验收指标

不要只看最终位置。至少比较：

```text
q trajectory RMSE
qdot trajectory RMSE
command-measurement phase lag
peak velocity
peak acceleration
overshoot
settling time
TCP trajectory error
不同负载和配置下的误差
```

建议以任务容差定义门槛，而不是绝对使用某个毫米值。例如：

```text
TCP median 误差 < 最小任务间隙的 25%
TCP p95 误差 < 最小任务间隙的 50%
phase error < 一个 150 Hz command frame
```

对于精密插入，标准需要更严；对于粗粒度 pick-and-place，可以放宽。

---

# 五、机器人刚体参数该辨识到什么程度

## 1. 位控模式下不要过度追求惯量真值

如果只能得到：

```text
command qpos
measured qpos/qvel
```

而没有可靠的：

```text
joint torque
motor current
external wrench
```

那么 link mass、COM、inertia、Coulomb friction、servo gain 往往相互混淆。

例如同样的 tracking lag，可能来自：

* link inertia 更大；
* servo gain 更低；
* command filter 更强；
* force limit 更低；
* friction 更大。

它们不一定可唯一辨识。

因此第一阶段应优先拟合：

```text
有效的 qcmd → qmeasured 行为
```

而不是声称恢复了真实电机和连杆动力学。

## 2. 推荐处理

* 几何和 link mass 使用厂商 URDF/CAD 作为先验；
* payload 和 tool mass 尽量实测；
* TCP 和 COM 单独校准；
* 如果有电流或关节力矩估计，再做标准刚体动力学辨识；
* 没有力矩数据时，只对明显影响任务的参数做小范围调节；
* 对无法唯一确定的参数保留分布，而不是强行取一点估计。

---

# 六、夹爪和灵巧手 Real2Sim

对于 Inspire 手、DMTAC 夹爪之类末端，不应只建一个“开合宽度”。

需要识别：

```text
command value → 实际关节/开口映射
开合延迟
最大速度
加速度
闭合停止逻辑
物体阻挡时的行为
位置误差
force/current limit
手指耦合
柔性和回弹
指腹接触几何
摩擦
死区
迟滞
左右方向不对称
```

## 1. 空载实验

运行多组：

```text
完全打开 → 完全闭合
不同中间目标
慢速和快速
反复往返
```

记录：

```text
command
measured joint/aperture
到达时间
重复性
迟滞曲线
```

## 2. 阻挡实验

使用若干已知厚度的刚性块：

```text
5 mm
10 mm
20 mm
...
```

观察：

* 夹爪是否继续增加内部力；
* measured position 是否停在物体表面；
* SDK 是否报告到达；
* command 是否被重发；
* 是否发生回弹；
* 不同厚度下夹持稳定性。

## 3. 摩擦和滑移

对典型物体表面做：

```text
不同夹紧 command
逐渐增加切向负载
记录首次滑移
```

需要辨识的不是一个全局 `gripper_friction`，而是：

```text
finger material × object material
```

的接触分布。

---

# 七、桌面、物体和接触模型

## 1. 几何要拆成三套

同一个物体最好分别具有：

```text
visual geometry
collision geometry
inertial geometry
```

不要直接把高精度扫描 mesh 同时用于三者。

### visual geometry

用于：

* RGB；
* 深度；
* 遮挡；
* 材质。

### collision geometry

用于：

* 接触位置；
* 指尖抓取；
* 推动；
* 插入；
* 碰撞效率。

通常应使用：

* convex decomposition；
* primitive；
* 低面数 convex mesh；
* 手工修正关键接触边缘。

### inertial geometry

用于估计：

```text
mass
COM
inertia tensor
```

关键问题不是表面看起来是否完全一致，而是：

* 接触边缘在哪里；
* 指尖厚度是否正确；
* 物体底面是否平；
* 圆角大小是否正确；
* 碰撞 proxy 是否导致错误接触法向。

2025 年的 Scalable Real2Sim 工作展示了利用外部相机和机器人交互自动获得视觉几何、碰撞几何和惯性属性的完整资产生成思路，说明现代 Real2Sim 越来越倾向于把“外观 mesh、碰撞 mesh、物理属性”作为不同输出处理。([Scalable Real2Sim][2])

## 2. 物体质量、COM 和惯量

优先顺序：

```text
质量：称重
尺寸：测量或扫描
COM：悬挂/支撑实验或 CAD
惯量：由质量分布近似
```

对于规则刚体，通常不需要一开始复杂辨识。

例如长方体可由尺寸和质量估计惯量。只有高速旋转、翻滚或者动态抛掷任务，惯量精度才会变得非常关键。

## 3. 摩擦不能只调一个系数

至少需要区分：

```text
object-table
object-finger
finger-table
object-object
object-wall/barrier
```

对于每一对材料，需要考虑：

* 静摩擦；
* 动摩擦；
* 扭转摩擦；
* 滚动摩擦；
* 各向异性；
* 接触面软硬；
* 表面磨损和污染带来的变化。

## 4. 推荐接触实验

### 倾斜桌面实验

逐渐提高倾角，记录首次滑动角：

[
\mu_s \approx \tan\theta_{\text{slip}}
]

用于静摩擦初值。

### 自由滑动实验

给物体一个初速度，记录减速度和停止距离，用于估计动态摩擦。

### 机器人推动实验

使用固定高度、固定方向、不同速度推动物体，记录：

```text
机器人 TCP
物体位置和旋转
接触开始时间
接触结束时间
最终位姿
```

### 偏心推动实验

改变接触点，用于验证：

* COM；
* 扭转摩擦；
* 接触几何；
* 物体惯量。

### 抓取滑移实验

使用不同夹持力度和物体质量，记录何时开始下滑。

### 放置和轻微跌落实验

用于识别：

* 接触回弹；
* damping；
* settle 时间；
* 接触软硬。

## 5. MuJoCo 中不能只拟合 friction

MuJoCo 使用软约束接触模型。接触滑移和穿透不仅由 `friction` 决定，还受到 `solref`、`solimp`、`impratio`、solver、friction cone 和 timestep 影响。官方文档特别指出，出现缓慢滑移时，单纯增加摩擦通常不一定能解决，约束阻抗和 solver 设置同样重要。([MuJoCo Documentation][3])

因此接触参数应联合拟合：

```text
friction
condim
solref
solimp
impratio
margin
timestep
solver/integrator
```

但不要一次把所有参数都交给优化器。推荐顺序：

```text
collision geometry
→ friction
→ contact softness/damping
→ torsional/rolling friction
→ solver refinements
```

否则优化器可能用错误的软接触参数补偿错误几何。

---

# 八、相机与视觉 Real2Sim

如果策略使用 RGB/RGB-D，则视觉 Real2Sim 应分为两部分：

```text
名义视觉数字孪生
+
由真机变化统计得到的 visual randomization
```

## 1. 相机几何

需要标定：

```text
intrinsics
distortion
camera resolution
FOV
camera-to-base extrinsics
camera-to-table extrinsics
wrist-camera-to-flange extrinsics
depth scale
depth invalid regions
RGB-depth alignment
```

仿真中不能只依据相机型号填写一个大致 FOV。

## 2. 相机时序

需要识别：

```text
capture latency
encoding/transmission latency
frame rate jitter
dropped frames
rolling shutter
exposure time
auto-exposure response
RGB 和 depth 的相对延迟
```

对策略而言，一张“视觉上很真实但晚了 100 ms”的图像，与一张同步图像的控制含义完全不同。

## 3. 场景外观

名义模型应尽量匹配：

* 桌面主色；
* 桌面纹理尺度；
* 围挡；
* 机械臂主体颜色；
* 夹爪颜色；
* 背景墙；
* 关键反光；
* 光源主要方向；
* 相机曝光和白平衡。

但不必追求电影级渲染。训练阶段更重要的是让真实图像处于 sim 分布之内。

visual randomization 不应使用毫无依据的全范围随机 RGB，而应来自：

```text
不同日期真实图像
不同曝光
不同灯光开关
摄像头轻微移动
桌面污染
机械臂遮挡
物体反光
```

的实测变化。

---

# 九、观测噪声和状态估计链路

Real2Sim 不能直接把 MuJoCo ground truth 全部送给策略，除非真机也能获得相同信息。

需要模拟：

```text
joint encoder noise
joint velocity estimation filtering
TCP FK error
object detector noise
pose estimator bias
遮挡导致的丢失
depth noise
tracking jump
force/tactile bias
frame dropout
不同观测通道的数据年龄
```

尤其需要区分：

### 白噪声

每 step 独立变化：

[
\epsilon_t\sim \mathcal N(0,\sigma^2)
]

### episode bias

一次运行中基本固定：

[
b_{\text{episode}}
]

例如相机外参、关节零偏。

### 慢漂移

随温度、机械安装或者时间逐渐变化。

### 丢帧和离群点

不能用普通高斯噪声覆盖。

这些误差应在与现实相同的时间尺度上随机化：

| 参数             | 推荐变化周期                 |
| -------------- | ---------------------- |
| link mass、物体质量 | model/deployment 级     |
| 相机安装误差         | deployment 或 episode 级 |
| 摩擦             | episode 级              |
| 控制延迟           | episode + step jitter  |
| encoder bias   | episode 级              |
| encoder noise  | step 级                 |
| frame dropout  | step/event 级           |
| 灯光和材质          | episode 级              |

---

# 十、真机辨识数据集如何设计

建议建立独立的 `real2sim_calibration` 数据集，而不是直接使用普通遥操数据。

## 数据集 A：静态几何和坐标

内容：

```text
50–100 个机械臂姿态
不同 workspace 区域
TCP 外部测量
相机观测
桌面/围挡/目标点
```

用于：

```text
joint zero
base transform
TCP
camera extrinsics
table frame
```

## 数据集 B：无接触位控响应

内容：

```text
单关节 step
梯形轨迹
chirp/multisine
PRBS
多关节耦合轨迹
多个 payload
多个起始配置
重复执行
```

用于：

```text
delay
filter
kp/kv effective parameters
velocity/acceleration limit
deadband
configuration dependence
random variability
```

## 数据集 C：夹爪

内容：

```text
空载开合
不同目标位置
阻挡物
不同速度
不同夹持力度
滑移实验
```

## 数据集 D：单一接触原语

内容：

```text
推
滑
轻触
压
放置
抓住后缓慢移动
```

避免一开始使用完整任务，因为完整任务中太多参数同时发生作用。

## 数据集 E：物体辨识

内容：

```text
不同位置推动
不同接触高度
偏心推动
倾斜滑动
抓取滑移
轻微放置/跌落
```

## 数据集 F：完整任务

内容：

```text
reach
grasp
lift
place
push-to-target
可能的插入/装配
```

用于最终验证，而不是主要用来拟合底层参数。

## 数据划分

必须按轨迹划分：

```text
train identification traces
validation traces
held-out test traces
```

不能把同一条长轨迹切成前半段训练、后半段测试，这会造成严重的数据泄漏。

最好额外留出：

* 未见过的速度；
* 未见过的起始姿态；
* 未见过的 payload；
* 未参与辨识的策略；
* 不同日期采集的数据。

---

# 十一、参数辨识的正确顺序

不要一次优化所有参数。

推荐严格采用：

```text
1. 时间戳和延迟
2. 坐标系和运动学
3. 位控闭环
4. 夹爪闭环
5. 物体几何和质量
6. 碰撞几何
7. 摩擦
8. 接触软硬和 damping
9. 观测噪声
10. 视觉参数
11. 剩余 residual
```

原因是上层参数会补偿下层错误。

例如：

* 时间错位会被误认为 servo 响应慢；
* TCP 偏差会被误认为关节零偏；
* 碰撞几何过大可能被优化器用较低摩擦补偿；
* 指尖过厚可能被错误接触刚度补偿；
* object pose estimator 延迟可能被误认为物体惯量大。

---

# 十二、优化目标应该怎样定义

定义参数向量：

[
\theta =
[
\theta_{\text{time}},
\theta_{\text{kin}},
\theta_{\text{servo}},
\theta_{\text{gripper}},
\theta_{\text{object}},
\theta_{\text{contact}},
\theta_{\text{sensor}}
]
]

目标函数不应只有 state MSE：

[
\begin{aligned}
J(\theta)=&
w_qL_q+
w_{\dot q}L_{\dot q}+
w_xL_{\mathrm{TCP}}+
w_oL_{\mathrm{object}}\
&+
w_cL_{\mathrm{contact}}+
w_eL_{\mathrm{event}}+
w_rL_{\mathrm{task}}
\end{aligned}
]

其中：

### 轨迹损失

```text
joint position
joint velocity
TCP position
TCP orientation
object pose
```

### 事件损失

```text
首次接触时间
开始滑移时间
物体离桌时间
抓取成功时间
落地时间
termination reason
```

### 分布损失

比较多次重复实验的：

```text
均值
方差
分位数
失败概率
```

### 任务损失

```text
success rate
完成时间
最终误差
失败类型
安全触发率
```

接触数据建议使用 robust loss，例如 Huber loss，避免少量 tracking jump 或视觉离群点支配优化。

---

# 十三、使用哪种辨识方法

## 1. 解析或最小二乘

适合：

```text
固定延迟
简单一阶滤波
ARX 模型
关节 zero offset
刚体运动学
```

优点是可解释、稳定。

## 2. 网格搜索 / 随机搜索

适合：

```text
delay steps
solver choice
integrator choice
少量离散参数
```

## 3. CMA-ES / Bayesian Optimization

适合：

```text
10–30 个连续参数
目标函数不可微
包含接触和离散事件
```

对于 MuJoCo 接触参数通常比直接梯度下降更稳。

## 4. likelihood-free Bayesian inference

当多个参数组合都能解释真机数据时，不要只求一个最优点，而应估计后验：

[
p(\theta\mid D_{\text{real}})
]

BayesSim 的核心思想就是从真实轨迹推断模拟参数的后验分布，并使用该后验指导 domain randomization，而不是使用宽泛的均匀随机范围。DROPO 则进一步强调，可以利用有限的离线真实轨迹估计随机化分布，并显式表达参数不确定性。([Robotics Proceedings][4])

这非常适合你已有的遥操、replay 和 human-in-the-loop 数据。

## 5. 残差模型

在物理模型已经基本正确后，可以学习：

[
\Delta x_{t+1}
==============

f_\phi
(
x_t,
u_t,
x^{sim}_{t+1}
)
]

或者学习：

```text
command correction
actuator force correction
delay correction
observation correction
```

不建议第一版直接让神经网络任意修改 MuJoCo state，因为：

* 容易破坏接触稳定性；
* 可能产生不可物理解释的能量；
* 长程 rollout 易发散；
* 难以迁移到未见状态。

更安全的 residual 位置是：

```text
position reference
actuator force
低维物体 acceleration
observation model
```

---

# 十四、Real2Sim 的目标不是“最像”，而是“对任务最有用”

严格的轨迹匹配并不必然产生最好的 sim-to-real 策略。

原因包括：

* 模拟器模型族本身不包含真实系统；
* 多组参数可能产生近似轨迹；
* 策略只关心某些状态区域；
* 某些底层误差对任务完全不重要；
* 过度拟合辨识轨迹可能伤害策略泛化。

AdaptSim 的结论正是：对操作任务而言，任务驱动地调整模拟参数分布，可能比追求单纯的动力学轨迹匹配更有效。([Proceedings of Machine Learning Research][5])

因此建议分两阶段：

```text
阶段 1：physics-driven Real2Sim
    尽量复现真实轨迹和事件

阶段 2：task-driven Sim2Real adaptation
    根据真实任务成功、失败和策略排序调整参数分布
```

例如，某组摩擦参数虽然轨迹 MSE 略差，但能正确预测：

```text
哪些抓取会滑
哪些 pushing policy 会失败
哪些接触策略更鲁棒
```

它可能比 nominal MSE 最小的参数更适合 RL。

---

# 十五、参数随机化不应手工宽泛设置

Real2Sim 最终不应只输出：

```text
friction = 0.5
mass = 0.2 kg
delay = 20 ms
```

而应输出：

```text
friction ~ distribution
mass ~ distribution
delay ~ distribution
camera offset ~ distribution
servo bandwidth ~ distribution
```

并保留相关性。

例如：

```text
更大 payload
→ tracking lag 增大
→ overshoot 变化
→ gripper lifting slip 概率增加
```

不能把这些参数完全独立采样。

推荐形成：

```yaml
servo_profile:
  delay:
    mean_ms: ...
    std_ms: ...
  joint_1:
    time_constant: ...
    kp_effective: ...
    velocity_limit: ...
    covariance: ...

object_profile:
  mass:
    mean: ...
    std: ...
  friction_table:
    posterior: ...
```

现实 nominal 模型应位于随机化分布的高概率区域，而不是仅仅位于极宽范围内部。

---

# 十六、验证体系：怎样证明 Real2Sim 是好的

一个好的 Real2Sim 必须通过未参与拟合的数据验证。

## Level 1：组件验证

分别验证：

```text
运动学
servo
gripper
object sliding
contact
camera
observation latency
```

## Level 2：轨迹验证

在相同 command trace 下比较：

```text
q
qvel
TCP
object pose
contact events
```

## Level 3：任务验证

比较：

```text
success rate
完成时间
最终误差
安全触发
失败类型
```

## Level 4：策略排序验证

这是 RL 环境最关键但经常被忽略的一项。

选取多组性能不同的 policy：

```text
policy A
policy B
policy C
...
```

分别在 sim 和 real 上测试，观察：

```text
sim 中更好的 policy
是否通常在 real 中也更好
```

即使 sim success rate 与 real 有固定偏差，只要 policy ranking 高度一致，仿真仍然非常适合算法开发。

反过来，如果：

```text
sim A > B > C
real C > A > B
```

说明 simulator 对 RL 的优化方向是错误的，即使单条轨迹看起来很逼真。

## 推荐的初始验收标准

这些不是通用标准，但可以作为第一版工程目标：

| 指标                        | 建议起始目标                         |
| ------------------------- | ------------------------------ |
| command/state 时间对齐误差      | 不超过一个 120 Hz command frame     |
| TCP median 误差             | 小于最小任务间隙的 25%                  |
| TCP p95 误差                | 小于最小任务间隙的 50%                  |
| 轨迹相位差                     | 不超过一个 command frame            |
| 关键 contact event 误差       | 不超过一个 command frame或事件持续时间的10% |
| held-out task success gap | 5–10 个百分点以内                    |
| policy ranking            | Spearman 相关系数目标大于 0.8          |
| 随机化覆盖率                    | 约90–95%的真实轨迹落入sim预测区间          |
| 失败模式                      | real主要失败模式在sim中均可复现            |

---

# 十七、建议的软件结构

建议不要把 Real2Sim 参数散落在 MJCF 和 Python 代码里，而应建立版本化 artifact。

```text
sim/rm65/
  actuation/
    profile.py           # 后端无关、版本化执行器参数
    core.py              # 120 Hz qcmd -> causal actuator target

  real2sim/
    artifact.py          # evidence / claim / validation contract
    artifacts/           # source index 与拟合/held-out manifest

  sapien_renderer.py     # 仅 operator/debug preview

sim/mujoco/actuation/
  executor.py            # RM65 通用执行器的 MuJoCo 薄适配层
```

后续 calibration、identification 和 validation 工具如果只服务 RM65，应继续
放在 `sim/rm65/real2sim/` 下；只有确实跨 embodiment 的纯契约才提升到更高层。
`sim/real2sim/` 仅保留兼容导出，不再作为新实现位置。

每个 Real2Sim artifact 应包含：

```text
source real dataset hash
robot serial/profile
tool profile
scene profile
MuJoCo version
model artifact hash
identification code version
fit metrics
held-out metrics
parameter covariance/posterior
known limitations
```

建议形成四个明确 profile：

```text
canonical_debug
    可读、可审计
    direct qpos
    不是物理真值

real2sim_nominal_v1
    名义位控闭环和场景模型

real2sim_randomized_v1
    nominal + posterior domain randomization

rl_fast_real2sim_v1
    对 nominal 做性能裁剪
    但需要通过行为等价验证
```

不要让 `rl_fast_v0` 独立手调物理参数。它应该从 Real2Sim profile 派生，并且证明裁剪前后的任务行为一致。

---

# 十八、针对你当前系统的推荐落地顺序

## R0：打通真实 trace

首先确保一条真实运行能够完整记录：

```text
15 Hz observation
raw action
selected action
interpolated pose
IK qpos
safety projected qpos
150 Hz dispatched qpos
measured qpos/qvel
TCP
gripper
camera timestamps
object pose
task outcome
```

这是后续所有 Real2Sim 的数据底座。

## R1：时间与坐标系

完成：

```text
clock offset
command delay
camera delay
base-table transform
TCP
camera extrinsics
joint zero offsets
```

先不做接触。

## R2：自由空间位控闭环

将 direct-qpos 替换为：

```text
delay
command hold
reference filter
position actuator
velocity/acceleration/force limits
```

并完成单关节和多关节 held-out trajectory 验证。

## R3：末端执行器

分别为 Inspire/DMTAC 建立独立 profile：

```text
command mapping
delay
speed
deadband
stall
contact geometry
friction
```

## R4：单物体桌面接触

只选择一个规则方块，完成：

```text
mass
geometry
object-table friction
finger-object friction
push
slide
lift
place
```

不要一开始支持大量随机物体。

## R5：完整 CubeLift 闭环

用同一批 action trace 在 real/sim 重放，验证：

```text
reach trajectory
first contact
gripper closure
lift timing
cube pose
success/termination
```

## R6：参数后验与随机化

从重复真实实验估计：

```text
mean
variance
correlations
outliers
slow drift
```

生成 `real2sim_randomized_v1`。

## R7：RL policy transfer 验证

至少比较：

```text
sim-only nominal policy
domain-randomized policy
real-finetuned policy
多个 checkpoint
```

并检查 policy ranking，而不仅是最终单个 policy 成功率。

---

# 十九、最容易踩的错误

1. **先调灯光、颜色和纹理，却没有解决控制延迟。**

2. **继续使用 direct-qpos 训练，然后称为 Real2Sim。**

3. **一次优化几十个质量、摩擦、刚度和 servo 参数。**

4. **用拟合轨迹本身作为验证轨迹。**

5. **只建立一个 nominal 模型，不保存不确定性。**

6. **使用很宽的 uniform randomization 掩盖模型错误。**

7. **sim 给策略 ground-truth object pose，real 却使用有延迟的视觉估计。**

8. **视觉 mesh 直接作为 collision mesh。**

9. **忽略指尖厚度、圆角和 TCP 误差，却反复调 friction。**

10. **把真实内部位置控制器假设成简单高增益 PD，却不做轨迹验证。**

11. **没有记录 command dispatch 和 measurement source timestamp。**

12. **只比较成功率，不比较失败模式和策略排序。**

13. **让参数辨识只覆盖低速舒适区，RL 却探索到更快、更靠近边界的状态。**

14. **用物理参数补偿 observation latency 或 pose-estimation bias。**

15. **把 Real2Sim 模型当成完成一次就不再变化的静态资产。**

---

## 最终建议

结合你当前 RM65 窄本体闭环、15/45/150 Hz 控制链路和 `kinematic_direct_qpos_debug` 模型，第一版 Real2Sim 的核心不应该是新增更多场景随机化，而应该是新增一个明确的：

```text
identified_position_servo_v1
```

至少包含：

```text
per-joint q offset
command delay and jitter
150 Hz zero-order hold
internal reference filtering
position servo kp/kv
velocity and acceleration limits
force/torque saturation proxy
joint friction/damping prior
payload profile
gripper command-response model
observation timestamp and noise model
```

然后用“同一条 timestamped command trace 在真机和仿真分别执行”的 differential harness 做闭环验证。

只有这个层级通过后，再辨识桌面摩擦、物体质量、指尖接触和视觉外观。最终输出的也不应是一份单独 MJCF，而应是：

```text
名义模型
+ 参数后验
+ 随机化 profile
+ held-out validation report
+ task/policy ranking report
```

这才是一个真正适合后续大规模 RL、sim-to-real 和持续迭代的 Real2Sim 系统。

[1]: https://mujoco.readthedocs.io/en/latest/computation/index.html?utm_source=chatgpt.com "Computation - MuJoCo Documentation"
[2]: https://scalable-real2sim.github.io/?utm_source=chatgpt.com "Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups"
[3]: https://mujoco.readthedocs.io/en/stable/overview.html?utm_source=chatgpt.com "Overview - MuJoCo Documentation"
[4]: https://roboticsproceedings.org/rss15/p29.html?utm_source=chatgpt.com "Robotics: Science and Systems XV - Online Proceedings"
[5]: https://proceedings.mlr.press/v229/ren23b.html?utm_source=chatgpt.com "AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer"
