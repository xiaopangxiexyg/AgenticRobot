# DMTAC夹爪执行器辨识流程

本流程用于建立贴近真机的夹爪执行器模型。测试脚本只打开一只DMTAC/Inspire夹爪串口，
不会初始化或控制RM65机械臂，不会启动遥操、replay、正式数采或触觉读取。

脚本位置：

```text
hardware/DMTAC_gripper/examples/identify_gripper_actuator.py
```

## 1. 需要辨识的量

速度固定为归一化`1.0`。每个状态样本记录：

- 实际归一化闭合位置`joint_pos`和毫米开度`opening_mm`；
- 位置寄存器原始uint16、有符号解码值及端点夹紧标记；
- 原始电流`current`；
- `is_moving`、状态码和错误码；
- 温度；
- wall/monotonic纳秒时间戳、读取耗时和命令调用耗时。

默认用单次Modbus连续寄存器读取按50 Hz采样；这比15 Hz action更细，足以先估计快速行程的延迟和
速度曲线。最终以trace中的实际sample gap为准，不能把CLI目标频率当成真实采样频率。

默认`--command-mode runtime-equivalent`会在运动前单独配置并回读speed、force和continuous，
随后命令计时区间只包含position写入。这更贴近正式运行时“配置缓存、稳态只发位置目标”的链路。
`--command-mode inline-config`只用于与旧式单次组合写入做对照，其延迟不能直接当作正式链路的执行器延迟。
trace同时保存计划采样时刻、读写开始/结束时刻、读取中点和单调递增的`trace_sequence`；拟合时优先以
单调时钟的读取中点对齐状态，不要使用JSON落盘时刻或行号代替采样时刻。

力度参数是夹爪寄存器的归一化设定`0.0-1.0`，不是牛顿。没有外部标定力计时，可以辨识
“力度设定—电流—接触位置—力控停止”的关系，但不能声称得到真实N。若需要仿真中的真实接触力，
应在两指之间放置薄型标定力计/测力传感器，并使用`--external-force-reference manual-gauge`
逐次录入读数。

必须区分两类hold：

1. `continuous_grasp=false/true`：夹爪内部catch-mode寄存器；
2. `host_position_hold=one-shot/resend`：主机只下发一次position，或按15 Hz持续重发相同目标。

contact测试使用2×2矩阵将二者独立比较。不能仅测试一个“hold=true”，否则无法判断保持来自
夹爪固件还是上层action重发。
这里的`one-shot`准确含义是“主机不再提供position hold”，并不预设电机一定卸力；固件是否仍然
保持位置正是需要测量的结果。脚本不会把`REG_STOP`当作日常no-hold模式，因为当前驱动只把它
暴露为stop/estop，尚无证据证明它等价于可持续使用的电机卸力模式。

## 2. 安全与恢复边界

- 测试前必须关闭遥操、policy部署、replay和所有其他夹爪调试进程；机械臂保持安全静止。
- 每次只测试一只夹爪；`--side`必须与物理夹爪一致。
- 任何运动都要求CLI确认参数且必须在交互式TTY中启动。启动后只需一次全局Enter；free trial
  自动连续执行，contact trial仅在每次物体放置/复位完成后按一次Enter。
- 默认力度只测试`0.10,0.20,0.35`。超过`0.5`需要额外的high-force双重确认。
- 不要使用手指作为阻挡物。不要使用硬质、易碎、尖锐、液体填充或可能弹出的物体。
- 推荐使用宽度30–50 mm、尺寸明显大于指尖接触面的均匀闭孔泡棉块；固定方向和插入深度。
- 物体必须由非手持支架支撑或软系留，夹爪自动张开时不能掉落到设备上。
- 默认最高允许温度55°C；运动、hold、人工输入等待和每个trial间的默认2秒冷却均保持低频状态采样。
  出现异味、异常声、电流突变或损伤立即Ctrl-C。

脚本修改任何实验寄存器前，会保存原始speed、force和continuous寄存器并fsync recovery journal，
启动独立guardian，再做原值写入/回读验证。正常结束和异常退出都会恢复原始寄存器；父进程被kill
或SSH断开时，guardian会先发送停止，再恢复寄存器。脚本从不调用save/default，因此不会主动
写入持久参数。

异常时脚本只停止运动，不会自动移动到原始位置或强制张开，因为当时可能仍有物体或人员靠近。
整机掉电或串口物理断开期间无法即时恢复；journal必须保留，连接恢复后执行第7节的恢复命令。

## 3. 第零步：只生成实验计划

该命令不访问任何硬件：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py plan \
  --protocol contact \
  --force-levels 0.10,0.20,0.35 \
  --continuous-modes false,true \
  --host-hold-modes one-shot,resend \
  --repeats 1 \
  --target-position 1.0 \
  --hold-duration-sec 5 \
  --object-label foam_block_40mm \
  --object-width-mm 40
```

默认得到12个contact trial：3个力度×2种设备continuous模式×2种主机hold模式。

## 4. 第一步：只读检查

先检查左夹爪：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py inspect \
  --side left \
  --config latest
```

右夹爪将`left`改成`right`。该命令只读，不写寄存器、不运动。必须确认：

- `endpoint.port`是预期的`/dev/serial/by-path/...`，不能只凭易漂移的`/dev/ttyUSB*`判断；
- `state.is_moving=false`；
- `state.error_code=0`；
- 温度合理；
- `settings_raw`和`open_limits_raw`完整可读。

如果串口已被其他进程打开，脚本会列出PID并拒绝继续。若配置中的port过期，可显式使用
`--port /dev/serial/by-path/... --gripper-id 1 --baudrate 115200`，但应先修正正式硬件分配。

## 5. 第二步：无物体自由运动基线

自由运动用于估计命令延迟、位置响应、全行程时间、停止行为和空载电流。确认完整行程内没有物体：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py run \
  --side left \
  --config latest \
  --protocol free \
  --force-levels 0.10,0.20,0.35 \
  --target-position 1.0 \
  --free-duration-sec 3 \
  --sample-hz 50 \
  --repeats 1 \
  --apply \
  --confirmation-token GRIPPER_ONLY_TEST
```

脚本固定速度为`1.0`，不提供速度扫描。每个trial都会：

1. 整个run开始前按一次Enter确认机械臂/遥操已停、手和物体均离开完整行程；
2. 每个trial自动张开；
3. 提示1秒后自动闭合并记录；
4. 自动重新张开并冷却2秒。

free protocol在全局确认后不再逐trial停下来要求输入；任何时刻可按Ctrl-C进入停止/恢复路径。

先只完成一侧的一轮。检查数据无错误后，再用`--repeats 3`获取可用于统计的重复数据。

## 6. 第三步：标准软物体contact/hold矩阵

将泡棉块固定在相同中心位置、相同朝向和相同插入深度。无外部力计时：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py run \
  --side left \
  --config latest \
  --protocol contact \
  --force-levels 0.10,0.20,0.35 \
  --continuous-modes false,true \
  --host-hold-modes one-shot,resend \
  --target-position 1.0 \
  --hold-duration-sec 5 \
  --sample-hz 50 \
  --resend-hz 15 \
  --repeats 1 \
  --object-label foam_block_40mm \
  --object-width-mm 40 \
  --external-force-reference none \
  --apply \
  --confirmation-token GRIPPER_ONLY_TEST
```

如果安装了标定力计，将`none`改为`manual-gauge`。夹爪保持接触时，终端会要求录入当次N读数；
等待输入期间脚本仍以不高于10 Hz持续采样状态并执行温度保护，默认30秒超时，可用
`--manual-gauge-timeout-sec`调整。输入值会附带采集时的wall/monotonic时间戳，但它仍是人工读取的
单点静态力证据，不能伪装成同步力传感器时间序列。输入完成后夹爪才自动张开。
contact protocol每个trial只保留一次必要的Enter：夹爪自动张开后，放置或
重新定位标准物体，检查完成后按Enter即开始闭合。释放后无需再次确认；下一轮会在新的物体放置
提示处等待。不能用手持物体完成测试。

推荐顺序：

1. 左侧`repeats=1`检查安全性和数据完整性；
2. 左侧`repeats=3`正式记录；
3. 右侧按完全相同物体、安装和参数重复；
4. 如需测试更高力度，先单独审查低力度结果。不要直接把默认矩阵提升到1.0。

超过0.5的显式形式为：

```text
--allow-high-force --high-force-confirmation HIGH_FORCE
```

这只是解锁参数，不代表该力度对当前夹爪、触觉皮肤或测试物体安全。

## 7. 输出、验收与异常恢复

默认输出目录：

```text
teleop_records/gripper_actuator_identification/<timestamp>_<side>_<protocol>/
```

原有`logs/gripper_actuator_identification/`已随整个`logs`目录归档到
`legacy/logs/gripper_actuator_identification/`，仅作为历史记录和异常恢复journal的扫描来源；
新运行不再默认写入该归档目录。若要把一次新采集显式写入统一运行产物布局，给`run`增加：

```text
--artifact-layout runtime-artifacts-v1
```

此时目录为
`runtime_artifacts/runs/dmtac_gripper_actuator/<YYYY-MM-DD>/<timestamp>_<side>_<protocol>/`。
若同时提供`--output-root <path>`，该精确路径始终优先。工具只写一个目标，但开始运动前会跨旧目录、
统一布局的所有日期分区及显式目录检查未恢复journal，不能通过切换输出目录绕过恢复要求。

包含：

- `metadata.json`：端点、参数、完整trial矩阵、初始状态和原始寄存器；
- `gripper_trace.jsonl`：命令开始/结束、状态样本、trial summary和运行结束事件；
- `summary.json`：逐trial聚合；异常退出也会尽量保留已完成trial的部分summary；
- `gripper_settings.recovery.json`：恢复journal；
- `gripper_settings.recovery.guardian.log`和ready文件：guardian证据；
- `manifest.json`：正常完成后生成的证据清单，包含各文件大小和SHA-256，以及脚本、驱动、配置、
  Python、pyserial和git revision等可复现信息。

成功运行必须同时满足：

- 终端最后打印`COMPLETE settings_restored=true`；
- `summary.json`中`normal_completion=true`、`settings_restored=true`、
  `trace_queue_dropped=0`且trial数量与计划一致；
- recovery journal的`status=restored`，`restored_settings_raw`与
  `original_settings_raw`逐项一致；
- `manifest.json`列出的证据文件哈希与实际文件一致；
- 所有样本`error_code=0`，温度未越限。温度上限不仅在启动前检查，也在运动、hold、人工力计等待和
  trial间等待期间持续执行；越限会进入停止与恢复路径。

部分夹爪在完全闭合的机械端点会把开度报告为小负数，例如`-2`，其Modbus uint16表示为
`65534/0xFFFE`。辨识脚本会保留`opening_register_raw_u16=65534`和
`opening_register_signed=-2`，并仅在有符号值属于驱动既定容差`[-20,1020]`时将物理开度夹到
`[0,1000]`。超出该范围仍会立即失败，不能把任意异常数据吞掉。

异常退出后不要删除目录或journal，也不要立即启动遥操。若journal不是`restored`，执行：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py restore \
  --journal teleop_records/gripper_actuator_identification/<run>/gripper_settings.recovery.json \
  --apply \
  --confirmation-token RESTORE_GRIPPER_SETTINGS
```

若要恢复迁移前遗留的未完成journal，可把`--journal`精确指向
`legacy/logs/gripper_actuator_identification/<run>/gripper_settings.recovery.json`；普通新运行不会写入该归档树。

恢复命令会发送停止并恢复配置寄存器，但不会移动或张开夹爪。必须看到
`RESTORED verified=true`，再运行只读`inspect`核对。若物体仍被机械夹持，不要用手强拉；断开夹爪
动力或按现场硬件安全流程释放后再继续。

## 8. 如何从数据得到仿真模型

先分别拟合左右夹爪，不要默认完全一致：

1. 用free trial估计命令到首次位置变化的延迟、位置速度曲线、停止区间和空载电流；
2. 用contact trial的状态码6、位置停止点和电流确定接触/力控停止转移；
3. 固定`continuous_grasp`，比较one-shot与resend，得到主机position hold的额外作用；
4. 固定host hold，比较continuous false/true，得到固件内部持续夹取的作用；
5. 比较不同力度下的平衡开度、电流、位置漂移和外部N读数；
6. 用重复trial报告均值和方差，不用单次曲线硬编码确定性参数。

仿真action至少应包含`target_position`、归一化`force`、`continuous_grasp`和
`host_position_hold`；状态至少包含实际位置、速度估计、接触/力控停止状态和可选电流。只有外部
力计完成标定后，才把归一化力度映射为真实接触力N。

## 9. 局部行程、小步进与中途反向

只有free protocol接受`--target-sequence`。同一个trial内按顺序发送多个目标且不中途自动张开，
可用于辨识局部行程、死区、反向延迟和命令被新目标打断时的响应。先用`plan`审查矩阵，再把同样参数
用于显式`--apply`的`run`。以下是建议起始序列：

```text
# 分段覆盖完整行程
--target-sequence 0.25,0.50,0.75,1.0,0.75,0.50,0.25,0.0 --free-duration-sec 1.5

# 0.50附近的小步进/回差
--target-sequence 0.50,0.51,0.50,0.52,0.50,0.55,0.50 --free-duration-sec 1.0

# 较短驻留下的中途反向/新目标打断
--target-sequence 0.80,0.20,1.0,0.0 --free-duration-sec 0.5
```

`--free-duration-sec`是每个目标段的观测时间，不是整个序列的总时长。序列仍固定速度`1.0`，因此用于
拟合当前正式链路，而不是扫描速度参数。先在单侧低重复次数验证安全和数据完整性，再在左右两侧分别
取得至少3次重复；左右夹爪不能共用一条未经验证的拟合曲线。

## 10. 无力计的主链路位控辨识

没有外部力计时，先只建立空载位控模型。当前真机夹爪默认条件是`speed=1.0`、`torque=0.2`、
`continuous=true`；正式任务profile或CLI若覆盖了这些值，必须按运行时实际值重新采集，不能混合拟合。
`command_hz=20 Hz`和`send_threshold=0.002`属于已知的软件发送门控，执行器拟合器不从强制发送的探针中
反推它们。

每一侧建议分成独立run。先做全行程往返拟合数据：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py run \
  --side left --config latest --protocol free \
  --force-levels 0.20 --free-continuous-grasp \
  --command-mode runtime-equivalent \
  --target-sequence 1.0,0.0,1.0,0.0 \
  --free-duration-sec 1.3 --sample-hz 50 --repeats 3 \
  --apply --confirmation-token GRIPPER_ONLY_TEST
```

再做中间目标和反向的拟合数据：

```bash
python hardware/DMTAC_gripper/examples/identify_gripper_actuator.py run \
  --side left --config latest --protocol free \
  --force-levels 0.20 --free-continuous-grasp \
  --command-mode runtime-equivalent \
  --target-sequence 0.25,0.50,0.75,1.0,0.75,0.50,0.25,0.0 \
  --free-duration-sec 1.0 --sample-hz 50 --repeats 3 \
  --apply --confirmation-token GRIPPER_ONLY_TEST
```

用新的独立run重复同类序列作为held-out验证，不要把同一run同时传给fit和validation。完成左侧后将
`--side left`改成`right`原样采集。可额外采集`0.80,0.20,1.0,0.0`、每段0.5秒的快速反向数据，
但当前第一版拟合器会剔除带明显非零初速度/非单调响应的段；这些数据保留给下一版preemption模型。

离线拟合不连接硬件，输出位于`sim/rm65/real2sim/artifacts/`：

```bash
python -m sim.rm65.real2sim.gripper_position_fit \
  --input <left_fit_full_run> --input <left_fit_partial_run> \
  --input <right_fit_full_run> --input <right_fit_partial_run> \
  --validation-input <left_held_out_run> \
  --validation-input <right_held_out_run> \
  --expected-speed 1.0 --expected-force 0.2 --expected-continuous true \
  --output sim/rm65/real2sim/artifacts/dmtac_gripper_position_<date>_v1.json
```

产物按left/right、close/open分别给出从position写调用开始到实际响应的总因果延迟、最大归一化速度和
归一化加速度，并单独统计position写耗时、自动剔除段、源文件SHA-256和held-out误差。它只支持
`free_motion_normalized_position_only`声明；
Newton力、阻挡停止、柔顺、摩擦和滑移必须保持`unavailable/unmodeled`，等以后有力计和标准物体再补。

## 11. 推荐：单命令左右夹爪完整位控套件

如果左右夹爪均已正确分配，推荐使用总控脚本一次完成两侧全部数据。它按固定顺序执行：

1. 左侧拟合步进、独立held-out步进、快速反向诊断；
2. 右侧完全相同的三组实验；
3. 自动拟合left/right、close/open参数并计算held-out误差；
4. 写入总`suite_manifest.json`和position-only artifact，但不会自动更新canonical sim profile。

先生成完整计划；该命令不访问硬件：

```bash
python scripts/identify_bimanual_gripper_position.py plan
```

确认计划后，从项目根目录运行：

```bash
python scripts/identify_bimanual_gripper_position.py run \
  --config latest \
  --apply \
  --confirmation-token BIMANUAL_GRIPPER_POSITION_SUITE
```

脚本只要求每侧开始前各一次强确认：左侧输入`LEFT_GRIPPER_PATH_CLEAR`，右侧输入
`RIGHT_GRIPPER_PATH_CLEAR`。同一侧的三个free-motion阶段随后自动连续执行；任何时刻Ctrl-C都会终止套件，
当前子阶段仍沿用独立采集器的停止、寄存器恢复和guardian机制，后续阶段不会继续。

第一次确认前会只读解析并冻结左右串口，拒绝左右映射到同一设备、未恢复journal或端口已被其他进程占用。
自动分析要求`left/right × close/open`四组均达到最低拟合样本数且都有独立held-out结果；不满足时原始数据和
artifact仍会保留，但总manifest标记失败，不会把不完整结果当作可采用模型。

默认完整套件固定`speed=1.0`、`force=0.2`、`continuous=true`、
`runtime-equivalent`、50 Hz状态读取和55°C温度上限。数据默认保存到：

```text
teleop_records/gripper_position_suite/dmtac_bimanual_position_<timestamp>/
```

原有`logs/gripper_position_suite/`随整个`logs`目录归档到`legacy/logs/gripper_position_suite/`，
预检仍会只读扫描其中的历史恢复journal。显式增加
`--artifact-layout runtime-artifacts-v1`后，新套件写入
`runtime_artifacts/runs/dmtac_gripper_position_suite/<YYYY-MM-DD>/dmtac_bimanual_position_<timestamp>/`；
`--output-root <path>`仍具有最高优先级。套件预检会同时扫描独立采集器与套件的新旧根目录，避免遗漏
此前未恢复的夹爪设置journal。

其中`runs/<side>/<phase>/`保留每个原始trace、summary、recovery和manifest，
`analysis/dmtac_gripper_position_fit.json`是自动拟合结果。`fast_preemption`只作为后续中途反向模型证据，
不会被第一版零初速步进模型悄悄混入主拟合。

阶段切换时，恢复`continuous`寄存器可能让固件在夹爪已全开时持续报告moving。套件内部的下一阶段预检
只读等待最多3秒：正常情况要求连续3次status idle；若status保持moving，只有连续5次同时满足安全全开端
和低电流才允许继续。它不会发送清状态或位置命令。位置离开全开端、高电流、非零错误码或超温仍拒绝，
判定证据保存为每个子run的`preflight.json`。

若套件中途失败，可在排除故障并确认没有未恢复journal后续跑原目录：

```bash
python scripts/identify_bimanual_gripper_position.py run \
  --config latest \
  --resume-suite teleop_records/gripper_position_suite/dmtac_bimanual_position_<timestamp> \
  --apply \
  --confirmation-token BIMANUAL_GRIPPER_POSITION_SUITE
```

续跑会重新校验左右设备、阶段计划、源条件、已完成子阶段的manifest/summary哈希以及
`normal_completion/settings_restored`，只跳过证据完整的阶段；当前待运行侧必须重新人工确认。
如果latest中的左右assignment相对原suite发生变化，必须新建套件，不能用端口互换来重标旧阶段；
旧数据保留为侧别待确认证据，直到有当时的稳定USB identity快照证明物理设备映射。
