# 基础硬件 Probe 使用说明

本文档说明两个受限的硬件探针脚本：

- `scripts/check_dmtac_gripper_health.py`：DMTAC/Inspire 夹爪的预览和显式解锁微动检查。
- `scripts/read_rm65_status.py`：RM65 单臂或双臂的只读状态读取。

这两个脚本不是 teleoperation、collection、replay 或 policy deployment 的入口，也不构成硬件 ready、标定完成或安全验收证明。文件名中含有 health、status 或 probe 也不代表命令必然只读；执行前必须先看清下方的风险等级和实际参数。

## 风险等级和动作边界

| 脚本 | 默认行为 | 可能的设备动作 | 明确不会做的事 |
| --- | --- | --- | --- |
| check_dmtac_gripper_health.py | 只生成计划，不连接夹爪 | 只有同时传入 --execute --confirmation GRIPPER_HEALTH_CHECK 才会进行有限的夹爪微动 | 不启动 teleop/collection，不读取或启动 DM-Tac 相机/触觉链路 |
| read_rm65_status.py | 连接并读取状态 | 建立 RM65 SDK 连接并执行读取调用；不发送运动命令 | 不 enable/disable、stop、clear fault、改配置、homing 或发送轨迹 |

任何真实设备命令都必须在设备操作者明确知情、机器人控制程序已停止、急停可立即触达的条件下执行。探针输出只能说明本次调用返回了什么，不能替代下层安全门、现场检查或正式验收。

## 1. DMTAC 夹爪微动健康检查

### 1.1 默认预览（无设备连接、无夹爪动作）

默认不带 --execute，脚本只解析端点、检查参数并打印计划：

```bash
python scripts/check_dmtac_gripper_health.py \
  --side all \
  --config latest \
  --json
```

只查看一侧：

```bash
python scripts/check_dmtac_gripper_health.py \
  --side right \
  --config latest \
  --json
```

不使用 assignment 文件时可以显式传串口端点：

```bash
python scripts/check_dmtac_gripper_health.py \
  --side right \
  --config none \
  --right-port <right-gripper-serial-port> \
  --json
```

预览成功只表示端点和参数可以形成计划；它没有证明串口存在、夹爪在线或夹爪可以安全动作。

### 1.2 微动参数

| 参数 | 默认值 | 约束和语义 |
| --- | ---: | --- |
| --side | all | all、left 或 right；双侧执行时脚本按侧顺序执行，不并行抢占 |
| --amplitude | 0.04 | 0 < amplitude <= 0.10；driver joint position 在 [0,1] 上的总行程，不是毫米 |
| --force | 0.20 | 0 < force <= 0.5；设备归一化设置，不是牛顿 |
| --speed | 0.20 | 0 < speed <= 0.5；设备归一化设置 |
| --wait-timeout | 3.0 秒 | 必须为正；每次目标移动等待 idle 的最长时间 |

脚本以读取到的起始位置为中心构造有界窗口，然后依次执行：

```text
读取起始状态 -> lower target -> upper target -> 恢复起始位置
```

每个阶段都会检查：

- 夹爪不再运动；
- error_code == 0；
- 状态码属于允许的 idle 状态；
- 实际位置在 POSITION_TOLERANCE = 0.015 内到达目标。

正常完成后会恢复原 speed、force 和 continuous grasp 设置，并关闭连接。异常发生在微动已经开始后，脚本会尝试调用夹爪急停；急停调用仍需要现场人员确认，不能把 Python 返回值当作物理停止证明。

### 1.3 真实微动（显式解锁，禁止直接盲目复制）

下面是受保护的命令形态；缺少确认字符串时脚本会拒绝动作：

```bash
python scripts/check_dmtac_gripper_health.py \
  --side right \
  --config latest \
  --amplitude 0.04 \
  --force 0.20 \
  --speed 0.20 \
  --execute \
  --confirmation GRIPPER_HEALTH_CHECK \
  --json
```

执行前必须逐项确认：

1. 已停止 teleop、数据采集、policy deployment 和其它会写夹爪命令的进程。
2. 两个夹爪的 jaws 周围没有手、工具、工件或会被夹住的线缆。
3. 急停位置和断电路径可立即触达。
4. 只选择要检查的一侧；没有必要时不要使用 --side all。
5. assignment 中的串口、ID、baudrate 和当前物理接线已经人工核对。
6. 已先运行预览命令并检查 JSON 中的端点和目标范围。

执行期间不得同时启动其它夹爪控制程序。若出现运动不停止、位置明显偏离、错误码、通信异常或机械干涉，立即使用现场急停并保留完整输出；不要重复运行来“试到成功”。

### 1.4 结果解释

JSON 中的关键字段：

- mode：preview 或 execute；
- plans：预览的端点、行程、速度、力和等待时间；
- results[].states：start、low、high、restored 四个状态快照；
- results[].settings_restored：软件是否走过恢复设置路径；
- results[].ok：本次脚本内部检查是否通过。

ok=true 只表示脚本观测到的返回码、idle 状态和位置误差满足脚本阈值，不等于夹爪没有机械磨损、负载正常或可以进入正式采集。

## 2. RM65 机械臂状态读取

### 2.1 只读命令

使用 assignment 文件读取双臂：

```bash
python scripts/read_rm65_status.py \
  --side all \
  --config latest \
  --json
```

只读取单侧：

```bash
python scripts/read_rm65_status.py \
  --side left \
  --config latest \
  --json
```

不使用 assignment 文件时，必须显式提供 IP；以下是参数模板，不是实际地址：

```bash
python scripts/read_rm65_status.py \
  --side all \
  --config none \
  --left-ip <left-rm65-ip> \
  --right-ip <right-rm65-ip> \
  --json
```

### 2.2 读取内容和退出码

脚本为每个选择的 RM65 读取并记录：

- API version；
- robot information；
- arm software information；
- current arm state；
- joint enable state；
- joint temperature；
- joint current；
- joint voltage；
- joint error and brake state；
- 连接关闭返回码。

输出的 mode 固定为 read_only。所有选中机械臂的读取和关闭都没有错误时退出码为 0；状态返回异常或 SDK 查询异常时退出码为 1；assignment/IP 参数解析失败时退出码为 2。双臂读取时一侧失败不会静默丢弃，另一侧仍会被尝试，最终 ok 由全部结果共同决定。

只读不等于无风险：它仍会打开网络连接并调用 RM65 SDK，应该避免在网络或 SDK 状态异常时高频重试。读取结果也不是“机械臂可运动”的授权，不包含碰撞、制动器、力矩、标定、急停或现场环境验收。

## 3. 推荐检查顺序

1. 停止会写设备命令的上层进程，并确认没有遗留 policy/teleop worker。
2. 先执行 read_rm65_status.py，保存原始 JSON；如果状态读取失败，不进入夹爪微动。
3. 对夹爪只执行默认预览，人工核对串口和目标窗口。
4. 只有明确需要物理微动时，才在现场安全条件满足后加上 --execute --confirmation GRIPPER_HEALTH_CHECK。
5. 微动结束后确认 restored 快照、settings_restored 和现场机械状态；将输出与时间、操作者、设备端点一起留档。
6. 若随后要恢复 teleop/采集，重新走其正式启动和安全检查流程，不把 probe 输出直接注入控制链。

## 4. 不应做的事情

- 不要在 teleop、采集或 policy deployment 正在运行时执行夹爪 --execute。
- 不要把 --force 当作牛顿，把 --speed 当作 Hz，或把 --amplitude 当作毫米。
- 不要从 read_rm65_status.py 的成功退出码推断机械臂已经通过完整硬件验收。
- 不要把 preview、只读状态或离线单元测试结果写成真实设备已验证。
- 不要为了绕过确认字符串修改脚本、删除 --confirmation 或直接调用底层 vendor SDK。
- 不要在错误码、急停、通信中断或异常运动后自动重试；先保留日志并进行现场排查。

## 5. 验证边界

脚本的单元测试使用 fake endpoint/fake SDK，验证参数、动作序列、恢复路径和错误处理；它们不会连接真实 RM65，也不会移动真实夹爪。运行测试时使用项目规定的 Python 3.10 环境和 pytest wrapper，例如：

```bash
bash scripts/run_pytest.sh \
  tests/test_check_dmtac_gripper_health.py \
  tests/test_read_rm65_status.py -q
```

测试通过只能证明软件契约和安全解锁逻辑，没有提供现场硬件 ready 证据。真实设备验证必须另行记录设备身份、端点、操作者、时间、原始 JSON、急停可用性和任何异常。
