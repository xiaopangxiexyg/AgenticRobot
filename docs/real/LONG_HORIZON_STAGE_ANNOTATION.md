# 长程任务细粒度阶段标注

这是一个纯离线、独立于真机遥操页面的 Control Station 应用。它只打开已经完成
`episodes.jsonl` 发布、且 progress manifest 已封存的 Episode；不会连接设备、启动
机械臂、修改原始 Parquet/视频或进入当前仍在采集的最新 Episode。

这里有两种不同的“封存”：

- `episodes.jsonl` 已发布且 progress manifest 已封存：说明采集事实闭合，Episode
  可以进入标注队列；
- stage annotation set 已封存：说明某一版二次阶段标注已经通过校验并不可变发布。

前者不代表后者已经完成。界面分别显示 `Progress 已封存` 和
`阶段标注已封存 N 版`。

对于没有 capture-causal Progress facts 的 v2.3 writing 历史数据，正式方案是在对应数据集内
建立 namespaced `RETROSPECTIVE_MIGRATED` semantic layer；writing adapter 只负责一个 v2.3
数据集的标注，不承担跨数据集 workspace 管理。该入口尚待实现，目标操作合同见
[Writing 第一子任务 retrospective 三阶段标注](WRITING_STAGE1_RETROSPECTIVE_ANNOTATION.md)。

## 0. 独立标注机器环境

后期标注不需要完整 DexTeleop 真机环境。从仓库根创建专用 Python 3.10 环境：

```bash
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

该环境只安装标注 GUI、数据合同读取和 Parquet 时间轴所需依赖，不安装机器人 SDK、
RealSense、OpenCV、仿真、训练或策略部署栈；标注机器另外需要系统级 `ffmpeg` 和
`ffprobe`。数据集必须位于标注机器本地可访问的绝对路径或已经挂载的共享文件系统。
H.264 MP4 由浏览器直接解码；采集默认的 `mp4v` MP4 首次打开时由服务转为临时 H.264
缓存，缓存不写入数据集目录，原始视频保持不变。安装边界和自定义环境路径见
[`README_ENV.md`](../../README_ENV.md#2-只安装后期标注环境)。

## 1. 启动

激活 `.venv-annotation` 或完整的 `dexteleop310` 后，直接启动 Control Station，然后在
打开的页面里填写数据集绝对路径和标注员 ID。数据集路径可以直接点击“浏览目录”使用
页面内目录浏览器，不需要手打路径；选择后仍需点击“打开标注工作区”完成校验和加锁：

```bash
cd /path/to/DexTeleop
python -m control_station refinement-annotation
```

命令行路径和 `--annotator-id` 仍可作为 GUI 的可选预填值，但不再是启动必填项：

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/final_dataset \
  --annotator-id operator_02
```

普通标注员可以在不启用特殊修正工具的情况下给 Episode 标记“需要二次复核”。进行
二次复核时，在启动页勾选“二次复核模式”，或使用参数：

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/final_dataset \
  --annotator-id operator_02 \
  --second-review
```

二次复核不要求更换 `annotator_id`。该参数只区分本次封存版本的用途；打开待复核
Episode 时，工作草稿中的待复核标签会自动清除。只有再次校验并封存为
`SECOND_REVIEW` annotation set 后，已发布数据视图中的整条待复核排除才解除；仅打开
页面或只保存 mutable draft 不会改写旧封存版本。

页面会在确认后才取得 writer lock；路径必须是已经存在的数据集绝对路径，输入错误不会
创建新目录。浏览器会在本机保存最近使用过的路径和标注员 ID，供下次从下拉建议中选择。
进入标注后可点击顶部“更换数据集”，选择新目录并确认切换；如果当前存在未封存修改，
切换前必须选择“校验封存并切换”“放弃修改并切换”或“取消切换”。界面不再提供容易
与正式完成混淆的“保存草稿”操作。

只检查指定数据集当前有多少 Episode 可标注、不启动 GUI，也不取得标注 writer lock：

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/final_dataset \
  --dry-run
```

服务器只允许绑定 loopback。升级后的多个标注程序可以通过 SSHFS/NFS 同时打开同一个
数据集，但同一时刻只能有一个程序编辑某个 Episode。跨机器互斥使用服务端原子创建的
`meta/progress/stage_annotation_locks/<episode_uuid>.lock/` 目录；其中 `owner.json`
记录标注员、主机、PID、取得时间、随机 lock ID，以及经过哈希的机器身份、boot ID 和
进程启动标识。程序切换 Episode 或关闭时，只有再次核对 lock ID 后才会用同目录原子
rename 释放锁。锁不会按时间自动抢占；新客户端只会自动隔离能够证明属于本机、且原进程
已消失、成为 zombie、机器已重启或 PID 已复用的锁。其他机器、旧格式或身份信息不完整的
锁继续 fail-closed，必须先核实记录的 owner 已停止再人工处理，禁止仅凭“时间较久”删除。

历史的 `meta/progress/stage_annotations/.writer.lock` 现在只作为旧客户端兼容 guard：
新程序持有共享锁，旧程序仍请求独占锁。SSHFS 配置不一定跨机器传播 `flock`，因此共享
数据集上的所有标注机器必须一起升级，不能依赖该 guard 安全混用新旧版本。不同 Episode
的 draft 和不可变 annotation set 位于互不重叠的目录，可以并行写入；它仍可与采集程序
并行，因为采集只写新的 Episode。旧版本留下的本地 draft 文件仍可读取，但只作为内部
兼容工作缓存，不是用户可选择的正式保存结果。
如果旧 draft 没有对应的不可变 annotation set，页面会明确把它视为“尚未封存”，即使
本次打开后还没有继续拖动时间轴，切换 Episode 或数据集时也必须封存或明确放弃。只有
草稿的可编辑语义与最新封存版本一致时，才会作为已封存缓存处理；这里不依赖经常变化的
运行配置 hash。

旧数据集是否能使用“失败发生于”和 `failure_trigger`，取决于它自身已封存的 Progress
快照是否包含完整 stage profile、stage-entry binding 和采集时记录的 Failure fact。满足
这些既有事实条件的旧数据集可直接使用；只有回溯 Progress、没有真实 capture Failure
的旧数据不会被静默伪造成 Failure。当前版本也不会把后标注的 SUCCESS 改 FAILURE 当成
新的采集 Failure。

## 2.1 旧 coarse 标注的可选 detail 补充

detail 是叠加在 coarse annotation set 之上的独立补充，不修改原有 stage bundle、
Progress facts、视频、Parquet 或采集协议。当前配套 profile 使用同名的
`.progress_detailed.json` 和 `.stages_detailed.json`。一个 source profile 可以为多个
coarse Progress Step 提供 fine stages；进入标准 annotation 后，界面按每个 changed Step
投影成独立的 detail 选择，每个不可变 overlay 仍只绑定一个 target Step，未选中的 Step
继续使用 coarse stages。

已有 coarse annotation set 在没有 detail 补充时明确解析为 `COARSE_FALLBACK`，因此旧
数据集不需要迁移文件就能继续使用，且不会把均分出来的界面草稿当成 detail 数据。进入
标准 annotation 时即可选择兼容的 detail profile，不要求 coarse set 先封存；coarse 尚未
封存时，detail 草稿保持未绑定，待 coarse 版本确定后再绑定。detail 轨道独立于 coarse
轨道，可单独拖动其边界。原始 Failure 仍只读，Failure Attempt 必须在 detail 轨道中选择
失败发生的 fine stage；选择前不能确认或封存。

detail 封存采用独立的不可变目录：

```text
meta/progress/detail_stage_annotations/<episode_uuid>/<detail_set_id>/
  binding.cjson
  annotations.cjson
  manifest.cjson
```

它精确绑定当前 Episode、当前 coarse annotation set 和目标 detail profile。成功 Attempt
必须覆盖目标 fine-stage 的完整顺序；原始 Failure Attempt 只能是从第一 stage 开始、以
对应 Failure 结束的顺序前缀；Retry 是独立 Attempt，仍需覆盖完整序列。任一 Attempt 未
完成 detail，detail set 都不会发布，分析继续使用 `COARSE_FALLBACK`。粗标注重新封存会
产生新的 base 版本并使旧 detail overlay 失效，页面会从新 base 重新生成未封存的 detail
草稿，不会静默沿用旧 detail。

detail overlay 只供后标注和后续显式选择 detail 视图的分析使用；现有只读取 coarse
annotation set 的调用不改变。对于新 dataset，采集 GUI 也可以显式选择
`dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed`，此时三个 writing
fine stage 会作为 capture-time stage profile 直接进入 coarse timeline，不再需要重复的
detail overlay；这个 variant 不能与已经绑定 coarse stage profile 的 dataset 混用。用旧
coarse profile 采集的 Episode（包括只从 `right_write` 开始的窗口）仍可在粗标注封存前或
封存后按上面的 additive overlay 流程补齐 detail；两种标注草稿互不替换。采集时显式选择
的 `_detailed` profile 则是普通 capture-time stage profile，直接进入该 Episode 的阶段
事实，不需要再通过 overlay 重新表达同一份采集阶段。

当前实现把 detail profile 发现、当前 overlay 选择和完成/失效状态统一放在
`control_station.refinement_annotation.detail_annotation` 模块；GUI application 只协调
Episode lock、draft 和封存。粗标注重封存后，旧 overlay 会显示为“详细标注需重做”，
而不是与“尚未配置 detail profile”混为同一个布尔值。左侧栏使用版本化的 additive
`annotation_status` 投影突出“整条异常”“待二次复核”“详细标注需重做”、原始失败
Attempt 数和最新版本的失败阶段标注数，同时继续返回旧的 `review_state`、
`whole_episode_abnormal` 和 `detail_annotation_complete` 字段，因此旧 GUI/数据集仍按
原语义工作。原始失败计数直接来自不可变 Progress timeline；标注失败计数只读取最新
stage annotation set，两者不会混成“整条异常”。

大数据集仍按页加载，只有可见 Episode 才读取精确状态；摘要批次并行覆盖远程文件系统
延迟，服务端对象/摘要缓存都有上限。浏览器复用既有 Episode 行，只更新状态确实变化的
行；目录 revision 未变化时，4 秒轻量轮询保留已经读取的精确摘要，不会把行退回
“状态读取中”并重复访问远程文件。`RETROSPECTIVE_MIGRATED` 工作区轮询使用固定次数的
目录 metadata probe，只有检测到变化或到达保守复核周期才枚举完整历史 Episode；初次
接入和检测到变化后的完整合同校验不省略。

## 2. 时间轴语义

- 粗粒度 Progress Step、Attempt、Advance 和原始 Failure 来自第一次采集事实；
- 原始 Failure 刻度只读，不能由细标界面移动或删除；
- 每个 coarse Attempt 内的 fine stage 必须按其 task stage profile 顺序出现；
- 一个 stage 可以失败多次；Failure 后的 Retry 必须继续同一个 stage；
- stage 成功后才进入下一个 stage，不能倒退或重复已经成功的 stage；
- 对于开头恢复采集，正确做法是在采集启动时记录
  `entry_condition=RECOVERY_FROM_FAILURE` 和准确的 `entry_stage_id`；
- 界面也允许按顺序删除开头或末尾的 stage 标注，以表达 capture window 本身不完整；
  中间阶段不能任意删除，避免产生无法解释的错序。

第一次打开 Episode 时，系统按照 coarse Attempt 均分生成可编辑草稿。对于类似
“A1 成功、A2 失败”的第一次 Attempt：

1. 在失败 Attempt 卡片的“失败发生于”中选择 A2；
2. 系统生成 `A1 SUCCESS + A2 FAILURE`，原始 Failure 保持在原时间；
3. 后续 Retry 自动从 A2 开始，并按 profile 确定性重排；
4. 将视频定位到 A1 完成/A2 开始的画面，再拖动白色边界精调。

若选择 A3，系统会一次性生成
`A1 SUCCESS + A2 SUCCESS + A3 FAILURE`，而不是要求重复点击添加。更改失败 stage
会重新均分当前 Attempt 和同一 coarse step 的后续 Retry 边界，因此已有人工边界时会
先要求确认。系统不会添加任意 stage：成功前缀和 terminal Failure 都必须严格服从
profile；有效 sample 不足以让每个 occurrence 至少拥有一个样本时会拒绝重排。

如果正式数采在非最终子任务直接按“数采结束”，采集事实会保留为
`MANUAL_STOP + ABORTED`（窗口采集也可能是 `WINDOW_COMPLETED +
TRUNCATED_AT_CAPTURE_END`），不会伪造成 `FULL_TASK_COMPLETED`。后标注界面此时显示
“采集结束位置”选择器：明确的 `MANUAL_STOP` 新建草稿默认选择原始最后 Attempt
所在的子任务，不会推断未记录的后续流程已经完成；若本条实际完成到更晚子任务但采集端
没有记录对应 Advance，标注员可显式选择实际最后正常完成的子任务。
`ABORTED/UNKNOWN` 等旧式或模糊中止不自动套用默认值，必须由标注员显式选择。系统随后按线性
profile 重建所选结束位置对应的成功子阶段，并允许继续拖动白色边界精调；如果采集端
已经误启动了下一个 Attempt，再选择前一个子任务会从该子任务最近一次 Attempt 起点
覆盖到 Episode 结束，而不会把后续误启动 Attempt 当成已完成任务。封存结果在原 stage
annotation bundle 中以可选
`capture_end_step` 记录，绑定原 terminal fact、原 final Attempt、所选 Step 和 Episode
clock；训练阶段投影可以据此验证，而原 Progress facts、视频、样本及 full-task outcome
均不修改。安全停机、带真实 Failure 的 Attempt、非线性 profile、早于 Episode 入口或
没有对应原始 Attempt 可锚定的选择会被拒绝。

没有 `capture_end_step` 的旧 draft 和旧 annotation set 继续按原语义读取，哈希不变，
不会在打开时自动覆盖已有人工边界；但符合上述事实条件的旧草稿可通过选择器显式重排，
重排前会提示它将重置最终一段的人工子阶段边界。若该段已有可恢复失误区间或重叠异常
区间，必须先处理这些附加标注，服务端不会猜测迁移其引用。

拖动阶段间的白色游标时，前端会把一帧内的多个鼠标事件合并为一次视频定位，并只在
拖动结束时重绘阶段说明；上方普通视频进度条独立工作。当前视频使用浏览器自动预读，
当前 Episode 播放后会在后台预热下一个多阶段 Episode 的默认视角。只有一个阶段的
Episode 默认不自动读取视频，点击“加载视频”或切换视角时才读取，避免很少查看的视频
占用带宽。切换 front、left/right wrist 等视角不会改变时间轴，默认优先使用 `front`。

每个 coarse Progress Step 使用稳定且明显不同的主色系；同一 coarse step 内的 fine
stage 保持同一色相，并按照 profile ordinal 从深到浅改变明度与背景对比度。因此可以
同时识别“属于哪个子任务”和“处于该子任务的哪个阶段”，同一阶段的 Retry 仍保持同色。
Failure 在对应颜色上叠加红色斜纹，而不是抹掉 Progress Step 身份。
点击阶段块会选中该 occurrence；右上角红色“删除所选阶段”只允许依次删除 Episode
开头/末尾的成功阶段，或异常区间覆盖及其两侧相邻阶段，避免在普通中间位置制造无依据
的阶段跳跃。删除只移除阶段标签，并把其完整时间范围并入同一 coarse Attempt、同一
Progress Step 内的前一或后一有效阶段，不会产生未标注时间空洞，也不会跨 Attempt
或改写原始 Failure；承载原始 Failure 的阶段本身仍不可删除。“撤销删除”按后进先出
顺序恢复被删 occurrence 和合并前的精确边界。逆操作记录只属于当前页面编辑，不进入
正式 annotation contract。若后续边界编辑已经改变接收阶段，则回退 fail closed，不
猜测覆盖更新后的边界。

标注视频与时间轴以 Parquet 中正式接受的 sample timestamp 对齐，而不是按 Episode
首尾 wall time 做简单线性映射。短 Hold/Pause 空洞按实际时长显示，长空洞在 GUI
显示轴最多占 1 秒，并用黄色竖纹标出；视频、Parquet、原始时间戳、阶段 annotation
时间和训练索引均不修改。拖动阶段边界时只允许落在正式 sample 或记录明确的暂停
起止边缘；服务端会拒绝暂停空洞内部无法观测、也没有事实依据的任意时间点。

## 3. Failure 与训练排除

细标不会重写 Failure 事实。训练索引按下面的闭区间时间语义派生：

```text
当前失败 stage occurrence 的起点
→ 原始 Failure 时间戳（含该时间点）
```

因此 A1/A2 已成功、A3 失败时，只排除本次失败 A3，不会把 A1/A2 一起废弃。Retry 的
成功 A3 是新的 occurrence/Attempt，仍可进入默认训练视图。

### 3.1 failure_trigger 失败发生时刻细标

对于采集事实中已经记录了 Failure 的 Attempt，界面额外显示一条独立的
`failure_trigger` 时间轴。它只划分“失败前的正常行为”和“失败后的情况”，不移动或
删除原始 Failure、Attempt、阶段边界或既有训练排除结果。

- 可拖动范围严格为当前失败 fine stage 的 `[B2 起点, t_failure]`，也就是该失败最多
  覆盖当前一个子阶段；不能拖到上一个子阶段或 Failure 之后；
- 真值是 Parquet 正式 sample axis 上的 `trigger_sample_index`，界面拖动会吸附到离散
  sample，不保存连续时间。`trigger_at`、上下界时间只是同一 sample 的可读投影；如果
  原始 `t_failure` 落在两个 sample 之间，离散上界取不晚于它的最后一个正式 sample；
- 首次默认值是采集时的 `t_failure`。采集端默认在原始 Failure 成功持久化后写一份独立
  receipt，位置为 Episode progress 目录下的
  `failure_trigger_capture/<source-failure-hash>.cjson`；该 receipt 不进入既有 facts、
  不改变 reducer 或暂停/重试流程。`ReportAttemptFailureRequest` 的
  `capture_failure_trigger=False` 可作为回退开关；即使关闭，已有 Failure 时间仍可作为
  后标注默认值；
- 后标注 draft 使用可选的 `failure_triggers` 字段，封存后写入 stage annotation set
  内的 `failure_triggers.cjson` sidecar。原有 `binding.cjson`、`annotations.cjson`、
  `manifest.cjson` 和旧 draft 都保持兼容；没有该 sidecar 的旧集合会按上述默认值补齐；
- 本版本只处理采集时已有 Failure fact 的情况。后标注把成功结果改成 Failure 的后续
  语义仍不纳入本次 `failure_trigger` 协议；训练排除策略也不因该新增时间轴改变。

如果 Episode 末尾没有第一次采集 Failure，现有后标注仍可表达 post-capture tail outcome；
但本版本不会为这种“成功改失败”情况创建 `failure_trigger`，也不会把它伪装成 capture
Failure。

### 3.2 中间失误后恢复

一次 Attempt 最终成功，但某个中间子阶段内短暂失误并自行恢复时，不应伪造成 capture
Failure，也不应生成 Retry。界面的“中间失误后恢复”记录一份独立、分析用途的
`recoverable_failures` 信息：

- 依次选择“失误发生”和“恢复正常”两个点；服务端将两个时间吸附到 Parquet 的离散
  sample index，连续时间不是协议真值。超出 Episode sample axis 的时间会直接拒绝，
  不会静默夹到首尾样本；
- 两个样本必须共同落在同一个现有 `SUCCESS` stage occurrence 内，系统据此自动确定
  coarse Attempt、Progress Step 和 stage。跨越白色阶段边界、落在真实 Failure 阶段，
  或之后调整边界导致原 occurrence 引用变化时都会 fail closed，不会猜测改挂到相邻阶段；
- 同一 stage occurrence 内的多个区间不能重叠，也不能把同一个离散样本同时作为前一
  区间的“恢复正常”和后一段的“失误发生”；两段之间至少保留一个明确正常样本；
- GUI 取点时立即显示吸附后的 sample index 和所属 SUCCESS 子阶段，可取消当前取点、
  定位已有区间或进行安全重标。重标期间原区间保持不变，只有新起止点通过服务端校验
  后才原子替换页面草稿中的旧区间；
- 该信息不会改变 `StageOccurrenceAnnotationV1.outcome`、原始 Failure、Retry、
  `failure_trigger` 或默认训练 exclusion；需要使用它的后续分析必须显式读取 sidecar；
- 封存后写入同一 annotation-set 目录下可选的
  `recoverable_failures.cjson`。没有该字段和文件的旧 draft、旧 annotation set 均解释为
  “没有记录可恢复失误”，不会回填或修改旧数据。早期若已存在共享边界样本的两个
  V1 区间，仍可读取和查看，但生成新封存版本前需要通过重标消除该离散语义歧义。

## 4. 特殊工具

特殊工具默认关闭，需要显式勾选后才能使用：

- 异常区间：相当于局部 Episode 断开，前后 stage 不要求完整，区间不进入默认训练；
  添加时不会再自动裁剪或删除 stage。异常带显示在时间轴底部，被覆盖的 stage 会保留
  并高亮，标注员再通过边界调整或点选后删除来完成对齐。若异常区覆盖原始 Failure，
  固定 Failure 刻度仍永久保留；
- 粗进度边界修正：只改变 retrospective view，原始 Advance 事实永久保留；
  启用特殊工具后，主阶段时间轴会直接出现橙色拖动柄，列表中的滑杆仍可使用。移动
  主阶段边界会同时调整前一 Attempt 的末段终点和后一 Attempt 的首段起点，防止只改
  粗边界后留下重叠或空档；边界不能越过相邻主阶段；
- 整条 Episode 异常：coverage 变为 `EXCLUDED_EPISODE`，默认训练排除整条 Episode，
  但既有 stage 标注和原始内容仍保留，便于审计或以后采用不同 policy。

“需要二次复核”不是异常结论，而是独立的待决标签。带该标签的封存版本同样得到
`EXCLUDED_EPISODE` 和整条默认训练排除，但 exclusion kind 明确为
`WHOLE_EPISODE_REVIEW_PENDING`。普通标注版本记录为 `STANDARD`；二次复核版本记录为
`SECOND_REVIEW`。旧版本保持不可变，因此可以完整追溯“谁在什么版本提出待复核、后来
由哪个二次复核版本解除”。二次复核后，既有异常区间、失败阶段等其他排除不会被取消。

关闭特殊工具会要求确认，并清除尚未封存的特殊草稿字段。
同一浏览器会保留启用特殊工具前的内存快照；取消启用会恢复该阶段草稿，而不是留下被
异常操作破坏的时间轴。服务端也会拒绝
`privileged_tools_enabled=false` 但仍携带异常字段的请求。

Episode 列表每 4 秒重新读取一次，只会增量显示已经完成正式发布和 Progress seal 的
新 Episode；当前仍在写入的 Episode 始终不可见。若标注服务进程退出或连接断开，页面
会显示红色离线状态并冻结标注操作；连接恢复后自动解锁。

左侧 Episode 区可收起。展开态显示 Episode 编号、时长、帧数、封存版本、整条异常、
待二次复核、二次复核完成、原始失败 Attempt、最新失败阶段标注和真正尚未封存的草稿；
与最新不可变版本一致的本地工作缓存不会误报为未封存。这些 annotation 摘要只对当前
可见行按需读取，避免大数据集启动时扫描每个 annotation bundle。完整 UUID 保留在悬停
信息中；收起态显示当前 Episode 编号、列表位置以及异常/复核/失败状态，展开态和收起态
都可以用“上一条/下一条”切换。浏览器会记住该布局偏好；若刷新后处于收起态且尚未选择
Episode，点击“下一条”会从列表第一条开始。

## 5. 封存与回退

界面只有“校验并封存标注”一个正式保存动作。它会执行严格 profile 顺序、Attempt/Fault
归属、时钟、coverage 和 exclusion 检查，通过后把不可变集合发布到：

```text
meta/progress/stage_annotations/<episode_uuid>/<annotation_set_id>/
  binding.cjson
  annotations.cjson
  failure_triggers.cjson       # 可选附加信息
  recoverable_failures.cjson  # 可选、仅分析用途
  manifest.cjson
```

所有存在的 sidecar 先写入，`manifest.cjson` 最后发布并作为整套结果的可见性边界。
封存不会删除旧 annotation set，也不会在 Parquet 中添加
逐帧布尔列；loader 通过 sample-time 索引快速得到 occurrence 和排除区间。当前尚未
默认选择多个 annotation set 中的某一个，训练发布前仍需显式 selection policy。

为了兼容已经存在的数据，服务端仍可读取旧的 mutable draft 文件，并会在成功封存后
缓存最近发布状态；但 GUI 不再允许标注员把“只保存草稿”当作完成操作。切换 Episode
或数据集时，有修改就直接要求封存、放弃或取消。关闭标签页、刷新或离开页面时，如果
仍有仅存在于浏览器内的修改或封存后尚未补写成功的工作草稿，浏览器也会触发离开确认；
该确认不会替代不可变封存。

数据集位于 SSHFS 等远程文件系统时，封存发布仍遵守同一不可变、manifest 最后写入的
协议。对可证明可重入的瞬时 I/O 故障，应用只使用同一个 `annotation_set_id` 做一次短暂
重试；持续权限或连接错误会返回结构化错误，浏览器中的修改不会被清空。sidecar 写入
失败时 Stage manifest 不会发布，因此不会暴露半套结果；若完整不可变 set 已完成、仅
mutable draft 缓存更新失败，界面会明确报告“封存成功但缓存失败”，不会要求再次发布
一个重复 set；重新打开时可从最新封存版本重建。

已经存在封存阶段标注的 Episode 可以通过“强制重新编辑所选版本”进入二次编辑：

1. GUI 列出该 Episode 下所有完整且绑定一致的不可变 annotation set；
2. 标注员选择一个版本并二次确认；
3. 服务只读取该封存版本并在页面内物化为新的页面编辑，不立即发布任何结果；
4. 再次“校验并封存”始终生成新的 annotation set，旧版本永久保持只读。

该入口不会定义“最新版本自动获胜”的训练选择策略；下游仍必须显式选择 annotation
set，避免一次返修静默改变既有训练数据视图。

## 6. 恢复采集入口

managed progress profile 会自动绑定同名 `.stages.json`。采集一条从失败 stage 开始的
recovery Episode 时，在原真机命令上同时增加：

```bash
--collection-progress-profile catalog:<progress_profile_id> \
--collection-progress-entry-step <progress_step_id> \
--collection-progress-entry-stage <stage_id> \
--collection-progress-entry-condition recovery_from_failure
```

非首 stage 若没有 `recovery_from_failure` 会在设备启动前被拒绝。该入口写入
Episode-local `stages.cjson` 和 `stage_entry.cjson`，并在 dataset 级
`meta/progress/stages.cjson` 锁定唯一 stage profile；同一数据集不能混装不同 stage
profile 或 profiled/unprofiled progress Episode。

若只需要从某个 coarse 子任务的开头采集（不是从该子任务内部的失败 fine stage 恢复），
可直接在真机 progress GUI 的“下一个 Episode 起始子任务”中选择。GUI 会绑定所选 coarse
step 的第一个 fine stage，写入 `NOMINAL + ACTUAL_TRUNCATED_HISTORY`，因此后标注页面会从
该 fine stage 开始构造 occurrence，不会要求为未采集的前置子任务补标。fine-stage recovery
仍必须使用上面的显式三参数入口，GUI 不会猜测失败恢复语义。

## 7. 无真机 mock 流程（仅完整源码仓库）

完整 DexTeleop 源码仓库提供一个仅用于离线界面验证的 probe。它复用真实采集侧的
`RealTeleopProgressEpisode` 来生成 capture-causal Progress facts，因此刻意不进入不含
`teleop_runtime` 的 portable refinement-annotation projection。它不连接硬件，也不是生产采集入口：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
cd /home/zc/HandMotion/Developing/DexTeleop

python -m scripts.generate_refinement_annotation_mock \
  --output datasets/mock_refinement_annotation_test \
  --background-root utils/random_background \
  --seed 2502
```

输出路径必须是尚不存在的新目录。工具使用 managed 三阶段测试 profile，生成四种
Episode：

1. 完整成功；
2. 操作阶段失败、约 20 秒 wall-time Hold/Pause 后 Retry 成功；暂停期间不生成正式
   sample，用于验证 GUI 的最长 1 秒显示压缩、边界吸附和视频帧对齐；
3. 在中间阶段截断；
4. 从 `RECOVERY_FROM_FAILURE` 的中间 fine stage 进入。

每个 Episode 有 front/left_wrist/right_wrist 三个 15 Hz mock 视频和随机化阶段草稿；
前两条还各封存一版标注，用来测试 badge、coverage 和 failure exclusion。每个视频
都会检查精确解码帧数、全片解码 warning 和尾帧解码，任何不一致都会中止且不会发布
最终数据集目录。

生成后直接运行：

```bash
python -m control_station refinement-annotation
```

在 GUI 中选择 mock 数据集绝对路径并填写任意测试标注员 ID。mock 目录带有
`MOCK_DATASET_README.md`，明确禁止将它用于策略训练。
