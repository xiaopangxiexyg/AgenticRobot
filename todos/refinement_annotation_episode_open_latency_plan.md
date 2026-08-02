# 后标注 Episode 打开耗时优化方案

日期：2026-08-02

## 1. 目标与边界

目标是降低“点击 Episode 到时间轴可用”和“点击 Episode 到视频首帧可见”的等待，重点处理首次打开、切换 Episode、切换视角和服务重启后的冷缓存场景。

本方案只针对后标注 Control Station，不改变：

- 原始 Parquet、原始视频和既有时间轴/阶段/failure_trigger 协议；
- 采集端的视频编码与采集流程。采集端当前正在并行修改，不能把本优化强绑定到采集端；
- Episode 或 annotation 的事实所有权。所有缓存均为可丢弃的派生缓存；
- 当前配置语义，也不引入配置 hash 作为校验或缓存命中条件。

## 2. 当前链路与主要怀疑点

### 2.1 打开 Episode 的调用顺序

当前前端 `selectEpisode()` 先等待 `/api/episode/<uuid>` 返回，随后才执行 `renderEditor()`，而 `renderEditor()` 里的 `renderCameraTabs()` 再触发视频加载。也就是说，详情 API 和视频首帧没有并行启动。

服务端 `episode_detail()` 当前会同步完成：

1. 读取/校验封存 annotation sets；
2. 从 Parquet 读取完整 `timestamp` 列并序列化为 JSON；
3. 读取或生成 draft；
4. 组装完整 progress/stage/attempt/transition/failure_trigger 视图。

当前 Episode 数量较少、帧数较短时，Parquet 通常不是主要瓶颈；长程 Episode 或 annotation set 较多时，完整 sample axis 和封存集合读取仍可能放大 API 延迟。

### 2.2 视频冷启动路径

`/media/<episode>/<camera>` 首次请求会进入 `BrowserMediaCache.resolve()`：

1. 对源视频执行 `ffprobe`；
2. 对采集端常见的 `mp4v/mpeg4` 视频同步执行完整 `ffmpeg libx264` 转码；
3. 对临时输出再次 `ffprobe`；
4. 原子移动到进程级临时缓存后才向浏览器返回数据。

因此浏览器的第一个 Range 请求会被整个转码耗时阻塞。当前缓存目录在服务进程退出时删除，重启服务后全部重新检查/转码。源视频若位于共享盘，首次转码还会叠加共享盘读取延迟。

### 2.3 现有预热的局限

当前前端已有 `scheduleNextEpisodeWarmup()`，但它满足以下条件后才运行：

- 当前 Episode 的视频已经触发 `loadedmetadata`；
- 延迟 250 ms；
- 只为下一个多阶段 Episode 的默认视角创建隐藏 video；
- 单阶段 Episode 被刻意 lazy load，这一行为应保留。

这能改善“顺序浏览下一个 Episode”，但不能改善第一个 Episode 的冷启动，而且预热依赖隐藏浏览器请求，服务端仍可能在 HTTP 请求线程中同步转码。

## 3. 推荐方案

### P0：先增加耗时分段观测，不改变行为

必须先把耗时拆开，避免把 API、共享盘读取、ffprobe、ffmpeg 和浏览器解码混成一个“打开慢”。建议记录以下指标：

- 前端：点击时间、详情响应、编辑器可见、视频请求开始、首字节、`loadedmetadata`、`canplay`、首帧可见；
- 服务端：`/api/episode` 总耗时、封存 annotation 读取、sample axis 读取、draft 读取、媒体 resolve、源 probe、转码、输出 probe、缓存命中/未命中；
- 媒体维度：Episode、camera、源大小、帧数、源 codec、缓存路径、是否来自共享盘；
- 资源维度：ffmpeg CPU 时间、并发转码数、共享盘读吞吐、缓存大小。

观测数据应只进入日志或临时性能报告，不写入数据集，不改变已有数据合同。先用当前真实的 mpeg4 MP4、H.264 MP4、三视角和单阶段 Episode 做 cold/warm 两轮基线。

### P1：将兼容视频缓存改为本机持久派生缓存

优先改 `BrowserMediaCache`，但不修改原始视频：

- 默认使用标注机器本地 SSD 的用户缓存目录；继续允许显式指定缓存根目录；
- 复用现有源文件身份（路径、设备/inode、大小、mtime）作为缓存失效依据，不使用配置 hash；
- 使用临时文件、完整解码/探测校验和原子替换，失败时删除半成品；
- 进程重启后复用已经完成的 H.264 MP4，不再重复转码；
- 增加有界 LRU/大小上限和过期清理，避免长时间标注积累无限缓存；
- 同一个源视频仍由单一 source lock 负责，确保并发请求只产生一次转码；
- H.264、MP4、`yuv420p` 源继续直通，只缓存需要兼容转换的视频。

缓存是可重建的 UI 派生物，不是数据集 truth，也不能阻塞或改变 annotation commit。

### P2：增加服务端后台预热，避免首次媒体请求承担全部转码

在服务内增加一个有界的、低优先级的预热队列，建议最多一个后台 ffmpeg 转码任务，避免预热抢占标注机器 CPU：

1. Episode 列表加载完成后，预热当前列表第一条或当前选中 Episode 的默认视角；
2. 当前 Episode 视频开始播放/确认可用后，预热下一个多阶段 Episode 的默认视角；
3. 用户切换视角时，将该视角提升为高优先级；
4. 单阶段 Episode 不自动预热，除非用户点击“加载视频”或切换到该视角；
5. 只取消尚未开始的低优先级任务；已经开始的 ffmpeg 让其完成并原子发布，避免留下半成品或反复重启转码。

预热可以由服务端直接调用同一个 `BrowserMediaCache` owner，也可以增加只读的 media-ready/prewarm 内部接口；不建议继续仅依赖隐藏 `<video preload="metadata">` 来驱动转码，因为浏览器取消请求会制造无意义的 socket 中断和排队噪声。

首次打开时的兜底路径仍保留：若高优先级视频尚未完成，显示“正在准备视频”的明确状态，当前时间轴和 Episode 结构可以先显示；缓存完成后再挂载 video source。这样 API 慢和视频慢不会互相阻塞，也不会把未加载视频误当成数据损坏。

### P3：把 Episode 详情和视频加载解耦

前端建议调整为两个阶段：

**阶段 A：快速显示结构**

- `/api/episode` 返回必要的 Episode 摘要、时间轴结构、draft 和 media readiness；
- 先渲染时间轴、阶段、failure_trigger 和加载占位；
- 立即发起默认视角的 media-ready/加载请求。

**阶段 B：媒体就绪**

- media-ready 返回缓存命中、排队、转码中、可播放或失败；
- 可播放后再设置 `<video src>`，避免多个旧请求同时竞争；
- seek、切换视角、切换 Episode 时沿用已经修复的 client-disconnect 静默处理。

如果 P0 证明 sample axis JSON 是主要 API 瓶颈，再增加按需 sample-axis 接口：详情首包只返回首尾/采样统计，时间轴需要精确拖动时再取完整离散 sample axis。可优先使用紧凑二进制或压缩传输；这只是 UI 传输优化，不修改 Parquet 和时间轴语义。

### P4：增加进程内只读缓存，减少重复详情读取

在 `RefinementAnnotationApplication` 增加有界缓存：

- `EpisodeSampleAxis` 缓存：按 Parquet 文件身份和 frame_count 失效；
- 封存 annotation set 缓存：按 Episode annotation 目录的最新 mtime/目录指纹失效；
- immutable Episode detail base 缓存：只缓存 profile、timeline、stage 和 media metadata；draft 每次仍从 mutable workspace 读取并规范化；
- `episodes.jsonl` 列表刷新使用文件 mtime/大小或追加 offset，避免每 4 秒重复解析整个目录。

这些缓存必须是 bounded、可重建、不可反向修改事实的 UI 读缓存；不能将缓存提升为训练或 annotation 的权威来源。

### P5：采集端编码优化作为后续独立评估

不把采集端改成 H.264 放入本次落地。采集端正在并行修改，且编码器改变会影响 CPU、磁盘吞吐、文件大小、尾帧完整性和现有工具兼容性。

后续若 P0 证明转码是长期主要成本，再单独做采集端实验：保留原始/兼容输出边界，比较 `mp4v` 与 H.264 的实时采集负载、丢帧、最终帧数、尾帧解码、文件大小和后标注首帧延迟；通过独立验收后再决定是否默认启用。

## 4. 推荐落地顺序

1. P0 性能观测和 cold/warm 基线；
2. P1 本机持久媒体缓存，保持当前媒体协议不变；
3. P2 有界后台预热当前默认视角和下一个多阶段 Episode；
4. P3 UI 先显示结构、媒体异步就绪；
5. P4 sample axis/annotation/detail 只读缓存；
6. 根据数据决定是否开展 P5 采集端编码评估。

第一批建议只落地 P0～P2。它们能直接处理最可能的冷转码瓶颈，同时不改变数据合同和采集端。P3～P4 在基线证明 API/JSON 仍是主要瓶颈时再做，避免为了理论上的长 Episode 优化而过早改变 GUI 协议。

## 5. 验收标准

### 功能与语义

- 原始视频、Parquet、进度、阶段、failure_trigger 和 annotation set 字节内容不改变；
- H.264 源直通，mp4v/mpeg4 源可得到可解码的 H.264 MP4；
- 缓存中断、服务重启、源文件替换、ffmpeg 失败后都能安全重建；
- 单阶段 Episode 仍默认 lazy load；多阶段顺序浏览能预热下一个默认视角；
- 时间轴仍按 Parquet 离散 sample axis 对齐，seek 和 failure_trigger 吸附语义不改变；
- 浏览器取消旧请求不产生 `BrokenPipeError` 堆栈，也不影响后续媒体请求。

### 性能

建立基线后至少比较：

- 冷缓存首次打开 vs 热缓存再次打开；
- 首次 Episode vs 顺序打开下一个 Episode；
- 默认视角 vs wrist 视角；
- H.264 vs mpeg4；
- 小 Episode vs 长 Episode；
- 本地 SSD 数据 vs 共享盘数据。

目标应同时报告 `T_detail_visible`、`T_media_ready`、`T_loadedmetadata` 和 `T_first_frame`，而不是只看整个页面请求耗时。P1/P2 的最低验收是：热缓存打开不再触发 ffmpeg，顺序打开下一个多阶段 Episode 的默认视角命中预热缓存，且缓存 miss 不阻塞其他 API 请求。

### 测试

- `BrowserMediaCache`：冷 miss、热 hit、跨进程持久 hit、源 mtime/大小变化失效、并发同源只转码一次、半成品清理、LRU 清理；
- HTTP media：Range、HEAD、客户端中途断开、转码失败和后续请求恢复；
- GUI：单阶段不自动加载、多阶段预热、切换 Episode/视角、服务重启后的缓存命中；
- 数据完整性：原始视频字节不变，H.264 输出完整解码且尾帧可读；
- 性能 probe：记录上述分段耗时和缓存命中率，作为后续回归阈值。

## 6. 预期结果

短期通过持久缓存和后台预热，消除“第一次点击必然同步转码”的等待；中期通过详情/媒体解耦，让时间轴先可见、视频异步就绪；只有在实测证明 Parquet/JSON 或采集编码仍是主要瓶颈时，才继续推进 API 传输优化或采集端编码调整。
