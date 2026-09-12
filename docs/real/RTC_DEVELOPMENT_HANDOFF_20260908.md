# RTC 开发与部署工作交接说明

> 历史快照：本文中的分支、SHA、worktree、状态和命令只描述 2026-09-08
> 的交接证据，不是当前开发或部署入口。当前入口见 `docs/real/README.md`、
> `docs/real/DEPLOY.md` 与相应 package README。

> 文档用途：把本轮 RTC 训练、部署、验收、调试和协作边界固化为一份可审阅的交接材料。
> 本文是开发交接文档，不是 Harness Task、集成批准、远端执行授权或真机动作授权。
>
> 当前工作模式：独立 Git 开发 worktree。Harness、远端训练、OpenPI server、相机、夹爪、
> 机械臂和真实动作均不在本轮执行范围内。

## 0. 先读结论

本轮工作的首要目标不是给某一个 checkpoint 打临时补丁，而是建立一条可以审计、可以切换、
可以 fail-closed 的 Model-RTC 训练到部署链路：

~~~text
数据选择/训练 profile
    -> checkpoint 与 artifact
    -> OpenPI server/client binding
    -> observation 与 RTC transport
    -> completion/latency gate
    -> policy emission/plan admission
    -> qpos-init、ready、safety 和 authority
    -> （只有显式授权时）真机 dispatch
~~~

当前交接候选的核心状态：

| 项目 | 状态 |
| --- | --- |
| 开发分支 | task/codex/rtc-strict-latency-gate-20260907 |
| 当前独立 worktree | /home/zc/HandMotion/Developing/temp/worktrees/codex-rtc-strict-latency-gate-20260907 |
| 候选基线 | c35e903178805cc05c64f7624e43852b64a34a3a |
| 代码候选（文档提交前） | bc041c0d1b05cca98150d376d43e93e1ace309de |
| 主线 development（共享 ref，非本轮修改） | 3fa54d58b2b21aa0bcdb53c4f76afb2288003c55（此前只读快照为 4743d20…；接手前必须重查） |
| worktree 状态 | 文档开始前干净且 ready-for-integration；新增本文档期间为 active/dirty，验证后重新固定最终 tip |
| Harness | 不调用、不修改、不激活、不 promotion、不集成 |
| 远端/训练/服务/真机 | 本轮均未执行 |

文档提交会改变候选 tip；共享仓库的 development ref 也可能由其他工作区移动。因此，接收者
必须以交接时重新读取的完整 Git SHA、目标线 SHA 和分支 description 为准，不能继续使用上表
“文档提交前”的 SHA 或旧 development 快照作为最终集成输入。

## 1. 用户要求与任务边界

### 1.1 已确认的工作方向

交接过程中用户明确要求继续 RTC 开发，并反复确认以下边界：

1. 先理解 Harness Coding Tool 的核心要求和现有代码，再动手开发。
2. 当前只担任开发，不担任 integration；不合并、不 promotion、不推送、不向远端同步。
3. 不动 Harness，不调用 Harness 控制面，不借 Harness 状态制造集成权限。
4. 远端主机身份曾明确为 user@192.168.70.116，但本轮没有远端连接授权或远端执行。
5. 真机恢复可达不等于动作授权；真实动作必须由操作者在独立安全检查后显式验收。
6. 在完成所有不下发真机动作的测试后，才向用户呈现真实动作命令；自动化开发过程不能执行该命令。
7. 当前 agent 只担任开发；既有 rtc_development_integrator/Raman 不属于本轮执行路径，不调用、不委派 integration。

### 1.2 历史训练停止要求

此前曾要求训练运行到 Step 25,000 后终止，并建议使用 2 小时 watchdog。该要求适用于
得到明确远端执行授权的训练操作，执行时必须同时记录：

- 训练命令、代码和 profile 的 immutable identity；
- 进程启动时间、watchdog 到期时间和实际停止原因；
- 数字 checkpoint 是否完整、是否可读取；
- 停止后 GPU、磁盘和残留进程状态。

本次开发交接不重新启动训练，也不对远端当前进程状态作新的推断。

### 1.3 “不下发动作”的准确含义

必须区分三种检查：

| 检查 | 是否连接 server | 是否可能打开相机/状态 backend | 是否允许 policy action dispatch |
| --- | ---: | ---: | ---: |
| 静态 contract/config check | 否 | 否 | 否 |
| no-action shadow | 是 | 是 | 否；不含 --send-actions |
| Active 真机 | 是 | 是 | 只有显式 --send-actions 且所有安全门通过 |

因此，no-action 不是“无硬件检查”。它可以读取真实 observation、RM65 状态、DMTAC 和
RealSense，但不得下发 policy 动作。静态检查才是不连接设备的检查。

## 2. 首要核心目标

### 2.1 系统性 RTC 目标

RTC 系统需要同时解决训练、artifact、transport、runtime、authority 和安全边界，而不是只
验证某一个模型能够启动：

- client 不应无必要地绑定某个 checkpoint step；
- server 不应无必要地限制在某一个训练 run；
- continuation contract hash 不应被当作跨实例身份的强一致门槛，除非它确实属于运行时能力契约；
- 同一 client 在实际能力契约兼容时，应能切换不同 step、不同兼容模型 server；
- 真正必须保持的契约包括协议/能力版本、observation/action schema、action dimension、
  action horizon、camera/view key、RTC prefix/delay 能力、absolute action 语义、authority
  和安全 dispatch 边界。

### 2.2 训练目标

系统性修复后的 Model-RTC 训练路线应明确记录：

- Stage1 使用所有当前有效数据；
- 视角为 front + left_wrist + right_wrist；
- h32、absolute；
- 主流/default Model-RTC 配置；
- final keep-mask；不得回退到 raw wrist depth mask；
- 低内存配置，batch size=1，dataloader workers=0；
- 每阶段目标 100000 step，保存/保留间隔 25000；
- checkpoint 只写入远端本地 /home/user/zc/checkpoints/openpi；
- 新 Stage1 完成并验证后，再继续新的 Stage2、Stage3 等阶段；旧 Stage2 不能冒充系统性
  修复后的完整训练序列。

### 2.3 部署目标

最终部署应覆盖：

- 普通 OpenPI/non-RTC；
- raw RTC；
- corrected RTC；
- Model-RTC/conditioned RTC；
- Stage1–Stage4 以及不同 checkpoint step 的切换；
- no-action shadow、Active qpos-init、HOLD、emission、plan admission 和真机 dispatch；
- qpos-init、ready、碰撞、力限、观测新鲜度和 authority 门的独立证据。

详细命令仍以 OPENPI.md、DEPLOY.md、deployments/openpi/COMMANDS.md 和
deployments/openpi/cmd_list.md 为准；本文负责解释决策和边界，不复制所有长命令。

## 3. 当前实现要点

### 3.1 Active latency 门

Active RTC warmup 使用如下固定语义：

- 丢弃第 1 个 cold completion；
- 接受随后的 3 个 warm completion；
- 每个 admitted completion 的端到端 latency 必须严格满足 < 0.4 s；
- 等于 0.400000 s 也必须拒绝；
- bounded acceptance 还要求至少 1 个 committed RTC emission；
- Active --max-steps 只按已接受的 RTC dispatch 计数，不把 qpos-init 或等待 ready 的循环
  误算成 policy dispatch。

门逻辑位于 teleop_runtime/deploy/run_policy.py 的 _rtc_active_latency_gate_result。

### 3.2 时间戳 fail-closed 修复

此前的风险是：

~~~python
max(0.0, released_at - inference_started_at)
~~~

当时间戳倒序时，会把错误样本变成 0.0，从而可能通过严格 latency 门。

当前实现通过 _rtc_active_completion_latency_sec 要求：

- 两个时间戳都必须是 finite；
- released_at >= inference_started_at；
- 正常值返回精确差值；
- 非法值直接拒绝，不再钳制成零。

Active 主循环遇到非法 completion timing 时会：

1. 将 RTC frontend 置为 HOLD（若当前仍 eligible）；
2. 清空 position plan；
3. 请求 measured-qpos Hold；
4. 即使 frontend HOLD 本身抛错，也在 finally 中继续尝试硬件侧 Hold；
5. 抛出 RuntimeError 并退出当前 Active 路径。

实现位置为 teleop_runtime/deploy/run_policy.py 中的 helper 和 Active warmup 分支。

### 3.3 qpos-init、HOLD 和 authority

- RTC profile 不自动授予动作权限；权限仍由 --send-actions 独立控制。
- Active 首次运行必须经过 dataset-frame qpos-init、碰撞检查、ready 确认和 latency gate。
- Stage1 通常使用绑定数据集的 qpos-init。
- Stage2–4 只有在当前姿态确认为连续衔接状态时才可使用 --no-init-qpos。
- --no-init-qpos 不继承上一个 client 的 queue、model history、chunk 或 authority。
- producer 不 eligible、plan underrun、延迟超限或 backend Hold 能力缺失时，必须 fail closed。
- Hold 失败时要升级到更强 Hold 并报错，不能悄悄继续 dispatch。

### 3.4 模型和 checkpoint 解耦原则

连接时应校验实际 wire/runtime 能力，而不是伪造或强制 model/step 身份：

- layout；
- camera routing/view key；
- action schema/dimension/horizon；
- normalization；
- prompt/text binding；
- RTC prefix/delay/continuation capability；
- absolute/relative action 语义；
- artifact/profile 的来源收据。

不能通过手写 model id、checkpoint step、layout hash、camera routing 或 action pose mode
绕过不兼容错误。prepared policy 不能混用普通 --config 或不匹配的 artifact。

## 4. 训练与 checkpoint 状态

本节分为“历史交接事实”和“当前需要重新核验的站点事实”。路径、step、hash、进程和磁盘
属于会漂移的外部状态；任何新部署前必须重新检查，不能把本文快照当作 registry。

### 4.1 历史旧 Stage2 RTC checkpoint

历史交接记录确认旧 Stage2 RTC 已完成 100000 step，并存在完整 checkpoint：

~~~text
/home/user/zc/checkpoints/openpi/
  writing-stage2-all-valid-all-view-h32-absolute-prepared-model-rtc-v1/
  writing-stage2-all-valid-all-view-h32-absolute-model-rtc-b1-nw0-lowmem-semanticbounds-v5-20260904/
  100000/
~~~

其中包含 _CHECKPOINT_METADATA、assets/、params/、params/_METADATA、params/_sharding、
params/manifest.ocdbt、policy_artifact.json、training_step/、
model_continuation_checkpoint_receipt.json 和 model_continuation_training_run.json，历史记录中的
大小约为 1.7G。

历史指标约为：final loss 0.0145、grad_norm 0.2226、param_norm 1810.455，恢复后运行
时间约 19:57:43。

该 checkpoint 可以作为：

- 兼容性和部署回归对象；
- old Stage2 对照对象；
- artifact、transport、no-action 和协议检查对象。

它不能被表述为“系统性修复后从新 Stage1 重新开始得到的最终 Stage2”。

历史部署配方还记录了 corrected Stage1 v2 的最新完整 checkpoint 为 25006；Stage2 的 100000
checkpoint 仍是 historical v1 对照对象。两者的 profile、artifact、runtime gitlink 和实际
数字目录必须分别核验，不能因 step 数字存在就交叉绑定。

### 4.2 新 Stage1 准备快照

历史准备根目录：

~~~text
/home/user/zc/openpi_stage1_model_rtc_runs/20260905_systemic_v2
~~~

历史隔离代码身份：

~~~text
AgenticRobot: 1ff4f5237145e5497ee83303cb5189fa77a21921
OpenPI:      a1bc3542f41c48381113bb8c61ba9fe2f62fa2f8
~~~

历史 profile 与 hash：

~~~text
profile:
/home/user/zc/openpi_stage1_model_rtc_runs/20260905_systemic_v2/profiles/writing_stage1_all_valid_all_view_h32_absolute_non_rtc_100k_systemic_v2.json
sha256: 28cc47488cd35063e2ff154ed76a2e9ad0818c6b525e85b606107b03395e9ba7

model-RTC overlay:
/home/user/zc/openpi_stage1_model_rtc_runs/20260905_systemic_v2/code/AgenticRobot/deployments/openpi/configs/model_rtc/writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json
sha256: 2ca8f26bb41234ebe63edc7c5b8dfecfa9cfbe739095ef86527b9d5e8e98cf2c
~~~

历史数据选择计数：

- sources：13；raw episodes：3587；eligible episodes：3380；excluded：207；
- complete h32 windows：215422；quality-valid windows：108754；sampler unique samples：108754；
- 排除原因包括 NO_SEALED_STAGE_ANNOTATION、WHOLE_EPISODE_ABNORMAL、
  WHOLE_EPISODE_REVIEW_PENDING、TARGET_SKILL_NOT_SUCCESSFUL 和 LOCAL_ABNORMAL_INTERVAL；
- 训练视角：front、left_wrist、right_wrist；动作：absolute。

这些是历史准备证据，不是本轮已重新执行的远端训练证据。接手者必须重新检查源代码、profile、
manifest、checkpoint 和远端剩余空间。

### 4.3 远端资源风险

历史交接记录显示远端系统盘曾约 99% 使用率、只剩约 35 GB。正式训练或复制 checkpoint
前必须先做只读容量审计，并确认：

- checkpoint root 所在分区和 inode 足够；
- 没有残留训练进程或旧 server 占用 GPU/磁盘；
- 不会通过删除未知目录制造空间；
- 训练输出与临时缓存位于明确的授权路径；
- 训练停止后保留 metadata、receipt、日志和完整数字 step。

### 4.4 Stage1 smoke 的历史 blocker

历史新 Stage1 smoke 在 Step 0 之前因 OpenPI 基础 checkpoint 参数初始化缺陷失败，缺失字段为
rtc_prefix_mask_embedding。该 smoke 目录不能直接 resume。历史记录中的最小 loader 修复已
提交为 359f1ced，但当时尚未集成到 OpenPI development，也未同步远端。接手者必须重新确认
该修复是否已经进入目标 OpenPI checkout，再决定是否重新做 smoke；不能把失败 smoke 当作训练
失败后的可恢复 checkpoint。

## 5. 部署路线与阶段切换

### 5.1 路线矩阵

| 路线 | 主要 contract | no-action | Active 前的额外门 |
| --- | --- | --- | --- |
| 普通 OpenPI/non-RTC | artifact、layout、camera、action、normalization | 必须 | qpos-init/ready/safety/authority |
| raw RTC | RTC final-chunk profile、transport、timed plan | 必须 | completion、emission、latency、HOLD |
| corrected RTC | 修正后的 profile、checkpoint 和 runtime identity | 必须 | 同上；不能混用旧 profile |
| Model-RTC | conditioned profile、prefix/delay、continuation binding | 必须 | qpos-init、strict <0.4 s、emission、ready |
| prepared Stage1 | stage1 training profile + checkpoint | 必须 | dataset-frame qpos-init |
| prepared Stage2–4 | 对应 stage profile + checkpoint | 必须 | 当前姿态连续衔接、--no-init-qpos、ready |

不同路线不能拼接参数。每次切换 checkpoint 或 stage 都应停止旧 server/client，重新做 artifact、
profile、metadata 和 no-action 检查；不能复用上一进程的 queue、history、authority 或旧 hash。

### 5.2 Stage1–4 的启动姿态

- Stage1：通常执行绑定数据集的 qpos-init，并完成 collision check。
- Stage2、Stage3、Stage4：只有在现场确认当前姿态就是正确连续接管点时，才使用
  --no-init-qpos。
- --no-init-qpos 不是“跳过安全检查”，仍须执行 ready、碰撞、力限、观测新鲜度和
  dispatch 门。
- Stage 切换时要在 handoff receipt 中写清楚：checkpoint path、step、profile hash、
  server identity、client identity、起始姿态和是否 qpos-init。

## 6. 协作、所有权和 Harness 规则

### 6.1 当前采用的管理方式

本任务选择“独立 Git 开发”，不是 Harness-managed Task。共享 worktree 规则要求：

1. 先 inventory，再 declare/overlap，再 pin；
2. 写入前确认 owner、branch、base、tip、dirty 状态和 write scope；
3. peer worktree 一律只读，不 reset、clean、删除、同步或覆盖；
4. 新提交会使旧 ready-for-integration 失效，必须重新验证并重新固定 tip；
5. inventory 只是事实快照，不是锁，也不是语义兼容证明；
6. 集成必须由明确授权的集成者在独立 integration worktree 完成；
7. 目标线和输入 SHA 变化后，旧验证不能直接复用。

当前候选的声明范围是：

~~~text
owner: codex-rtc-developer
base: c35e903178805cc05c64f7624e43852b64a34a3a
semantic scope: strict fail-closed RTC Active latency + complete RTC handoff documentation
excludes: Harness, integration, promotion, remote evidence, training, deployment services, hardware actions
~~~

### 6.2 Harness 明确边界

本轮不能：

- 调用 harness 命令创建/关闭/恢复/集成 Task；
- 修改 HarnessCodingTool、HarnessControl、HarnessState、HarnessConfig 或其 receipt；
- 伪造 lease、claim、runner、actor、promotion receipt 或 protected control-plane identity；
- 把普通 Git branch description 当作 Harness 授权；
- 把历史 Harness integration receipt 当作当前代码或远端部署已经生效的证明。

如果未来切换回 Harness，必须先做正式 handoff，由新 Task contract、scope、lease 和 evidence
接管；不能在同一任务上叠加两套所有权记录。

### 6.3 远端和设备边界

- 未得到本轮明确授权，不连接 user@192.168.70.116，不启动远端训练或 server。
- 远端操作只能通过受支持的 codex_remote wrapper，并保留命令、身份、输出根和结果 receipt。
- 不使用临时 SSH、任意 scp/rsync 或未经审计的远端 shell 替代 wrapper。
- 不启动相机、DMTAC、夹爪、RM65、servo、force-control、homing、reset、replay 或动作。
- 真实动作命令只交给用户现场验收，不由开发 agent 执行。

## 7. 开发和调试流程

### 7.1 开发前

~~~bash
git worktree list --porcelain
git status --short --branch
git rev-parse HEAD
git rev-parse development
~~~

然后运行指定分支 inventory，核对 owner、base、tip、scope、dirty entries 和语义重叠。需要
写入时先把 branch description 从 ready-for-integration 改回 active。

### 7.2 实现时

- 先定位真正的 authority owner；不要在 observer、UI cache 或日志层重复重建状态。
- 只改当前声明的 scope；发现需要跨 scope 或跨 worktree 时先暂停并重新评估所有权。
- 对每一个 failure mode 增加边界测试，不用大面积 formatter 制造无关 diff。
- 任何可能影响 Active 的异常都要先 HOLD/fail closed，再抛错；不能只依赖最终 close。
- 不以“测试进程退出码为 0”推断动作成功；必须检查 emission、authority、dispatch 和 safety receipt。

### 7.3 推荐验证层次

1. 纯 helper/unit 边界：finite、倒序、相等、阈值边界、空输入和配置错误。
2. focused RTC shadow/Active 测试：验证 no-action、qpos-init、HOLD、emission 和 authority。
3. prepared deployment 组合：验证普通、Model-RTC、Stage1/2 绑定和 artifact receipt。
4. 全 RTC 回归：记录 socket、GUI、设备或环境限制，不把 skip 当作 pass。
5. 最终交接前重新 inventory、检查实际 HEAD、检查工作树和 diff。

### 7.4 当前候选使用过的命令与结果

支持环境由 scripts/run_pytest.sh 负责。当前候选已验证：

~~~bash
bash scripts/run_pytest.sh -q \
  tests/test_rtc_real_shadow_no_action.py \
  tests/test_model_rtc_real_active_qpos.py
# 41 passed

bash scripts/run_pytest.sh -q \
  tests/test_openpi_prepared_deployment.py \
  tests/test_openpi_model_rtc_artifact_binding.py \
  tests/test_openpi_stage12_all_view_model_rtc_deployment.py \
  tests/test_rtc_real_shadow_no_action.py \
  tests/test_model_rtc_real_active_qpos.py
# 98 passed

bash scripts/run_pytest.sh -q 'tests/*rtc*.py'
# 477 passed, 2 skipped；一个 AF_INET loopback 节点受沙箱限制

/home/zc/anaconda3/envs/dexteleop310/bin/ruff check \
  --select E4,E7,E9,F821 \
  teleop_runtime/deploy/run_policy.py \
  tests/test_rtc_real_shadow_no_action.py
# All checks passed

git diff --check
# passed
~~~

唯一受限 RTC 节点为：

~~~text
tests/test_deploy_rtc_frontend_shadow.py::test_cs3_loopback_api_only_coordinates_passive_shadow_lifecycle
~~~

它在沙箱内因 PermissionError: [Errno 1] Operation not permitted 无法创建 AF_INET socket；
同一固定测试在 socket-capable 本机上下文通过 1 passed。这属于环境证据，不得改测试绕过。

全文件 Ruff 仍有原有导入、mutable class default 和 SIM 告警；本轮只运行并记录受影响的
语法/未定义名规则，不用全文件自动格式化覆盖历史代码。

## 8. 现场部署和验收阶梯

### 阶梯 A：完全离线

1. 核对 checkpoint 目录真实存在数字 step、params、policy_artifact.json、norm stats。
2. 核对 prepared training profile、model-RTC overlay、artifact hash 和 runtime gitlink。
3. 执行 --contract-check-only。
4. 执行 server --dry-run 或 --prepared-config-check-only。
5. 检查 schema、camera slots、action dimension/horizon、normalization、prompt 和 RTC capability。

这些步骤不连接 server、相机、夹爪或机械臂。

### 阶梯 B：server/transport no-action

1. 启动与 checkpoint/profile 完全匹配的 server。
2. 只读检查 /healthz 和 /metadata。
3. client 不含 --send-actions，运行 bounded no-action。
4. 至少观察一次 committed RTC emission；零 emission 不能判定为通过。
5. 保存 client/server 命令、metadata、artifact/profile hash、日志目录和退出原因。

no-action 期间可以打开真实 observation/backend，但不得 dispatch policy action。

### 阶梯 C：Active 前置

只有在 A、B 证据完整后，才允许进入 Active preflight：

- 机械臂和相机状态由现场操作者确认；
- Stage1 完成 dataset-frame qpos-init；Stage2–4 明确确认连续姿态；
- collision、force guard、camera freshness、backend readiness 通过；
- 操作者完成 ready 确认；
- RTC warmup 完成：discard=1、warm=3、每次 latency <0.4 s；
- 至少一个 committed emission；
- producer eligible、authority 合法、没有 plan underrun；
- --max-steps 和退出条件明确。

### 阶梯 D：用户现场真机验收

开发 agent 不执行真实动作。交给用户的命令必须：

- 来自已经通过 A–C 的同一 server/client/profile/checkpoint 组合；
- 只额外加入明确的 --send-actions；
- 不加入 --auto-start、--start-init_qpos 或跳过 ready 的参数；
- 明确写出回退、Hold、Ctrl+C、急停和日志保存方式；
- 由用户在机械臂可达、无人员风险、现场安全门确认后执行。

真机命令的具体长版本继续维护在 deployments/openpi/cmd_list.md，本文不复制一条容易被
误粘贴执行的动作命令。

## 9. 调试手册

### 9.1 Active latency 超时或非法时间戳

先保留：completion receipt、inference start/release、worker session、clock domain、
producer state 和 exact command。不要放宽 --rtc-active-max-inference-latency-sec 代替修复。

- released_at < inference_started_at：判定为时钟/receipt 契约错误，必须 HOLD 并退出。
- NaN/Inf：判定为无效测量，不能进入统计。
- 0.400000：边界拒绝；只有严格小于 0.4 才能通过。
- warm sample 不足：保持 HOLD，不把未完成 warmup 当作通过。
- rtc_producer_not_eligible 或 plan underrun：清空计划、保持 measured qpos、必要时升级 Hold。

### 9.2 零 emission

进程正常退出不代表 RTC 成功。检查：

- selected_emission_count；
- committed emission receipt；
- source binding hash；
- producer lifecycle；
- worker completion/admission receipt；
- --min-rtc-emissions-before-exit 是否生效。

### 9.3 schema/profile mismatch

不要直接改 hash 或手填 model id。重新确认：

- server/client 是否使用同一条 prepared/profile 路线；
- checkpoint step 是否真实存在；
- artifact、norm stats、camera key、action mode 是否一致；
- Model-RTC overlay 与 runtime gitlink 是否来自同一固定部署快照；
- server 的 OpenPI root 和 profile 固定 training checkout 是否被混用。

### 9.4 训练慢、停止和恢复

只读检查 GPU、CPU、dataloader workers、I/O、磁盘、日志和进程树。若按历史要求运行到
25,000 step：

- watchdog 设为 2 小时；
- 到 step 25,000 后正常停止并记录 receipt；
- 不从只有 wandb/、stats 或临时文件的目录 resume；
- 只有存在完整正整数 step checkpoint 时才允许 --resume；
- 恢复前重新绑定代码、profile、数据选择和 checkpoint root。

### 9.5 Socket/环境限制

PermissionError: [Errno 1] Operation not permitted 若来自 AF_UNIX/AF_INET 创建，是环境
证据，不是业务失败。保留原始失败，再在 socket-capable 上下文重跑同一 pinned command；不要
修改测试使其不创建 socket，也不要把未重跑的结果报告为通过。
## 10. 证据和呈现方式

所有汇报按以下四类分开：

| 证据类别 | 可证明内容 | 不能证明的内容 |
| --- | --- | --- |
| offline | 代码、schema、profile、artifact 和 unit/pytest | 远端可用、设备存在、动作安全 |
| simulation/local | 本机运行时、socket、模拟 backend、协议流程 | 真机动力学、现场姿态和人员安全 |
| remote | 远端路径、进程、GPU、checkpoint、server metadata | 本地候选已集成、真机动作安全 |
| real hardware | 现场 backend、qpos-init、ready、dispatch 和急停证据 | 其他 checkpoint/stage 自动兼容 |

汇报模板应包含：

~~~text
目标：
当前 worktree / branch / owner：
base SHA：
candidate SHA / tree SHA：
代码变更：
影响的 API/schema/authority：
验证命令：
验证环境：
通过/失败/skipped：
已知环境限制：
远端证据：未执行/具体 receipt
真机证据：未执行/具体现场记录
未验证范围：
下一步及所需授权：
~~~

不能使用“已测试”这种没有命令、环境和结果的笼统表述；不能把 communication record、旧
receipt、分支名字或历史 checkpoint 名字当成当前 authority。

## 11. 当前已知风险和未完成事项

1. 本候选仍未集成到 development；集成前必须重新检查 development tip、候选 tip、路径/API
   overlap 和当前 worktree dirty 状态。
2. 文中远端 checkpoint、磁盘、GPU、训练进程和 server 状态均需在新一轮远端操作前刷新。
3. 历史旧 Stage2 checkpoint 不是系统性修复后新 Stage1→Stage2 序列的最终证明。
4. 新 Stage1 loader 修复、训练重跑和后续阶段推进需要独立记录代码 identity、profile hash、
   数据计数和 checkpoint receipt。
5. 当前 AF_INET 节点只在 socket-capable 本机验证过；普通沙箱中仍会失败，不能删除该环境注记。
6. 全文件 lint 的历史告警没有在本轮扩大修复；若后续要清理，必须另开明确 scope，不能混进
   RTC latency 候选。
7. 真机动作、远端训练和 OpenPI service 仍没有本轮 evidence。

## 12. 接手者操作清单

### 接手开发

- [ ] 阅读本文件、AGENTS.md、docs/development/git-workflow.md、OPENPI.md 和 cmd_list.md。
- [ ] 重新运行 git worktree list --porcelain 和指定分支 inventory。
- [ ] 确认 branch owner、base、tip、scope、dirty 状态和 peer overlap。
- [ ] 如果继续写代码，把 branch state 改为 active。
- [ ] 只在当前独立 worktree 修改，不碰 Harness 和 peer worktree。

### 接手集成（需要额外授权）

- [ ] 获取明确的 integration 授权和唯一集成者身份。
- [ ] 重新 pin 当前候选完整 SHA 和 target development SHA。
- [ ] 在独立 integration worktree 合并并运行组合验收。
- [ ] 重新检查 artifact/profile/runtime gitlink、测试、diff 和目标线。
- [ ] 只有 verified integration tip 才能更新目标线；不能把 ready 候选直接宣称已生效。

### 接手远端训练/部署（需要额外授权）

- [ ] 只读审计远端主机、磁盘、GPU、进程、checkpoint 和环境。
- [ ] 绑定 immutable AgenticRobot/OpenPI/profile/runtime identities。
- [ ] 训练时按 25,000 step/2 小时 watchdog 规则记录停止证据（若该轮仍适用）。
- [ ] 先做 contract/config check，再 server metadata，再 no-action，再 Active preflight。
- [ ] 真机动作由用户现场验收，开发 agent 不代执行。

## 13. 相关入口

- 现场 OpenPI 顺序：[docs/real/OPENPI.md](./OPENPI.md)
- 通用部署边界：[docs/real/DEPLOY.md](./DEPLOY.md)
- OpenPI 详细命令：[deployments/openpi/cmd_list.md](../../deployments/openpi/cmd_list.md)
- OpenPI 统一命令入口：[deployments/openpi/COMMANDS.md](../../deployments/openpi/COMMANDS.md)
- 独立 Git 工作流：[docs/development/git-workflow.md](../../docs/development/git-workflow.md)
- AgenticRobot 项目约束：[AGENTS.md](../../AGENTS.md)
- RTC Active 实现：[teleop_runtime/deploy/run_policy.py](../../teleop_runtime/deploy/run_policy.py)
- RTC shadow/Active tests：[tests/test_rtc_real_shadow_no_action.py](../../tests/test_rtc_real_shadow_no_action.py)
