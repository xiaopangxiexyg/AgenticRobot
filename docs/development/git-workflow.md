# 独立 Git 开发与交接

这是 AgenticRobot 自身的轻量 Git 流程；其他仓库应维护自己的开发规则，不依赖
本仓脚本或资产。默认开发入口仍为 HarnessCodingTool 定义的 Harness。选择独立 Git 时说明一次即可，不需要
Harness 服务、Task、lease 或机器人资产。已受 Harness 管理的工作必须先完成
明确交接才能切换；不能用本流程绕过其冲突、写入、证据或集成门禁。

## 创建与所有权

先确认目标仓库、开发线和当前相关工作。默认盘点只精查当前和主线；其他声明为
active 的 branch 只显示 owner/state 和 scope 数量，同时保留全部 branch/detached 数量。
不要把大型共享仓库的原始 worktree 清单直接放进 Agent 上下文：

```bash
git status --short --branch
python scripts/worktree_inventory.py . --active --mainline development
```

如果输出存在未分类 branch worktree，或相关 owner/scope 仍不明确，再根据候选名称用重复
`--branch` 精确扩展；只有恢复、清理或仍无法归属时才做完整盘点或读取原始
`git worktree list --porcelain`。

本项目通常以 AgenticRobot `development` 为基线；其他开发线必须来自实际任务。
复用清楚归属的现有任务，或为独立范围建立新工作区。短路径、
不同基线和隔离集成是合理的新建原因；后续修复复用本任务归属明确的工作区，
不按对话轮次新建副本。保留分支不要求常驻 checkout。不要仅为命名统一重命名旧分支。

从目标 AgenticRobot checkout 运行认领工具：

```bash
python scripts/claim_worktree.py \
  --branch codex/<task-name> \
  --path /absolute/path/to/new-worktree \
  --base <development-line-or-commit> --owner <actual-owner> \
  --write-scope 'src/feature/**' --write-scope 'tests/test_feature.py' \
  --semantic-scope '<changed behavior or public contract>' \
  --validation '<planned focused checks>'
```

`task/` 旧命名仍可用。工具只负责创建和声明，不自动判断语义独立性，也不持有
排他锁。写入前确认没有其他所有者修改相同文件或公共契约；如果同仓库存在
Harness Task，检查其当前范围并与其协调者解决重叠。

资产不是认领前提。已配置且完整的只读资产可以自动链接；缺失资产会明确报告，
纯代码或文档任务仍可创建。`--asset-root` 或 `--require-assets` 表示本次要求完整
资产，验证失败则不创建分支。后续资产任务参考 `docs/setup/LOCAL_RESOURCES.md`。
兼容的外部 `--repo-root` 只创建 Git worktree 和分支描述，不读取或挂接本仓资产；
外部项目应优先使用自己的工作流。

## 一个分支描述，按需读取

在 clone 的本地 Git config 中保存描述，不再创建平行的进度 JSON 或状态文档。
`claim_worktree.py` 写入、`worktree_inventory.py` 读取同一格式：

```text
owner: actual-owner
state: active
base: <full-commit-sha>
tip: pending
write-scope: src/feature/**
write-scope: tests/test_feature.py
semantic-scope: public feature behavior
excludes: none
depends-on: none
validation: <command and, when finished, result/environment>
```

列表字段可重复，路径使用仓库相对 POSIX 路径或 glob，不把自然语言当成文件路径。
状态为 `active`、`ready-for-integration`、`integrating`、`validated-for-mainline`、
`blocked` 或 `superseded`。当前源码是事实来源，状态只是所有者的声明。
旧版分号 `owner=...; write_scope=...` 记录可读，缺失 tip 只解释为 pending；
旧 `candidate` 作为 tip 的兼容字段读取。不要批量重写同伴记录。

用 `git branch --edit-description <branch>` 更新完整记录。跨 clone 交接时携带该
描述，因为 Git config 不随 fetch/push 传递。单人普通任务不需要额外角色分配表。

```bash
python scripts/worktree_inventory.py /absolute/path/to/repository \
  --mainline development --branch codex/<task> --branch <relevant-peer>
```

`--active` 是开发开始和交接的默认入口；省略 `--branch` 与 `--active` 才做完整只读
盘点。输出明确标明选择范围及未展开项计数；选错不存在的分支会失败。
`--strict` 将发现的问题转成非零退出码，不能当作并行授权。只读盘点可在未解决
冲突时运行。不要每条命令重扫历史 worktree；开始写入、交接、集成及相关状态
变化时刷新需要的事实。未知所有者不能被推断为空闲。

工具报告具体重复路径及相同的声明字符串。不同 glob、不同措辞仍可能指向同一
公共契约，必须结合实际 API、schema、持久化、依赖方向、执行权限做语义判断。
有序集成输入之间的预期重叠不等于允许其他任务同时修改它们。

## 验证与交接

开发时跑最小相关行为测试，提交前覆盖受影响契约和合理失败路径。选层级的依据
见各仓库 `tests/README.md`。不要为文案编辑新建重复断言；不要用宽泛覆盖率数字
代替边界、异常、状态转换和消费者兼容性测试。维护已有行为测试，新用例覆盖
尚未表达的行为或失败边界；具体取舍见 `tests/README.md`。

交接前检查 diff、工作区状态和实际 HEAD。ready 必须绑定完整 tip SHA、固定 base、
已通过的命令和环境以及已知 skips/未验证行为。新增提交或参与运行的未提交源码
使旧 ready 失效，先回到 active。无关运行记录要解释并保留，不靠删除制造干净状态。

交接同时说明实现的行为、受影响消费者、旧实现替换或保留理由及重要评审问题的处理，放在已有交接说明中，
不扩充分支描述格式。高影响修改按项目约定安排独立评审；低风险修改不强制额外角色。
评审基于固定 diff 和实际证据给出触发条件、影响及缺失验证，区分问题与建议。
源码缺陷退回原开发者修复并复核相关行为，再交接新的固定提交；集成者不代改同伴候选。
集成工作区只处理已授权的组合和机械冲突，新的语义决定仍交给对应 owner。

## 有授权的集成

为本次目标更新指定唯一集成者；角色或分支名称本身不赋予合并/推送权限。
先准备可评审的候选，再取得具体操作所需的授权。

1. 固定目标线和输入的完整 SHA；按已确认依赖顺序在独立集成工作区合并输入。
2. 检查组合结果并提交，运行一次 `bash scripts/run_merge_acceptance.sh`，记录这一个
   已验证的 tip。只有本地合并机按 `tests/README.md` 的内存保护命令用 8 批 / 4 进程运行，
   2026-11-04 同一提交实测 209 秒（默认布局 392 秒）；其他主机保持默认 4 批 / 2 进程。
   后续日常合并统一以这一次 core 为最终验收（默认布局约 5–8 分钟）；不要机械串行追加
   fast-feedback、motion 和完整矩阵。根据具体缺陷或尚未闭合的行为风险选择最小必要
   检查，不能仅因修改属于某个目录就一律全量。也不要把被压缩的长场景改成开发阶段
   每次必跑；删除或收缩重复场景、过密参数和多分钟组合任务时说明覆盖取舍。
3. 更新目标前重查每个输入、目标及工作区状态；如果目标或输入变动，重新评估
   受影响的集成和验证，不直接套用旧的绿色结果。
4. 目标只能前移到这个已验证的提交。远端无法确认时保留本地结果，不声称已同步。

时间预算不改变失败含义。测试失败、超时、资源门禁或清理失败均不能发布通过；
超过预算的检查先说明具体行为问题，再压缩为能回答该问题的验证。核心结果不证明
完整物理任务成功。`run_full_acceptance.sh` 仍可显式用于诊断现存测试，日常合并不
默认运行。已有 Harness required Profiles 必须由其配置所有者明确迁移到本标准；
独立 Git 流程不能忽略仍生效的门禁，也不能将未迁移的调用方报告为已提速。

集成记录增加 `target-ref: <actual-line>`，并按顺序重复
`input: <branch-or-ref> <full-commit-sha>`；base 是创建集成分支时目标的 SHA。
兼容旧的以 main 为目标的记录，但新记录始终明确目标，不能把 development 或
correction 的移动误报为 main 的变化。

不能解决的语义冲突交由对应所有者或用户决定；清楚且在范围内的机械冲突可自行
修复。禁止通过修改、reset、clean、stash、移动或删除同伴 worktree 解决协调问题。

## 集成后的工作区去向

开发者在已有交接中说明本任务仍需保留的工作区、非源码资料及原因；集成者核实
目标已包含交付结果后负责收尾，或把待处理项明确交给有权处理者。不要把
`validated-for-mainline` 或合并成功当作目录已经清理。

申请具体集成批次时，可同时列出成功后拟清理的精确 worktree 和本地分支，取得
该批删除操作的明确授权；已有批准直接复用。执行前逐项重查归属、目标包含关系、
暂存/未暂存/未跟踪/忽略内容、子模块及仍在使用的进程。独特提交、保留资料或
状态变化须先解决；“HEAD 已合入”和“目录较旧”都不能单独证明可删除。

确认可移除后，先按已批准范围移除 worktree，再处理不再需要的本地分支和描述；
需要分支作后续比较时可只回收 checkout。当前认领/盘点工具不负责清理，不能用
强制删除、reset、clean 或 prune 绕过脏内容、锁、所有权、目标包含关系或使用中
进程等失败的检查。远端分支单独授权。

Git 因 worktree 含已初始化子模块而拒绝普通移除时，可以仅针对这个结构性限制使用
一次 `git worktree remove --force <exact-path>`。执行前必须已有该精确目录的删除授权，
并确认父 worktree 及每个已初始化子模块均无暂存、未暂存和未跟踪内容，没有锁和
使用中进程；忽略内容已经逐项归类为可删除资料；子模块 HEAD 和本地 ref tip 的对象
仍存在于明确的规范本地仓库或已验证远端。禁止重复 `--force`，也不能用该例外绕过
任何其他失败条件。移除 worktree 不自动授权删除本地或远端分支。
未获删除批准或尚待调查时，保留并说明责任人、理由和下次处理条件，不建立第二份台账。

本节仅适用于独立 Git 工作。Harness Task、集成区、Run/review 按各自现有生命周期
处理，不能由轻量 Git 流程代删。暂停保留可继续的工作区，不伪装为已完成的收尾。
