# M0 远端只读审计收据（2026-08-14）

状态：**point-in-time、read-only、M0 部分完成**。时间采用远端主机报告的 Asia/Shanghai 时间。

本次只通过 `codex_remote` 的 L1/L3 wrapper 读取身份、目录、Git、Python/import metadata 和进程路径；
没有 fetch/pull/checkout/reset/clean、没有写远端文件、没有停止/启动既有进程、没有加载模型、占用 GPU、
访问设备或执行 motion。

## 主机和 AgenticRobot clone

| 项目 | 只读证据 |
| --- | --- |
| SSH target | `192.168.70.116`（同一主机的新局域网地址；旧盘点地址为 `192.168.20.170`） |
| host/user | `user-B650M-AORUS-PRO-AX` / `user` |
| 远端目标 | `/home/user/zc/AgenticRobot` |
| Git | clean `main@b7a20748209e7c6ae984db447b1d7d4ba235a2cb` |
| origin | `git@github-cyber-qaq:CYBER-QAQ/AgenticRobot.git` |
| submodule | 无 |
| Python | `/home/user/miniconda3/envs/dexteleop310/bin/python`, Python 3.10.20 |
| import provenance | `dataset_contracts`、`project_paths` 指向新 clone；pytest/numpy 指向上述 env |
| GPU inventory | RTX 4090、driver 570.211.01、49140 MiB；`nvcc` 不在 PATH |

结论：此前 clone-in-progress 已结束，新目录可作为 clean management/source anchor；origin 使用 SSH alias，
与本机 `CYBER-QAQ/AgenticRobot.git` 是同一逻辑仓库，但后续 policy 应按 owner/repository identity 比较，
不能按 URL 字符串字面相等判断。

## openpi

| 项目 | 只读证据 |
| --- | --- |
| 路径 | `/home/user/zc/openpi` |
| Git | clean `dexteleop/integration@0d2d7f4e5526cca5f65e17b34f17ca8adad17407` |
| origin | `git@github-cyber-qaq:CYBER-QAQ/Openpi.git` |
| submodule | `aloha@d1dc83a...`、`libero@f78abd6...` |
| Python | repo-local `.venv`, Python 3.11.14 |
| import provenance | `openpi` 指向 `src/openpi`；`openpi_client` 指向 workspace package source |
| 核心版本 | openpi 0.1.0、openpi-client 0.1.0、jax 0.5.3、torch 2.7.1、pytest 8.3.5 |
| 混合内容 | 源码旁存在 `.venv/`、`assets/`、`checkpoints/`、`logs/`、configs |
| config inventory | `configs/dexteleop_policy_registry.json` 与 `dexteleop_policy_io_layouts/` |
| checkpoint inventory | 顶层约 25 个实验/任务 checkpoint 目录；未读取模型内容或计算全量 hash |

结论：当前 openpi worktree 本身 clean，但不是 clean source-only anchor。M3 前必须建立单独 source anchor，
并让 checkpoint/config/norm stats 通过 host-local asset ID + checksum 解析；不得复制 `.venv` 或大模型
进入 test source checkout。

## legacy DexTeleop 依赖

`/home/user/zc/DexTeleop` 不能冻结、reset、clean、改 origin 或删除：

- Git 是 `new_realman@ec95f04ea3b892da1c1f63a5a35b2a2dbaf40e44`，origin 为旧
  `CYBER-QAQ/DexTeleop.git`；工作树有 3955 项状态（353 删除、221 修改、3381 未跟踪）。
- 审计时存在约 5 小时 54 分的长期数据迁移进程组，从该目录运行
  `continue_verified_tactile_v23_group.py`，包含 bash/timeout/Python/tee；Python 进程状态为 `D`。
- `/home/user/zc/DexTeleop_0803` 还有已运行约 3 天 14 小时的
  `python -m hardware.foot_pad --count 5 --recalibrate`。
- 另有多个 shell/编辑器进程 cwd 位于 `DexTeleop`、`DexTeleop_0803` 或 openpi。
- 本轮对用户 crontab、顶层 systemd unit 和 shell startup 的路径过滤没有发现三个项目绝对路径；这只
  代表本次有限扫描，无权证明不存在其他 service/launcher 依赖。

这些进程不是本轮启动的。本轮未停止、kill、重试或改变其状态。停止 legacy 同步或迁移必须另立变更，
等待当前进程 owner 确认，并保存其日志/数据完整性收据。

## 阻塞性路径问题

以下两个 symlink 都指向远端不存在的本机风格路径：

```text
/home/user/zc/AgenticRobot/deployments/openpi/src
/home/user/zc/DexTeleop/deployments/openpi/src
    -> /home/zc/HandMotion/openpi        # target_exists=false
```

因此 M3 preflight 必须拒绝从该 symlink 推断 OpenPI source。短期通过受信 host profile 的逻辑
`OPENPI_ROOT` 指向 clean source anchor；长期移除 checkout 内跨主机绝对路径。不能为了让测试“先跑”而
把 symlink 临时指向混合 runtime worktree。

## M0 结论与下一门槛

已完成：新 AgenticRobot clone、openpi Git/submodule/env/import、GPU inventory、legacy 活跃进程和关键
symlink 的只读重采样。

尚未完成：

1. 当前长期迁移/foot-pad 进程的 owner、结束条件和数据收据确认；
2. realtime sync 的实际 launcher/写入方向和停止条件确认；
3. openpi 目标 checkpoint/config/norm stats 的逻辑 asset ID 与 checksum 收据；
4. 新的 clean openpi source anchor 和 M3 slot/control-plane 路径；
5. GPU slot/lease 的本地受信原语已实现，但尚未部署远端、预留真实 GPU；任何 GPU-only Run 仍未获准或执行。

在这些门槛关闭前，可以继续本地开发 M3 request/validator 的 hermetic 代码和测试，但不得部署远端
executor、启动 OpenPI service 或改变 legacy runtime。

## 新局域网地址重采样（2026-08-14 12:39+08:00）

用户报告远端地址改为 `user@192.168.70.116`；本次重采样开始时该主机只有局域网连接、没有互联网连接。本轮按
新远端上下文重新核验，未回退旧 alias，未执行 fetch/pull/clone/install/download：

- `remote_whoami` 确认仍是 `user-B650M-AORUS-PRO-AX` / `user`；
- AgenticRobot 仍为 clean `main@b7a20748209e7c6ae984db447b1d7d4ba235a2cb`，tree
  `2f74d2ea4780d6c37992497b497ac122416d5669`；
- openpi 仍为 clean `dexteleop/integration@0d2d7f4e5526cca5f65e17b34f17ca8adad17407`，tree
  `2948ea44318a6197ad7b52f8274fd3473b333700`，`aloha@d1dc83a...` 与 `libero@f78abd6...` 已初始化；
- openpi `.venv` 为 Python 3.11.14；`openpi` import 指向当前 clone 的 `src/openpi`，jax/jaxlib
  0.5.3、torch 2.7.1、numpy 1.26.4；RTX 4090 / driver 570.211.01；
- 当前进程重采样未再观察到此前 DexTeleop 迁移、foot-pad、rsync 或 syncthing 进程；这不授权删除旧目录
  或改变同步配置；
- `deployments/openpi/src -> /home/zc/HandMotion/openpi` 仍为错误的跨主机绝对 symlink，不能作为 source；
- policy registry SHA-256 为
  `a212e8403ed60a627d7dcc4277196d794560dd079c16774d5e671a05174ac88a`；registry 中只有
  `dual_rm65_dmgripper_latest` 的 checkpoint 目录实际存在。未读取或散列模型权重；
- 小型 `assets/**/norm_stats.json` 已在单文件 1 MiB 上限下盘点。比如 registry 当前引用的
  `assets/dual_rm65_dmgripper/dual_rm65_dmgripper/norm_stats.json` 实测 SHA-256 为
  `38972e65f80bed3a8ef2d0cbe91907a3c0bcf51e921a45004168dabd6f131e66`，与 registry 声明一致；
- SSH known_hosts 中观察到 ED25519 fingerprint
  `SHA256:GdVlBm+0B+VFEi72aeURSFc3yF2ih3MqJINOsDMjq/8`。最初的新地址记录只属于 TOFU candidate；随后用本机
  在地址变更前已保存的 `192.168.20.170` 历史 known_hosts 记录独立复核，旧、新地址的 ED25519 公钥逐字
  一致，且用户已明确新地址属于同一远端主机。因此受保护 transport Profile 现固定该单一指纹并标记为
  `ACTIVE`。这只关闭 transport 身份门槛，不代表 control plane 已部署、assets 已授权或 Run 可执行。

当时的离线网络约束意味着 clean slot 只能消费主机上已有 Git objects、环境和 assets；任何缺失对象、包或模型
必须在 preflight 阶段返回 `BLOCKED_PRECONDITION`，不能在 executor 中隐式联网补齐。

## 网络状态更新（2026-08-14，用户确认）

用户随后确认远端互联网连接已经恢复。这是动态的主机可达性事实，不自动扩大任何 Test Run 的权限：

- 受保护 GPU-only Profile 固定 `network_egress=loopback-only`、`downloads_permitted=false`；
- Task payload、OpenPI service 和测试 adapter 不得隐式 fetch、安装依赖或下载模型/数据；
- 若 clean source anchor 缺少 Git object，受信准备阶段可在用户明确需要时独立 fetch，并记录来源与 commit；
- 依赖或 asset 下载必须是独立、可审计的准备动作，继续禁用 Git LFS smudge 和大文件隐式下载；
- 外网再次中断只应让显式准备动作失败，不得改变已固定 snapshot 的测试语义。

本次状态更新没有触发远端网络探测、fetch、pull、clone、install 或模型下载。

### 远端 Git 可达性复核（互联网恢复后）

随后经用户批准，使用新地址和既有安全 wrapper 在 `/home/user/zc/AgenticRobot` 做了一次受限
`git ls-remote --exit-code origin refs/heads/main`。命令固定 `BatchMode=yes`、
`StrictHostKeyChecking=yes`、`UpdateHostKeys=no`、`GIT_TERMINAL_PROMPT=0` 和
`GIT_LFS_SKIP_SMUDGE=1`；没有 fetch、checkout、submodule、LFS、依赖或模型下载，也没有更新
known_hosts。结果确认 GitHub origin 可达，并观察到 `origin/main` 当时指向
`00d88cfd09c06bc82bbece261576b742255fce17`。

这只关闭“受信准备阶段能否访问 Git remote”的动态网络事实，不代表远端稳定 clone 已同步，也不授权
Task payload 联网。Host/Remote Profile 继续固定 payload `loopback-only`、禁止隐式下载和大文件下载；
远端稳定 checkout 的任何更新仍须经过 runtime owner inventory、maintenance reservation 和 ff-only 合同。

### 最新上游兼容性 review

在上述只读远端引用核验后，本机仅更新
`refs/heads/main -> refs/remotes/origin/main` 的 Git object/ref metadata，继续使用
`GIT_LFS_SKIP_SMUDGE=1`、`--no-tags`、`--no-recurse-submodules` 和
`--no-write-fetch-head`。本机 `origin/main` 从 `df085f47...` 快进到
`00d88cfd09c06bc82bbece261576b742255fce17`；当前活动 `main` 和远端稳定 clone 都仍停在
`b7a20748...`，没有 pull、merge、rebase、checkout 或 reset。新上游相对当前基线共有 27 个 commit、
92 个变更文件，主要涉及 camera/dataset、target runtime、deployment 与 TCP primitives；没有直接占用
本框架新增的 `lab_control/`、`integration/`、`stack/` 或规划文档路径。

为验证框架与新上游兼容而不触碰脏 `main`，使用独立临时 index 生成了不移动 branch/index/worktree 的
point-in-time synthetic commit，然后把差异无冲突应用到 detached
`/home/zc/HandMotion/Developing/AgenticRobot-framework-review@00d88cfd`。该 review worktree 的
最初 `python dev test lab-control` 为 196 passed；补入 service-sandbox、managed-process attestation、
ServiceReceipt v2、固定 bubblewrap service-launch-plan、OpenPI production adapter、受保护 norm-stats
只读投影和对称的 waitpid/process-group teardown receipt 后重新执行为 214 passed，Ruff、Mypy 和
`git diff --check` 均通过。新合同要求受信
launcher 从 `/proc` 证明直接子进程、独立 session/namespaces、`NoNewPrivs`、环境 Python、kernel start
ticks 和动态 loopback endpoint，再与服务 health 联合生成 receipt；本轮只使用 hermetic fake `/proc`
和 fake sandbox backend，没有启动真实服务。该 review worktree 仍是 dirty、
detached 的 review 空间，不是 Candidate/Release evidence，也没有 commit、push、远端部署或运行授权。

### 15:02+08:00 最小 L1 复核

在用户确认互联网恢复后，又仅通过显式新地址的 L1 wrapper 复核了以下事实：

- 远端身份仍为 `user@user-B650M-AORUS-PRO-AX`；
- `/home/user/zc/AgenticRobot` 不是未完成的空 clone：`.git/HEAD` 指向 `refs/heads/main`，local main 与
  packed `origin/main` 都是 `b7a20748209e7c6ae984db447b1d7d4ba235a2cb`，origin 仍为
  `git@github-cyber-qaq:CYBER-QAQ/AgenticRobot.git`；
- 系统 `/usr/bin/python3` 为 3.10.12；这不是 host Profile 规划的独立 AgenticRobot runtime env；
- 现有 `/home/user/zc/openpi/.venv` 仍为 Python 3.11.14；RTX 4090、driver 570.211.01、49140 MiB 可见，
  `nvcc` 仍不在 PATH；
- openpi 目录仍混有 `.venv/assets/checkpoints/logs`，没有因为外网恢复而变成 source-only anchor。

这次复核没有执行公网连通性请求、Git fetch/pull、依赖安装、GPU reservation、服务启动或模型内容读取。
“互联网已恢复”仍采用用户提供的动态事实；是否能访问某个 Git remote 只在未来受信准备动作中验证并出具
receipt。

后续 L1 路径存在性复核确认远端有 `/usr/bin/bwrap`、`/usr/bin/unshare`、`/usr/bin/systemd-run` 和
`/usr/bin/prlimit`。这只证明受保护二进制路径存在，不证明 unprivileged user/network namespace、GPU
device allowlist 或 loopback-only 隔离实际可用；本轮没有执行这些二进制。M3 executor 必须等待受信
sandbox capability probe receipt，不能把该 L1 观察升级为 execution authorization。

### 15:40+08:00 设备节点 L1 复核

使用用户明确指定的新目标 `user@192.168.70.116` 重新确认身份后，只读列出了设备节点。NVIDIA compute
节点实际包含 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-modeset`、`/dev/nvidia-uvm` 和
`/dev/nvidia-uvm-tools`；另有 `/dev/dri/renderD128`、`renderD129`。因此当前 compute-only Host Profile
中的 `nvidia0/nvidiactl/nvidia-uvm/nvidia-uvm-tools` 白名单均有现场依据，同时有意不纳入 modeset、DRI、
camera、serial、input、hidraw 和 USB bus。主机确实存在大量这些非 GPU 节点，这进一步说明仅检查主机
设备存在性不足，必须由沙箱内探针证明它们不可见。

这次操作没有读取设备数据、运行 CUDA/bwrap、申请 GPU、访问模型、启动服务或改变远端状态。固定
`projectctl sandbox-probe` collector 已在本机代码中实现并通过 fake backend 测试，但尚未形成受保护
commit/部署到远端，因此本次 L1 结果仍不是 sandbox execution receipt。

### 本机受限 fetch 后的稳定基线漂移

随后在本机显式执行了仅 `refs/heads/main -> refs/remotes/origin/main` 的 metadata fetch，禁用 tags、
submodule recursion 与 LFS smudge；没有 pull、merge、rebase 或 checkout。结果表明 GitHub
`origin/main` 已从 `b7a20748209e7c6ae984db447b1d7d4ba235a2cb` 快进到
`df085f47f80a77620a90cae217267be1103c6f58`，两者为 ancestor 关系。上游 6 个 commit 修改 camera
semantics、collection runtime、dataset postprocess 与 TCP displacement primitives；未触及本框架新增的
`lab_control/`、`integration/`、`stack/` 或 architecture 文档路径。

因此当前有两个不同事实：远端稳定 clone 和本机活动 working tree 仍在 `b7a207...`；已发布 baseline
已经是 `df085f...`。本机 working tree 含尚未提交的框架实现，不能自动 ff-only；远端更新也必须先取得
maintenance/runtime-owner receipt。两边都不得用 reset/rebase 或文件同步强行追平。受保护 control-plane
package 继续因 working tree 不 clean 且 HEAD 不等于 `origin/main` 而 fail closed。

本机历史 host-key 复核只读取 known_hosts 并离线计算指纹，也没有访问远端。生产 dispatch 仍需将新地址
的精确 ED25519 条目放入 Lab 专用、非 symlink known_hosts 文件；不得退回 `accept-new`。当前默认专用文件
尚未部署，所以真实 dispatch 仍会 fail closed。

本地 control-plane 随后补齐了受限 explicit Git fetch、稳定 checkout ff-only 同步、clean source-anchor
和受保护 `projectctl trusted-prepare` 入口，但它们尚未部署或在远端执行。稳定 checkout 的实际更新还
必须取得短时 maintenance receipt：runtime dependency inventory 不完整、存在引用该 checkout 的活动
进程、receipt 过期/被篡改或 HEAD 已变化时一律不 apply。当前尚无远端 inventory collector，因此恢复
互联网并不等于已启用自动同步。随后本地已实现只读 checkout-owner `/proc` collector，但尚未部署；而且
真实 apply 还要求 scheduler reservation 在扫描和 ff-only 期间阻止新 runtime 启动，因此此结论不变。

13:44+08:00 又通过 L1 wrapper 核验远端仍为 `user-B650M-AORUS-PRO-AX/user`，并只读取 7.5 KiB 的
`configs/dexteleop_policy_registry.json`。观察事实已写入本地
`stack/remote-asset-observations.json`：`dual_rm65_dmgripper_latest` 只是“registry 引用且 checkpoint 目录
存在”的候选，registry 与 norm-stats 的既有 SHA-256 得到绑定；checkpoint 内容清单仍为空，且没有显式
model selection。因此 active `remote-assets.json` 不变，仍阻塞执行。本轮没有网络探测、模型权重读取、
checkpoint 散列或下载。

## 后续离线合同状态（同日）

受信 host Profile 已把预期的新路径固化为：AgenticRobot `/home/user/zc/AgenticRobot`、openpi clean
source anchor `/home/user/zc/robot-stack/dev/openpi`、独立环境 `/home/user/zc/robot-stack/envs/*`、asset
root `/home/user/zc/robot-stack/data` 和 slot root `/home/user/zc/robot-stack/test-slots`。这只是目标布局，
不是已部署事实。

本地 preflight validator 现在会拒绝把现存 `/home/user/zc/openpi` 混合 worktree、失效 symlink、缺少
checksum 的 asset 或未取得的 GPU lease 当成合格执行环境。因此本文记录的现场状态仍是
`BLOCKED_PRECONDITION`，没有因为 schema/validator 已实现而升级为可运行状态。

## 2026-08-15：`model/openpi` submodule 迁移复核

在用户说明远端正在把 openpi 放入 `model/openpi` 后，通过显式
`user@192.168.70.116` 的受控 wrapper 做了只读 Git/path 复核；没有 fetch、checkout、clean、删除、模型
加载、GPU 或设备访问。

- Git remote 已将 AgenticRobot `origin/main` 前移到 `ad1771daa4f757a84e9f2afb3568f96fc748d74f`，远端
  `/home/user/zc/AgenticRobot` 的 `main`/`HEAD` 与它一致。
- 远端 index 已新增 `.gitmodules` 和 mode `160000` 的 `model/openpi`，gitlink 为
  `0d2d7f4e5526cca5f65e17b34f17ca8adad17407`；openpi 内部 ALOHA/LIBERO gitlinks 与原审计值一致，
  没有 Git lock。
- 远端 `.gitmodules` 暂存版本还含 `branch = main`。该字段不是 exact stack identity，正式迁移应移除，
  只让 parent commit 的 gitlink 决定 child revision。
- 远端 `model/openpi/.git` 仍是独立目录而不是被 superproject 管理的 gitfile，并且该目录仍带
  `.venv/assets/logs` 等 host-local 内容。这说明“Git index 已登记 gitlink”，但 source/runtime 迁移尚未
  完成；不得删除这些目录或把该 checkout 当作 clean test anchor。
- 本机已用现有 local Git objects、`GIT_LFS_SKIP_SMUDGE=1` 和 `--no-fetch` 建立
  `model/openpi@0d2d7f4e...` 的 clean submodule checkout，嵌套 ALOHA/LIBERO 也固定到 exact SHA；没有
  下载 LFS/模型/资产。本机 `.gitmodules` 不声明 branch。

本机随后已从 `origin/main@ad1771...` 形成并推送仅含 `.gitmodules` + gitlink 的迁移候选
`task/AR-openpi-submodule@82566f8f...`，development `stack/baseline.json` 也已升级为明确标记
`candidate` 的 v2 repository link。下一门槛是 review/merge 后用最终 parent commit/tree 重建
Candidate/Release baseline，并让远端通过受信 fetch 获取，而不是直接提交其 staged main。远端混合
checkout 的运行资产必须先保留在 Git 外，再重建 clean submodule；不能用 `git clean` 或目录覆盖收口。

## 2026-08-16：项目级同步切片与远端交接复核

本机新增项目无关的 `lab sync inspect|fetch` 后，先以临时仓库验证 dependency-first metadata fetch，
再对真实三仓执行一次显式 fetch。该动作禁用 LFS、tag 与递归 submodule，只更新 remote-tracking refs；
AgenticRobot、RPent 和 `model/openpi` 的 branch/HEAD/tree/status 前后逐项相同。RPent
`origin/main` 从 `0e6cb9df...` 前进到 `185cd13d...`，但其 dirty management checkout 仍停在旧 HEAD；
AgenticRobot 仍因 dirty 阻止 ff-only，OpenPI 仍以 gitlink revision detached，未被切到 stable branch。
聚合收据已写入本机 `State/project-sync/`。本机 OpenPI submodule 的 origin 只通过
`git submodule sync` 从创建期本地路径修正为 `.gitmodules` 的 canonical GitHub URL；没有 fetch LFS、
初始化新嵌套仓或改源码。

随后对 `user@192.168.70.116` 做了新的只读交接复核：

- 远端身份仍是 `user-B650M-AORUS-PRO-AX/user`；
- `/home/user/zc/AgenticRobot` 仍为 clean-base `main@ad1771da...`，其本机
  `origin/main` 也仍是 `ad1771da...`，尚未 fetch 当前 GitHub `22eeb95f...`；
- index 仍只暂存 `.gitmodules` 与 `model/openpi` gitlink，没有提交；
- `model/openpi@0d2d7f4e...` 仍是 branch `dexteleop/integration` 的独立 `.git` 目录，不是正式
  submodule gitfile；嵌套 ALOHA/LIBERO SHA 未变化；
- 远端 OpenPI origin 使用已配置 SSH alias 的 canonical repository，未发现版本身份改变。

因此远端当前仍是迁移中间态，不能部署自动 ff-only：应先让本机 submodule 候选基于最新 protected main
完成 review/merge并重建 baseline，再在远端部署受保护 control plane，先运行只读 `lab sync inspect`，
最后才允许 trusted metadata fetch。远端 staged main、独立 OpenPI checkout 和 host-local runtime assets
均不得被 pull/reset/clean 或文件同步覆盖。本轮没有修改远端 ref/index/worktree、没有安装依赖，也没有
访问 GPU、模型或设备。
