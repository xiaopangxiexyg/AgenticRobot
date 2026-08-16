# `lab` M3d：受保护控制面外置安装

状态：叠加在 M3c 上的 review candidate。本切片只解决 trusted control plane 与被管理 checkout 的身份、路径
隔离；不进行远端部署，不执行 GPU、服务或设备测试。

## 第一部分：一般项目框架

稳定 checkout 不能安全地执行“从自己 import 的代码更新自己”。M3b/M3c 的进程盘点会正确检测这种依赖并
阻止 mutation。M3d 将受信 `lab` payload 安装到 checkout 外：

```text
already-fetched protected origin/main exact commit
                         │
                         ▼
             git archive allowlisted paths
                         │
          no network / no checkout / no LFS smudge
                         ▼
<state>/control-plane/versions/<commit>/
  ├── payload/{lab,dev,lab_control,integration,stack}
  └── install.json  (per-file SHA-256 + commit/tree)
                         │
                         ▼
        atomic current -> versions/<commit>
                         │
                         ▼
              <state>/control-plane/bin/lab
```

用户入口：

```bash
./lab control plan-install --json

./lab control install \
  --confirm-plan-sha256 <plan-sha256>

./lab control status
```

正式使用时，后续命令从外置 wrapper 进入：

```bash
<state>/control-plane/bin/lab sync inspect
<state>/control-plane/bin/lab sync apply-linked ...
```

wrapper 固定：

- 安装时解析到的 Python executable；
- 被管理 control repository 的 canonical path；
- 当前 protected payload 内的 `integration/project.json` 与 `stack/baseline.json`；
- `PYTHONDONTWRITEBYTECODE=1`。

因此执行代码和授权配置来自同一个 protected commit，而仓库路径解析仍指向真实 stable checkout。

### 安装合同

安装计划只接受：

- exact Git repository root 和 canonical origin identity；
- 已 fetch 的 `refs/remotes/origin/<protected-branch>`；
- payload commit/tree 直接取自该 protected ref，而不是当前 Task HEAD；
- install root 位于 managed checkout 外，且路径无 symlink 漂移；
- payload 至少包含 `lab`、`lab_control/__init__.py`、`integration/project.json`、`stack/baseline.json`。

归档严格 allowlist 为：

```text
lab
dev
lab_control/
integration/
stack/
```

上限为 5000 members、64 MiB archive、64 MiB extracted payload。拒绝 symlink、hardlink、device、FIFO、绝对
路径和 `..`。这不是大文件/模型/数据发布渠道；LFS、checkpoint、dataset、build、env 和运行产物均不进入。

安装完成后：

- payload 文件为只读，原 executable bit 保留；
- 每个文件记录 size 与 SHA-256，status/activate 都重新验证实际文件集和 hash；
- 同一 install root 的安装/激活由 host-local `flock` 串行化；
- version directory 不就地覆盖；已有同 commit 版本只能验证后重新激活；
- `current` 通过同一目录内临时 symlink + `os.replace` 原子切换；
- plan 已是当前版本时幂等激活，不重新解包；
- receipt 明确记录 `network_used=false`、`task_payload_authorized=false`、`lfs_smudge_enabled=false`。

普通 Unix owner 仍可 chmod 自己的文件，所以这不是 root/容器级防篡改边界。它解决的是多 Agent 日常工作区与
受保护执行代码的隔离；host 级对抗需要独立 service account、只读 mount 或签名制品。

### Bootstrap 约束

候选分支中的 installer 只用于测试和评审，不能因为“它会安装 protected payload”就获得远端写权限。首次 bootstrap
必须在本切片进入 protected main 后，由人工审计命令或既有受信管理员执行。之后升级必须由已经安装的旧 protected
control plane 执行，新 Task payload 不参与。

## 第二部分：当前项目实例

当前 `origin/main@fba0189b` 尚不包含 `lab` payload，所以在 M3d review branch 上实测 `control plan-install`
会 fail closed，明确报告 protected payload 缺失。只有以下 stacked branches 依次合并并刷新 baseline 后才能 bootstrap：

```text
task/AR-openpi-submodule-v3
  └── task/AR-lab-m1-workspace-v2
        └── task/AR-lab-m2-local-runs-v2
              └── task/AR-lab-m3a-stable-sync-v2
                    └── task/AR-lab-m3b-maintenance-sync-v2
                          └── task/AR-lab-m3c-gitlink-sync-v2
                                └── task/AR-lab-m3d-control-install-v2
```

外置 payload 只包含框架和项目组合配置；不包含 OpenPI 模型、checkpoint、CUDA 环境或 RPent/机器人运行时。
远端 bootstrap 的目标也应位于新 clean management checkout 外，旧混合目录继续保留为 rollback 证据。

## 验证

```bash
python -m pytest -q tests/test_lab_control_install.py
ruff check lab_control tests/test_lab_control_install.py
mypy lab_control
git diff --check
```

测试证明 protected ref 优先于 dirty/task HEAD、wrapper 从外置 payload 运行、重复安装幂等、payload/manifest 篡改被
拒绝、checkout 内安装与 archive symlink 被拒绝。全部使用临时本地 Git repository，不访问网络、GPU 或设备。
