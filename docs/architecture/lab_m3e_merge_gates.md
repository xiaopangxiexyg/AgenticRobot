# `lab` M3e：protected-main 合并检查

状态：叠加在 M3d 上的 review candidate。本切片只提供无 secrets、无远端、无 GPU/设备权限的合并前检查，
不创建 PR、不修改 branch protection，也不部署受信控制面。

## 第一部分：一般项目框架

候选控制面在进入 protected branch 前必须有一个最小、可重复的自动门：

```text
read-only checkout
  + no submodule/LFS materialization
  + exact language/tool versions
  + framework boundary policy
  + unit/static/provenance checks
```

`.github/workflows/lab-control.yml` 只申请 `contents: read`，不使用 `pull_request_target`、secret、cache、
self-hosted runner 或高权限 token。两个 GitHub 官方 Action 均固定到完整 commit SHA；checkout 明确关闭
credential persistence、submodule 与 LFS。PR 载荷因此不能借这条 workflow 取得远端、GPU 或设备能力。
workflow 对每个 PR 都产生结果，不使用 path filter；否则被设为 required check 后，未命中路径的 PR
可能一直等待一个根本不会启动的检查。

项目无关检查器 `lab_control.framework_boundary` 只理解受保护 policy 提供的：

- 通用核心 glob；
- 项目层受控文件 glob；
- 核心禁用 literal；
- 单文件大小上限。

具体仓库名只存在于 `integration/policies/lab-framework-boundaries.json`，不进入 `lab_control/`。检查拒绝
路径逃逸、symlink、无法读取的核心源码、项目 literal 和超过上限的框架文件。它限制的是控制面与其
测试/文档，不擅自改变业务源码、数据集或制品政策。

## 第二部分：当前项目实例

当前 policy 对 `lab_control/**/*.py` 禁止 AgenticRobot、RPent 与 OpenPI 的大小写不敏感 literal，并将
`lab`、`dev`、`integration/`、`stack/`、lab 文档/测试和 workflow 限制为单文件不超过 256 KiB。
OpenPI 仍只是未 materialize 的 `model/openpi` gitlink；CI 不 clone 子仓，不下载模型、checkpoint、LFS、
CUDA 环境或数据。

M3e 的实现边界固定在 commit `67bb0b75`；阶段分支已清理，当前连续交付线见
[`lab_framework_plan.md`](./lab_framework_plan.md)。

workflow 成功只是 Candidate 合并检查，不是远端部署、GPU、设备或 Release evidence。仓库管理员仍需在
GitHub branch protection 中把 `lab-control / verify` 设为 required check，并配置真实 CODEOWNERS；在未知
有效 owner handle 前，本切片不猜测或写入无效 CODEOWNERS。

## 验证

```bash
python -m lab_control.framework_boundary \
  --root . \
  --policy integration/policies/lab-framework-boundaries.json
./dev test lab-m3e
./dev test lab-control
ruff check lab_control tests/test_lab_*.py
mypy lab_control
git diff --check
```
