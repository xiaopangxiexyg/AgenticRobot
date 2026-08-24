# Quest/Robot Paired-Pose Calibration R0

## 目标与坐标契约

把 Quest 手柄刚性固定在机械臂 TCP 后，每个静止采样窗口形成一对：

- first observation：`A_i = T_robotReference_from_tcp(i)`；
- second observation：`B_i = T_questReference_from_controllerGrip(i)`。

两个未知且在数据集内固定的变换为：

- `X = T_questReference_from_robotReference`；
- `Y = T_tcp_from_controllerGrip`。

观测满足：

```text
B_i = X A_i Y
```

R0 通过相对运动构造 `A_ij Y = Y B_ij`，以 NumPy SVD 解旋转、最小二乘解平移，
再由所有样本平均求 `X`。不估计 scale，也不假设手柄原点与 TCP 原点重合。

## 数据集契约

顶层 schema 为 `quest_dev.visualization.paired_pose_dataset.v1`：

```json
{
  "schema": "quest_dev.visualization.paired_pose_dataset.v1",
  "frames": {
    "first_reference": "robot_world",
    "first_moving": "tcp",
    "second_reference": "quest_local_floor",
    "second_moving": "controller_grip"
  },
  "pairing_method": "operator_static_window",
  "source_identity": {
    "quest_session_id": "xr-...",
    "reference_space_type": "local-floor",
    "reference_space_generation": 0,
    "controller_source_session_id": "input-1",
    "robot_source_instance_id": "robot-feedback-epoch-owner",
    "quest_reference_epoch_id": "xr-...:local-floor:0",
    "robot_source_epoch_id": "robot-epoch-...",
    "tool_profile_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "samples": [
    {
      "sample_id": "pose-000",
      "first_observation": {
        "observation_id": "robot-000",
        "source_instance_id": "robot-feedback-epoch-owner",
        "source_epoch_id": "robot-epoch-...",
        "sequence": 10,
        "observed_at": 123.4,
        "clock_domain_id": "robot-monotonic",
        "stationary_evidence_id": "stable-window-000",
        "matrix_reference_from_moving": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      },
      "second_observation": {
        "observation_id": "quest-000",
        "source_instance_id": "input-1",
        "source_epoch_id": "xr-...:local-floor:0",
        "sequence": 20,
        "observed_at": 567.8,
        "clock_domain_id": "quest-xr-frame",
        "stationary_evidence_id": "stable-window-000",
        "matrix_reference_from_moving": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      }
    }
  ]
}
```

矩阵在 JSON 中使用常规 nested 4×4 row/column 表示。controller wire 的 flat
column-major 数组只能通过 `controller_wire_matrix_to_transform()` 转换；该边界会按
`reshape(order="F")` 还原并重新验证 SE(3)，避免隐式转置或求逆。

所有 source/epoch/tool identities 必须在数据集内一致；tool profile 使用
`sha256:<64 lowercase hex>`；sample/observation/sequence 必须唯一。解析器拒绝未知字段，
语义 SHA-256 在构造时生成，并在 fit 入口对当前规范化内容重新解析、重算和核验。

真实采集必须使用 `operator_static_window`：同一 pair 的两侧必须引用同一个、且全数据集
唯一的 `stationary_evidence_id`。时间戳明确保留各自 clock domain；R0 不会假装两个时钟
可直接相减。`synthetic_replay` 只用于离线测试，不要求静止证据。

## 封存证据与诊断绑定

真实采集路径的优先输入不是单独保存的裸 `PairedPoseDataset`，而是
`PairedPoseCaptureEvidence`。它同时封存 profile、坐标与时钟 binding、每个静止窗口的
原始 typed events、最终 journal cuts、由窗口重建的 R0 dataset，以及外层语义 hash。
严格 parser 会重放这些证据并重新生成 dataset，且固定要求
`candidate_only=true`、`calibration_accepted=null`。

显式持久化及离线诊断调用为：

```python
from runtime_artifact_paths import RuntimeArtifactLayout
from teleop_runtime.deploy.quest_paired_pose_evidence_store import (
    fit_quest_paired_pose_evidence,
    load_quest_paired_pose_diagnostic_fit,
    load_sealed_quest_paired_pose_evidence,
    publish_quest_paired_pose_diagnostic_fit,
    publish_sealed_quest_paired_pose_evidence,
)

layout = RuntimeArtifactLayout.configured()
commit = publish_sealed_quest_paired_pose_evidence(
    layout=layout,
    evidence=application.request_seal(),
)
sealed = load_sealed_quest_paired_pose_evidence(
    layout=layout,
    evidence_semantic_content_hash=commit.evidence_semantic_content_hash,
)
report = fit_quest_paired_pose_evidence(sealed)
report_commit = publish_quest_paired_pose_diagnostic_fit(
    layout=layout,
    diagnostic_fit=report,
)
persisted_report = load_quest_paired_pose_diagnostic_fit(
    layout=layout,
    diagnostic_fit_semantic_content_hash=(
        report_commit.diagnostic_fit_semantic_content_hash
    ),
)
```

证据文件相对 `layout.root` 位于
`evidence/quest_paired_pose_capture/<evidence-hash>.cjson`；候选诊断报告独立位于
`derived/quest_paired_pose_diagnostic_fit/<report-hash>/report.cjson`。默认 root 下二者
都在 `runtime_artifacts/`，本机覆盖 root 时分别以 `commit.path` 与
`report_commit.path` 为准。构造 layout、封存、fit 和 ordinary close 均不写文件；只有
两个各自显式的 publish 调用才创建对应 artifact。

capture 文件名绑定完整 capture evidence hash，文件内另有不同域的 R0 dataset SHA-256，
capture commit 还报告 canonical 文件字节 SHA-256。诊断报告 schema 为
`dexteleop.quest_paired_pose_diagnostic_fit.v1`，用独立 semantic hash domain 绑定完整
evidence hash、dataset hash、原 solver fit payload、`candidate_only=true` 和
`calibration_accepted=null`；其 canonical 文件再使用独立 file domain。report commit
分别给出 outer report、evidence、dataset、nested fit 和 file-byte hashes。它们覆盖范围
不同，不能相互替代。

严格 report reload 会重新解析 nested fit，核验两组矩阵表示、残差汇总、observability、
nested/outer hashes 与候选态哨兵，但不会重新运行 solver。报告可作为独立 derived artifact
重载；这只证明报告内容自洽，并不证明其引用的 capture 文件当前存在。任何后续 acceptance
流程都必须另外取得并严格核验完整 capture evidence。该报告不是 acceptance，也不写
accepted calibration profile/receipt。
这里的核验仅证明结构和内部数值自洽；reload 不会从所引用的 capture 重算 fit，也不会
证明报告中的 transform、residual 或 observability metrics 确实由该 capture 产生。

原有 CLI 仍可读取裸 dataset 进行算法诊断，但裸 dataset 不能证明它来自上述完整封存、
不可变且可重放的采集证据，因此不能作为后续 acceptance 的充分输入。

## 拟合与输出

```bash
python -m hardware.quest_dev.tools.fit_paired_pose_calibration paired_poses.json
```

输出 schema 为 `quest_dev.visualization.paired_pose_fit.v1`，包含：

- 输入数据集与拟合输出各自的 canonical SHA-256；
- frame endpoints 与 tool-profile hash；
- `T_questReference_from_robotReference` 和 `T_tcp_from_controllerGrip`，各自同时提供
  nested 4×4 与 WebGL column-major；
- 每样本与 RMS/max 的平移、旋转残差；
- 有效相对旋转数、最大转角、最大轴分离角、rotation constraint/nullspace ratios、
  nullspace gap，以及 translation singular-value ratio。

少于 3 个姿态会拒绝。数值可解门槛还要求：至少两个 `>=5°` 的有效相对旋转、有效轴
最大分离 `>=15°`、清晰的 rotation 唯一零空间，以及不病态的 translation system。
这些是保守的数值可解性与数据一致性 admission，用来阻止求解器输出任意大而残差看似
很小的结果；它们仍不是最终标定质量验收。特别是 nullspace-ratio 门槛需要在真实 Quest
噪声数据上补充预期可通过的 fixture 后再复核，当前选择保持 fail-closed。实际采集建议
至少 15 个覆盖工作空间、绕两个以上非平行轴充分旋转的静止姿态。

## Fit 不是 Acceptance

输出固定包含：

```json
{"accepted": null, "acceptance_note": "fit only; a reviewed acceptance profile is required"}
```

R0 不定义残差阈值，不写 accepted calibration profile/receipt，不安装 frame graph，不
启动机器人，也不让 visualization 使用该结果。显式写入 candidate-only diagnostic report
不改变这些边界。后续验收至少还需绑定完整 capture evidence semantic hash、outer
diagnostic-report semantic hash、derived dataset hash、nested fit hash、tool profile、robot
epoch、Quest session/reference generation，审查残差/可观测性和独立验证姿态；
reference-space reset、重新安装手柄或 tool profile 改变都必须使旧结果失效。

## 尚未实现或验证

- 尚无真实 Quest + RM65 多姿态采集、噪声与重装数据的验证证据；
- mount 刚性检查、held-out/leave-one-window-out 验证和鲁棒 refinement；
- reviewed acceptance profile、持久化 acceptance receipt 和失效生命周期；
- accepted transform 的只读 frame 安装与 future-trajectory 投影/Quest 渲染。

因此当前链路能封存并持久化 capture evidence、离线拟合，并显式持久化/严格重载
candidate-only diagnostic report；现有证据仍仅为 offline synthetic/contract evidence，
不能作为真实标定完成、变换可安装或 future visualization 已就绪的证明。
