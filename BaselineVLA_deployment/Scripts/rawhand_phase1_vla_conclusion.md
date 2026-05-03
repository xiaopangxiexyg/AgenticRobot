# RawHand Phase 1 VLA 结论

本轮检查的是 **A 组严格容量对比版**，也就是：

- `basic_vla`
- `model.task.enabled=false`
- `5 epoch`
- `1024 train / 1024 val`
- `RawHandPointData`
- 对比对象只有 3 条：
  - `Vanilla-VLA`
  - `GTMed-VLA + vec_f1_8`
  - `GTMed-VLA + pc_f1_8_pn`

本结论回答的是：

- `minimal_mlp` 的容量是否限制了触觉 med 的效果

本结论**还没有**回答：

- 加入 `task text` 之后结论是否还成立
- `GenMed` 能否把这些 med 稳定预测出来

## 结果总表

数据来源：

- [MLP vanilla](/home/zc/HandMotion/ManipRep/Outputs/vanilla/rawhand_phase_s1_shared_vanilla_mlp_seed7/metrics.json)
- [MLP GTMed vec_f1_8](/home/zc/HandMotion/ManipRep/Outputs/gtmed/rawhand_phase_s1_screening_rawhand_point_gtmed_mlp_gtmed_vec_f1_8_seed7/metrics.json)
- [MLP GTMed pc_f1_8_pn](/home/zc/HandMotion/ManipRep/Outputs/gtmed/rawhand_phase_s1_screening_rawhand_point_gtmed_mlp_gtmed_pc_f1_8_pn_seed7/metrics.json)
- [VLA vanilla](/home/zc/HandMotion/ManipRep/Outputs/vanilla/rawhand_phase1_vla_iso_shared_vanilla_seed7/metrics.json)
- [VLA GTMed vec_f1_8](/home/zc/HandMotion/ManipRep/Outputs/gtmed/rawhand_phase1_vla_iso_gtmed_vec_f1_8_seed7/metrics.json)
- [VLA GTMed pc_f1_8_pn](/home/zc/HandMotion/ManipRep/Outputs/gtmed/rawhand_phase1_vla_iso_gtmed_pc_f1_8_pn_seed7/metrics.json)

下面统一使用 `metrics.json` 里的 `best_val_action_loss` 作为主指标；`RMSE` 为其开方后的直观量。

| family | run | best val action loss | derived RMSE | vs same-family vanilla |
| --- | --- | ---: | ---: | ---: |
| `MLP` | `vanilla` | `0.000744377` | `0.027283` | `0%` |
| `MLP` | `GTMed vec_f1_8` | `0.000755216` | `0.027481` | `+1.46%` |
| `MLP` | `GTMed pc_f1_8_pn` | `0.000745698` | `0.027307` | `+0.18%` |
| `VLA` | `vanilla` | `0.000744832` | `0.027292` | `0%` |
| `VLA` | `GTMed vec_f1_8` | `0.000719382` | `0.026821` | `-3.42%` |
| `VLA` | `GTMed pc_f1_8_pn` | `0.000722312` | `0.026876` | `-3.02%` |

说明：

- 这里的相对增益按 `best_val_action_loss` 计算。
- `VLA vanilla` 和 `MLP vanilla` 几乎一样，差异只有 `+0.06%`，可以视为 **纯换 backbone 并没有自然抬高 baseline**。
- 但 `GTMed` 在 VLA 下开始显著优于 `Vanilla-VLA`，这和 MLP 阶段形成了清楚反差。

## 核心结论

结论可以直接下：

**`minimal_mlp` 的容量/架构确实是此前 Phase 1 没有观察到触觉收益的重要原因。**

证据链很完整：

1. 在 `MLP` 下，`GTMed vec_f1_8` 和 `GTMed pc_f1_8_pn` 都没有超过 `Vanilla`。
2. 在 `VLA` 下，`Vanilla` 本身几乎没有提升。
3. 但在 `VLA` 下，两个 `GTMed` 候选都稳定优于 `Vanilla-VLA`，而且幅度达到 `3%` 量级。

这说明：

- 不是 “VLA 更强所以所有方法都更好”
- 而是 **VLA 更有能力利用 med 条件信息**

这正是你想验证的点。

## 表示层面的判断

在当前单 seed、`task=false` 的 VLA 结果里：

- `vec_f1_8` 略优于 `pc_f1_8_pn`
- 但差距很小，`0.000719382` vs `0.000722312`

因此现在**不能**草率宣布 `vector` 已经最终胜出，只能说：

- `vec_f1_8` 是当前的名义第一
- `pc_f1_8_pn` 仍然是强竞争者，应该继续保留

这两个都应该进入下一阶段。

## 训练曲线解读

从逐 epoch 结果看：

- `Vanilla-VLA` 在 `epoch 2` 后基本进入平台，并有轻微回摆
- `GTMed-VLA + vec_f1_8` 基本持续下降，到 `epoch 5` 取得最佳
- `GTMed-VLA + pc_f1_8_pn` 在 `epoch 2` 就达到最佳，后面略回升

这意味着：

- `vec_f1_8` 目前更稳
- `pc_f1_8_pn` 更敏感，后续若继续用这条线，最好显式做 early-stop 口径，或者先修 `best checkpoint` 保存逻辑

## 需要注意的技术问题

在这份结论对应的原始分析时，仓库里有两个会影响评测口径的问题：

1. `Vanilla` 单独 `val` 评测会在 [Training/VanillaTrain/build.py](/home/zc/HandMotion/ManipRep/Training/VanillaTrain/build.py:33) 触发 `KeyError: 'train'`
2. [Training/engine.py](/home/zc/HandMotion/ManipRep/Training/engine.py:71) 保存 `best.pt` 时没有深拷贝 `state_dict()`，所以 `best.pt` 可能实际接近最后一轮参数

这两个问题现在都已经修复。  
因此这份历史结论仍然刻意**不依赖** 当时 `evaluate_checkpoint.py` 的 `best.pt` 后处理结果，而是直接用 `metrics.json` 的 `best_val_action_loss`。

## 新的方向判断

原来的 MLP 结论是：

- 触觉表示还没真正打赢 vanilla

现在这个判断应当更新为：

- **在更有表达能力的 backbone 上，触觉中间表示已经证明自己是有用的**

因此后续方向不该再是“继续怀疑 med 本身没用”，而应该是：

1. 保留 `vec_f1_8`
2. 保留 `pc_f1_8_pn`
3. 把后续 `GenMed` 和 `pipeline` 验证切到 **VLA 主线**

## 我建议的下一步

下一阶段最合理的是：

- `GenMed-VLA + vec_f1_8`
- `GenMed-VLA + pc_f1_8`
- 再分别接到对应的 `GTMed-VLA` 做 pipeline 验证

如果你只想先推进一条主线：

- 优先 `vec_f1_8`

原因不是因为它已经绝对胜出，而是：

- 当前单 seed 下它略优
- 曲线更稳
- 最优点出现在最后一轮，不像 `pc_f1_8_pn` 那样依赖早停

但如果资源允许，**最佳做法仍是两条都保留**，不要过早砍掉 point-cloud。
