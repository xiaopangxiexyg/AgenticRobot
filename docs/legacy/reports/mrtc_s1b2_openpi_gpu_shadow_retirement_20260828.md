# MRTC-S1b2 synthetic OpenPI shadow 探针退役说明（2026-08-28）

## 结论

在精确 `development` 基线
`8eabcd47ab2a190f655d2eb1b548c3193f7fc32f` 上，
`scripts/probes/mrtc_s1b2_openpi_gpu_shadow.py` 是 479 行、16,541 字节的一次性
synthetic-observation/no-action 验证入口。它只启动本地主机 OpenPI server，执行
conditioned wire 与 bounded worker/timed-plan frontend 检查，不初始化相机、目标、机械臂、
安全、位置控制或 dispatch。

该探针已被后续正式 RTC client/frontend、deploy runtime 及注册测试覆盖，且没有生产入口、
测试、Profile 或命令映射调用。因此删除版本化源码入口；原始数字和边界继续保留在
[MRTC-S1b2 evidence](../../architecture/evidence/rtc/mrtc_s1b2_openpi_gpu_wire_frontend_shadow_evidence_v1.json)。

## 调用与替代复核

全树静态复核只发现 evidence JSON 中的历史路径记录，没有 Python import 或运行时调用。
当前能力由以下正式路径拥有：

- `deployments.client.rtc_frontend.build_rtc_conditioned_model_shadow_frontend_v3`：frontend 组合；
- `deployments.openpi.client.rtc_conditioned.OpenPIConditionedRtcBackendV3`：conditioned wire backend；
- `teleop_runtime.deploy.run_policy`：部署入口的 RTC profile 接线；
- `tests/test_deploy_rtc_frontend_shadow.py` 及 OpenPI/RTC contract tests：可重复的本地验证。

Evidence JSON 已将 `probe` 标记为 retired，并指向本说明；不再引用被忽略的 `todos/` 路径。

## 历史与安全边界

- 2026-07-27 的 provider warmup、wire/frontend receipt、27 次 emission、零 action send 等
  原始结果不变；
- evidence 中的 `real_observation_or_target_execution_proven=false`、
  `production_profile_enabled=false` 和 `motion_readiness_proven=false` 继续有效；
- 如需逐行复现旧实现，应从 Git 精确提交离线读取，而不是恢复 tracked 入口；
- 根 `legacy/` 仍严格本地、完全忽略，本次不写入任何 tracked 代码副本；
- 本次不连接设备、不启动真实相机、不发送动作、不运行硬件或科学重新认证。
