# Remote wrapper catalog

Use this reference when the dedicated wrapper or effect class is not already
clear. Wrapper `--help` remains authoritative for current arguments; consult it
when needed under the entrypoint's reuse rule. Read only the matching
`codex_remote/README.md` section when more context is needed.

## Effect classes

| Class | Examples | Gate |
| --- | --- | --- |
| Read-only | identity, environment, USB/device inventory, ports, processes, logs, camera listing | Run when relevant through the narrow wrapper |
| Non-motion write | explicit sync, result pull, debug bundle, camera snapshot | State the target and exact write scope first |
| Process execution | Python, module, pytest, preview or bounded service | State process, timeout, and cleanup expectations first |
| Device/host state | USB reset, udev reload, enable/disable, controlled kill, zero-velocity stop | Require explicit current-request authority and fresh safety preflight |
| Real motion/system control | homing, replay, trajectory, servo, gripper, force control, reboot, shutdown, firmware or bus reset | Require the exact action explicitly; never infer permission from a broader task |

Classify an unlisted or ambiguous operation as the highest plausible risk.
Target/scope statements may reference the bounded batch defined in `SKILL.md`;
this does not extend its authority or replace fresh checks.

## Wrapper selection

- Identity and environment: `remote_whoami.sh`, `remote_env.sh`.
- Passive host/device checks: `remote_lsusb.sh`, `remote_dev_nodes.sh`,
  `remote_groups.sh`, `remote_dmesg_usb.sh`, `remote_ps.sh`.
- Passive camera/arm connectivity: `remote_camera_list.sh`,
  `remote_opencv_camera_check.sh`, `remote_arm_ping.sh`,
  `remote_arm_port_check.sh`.
- Camera capture and retrieval: `remote_camera_snapshot.sh`; declare its local
  output path before running it.
- Frozen RGB-D/qpos evidence: `remote_rm65_grasp_snapshot.sh`; requires explicit
  device-read authorization and a new output directory. It does not issue
  motion, HOLD, STOP, gripper commands or camera reset.
- Read-only Agent acceptance: `remote_readonly_agent_acceptance.sh`; requires
  explicit current device-read authorization. Uses fixed camera-only
  `real-perception --camera-single-start` or TCP-only `real-tcp-observe` and
  the existing RPent CLI; stdin credentials only, 300s budget. No motion,
  gripper, controller, HOLD, STOP or camera reset authority.
- `remote_rm65_left_udp_destination.sh`, `remote_rm65_udp_force_coordinate.sh`:
  preflight by default; `--apply` requires exact controller/config authority.
- `remote_camera_v4l2_pair.sh`: bounded capture requires exact serial/start
  authority and new output. Verify child exits; no production readiness claim.
- `remote_camera_driver_bind.sh`: inspect by default; `--bind` requires exact
  port/serial authority, fresh idle checks; only unbound VideoControl binding.
  These wrappers grant no motion/reset authority. Read matching README scopes,
  limits and recovery rules before use.
- Offline frozen grasp inference: `remote_grasp_snapshot_inference.sh`; runs
  only the fixed inference probe on retained evidence, without device access.
- Inspection: `remote_view.sh`, `remote_tail_log.sh`,
  `remote_sync_check.sh`.
- Remote execution: `remote_python.sh`, `remote_module.sh`,
  `remote_pytest.sh`.
- Explicit transfer: `remote_pull.sh`, `remote_rsync_project.sh`,
  `remote_collect_debug_bundle.sh`.
- Protected Harness execution: `remote_harness.sh`, limited to authenticated
  `runner preflight` and `transfer exchange-run`; its availability grants no
  request, device, motion, or arbitrary-command authority.
- Protected Harness deployment: `remote_harness_deploy.sh`, only for an exact
  commit explicitly authorized by the deployment owner and a fresh Harness
  plan; it grants no project command, device, or motion authority.
- Before either protected Harness wrapper, read `codex_remote/README.md` section
  `3.1 远端受保护 Harness`. Resolve site identity, exchange roots, and key paths
  from the host-private configuration; do not guess them or add them to project
  source.
- Dry-run arm checks: `remote_arm_dry_run.sh`; do not disable its dry-run
  behavior.
- Control-risk preflight evidence: `remote_safety_check.sh`. It is read-only
  and does not authorize a later action.

Use dataset-specific wrappers only for the exact documented dataset audit,
quarantine, promotion, or storage task. Do not reinterpret them as arbitrary
remote shell access.

## Target and sync rules

- Pass the user-selected host or alias unchanged across identity, environment,
  sync, execution, inspection, and retrieval.
- If the user provides no target, resolve the project's configured default alias
  and still pass it explicitly; do not copy its current IP into durable
  instructions.
- When no-delete realtime sync is confirmed active, do not perform routine
  one-shot project sync. Edit locally and verify explicit paths if stale code is
  suspected.
- Inspect remote artifacts in place or retrieve them explicitly. Never assume
  runtime outputs are synchronized back.
