# Connection Codex Agent Rules

These are agent instructions, not a full user manual. Apply them only when the
Codex session current directory is
`/home/zc/Release/util_scripts/Connection/codex` or a child directory. If they
are loaded during unrelated local-only work, do not let them steer that task.

## Hard Rules

- Keep Codex local. Treat remote machines only as SSH-reachable execution and
  hardware hosts.
- Use the wrappers in this directory for remote work. Do not handwrite SSH
  commands for remote writes, process control, hardware access, robot control,
  USB reset, reboot/shutdown, or motion.
- Do not modify wrapper behavior, shared safety logic, or safety policy
  documents by default.
- Only modify wrappers or safety policy when the user explicitly asks to modify
  this wrapper set. Before doing so, read `README.md`, state the intended safety
  impact, keep the edit narrowly scoped, and run relevant verification.
- Do not store passwords, tokens, private keys, host secrets, or credentials in
  this directory.
- Do not assume `Agent (Full Access)`. Respect the active sandbox and approval
  mode.

## Sync Model

- When the user states that realtime sync is active via
  `/home/zc/Release/util_scripts/Connection/realtime_sync_ssh.sh` in no-delete
  mode, treat the local directory as the only code editing surface.
- Edit source code locally only. Do not edit remote source files with SSH, sed,
  vim, Python snippets, or ad hoc shell commands.
- Do not run `remote_rsync_project.sh` for routine edits while realtime sync is
  active, unless the user explicitly requests a one-shot sync.
- Run tests and runtime commands on the remote host when they need remote
  hardware, drivers, devices, GPUs, robot networks, or the remote Python
  environment.
- Remote runtime outputs are not assumed to sync back. Inspect them with
  `remote_view.sh` or `remote_tail_log.sh`, or pull them intentionally with
  `remote_pull.sh` or `remote_collect_debug_bundle.sh`.
- If sync health matters or stale remote code is suspected, verify explicit
  files/directories with `remote_sync_check.sh`.

## Safety Levels

- If an operation is not listed or cannot be confidently classified, treat it
  as L5 and stop for explicit user direction.
- L0/L1: local or remote read-only diagnostics; may run when relevant.
- L2: non-motion writes such as pulling remote results, debug bundles, camera
  snapshots, or explicit one-shot sync; first state target and write scope.
- L3: remote process execution/control such as Python, modules, pytest, preview
  services, or controlled sessions; first state target process and stop/timeout
  expectations.
- L4: device or host state changes without planned motion, including USB reset,
  udev reload, enable/disable, controlled kill, or stop/zero-velocity commands;
  never run automatically.
- L5: real motion or system-level action, including homing, trajectory/replay,
  servo, gripper motion, force control, reboot, shutdown, firmware flashing,
  CAN bus reset, or unknown robot-control actions; never run automatically.

## Required Wrappers

- Read-only checks: `remote_whoami.sh`, `remote_env.sh`, `remote_lsusb.sh`,
  `remote_dev_nodes.sh`, `remote_groups.sh`, `remote_dmesg_usb.sh`,
  `remote_camera_list.sh`, `remote_arm_ping.sh`, `remote_arm_port_check.sh`,
  `remote_ps.sh`, `remote_view.sh`, `remote_tail_log.sh`,
  `remote_sync_check.sh`.
- Remote Python/test execution: `remote_python.sh`, `remote_module.sh`,
  `remote_pytest.sh`. Do not bypass their control-risk checks.
- Arm dry-run work: `remote_arm_dry_run.sh`. Do not pass flags that disable
  dry-run behavior.
- Intentional result retrieval: `remote_pull.sh` or
  `remote_collect_debug_bundle.sh`.

## Real Control

- Do not execute real motion, homing, enable/disable, replay, trajectory,
  servo, gripper, force-control, emergency-stop, or other L4/L5 actions unless
  the user explicitly requests that specific action in the current turn.
- Before any L4/L5 or control-risk command, require a fresh
  `remote_safety_check.sh` with relevant `--arm-ip`, `--arm-port`, and/or
  `--require-device` checks.
- Do not add `--allow-control-risk` just to make a generic Python command run.
  If a command is blocked by control-risk checks, stop and explain what was
  blocked.
- If the user explicitly confirms a control-risk command, require
  `--require-safety-ok TOKEN_PATH` and keep `--safety-max-age` short.
