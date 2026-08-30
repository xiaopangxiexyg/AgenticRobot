# Connection Codex Agent Rules

These are agent instructions, not a full user manual. Apply them to changes
under this directory and to tasks that invoke these wrappers, including calls
made from the repository root. Do not let them steer unrelated local-only work.

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

## Target Selection

- Pass `-H/--host` explicitly on every wrapper invocation. Do not rely on an
  ambient `REMOTE_HOST` value.
- If the user identifies an SSH host, IP, `user@host[:port]`, or SSH alias, use
  that exact target. Otherwise use `-H dex-remote`, the configured default for
  the 192.168.20.170 host.
- Keep one target across a related operation chain. If an explicit target
  fails, stop; never fall back to `dex-remote`.
- Treat a target change as a new remote context. Re-verify identity, project
  path, environment, devices, safety token, and write scope as applicable.
- Distinguish the SSH execution target (`-H/--host`) from robot or controller
  addresses such as `--arm-ip`; ask when an IP is ambiguous.
- Do not infer usernames, project paths, environments, or device layouts from
  an IP. Run `remote_whoami.sh` first on a non-default target before L2/L3 or
  hardware work.

## Sync Model

- When the user states that a confirmed no-delete realtime sync is active, such
  as `connection/realtime_sync_ssh.sh`, treat the local directory as the only
  code editing surface.
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
- Protected Harness execution: `remote_harness.sh`, limited to `runner
  preflight` and authenticated `transfer exchange-run`. It does not grant
  request, device, motion, or arbitrary-command authority.
  Wrapper availability is not transfer readiness: the local controller must
  resolve its sender, recipient, key, exchange root, and exact receiver bases
  from host-private Harness configuration. Never invent or commit those values.
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
