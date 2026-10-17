"""Bounded perception or TCP-only acceptance using canonical Gateway/RPent CLIs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "rpent-repo", "rpent-python", "output"):
        value.add_argument("--" + name, type=Path, required=True)
    for name in ("camera-config", "camera-registry", "localization-bundle", "arm-ip-assignment",
                 "left-tool-profile", "right-tool-profile", "envelope-profile"):
        value.add_argument("--" + name, type=Path)
    value.add_argument("--acceptance-scope", choices=("perception", "tcp"), default="perception")
    value.add_argument("--check-only", action="store_true")
    value.add_argument("--camera-sdk-debug-log", action="store_true",
        help="retain native SDK DEBUG log and Gateway startup context in this new run")
    value.add_argument("--hf-home", type=Path, help="existing offline Hugging Face cache root; no download")
    value.add_argument("--camera-scope", choices=("all", "main"), default="all",
        help="explicit camera inventory; main starts only the configured front RGB-D camera")
    return value


def select_camera_config(path, scope):
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("Camera configuration must be an object")
    if scope == "main":
        if not isinstance(payload.get("front"), dict):
            raise ValueError("Main-only acceptance requires the configured front camera")
        return {"front": payload["front"]}
    return payload


def wait_ready(process, log_path, timeout=60, *, required_roles=("main", "wrist_left", "wrist_right"), tcp=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                event = json.loads(line)
                if event.get("event") == "gateway_ready":
                    if tcp:
                        if (event.get("robot_backend") != "readonly_sdk_sampler"
                            or event.get("current_mode") != "real_observe"
                            or event.get("command_owner_count") != 0
                            or event.get("motion_enabled") is not False
                            or event.get("gripper_enabled") is not False
                            or set(event.get("component_ids", ())) != {"left_arm", "right_arm"}):
                            raise RuntimeError("Expected a dual-arm TCP-only read-only Gateway")
                        return event
                    if event.get("robot_backend") != "absent" or event.get("current_mode") != "real_observe":
                        raise RuntimeError("Expected a camera-only real_observe Gateway")
                    if set(event.get("rgb_roles", ())) != set(required_roles):
                        raise RuntimeError("Camera inventory differs from the explicitly selected scope")
                    return event
        if process.poll() is not None:
            raise RuntimeError(f"Gateway exited before ready: {process.returncode}; inspect {log_path}")
        time.sleep(0.1)
    raise TimeoutError("Camera-only Gateway did not become ready within 60s")


def _close_owned_children(children):
    """Confirm owned child exits within a shared budget below outer KILL grace."""
    errors = []
    owned = list(reversed(children))
    # main records cleanup cancellation; process API errors cannot skip siblings.
    for child in owned:
        if child.poll() is None:
            try:
                child.terminate()
            except OSError as error:
                errors.append({"pid": child.pid, "error": "child_TERM_failed", "error_type": type(error).__name__})
    deadline = time.monotonic() + 70.0
    for child in owned:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass  # Escalate only still-live owned children below.
            except OSError as error:
                errors.append({"pid": child.pid, "error": "child_TERM_wait_failed", "error_type": type(error).__name__})
    for child in owned:
        if child.poll() is None:
            try:
                child.kill()
            except OSError as error:
                errors.append({"pid": child.pid, "error": "child_KILL_failed", "error_type": type(error).__name__})
    deadline = time.monotonic() + 10.0
    for child in owned:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                errors.append({"pid": child.pid, "error": "child_did_not_exit_after_KILL"})
            except OSError as error:
                errors.append({"pid": child.pid, "error": "child_KILL_wait_failed", "error_type": type(error).__name__})
    exit_codes = {str(child.pid): child.poll() for child in children}
    for child in children:
        code = exit_codes[str(child.pid)]
        if code != 0 and not any(row["pid"] == child.pid for row in errors):
            errors.append({"pid": child.pid, "error": "child_exit_not_clean", "exit_code": code})
    return errors, exit_codes


def main(argv=None):
    args = parser().parse_args(argv)
    for name, path in vars(args).items():
        if isinstance(path, Path) and (not path.is_absolute() or path == Path("/")):
            raise ValueError(f"{name} must be an explicit absolute path")
    tcp = args.acceptance_scope == "tcp"
    scope_paths = (("arm_ip_assignment", "left_tool_profile", "right_tool_profile", "envelope_profile")
                   if tcp else ("camera_config", "camera_registry", "localization_bundle"))
    for name in ("repo", "python", "rpent_repo", "rpent_python", *scope_paths):
        if getattr(args, name) is None:
            raise ValueError(f"{name} is required for {args.acceptance_scope} acceptance")
        if not getattr(args, name).exists():
            raise FileNotFoundError(getattr(args, name))
    if args.hf_home is not None and not args.hf_home.is_dir():
        raise NotADirectoryError(args.hf_home)
    selected_cameras = None if tcp else select_camera_config(args.camera_config, args.camera_scope)
    required_roles = () if tcp else (("main",) if args.camera_scope == "main" else ("main", "wrist_left", "wrist_right"))
    gateway_command = [str(args.python), "-u", "-m", "agent_gateway", "real-perception",
        "--socket-path", str(args.output / "gateway.sock"), "--artifact-root", str(args.output / "artifacts"),
        "--camera-config", str(args.camera_config), "--camera-registry", str(args.camera_registry),
        "--camera-single-start", "--client-id", "rpent.readonly.acceptance", "--client-uid", str(os.getuid()),
        "--enable-object-localization", "--object-localization-asset-bundle", str(args.localization_bundle)]
    if args.camera_sdk_debug_log:
        gateway_command.extend(("--camera-sdk-debug-log", str(args.output / "librealsense.debug.log")))
    if tcp:
        if args.camera_sdk_debug_log or args.camera_scope != "all":
            raise ValueError("TCP-only acceptance cannot select or start cameras")
        gateway_command = [str(args.python), "-u", "-m", "agent_gateway", "real-tcp-observe",
            "--socket-path", str(args.output / "gateway.sock"),
            "--client-id", "rpent.readonly.acceptance", "--client-uid", str(os.getuid())]
        for name in scope_paths:
            gateway_command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    upper_env = dict(os.environ, PYTHONPATH=f"{args.rpent_repo}:{args.rpent_repo}/integrations/agent_gateway/src",
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        RPENT_DEEPSEEK_THINKING="enabled")
    lower_env = dict(os.environ, PYTHONPATH=str(args.repo), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # The read-only Gateway has no planner and must not inherit its credential.
    lower_env.pop("OPENAI_API_KEY", None)
    if args.hf_home is not None:
        lower_env["HF_HOME"] = str(args.hf_home)
    if args.check_only:
        for command, cwd, env in (
            ([str(args.python), "-m", "agent_gateway", "real-tcp-observe" if tcp else "real-perception", "--help"], args.repo, lower_env),
            ([str(args.rpent_python.parent / "rpent"), "--robot", "agentic_robot", "--help"], args.rpent_repo, upper_env),
        ):
            subprocess.run(command, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL, timeout=20)
        print(json.dumps({"check_only": True, "devices_started": False, "commands_valid": True}))
        return 0

    # Credentials arrive on SSH stdin, never in argv, source, or output files.
    secret = sys.stdin.readline().strip()
    if not secret:
        raise ValueError("API credential is required on stdin")
    upper_env["OPENAI_API_KEY"] = secret
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output / "artifacts").mkdir(mode=0o700)
    (args.output / "memory").mkdir(mode=0o700)
    if not tcp and args.camera_scope == "main":
        effective_config = args.output / "selected-camera-config.json"
        effective_config.write_text(json.dumps(selected_cameras, ensure_ascii=False, indent=2))
        gateway_command[gateway_command.index("--camera-config") + 1] = str(effective_config)
    report = {"schema": "agenticrobot.readonly_agent_acceptance.v1", "status": "failed",
        "robot_backend": "readonly_sdk_sampler" if tcp else "absent", "motion_enabled": False, "gripper_enabled": False,
        "camera_single_start": not tcp, "output": str(args.output), "acceptance_scope": args.acceptance_scope}
    report["hf_home"] = lower_env.get("HF_HOME")
    report["camera_scope"] = "none" if tcp else args.camera_scope
    report["camera_sdk_debug_log"] = "librealsense.debug.log" if args.camera_sdk_debug_log else None
    report["required_rgb_roles"] = list(required_roles)
    children = []
    started = time.monotonic()

    cleanup_phase = False
    cleanup_stop_signals = []

    def interrupted(signum, _frame):
        if cleanup_phase:
            # Record cancellation without interrupting owned-child confirmation.
            cleanup_stop_signals.append(signum)
            return
        raise InterruptedError(f"Acceptance interrupted by signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name, repo in (("gateway_commit", args.repo), ("rpent_commit", args.rpent_repo)):
            report[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        with (args.output / "gateway.console.log").open("w") as gateway_log:
            gateway = subprocess.Popen(gateway_command, cwd=args.repo, env=lower_env,
                stdin=subprocess.DEVNULL, stdout=gateway_log, stderr=subprocess.STDOUT)
            children.append(gateway)
            report["gateway_pid"] = gateway.pid
            ready_options = {"required_roles": required_roles}
            if tcp:
                ready_options["tcp"] = True
            report["gateway_ready"] = wait_ready(gateway, args.output / "gateway.console.log", **ready_options)
            print(json.dumps({"event": "readonly_gateway_ready", "scope": args.acceptance_scope, "pid": gateway.pid}), flush=True)
            attachment = args.output / "perception.json"
            prepare = [str(args.rpent_python), "-m", "robots.agentic_robot.gateway_attachment", "--workflow", "perception",
                "--include-localization", "--socket-path", str(args.output / "gateway.sock"),
                "--client-id", "rpent.readonly.acceptance", "--expected-server-uid", str(os.getuid()),
                "--output", str(attachment)]
            if tcp:
                prepare[prepare.index("--workflow") + 1] = "state"
                prepare.remove("--include-localization")
            with (args.output / "attachment.console.log").open("w") as log:
                subprocess.run(prepare, cwd=args.rpent_repo, env=upper_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=20)
            instruction = (
                "只读验收，不执行任何动作。获取当前公开的所有 RGB 视角，使用640x480输出，并获取主相机 main 的深度。"
                "报告每个真实可用视角的遮挡和可见目标，不根据相机序列号或顺序猜测角色。"
                "从这些图像中选一个清晰可见的小物体，明确选中的图像角色和 ordinal，"
                "使用英文物体描述调用 localize_objects；必须引用这次 capture_rgb 返回的同一张选中视角 Artifact，"
                "不要传本地路径或换图。报告物体、像素定位、深度是否获取成功及来源角色。"
                "无可靠物体则明确报告，不能猜测。不要将深度 Artifact 解释为已得到物体三维或6D位姿。"
                "另外仅做方案推理：假设后续请求右臂上移0.03米再返回，列出还需要读取的状态、"
                "目标参数来源、坐标身份和执行前提。没有 state.read_tcp_pose 就明确缺失，不猜当前TCP、"
                "四元数或目标数值，也不调用运动格式适配器、规划或执行工具。"
                "说明主相机深度不能直接用于另一个视角的检测像素，本轮没有移动机械臂或开闭夹爪。"
                "报告完整时调用 finish(status='completed', summary=报告)，仅表示只读报告完成，不声明物理任务成功。"
                "报告未完成则用 failure/stuck；不要读取本次运行日志或 memory 来寻找 native success。"
            )
            task_tag = "readonly_perception"
            if tcp:
                task_tag = "readonly_tcp"
                instruction = (
                    "只读真机 TCP 验收，不执行任何动作。本会话只有 state.read_tcp_pose。"
                    "先同时读取 left_arm 和 right_arm；随后再分别读取左右臂，报告真实xyz与quat_xyzw、"
                    "测量时间、可用时间、cutoff、新鲜度、参考坐标系和工具/标定身份。"
                    "不要调用图像、运动、夹爪、HOLD或STOP；不要从文件/日志补造测量。"
                    "只做推理：依据第一次右臂位姿，给出在同一参考系+Z方向上移0.03米并返回的目标，"
                    "四元数保持不变。明确当前没有运动控制权且未验证可达性/碰撞，不能执行。"
                    "报告完整时调用 finish(status='completed', summary=报告)，不声明物理任务成功。"
                )
            command = [str(args.rpent_python.parent / "rpent"), "--robot", "agentic_robot", "--session-extension", "agent_gateway",
                "--session-extension-config", str(attachment), "--instruction", instruction, "--task-tag", task_tag,
                "--planner", "api", "--model", "openai-chat:deepseek-flash", "--base-url", "https://api.deepseek.com",
                "--reasoning-effort", "high", "--max-turns", "10", "--max-tokens", "8192", "--planner-timeout-s", "120",
                "--memory-profile", "local", "--memory-dir", str(args.output / "memory"),
                "--output-dir", str(args.output / "agent"), "--save-full-conversation"]
            with (args.output / "agent.console.log").open("w") as agent_log:
                agent = subprocess.Popen(command, cwd=args.rpent_repo, env=upper_env, stdin=subprocess.DEVNULL,
                    stdout=agent_log, stderr=subprocess.STDOUT)
                children.append(agent)
                report["agent_pid"] = agent.pid
                report["agent_exit_code"] = agent.wait(timeout=150)
            if report["agent_exit_code"] != 0:
                raise RuntimeError(f"RPent exited with {report['agent_exit_code']}; inspect agent.console.log")
            transcript = json.loads((args.output / "agent" / f"transcript_{task_tag}.json").read_text())
            report["agent_finish"] = transcript.get("finish")
            report["environment_success"] = transcript.get("environment_success")
            finish = report["agent_finish"]
            if not isinstance(finish, dict) or finish.get("_finish") is not True or finish.get("status") != "completed":
                raise RuntimeError("RPent did not complete its read-only report; inspect agent transcript, not just exit code")
            if report["environment_success"] is not False:
                raise RuntimeError("Read-only acceptance must not assert native physical task success")
            report["status"] = "agent_run_completed_pending_evidence_review"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error).replace(secret, "[redacted]")
    finally:
        cleanup_phase = True
        try:
            cleanup_errors, exit_codes = _close_owned_children(children)
            if cleanup_stop_signals:
                cleanup_errors.append({"error": "cleanup_stop_requested", "signals": list(cleanup_stop_signals)})
            report["child_exit_codes"] = exit_codes
            report["cleanup_complete"] = not cleanup_errors
            report["cleanup_errors"] = cleanup_errors
            report["cleanup_stop_signals"] = cleanup_stop_signals
            report["elapsed_sec"] = round(time.monotonic() - started, 3)
            (args.output / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if report["status"] == "agent_run_completed_pending_evidence_review" and report["cleanup_complete"] and not cleanup_stop_signals else 1


if __name__ == "__main__":
    raise SystemExit(main())
