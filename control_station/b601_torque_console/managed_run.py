"""Bounded process management for the existing owner, never a CAN sender."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .managed_cycle import assess_managed_result

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT/'scripts/hardware/rebot_b601_rs_owner.py'
PROFILE = ROOT/'docs/hardware/profiles/b601_j2_j3_near_zero_command_field_v1.json'
PROFILES = {'near-zero': PROFILE,
    'four-centres': ROOT/'docs/hardware/profiles/b601_j2_j3_four_centres_cruise_identification_v1.json'}
OWNER_NAMES = {'rebot_b601_rs_owner.py', 'rebot_b601_rs_low_gain_probe.py'}
urlopen = build_opener(ProxyHandler({})).open


def command_for(args, run):
    command = [sys.executable, str(OWNER), '--config', str(args.config),
        '--rebot-source-root', str(args.rebot_source_root), '--motorbridge-root', str(args.motorbridge_root),
        '--output-root', str(run/'owner'), '--mode', 'hold', '--duration-sec', '4000',
        '--rate-hz', '100', '--feedback-source', 'active-report', '--driver-kd', '0', '0', '0', '0', '0', '0',
        '--torque-console-auto-calibration-profile', str(PROFILES[args.recipe]), '--torque-console-managed-cycle',
        '--torque-console-max-assist-nm', '1', '--torque-console-max-joint-velocity-rad-s', '.195',
        '--torque-console-max-excursion-rad', '.15', '--torque-console-max-support-drift-rad', '.05235987755982989',
        '--torque-console-staged-target-offset-limit-rad', '.09', '--torque-console-staged-breakaway-torque-nm', '1',
        '--torque-console-staged-breakaway-rise-slew-nm-s', '1', '--torque-console-staged-breakaway-duration-sec', '3',
        '--torque-console-staged-breakaway-impulse-nm-sec', '1.5', '--torque-console-staged-tracking-overlay-cap-nm', '1',
        '--torque-console-confirmation', 'B601_RS_TORQUE_CONSOLE', '--allow-motion', '--gzip-dynamics-trace']
    if args.field_report is not None:
        command += ['--torque-console-command-field-report', str(args.field_report),
                    '--torque-console-command-field-mode', 'shadow']
    return command


def check_revision(expected):
    if not re.fullmatch('[0-9a-f]{40}', expected or ''):
        raise ValueError('an exact expected revision is required')
    def git(*values):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), *values],
            text=True, timeout=10).strip()
    if git('rev-parse', 'HEAD') != expected or git('status', '--porcelain=v1'):
        raise ValueError('managed run requires the exact clean candidate')


def check_idle():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            names = {Path(arg).name for arg in path.read_bytes().decode(errors='replace').split('\0') if arg}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if names & OWNER_NAMES:
            raise ValueError(f'an existing B601 owner requires resolution: pid={path.parent.name}')


def process_start(pid):
    try:
        # comm may contain spaces or parentheses; fields after its final ')'
        # start at field 3, and starttime is field 22.
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rindex(')')+2:].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_matches(receipt):
    pid, command, start = receipt.get('pid'), receipt.get('command'), receipt.get('start_ticks')
    if (type(pid) is not int or pid <= 0 or not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or command[:2] != [sys.executable, str(OWNER)]
        or '--torque-console-managed-cycle' not in command
        or start is not None and (not isinstance(start, str) or not start.isdigit())):
        raise ValueError('invalid managed process receipt')
    if start is None or process_start(pid) != start:
        return False
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return argv == command


def _write_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args, run):
    check_revision(args.expected_revision)
    check_idle()
    command = command_for(args, run)
    _write_new(run/'plan.json', {'revision': args.expected_revision, 'project': str(ROOT), 'operation': args.operation,
        'command': command, 'recipe': args.recipe,
        'profile_sha256': hashlib.sha256(PROFILES[args.recipe].read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'max_cycles': 1, 'automatic_retry': False})
    with (run/'preflight.log').open('x') as output:
        result = subprocess.run([*command, '--preflight-only'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, timeout=60, check=False)
    paths = list((run/'owner').glob('b601-owner-*/b601_owner.json'))
    if result.returncode or len(paths) != 1:
        raise ValueError(f'owner preflight failed; see {run}/preflight.log')
    preflight = json.loads(paths[0].read_text())
    if (preflight.get('status') != 'PASS' or preflight.get('can_opened') is not False
        or preflight.get('motor_command_sent') is not False or preflight.get('managed_cycle_enabled') is not True):
        raise ValueError('owner did not confirm the exact no-CAN managed preflight')
    return command, preflight


def check_channel(channel):
    if not isinstance(channel, str) or not re.fullmatch('[a-zA-Z0-9_-]{1,15}', channel):
        raise ValueError('managed run requires a named SocketCAN interface')
    socket.if_nametoindex(channel)
    links = json.loads(subprocess.check_output(['ip', '-j', '-details', 'link', 'show', 'dev', channel],
        text=True, timeout=5))
    if len(links) != 1 or 'UP' not in links[0].get('flags', []):
        raise ValueError('CAN interface is down; no automatic reset is attempted')
    state = links[0].get('linkinfo', {}).get('info_data', {}).get('state')
    if state != 'ERROR-ACTIVE':
        raise ValueError(f'CAN interface cannot start a cycle: {state}')


def launch(args):
    output = args.output_root.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('runtime output must be outside the source checkout')
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='managed-', dir=output))
    # Keep the lock in the parent owner until its child has completed. The
    # canonical receiver/dispatch owner remains the existing probe process.
    lock_root = Path(tempfile.gettempdir())/f'b601-managed-{os.getuid()}'
    lock_root.mkdir(mode=0o700, exist_ok=True)
    if lock_root.stat().st_uid != os.getuid() or lock_root.stat().st_mode & 0o077:
        raise ValueError('managed lock directory must be private to this user')
    with (lock_root/'six-axis-owner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another managed B601 owner still holds the device lease') from exc
        command, preflight = prepare(args, run)
        if args.operation == 'preflight':
            return {'status': 'PREFLIGHT_PASS', 'run_directory': str(run), 'can_opened': False}
        check_channel(preflight['channel'])
        check_idle()
        check_revision(args.expected_revision)
        with (run/'management.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        receipt = {'pid': process.pid, 'start_ticks': process_start(process.pid), 'command': command,
                   'revision': args.expected_revision, 'run_directory': str(run)}
        _write_new(run/'process.json', receipt)
    return {'status': 'STARTED_NOT_YET_ACCEPTED', **receipt}


def inspect(run):
    run = run.resolve()
    process_file = run/'process.json'
    process = json.loads(process_file.read_text()) if process_file.exists() else None
    alive = process is not None and process_matches(process)
    results = sorted((run/'owner').glob('b601-owner-*/b601_owner.json'))
    # The preflight result is retained independently from the actual run.
    payload = report = None
    result_path = None
    for path in results:
        data = json.loads(path.read_text())
        if not data.get('preflight_only'):
            if payload is not None:
                raise ValueError('ambiguous owner result: expected one active cycle')
            payload, result_path = data, path
    if result_path is not None:
        report_path = result_path.parent/'automatic_calibration_report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
    assessment = None if payload is None else assess_managed_result(payload, report)
    state = ('RUNNING' if payload is None else 'POSTPROCESSING') if alive else (
        'STOPPED_WITHOUT_CONFIRMED_CLEANUP' if payload is None else assessment['status'])
    live, live_error = None, None
    if alive and payload is None:
        try:
            _, snapshot = console_state(run, process)
        except (OSError, ValueError) as exc:
            live_error = str(exc)
        else:
            batch = snapshot.get('automatic_calibration', {})
            workspace = batch.get('workspace', {})
            holding = snapshot.get('coordinated_holding') or {}
            live = {'managed_cycle': snapshot.get('managed_cycle'), 'fault': snapshot.get('fault'),
                'owner_ready': snapshot.get('owner_ready'),
                'state': batch.get('state'), 'reason': batch.get('reason'),
                'current_task': workspace.get('current_task'),
                'continuous_identification': workspace.get('continuous_identification'),
                'arrival': workspace.get('arrival'),
                'motion_control': snapshot.get('motion_control'),
                'cruise_velocity_control': holding.get('cruise_velocity_control'),
                'command_authority': {k: holding.get(k) for k in (
                    'bias_nm', 'proposed_bias_nm', 'combined_command_nm', 'combined_l1_cap_nm',
                    'combined_cap_shortfall_nm', 'saturated', 'composed_speed_scales', 'quantity')}
                    if holding else None,
                'supported_return': workspace.get('supported_return'), 'qpos_rad': snapshot.get('qpos_rad')}
            phase = (live['managed_cycle'] or {}).get('phase')
            if phase == 'origin_confirmed':
                state = 'DISABLE_PENDING'
            elif (live['fault'] or phase in {'operator_required', 'cancelled', 'stopped'}
                  or phase == 'running' and live['owner_ready'] is not True):
                state = 'OPERATOR_REQUIRED'
    return {'run_directory': str(run), 'launcher_running': alive, 'status': state,
        'result_path': None if result_path is None else str(result_path),
        'assessment': assessment, 'process': process, 'live': live, 'live_status_error': live_error}


def console_state(run, process):
    # Only the exact run's local child endpoint is eligible. Never send a
    # second SIGINT or infer permission to disable from a stale PID.
    # The detached launcher's stdout may still be buffered in an older run.
    # The child's own flushed runtime log is the canonical endpoint record.
    logs = list((run/'owner').glob('b601-owner-*/b601_owner.log'))
    if not logs:
        logs = [run/'management.log']
    urls = {url for path in logs for url in
            re.findall(r'http://127\.0\.0\.1:[0-9]+', path.read_text())}
    if not urls:
        raise ValueError('owner console endpoint is not available yet')
    if len(urls) != 1:
        raise ValueError('ambiguous owner console endpoints in this run')
    url = urls.pop()
    with urlopen(url+'/api/status?view=operator', timeout=2) as response:
        status = json.load(response)
    owner_pid = status.get('owner_pid')
    if type(owner_pid) is not int or os.getpgid(owner_pid) != process['pid']:
        raise ValueError('console is not bound to the exact managed process group')
    if not Path(status.get('runtime_log_path') or '/').resolve().is_relative_to(run.resolve()):
        raise ValueError('console belongs to a different run')
    return url, status


def request_protective_hold(run):
    return _request_stop(run, full_disable=False)


def request_supported_disable(run):
    """Explicit operator-confirmed physical support; never called by the cycle."""
    return _request_stop(run, full_disable=True)


def _request_stop(run, *, full_disable):
    state = inspect(run)
    if not state['launcher_running']:
        raise ValueError('the exact managed launcher is no longer running')
    url, snapshot = console_state(run, state['process'])
    if full_disable and (snapshot.get('managed_cycle') or {}).get('phase') not in {
        'stopped', 'operator_required', 'cancelled', 'origin_confirmed',
    }:
        raise ValueError('supported disable requires a stopped batch')
    with urlopen(url, timeout=2) as response:
        page = response.read().decode()
    token = re.search(r'<meta name="csrf-token" content="([A-Za-z0-9_-]+)">', page)
    if token is None:
        raise ValueError('local console token is unavailable')
    action = 'full_disable' if full_disable else 'protective_hold'
    request = Request(url+'/api/stop', method='POST', data=json.dumps({'action':action}).encode(),
        headers={'Content-Type': 'application/json', 'Origin': url, 'X-B601-Console-Token': token[1]})
    if not process_matches(state['process']):
        raise ValueError('managed process identity changed before the stop request')
    with urlopen(request, timeout=2) as response:
        json.load(response)
    return {'status': 'SUPPORTED_FULL_DISABLE_REQUESTED' if full_disable else 'PROTECTIVE_HOLD_REQUESTED',
            'run_directory': str(run), 'cleanup_confirmed': False,
            'full_disable_requested': full_disable, 'automatic_retry': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('preflight', 'start', 'status', 'stop', 'supported-disable'), default='preflight')
    parser.add_argument('--recipe', choices=tuple(PROFILES), default='near-zero')
    parser.add_argument('--expected-revision')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--rebot-source-root', type=Path)
    parser.add_argument('--motorbridge-root', type=Path)
    parser.add_argument('--field-report', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-directory', type=Path)
    args = parser.parse_args(argv)
    if args.operation in {'status', 'stop', 'supported-disable'}:
        if args.run_directory is None:
            parser.error('status/stop/supported-disable require --run-directory')
        operation = {'status': inspect, 'stop': request_protective_hold, 'supported-disable': request_supported_disable}
        result = operation[args.operation](args.run_directory)
    else:
        if any(getattr(args, k) is None for k in ('config', 'rebot_source_root', 'motorbridge_root', 'output_root')):
            parser.error('preflight/start require config, rebot-source-root, motorbridge-root and output-root')
        result = launch(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
