"""Fake process/transport boundaries; these tests cannot launch the B601 owner."""

import json
import inspect
import os
import select
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_station.b601_torque_console import managed_run as managed

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def options(tmp_path, **changes):
    config = tmp_path/'config.yaml'
    config.write_text('channel: can0')
    values = {'config': config, 'rebot_source_root': tmp_path/'vendor', 'motorbridge_root': tmp_path/'bridge',
        'field_report': None, 'expected_revision': 'a'*40, 'output_root': tmp_path/'runs', 'operation': 'preflight',
        'recipe': 'near-zero'}
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('operation', ['preflight', 'start'])
@pytest.mark.parametrize('recipe', ['near-zero', 'four-centres'])
def test_fixed_process_recipe_runs_preflight_before_any_detached_owner(tmp_path, monkeypatch, operation, recipe):
    args = options(tmp_path, operation=operation, recipe=recipe)
    calls = []
    child_reads = []
    monkeypatch.setattr(managed, 'check_revision', lambda sha: calls.append(('revision', sha)))
    monkeypatch.setattr(managed, 'check_idle', lambda: calls.append(('idle',)))
    monkeypatch.setattr(managed, 'check_channel', lambda channel: calls.append(('channel', channel)))
    monkeypatch.setattr(managed.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(managed, 'process_start', lambda pid: '100')

    def preflight(command, **kwargs):
        assert command[-1] == '--preflight-only'
        assert '--torque-console-managed-cycle' in command
        assert '--pause-on-exit' not in command
        assert command[command.index('--torque-console-max-assist-nm')+1] == '1'
        assert command[command.index('--torque-console-auto-calibration-profile')+1] == str(managed.PROFILES[recipe])
        directory = Path(command[command.index('--output-root')+1])/'b601-owner-offline'
        directory.mkdir(parents=True)
        (directory/'b601_owner.json').write_text(json.dumps({'status': 'PASS', 'preflight_only': True,
            'can_opened': False, 'motor_command_sent': False, 'managed_cycle_enabled': True, 'channel': 'can0'}))
        calls.append(('preflight',))
        return SimpleNamespace(returncode=0)

    def popen(command, **kwargs):
        assert calls[-1] == ('revision', 'a'*40)
        assert kwargs['start_new_session'] and kwargs['stdin'] == subprocess.DEVNULL
        assert kwargs['pass_fds']
        assert kwargs['env']['PYTHONUNBUFFERED'] == '1'
        assert '--preflight-only' not in command
        assert managed.process_start(22) == '100'
        fd = int(command[command.index('--managed-startup-fd')+1])
        assert fd in kwargs['pass_fds']
        child_reads.append(os.dup(fd))  # The fake child owns the inherited pipe.
        calls.append(('launch',))
        return SimpleNamespace(pid=22)

    monkeypatch.setattr(managed.subprocess, 'run', preflight)
    monkeypatch.setattr(managed.subprocess, 'Popen', popen)
    result = managed.launch(args)
    plan = json.loads((Path(result['run_directory'])/'plan.json').read_text())
    assert plan['recipe'] == recipe
    assert ('preflight',) in calls
    assert (('launch',) in calls) == (operation == 'start')
    if operation == 'start':
        assert os.read(child_reads[0], 1) == b'G'
        os.close(child_reads.pop())
        receipt = json.loads((Path(result['run_directory'])/'process.json').read_text())
        assert receipt['start_ticks'] == '100' and receipt['pid'] == 22
        assert result['status'] == 'STARTED_NOT_YET_ACCEPTED'
    else:
        assert ('channel', 'can0') not in calls
        assert result['can_opened'] is False


def _start_inert_pipe_owner(tmp_path, monkeypatch):
    """Run only the exact gate function, then a marker; no B601 launcher body."""
    from scripts.hardware import rebot_b601_rs_owner as owner
    run = tmp_path/'inert-run'
    run.mkdir()
    marker = run/'device-path-reached'
    process = []
    actual_popen = subprocess.Popen
    source = inspect.getsource(owner._wait_for_managed_startup_release)
    code = ('import os, select, stat, sys\nfrom pathlib import Path\n' + source +
            '\ntry:\n _wait_for_managed_startup_release(int(sys.argv[1]))\n'
            'except ValueError:\n sys.exit(17)\n'
            'Path(sys.argv[2]).write_text("released")\n')

    def popen(command, **kwargs):
        fd = command[command.index('--managed-startup-fd')+1]
        child = actual_popen([sys.executable, '-c', code, fd, str(marker)], **kwargs)
        process.append(child)
        return child

    monkeypatch.setattr(managed.subprocess, 'Popen', popen)
    return run, marker, process, popen


@pytest.mark.parametrize('stage', ['open', 'write', 'flush', 'file-fsync', 'directory-open', 'directory-fsync'])
def test_receipt_persistence_fault_never_releases_existing_owner(tmp_path, monkeypatch, stage):
    run, marker, children, _ = _start_inert_pipe_owner(tmp_path, monkeypatch)
    original_open, original_fsync, original_os_open = Path.open, os.fsync, os.open

    class ReceiptStream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): self.stream.__enter__(); return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def write(self, data):
            if stage == 'write': raise OSError('injected receipt write')
            return self.stream.write(data)
        def flush(self):
            if stage == 'flush': raise OSError('injected receipt flush')
            return self.stream.flush()
        def fileno(self): return self.stream.fileno()

    def open_receipt(path, *args, **kwargs):
        if path == run/'process.json' and args and args[0] == 'x':
            if stage == 'open': raise OSError('injected receipt open')
            return ReceiptStream(original_open(path, *args, **kwargs))
        return original_open(path, *args, **kwargs)

    def fsync(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if (stage == 'directory-fsync' and directory) or (stage == 'file-fsync' and not directory):
            raise OSError('injected receipt fsync')
        return original_fsync(fd)

    def open_directory(path, *args, **kwargs):
        if stage == 'directory-open' and Path(path) == run:
            raise OSError('injected receipt directory open')
        return original_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_receipt)
    monkeypatch.setattr(managed.os, 'fsync', fsync)
    monkeypatch.setattr(managed.os, 'open', open_directory)
    with (run/'lease').open('w') as lock:
        with pytest.raises(OSError, match='injected receipt'):
            managed._launch_after_durable_receipt(
                [sys.executable, str(managed.OWNER), '--torque-console-managed-cycle'],
                run, lock, revision='a'*40)
    assert len(children) == 1 and children[0].returncode == 17
    assert not marker.exists()
    if stage in {'file-fsync', 'directory-open', 'directory-fsync'}:
        receipt = json.loads((run/'process.json').read_text())
        assert receipt['pid'] == children[0].pid and receipt['start_ticks'].isdigit()
    elif stage == 'open':
        assert not (run/'process.json').exists()


def test_durable_file_and_directory_precede_startup_release(tmp_path, monkeypatch):
    run, marker, children, _ = _start_inert_pipe_owner(tmp_path, monkeypatch)
    original_fsync, original_write = os.fsync, os.write
    synced = []

    def fsync(fd):
        original_fsync(fd)
        synced.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')

    def release(fd, data):
        assert synced == ['file', 'directory'] and not marker.exists()
        receipt = json.loads((run/'process.json').read_text())
        assert receipt['pid'] == children[0].pid
        return original_write(fd, data)

    monkeypatch.setattr(managed.os, 'fsync', fsync)
    monkeypatch.setattr(managed.os, 'write', release)
    with (run/'lease').open('w') as lock:
        receipt = managed._launch_after_durable_receipt(
            [sys.executable, str(managed.OWNER), '--torque-console-managed-cycle'],
            run, lock, revision='a'*40)
    assert children[0].wait(timeout=5) == 0 and marker.read_text() == 'released'
    assert receipt['command'][-2] == '--managed-startup-fd'


def test_uncertain_popen_result_closes_release_pipe_without_starting_device_path(tmp_path, monkeypatch):
    run, marker, children, popen = _start_inert_pipe_owner(tmp_path, monkeypatch)

    def uncertain(*args, **kwargs):
        popen(*args, **kwargs)
        raise OSError('unknown Popen result')

    monkeypatch.setattr(managed.subprocess, 'Popen', uncertain)
    with (run/'lease').open('w') as lock:
        with pytest.raises(OSError, match='unknown Popen result'):
            managed._launch_after_durable_receipt([], run, lock, revision='a'*40)
    assert children[0].wait(timeout=5) == 17
    assert not marker.exists() and not (run/'process.json').exists()


def test_uncertain_release_retains_durable_management_and_does_not_kill_child(tmp_path, monkeypatch):
    run, marker, children, _ = _start_inert_pipe_owner(tmp_path, monkeypatch)
    original_write = os.write

    def uncertain(fd, data):
        original_write(fd, data)
        raise OSError('unknown release result')

    monkeypatch.setattr(managed.os, 'write', uncertain)
    with (run/'lease').open('w') as lock:
        with pytest.raises(OSError, match='unknown release result'):
            managed._launch_after_durable_receipt([], run, lock, revision='a'*40)
    receipt = json.loads((run/'process.json').read_text())
    assert receipt['pid'] == children[0].pid
    assert children[0].wait(timeout=5) == 0 and marker.exists()


@pytest.mark.parametrize('after_release', [False, True])
def test_startup_interrupt_preserves_original_and_exact_release_boundary(tmp_path, monkeypatch, after_release):
    run, marker, children, _ = _start_inert_pipe_owner(tmp_path, monkeypatch)
    interrupted = KeyboardInterrupt('operator interrupt')
    original_write = os.write

    if after_release:
        def release_then_interrupt(fd, data):
            original_write(fd, data)
            raise interrupted
        monkeypatch.setattr(managed.os, 'write', release_then_interrupt)
    else:
        def interrupt_receipt(*args):
            raise interrupted
        monkeypatch.setattr(managed, '_write_new', interrupt_receipt)

    with (run/'lease').open('w') as lock:
        with pytest.raises(KeyboardInterrupt) as caught:
            managed._launch_after_durable_receipt([], run, lock, revision='a'*40)
    assert caught.value is interrupted
    assert children[0].wait(timeout=5) == (0 if after_release else 17)
    assert marker.exists() is after_release
    assert (run/'process.json').exists() is after_release
    if after_release:
        assert json.loads((run/'process.json').read_text())['pid'] == children[0].pid


@pytest.mark.parametrize('value', [None, '', 'not-an-identity'])
def test_unconfirmed_child_identity_denies_startup(tmp_path, monkeypatch, value):
    run, marker, children, _ = _start_inert_pipe_owner(tmp_path, monkeypatch)
    monkeypatch.setattr(managed, 'process_start', lambda pid: value)
    with (run/'lease').open('w') as lock:
        with pytest.raises(ValueError, match='no confirmed process identity'):
            managed._launch_after_durable_receipt([], run, lock, revision='a'*40)
    assert children[0].returncode == 17 and not marker.exists()


def test_denied_child_exit_must_be_confirmed_without_a_kill_fallback(tmp_path, monkeypatch):
    run = tmp_path/'pending-run'
    run.mkdir()
    child = SimpleNamespace(pid=22, wait=lambda timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired('inert', timeout)))
    monkeypatch.setattr(managed.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(managed, 'process_start', lambda pid: None)
    with (run/'lease').open('w') as lock:
        with pytest.raises(managed.ManagedStartupCleanupPending) as pending:
            managed._launch_after_durable_receipt([], run, lock, revision='a'*40)
    assert pending.value.process is child
    assert isinstance(pending.value.__cause__, ValueError)
    assert isinstance(pending.value.cleanup_error, subprocess.TimeoutExpired)


@pytest.mark.parametrize('token', [b'', b'X', b'G'])
def test_original_owner_gate_uses_exact_release_and_closes_fd(token):
    from scripts.hardware import rebot_b601_rs_owner as owner
    read_fd, write_fd = os.pipe()
    if token: os.write(write_fd, token)
    os.close(write_fd)
    if token == b'G':
        owner._wait_for_managed_startup_release(read_fd, timeout_sec=0)
    else:
        with pytest.raises(ValueError, match='not released'):
            owner._wait_for_managed_startup_release(read_fd, timeout_sec=0)
    with pytest.raises(OSError): os.fstat(read_fd)


def test_original_owner_gate_timeout_and_file_input_fail_closed(tmp_path):
    from scripts.hardware import rebot_b601_rs_owner as owner
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(ValueError, match='not released'):
            owner._wait_for_managed_startup_release(read_fd, timeout_sec=0)
    finally:
        os.close(write_fd)
    fd = os.open(tmp_path/'ordinary-file', os.O_CREAT | os.O_RDWR, 0o600)
    with pytest.raises(ValueError, match='not a file'):
        owner._wait_for_managed_startup_release(fd, timeout_sec=0)


def test_owner_rejects_denied_startup_before_resource_or_probe_work(monkeypatch, capsys):
    from scripts.hardware import rebot_b601_rs_owner as owner
    monkeypatch.setattr(owner, '_motorbridge_environment', lambda *a: pytest.fail('denied startup reached resource work'))
    monkeypatch.setattr(owner, '_run_foreground', lambda *a, **k: pytest.fail('denied startup reached probe'))
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with pytest.raises(SystemExit) as error:
        owner.main(['--allow-motion', '--torque-console-managed-cycle', '--managed-startup-fd', str(read_fd)])
    assert error.value.code == 2
    assert 'not released' in capsys.readouterr().err
    assert owner._parser().parse_args([]).managed_startup_fd is None


def test_bad_preflight_cannot_launch_and_runtime_cannot_enter_checkout(tmp_path, monkeypatch):
    args = options(tmp_path, operation='start')
    monkeypatch.setattr(managed.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(managed, 'check_revision', lambda sha: None)
    monkeypatch.setattr(managed, 'check_idle', lambda: None)
    monkeypatch.setattr(managed.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=2))
    monkeypatch.setattr(managed.subprocess, 'Popen', lambda *a, **k: pytest.fail('preflight failure reached owner launch'))
    with pytest.raises(ValueError, match='preflight failed'):
        managed.launch(args)
    args.output_root = ROOT/'forbidden-runtime'
    with pytest.raises(ValueError, match='outside'):
        managed.launch(args)


@pytest.mark.parametrize('flags,state', [([], 'ERROR-ACTIVE'), (['UP'], 'BUS-OFF'), (['UP'], None)])
def test_invalid_can_state_cannot_be_repaired_or_started_implicitly(monkeypatch, flags, state):
    monkeypatch.setattr(managed.socket, 'if_nametoindex', lambda name: 3)
    monkeypatch.setattr(managed.subprocess, 'check_output', lambda *a, **k:
        json.dumps([{'flags': flags, 'linkinfo': {'info_data': {'state': state}}}]))
    with pytest.raises(ValueError):
        managed.check_channel('can0')


def test_old_pid_and_exit_zero_cannot_become_success_or_a_stop_target(tmp_path, monkeypatch):
    run = tmp_path/'managed-closed'
    directory = run/'owner/b601-owner-test'
    directory.mkdir(parents=True)
    (run/'process.json').write_text(json.dumps({'pid': 22, 'start_ticks': '100',
        'command': [sys.executable, str(managed.OWNER), '--torque-console-managed-cycle']}))
    (directory/'b601_owner.json').write_text(json.dumps({'status': 'OK'}))
    monkeypatch.setattr(managed, 'process_start', lambda pid: '101')
    state = managed.inspect(run)
    assert not state['launcher_running'] and state['status'] == 'OPERATOR_REQUIRED'
    with pytest.raises(ValueError, match='no longer running'):
        managed.request_protective_hold(run)


def test_live_status_exposes_existing_shared_command_authority_without_a_control_request(tmp_path, monkeypatch):
    run = tmp_path/'managed-live'
    run.mkdir()
    (run/'process.json').write_text(json.dumps({'pid':22, 'start_ticks':'100',
        'command':[sys.executable, str(managed.OWNER), '--torque-console-managed-cycle']}))
    holding = {'combined_command_nm':[0,.037,.963,0,0,0], 'combined_l1_cap_nm':1,
        'saturated':True, 'quantity':'software_command_not_measured_force'}
    monkeypatch.setattr(managed, 'process_matches', lambda receipt: True)
    monkeypatch.setattr(managed, 'console_state', lambda *a: ('http://127.0.0.1:1234', {
        'managed_cycle':{'phase':'running'}, 'owner_ready':True, 'coordinated_holding':holding}))
    monkeypatch.setattr(managed, 'urlopen', lambda *a, **k: pytest.fail('status attempted an additional HTTP request'))
    result = managed.inspect(run)
    assert result['status'] == 'RUNNING'
    authority = result['live']['command_authority']
    assert authority['saturated'] and authority['combined_l1_cap_nm'] == 1
    assert authority['combined_command_nm'] == holding['combined_command_nm']
    assert authority['quantity'] == 'software_command_not_measured_force'


def test_managed_stop_cannot_escalate_to_full_disable(tmp_path, monkeypatch):
    monkeypatch.setattr(managed, 'inspect', lambda run: {'launcher_running': True, 'process': {'pid': 2}})
    monkeypatch.setattr(managed, 'process_matches', lambda receipt: True)
    monkeypatch.setattr(managed, 'console_state', lambda *a: ('http://127.0.0.1:1234', {}))
    requests = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'<meta name="csrf-token" content="local-token">' if not requests else b'{"ok": true}'
    def open_url(request, **kwargs):
        if not isinstance(request, str):
            requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr(managed, 'urlopen', open_url)
    for _ in range(2):
        # Two API requests remain protective hold, unlike two terminal SIGINTs.
        requests.clear()
        result = managed.request_protective_hold(tmp_path)
        assert requests == [{'action': 'protective_hold'}]
        assert result['full_disable_requested'] is False


@pytest.mark.parametrize('phase,matching', [('running', True), ('stopped', False), ('stopped', True)])
def test_supported_disable_wrapper_only_addresses_stopped_identity_bound_owner(tmp_path, monkeypatch, phase, matching):
    monkeypatch.setattr(managed, 'inspect', lambda run: {'launcher_running': True, 'process': {'pid': 2}})
    monkeypatch.setattr(managed, 'process_matches', lambda receipt: matching)
    monkeypatch.setattr(managed, 'console_state', lambda *a:
        ('http://127.0.0.1:1234', {'managed_cycle': {'phase': phase}}))
    requests = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"ok":true}' if requests else b'<meta name="csrf-token" content="local-token">'
    def open_url(request, **kwargs):
        if not isinstance(request, str):
            assert request.full_url == 'http://127.0.0.1:1234/api/stop'
            requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr(managed, 'urlopen', open_url)
    if phase != 'stopped' or not matching:
        with pytest.raises(ValueError):
            managed.request_supported_disable(tmp_path)
        assert not requests
    else:
        result = managed.request_supported_disable(tmp_path)
        assert requests == [{'action': 'full_disable'}]
        assert result['full_disable_requested'] and not result['cleanup_confirmed']


def test_live_endpoint_uses_flushed_child_log_even_when_launcher_output_is_buffered(tmp_path, monkeypatch):
    run = tmp_path/'managed-live'
    runtime = run/'owner/b601-owner-live/b601_owner.log'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('TORQUE CONSOLE: http://127.0.0.1:44395\n')
    (run/'management.log').write_text('foreground owner starts\n')
    snapshot = {'owner_pid': 33, 'runtime_log_path': str(runtime), 'owner_ready': True}
    urls = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(snapshot).encode()
    def open_url(url, **kwargs):
        urls.append(url)
        return Response()
    monkeypatch.setattr(managed, 'urlopen', open_url)
    monkeypatch.setattr(managed.os, 'getpgid', lambda pid: 22 if pid == 33 else 0)
    url, result = managed.console_state(run, {'pid': 22})
    assert url == 'http://127.0.0.1:44395' and result['owner_ready'] is True
    assert urls == [url+'/api/status?view=operator']
    snapshot['runtime_log_path'] = str(tmp_path/'other-owner.log')
    with pytest.raises(ValueError, match='different run'):
        managed.console_state(run, {'pid': 22})
    snapshot['runtime_log_path'] = str(runtime)
    monkeypatch.setattr(managed.os, 'getpgid', lambda pid: 44)
    with pytest.raises(ValueError, match='exact managed process group'):
        managed.console_state(run, {'pid': 22})


def test_live_endpoint_discovery_rejects_multiple_owner_endpoints(tmp_path, monkeypatch):
    run = tmp_path/'managed-ambiguous'
    for name, port in [('first', 1234), ('second', 4321)]:
        runtime = run/f'owner/b601-owner-{name}/b601_owner.log'
        runtime.parent.mkdir(parents=True)
        runtime.write_text(f'TORQUE CONSOLE: http://127.0.0.1:{port}\n')
    monkeypatch.setattr(managed, 'urlopen', lambda *a, **k: pytest.fail('ambiguous endpoint reached HTTP'))
    with pytest.raises(ValueError, match='ambiguous owner console'):
        managed.console_state(run, {'pid': 22})


def test_purpose_built_wrapper_previews_and_rejects_arbitrary_commands(tmp_path):
    env = dict(os.environ, KEY_FILE=str(tmp_path/'no-key'))
    wrapper = str(ROOT/'codex_remote/remote_b601_managed_cycle.sh')
    base = ['bash', wrapper, '-H', 'trusted-target', '--project', '/checkout', '--python', '/env/bin/python']
    result = subprocess.run([*base, '--operation', 'start', '--expected-revision', 'a'*40, '--dry-run'],
        env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0
    assert 'control_station.b601_torque_console.managed_run' in result.stdout
    for extra in (['--command', 'arbitrary'], ['--operation', 'kill'], ['--recipe', 'arbitrary'], ['--', '--allow-control-risk']):
        result = subprocess.run([*base, *extra, '--dry-run'], env=env, capture_output=True,
            text=True, timeout=10, check=False)
        assert result.returncode != 0
    for wrapper in ('remote_python.sh', 'remote_module.sh', 'remote_pytest.sh'):
        result = subprocess.run(['bash', str(ROOT/'codex_remote'/wrapper), '-H', 'trusted-target',
            '--', 'control_station.b601_torque_console.managed_run'], env=env,
            capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 5 and 'Refusing a control-risk command' in result.stderr


def test_owner_inherits_the_launch_lease_until_process_completion(tmp_path):
    import fcntl

    path = tmp_path/'lease'
    with path.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read(1)'],
            stdin=subprocess.PIPE, pass_fds=(lease.fileno(),))
    try:
        with path.open('a') as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process.communicate(b'x', timeout=3)
        with path.open('a') as competitor:
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
