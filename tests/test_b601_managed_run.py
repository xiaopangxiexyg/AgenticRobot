"""Fake process/transport boundaries; these tests cannot launch the B601 owner."""

import json
import os
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
        'field_report': None, 'expected_revision': 'a'*40, 'output_root': tmp_path/'runs', 'operation': 'preflight'}
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('operation', ['preflight', 'start'])
def test_fixed_process_recipe_runs_preflight_before_any_detached_owner(tmp_path, monkeypatch, operation):
    args = options(tmp_path, operation=operation)
    calls = []
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
        calls.append(('launch',))
        return SimpleNamespace(pid=22)

    monkeypatch.setattr(managed.subprocess, 'run', preflight)
    monkeypatch.setattr(managed.subprocess, 'Popen', popen)
    result = managed.launch(args)
    assert ('preflight',) in calls
    assert (('launch',) in calls) == (operation == 'start')
    if operation == 'start':
        receipt = json.loads((Path(result['run_directory'])/'process.json').read_text())
        assert receipt['start_ticks'] == '100' and receipt['pid'] == 22
        assert result['status'] == 'STARTED_NOT_YET_ACCEPTED'
    else:
        assert ('channel', 'can0') not in calls
        assert result['can_opened'] is False


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
    for extra in (['--command', 'arbitrary'], ['--operation', 'kill'], ['--', '--allow-control-risk']):
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
