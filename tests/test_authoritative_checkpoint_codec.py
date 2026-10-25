"""Fresh interpreter core restore; native/backend state remains separately owned."""

import json
import subprocess
import sys

import pytest

from robot_motion.action.authoritative_checkpoint import capture_authoritative_control
from robot_motion.action.checkpoint_codec import (
    decode_control_checkpoint,
    encode_control_checkpoint,
)
from tests.test_authoritative_checkpoint import _moving, _restore

pytestmark = pytest.mark.unit


def test_cold_json_component_retains_history_and_resumes_in_a_fresh_interpreter(
    tmp_path,
):
    source, clock = _moving()
    state = capture_authoritative_control(source.core)
    blob = encode_control_checkpoint(state)
    path = tmp_path / "control.json"
    path.write_bytes(blob)
    child = r"""
import json,sys
from pathlib import Path
from robot_motion.action.checkpoint_codec import decode_control_checkpoint
from tests.test_authoritative_checkpoint import _restore
s=decode_control_checkpoint(Path(sys.argv[1]).read_bytes(),max_bytes=8*1024*1024)
core,rig,clock=_restore(s)
try:
 clock.value=10.1
 result=core.run_decision()
 print(json.dumps({"status":result.status,"epoch":core.execution_epoch,
   "facts":[fact.semantic_content_hash for fact in core.trace_facts]}))
finally:rig.close()
"""
    run = subprocess.run(
        [sys.executable, "-c", child, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(run.stdout)
    clock.value = 10.1
    expected = source.core.run_decision()
    assert result == {
        "status": expected.status,
        "epoch": source.core.execution_epoch,
        "facts": [fact.semantic_content_hash for fact in source.core.trace_facts],
    }
    assert path.read_bytes() == blob
    assert (
        encode_control_checkpoint(decode_control_checkpoint(blob, max_bytes=len(blob)))
        == blob
    )
    source.close()


def test_codec_rejects_unknown_types_fields_duplicate_keys_and_small_budget():
    source, _ = _moving()
    blob = encode_control_checkpoint(capture_authoritative_control(source.core))
    with pytest.raises(ValueError, match="budget"):
        decode_control_checkpoint(blob, max_bytes=len(blob) - 1)
    data = json.loads(blob)
    data["state"][1] = "ArbitraryExecutableFactory"
    with pytest.raises(ValueError, match="record"):
        decode_control_checkpoint(json.dumps(data).encode(), max_bytes=8 * 1024 * 1024)
    with pytest.raises(ValueError, match="Duplicate"):
        decode_control_checkpoint(b'{"schema":1,"schema":2}', max_bytes=1024)
    source.close()


def test_decoding_does_not_replace_configuration_or_history_validation():
    source, _ = _moving()
    data = json.loads(
        encode_control_checkpoint(capture_authoritative_control(source.core))
    )
    data["state"][2]["configuration_hash"] = "untrusted-configuration"
    parsed = decode_control_checkpoint(
        json.dumps(data).encode(), max_bytes=8 * 1024 * 1024
    )
    with pytest.raises(ValueError, match="configuration"):
        _restore(parsed)
    source.close()
