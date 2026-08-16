from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_control.asset_observation import load_asset_observations
from lab_control.workspace import LabError


def _production_path() -> Path:
    return Path(__file__).resolve().parents[1] / "stack" / "remote-asset-observations.json"


def test_production_asset_observation_is_non_authorizing() -> None:
    observations = load_asset_observations(_production_path())

    assert observations.host_id == "remote-gpu-hardware"
    assert set(observations.candidates) == {"dual_rm65_dmgripper_latest"}
    candidate = observations.candidates["dual_rm65_dmgripper_latest"]
    assert candidate.content_role == "openpi-checkpoint"
    assert candidate.content_relative_path.endswith("/99999")
    assert set(candidate.blockers) == {
        "content-manifest",
        "explicit-selection",
    }
    assert len(observations.sha256) == 64


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("activation_ready", True, "unsafe activation envelope"),
        ("selection_authorized", True, "unsafe activation envelope"),
        ("content_manifest", "f" * 64, "cannot claim content identity"),
        ("content_path", "/tmp/model", "unsafe content path"),
    ],
)
def test_asset_observation_cannot_promote_candidate_or_escape_paths(
    tmp_path: Path,
    field: str,
    replacement: object,
    error: str,
) -> None:
    payload = json.loads(_production_path().read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    candidate = tampered["candidates"]["dual_rm65_dmgripper_latest"]
    if field == "content_manifest":
        candidate["content"]["content_manifest_sha256"] = replacement
    elif field == "content_path":
        candidate["content"]["relative_path"] = replacement
    else:
        candidate[field] = replacement
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(LabError, match=error):
        load_asset_observations(path)
