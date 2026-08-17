#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python scripts/dev_environment_doctor.py
bash scripts/run_pytest.sh -q \
  tests/test_dev_environment_doctor.py \
  tests/test_annotation_environment.py \
  tests/test_collection_runtime_boundaries.py \
  tests/test_control_composition_boundaries.py \
  tests/test_control_station_boundaries.py \
  tests/test_dataset_dependency_boundaries.py
git diff --check
