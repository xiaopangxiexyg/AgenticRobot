#!/usr/bin/env bash
set -euo pipefail

# Run the bounded edit-feedback tiers concurrently.  Each child still goes
# through run_pytest.sh, which owns the Python selection and basetemp.  The
# unit tier is split by test module into a small number of deterministic file
# shards; the aggregate JUnit/log keep the historical evidence paths intact.
# This wrapper owns the concurrency, per-child pytest cache, report paths, and
# the temporary shared assets link needed by a clean worktree.
# DEXTELEOP_FAST_FEEDBACK_RUN_ROOT may name a new absolute directory when a
# caller such as Harness needs the logs and JUnit reports under its Run root.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
umask 022

die() {
  echo "[dexteleop-fast-feedback] ERROR: $*" >&2
  exit 2
}

for argument in "$@"; do
  case "$argument" in
    --basetemp|--basetemp=*|-m|--junitxml|--junitxml=*|-o|--override-ini|--override-ini=*)
      die "$argument is owned by this wrapper; pass only shared pytest options such as --durations"
      ;;
  esac
done

RUN_ROOT=""
ASSET_LINK="$REPO_ROOT/assets"
ASSET_ROOT=""
ASSET_LINK_CREATED=0
UNIT_SHARD_COUNT=""
UNIT_SHARD_PATHS=()
UNIT_SHARD_LOGS=()
UNIT_SHARD_XMLS=()
UNIT_SHARD_PIDS=()
UNIT_SHARD_STATUSES=()
UNIT_EXPLICIT_TEST_TARGET=0
INTEGRATION_PID=""

prepare_run_root() {
  local requested_root="${DEXTELEOP_FAST_FEEDBACK_RUN_ROOT:-}"
  local parent

  if [[ -z "$requested_root" ]]; then
    RUN_ROOT="$(mktemp -d "/tmp/dt-fast-feedback-${UID:-user}-XXXXXX")"
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must be an absolute path: $requested_root" ;;
  esac
  if [[ -e "$requested_root" || -L "$requested_root" ]]; then
    die "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must not already exist: $requested_root"
  fi
  parent="$(dirname -- "$requested_root")"
  [[ -d "$parent" ]] || die \
    "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT parent is not a directory: $parent"
  mkdir -- "$requested_root" || die \
    "cannot create DEXTELEOP_FAST_FEEDBACK_RUN_ROOT: $requested_root"
  RUN_ROOT="$requested_root"
}

prepare_asset_compat_link() {
  local requested_root="${DEXTELEOP_ASSET_ROOT:-}"
  local existing_root

  if [[ -z "$requested_root" ]]; then
    [[ -e "$ASSET_LINK" ]] || die \
      "assets path is missing; set DEXTELEOP_ASSET_ROOT to an absolute shared asset root"
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_ASSET_ROOT must be an absolute directory: $requested_root" ;;
  esac
  [[ -d "$requested_root" ]] || die \
    "DEXTELEOP_ASSET_ROOT is not a directory: $requested_root"
  ASSET_ROOT="$(cd -- "$requested_root" && pwd -P)"

  if [[ -e "$ASSET_LINK" || -L "$ASSET_LINK" ]]; then
    existing_root="$(readlink -f -- "$ASSET_LINK")" || die \
      "cannot resolve existing assets path: $ASSET_LINK"
    [[ "$existing_root" == "$ASSET_ROOT" ]] || die \
      "existing assets path does not match DEXTELEOP_ASSET_ROOT: $existing_root != $ASSET_ROOT"
    return
  fi

  ln -s -- "$ASSET_ROOT" "$ASSET_LINK"
  ASSET_LINK_CREATED=1
}

configure_unit_shards() {
  local requested_count="${DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS:-2}"
  local argument
  local explicit_test_target=0

  case "$requested_count" in
    1|2|3|4) ;;
    *) die "DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS must be an integer from 1 to 4: $requested_count" ;;
  esac

  # The normal command accepts shared pytest options.  A path/nodeid is a
  # targeted invocation, however, and adding the complete shard file list
  # would duplicate that target.  Preserve the old targeted behavior by using
  # one unit child in that case.
  for argument in "$@"; do
    case "$argument" in
      --|.|./|tests/*|./tests/*|*/tests/*|*::* )
        explicit_test_target=1
        ;;
    esac
  done
  UNIT_SHARD_COUNT="$requested_count"
  UNIT_EXPLICIT_TEST_TARGET=0
  if ((explicit_test_target)); then
    if ((UNIT_SHARD_COUNT > 1)); then
      echo "[dexteleop-fast-feedback] unit sharding disabled for explicit test target" >&2
    fi
    UNIT_SHARD_COUNT=1
    UNIT_EXPLICIT_TEST_TARGET=1
  fi
}

prepare_unit_shards() {
  local index=0

  UNIT_SHARD_PATHS=()
  UNIT_SHARD_LOGS=()
  UNIT_SHARD_XMLS=()
  if ((UNIT_EXPLICIT_TEST_TARGET)); then
    UNIT_SHARD_PATHS[0]="$RUN_ROOT/unit-target-paths.txt"
    UNIT_SHARD_LOGS[0]="$RUN_ROOT/unit.log"
    UNIT_SHARD_XMLS[0]="$RUN_ROOT/unit.xml"
    : >"${UNIT_SHARD_PATHS[0]}"
    return
  fi
  for ((index = 0; index < UNIT_SHARD_COUNT; index++)); do
    UNIT_SHARD_PATHS[index]="$RUN_ROOT/unit-${index}-paths.txt"
    UNIT_SHARD_LOGS[index]="$RUN_ROOT/unit-${index}.log"
    UNIT_SHARD_XMLS[index]="$RUN_ROOT/unit-${index}.xml"
    : >"${UNIT_SHARD_PATHS[index]}"
  done

  python3 "$REPO_ROOT/scripts/pytest_sharding.py" plan \
    --repository "$REPO_ROOT" \
    --tests-root "$REPO_ROOT/tests" \
    --output-dir "$RUN_ROOT" \
    --shards "$UNIT_SHARD_COUNT" \
    --profile fast \
    --prefix unit || die "could not build a complete weighted unit shard plan"
}

stop_children() {
  local pid
  for pid in "${UNIT_SHARD_PIDS[@]}" "$INTEGRATION_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

cleanup() {
  local status=$?
  local current_root

  trap - EXIT INT TERM
  if ((${#UNIT_SHARD_PIDS[@]} > 0)) || [[ -n "$INTEGRATION_PID" ]]; then
    stop_children
  fi
  if ((ASSET_LINK_CREATED)); then
    if [[ -L "$ASSET_LINK" ]]; then
      current_root="$(readlink -f -- "$ASSET_LINK")" || current_root=""
      if [[ "$current_root" == "$ASSET_ROOT" ]]; then
        unlink -- "$ASSET_LINK" || status=1
      else
        echo "[dexteleop-fast-feedback] ERROR: refusing to remove changed assets link: $ASSET_LINK" >&2
        status=1
      fi
    else
      echo "[dexteleop-fast-feedback] ERROR: temporary assets link was replaced: $ASSET_LINK" >&2
      status=1
    fi
  fi
  if [[ -n "$RUN_ROOT" ]]; then
    echo "[dexteleop-fast-feedback] artifacts: $RUN_ROOT"
  fi
  exit "$status"
}

on_signal() {
  stop_children
  local pid
  for pid in "${UNIT_SHARD_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  wait "$INTEGRATION_PID" 2>/dev/null || true
  UNIT_SHARD_PIDS=()
  INTEGRATION_PID=""
  exit 130
}

run_tier() {
  local name="$1"
  local marker="$2"
  shift 2
  PYTHONNOUSERSITE=1 \
  PYTHONPYCACHEPREFIX="$RUN_ROOT/${name}-python-bytecode" \
  SIM_MUJOCO_CACHE_DIR="$RUN_ROOT/${name}-mujoco-cache" \
  "$REPO_ROOT/scripts/run_pytest.sh" \
    -m "$marker" \
    -q \
    -o "cache_dir=$RUN_ROOT/${name}-pytest-cache" \
    --junitxml="$RUN_ROOT/${name}.xml" \
    "$@" \
    >"$RUN_ROOT/${name}.log" 2>&1
}

merge_junit_reports() {
  local destination="$1"
  shift
  [[ "$#" -gt 0 ]] || die "no JUnit reports were supplied for $destination"

  python3 - "$destination" "$@" <<'PY'
from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

destination = Path(sys.argv[1])
sources = [Path(value) for value in sys.argv[2:]]
roots = []
for source in sources:
    if not source.is_file():
        raise SystemExit(f"missing JUnit shard report: {source}")
    root = ET.parse(source).getroot()
    if root.tag == "testsuite":
        roots.append(root)
    elif root.tag == "testsuites":
        suites = [child for child in root if child.tag == "testsuite"]
        if not suites:
            raise SystemExit(f"JUnit shard has no <testsuite>: {source}")
        roots.extend(suites)
    else:
        raise SystemExit(
            f"JUnit shard root is not <testsuite> or <testsuites>: {source}"
        )

merged = ET.Element("testsuite")
first_attributes = dict(roots[0].attrib)
merged.attrib.update(first_attributes)
integer_attributes = ("tests", "errors", "failures", "skipped")
for name in integer_attributes:
    merged.set(
        name,
        str(sum(int(root.attrib.get(name, "0")) for root in roots)),
    )
merged.set(
    "time",
    format(sum(float(root.attrib.get("time", "0")) for root in roots), ".6f"),
)

seen_cases: set[tuple[str, str]] = set()
for root in roots:
    for child in root:
        if child.tag == "testcase":
            identity = (child.attrib.get("classname", ""), child.attrib.get("name", ""))
            if identity in seen_cases:
                raise SystemExit(
                    "duplicate testcase in JUnit shards: "
                    f"{identity[0]}::{identity[1]}"
                )
            seen_cases.add(identity)
        merged.append(copy.deepcopy(child))

ET.indent(merged, space="  ")
destination.parent.mkdir(parents=True, exist_ok=True)
ET.ElementTree(merged).write(
    destination,
    encoding="utf-8",
    xml_declaration=True,
)
PY
}

merge_unit_logs() {
  local index
  if ((UNIT_EXPLICIT_TEST_TARGET)); then
    return
  fi
  : >"$RUN_ROOT/unit.log"
  for ((index = 0; index < UNIT_SHARD_COUNT; index++)); do
    {
      echo "===== unit shard $index ====="
      if [[ -f "${UNIT_SHARD_LOGS[index]}" ]]; then
        cat "${UNIT_SHARD_LOGS[index]}"
      else
        echo "missing unit shard log: ${UNIT_SHARD_LOGS[index]}" >&2
        UNIT_STATUS=1
      fi
    } >>"$RUN_ROOT/unit.log"
  done
}

trap cleanup EXIT
prepare_run_root
prepare_asset_compat_link
configure_unit_shards "$@"
prepare_unit_shards
trap on_signal INT TERM

if ((UNIT_EXPLICIT_TEST_TARGET)); then
  run_tier unit 'unit and not optional and not hardware and not slow_sim' "$@" &
  UNIT_SHARD_PIDS[0]=$!
else
  for ((index = 0; index < UNIT_SHARD_COUNT; index++)); do
    mapfile -t shard_paths <"${UNIT_SHARD_PATHS[index]}"
    run_tier "unit-${index}" 'unit and not optional and not hardware and not slow_sim' \
      "${shard_paths[@]}" "$@" &
    UNIT_SHARD_PIDS[index]=$!
  done
fi
run_tier integration 'integration and not optional and not hardware and not slow_sim' "$@" &
INTEGRATION_PID=$!

set +e
UNIT_STATUS=0
UNIT_SHARD_STATUSES=()
for ((index = 0; index < UNIT_SHARD_COUNT; index++)); do
  wait "${UNIT_SHARD_PIDS[index]}"
  shard_status=$?
  UNIT_SHARD_STATUSES[index]=$shard_status
  if ((shard_status != 0)); then
    UNIT_STATUS=1
  fi
done
UNIT_SHARD_PIDS=()
wait "$INTEGRATION_PID"
INTEGRATION_STATUS=$?
INTEGRATION_PID=""
set -e

merge_unit_logs
if ((UNIT_EXPLICIT_TEST_TARGET)); then
  [[ -f "$RUN_ROOT/unit.xml" ]] || UNIT_STATUS=1
else
  if ! merge_junit_reports "$RUN_ROOT/unit.xml" "${UNIT_SHARD_XMLS[@]}"; then
    UNIT_STATUS=1
  fi
fi

echo "===== fast unit (status=$UNIT_STATUS) ====="
cat "$RUN_ROOT/unit.log"
echo "===== fast integration (status=$INTEGRATION_STATUS) ====="
cat "$RUN_ROOT/integration.log"
echo "[dexteleop-fast-feedback] unit JUnit: $RUN_ROOT/unit.xml"
echo "[dexteleop-fast-feedback] integration JUnit: $RUN_ROOT/integration.xml"

if ((UNIT_STATUS != 0 || INTEGRATION_STATUS != 0)); then
  exit 1
fi
