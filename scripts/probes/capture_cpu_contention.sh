#!/usr/bin/env bash
# Capture read-only host/process evidence for teleop versus model CPU contention.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/probes/capture_cpu_contention.sh \
    --phase NAME --duration-sec N --pid PID [--pid PID ...] \
    [--output-root DIR]

The script does not start, stop, renice, or change affinity for any process.
Pass the teleop PID and each model/training root PID. Existing descendants are
included automatically.
EOF
}

phase=""
duration_sec=120
output_root="/home/user/zc/teleop_cpu_contention"
declare -a target_pids=()

while (($# > 0)); do
  case "$1" in
    --phase)
      phase="${2:-}"
      shift 2
      ;;
    --duration-sec)
      duration_sec="${2:-}"
      shift 2
      ;;
    --pid)
      target_pids+=("${2:-}")
      shift 2
      ;;
    --output-root)
      output_root="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$phase" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf '%s\n' '--phase must use only letters, digits, dot, underscore, or dash' >&2
  exit 2
fi
if [[ ! "$duration_sec" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--duration-sec must be a positive integer' >&2
  exit 2
fi
if ((${#target_pids[@]} == 0)); then
  printf '%s\n' 'at least one --pid is required' >&2
  exit 2
fi

declare -a root_pids=()
declare -A seen_pids=()
for pid in "${target_pids[@]}"; do
  if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
    printf 'invalid PID: %s\n' "$pid" >&2
    exit 2
  fi
  if [[ ! -d "/proc/$pid" ]]; then
    printf 'PID is not running: %s\n' "$pid" >&2
    exit 2
  fi
  if [[ -z "${seen_pids[$pid]:-}" ]]; then
    seen_pids[$pid]=1
    root_pids+=("$pid")
  fi
done

declare -a process_pairs=()
mapfile -t process_pairs < <(ps -e -o pid=,ppid=)
descendant_added=1
while ((descendant_added != 0)); do
  descendant_added=0
  for pair in "${process_pairs[@]}"; do
    read -r child_pid parent_pid <<<"$pair"
    if [[ -n "${seen_pids[$parent_pid]:-}" && -z "${seen_pids[$child_pid]:-}" ]]; then
      seen_pids[$child_pid]=1
      descendant_added=1
    fi
  done
done
mapfile -t unique_pids < <(printf '%s\n' "${!seen_pids[@]}" | sort -n)

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root%/}/${timestamp}_${phase}"
mkdir -p -- "$run_dir"

pid_csv="$(IFS=,; printf '%s' "${unique_pids[*]}")"
root_pid_csv="$(IFS=,; printf '%s' "${root_pids[*]}")"

{
  printf 'phase=%s\n' "$phase"
  printf 'start_utc=%s\n' "$(date -u --iso-8601=ns)"
  printf 'start_epoch_ns=%s\n' "$(date +%s%N)"
  printf 'duration_sec=%s\n' "$duration_sec"
  printf 'root_pids=%s\n' "$root_pid_csv"
  printf 'sampled_pids=%s\n' "$pid_csv"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'kernel=%s\n' "$(uname -srmo)"
} >"$run_dir/metadata.txt"

if command -v lscpu >/dev/null 2>&1; then
  lscpu >"$run_dir/lscpu.txt"
  lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ >"$run_dir/lscpu_extended.txt"
fi

ps -eLo pid,ppid,tid,psr,cls,rtprio,pri,ni,stat,pcpu,pmem,comm \
  --sort=-pcpu >"$run_dir/process_threads_before.txt"

for pid in "${unique_pids[@]}"; do
  if [[ -d "/proc/$pid" ]]; then
    {
      printf 'pid=%s\n' "$pid"
      taskset -pc "$pid" 2>&1 || true
      grep -E '^(Name|State|Pid|PPid|Threads|Cpus_allowed_list|Mems_allowed_list|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' \
        "/proc/$pid/status" || true
      printf '%s\n' '[native thread environment]'
      tr '\0' '\n' <"/proc/$pid/environ" \
        | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR|BLIS|VECLIB_MAXIMUM)_NUM_THREADS=' \
        || true
    } >"$run_dir/pid_${pid}_before.txt"
  else
    printf 'PID exited before capture started: %s\n' "$pid" \
      >"$run_dir/pid_${pid}_before.txt"
  fi
done

for pressure_kind in cpu memory io; do
  if [[ -r "/proc/pressure/$pressure_kind" ]]; then
    cp -- "/proc/pressure/$pressure_kind" "$run_dir/pressure_${pressure_kind}_before.txt"
  fi
done

declare -a sampler_pids=()
start_sampler() {
  "$@" &
  sampler_pids+=("$!")
}

# Procfs provides per-thread CPU ticks and scheduler wait without sysstat.
probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
start_sampler python3 "$probe_dir/sample_procfs_cpu.py" \
  --duration-sec "$duration_sec" \
  "${root_pids[@]/#/--pid=}" \
  >"$run_dir/procfs_cpu.jsonl" 2>"$run_dir/procfs_cpu.stderr"

if command -v mpstat >/dev/null 2>&1; then
  start_sampler mpstat -P ALL 1 "$duration_sec" \
    >"$run_dir/mpstat.txt" 2>&1
else
  printf '%s\n' 'mpstat unavailable (install the sysstat package)' >"$run_dir/mpstat.txt"
fi

if command -v pidstat >/dev/null 2>&1; then
  start_sampler pidstat -u -r -w -t -p "$pid_csv" 1 "$duration_sec" \
    >"$run_dir/pidstat.txt" 2>&1
else
  printf '%s\n' 'pidstat unavailable (install the sysstat package)' >"$run_dir/pidstat.txt"
fi

if command -v vmstat >/dev/null 2>&1; then
  start_sampler vmstat -w 1 "$duration_sec" >"$run_dir/vmstat.txt" 2>&1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 1 >"$run_dir/nvidia_smi.csv" 2>&1 &
  nvidia_sampler_pid="$!"
  sampler_pids+=("$nvidia_sampler_pid")
  (
    sleep "$duration_sec"
    kill "$nvidia_sampler_pid" 2>/dev/null || true
  ) &
fi

sampler_failed=0
for sampler_pid in "${sampler_pids[@]}"; do
  if ! wait "$sampler_pid"; then
    sampler_failed=1
  fi
done

ps -eLo pid,ppid,tid,psr,cls,rtprio,pri,ni,stat,pcpu,pmem,comm \
  --sort=-pcpu >"$run_dir/process_threads_after.txt"

for pid in "${unique_pids[@]}"; do
  if [[ -d "/proc/$pid" ]]; then
    grep -E '^(Name|State|Pid|PPid|Threads|Cpus_allowed_list|Mems_allowed_list|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' \
      "/proc/$pid/status" >"$run_dir/pid_${pid}_after.txt" || true
  else
    printf 'PID exited before capture ended: %s\n' "$pid" \
      >"$run_dir/pid_${pid}_after.txt"
  fi
done

for pressure_kind in cpu memory io; do
  if [[ -r "/proc/pressure/$pressure_kind" ]]; then
    cp -- "/proc/pressure/$pressure_kind" "$run_dir/pressure_${pressure_kind}_after.txt"
  fi
done

{
  printf 'end_utc=%s\n' "$(date -u --iso-8601=ns)"
  printf 'end_epoch_ns=%s\n' "$(date +%s%N)"
  printf 'sampler_failed=%s\n' "$sampler_failed"
} >>"$run_dir/metadata.txt"

printf 'capture_dir=%s\n' "$run_dir"
if ((sampler_failed != 0)); then
  printf '%s\n' 'one or more optional samplers exited non-zero; inspect their logs' >&2
fi
