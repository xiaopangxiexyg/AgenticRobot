#!/usr/bin/env bash

(
set -Eeuo pipefail

source_path=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
target_path=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1.accidental_20260806_234020

if [[ ! -e "$source_path" ]]; then
  printf 'Nothing to quarantine: %s\n' "$source_path"
  exit 0
fi

if [[ -e "$target_path" ]]; then
  printf 'Refusing to overwrite quarantine path: %s\n' "$target_path" >&2
  exit 2
fi

mv -- "$source_path" "$target_path"
printf 'Quarantined accidental run at: %s\n' "$target_path"
)
