#!/usr/bin/env bash
set -euo pipefail

PRESET="smoke"
DATASETS="SEED"
MODES="lp"
SEEDS="42"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Output directory: outputs/course_project"
echo "TensorBoard: tensorboard --logdir outputs/course_project --host 0.0.0.0 --port 6006"
echo "SSH port forward: ssh -L 6006:127.0.0.1:6006 <server>"
echo "Browser: http://127.0.0.1:6006"

python scripts/our_tasks/run_course_project.py \
  --resource l40 \
  --preset "${PRESET}" \
  --datasets "${DATASETS}" \
  --modes "${MODES}" \
  --seeds "${SEEDS}" \
  "${EXTRA_ARGS[@]}"
