#!/bin/bash
#
# Policy-aware GPU submitter for SolidNES FermiNet PBC smoke jobs.

set -eo pipefail

SOLIDNES_SUBMIT_SCRIPT="${SOLIDNES_SUBMIT_SCRIPT:-scripts/slurm/run_ferminet_gpu_smoke.slurm}"
SOLIDNES_SLURM_ACCOUNT="${SOLIDNES_SLURM_ACCOUNT:-hmt03}"
SOLIDNES_JOB_NAME="${SOLIDNES_JOB_NAME:-solidnes-ferminet-gpu}"
SOLIDNES_TIME_LIMIT="${SOLIDNES_TIME_LIMIT:-00:30:00}"
SOLIDNES_GPU_PARTITIONS="${SOLIDNES_GPU_PARTITIONS:-auto}"
SOLIDNES_GPU_ALLOW_TEST="${SOLIDNES_GPU_ALLOW_TEST:-0}"
SOLIDNES_GPU_BLOCKED_PARTITIONS_DEFAULT="gpu80gllm,gpu40gllm,h20llm,gpu4090,gpu4090_8,gpu4090_128,h800,test"
if [ "$SOLIDNES_GPU_ALLOW_TEST" = "1" ] || [ "$SOLIDNES_GPU_ALLOW_TEST" = "true" ]; then
  SOLIDNES_GPU_BLOCKED_PARTITIONS_DEFAULT="gpu80gllm,gpu40gllm,h20llm,gpu4090,gpu4090_8,gpu4090_128,h800"
fi
SOLIDNES_GPU_BLOCKED_PARTITIONS="${SOLIDNES_GPU_BLOCKED_PARTITIONS:-$SOLIDNES_GPU_BLOCKED_PARTITIONS_DEFAULT}"
SOLIDNES_GPU_ALLOWED_COUNTS="${SOLIDNES_GPU_ALLOWED_COUNTS:-}"
SOLIDNES_GPU_TARGET_GPUS="${SOLIDNES_GPU_TARGET_GPUS:-1}"
SOLIDNES_GPU_MIN_GPUS="${SOLIDNES_GPU_HARD_MIN_GPUS:-1}"
SOLIDNES_GPU_MIN_CPUS="${SOLIDNES_GPU_HARD_MIN_CPUS:-8}"
SOLIDNES_GPU_MIN_MEMORY_MB="${SOLIDNES_GPU_MIN_MEMORY_MB:-0}"
SOLIDNES_GPU_QUEUE_MIN_GPUS="${SOLIDNES_GPU_QUEUE_MIN_GPUS:-4}"
SOLIDNES_GPU_QUEUE_MIN_CPUS="${SOLIDNES_GPU_QUEUE_MIN_CPUS:-64}"
SOLIDNES_GPU_QUEUE_MEMORY_MB="${SOLIDNES_GPU_QUEUE_MEMORY_MB:-64000}"
SOLIDNES_GPU_QUEUE_MODE="${SOLIDNES_GPU_QUEUE_MODE:-auto}"
SOLIDNES_GPU_PRECISION_PROFILE="${SOLIDNES_GPU_PRECISION_PROFILE:-tf32}"
SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE="${SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE:-0}"
SOLIDNES_TASK_ROOT="${SOLIDNES_TASK_ROOT:-}"
if [ -n "$SOLIDNES_TASK_ROOT" ]; then
  SOLIDNES_SLURM_LOG_DIR="${SOLIDNES_SLURM_LOG_DIR:-$SOLIDNES_TASK_ROOT/logs/slurm}"
  SOLIDNES_PLAN_JSON="${SOLIDNES_PLAN_JSON:-$SOLIDNES_TASK_ROOT/outputs/slurm_plans/plan.json}"
else
  SOLIDNES_SLURM_LOG_DIR="${SOLIDNES_SLURM_LOG_DIR:-}"
  SOLIDNES_PLAN_JSON="${SOLIDNES_PLAN_JSON:-}"
fi
if [ -z "$SOLIDNES_SLURM_LOG_DIR" ] || [ -z "$SOLIDNES_PLAN_JSON" ]; then
  echo "Set SOLIDNES_TASK_ROOT to a numbered tasks/.../NNNN_slug bundle, or set both SOLIDNES_SLURM_LOG_DIR and SOLIDNES_PLAN_JSON." >&2
  exit 2
fi
SOLIDNES_DRY_RUN="${SOLIDNES_DRY_RUN:-0}"
SOLIDNES_PLANNER_PYTHON="${SOLIDNES_PLANNER_PYTHON:-python}"

export SOLIDNES_CONDA_ENV="${SOLIDNES_CONDA_ENV:-}"
if [ -n "${SOLIDNES_CONDA_ENV}" ]; then
  export SOLIDNES_VENV_DIR="${SOLIDNES_VENV_DIR:-}"
else
  export SOLIDNES_VENV_DIR="${SOLIDNES_VENV_DIR:-.venv/ferminet-jax0101-cuda12}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

mkdir -p "$SOLIDNES_SLURM_LOG_DIR"
mkdir -p "$(dirname "$SOLIDNES_PLAN_JSON")"

planner_args=(
  "$SOLIDNES_PLANNER_PYTHON"
  "scripts/slurm/plan_slurm_job.py"
  "--kind" "gpu"
  "--min-gpus" "$SOLIDNES_GPU_MIN_GPUS"
  "--target-gpus" "$SOLIDNES_GPU_TARGET_GPUS"
  "--min-cpus" "$SOLIDNES_GPU_MIN_CPUS"
  "--min-memory-mb" "$SOLIDNES_GPU_MIN_MEMORY_MB"
  "--queue-min-gpus" "$SOLIDNES_GPU_QUEUE_MIN_GPUS"
  "--queue-min-cpus" "$SOLIDNES_GPU_QUEUE_MIN_CPUS"
  "--queue-memory-mb" "$SOLIDNES_GPU_QUEUE_MEMORY_MB"
  "--gpu-queue-mode" "$SOLIDNES_GPU_QUEUE_MODE"
  "--precision-profile" "$SOLIDNES_GPU_PRECISION_PROFILE"
  "--job-name" "$SOLIDNES_JOB_NAME"
  "--account" "$SOLIDNES_SLURM_ACCOUNT"
  "--time" "$SOLIDNES_TIME_LIMIT"
  "--log-dir" "$SOLIDNES_SLURM_LOG_DIR"
  "--submit-script" "$SOLIDNES_SUBMIT_SCRIPT"
  "--output-json" "$SOLIDNES_PLAN_JSON"
)

if [ "$SOLIDNES_GPU_PARTITIONS" != "auto" ] && [ "$SOLIDNES_GPU_PARTITIONS" != "all" ]; then
  for partition in ${SOLIDNES_GPU_PARTITIONS//,/ }; do
    planner_args+=("--partition" "${partition%\*}")
  done
fi

for gpu_count in ${SOLIDNES_GPU_ALLOWED_COUNTS//,/ }; do
  if [ -n "$gpu_count" ]; then
    planner_args+=("--allowed-gpu-count" "$gpu_count")
  fi
done

for partition in ${SOLIDNES_GPU_BLOCKED_PARTITIONS//,/ }; do
  if [ -n "$partition" ]; then
    planner_args+=("--blocked-partition" "${partition%\*}")
  fi
done

if [ "$SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE" = "1" ] || [ "$SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE" = "true" ]; then
  planner_args+=("--exclusive-when-full-node")
fi

if [ "$SOLIDNES_DRY_RUN" != "1" ] && [ "$SOLIDNES_DRY_RUN" != "true" ]; then
  planner_args+=("--run-sbatch")
fi

echo "SolidNES FermiNet GPU submission policy:"
echo "  gpu_partitions=${SOLIDNES_GPU_PARTITIONS}"
echo "  blocked_partitions=${SOLIDNES_GPU_BLOCKED_PARTITIONS:-none}"
echo "  allowed_gpu_counts=${SOLIDNES_GPU_ALLOWED_COUNTS:-any}"
echo "  target_gpus=${SOLIDNES_GPU_TARGET_GPUS}"
echo "  hard_min_gpus=${SOLIDNES_GPU_MIN_GPUS}"
echo "  hard_min_cpus=${SOLIDNES_GPU_MIN_CPUS}"
echo "  queue_min_gpus=${SOLIDNES_GPU_QUEUE_MIN_GPUS}"
echo "  queue_min_cpus=${SOLIDNES_GPU_QUEUE_MIN_CPUS}"
echo "  allow_test=${SOLIDNES_GPU_ALLOW_TEST}"
echo "  queue_mode=${SOLIDNES_GPU_QUEUE_MODE}"
echo "  precision_profile=${SOLIDNES_GPU_PRECISION_PROFILE}"
echo "  exclusive_when_full_node=${SOLIDNES_GPU_EXCLUSIVE_WHEN_FULL_NODE}"
echo "  venv_dir=${SOLIDNES_VENV_DIR:-none}"
echo "  conda_env=${SOLIDNES_CONDA_ENV:-none}"
echo "  submit_script=${SOLIDNES_SUBMIT_SCRIPT}"
echo "  plan_json=${SOLIDNES_PLAN_JSON}"

printf 'planner command:'
printf ' %q' "${planner_args[@]}"
printf '\n'

"${planner_args[@]}"
