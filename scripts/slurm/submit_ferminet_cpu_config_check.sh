#!/bin/bash
#
# Policy-aware submitter for no-compute FermiNet config checks.

set -eo pipefail

SOLIDNES_SUBMIT_SCRIPT="${SOLIDNES_SUBMIT_SCRIPT:-scripts/slurm/run_ferminet_cpu_config_check.slurm}"
SOLIDNES_SLURM_ACCOUNT="${SOLIDNES_SLURM_ACCOUNT:-hmt03}"
SOLIDNES_JOB_NAME="${SOLIDNES_JOB_NAME:-solidnes-ferminet-check}"
SOLIDNES_TIME_LIMIT="${SOLIDNES_TIME_LIMIT:-00:05:00}"
SOLIDNES_EXPECTED_MINUTES="${SOLIDNES_EXPECTED_MINUTES:-5}"
SOLIDNES_FORCE_PARTITION="${SOLIDNES_FORCE_PARTITION:-test}"
SOLIDNES_PARTITION_ORDER="${SOLIDNES_PARTITION_ORDER:-test regular regular128 regular256}"
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

export SOLIDNES_VENV_DIR="${SOLIDNES_VENV_DIR:-.venv/ferminet-jax0101-cuda12}"
export SOLIDNES_CONDA_ENV="${SOLIDNES_CONDA_ENV:-}"
export SOLIDNES_REQUIRE_JAX_GPU="${SOLIDNES_REQUIRE_JAX_GPU:-0}"

mkdir -p "$SOLIDNES_SLURM_LOG_DIR"
mkdir -p "$(dirname "$SOLIDNES_PLAN_JSON")"

planner_args=(
  "$SOLIDNES_PLANNER_PYTHON"
  "scripts/slurm/plan_slurm_job.py"
  "--kind" "cpu"
  "--partition-order" "$SOLIDNES_PARTITION_ORDER"
  "--expected-minutes" "$SOLIDNES_EXPECTED_MINUTES"
  "--job-name" "$SOLIDNES_JOB_NAME"
  "--account" "$SOLIDNES_SLURM_ACCOUNT"
  "--time" "$SOLIDNES_TIME_LIMIT"
  "--log-dir" "$SOLIDNES_SLURM_LOG_DIR"
  "--submit-script" "$SOLIDNES_SUBMIT_SCRIPT"
  "--output-json" "$SOLIDNES_PLAN_JSON"
)

if [ -n "$SOLIDNES_FORCE_PARTITION" ]; then
  planner_args+=("--force-partition" "$SOLIDNES_FORCE_PARTITION")
fi

if [ "$SOLIDNES_DRY_RUN" != "1" ] && [ "$SOLIDNES_DRY_RUN" != "true" ]; then
  planner_args+=("--run-sbatch")
fi

echo "SolidNES FermiNet no-compute config-check policy:"
echo "  partition_order=${SOLIDNES_PARTITION_ORDER}"
echo "  expected_minutes=${SOLIDNES_EXPECTED_MINUTES}"
echo "  force_partition=${SOLIDNES_FORCE_PARTITION:-none}"
echo "  venv_dir=${SOLIDNES_VENV_DIR:-none}"
echo "  conda_env=${SOLIDNES_CONDA_ENV:-none}"
echo "  submit_script=${SOLIDNES_SUBMIT_SCRIPT}"
echo "  plan_json=${SOLIDNES_PLAN_JSON}"

printf 'planner command:'
printf ' %q' "${planner_args[@]}"
printf '\n'

"${planner_args[@]}"

