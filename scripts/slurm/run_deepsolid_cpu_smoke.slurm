#!/bin/bash
#
# CPU runtime smoke for SolidNES DeepSolid integration.
# Recommended entrypoint:
#   bash scripts/slurm/submit_deepsolid_cpu_smoke.sh
#
# Direct sbatch is also supported:
#   sbatch scripts/slurm/run_deepsolid_cpu_smoke.slurm
#
#SBATCH --account=hmt03
#SBATCH --nodes=1
#SBATCH --partition=test
#SBATCH --job-name=solidnes-cpu-smoke

set -eo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MV2_ENABLE_AFFINITY="${MV2_ENABLE_AFFINITY:-0}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-localhost}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unknown}"

if [ -f "${HOME}/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

set -u
ulimit -s unlimited
ulimit -c unlimited

if command -v module >/dev/null 2>&1 && [ -n "${SOLIDNES_MODULES:-}" ]; then
  for module_name in ${SOLIDNES_MODULES}; do
    echo "Loading module: ${module_name}"
    module load "${module_name}"
  done
fi

if [ -n "${SOLIDNES_CONDA_ENV:-}" ]; then
  echo "Activating conda environment: ${SOLIDNES_CONDA_ENV}"
  set +u
  conda activate "${SOLIDNES_CONDA_ENV}"
  set -u
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Working directory: $(pwd)"

SOLIDNES_PYTHON="${SOLIDNES_PYTHON:-python}"
SOLIDNES_BACKEND_SCRIPT="${SOLIDNES_BACKEND_SCRIPT:-scripts/backends/run_deepsolid_process_smoke.py}"
SOLIDNES_EXPERIMENT="${SOLIDNES_EXPERIMENT:-configs/experiment/diamond_c_deepsolid_runtime_smoke.yaml}"
SOLIDNES_DEEPSOLID_ROOT="${SOLIDNES_DEEPSOLID_ROOT:-external/deepsolid}"
export PYTHONPATH="${SOLIDNES_DEEPSOLID_ROOT}:src${PYTHONPATH:+:${PYTHONPATH}}"

echo "SOLIDNES_EXPERIMENT=${SOLIDNES_EXPERIMENT}"
echo "SOLIDNES_BACKEND_SCRIPT=${SOLIDNES_BACKEND_SCRIPT}"
echo "SOLIDNES_DEEPSOLID_ROOT=${SOLIDNES_DEEPSOLID_ROOT}"
echo "SOLIDNES_TOTAL_CPU_CORES=${SOLIDNES_TOTAL_CPU_CORES:-unknown}"
echo "DeepSolid CPU smoke starts at $(date)"

"${SOLIDNES_PYTHON}" "${SOLIDNES_BACKEND_SCRIPT}" \
  "${SOLIDNES_EXPERIMENT}"

echo "DeepSolid CPU smoke ends at $(date)"
