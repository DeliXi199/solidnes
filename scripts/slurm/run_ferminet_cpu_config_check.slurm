#!/bin/bash
#
# No-compute CPU config/environment check for SolidNES FermiNet integration.

#SBATCH --account=hmt03
#SBATCH --nodes=1
#SBATCH --partition=test
#SBATCH --job-name=solidnes-ferminet-check

set -eo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MV2_ENABLE_AFFINITY="${MV2_ENABLE_AFFINITY:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-localhost}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unknown}"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}"

if [ -f "${HOME}/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

set -u

if [ -n "${SOLIDNES_VENV_DIR:-}" ]; then
  echo "Activating venv: ${SOLIDNES_VENV_DIR}"
  # shellcheck disable=SC1091
  source "${SOLIDNES_VENV_DIR}/bin/activate"
elif [ -n "${SOLIDNES_CONDA_ENV:-}" ]; then
  echo "Activating conda environment: ${SOLIDNES_CONDA_ENV}"
  set +u
  conda activate "${SOLIDNES_CONDA_ENV}"
  set -u
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Working directory: $(pwd)"

SOLIDNES_PYTHON="${SOLIDNES_PYTHON:-python}"
SOLIDNES_BACKEND_SCRIPT="${SOLIDNES_BACKEND_SCRIPT:-scripts/backends/check_ferminet_environment.py}"
SOLIDNES_EXPERIMENT="${SOLIDNES_EXPERIMENT:-configs/experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml}"
SOLIDNES_FERMINET_ROOT="${SOLIDNES_FERMINET_ROOT:-external/ferminet}"
export SOLIDNES_REQUIRE_JAX_GPU="${SOLIDNES_REQUIRE_JAX_GPU:-0}"
export PYTHONPATH="${SOLIDNES_FERMINET_ROOT}:src${PYTHONPATH:+:${PYTHONPATH}}"

echo "SOLIDNES_EXPERIMENT=${SOLIDNES_EXPERIMENT}"
echo "SOLIDNES_BACKEND_SCRIPT=${SOLIDNES_BACKEND_SCRIPT}"
echo "SOLIDNES_FERMINET_ROOT=${SOLIDNES_FERMINET_ROOT}"
echo "SOLIDNES_VENV_DIR=${SOLIDNES_VENV_DIR:-none}"
echo "SOLIDNES_CONDA_ENV=${SOLIDNES_CONDA_ENV:-none}"
echo "FermiNet CPU config check starts at $(date)"

"${SOLIDNES_PYTHON}" "${SOLIDNES_BACKEND_SCRIPT}" "${SOLIDNES_EXPERIMENT}"

echo "FermiNet CPU config check ends at $(date)"
