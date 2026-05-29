#!/bin/bash
#
# GPU runtime smoke for SolidNES FermiNet PBC integration.

#SBATCH --account=hmt03
#SBATCH --nodes=1
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=solidnes-ferminet-gpu

set -eo pipefail

export MV2_ENABLE_AFFINITY="${MV2_ENABLE_AFFINITY:-0}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

SOLIDNES_EFFECTIVE_CPUS="${SOLIDNES_TOTAL_CPU_CORES:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}"
if ! [[ "${SOLIDNES_EFFECTIVE_CPUS}" =~ ^[0-9]+$ ]] || [ "${SOLIDNES_EFFECTIVE_CPUS}" -lt 1 ]; then
  SOLIDNES_EFFECTIVE_CPUS=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SOLIDNES_EFFECTIVE_CPUS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SOLIDNES_EFFECTIVE_CPUS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SOLIDNES_EFFECTIVE_CPUS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${SOLIDNES_EFFECTIVE_CPUS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SOLIDNES_EFFECTIVE_CPUS}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-localhost}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "SOLIDNES_EFFECTIVE_CPUS=${SOLIDNES_EFFECTIVE_CPUS}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "XLA_FLAGS=${XLA_FLAGS:-none}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

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

if [ -n "${SOLIDNES_VENV_DIR:-}" ]; then
  echo "Activating venv: ${SOLIDNES_VENV_DIR}"
  # shellcheck disable=SC1091
  source "${SOLIDNES_VENV_DIR}/bin/activate"
  SOLIDNES_NVIDIA_LIB_DIRS="$(
    find "${SOLIDNES_VENV_DIR}/lib" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | sort | paste -sd:
  )"
  if [ -n "${SOLIDNES_NVIDIA_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${SOLIDNES_NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "SOLIDNES_NVIDIA_LIB_DIRS=${SOLIDNES_NVIDIA_LIB_DIRS}"
  fi
  SOLIDNES_NVIDIA_BIN_DIRS="$(
    find "${SOLIDNES_VENV_DIR}/lib" -type d -path '*/site-packages/nvidia/*/bin' 2>/dev/null | sort | paste -sd:
  )"
  if [ -n "${SOLIDNES_NVIDIA_BIN_DIRS}" ]; then
    export PATH="${SOLIDNES_NVIDIA_BIN_DIRS}:${PATH}"
    echo "SOLIDNES_NVIDIA_BIN_DIRS=${SOLIDNES_NVIDIA_BIN_DIRS}"
  fi
elif [ -n "${SOLIDNES_CONDA_ENV:-}" ]; then
  echo "Activating conda environment: ${SOLIDNES_CONDA_ENV}"
  set +u
  conda activate "${SOLIDNES_CONDA_ENV}"
  set -u
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Working directory: $(pwd)"

SOLIDNES_PYTHON="${SOLIDNES_PYTHON:-python}"
SOLIDNES_BACKEND_SCRIPT="${SOLIDNES_BACKEND_SCRIPT:-scripts/backends/run_ferminet_train.py}"
SOLIDNES_BACKEND_ARGS="${SOLIDNES_BACKEND_ARGS:-}"
SOLIDNES_EXCITED_STATE_MAINLINE="${SOLIDNES_EXCITED_STATE_MAINLINE:-0}"
SOLIDNES_MAINLINE_EXCITED_STATE_EXPERIMENT="${SOLIDNES_MAINLINE_EXCITED_STATE_EXPERIMENT:-configs/experiment/diamond_c_psiformer_pbc_gamma_attention_paper_fullnode_anygpu_fused_qkv_x64_attnfix_b4096_iter10000_levmap128_jaxattn.yaml}"
if [ "${SOLIDNES_EXCITED_STATE_MAINLINE}" = "1" ] || [ "${SOLIDNES_EXCITED_STATE_MAINLINE}" = "true" ]; then
  SOLIDNES_EXPERIMENT="${SOLIDNES_EXPERIMENT:-${SOLIDNES_MAINLINE_EXCITED_STATE_EXPERIMENT}}"
else
  SOLIDNES_EXPERIMENT="${SOLIDNES_EXPERIMENT:-configs/experiment/diamond_c_ferminet_pbc_gamma_smoke.yaml}"
fi
SOLIDNES_FERMINET_ROOT="${SOLIDNES_FERMINET_ROOT:-external/ferminet}"
SOLIDNES_REQUIRE_JAX_GPU="${SOLIDNES_REQUIRE_JAX_GPU:-1}"
export PYTHONPATH="${SOLIDNES_FERMINET_ROOT}:src${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${SOLIDNES_REQUIRE_JAX_GPU}" = "1" ] || [ "${SOLIDNES_REQUIRE_JAX_GPU}" = "true" ]; then
  "${SOLIDNES_PYTHON}" - <<'PY'
import jax
import jaxlib

print(f"jax={jax.__version__}")
print(f"jaxlib={jaxlib.__version__}")
print(f"jax_default_backend={jax.default_backend()}")
print(f"jax_devices={jax.devices()}")
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise SystemExit("JAX did not report a GPU device")
PY
fi

echo "SOLIDNES_EXPERIMENT=${SOLIDNES_EXPERIMENT}"
echo "SOLIDNES_BACKEND_SCRIPT=${SOLIDNES_BACKEND_SCRIPT}"
echo "SOLIDNES_BACKEND_ARGS=${SOLIDNES_BACKEND_ARGS:-none}"
echo "SOLIDNES_FERMINET_ROOT=${SOLIDNES_FERMINET_ROOT}"
echo "SOLIDNES_VENV_DIR=${SOLIDNES_VENV_DIR:-none}"
echo "SOLIDNES_CONDA_ENV=${SOLIDNES_CONDA_ENV:-none}"
echo "FermiNet GPU smoke starts at $(date)"

SOLIDNES_BACKEND_ARGV=()
if [ -n "${SOLIDNES_BACKEND_ARGS}" ]; then
  mapfile -t SOLIDNES_BACKEND_ARGV < <(
    "${SOLIDNES_PYTHON}" - <<'PY'
import os
import shlex

for item in shlex.split(os.environ.get("SOLIDNES_BACKEND_ARGS", "")):
    print(item)
PY
  )
fi

"${SOLIDNES_PYTHON}" "${SOLIDNES_BACKEND_SCRIPT}" "${SOLIDNES_EXPERIMENT}" "${SOLIDNES_BACKEND_ARGV[@]}"

echo "FermiNet GPU smoke ends at $(date)"
