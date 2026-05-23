#!/bin/bash
#
# GPU runtime smoke for SolidNES DeepSolid integration.
# Recommended entrypoint:
#   bash scripts/slurm/submit_deepsolid_gpu_smoke.sh
#
#SBATCH --account=hmt03
#SBATCH --nodes=1
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=solidnes-gpu-smoke

set -eo pipefail

export MV2_ENABLE_AFFINITY="${MV2_ENABLE_AFFINITY:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

count_visible_gpus() {
  if [ -n "${SOLIDNES_TOTAL_GPUS:-}" ]; then
    printf '%s\n' "${SOLIDNES_TOTAL_GPUS}"
    return
  fi
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "${CUDA_VISIBLE_DEVICES}" = "none" ]; then
    printf '1\n'
    return
  fi
  local visible="${CUDA_VISIBLE_DEVICES//,/ }"
  # shellcheck disable=SC2086
  set -- ${visible}
  printf '%s\n' "$#"
}

build_cpu_batches() {
  local total_cpus="$1"
  local total_gpus="$2"
  local base=$((total_cpus / total_gpus))
  local remainder=$((total_cpus % total_gpus))
  local start=0
  local batches=""
  local sizes=""
  local gpu_index size end cpu_range
  for ((gpu_index = 0; gpu_index < total_gpus; gpu_index++)); do
    size="${base}"
    if [ "${gpu_index}" -lt "${remainder}" ]; then
      size=$((size + 1))
    fi
    if [ -n "${sizes}" ]; then
      sizes="${sizes}:"
      batches="${batches};"
    fi
    sizes="${sizes}${size}"
    if [ "${size}" -le 0 ]; then
      batches="${batches}gpu${gpu_index}:none"
      continue
    fi
    end=$((start + size - 1))
    if [ "${start}" -eq "${end}" ]; then
      cpu_range="${start}"
    else
      cpu_range="${start}-${end}"
    fi
    batches="${batches}gpu${gpu_index}:${cpu_range}"
    start=$((end + 1))
  done
  export SOLIDNES_CPUS_PER_GPU_BASE="${SOLIDNES_CPUS_PER_GPU_BASE:-${base}}"
  export SOLIDNES_CPU_REMAINDER="${SOLIDNES_CPU_REMAINDER:-${remainder}}"
  export SOLIDNES_CPUS_PER_GPU_LIST="${SOLIDNES_CPUS_PER_GPU_LIST:-${sizes}}"
  export SOLIDNES_CPU_BATCHES="${SOLIDNES_CPU_BATCHES:-${batches}}"
}

SOLIDNES_EFFECTIVE_GPUS="$(count_visible_gpus)"
SOLIDNES_EFFECTIVE_CPUS="${SOLIDNES_TOTAL_CPU_CORES:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}"
if ! [[ "${SOLIDNES_EFFECTIVE_CPUS}" =~ ^[0-9]+$ ]] || [ "${SOLIDNES_EFFECTIVE_CPUS}" -lt 1 ]; then
  SOLIDNES_EFFECTIVE_CPUS=1
fi
if ! [[ "${SOLIDNES_EFFECTIVE_GPUS}" =~ ^[0-9]+$ ]] || [ "${SOLIDNES_EFFECTIVE_GPUS}" -lt 1 ]; then
  SOLIDNES_EFFECTIVE_GPUS=1
fi
build_cpu_batches "${SOLIDNES_EFFECTIVE_CPUS}" "${SOLIDNES_EFFECTIVE_GPUS}"

SOLIDNES_CPU_THREAD_MODE="${SOLIDNES_CPU_THREAD_MODE:-all}"
case "${SOLIDNES_CPU_THREAD_MODE}" in
  all)
    SOLIDNES_THREAD_COUNT="${SOLIDNES_CPU_PARALLELISM:-${SOLIDNES_EFFECTIVE_CPUS}}"
    ;;
  per_gpu)
    SOLIDNES_THREAD_COUNT="${SOLIDNES_CPUS_PER_GPU_BASE:-$((SOLIDNES_EFFECTIVE_CPUS / SOLIDNES_EFFECTIVE_GPUS))}"
    if [ "${SOLIDNES_THREAD_COUNT}" -lt 1 ]; then
      SOLIDNES_THREAD_COUNT=1
    fi
    ;;
  *)
    echo "Unknown SOLIDNES_CPU_THREAD_MODE=${SOLIDNES_CPU_THREAD_MODE}; expected all or per_gpu" >&2
    exit 2
    ;;
esac
export SOLIDNES_CPU_PARALLELISM="${SOLIDNES_CPU_PARALLELISM:-${SOLIDNES_EFFECTIVE_CPUS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SOLIDNES_THREAD_COUNT}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SOLIDNES_THREAD_COUNT}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SOLIDNES_THREAD_COUNT}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${SOLIDNES_THREAD_COUNT}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SOLIDNES_THREAD_COUNT}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${SOLIDNES_THREAD_COUNT}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-localhost}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "SOLIDNES_EFFECTIVE_CPUS=${SOLIDNES_EFFECTIVE_CPUS}"
echo "SOLIDNES_EFFECTIVE_GPUS=${SOLIDNES_EFFECTIVE_GPUS}"
echo "SOLIDNES_CPU_THREAD_MODE=${SOLIDNES_CPU_THREAD_MODE}"
echo "SOLIDNES_CPU_PARALLELISM=${SOLIDNES_CPU_PARALLELISM}"
echo "SOLIDNES_CPUS_PER_GPU_BASE=${SOLIDNES_CPUS_PER_GPU_BASE}"
echo "SOLIDNES_CPU_REMAINDER=${SOLIDNES_CPU_REMAINDER}"
echo "SOLIDNES_CPUS_PER_GPU_LIST=${SOLIDNES_CPUS_PER_GPU_LIST}"
echo "SOLIDNES_CPU_BATCHES=${SOLIDNES_CPU_BATCHES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"

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
SOLIDNES_REQUIRE_JAX_GPU="${SOLIDNES_REQUIRE_JAX_GPU:-1}"
export PYTHONPATH="${SOLIDNES_DEEPSOLID_ROOT}:src${PYTHONPATH:+:${PYTHONPATH}}"

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
echo "SOLIDNES_DEEPSOLID_ROOT=${SOLIDNES_DEEPSOLID_ROOT}"
echo "SOLIDNES_TOTAL_GPUS=${SOLIDNES_TOTAL_GPUS:-unknown}"
echo "SOLIDNES_TOTAL_CPU_CORES=${SOLIDNES_TOTAL_CPU_CORES:-unknown}"
echo "DeepSolid GPU smoke starts at $(date)"

"${SOLIDNES_PYTHON}" "${SOLIDNES_BACKEND_SCRIPT}" \
  "${SOLIDNES_EXPERIMENT}"

echo "DeepSolid GPU smoke ends at $(date)"
