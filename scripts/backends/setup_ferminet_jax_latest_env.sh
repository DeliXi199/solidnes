#!/bin/bash
#
# Create/update the latest-JAX FermiNet environment used by SolidNES.

set -eo pipefail

ENV_NAME="${SOLIDNES_FERMINET_CONDA_ENV:-solidnes-ferminet-jax0101-cuda12}"
JAX_VERSION="${SOLIDNES_JAX_VERSION:-0.10.1}"
ENV_MODE="${SOLIDNES_FERMINET_ENV_MODE:-conda}"
VENV_DIR="${SOLIDNES_FERMINET_VENV_DIR:-.venv/ferminet-jax0101-cuda12}"

case "${ENV_MODE}" in
  conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "conda is required when SOLIDNES_FERMINET_ENV_MODE=conda" >&2
      exit 2
    fi
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
      echo "Conda environment already exists: ${ENV_NAME}"
    else
      conda env create -n "${ENV_NAME}" -f configs/env/ferminet_jax0101_cuda12.yml
    fi
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    set -u
    ;;
  venv)
    if [ ! -d "${VENV_DIR}" ]; then
      python -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    ;;
  *)
    echo "Unknown SOLIDNES_FERMINET_ENV_MODE=${ENV_MODE}; expected conda or venv" >&2
    exit 2
    ;;
esac

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "jax[cuda12]==${JAX_VERSION}"
python -m pip install --upgrade -e external/ferminet

SOLIDNES_EXPECT_JAX_VERSION="${JAX_VERSION}" \
  python scripts/backends/check_ferminet_environment.py \
  --expected-jax-version "${JAX_VERSION}"
