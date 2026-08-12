#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
VENV_DIR="${STARVLA_VENV_DIR:-${REPO_ROOT}/.venv}"
BOOTSTRAP_DIR="${STARVLA_BOOTSTRAP_DIR:-${REPO_ROOT}/.virtualenv-bootstrap}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}" || true
fi

if ! [[ -x "${VENV_DIR}/bin/python" ]] || ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "python venv/ensurepip is unavailable; bootstrapping virtualenv into ${BOOTSTRAP_DIR}"
  mkdir -p "${BOOTSTRAP_DIR}"
  "${PYTHON_BIN}" -m pip install --upgrade --target "${BOOTSTRAP_DIR}" virtualenv
  PYTHONPATH="${BOOTSTRAP_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m virtualenv --clear --system-site-packages "${VENV_DIR}"
fi

RUNTIME_PYTHON="${VENV_DIR}/bin/python"
"${RUNTIME_PYTHON}" -m pip install -r requirements.txt
"${RUNTIME_PYTHON}" -m pip install --no-deps -e .

DS_ACCELERATOR=musa "${RUNTIME_PYTHON}" - <<'PY'
import torch
import torch_musa
from deepspeed.accelerator import get_accelerator
from diffusers import ModelMixin
from torch_musa.optim import FusedAdamW

assert torch.musa.is_available(), "MUSA is not available"
assert torch.musa.device_count() == 8, f"expected 8 MUSA devices, got {torch.musa.device_count()}"
assert get_accelerator().device_name() == "musa", "DeepSpeed did not select MUSA"
print("StarVLA environment ready:")
print(f"  torch={torch.__version__}")
print(f"  musa_devices={torch.musa.device_count()}")
print(f"  deepspeed_accelerator={get_accelerator().device_name()}")
print(f"  diffusers_model_mixin={ModelMixin.__name__}")
print(f"  fused_optimizer={FusedAdamW.__name__}")
PY
