#!/usr/bin/env bash
# Build the two CUDA source extensions that have NO redistributable wheels and must
# be compiled against the target's torch + CUDA 12.6 toolchain:
#   - detectron2 (pinned git commit)
#   - MultiScaleDeformableAttention / MSDeformAttn (vendored under DVIS_Plus)
# and install the whole inf_server venv fully OFFLINE from the matching wheelhouse.
#
# Works on BOTH Linux targets (x86_64 and aarch64); it auto-selects the wheelhouse
# from `uname -m`. Windows uses build_cuda_ext_on_target.ps1 instead.
#
# Prereqs on the target: `uv` on PATH, a CUDA 12.6 toolkit (`nvcc`), a C/C++ host
# compiler, and TORCH_CUDA_ARCH_LIST set for the target GPU (e.g. 8.6 for RTX 30xx,
# 8.7 Orin, 9.0 GH200). No conda/micromamba needed -- the VM already has CUDA 12.6.
set -euo pipefail

D2_PIN="02b5c4e295e990042a714712c21dc79b731e8833"   # keep in sync with pyproject.toml

HERE=$(cd "$(dirname "$0")" && pwd)     # inf_server/export
ROOT=$(cd "$HERE/.." && pwd)            # inf_server
OPS="$ROOT/DVIS_Plus/mask2former/modeling/pixel_decoder/ops"
PATCHFILE="$ROOT/patches/dvis_ms_deform_attn_scalar_type.patch"

case "$(uname -m)" in
    x86_64|amd64)   WHEELHOUSE="$HERE/wheelhouse-linux-x86_64" ;;
    aarch64|arm64)  WHEELHOUSE="$HERE/wheelhouse-linux-aarch64" ;;
    *) echo "ERROR: unsupported arch $(uname -m)"; exit 1 ;;
esac

# --- 0. sanity ---
command -v uv   >/dev/null || { echo "ERROR: uv not on PATH"; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc (CUDA 12.6 toolkit) not on PATH"; exit 1; }
[ -d "$WHEELHOUSE" ] || { echo "ERROR: $WHEELHOUSE missing (ship it from fetch_wheelhouses.sh)"; exit 1; }
: "${TORCH_CUDA_ARCH_LIST:?set TORCH_CUDA_ARCH_LIST for the target GPU, e.g. 8.6 / 8.7 / 9.0}"
export FORCE_CUDA=1     # build CUDA ops even if torch.cuda.is_available() is False at build time

cd "$ROOT"

# --- 1. core venv, fully offline from the shipped wheelhouse ---
echo "=== [1/4] uv sync (core deps, offline) ==="
uv sync --no-default-groups --offline --find-links "$WHEELHOUSE"
VENVPY="$ROOT/.venv/bin/python"

# a builder pip inside the venv (uv venvs omit pip; `uv pip` has no `wheel` verb)
uv pip install --python "$VENVPY" --offline --find-links "$WHEELHOUSE" pip

# --- 2. MSDeformAttn scalar_type() patch (idempotent; required for torch 2.6) ---
echo "=== [2/4] MSDeformAttn scalar_type patch ==="
if grep -q 'value.type()' "$OPS/src/cuda/ms_deform_attn_cuda.cu"; then
    ( cd "$ROOT/DVIS_Plus" && git apply "$PATCHFILE" ) && echo "applied"
else
    echo "already applied -- skipping"
fi

# --- 3. compile the two CUDA-ext wheels against the venv torch ---
echo "=== [3/4] building detectron2 + MSDeformAttn wheels ==="
"$VENVPY" -m pip wheel --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/detectron2.git@${D2_PIN}" -w "$WHEELHOUSE"
"$VENVPY" -m pip wheel --no-build-isolation --no-deps "$OPS" -w "$WHEELHOUSE"

# --- 4. install both CUDA exts from the wheelhouse ---
echo "=== [4/4] installing CUDA-ext wheels ==="
uv pip install --python "$VENVPY" --no-index --find-links "$WHEELHOUSE" \
    detectron2 MultiScaleDeformableAttention

echo
echo "=== DONE. smoke test:  .venv/bin/python verify.py ==="
echo "The two freshly built wheels are now in $WHEELHOUSE (redistributable for identical targets)."
