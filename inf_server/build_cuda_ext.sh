#!/usr/bin/env bash
# Build the two source CUDA extensions inf_server needs, against the uv venv's
# torch 2.6.0+cu126 using a conda CUDA-toolkit 12.6 + gcc userspace
# (~/micromamba/envs/$CUDATK_ENV), exactly like mask3d_feat/build_me.sh. Run AFTER
# `uv sync`. nvcc rejects a too-new system gcc, so we shim gcc/g++ to conda's.
#
# NOTE: torch 2.6/cu126 needs a CUDA **12.6** toolkit. The repo's default `cudatk`
# env is 11.8 (too old) and the system toolkit here is 13.2 (too new); create a
# 12.6 env and point CUDATK_ENV at it, e.g.:
#   micromamba create -n cudatk126 -c nvidia/label/cuda-12.6.0 cuda-toolkit gxx
# For the aarch64 offline target use export/build_on_target_aarch64.sh instead.
#
#   1. detectron2 (its _C ops) -- installed from git (cuda-ext group), no build
#      isolation so it links against the venv torch.
#   2. MSDeformAttn (DVIS_Plus pixel decoder) -- compiled in place. torch 2.6
#      removed Tensor::type(); apply patches/dvis_ms_deform_attn_scalar_type.patch
#      to DVIS_Plus first if not already applied.
set -e
ENV=$HOME/micromamba/envs/${CUDATK_ENV:-cudatk126}
HERE=$(cd "$(dirname "$0")" && pwd)
VENVPY=$HERE/.venv/bin/python

mkdir -p /tmp/d2cc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc /tmp/d2cc/gcc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-g++ /tmp/d2cc/g++
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc /tmp/d2cc/cc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-g++ /tmp/d2cc/c++

export PATH=/tmp/d2cc:$ENV/bin:$PATH
export CUDA_HOME=$ENV
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CONDA_BUILD_SYSROOT=$ENV/x86_64-conda-linux-gnu/sysroot
# conda's cuda-cudart headers (cuda_runtime_api.h) live under targets/<triple>/include,
# NOT $ENV/include (which is all torch's CUDAExtension adds). The host g++ compile of the
# MSDeformAttn .cpp files #includes <ATen/cuda/CUDAContext.h> -> cuda_runtime_api.h, so put
# the real header dir on CPATH or that compile fails with "No such file or directory".
export CPATH=$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=7.5
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

echo "=== nvcc: $(nvcc --version | grep -o 'release [0-9.]*') | host gcc: $(gcc --version | head -1) ==="

echo "=== [1/2] building detectron2 (no build isolation) ==="
# Pinned commit -- keep in sync with pyproject.toml [tool.uv.sources].detectron2.rev
D2="git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833"
uv pip install --python "$VENVPY" --no-build-isolation "$D2"

echo "=== [2/2] building MSDeformAttn kernel ==="
cd "$HERE/DVIS_Plus/mask2former/modeling/pixel_decoder/ops"
$VENVPY setup.py build install

echo "=== BUILD_CUDA_EXT_DONE ==="
