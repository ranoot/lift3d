#!/usr/bin/env bash
# Build MinkowskiEngine 0.5.4 (CUDA) against:
#   - conda cuda-toolkit 11.8 + gcc-11 (userspace, ~/micromamba/envs/cudatk)
#   - the uv venv's torch 2.0.1+cu118 / python 3.10
# Memory-throttled (MAX_JOBS=1) for a 7.5 GB WSL box; arch fixed to 7.5 (2080 Ti).
set -e
ENV=$HOME/micromamba/envs/cudatk
VENVPY=$HOME/lift3d/mask3d_feat/.venv/bin/python

# Make plain gcc/g++ resolve to conda's 11.x so nvcc's default host compiler is
# accepted (nvcc 11.8 rejects the system gcc-15).
mkdir -p /tmp/mecc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc /tmp/mecc/gcc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-g++ /tmp/mecc/g++
ln -sf $ENV/bin/x86_64-conda-linux-gnu-gcc /tmp/mecc/cc
ln -sf $ENV/bin/x86_64-conda-linux-gnu-g++ /tmp/mecc/c++

export PATH=/tmp/mecc:$ENV/bin:$PATH
export CUDA_HOME=$ENV
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CONDA_BUILD_SYSROOT=$ENV/x86_64-conda-linux-gnu/sysroot
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=7.5
export BLAS=openblas
export BLAS_INCLUDE_DIRS=$ENV/include
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

cd $HOME/lift3d/mask3d_feat/MinkowskiEngine
echo "=== nvcc: $(which nvcc) ==="; nvcc --version | grep release
echo "=== host gcc: $(gcc --version | head -1) ==="
echo "=== building (this is slow, single-threaded) ==="
$VENVPY setup.py install --force_cuda --blas=openblas
echo "=== BUILD_ME_DONE ==="
