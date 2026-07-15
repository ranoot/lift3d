# Build the two CUDA source extensions (detectron2 + MultiScaleDeformableAttention)
# on a WINDOWS x86_64 target and install the inf_server venv offline from
# wheelhouse-windows-amd64. Windows equivalent of build_cuda_ext_on_target.sh.
#
# CAVEAT: detectron2 upstream does NOT officially support Windows. It generally
# compiles with the toolchain below, but expect to nudge it (see notes at bottom).
# The pure-python + torch/torchvision/opencv stack in the wheelhouse installs
# cleanly regardless; only these two CUDA exts are the hard part on Windows.
#
# Prereqs on the target:
#   - Python 3.11 (x86_64) + `uv` on PATH
#   - CUDA 12.6 toolkit (nvcc) on PATH
#   - Visual Studio 2022 Build Tools (MSVC v143, "Desktop development with C++")
#     -- run this from a "x64 Native Tools Command Prompt for VS 2022" / Dev PowerShell
#   - $env:TORCH_CUDA_ARCH_LIST set for the target GPU, e.g. "8.6"
$ErrorActionPreference = "Stop"

$D2_PIN = "02b5c4e295e990042a714712c21dc79b731e8833"   # keep in sync with pyproject.toml

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path      # inf_server\export
$ROOT = Split-Path -Parent $HERE                             # inf_server
$OPS  = Join-Path $ROOT "DVIS_Plus\mask2former\modeling\pixel_decoder\ops"
$PATCH = Join-Path $ROOT "patches\dvis_ms_deform_attn_scalar_type.patch"
$WHEELHOUSE = Join-Path $HERE "wheelhouse-windows-amd64"

# --- 0. sanity ---
foreach ($c in @("uv","nvcc","cl")) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { throw "$c not on PATH (need uv, CUDA 12.6 nvcc, and MSVC cl.exe)" }
}
if (-not (Test-Path $WHEELHOUSE)) { throw "$WHEELHOUSE missing (ship it from fetch_wheelhouses.sh)" }
if (-not $env:TORCH_CUDA_ARCH_LIST) { throw "set `$env:TORCH_CUDA_ARCH_LIST for the target GPU, e.g. 8.6" }
$env:FORCE_CUDA = "1"

Set-Location $ROOT

# --- 1. core venv, offline from the wheelhouse ---
Write-Host "=== [1/4] uv sync (core deps, offline) ==="
uv sync --no-default-groups --offline --find-links $WHEELHOUSE
$VENVPY = Join-Path $ROOT ".venv\Scripts\python.exe"
uv pip install --python $VENVPY --offline --find-links $WHEELHOUSE pip

# --- 2. MSDeformAttn scalar_type() patch (required for torch 2.6) ---
Write-Host "=== [2/4] MSDeformAttn scalar_type patch ==="
$cu = Join-Path $OPS "src\cuda\ms_deform_attn_cuda.cu"
if (Select-String -Path $cu -Pattern "value.type\(\)" -Quiet) {
    Push-Location (Join-Path $ROOT "DVIS_Plus"); git apply $PATCH; Pop-Location
    Write-Host "applied"
} else { Write-Host "already applied -- skipping" }

# --- 3. compile the two CUDA-ext wheels ---
Write-Host "=== [3/4] building detectron2 + MSDeformAttn wheels ==="
& $VENVPY -m pip wheel --no-build-isolation --no-deps `
    "git+https://github.com/facebookresearch/detectron2.git@$D2_PIN" -w $WHEELHOUSE
& $VENVPY -m pip wheel --no-build-isolation --no-deps $OPS -w $WHEELHOUSE

# --- 4. install both CUDA exts from the wheelhouse ---
Write-Host "=== [4/4] installing CUDA-ext wheels ==="
uv pip install --python $VENVPY --no-index --find-links $WHEELHOUSE detectron2 MultiScaleDeformableAttention

Write-Host ""
Write-Host "=== DONE. smoke test:  .venv\Scripts\python.exe verify.py ==="

# --- Windows detectron2 build notes (if step 3 fails) ---
# * Use MSVC v143 + CUDA 12.6; ensure cl.exe and nvcc agree on the host compiler.
# * If nvcc errors on the host pass, set:  $env:DISTUTILS_USE_SDK=1  and build from
#   the x64 Native Tools prompt so INCLUDE/LIB are populated.
# * Some detectron2 commits need a one-line patch removing POSIX-only `-Wall`/`-O3`
#   nvcc flags in setup.py on Windows; edit them out if nvcc rejects them.
