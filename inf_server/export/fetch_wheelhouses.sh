#!/usr/bin/env bash
# Download the BINARY-wheel portion of the inf_server bundle for all three targets
# into export/wheelhouse-<platform>/, using the per-platform marker-resolved lists
# produced by compile_requirements.sh. Runs entirely from ONE host (any arch/OS):
# `pip download --platform` only sets wheel tags, and because the requirement files
# are already resolved AS the target, no host markers leak in.
#
# NOT fetched here (no wheels exist -- compiled on the target):
#   - detectron2            (git source, CUDA build)   -> build_cuda_ext_on_target.*
#   - MultiScaleDeformableAttention / MSDeformAttn      -> build_cuda_ext_on_target.*
# The `detectron2 @ git+...` line is stripped from each list before download.
#
# If pip aborts with "No matching distribution" for some package on a target, that
# package ships no wheel for that platform/cp311 -- note it and build it on target
# from sdist (rare; the graph is otherwise all-binary).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CU126="https://download.pytorch.org/whl/cu126"
PY=${PYTHON:-python3}

# pip --abi is exact, so list every abi tag cp311 can consume: the version-specific
# cp311, the stable ABI abi3, and pure-python none.
ABIS=(--abi cp311 --abi abi3 --abi none)

# Two graph deps ship ONLY as sdists (antlr4-python3-runtime, fvcore). Both are pure
# python, so we pre-build them into universal py3-none-any wheels once (see the
# `pip wheel` step in the README / build_noarch_wheels below) and satisfy them from
# this shared dir via --find-links; they are then copied into every wheelhouse so
# each is self-contained. --only-binary=:all: (required by --platform) would abort on
# them otherwise.
NOARCH="$HERE/wheelhouse-noarch"

fetch() {   # fetch <req-file> <out-dir> <platform-tag...>
    local req="$1" out="$2"; shift 2
    local plats=(); local p
    for p in "$@"; do plats+=(--platform "$p"); done

    [ -f "$req" ] || { echo "MISSING $req -- run compile_requirements.sh first"; exit 1; }
    [ -d "$NOARCH" ] || { echo "MISSING $NOARCH -- run: pip wheel --no-deps -w $NOARCH 'antlr4-python3-runtime==4.9.3' 'fvcore==0.1.5.post20221221'"; exit 1; }
    mkdir -p "$out"
    # drop the git detectron2 line (built on target, not downloadable as a wheel)
    local tmp; tmp=$(mktemp)
    grep -v 'git+' "$req" > "$tmp"

    echo "=== downloading -> $(basename "$out") ==="
    "$PY" -m pip download \
        --requirement "$tmp" \
        --dest "$out" \
        --only-binary=:all: \
        --python-version 311 \
        --implementation cp \
        "${ABIS[@]}" \
        "${plats[@]}" \
        --find-links "$NOARCH" \
        --extra-index-url "$CU126" || {
            echo "!! pip download failed for $out -- see above"; rm -f "$tmp"; return 1; }
    rm -f "$tmp"
    # copy the shared pure-python wheels in so each wheelhouse installs standalone
    cp -n "$NOARCH"/*.whl "$out"/ 2>/dev/null || true
    echo "    $(ls -1 "$out" | wc -l) wheels in $(basename "$out")"
}

# manylinux tags: newest first so pip prefers modern glibc but falls back for
# older-built wheels. linux_<arch> (plain) covers the pytorch torch/vision wheels.
fetch "$HERE/requirements-linux-x86_64.txt"  "$HERE/wheelhouse-linux-x86_64" \
    manylinux_2_39_x86_64 manylinux_2_35_x86_64 manylinux_2_34_x86_64 \
    manylinux_2_31_x86_64 manylinux_2_28_x86_64 manylinux_2_27_x86_64 \
    manylinux_2_24_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64 \
    manylinux2010_x86_64 manylinux1_x86_64 linux_x86_64

fetch "$HERE/requirements-linux-aarch64.txt" "$HERE/wheelhouse-linux-aarch64" \
    manylinux_2_39_aarch64 manylinux_2_35_aarch64 manylinux_2_34_aarch64 \
    manylinux_2_31_aarch64 manylinux_2_28_aarch64 manylinux_2_27_aarch64 \
    manylinux_2_24_aarch64 manylinux_2_17_aarch64 manylinux2014_aarch64 \
    linux_aarch64

fetch "$HERE/requirements-windows-amd64.txt" "$HERE/wheelhouse-windows-amd64" \
    win_amd64

echo
echo "=== wheelhouse sizes ==="
du -sh "$HERE"/wheelhouse-* 2>/dev/null
