#!/usr/bin/env bash
# Resolve the inf_server dependency graph SEPARATELY for each of the three target
# platforms and emit one fully-pinned, marker-RESOLVED requirements file per target:
#
#   requirements-linux-x86_64.txt   (x86_64-unknown-linux-gnu)
#   requirements-linux-aarch64.txt  (aarch64-unknown-linux-gnu)
#   requirements-windows-amd64.txt  (x86_64-pc-windows-msvc)
#
# Why per-platform files instead of the single markered export: `pip download
# --platform` sets wheel *tags* but still evaluates environment markers against the
# HOST interpreter, so from an x86_64 Linux box it would (a) keep the Linux-only
# nvidia-*-cu12 wheels for the Windows/aarch64 targets, (b) pick the wrong
# torchvision variant for aarch64 (needs the bare 0.21.0, not +cu126), and (c) drop
# the Windows-only colorama / pywin32. `uv pip compile --python-platform` runs the
# resolver AS the target, so markers are already evaluated and the emitted list is
# exactly what that platform installs -- no host contamination.
#
# The cuda-ext group is included so detectron2's transitive deps (fvcore, iopath,
# omegaconf, pywin32-on-Windows, ...) land in the list. detectron2 itself resolves
# to a `git+` line (no wheel); fetch_wheelhouses.sh strips it -- detectron2 and the
# vendored MSDeformAttn kernel are compiled on the target (see build_cuda_ext_on_target.*).
#
# Run anywhere with `uv` + network (host arch/OS is irrelevant -- that is the point).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PYPROJECT="$ROOT/pyproject.toml"
CU126="https://download.pytorch.org/whl/cu126"

compile() {   # compile <uv-python-platform> <out-file>
    local plat="$1" out="$2"
    echo "=== resolving for $plat -> $(basename "$out") ==="
    uv pip compile \
        --quiet \
        --python-version 3.11 \
        --python-platform "$plat" \
        --group cuda-ext \
        --extra-index-url "$CU126" \
        --index-strategy unsafe-best-match \
        --emit-index-url \
        "$PYPROJECT" -o "$out"
}

compile x86_64-unknown-linux-gnu  "$HERE/requirements-linux-x86_64.txt"
compile aarch64-unknown-linux-gnu "$HERE/requirements-linux-aarch64.txt"
compile x86_64-pc-windows-msvc    "$HERE/requirements-windows-amd64.txt"

echo
echo "=== done. per-platform package counts (incl. the git detectron2 line) ==="
for f in "$HERE"/requirements-linux-x86_64.txt "$HERE"/requirements-linux-aarch64.txt "$HERE"/requirements-windows-amd64.txt; do
    printf '  %-40s %s pkgs\n' "$(basename "$f")" "$(grep -cE '^[a-zA-Z0-9]' "$f")"
done
