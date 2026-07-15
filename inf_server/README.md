# inf_server

IPC inference server that abstracts running the fine-tuned **OV-DVIS++ online**
model (open-vocabulary video instance segmentation). It is the 2D producer for
the `lift3d` pipeline: a client streams camera frames, the server runs the
stateful frame-by-frame tracker, and returns a per-pixel label + instance map
(plus per-instance embeddings) that `universe`/`point_pixel_mapping` lift onto
the 3D world map.

Same shape as `mask3d_feat/`: an isolated `uv` venv pinned to the model's
torch/CUDA stack, talking to the rest of the repo over ZeroMQ + msgpack. The
vendored `DVIS_Plus/` tree is put on `sys.path` at import time.

The venv is pinned to **Python 3.11 + torch 2.6.0 + cu126** (`pyproject.toml`).
`detectron2` and the `MSDeformAttn` (`MultiScaleDeformableAttention`) kernel are
CUDA source builds compiled in a second step against the venv's torch — they are
not part of a plain `uv sync` (see below).

## Layout

- `main.py` — `zmq.REP` server; dispatches `set_vocab` / `reset` / `frame` / `ping`.
- `dvis_runner.py` — `DvisRunner`: cfg assembly, model build/load, dynamic text
  classifier, stateful tracker, per-pixel resolution.
- `protocol.py` — msgpack (de)serialization + the wire schema.
- `DVIS_Plus/` — vendored model code + base weights (`ov_online_supervised_convnextl.pth`).

## Setup

```bash
export PATH=/usr/local/cuda/bin:$PATH      # nvcc (needs a CUDA 12.6 toolkit) for the MSDeformAttn build
export CUDA_HOME=/usr/local/cuda

cd inf_server
uv sync                                     # core stack: torch 2.6.0+cu126 (detectron2 NOT synced)

# Second step — the two CUDA source builds against the venv's torch:
#   - detectron2 (its _C ops), pinned in the `cuda-ext` group of pyproject.toml
uv pip install --no-build-isolation --group cuda-ext
#   - MSDeformAttn (pixel-decoder kernel), compiled in place:
cd DVIS_Plus/mask2former/modeling/pixel_decoder/ops
uv run python setup.py build install
cd -
```

Notes / gotchas:
- **torch 2.6.0 + cu126** needs a **CUDA 12.6 toolkit** to build the exts (the repo's
  micromamba `cudatk` env is 11.8 — too old; use a 12.6 toolkit).
- **detectron2** is a CUDA source build (no matching wheel): it lives in the
  non-default `cuda-ext` dependency group and `no-build-isolation-package` in
  `pyproject.toml` lets it see the already-installed torch. Keeping it out of the
  default sync avoids the fresh-venv ordering trap (it needs torch present first).
- **MSDeformAttn** must be compiled against the venv's torch; set
  `TORCH_CUDA_ARCH_LIST` for the GPU (`7.5` for the RTX 2080 Ti dev box). torch 2.6
  removed `Tensor::type()`, so apply `patches/dvis_ms_deform_attn_scalar_type.patch`
  to the `DVIS_Plus` tree first if it isn't already.
- The **ConvNeXt-L CLIP** backbone weights (`laion2b_s29b_b131k_ft_soup`) are
  fetched by `open_clip` on first build. Behind a corporate proxy, set
  `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` (as the training env did) or pre-place
  the weights in the open_clip cache.
- The notebook's `patch_clip_text_attention` was a training-time grad trick for
  classifier caching; not needed for no-grad inference.

## Export / offline install (3 platforms: Linux x86_64, Linux aarch64, Windows x86_64)

To install this stack on another machine offline, ship a wheel bundle. `export/`
builds **three** self-contained wheelhouses — one per target — all pinned to
**Python 3.11 + torch 2.6.0 + CUDA 12.6 (cp311)**. Everything except the two CUDA
source builds (`detectron2`, `MultiScaleDeformableAttention`) is shipped as a
prebuilt wheel; those two have no redistributable wheel and are compiled on the
target against its torch + CUDA 12.6 toolchain.

```bash
# 1. resolve the dep graph SEPARATELY per target (marker-resolved, fully pinned).
#    Runs on any host/arch -- uv evaluates markers AS the target, unlike pip.
bash export/compile_requirements.sh
#    -> export/requirements-{linux-x86_64,linux-aarch64,windows-amd64}.txt

# 2. build the two pure-python sdist-only deps into universal wheels (once), then
#    download every target's binary wheels. Both steps run from ONE x86_64 host --
#    pip --platform only sets wheel tags, and the requirement files are already
#    marker-resolved, so no host contamination.
pip wheel --no-deps -w export/wheelhouse-noarch \
    "antlr4-python3-runtime==4.9.3" "fvcore==0.1.5.post20221221"
bash export/fetch_wheelhouses.sh
#    -> export/wheelhouse-linux-x86_64/  (incl. 13 nvidia-*-cu12 runtime wheels)
#    -> export/wheelhouse-linux-aarch64/ (torch wheel self-contained, no nvidia-*)
#    -> export/wheelhouse-windows-amd64/ (incl. colorama, pywin32; no nvidia-*)

# 3. ON THE TARGET (CUDA 12.6 toolkit on PATH, TORCH_CUDA_ARCH_LIST set): sync the
#    core venv offline from the matching wheelhouse, apply the MSDeformAttn patch,
#    build + install detectron2 + MSDeformAttn.
#    Linux (auto-selects the x86_64 or aarch64 wheelhouse from `uname -m`):
TORCH_CUDA_ARCH_LIST=8.6 bash export/build_cuda_ext_on_target.sh
#    Windows (from an x64 Native Tools / VS 2022 Dev PowerShell):
$env:TORCH_CUDA_ARCH_LIST="8.6"; ./export/build_cuda_ext_on_target.ps1
```

Notes:
- **Per-platform resolution is the crux.** `pip download --platform` sets wheel tags
  but evaluates environment markers against the *host*, so from x86_64 it would keep
  the Linux-only `nvidia-*-cu12` wheels for the Windows/aarch64 bundles, pick the
  wrong `torchvision` for aarch64, and drop the Windows-only `colorama`/`pywin32`.
  `uv pip compile --python-platform` runs the resolver as the target, so each
  requirements file is exactly what that platform installs. That's why all three
  bundles can be produced from a single x86_64 box.
- The **aarch64** torch wheel is self-contained (bundles its CUDA libs, no `nvidia-*`
  deps). The **x86_64** cu126 wheel pulls 13 `nvidia-*-cu12` runtime wheels (staged in
  its wheelhouse). **Windows** torch bundles its CUDA DLLs (no `nvidia-*`).
- `torchvision` is pinned bare `==0.21.0`: the cu126 index publishes the aarch64
  torchvision wheel without the `+cu126` local tag (x86_64/Windows get `0.21.0+cu126`),
  and the lock carries both variants behind `platform_machine` markers.
- Two graph deps ship **only** as sdists (`antlr4-python3-runtime`, `fvcore`); both are
  pure python, pre-built once into `wheelhouse-noarch/` and copied into every bundle.
- `detectron2` is pinned to a git commit in `pyproject.toml` (`cuda-ext` group) so the
  bundle is reproducible; keep the pin in the `build_cuda_ext_on_target.*` scripts in sync.
- **Windows caveat:** detectron2 upstream doesn't officially support Windows. The binary
  wheelhouse installs cleanly; the two CUDA exts build with MSVC v143 + CUDA 12.6 but may
  need minor nudging (see the notes at the bottom of `build_cuda_ext_on_target.ps1`).

## Run

```bash
uv run python main.py --endpoint ipc:///tmp/inf_server.ipc --device cuda
```

## Protocol (msgpack over zmq REP)

Numpy arrays are sent as `{"__ndarray__": true, "shape", "dtype", "data": <bytes>}`
(see `protocol.py`).

| `cmd`       | request fields | reply |
|-------------|----------------|-------|
| `ping`      | —              | `{ok}` |
| `set_vocab` | `thing_classes[]`, `stuff_classes[]` | `{ok, num_classes}` |
| `reset`     | —              | `{ok}` (start a new tracking sequence) |
| `frame`     | `image` = uint8 `(H,W,3)` RGB | `{ok, h, w, label_map(int16 HxW), id_map(int16 HxW), instances[]}` |

`label_map[y,x]` = class id (`-1` = background); `id_map[y,x]` = instance/query id
(`-1` = background). `instances` = `[{id, label, score, embedding(float32 C)}]` for
each instance present in `id_map`. The client gathers per-pixel embeddings via
`id_map[y,x] -> instances[id].embedding` (a dense `H×W×C` float map — ~314 MB at
480×640×256 — is intentionally not sent over IPC).

The server owns preprocessing: it resizes the shortest edge to
`INPUT.MIN_SIZE_TEST` (480) for the network and normalizes with
`PIXEL_MEAN/STD`, then returns maps at the original `(H, W)` so pixel coordinates
line up with the caller's camera image. Send plain RGB.

## Smoke test

Helper scripts are included: `verify.py` exercises `DvisRunner` directly
(build/load/classifier + two-frame forward + reset); `smoke_client.py <endpoint>`
drives a running server over IPC. Or inline:

```bash
uv run python - <<'PY'
import numpy as np, zmq, protocol
s = zmq.Context.instance().socket(zmq.REQ); s.connect("ipc:///tmp/inf_server.ipc")
def call(o): s.send(protocol.pack(o)); return protocol.unpack(s.recv())
print(call({"cmd": "set_vocab",
            "thing_classes": ["chair", "table", "person"],
            "stuff_classes": ["wall", "floor", "ceiling"]}))
img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
r = call({"cmd": "frame", "image": img})
print(r["label_map"].shape, r["id_map"].shape, len(r["instances"]))
print(call({"cmd": "frame", "image": img})["ok"])   # 2nd frame: keep=True path
print(call({"cmd": "reset"}))
PY
```
