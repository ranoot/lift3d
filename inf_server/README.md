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

## Layout

- `main.py` — `zmq.REP` server; dispatches `set_vocab` / `reset` / `frame` / `ping`.
- `dvis_runner.py` — `DvisRunner`: cfg assembly, model build/load, dynamic text
  classifier, stateful tracker, per-pixel resolution.
- `protocol.py` — msgpack (de)serialization + the wire schema.
- `DVIS_Plus/` — vendored model code + base weights (`ov_online_supervised_convnextl.pth`).

## Setup

```bash
export PATH=/usr/local/cuda/bin:$PATH      # nvcc for the MSDeformAttn build
export CUDA_HOME=/usr/local/cuda

cd inf_server
uv sync                                     # torch cu118 + detectron2 (built from source)

# Compile the MSDeformAttn CUDA kernel used by the pixel decoder, inside the venv:
cd DVIS_Plus/mask2former/modeling/pixel_decoder/ops
uv run python setup.py build install
cd -
```

Notes / gotchas:
- **detectron2** is built from source (no cu118 wheel); `no-build-isolation-package`
  in `pyproject.toml` lets it see the already-installed torch.
- **MSDeformAttn** must be compiled against the venv's torch; the RTX 2080 Ti is
  `sm_75` (set `TORCH_CUDA_ARCH_LIST=7.5` if the build doesn't autodetect).
- The **ConvNeXt-L CLIP** backbone weights (`laion2b_s29b_b131k_ft_soup`) are
  fetched by `open_clip` on first build. Behind a corporate proxy, set
  `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` (as the training env did) or pre-place
  the weights in the open_clip cache.
- The notebook's `patch_clip_text_attention` was a training-time grad trick for
  classifier caching; not needed for no-grad inference.

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
