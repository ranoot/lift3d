# Python services

Three `uv` projects, each an **isolated venv** pinned to its model's torch/CUDA stack. They
talk to the C++ side only over ZeroMQ + msgpack, which is exactly why their mutually
incompatible stacks can coexist in one repo.

| Service | Socket | Endpoint (default) | Needed? |
|---------|--------|--------------------|---------|
| [`inf_server`](#inf_server) | `zmq.REP` | `ipc:///tmp/inf_server.ipc` | **Required** for any real run |
| [`viz_server`](#viz_server) | `zmq.PULL` | `ipc:///tmp/lift3d_viz.ipc` | Optional |
| [`mask3d_feat`](#mask3d_feat) | `zmq` | `ipc:///tmp/mask3d_feat.ipc` | **Currently dormant** — skip |

All three use the same **ndarray envelope** on the wire, so numpy arrays cross the boundary
without a copy through JSON:

```python
{"__ndarray__": True, "shape": [...], "dtype": "<str>", "data": <raw bytes>}
```

---

## `inf_server`

Open-vocabulary **video** instance segmentation with the fine-tuned OV-DVIS++ online model.
This is the pipeline's 2D producer: `universe`/`point_pixel_mapping` lift its per-pixel output
onto the 3D world map.

### Run

```bash
cd inf_server
uv run python main.py --endpoint ipc:///tmp/inf_server.ipc --device cuda
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--endpoint` | `ipc:///tmp/inf_server.ipc` | ZMQ bind address. Must match `run.yaml`'s `endpoint:`. |
| `--device` | `cuda` | Torch device. |
| `--weights` | `DEFAULT_WEIGHTS` (in `dvis_runner.py`) | Checkpoint path. |
| `--config` | `DEFAULT_CONFIG` (in `dvis_runner.py`) | Model config path. |

### Files

| File | Role |
|------|------|
| `main.py` | The `zmq.REP` server; dispatches `ping` / `set_vocab` / `reset` / `frame`. Also holds the tuning knobs below. |
| `dvis_runner.py` | `DvisRunner`: cfg assembly, model build/load, dynamic text classifier, the stateful tracker, per-pixel resolution. |
| `protocol.py` | msgpack (de)serialization + the wire schema. |
| `DVIS_Plus/` | Vendored model code + base weights. Put on `sys.path` at import time. |
| `verify.py` | Exercises `DvisRunner` directly (build/load/classifier + two-frame forward + reset). |
| `smoke_client.py <endpoint>` | Drives a *running* server over IPC. |
| `build_cuda_ext.sh` | Automates the two CUDA source builds. |
| `export/` | Builds three offline wheelhouses (Linux x86_64 / aarch64, Windows) from one x86_64 host. |
| `panopticapi/` | Local shim (upstream package is awkward to install). |

### Protocol

| `cmd` | Request | Reply |
|-------|---------|-------|
| `ping` | — | `{ok}` |
| `set_vocab` | `thing_classes[]`, `stuff_classes[]` | `{ok, num_classes}` |
| `reset` | — | `{ok}` — begins a new tracking sequence |
| `frame` | `image` = uint8 `(H,W,3)` **RGB** | `{ok, h, w, label_map(int16 HxW), id_map(int16 HxW), instances[]}` |

- `label_map[y,x]` = class id (`-1` = background); `id_map[y,x]` = instance/query id
  (`-1` = background).
- `instances` = `[{id, label, score, embedding(float32 C)}]` for each instance present in
  `id_map`. Per-pixel embeddings are gathered client-side via
  `id_map[y,x] → instances[id].embedding`; a dense `H×W×C` map (~314 MB at 480×640×256) is
  deliberately **not** sent over IPC.
- The server owns preprocessing: it resizes the shortest edge to `INPUT.MIN_SIZE_TEST` (480)
  and normalizes with `PIXEL_MEAN/STD`, then returns maps at the **original** `(H,W)` so pixel
  coordinates line up with the caller's image. Send plain RGB.

> **It is a stateful tracker.** Frames must be fed in order; `reset` starts a new video. The
> C++ client calls `reset` in `OnlineSemantic::begin()`.

### Tuning knobs (edited in `main.py`, not `run.yaml`)

The runner resolves each frame with **VPS panoptic assignment** on top of the FC-CLIP
**geometric ensemble**. (An earlier version used VIS argmax and skipped the ensemble entirely,
which is why novel classes like `sign` and `extinguisher` failed to appear.) These are the
levers for open-vocabulary recall:

| Constant | Default | Effect |
|----------|---------|--------|
| `OBJECT_MASK_THRESH` | `0.10` | VPS keep gate — drop queries whose winning class score is below this. |
| `OVERLAP_THRESH` | `0.7` | Per-segment stability: min `kept / original` mask-area ratio. |
| `ENSEMBLE_ALPHA` | `0.5` | Geometric-ensemble weight for **seen** (in-training) classes. |
| `ENSEMBLE_BETA` | `1.3` | Geometric-ensemble weight for **unseen** classes. Raise toward 1.0 to trust CLIP more for novel classes. |

### `SYNONYMS`

A per-class dict in `main.py`. Key = the canonical class name **exactly** as spelled in
`run.yaml`'s `thing:`/`stuff:` list; value = extra phrasings. The model embeds every synonym and
keeps the best-matching one per pixel (max-ensemble), so more phrasings only raise recall.

```python
SYNONYMS = {
    "extinguisher": ["fire extinguisher", "red fire extinguisher", ...],
    "sign":         ["signage", "placard", "wall sign", "sign with text", ...],
}
```

Two rules:

1. **Synonyms stay on the Python side.** The C++ never sees them, so object labels, topics and
   dynamic-class matching keep the clean canonical name. **Do not put synonyms in `run.yaml`.**
2. Expansion happens **after** the canonical sort, so label-id ordering holds.

---

## `viz_server`

The visualizer half of the viz/logic split. `run_semantic_universe` streams pipeline state over
ZMQ + msgpack; this process renders it with [rerun](https://rerun.io). **All rerun/Arrow
dependencies live here**, so the C++ build carries none and configures on locked-down systems.

Pure Python — no compiled extensions, no GPU. `rerun-sdk` is pinned to `0.33.1` (the version
the removed C++ SDK used) so recordings stay format-compatible.

### Run

**Start it first** — it *binds* the PULL socket.

```bash
# live viewer
uv run --project viz_server viz_server/main.py --spawn
./build/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc

# record to a .rrd instead
uv run --project viz_server viz_server/main.py --out /tmp/sem.rrd --exit-on-finish
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--endpoint` | `ipc:///tmp/lift3d_viz.ipc` | ZMQ PULL bind address. Must match the C++ `--viz` / `viz_endpoint`. |
| `--spawn` | off | Spawn a live rerun viewer. |
| `--out` | `""` | Also save the recording to this `.rrd`. |
| `--port` | `9876` | Port for the spawned viewer. |
| `--exit-on-finish` | off | Exit after the run's `finish` message (useful for scripted recordings). |

### Message catalogue

Defined in `protocol.py`; produced by `universe/viz_publisher.cpp`.

```
{"type": "begin",  "classes": [{"id", "name", "kind"}, ...],   # Universe vocabulary
                   "seg_vocab": [str, ...],                    # index == label_map class id
                   "seg_overlay": str, "seg_alpha": float, "seg_min_area": int}

{"type": "frame",  "idx": int, "capture_secs": float,
                   "robot_pose": f4(16), "robot": f4(3),
                   "cam": {"K": f4(9), "T": f4(16), "w": int, "h": int},
                   "image": u1(H,W,3),                              # optional
                   "seg": {"label_map": i2(H,W), "id_map": i2(H,W)},# optional
                   "objects": [entity, ...],                        # optional
                   "proposals": [entity, ...],                      # optional
                   "superpoints": [entity+kind, ...]}               # only on VCCS refresh

{"type": "finish", "positions": f4(N,3), "class_ids": i2(N,), "kinds": i1(N,)}
```

where `entity = {"id", "class_id", "name", "centroid": f4(3), "points": f4(M,3)}`. Objects also
carry `"level"` (the component's member-seed count — its multi-view support/confidence) and may
carry `"text"` (free text attached to the object, e.g. sign content).

Per-frame entities are logged on the timeline and overwrite each frame; **the full world map
arrives once at `finish` and is logged as a static snapshot** — per-frame re-logging made
recordings O(frames × map_size).

Transport is PUSH→PULL and non-blocking: per-frame messages are sent `DONTWAIT` and **dropped
on backpressure**, because a live viewer only needs the latest. A slow viewer therefore skips
frames rather than stalling the pipeline.

---

## `mask3d_feat`

> **Dormant.** The Mask3D deep-feature affinity was replaced by SAI3D class-histogram affinity,
> which is derived from the accumulated per-point votes and costs nothing extra — measurements
> showed the Mask3D backbone dominated per-view cost (0.7–1.9 s versus ~200–320 ms for
> VCCS + HDBSCAN). There is **no `FeatClient` or `PointFeatures` in the current C++ source** and
> **no `features:` block in `run.yaml`**. You do not need to run this. It is documented here for
> reference and possible revival.

Serves the Res16UNet34C **backbone** of Mask3D — never the full head — turning a sparse voxel
cloud into 96-d per-voxel feature vectors.

Pinned to **Python 3.10 + torch 2.0.1 + cu118**, because that is what MinkowskiEngine 0.5.4 can
build against. `build_me.sh` builds the extension; the checkpoint is expected at
`Mask3D/checkpoints/scannet_val.ckpt`.

`main.py` takes **no CLI arguments** — the endpoint is the module constant
`ipc:///tmp/mask3d_feat.ipc`.

### Protocol

| `cmd` | Request | Reply |
|-------|---------|-------|
| `ping` | — | `{ok: True}` |
| `features` | `coords` int32 `(N,3)`, `feats` float32 `(N,C)` | `{ok, coords: int32 (M,4), feats: float32 (M,96)}` |

- `coords` are **voxel** coordinates = `round(xyz / voxel)`; `feats` is **normalized RGB**.
  Colour is a required input (ScanNet normalization `feat = (rgb/255 − mean)/std`).
- MinkowskiEngine **coalesces duplicate voxel coordinates**, so `M` may be `< N` — the response
  includes the output coords precisely so the caller can realign features to voxels.
- Any handler error returns `{"ok": False, "error": "<message>"}`.

### Two import hacks worth knowing about

Both happen before any Mask3D import in `main.py`:

1. Mask3D's `models/__init__.py` eagerly imports `hydra`, `torch_scatter` and the compiled
   `third_party.pointnet2` extension — none installed in this venv, none needed for the
   backbone. Lightweight stubs are pre-seeded into `sys.modules` so the real module still
   imports (keeping the package `__init__` intact) while its heavy deps stay inert.
2. `Res16UNet34C` requires a `config` object (attribute access for `bn_momentum`,
   `conv1_kernel_size`, `dilations`), supplied as a minimal `SimpleNamespace` with the ScanNet
   backbone values from `conf/model/mask3d.yaml`.

It also binds a `REQ` socket, which is the *wrong* role for a request/reply server —
`inf_server` uses `REP` and is the pattern to copy if this is ever revived.
