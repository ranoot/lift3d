# lift3d — online 2D→3D semantic mapping

`lift3d` turns a quadruped's raw sensor stream into a **persistent, semantically labelled 3D
map with discovered object instances** — online, while the robot walks.

A dog streams LiDAR scans + poses (~10 Hz) and camera images (~5 Hz). The pipeline fuses the
LiDAR into a voxel-deduplicated world point map ("the Universe"), runs an open-vocabulary 2D
segmentation model on every image, **lifts** those 2D labels onto the 3D points through a
self-reconstructed z-buffer, and discovers object instances via a three-tier hierarchy
(VCCS superpoints → per-class HDBSCAN\* proposals → Union-Find object consolidation).

The structure follows **SGS-3D** (arXiv:2509.05144v2) with **SAI3D**-style affinity growing,
adapted from "static, complete scene" to **online and grow-only**.

```
                 ┌──────────────┐        ┌──────────────┐
   LiDAR+pose ──►│ FrameSource  │        │  inf_server  │  (Python, GPU)
   camera     ──►│  + DogStream │        │  OV-DVIS++   │
                 │ (time sync)  │        └──────┬───────┘
                 └──────┬───────┘   ZMQ+msgpack │ label_map / id_map
                        │ SyncedFrame           ▼
                        └────────────►┌───────────────────────┐
                                      │   OnlineSemantic      │
                                      │  fuse → project/vote  │
                                      │  → superpoints → HDB* │
                                      │  → grow → objects     │
                                      └───┬───────────────┬───┘
                       EAIRoomObject list │               │ ZMQ+msgpack
                        / labelled PCD ◄──┘               ▼
                                                  ┌──────────────┐
                                                  │  viz_server  │ (Python, rerun)
                                                  └──────────────┘
```

## Quickstart

```bash
git submodule update --init --recursive
export PATH=/usr/local/cuda/bin:$PATH          # nvcc is not on PATH by default

# C++ build (CPU back end; ~6 min configure because of PCL/VTK probing)
cmake -S . -B build && cmake --build build -j

# GPU-free smoke tests — no PCL servers, no GPU, no log needed
./build/gmd_stream_check
./build/object_publisher_check

# Full run: start the 2D segmentation server, then (optionally) the visualizer, then the pipeline
cd inf_server && uv run python main.py --endpoint ipc:///tmp/inf_server.ipc --device cuda &
uv run --project viz_server viz_server/main.py --spawn &
./build/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc
```

Full details in [docs/installation.md](docs/installation.md).

## Documentation

| Document | What's in it |
|----------|--------------|
| [docs/architecture.md](docs/architecture.md) | **Every module, what it owns, and how they connect.** Start here. |
| [docs/installation.md](docs/installation.md) | System packages, submodules, C++ build, the Python venvs. |
| [docs/build-options.md](docs/build-options.md) | CMake options, every target, vendored deps, build gotchas. |
| [docs/configuration.md](docs/configuration.md) | Complete `run.yaml` reference + CLI flags for every executable. |
| [docs/python-services.md](docs/python-services.md) | `inf_server`, `viz_server`, `mask3d_feat`: options, wire protocols, tuning knobs. |
| [doc.md](doc.md) | **Algorithmic design reference** — the math and the reasoning at each stage. |
| [CLAUDE.md](CLAUDE.md) | Condensed build/gotcha notes for agents working in this repo. |

## Repository layout

| Path | Role |
|------|------|
| `point_pixel_mapping/` | The SGS-3D lifting core — z-buffered point→pixel projection. **PCL-free**, CPU + CUDA back ends behind one API. |
| `universe/` | The pipeline: world map, semantic voting, superpoints, proposals, growing, object consolidation, config loader, viz egress, drivers. |
| `dog_log_adaptor/` | Offline `~/dog_logs` decoding + the transport-agnostic sync layer (`PoseInterpolator`, `DogStream`, `FrameSource`). |
| `gmd_adaptor/` | Robot deployment interface: GMD entity types ↔ pipeline (frame ingress + `EAIRoomObject` egress). |
| `gmdDataTypesForInterns2/` | Vendored robot message/entity type definitions. |
| `inf_client/` | C++ ZMQ+msgpack client for `inf_server` (hand-rolled msgpack subset). |
| `sign_adaptor/` | Async sign-text egress bridge; mock or real VLM detector backend. |
| `inf_server/` | **Python.** OV-DVIS++ open-vocabulary segmentation IPC server (the 2D producer). |
| `viz_server/` | **Python.** rerun visualizer; consumes the C++ viz stream. |
| `mask3d_feat/` | **Python.** Mask3D backbone feature server. *Currently dormant* — see architecture doc. |
| `hdbscan/`, `yaml-cpp/`, `cppzmq/`, `libzmq/`, `sign-unds/` | Vendored submodules. |
| `tests/` | Two GPU-free smoke checks. |

## Known constraints

- **Pinhole cameras only.** Cams `111`/`211` are pinhole and correct for this pipeline;
  `11`/`21` are fisheye and need an omnidirectional model that is not implemented.
- **Grow-only, index-stable map.** Points are never erased or reindexed — "deletion" is
  tombstoning. Every tier keys on the global point index.
- **Objects never un-merge.** Consolidation is monotonic.
- **GPU passthrough under WSL2 is intermittent** (host-side); `nvcc` always compiles
  regardless. See [docs/installation.md](docs/installation.md#gpu--wsl2).
