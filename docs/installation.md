# Installation

Three things get installed independently: **system packages**, the **C++ build**, and one
**isolated Python venv per service**. The Python venvs are pinned to their models' torch/CUDA
stacks and talk to the C++ side only over ZeroMQ + msgpack, so their (mutually incompatible)
stacks never interact with each other or with the C++ build.

---

## 1. System prerequisites

| Dependency | Needed for | Notes |
|------------|-----------|-------|
| CMake ≥ 3.18 | everything | |
| A C++20 compiler | everything | The z-buffer's `cuda::atomic_ref` / `bit_cast` require C++20. |
| **PCL ≥ 1.3** | `universe` and above | Pulls in VTK — see the `C`-language gotcha below. |
| **Eigen 3.3+** | pose math | |
| **libzmq runtime** | IPC | The *runtime* only (`libzmq.so.5`) — no `-dev` package needed, headers come from the vendored submodules. |
| pthreads | parlay's scheduler | |
| CUDA toolkit | only `-DLIFT3D_USE_CUDA=ON` | `nvcc` at `/usr/local/cuda`. |
| `uv` | the Python services | https://docs.astral.sh/uv/ |

On Debian/Ubuntu:

```bash
sudo apt install cmake build-essential libpcl-dev libeigen3-dev libzmq5
```

Optional, only for a `-DLIFT3D_USE_SIGN=ON` build: `libcurl`, OpenCV, libjpeg, libpng, and
nlohmann/json (fetched automatically if no system package is found).

---

## 2. Clone with submodules

The build needs the vendored submodules — `yaml-cpp`, `cppzmq`, `libzmq` and `hdbscan` are
**required** to configure at all.

```bash
git clone <repo> lift3d && cd lift3d
git submodule update --init --recursive
```

If you only want the C++ build and not the Python model code (the model submodules are large):

```bash
git submodule update --init yaml-cpp cppzmq libzmq hdbscan
```

Add `sign-unds` if you intend to build with `-DLIFT3D_USE_SIGN=ON`.

---

## 3. Build the C++ side

```bash
export PATH=/usr/local/cuda/bin:$PATH        # only needed for the CUDA back end

# CPU back end — the default; builds and runs without a GPU
cmake -S . -B build && cmake --build build -j

# CUDA back end — a SEPARATE build dir, so the two never clobber each other
cmake -S . -B build-cuda -DLIFT3D_USE_CUDA=ON && cmake --build build-cuda -j2
```

**Configure takes ~6 minutes**, dominated by PCL's dependency probing. Reuse the build
directory for incremental rebuilds; only `rm -rf` it when changing project-level settings.

Verify with the two GPU-free smoke tests — they need no PCL servers, no GPU, no log:

```bash
./build/gmd_stream_check           # GMD -> pipeline sync/lift/pixel translation
./build/object_publisher_check     # object egress: 2D hull + keyed topic
```

Build options, targets, and per-target dependencies: [build-options.md](build-options.md).

---

## 4. Set up the Python services

Each directory is a self-contained `uv` project. See
[python-services.md](python-services.md) for what each one does and its options.

### 4.1 `viz_server` — the visualizer (easy, no GPU)

Pure Python, no compiled extensions, no GPU.

```bash
uv sync --project viz_server
```

Python ≥ 3.10; `rerun-sdk` is pinned to `0.33.1` so recordings stay format-compatible with the
(now removed) C++ SDK.

### 4.2 `inf_server` — 2D segmentation (required for a real run, GPU)

Pinned to **CUDA 12.6 torch 2.6.0** (`requires-python` and `.python-version` pin Python 3.10;
the prose in `inf_server/README.md` says 3.11 — the pyproject is what `uv` obeys). Two of its
dependencies are **CUDA source builds** and are deliberately *not* part of a plain `uv sync`,
because they must compile against the already-installed venv torch:

```bash
export PATH=/usr/local/cuda/bin:$PATH        # needs a CUDA 12.6 toolkit
export CUDA_HOME=/usr/local/cuda

cd inf_server
uv sync                                       # core stack (detectron2 NOT synced)

# Step 2 — the two CUDA extensions:
uv pip install --no-build-isolation --group cuda-ext          # detectron2 and its _C ops
cd DVIS_Plus/mask2former/modeling/pixel_decoder/ops
uv run python setup.py build install                          # MSDeformAttn kernel
cd -
```

`build_cuda_ext.sh` automates step 2.

Known traps (the full list is in [`inf_server/README.md`](../inf_server/README.md)):

- **A CUDA 12.6 toolkit is required** to build the extensions. An 11.8 toolkit is too old.
- Set `TORCH_CUDA_ARCH_LIST` for your card (`7.5` for the RTX 2080 Ti dev box).
- torch 2.6 removed `Tensor::type()`, so apply `patches/dvis_ms_deform_attn_scalar_type.patch`
  to `DVIS_Plus` first if it isn't already applied.
- The **ConvNeXt-L CLIP** backbone weights are fetched by `open_clip` on first build. Behind a
  proxy, set `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` or pre-place them in the open_clip cache.

Smoke test the server: `uv run python verify.py` (drives `DvisRunner` directly) or
`uv run python smoke_client.py <endpoint>` (drives a running server over IPC).

#### Offline / air-gapped install

`inf_server/export/` builds **three** self-contained wheelhouses — Linux x86_64, Linux
aarch64, Windows x86_64 — all from a **single x86_64 host**, because `uv pip compile
--python-platform` runs the resolver *as* the target (unlike `pip download --platform`, which
only sets wheel tags and evaluates markers against the host). Everything except the two CUDA
source builds ships prebuilt. See the "Export / offline install" section of
[`inf_server/README.md`](../inf_server/README.md) for the full procedure.

### 4.3 `mask3d_feat` — currently dormant, usually skip

The Mask3D feature tier was replaced by SAI3D class-histogram affinity; there is no
`FeatClient` in the current C++ source and no `features:` block in `run.yaml`.
**You do not need this server to run the pipeline.** It is kept for reference and possible
revival.

If you do want it: it is pinned to **Python 3.10 + torch 2.0.1 + cu118**, because that is what
MinkowskiEngine 0.5.4 can build against. `uv sync --project mask3d_feat`, then `build_me.sh`
for the MinkowskiEngine extension. It also needs the ScanNet checkpoint at
`mask3d_feat/Mask3D/checkpoints/scannet_val.ckpt`.

---

## 5. Data

The pipeline reads a **dog log folder** plus a **calibration XML**, both set in `run.yaml`.

```
~/dog_logs/20260608_test_log_officePantryNarrowCorridor/
    PoseAndPointClouds.bin      # MLOAMC LiDAR scans + poses (body frame; pose is body->world)
    camera211.gclf              # the image stream for sensor id 211
~/dog_logs/CameraCalRaiboAsQUGV113.xml    # in the log's PARENT dir; calib: "" resolves here
```

**Camera choice matters:** cams **111 / 211 are pinhole** (`xi=0`, no distortion) and are the
correct ones. Cams **11 / 21 are fisheye** and need an omnidirectional model the pipeline does
not implement.

---

## 6. Run

Order matters: **start the servers before the pipeline**, and start `viz_server` before the C++
side because it *binds* the PULL socket.

```bash
# 1. 2D segmentation server (required)
cd inf_server && uv run python main.py --endpoint ipc:///tmp/inf_server.ipc --device cuda

# 2. visualizer (optional) — live viewer, or --out to record a .rrd
uv run --project viz_server viz_server/main.py --spawn

# 3. the pipeline
./build/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc
```

Without `--viz` (and with an empty `viz_endpoint`) the run is headless and pays nothing for
visualization. If a run prints *"inf_server not responding"*, the server isn't up or the GPU is
down.

---

## 7. Gotchas

### GPU / WSL2

The GPU (RTX 2080 Ti on the dev box) **does** execute CUDA, but WSL passthrough is
**intermittent and host-side**. When it breaks, `nvidia-smi` fails with *"Failed to initialize
NVML"* and CUDA programs fail at runtime with *"CUDA driver version is insufficient…"*.

- **Fix:** refresh the **Windows-host** NVIDIA driver + `wsl --update`, then **restart the
  Python IPC servers**.
- **Never install a Linux NVIDIA driver inside WSL** — it breaks passthrough further.
- `nvcc` **compiles** fine regardless, so verify CUDA changes by compiling; don't assume you
  can run without checking passthrough first.

### `nvcc` is not on `PATH` by default

The toolkit lives at `/usr/local/cuda` but the installer doesn't export its `bin/`. `~/.bashrc`
sets `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` so new shells are fine; otherwise prepend
`/usr/local/cuda/bin` manually.

### PCL pulls in VTK, which needs the `C` language enabled

The CMake `project()` line **must** declare `C` (it declares `LANGUAGES C CXX`, adding `CUDA`
conditionally). Without `C`, VTK 9.5 fails configure on a missing `MPI::MPI_C` target.

### Two build directories

`build/` (CPU) and `build-cuda/` (CUDA) both exist. **Rebuild the one you actually run** —
editing source and rebuilding `build/` while running a stale `build-cuda/` binary has bitten
before.

### clangd

`.clangd` is in the repo and papers over two traps: (a) `gcc-16` is installed but its libstdc++
headers are not, so clang auto-picks GCC 16 and every `std::` symbol vanishes — pinned with
`--gcc-install-dir=…/15`; (b) CUDA 13 moved libcu++ into `include/cccl/`, so `<cuda/atomic>`
needs that extra `-I`. nvcc-only flags in `compile_commands.json` are stripped there too.

Validate outside the editor with `clangd --check=point_pixel_mapping.cu` (the
`tweak: DefineInline` lines it reports are not real errors).

`compile_commands.json` is a **symlink into `build/`**, regenerated by CMake. If clangd reports
`Root directory: nil`, the symlink is stale — re-point it at `build/compile_commands.json`.

### HDBSCAN thread count

The vendored parallel HDBSCAN had a crash when the parlay worker count exceeded the point count
(zero-capacity buffer → SIGSEGV). It is fixed in-tree, but on boxes with more than 16 cores the
mitigation `PARLAY_NUM_THREADS<=16` is a safe belt-and-braces setting.
