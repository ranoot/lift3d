# Build options and targets

## CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `LIFT3D_USE_CUDA` | `OFF` | Selects the **CUDA** back end for `point_pixel_mapping` instead of the CPU one. Enables the `CUDA` language and `CMAKE_CUDA_STANDARD 20`. |
| `LIFT3D_USE_SIGN` | `OFF` | Compiles the **real VLM** sign detector backend (`sign-unds/` + `sign_detector_real.cpp`) instead of the mock. Pulls in CURL, OpenCV, libjpeg, libpng and nlohmann/json. |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90` | GPU targets. The card isn't detectable at configure time under WSL, hence the broad default. **Override when you know the card** (`-DCMAKE_CUDA_ARCHITECTURES=86`) for a much faster build. |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | forced `ON` | For clangd. `compile_commands.json` at the repo root is a symlink into the build dir. |

`CMAKE_CXX_STANDARD` is fixed at **20** and required — the z-buffer's `cuda::atomic_ref` and
`cuda::std::bit_cast` depend on it.

### `LIFT3D_USE_CUDA` in detail

Exactly one back end is compiled into the `point_pixel_mapping` static library; the public API
is identical either way, so nothing downstream changes.

```bash
cmake -S . -B build                              # point_pixel_mapping_cpu.cpp
cmake -S . -B build-cuda -DLIFT3D_USE_CUDA=ON    # point_pixel_mapping.cu
```

The CUDA target sets `CUDA_SEPARABLE_COMPILATION ON` **and** `CUDA_RESOLVE_DEVICE_SYMBOLS ON`.
The second matters: it performs the CUDA device link when the *static library* is built, so
plain C++ executables can link it without an `__cudaRegisterLinkedBinary` undefined reference
at their own link step.

**Use a separate build directory for each.** Sharing one silently leaves you running a binary
built with the other back end.

### `LIFT3D_USE_SIGN` in detail

The sign **bridge** (`sign_adaptor/sign_bridge.cpp` — mask scan, single-in-flight gate, 3D
footprint, object matching, logging) is **always built**; it needs nothing beyond `universe`.
Only the **detector backend** is conditional:

| | `OFF` (default) | `ON` |
|---|---|---|
| backend TU | `sign_detector_mock.cpp` | `sign_detector_real.cpp` + `sign-unds/` chain |
| behaviour | sleeps `mock_delay_ms`, answers `mock_text` | talks to the real VLM at `vlm_url` |
| extra deps | none | CURL, OpenCV, JPEG, PNG, nlohmann/json |
| define | — | `LIFT3D_HAVE_SIGN` |

This means the **whole sign path** — logs, `directionContent`, rerun labels — can be exercised
on any machine with the default build. Either way nothing runs unless `run.yaml` sets
`sign.enabled: true`.

Use a separate build dir (e.g. `build-sign`) when turning it on, so the normal build stays free
of the OpenCV/CURL/VLM dependency set.

> `sign-unds/GmdTimeStubs.cpp` is intentionally excluded from the source list — its
> `Common::Time` symbols are already defined by `gmd_types`' real `TimeStamp.cpp`, and
> compiling both duplicates them.

---

## Targets

### Libraries

| Target | Sources | Links | Notes |
|--------|---------|-------|-------|
| `point_pixel_mapping` | one of `.cu` / `_cpu.cpp` | — | PCL-free, dependency-light. |
| `dog_log_adaptor` | log decode + `pose_interp` + `dog_stream` + `dog_log_ingestor` | Eigen (PRIVATE) | PCL-free. |
| `inf_client` | `inf_client.cpp` | system libzmq | Headers from the `cppzmq`/`libzmq` submodules; **libzmq itself is never built**. |
| `universe` | `universe.cpp` + `superpoints.cpp` | `point_pixel_mapping`, PCL, `inf_client` | VCCS comes from `pcl_segmentation`, already in `PCL_LIBRARIES`. |
| `pargeo_hdbscan` | `hdbscan/src/{hdbscan,dendrogram}.cpp` | Threads | All dims 2–20 instantiated (one-time cost). `HOMEGROWN` selects parlay's built-in scheduler; built `-O3`. |
| `object_seeds` | `object_seeds` + `object_grow` + `object_consolidate` + `hdbscan_extract` | `universe` PUBLIC, `pargeo_hdbscan` **PRIVATE** | `object_seeds.cpp` is the only TU pulling in parlay — hence the PRIVATE link, so parlay's heavy includes never leak to the drivers. |
| `run_config` | `run_config.cpp` | `universe`, `dog_log_adaptor`, `inf_client` PUBLIC; `yaml-cpp` **PRIVATE** | Also exposes `sign_adaptor/` on the include path for the header-only `sign_config.h`. |
| `viz_publisher` | `viz_publisher.cpp` | `universe`, `dog_log_adaptor`, `inf_client`, libzmq, PCL | PUSH-socket msgpack publisher. |
| `sign_adaptor` | `sign_bridge.cpp` + one backend | `universe`, `dog_log_adaptor` | See above. |
| `gmd_types` | `TimeStamp.cpp`, `Transformation.cpp` | Eigen | Only the **OpenCV-free** GMD entity TUs — `ImageUtils.cpp`/`PoseUtils.cpp` are skipped. |
| `gmd_adaptor` | `gmd_frame_source.cpp`, `object_publisher.cpp` | `gmd_types`, `dog_log_adaptor`, Eigen | PCL/CUDA/server-free. |
| `semantic_runner` | `semantic_runner.cpp` | everything above + `gmd_adaptor` | The all-in-one `runOnInput` entry point behind a PImpl. |

All libraries are built `POSITION_INDEPENDENT_CODE ON`.

### Executables

| Target | Requires at runtime |
|--------|---------------------|
| `run_semantic_universe` | `inf_server` + a log. The main driver. |
| `run_on_input_demo` | `inf_server` + a log. Replays a recording through `SemanticRunner::runOnInput`. Always compiles. |
| `exp_per_view` | `inf_server` + a log. Throwaway per-view cost / DVIS id-overlap measurement. |
| `gmd_stream_check` | **nothing** — no PCL, no GPU, no servers. |
| `object_publisher_check` | **nothing** — no PCL, no GPU, no servers. |

Both drivers link `sign_adaptor`; it stays dormant unless `run.yaml` enables it.

---

## Vendored dependency policy

The build is deliberately arranged so a **locked-down / offline machine can configure it**:

- **rerun and Apache Arrow are gone entirely.** All visualization moved to the Python
  `viz_server`. This was the point of the viz/logic split — Arrow could not be built on the
  target systems.
- **yaml-cpp** is a pinned submodule (tag 0.8.0) built via `add_subdirectory` with tests, tools
  and contrib disabled — no system `-dev` package and no network fetch. yaml-cpp 0.8.0 still
  declares `cmake_minimum_required(VERSION 3.4)`, which newer CMake rejects, so
  `CMAKE_POLICY_VERSION_MINIMUM` is raised **just for its configure** and unset afterwards.
- **cppzmq / libzmq** submodules supply **headers only**. The system libzmq *runtime* is
  linked, so no `-dev` package is needed and libzmq is never compiled. If the find fails, the
  build globs common `libzmq.so.5*` locations before erroring; you can also point
  `-DZMQ_LIBRARY=` at it directly.
- **`FetchContent` is used for exactly one thing**: nlohmann/json, and only inside the
  `LIFT3D_USE_SIGN` block, and only if no system package is found.

---

## Build gotchas

- **Configure is slow (~6 min)** — PCL dependency probing. Reuse the build dir; only `rm -rf`
  when changing project-level settings.
- **The `project()` line must keep `C`.** PCL → VTK 9.5 fails configure on a missing
  `MPI::MPI_C` target without it.
- **Rebuild the build dir you actually run.** `build/` and `build-cuda/` are independent.
- **`nvcc` is not on `PATH`** — `export PATH=/usr/local/cuda/bin:$PATH`.
- The vendored HDBSCAN spams stdout on every call; the pipeline silences it by redirecting
  `std::cout` for the clustering scope.
