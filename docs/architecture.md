# Architecture — modules and how they connect

This document is the **module map**: what each unit owns, what it depends on, and the exact
seam it talks to its neighbours across. For the *algorithms* (projection math, voting rules,
affinity scoring) see [`doc.md`](../doc.md); for the knobs see
[configuration.md](configuration.md).

---

## 1. The shape of the system

Three processes, wired over ZeroMQ + msgpack:

```
┌───────────────────────────── C++ process ─────────────────────────────┐
│                                                                       │
│  FrameSource ──► DogStream ──► OnlineSemantic ──► object list         │
│  (ingress)       (time sync)   (the pipeline)     (egress)            │
│                                     │                                 │
└─────────────────────────────────────┼─────────────────────────────────┘
                        REQ/REP ──────┼────── PUSH/PULL
                            ▼                     ▼
                   ┌─────────────────┐   ┌─────────────────┐
                   │   inf_server    │   │   viz_server    │
                   │ Python + GPU    │   │ Python + rerun  │
                   └─────────────────┘   └─────────────────┘
```

Only `inf_server` is **required** for a real run: the pipeline cannot label anything without
2D segmentation. `viz_server` is optional (an empty viz endpoint makes every publish call a
no-op with zero overhead).

### The load-bearing invariant

The world map is **append-only and index-stable**. Every downstream structure — vote
histograms, superpoint membership, proposal membership, object membership — stores **global
point indices** into that map. Nothing may erase or reindex a point; soft deletion is
*tombstoning* (`Universe::killPoint`), which keeps the index and skips the point in every
read-time enumeration. A later fuse into the same voxel revives the slot.

Every stage therefore follows the same contract: *read a const `Universe`, operate on a local
crop, store results by global index, never mutate or reindex the map.*

---

## 2. Layer by layer

### 2.1 `point_pixel_mapping/` — the lifting core

The SGS-3D point-to-pixel projection. **Deliberately PCL-free and dependency-light** so any
front end can reuse it.

| File | Contents |
|------|----------|
| `point_pixel_mapping.cuh` | The whole public API: the `Camera` struct (row-major 3×3 `K`, row-major 4×4 camera-to-world **rigid** `T`, image `W`/`H`) and `projectPointsToPixels(...)`. |
| `point_pixel_mapping.cu` | CUDA back end: `projectKernel` (project + atomically z-buffer the nearest depth, with a depth splat) and `visibilityKernel` (visible iff in-frame and within `tau` of the buffered depth). |
| `point_pixel_mapping_cpu.cpp` | CPU back end, **same API**. Built by default. |

Exactly one of the two back ends is compiled into the `point_pixel_mapping` static library,
selected by `-DLIFT3D_USE_CUDA`. Callers link the library and never know which one they got.

Two conventions the rest of the repo depends on:

- **`Camera.T` is camera-to-world** and is inverted internally (closed-form rigid inverse).
  A caller supplying world-to-camera must skip the inverse.
- The **z-buffer stores depths as ints** via `cuda::std::bit_cast` + `cuda::atomic_ref::fetch_min`
  (IEEE-754 bit patterns are monotonic for positive depths). This is why the project requires
  **C++20**.

### 2.2 `dog_log_adaptor/` — input decoding and time sync

Two separable concerns live here.

**(a) Log decoding** — `dog_log_adaptor.{h,cpp}`, `euler_pose.h`. `DogLogReader` indexes a
`~/dog_logs` folder: `PoseAndPointClouds.bin` (MLOAMC scans — 60-byte header, then 13-byte
points, body frame, body→world extrinsic-XYZ euler pose) and `camera<id>.gclf` (the image
stream). It also parses the calibration XML into a `CameraCalib` (projection `fu,fv,cu,cv`,
image size, and the camera-optical→body mount from an offset + quaternion).

**(b) The sync layer** — the interesting part, and transport-agnostic:

| Unit | Role |
|------|------|
| `PoseInterpolator` (`pose_interp.{h,cpp}`) | Buffers body→world poses and interpolates to an arbitrary timestamp — **SLERP on rotation, LERP on translation**. |
| `DogStream` (`dog_stream.{h,cpp}`) | The push-based entry point. Two calls only: `pushScan(t, pose6, xyz_body, intensity, n)` and `pushImage(DogImage)`. For each image it interpolates the body pose to the **image's exact timestamp**, composes `T_world_cam = T_world_body(image_t) · T_body_cam`, attaches the nearest world-lifted scan, and emits a **`SyncedFrame`**. Policy: *wait for the bracketing pose* — an image is held until a pose with `t ≥ image_t` arrives, so interpolation is always two-sided; unbracketable images are dropped. PCL-free. |
| `FrameSource` (`frame_source.h`) | Abstract base that **owns** the `DogStream` engine and hides it. A concrete source adds only its transport-specific typed ingest methods. The engine is built lazily in `setSyncedHook`, so `setSyncedHook` (or `OnlineSemantic::attach`) must precede any push. |
| `LogFrameSource` (`log_frame_source.h`) | Offline replay as a `FrameSource`. |
| `DogLogIngestor` (`dog_log_ingestor.{h,cpp}`) | Replays a recording into a `DogStream` through the *same* `pushScan`/`pushImage` a live driver would use, merging the 10 Hz scan and 5 Hz image streams into one increasing-timestamp order. |

**Why this matters:** poses arrive at ~10 Hz and images at ~5 Hz, so an image's capture time
almost never lands on a pose sample. Projecting through the nearest *scan's* pose introduced
up to ~100 ms of lag while the robot turned, smearing labels across the map. Everything
downstream lifts labels through the camera pose *at the image instant*.

### 2.3 `inf_client/` — the 2D segmentation client

`InfClient` speaks ZeroMQ REQ + msgpack to the Python `inf_server`. Inference cannot run in
C++, so the model lives behind an IPC server and this is the consumer side.

```cpp
InfClient inf("ipc:///tmp/inf_server.ipc");
inf.setVocab({"door","chair"}, {"wall","floor"});  // -> num_classes
FrameResult r = inf.frame(rgb, H, W);              // rgb = HxWx3 uint8
int16_t cls = r.labelAt(u, v);                     // class id, -1 = background
```

`msgpack_lite.h` is a hand-rolled msgpack subset (pack/unpack of the maps, arrays and the
ndarray envelope this repo actually uses) — no msgpack-c dependency. It is reused by
`viz_publisher` for the outbound viz stream.

`FrameResult` carries `label_map` (per-pixel class), `id_map` (per-pixel instance), and the
per-instance records including embeddings. A dense `H×W×C` embedding map is deliberately
*not* sent over IPC (~314 MB at 480×640×256); the client gathers embeddings via
`id_map[y,x] → instances[id].embedding`.

### 2.4 `universe/` — the pipeline

This is the bulk of the system. Compiled as several static libraries so the heavy
dependencies stay confined.

#### `Universe` (`universe.{h,cpp}`) — the persistent world

A `pcl::PointCloud<PointXYZI>` that **grows to cover the explored world and is never pruned**.

- **Incremental voxel fusion.** Each incoming point folds into its voxel's running centroid
  and mean intensity through a `voxel → index` hash. Revisiting a place *refines* points
  instead of duplicating them. Cost is O(incoming) per frame, independent of world size — a
  batch `VoxelGrid` re-run every frame would be quadratic.
- **Local crops.** `local()` / `projectLocal()` return the subset within a radius of a centre
  **plus the matching global indices**, so anything computed locally lifts straight back onto
  the persistent map.
- **Views.** Each image registers a `View` (camera pose + intrinsics + image handle + robot
  world position).
- **Per-point semantics.** A `SemanticVocabulary` interns class **names** to stable ids, so a
  runtime vocabulary change never shifts an id already assigned and earlier votes stay valid.
  Optional persistent per-point vote histograms (`setVoting`) with a stuff bias and a label
  gate.
- **Tombstoning** (`killPoint` / `alive_`) as described in §1.

#### `Superpoints` (`superpoints.{h,cpp}`) — tier 1, ephemeral

PCL `SupervoxelClustering` (VCCS) over the current view, run on the **full** working set
(including stuff and unlabelled points) so superpoints follow real surfaces. The list is
**replaced** each refresh — nothing is merged across frames. Each superpoint carries a class
histogram (the semantics that growing matches on) and a dominant thing class by majority.

#### `ObjectSeeds` (`object_seeds.{h,cpp}`, `object_grow.cpp`, `hdbscan_extract.cpp`) — tier 2

- **Proposals.** Per-class HDBSCAN\* in 3-D over the frame's visible **thing** points, using
  the vendored parallel `hdbscan` library. Optional statistical outlier removal beforehand and
  a Euclidean connected-component split afterwards so no proposal spans a physical gap.
- **Growing** (`object_grow.cpp`). SAI3D region affinity —
  `cosine(class histograms) × containment(shared voxels / superpoint voxels)` — merges each
  superpoint into its matching proposal(s), optionally pooling each superpoint's histogram
  over its KNN neighbours and running several assignment sweeps.
- `object_seeds.cpp` is the **only** translation unit that pulls in parlay, which is why
  `pargeo_hdbscan` is linked PRIVATE and its heavy includes never leak to the drivers.
  `object_grow.cpp` and `object_consolidate.cpp` are parlay-free and read only the
  `Universe`/`Superpoints` headers (shared voxel math in `voxel_util.h`).

#### `Objects` (`objects.h`, `object_consolidate.cpp`) — tier 3, the primary output

A single-tier **Union-Find over the seed layer**. Every per-frame proposal is retained as a
persistent seed; a new seed is unioned into every same-class component it is sufficiently
*contained* in, with the bar depending on the component's level. A component's **support** is
its member-seed count, and it becomes a *reported object* once support reaches `min_merges` —
progressive merging (SGS-3D / SAI3D), not a one-shot promotion.

Optional **global-context merging**: instead of deciding inter-object merges from one frame's
voxel overlap, each frame deposits confidence-weighted affinity onto a persistent edge graph,
and a merge fires only once the accumulated affinity clears the bar with enough evidence *and*
a neighbourhood-aggregated score agrees. This kills single-frame spurious bridges.

Optional **overlapping point-set model**: higher layers stop *owning* points — a superpoint is
grown into every matching proposal, proposals become overlapping bags of points, and the final
objects are made disjoint only at the end by assigning each contested voxel to its
highest-support component.

#### `OnlineSemantic` (`online_semantic.h`) — the conductor

Owns every stage and drives one `SyncedFrame` at a time through `onSynced`:

1. **Infer 2D labels** first — labels gate which geometry is admitted.
2. **Reject dynamic classes at ingest** — scan points projecting onto a person mask are
   dropped *before* fusion, so a moving object never smears into the map.
3. **Fuse** the surviving cloud.
4. **Project + vote** — stamp or vote the visible points with the class at their pixel; fuse
   per-point RGB from the image.
5. **Discovery** — superpoints → HDBSCAN\* proposals → growing → object consolidation, each
   gated by its config flag. Growing requires superpoints **and** HDBSCAN; the objects tier
   requires growing. Enabling `objects` therefore implicitly requires the whole chain.
6. **Fire hooks** — frame, objects, sign, and segmentation-overlay consumers.

`stepFrameSynced` returns **two** index sets, because the stages need different things:
`out_gidx` (the observed *shell*: nearest visible point per pixel plus a depth band) always
seeds proposals — a proposal is a hypothesis about geometry the camera *resolved*. `out_seg_gidx`
(the *segmentation* set that superpoints and growing consume) is either the same shell or,
in frustum mode, the shell **plus** occluded live map points inside the camera cone. See
[configuration.md](configuration.md#working-set) for the trade-off.

#### `SemanticRunner` (`semantic_runner.{h,cpp}`) — the deployment entry point

The whole pipeline behind **one blocking call**, for the robot's own code:

```cpp
semantic::SemanticRunner runner("run.yaml");
for (;;) {
    std::optional<OpenVocabSegInput> in = robotGetSynchronizedInputs();
    std::vector<Common::Entity::EAIRoomObject> objects = runner.runOnInput(in);
    publish(objects);
}
```

It owns `OnlineSemantic` + the viz publisher + the object egress behind a PImpl, so the caller
sees **only GMD entity types** — no PCL, no CUDA, no pipeline internals. Inputs are assumed
already time-synced and pose-interpolated (that is the robot system's job), so no `DogStream`
is involved on this path. Returns the full current object list every call.

#### `VizPublisher` (`viz_publisher.{h,cpp}`) — viz egress

Serializes per-frame and final pipeline state onto a ZeroMQ **PUSH** socket for `viz_server`.
Per-frame messages are sent `DONTWAIT` (dropped on backpressure — a live viewer only needs the
latest). An empty endpoint makes every call a no-op. The wire schema mirrors `inf_server`'s
ndarray envelope so Python rebuilds numpy arrays directly.

**This is why the C++ build carries no rerun/Arrow dependency** — it was removed entirely when
visualization moved to Python, so the project configures on locked-down systems that cannot
build Arrow.

#### `run_config.{h,cpp}` — configuration

Loads `run.yaml` (via the vendored yaml-cpp) into a `RunConfig` containing `semantic::Params`,
the viz settings and the sign settings. **Every key is optional**; anything omitted falls back
to the C++ default. Shared by all drivers.

### 2.5 `gmd_adaptor/` + `gmdDataTypesForInterns2/` — the robot interface

`gmdDataTypesForInterns2/` is the vendored robot type library. Only two of its TUs compile into
`gmd_types` (`TimeStamp.cpp`, `Transformation.cpp`) — `ImageUtils.cpp` and `PoseUtils.cpp` need
OpenCV and are deliberately skipped.

- **Ingress** — `GmdFrameSource` is a concrete `FrameSource` that converts the robot's native
  `PrimaryPose` / `PointCloud` / `Image` messages and forwards them to the shared `DogStream`
  sync engine. Nothing on either side is modified; this only translates.
- **Egress** — `ObjectPublisher` converts a pipeline `Object` into a
  `Common::Entity::EAIRoomObject`. The overhead polygon is a 2D bird's-eye convex hull: project
  members onto X-Y (the LiDAR world is Z-up right-handed), reject strays with a kNN statistical
  outlier pass so grown boundary points don't balloon the hull, then take the convex hull
  (monotone chain). Each object carries a **frame-stable `objectId`** derived from the pipeline
  object id, so re-publishing an existing id updates rather than duplicates.

Both are PCL/CUDA/server-free, which is why `tests/` can exercise them with no GPU.

### 2.6 `sign_adaptor/` + `sign-unds/` — async sign-text egress

Optional side channel that reads text off signs and attaches it to the object containing them.

`SignBridge` hangs off `OnlineSemantic::setSignHook`. Per frame it scans the 2D masks for a
sign-class instance and, **if nothing is already in flight** (reading takes ~1–2 s, so frames
arriving meanwhile are dropped on purpose), computes the sign's pixel bbox and 3D voxel
footprint and dispatches to a worker thread. The detector answers asynchronously keyed by a
match id; `annotate()` then attaches the text to the object whose footprint best encompasses
the sign (biggest proposal wins on conflict), filling `EAIRoomObject::directionContent` and the
rerun label.

The detector sits behind an `IDetector` seam:

| Build | Backend |
|-------|---------|
| default | `sign_detector_mock.cpp` — sleeps `mock_delay_ms`, answers `mock_text`. No OpenCV/CURL/VLM. Exercises the entire path anywhere. |
| `-DLIFT3D_USE_SIGN=ON` | `sign_detector_real.cpp` + the vendored `sign-unds/` detector chain talking to a real VLM. |

**The bridge itself is always built** (it needs nothing beyond `universe`); only the backend is
conditional. Either way nothing runs unless `run.yaml` sets `sign.enabled: true`.

### 2.7 Vendored submodules

| Submodule | Why it's vendored |
|-----------|-------------------|
| `hdbscan/` (wangyiqiu) | Parallel HDBSCAN\*; header-only parlaylib bundled inside it. Built in-tree as `pargeo_hdbscan`. |
| `yaml-cpp/` (pinned 0.8.0) | Config parsing, built via `add_subdirectory` — no system `-dev` package, no network fetch. |
| `cppzmq/` + `libzmq/` | Supply the **headers only** (`zmq.hpp`, `zmq.h`). The system libzmq runtime is linked, so libzmq is never built and no `-dev` package is needed. |
| `sign-unds/` | Teammate's sign-understanding module (only compiled with `LIFT3D_USE_SIGN=ON`). |
| `inf_server/DVIS_Plus`, `mask3d_feat/Mask3D`, `mask3d_feat/MinkowskiEngine`, `inf_server/export/detectron2` | Python-side model code. |

---

## 3. The Python services

Full details in [python-services.md](python-services.md).

| Service | Transport | Role |
|---------|-----------|------|
| **`inf_server/`** | `zmq.REP`, msgpack, `ipc:///tmp/inf_server.ipc` | OV-DVIS++ open-vocabulary **video** instance segmentation. A stateful tracker: frames must be fed in order, `reset` begins a new video. Owns preprocessing and returns maps at the original `(H,W)`. **Required for a real run.** |
| **`viz_server/`** | `zmq.PULL`, msgpack, `ipc:///tmp/lift3d_viz.ipc` | rerun visualizer. Renders per-frame entities on the timeline and the full world map as a static snapshot on `finish`. All rerun/Arrow dependencies live here. Optional. |
| **`mask3d_feat/`** | `zmq`, msgpack, `ipc:///tmp/mask3d_feat.ipc` | Mask3D backbone → 96-d per-voxel deep features. **Currently dormant** (see below). |

### On `mask3d_feat` being dormant

The pipeline originally used Mask3D deep features as the growing/consolidation affinity
signal. That was **replaced by SAI3D class-histogram affinity**, which is derived from the
accumulated per-point votes and costs nothing extra. Measurements showed the Mask3D backbone
dominated per-view cost (0.7–1.9 s versus ~200–320 ms for VCCS + HDBSCAN).

There is consequently **no `FeatClient` or `PointFeatures` in the current C++ source**, and no
`features:` block in `run.yaml`. The directory, its server and the description in `doc.md` §7.2
are retained for reference and possible revival; **you do not need to run it.**

---

## 4. Executables

| Target | Needs | Purpose |
|--------|-------|---------|
| `run_semantic_universe` | inf_server (+ log) | The main headless driver. Optional labelled-PCD output, optional viz stream. |
| `run_on_input_demo` | inf_server (+ log) | Replays a recording through `SemanticRunner::runOnInput`, standing in for the robot's `getSynchronizedInputs`. The harness for the deployment entry point. |
| `exp_per_view` | inf_server (+ log) | Throwaway per-view cost + DVIS id-overlap measurement experiment. |
| `gmd_stream_check` | **nothing** | GMD → pipeline sync/lift/pixel translation check. No PCL, no GPU, no servers. |
| `object_publisher_check` | **nothing** | Object egress check: 2D hull + keyed topic. No PCL, no GPU, no servers. |

The last two are the quick smoke tests — they compile and run anywhere, which makes them the
right first thing to try after a fresh clone.

---

## 5. Dependency graph of the build targets

```
point_pixel_mapping ──┐
                      ├──► universe ──┬──► object_seeds ──► (pargeo_hdbscan, PRIVATE)
inf_client ───────────┘               │
   (cppzmq/libzmq hdrs, system libzmq)├──► viz_publisher
                                      ├──► run_config ──► (yaml-cpp, PRIVATE)
dog_log_adaptor ──► (Eigen)           └──► sign_adaptor ──► (sign-unds, only if USE_SIGN)

gmd_types ──► gmd_adaptor

run_semantic_universe = universe + dog_log_adaptor + inf_client + object_seeds
                      + run_config + viz_publisher + sign_adaptor
semantic_runner       = the above + gmd_adaptor + gmd_types   ──► run_on_input_demo
gmd_stream_check      = gmd_adaptor + gmd_types + dog_log_adaptor
object_publisher_check= gmd_adaptor + gmd_types
```

Note the two deliberate PRIVATE links: parlay (heavy templates) stays inside `object_seeds`,
and yaml-cpp stays inside `run_config`. Neither leaks to the drivers.
