# lift3d — Semantic Universe Pipeline

A design + operations reference for the online 2D→3D semantic mapping pipeline built on
top of `lift3d`. This document is self-contained: it describes the data flow, the math at
each stage, the configuration surface, the gotchas that are baked into the code, and the
outputs — enough to reason about and plan extensions without reading the source.

---

## 1. What this system does (one paragraph)

A quadruped robot ("dog") drives through an indoor space streaming **LiDAR scans + poses**
(~10 Hz) and **camera images** (~5 Hz). The pipeline fuses the LiDAR into a persistent,
voxel-deduplicated **world point map** ("the Universe"), runs an **open-vocabulary 2D
segmentation model** on each image, and **lifts** the 2D labels onto the 3D points via a
z-buffered point-to-pixel projection. It then discovers **object instances** through a
3-tier hierarchy: VCCS superpoints (oversegmentation) → HDBSCAN* seeds (per-class dense
clusters) → consolidated objects (feature-similar, spatially-adjacent seeds merged). Every
computation that isn't the raw fusion runs on a **local crop around the robot** and lifts
results back onto the global map by global point index. The primary output is a list of
persistent object instances, each a subset of the world map plus a centroid.

The overall structure follows **SGS-3D (arXiv:2509.05144v2)**: per-point semantics via
point-to-pixel lifting → oversegmentation → object discovery. The key adaptation is that
SGS-3D assumes a *static, complete* scene (ScanNet mesh); here everything is **online and
grow-only** as the robot explores.

---

## 2. External runtime dependencies (must be running before the C++ pipeline)

Two **separate Python GPU inference servers**, each in its own isolated `uv` venv, talking
over ZeroMQ + msgpack IPC. They are NOT built by CMake. Both need a working GPU (see
gotchas — WSL passthrough is intermittent here).

| Server | Dir | Endpoint (default) | Role |
|--------|-----|--------------------|------|
| **inf_server** | `inf_server/` | `ipc:///tmp/inf_server.ipc` | OV-DVIS++ open-vocab video instance segmentation. Produces per-pixel class + instance labels. The 2D "producer". |
| **mask3d_feat** | `mask3d_feat/` | `ipc:///tmp/mask3d_feat.ipc` | Mask3D backbone. Produces 96-d per-point deep features used by the growing/objects tiers. |

**inf_server protocol** (msgpack over zmq REP): `ping` / `set_vocab{thing,stuff}` / `reset`
(start a new tracking sequence) / `frame{image uint8 HxWx3 RGB}` → `{label_map(int16 HxW,
-1=bg), id_map(int16 HxW, -1=bg), instances[{id,label,score,embedding}]}`. The server owns
preprocessing (resize short edge to 480, normalize) and returns maps at the original (H,W).
It is a **stateful tracker**: frames must be fed in order; `reset` begins a new video.

**mask3d_feat protocol** (`cmd: features`): request is `coords` (N×3 int32 **voxel** coords =
round(xyz / voxel)) + `feats` (N×3 float32 **normalized RGB**); response is `coords` + `feats`
(N×96 per-voxel deep features). Colour is a required input (ScanNet normalization; see §7.2).

Both are launched roughly as:
```bash
cd inf_server   && uv run python main.py --endpoint ipc:///tmp/inf_server.ipc --device cuda
cd mask3d_feat  && uv run python main.py   # endpoint ipc:///tmp/mask3d_feat.ipc
```

---

## 3. Data inputs

Configured in `run.yaml`. A **dog log folder** (`log:`) plus a **calibration XML**.

**Per-log folder** (e.g. `~/dog_logs/20260608_test_log_officePantryNarrowCorridor/`):
- `PoseAndPointClouds.bin` — MLOAMC LiDAR scans. Per scan: 60-byte header (int64 timestamp
  in 100 ns units, 6× float64 pose `x,y,z,roll,pitch,yaw`, int32 numPoints), then
  numPoints × 13 B points (3× float32 xyz + uint8 intensity). Little-endian, no padding.
  **Cloud is in the robot BODY frame; pose is body→world (extrinsic-XYZ euler).**
  Zero-return beams (x=y=z=0) must be dropped.
- `camera<id>.gclf` — the image stream (e.g. `camera211.gclf` for `cam: 211`).

**Calibration XML** (`CameraCalRaiboAsQUGV113.xml`, in the log's parent dir): one
`<cameraCalibration>` per sensorId with projection `fu,fv,cu,cv`, imageWidth/Height, and
an `imuToCamera` offset + quaternion = the camera-optical→body mount transform.

**Camera choice matters:** cams **111 / 211 are pinhole** (xi=0, no distortion) — correct
for this pipeline. Cams **11 / 21 are fisheye** (xi≠0) and need an omnidirectional model
the pipeline does NOT implement. Use 111/211.

The offline path (`DogLogIngestor`) replays the bin/gclf files through the *same*
`pushScan`/`pushImage` interface a live sensor driver would use, so the pipeline is
transport-agnostic.

---

## 4. Coordinate conventions & the projection math

**Frames.** Camera optical frame is OpenCV convention (X right, Y down, Z forward). The
LiDAR cloud is in the body frame; the pose is body→world. The camera-to-world pose used for
projection is composed as:

```
T_world_cam = T_world_body · T_body_cam
```

where `T_body_cam` comes from the calibration mount (offset + quaternion) and `T_world_body`
is the **interpolated** body pose at the image timestamp (see §5).

**Point-to-pixel projection** (`point_pixel_mapping.cuh/.cu`, PCL-free, CPU + CUDA backends).
Given camera intrinsics `K` (row-major 3×3) and camera-to-world pose `T` (row-major 4×4,
**rigid**), the projection matrix is precomputed host-side:

```
M = K · (T⁻¹)[0:3, :]         (3×4)
```

Because `T` is a rigid camera-to-world pose, its inverse is closed-form:
`T⁻¹ = [Rᵀ | −Rᵀt]` (built in `buildProjection`). For a world point `p`:

```
[a, b, c]ᵀ = M · [x, y, z, 1]ᵀ
u = a/c,   v = b/c,   depth z = c
```

**Visibility via a self-reconstructed z-buffer** (no depth sensor). Two kernels:

1. `projectKernel` — computes `(u,v)` and depth per point, and atomically z-buffers the
   nearest depth per pixel. The **depth splat** (`zbuf_dilate`, config `splat`) rasterizes
   each point's depth into a `(2r+1)×(2r+1)` window around its pixel (nearest wins). This is
   the SGS-3D fix for the **resolution mismatch** between a sparse voxel map and a dense
   image: with r=0 the depth map is full of holes, so a point behind a wall that lands in a
   hole pixel is nobody's occludee and leaks through as "visible". r≥1 lets a nearby surface
   voxel win the hole and occlude it. The per-point `(u,v)` is always the exact centre pixel;
   only the *depth test* dilates.

2. `visibilityKernel` — a point is visible iff in-frame **and** `|zᵢ − D(u,v)| ≤ τ_vis`
   (config `tau`), where `D(u,v)` is the z-buffered nearest depth. Fills the `[u,v,1]`
   correspondence map.

**Z-buffer atomics gotcha (deliberate):** the z-buffer stores depths as **ints** via
`cuda::std::bit_cast<int>(float)` and uses `cuda::atomic_ref<int>::fetch_min`, NOT the legacy
`atomicMin`/`atomicCAS` float intrinsics. IEEE-754 bit patterns are monotonic for positive
depths, so an integral min over the reinterpreted bits gives the true nearest depth — and
integral `fetch_min` is portable across all archs (float atomic-min is not). Requires C++20.

**Pose convention gotcha:** `Camera.T` is treated as camera-to-world and inverted internally.
If a caller ever supplies world-to-camera poses, the inverse must be skipped.

---

## 5. The image/pose timeline sync (DogStream)

Poses+LiDAR arrive at ~10 Hz but images at ~5 Hz, so an image's true capture time almost
never lands on a pose sample. Projecting an image's labels through the *nearest scan's* pose
introduced up to ~100 ms of lag while the robot moved/turned, smearing labels.

**Fix:** `PoseInterpolator` buffers body→world poses and, for each image, interpolates the
pose to the image's exact timestamp — **SLERP on rotation, LERP on translation**. `DogStream`
then composes `T_world_cam = T_world_body(image_t) · T_body_cam`, attaches the nearest LiDAR
scan (already world-lifted with its own exact scan pose), and emits a **`SyncedFrame`**
(image + camera pose @ image_t + cloud). Every downstream stage lifts labels through the
camera pose **at the image instant**.

**Policy:** WAIT for the bracketing pose — an image is held until a pose with `t ≥ image_t`
arrives, so interpolation is always two-sided. Images older than the buffered pose history
are dropped (unrecoverable). The residual scan/image offset the sync absorbs is logged per
frame as `scan_dt`.

---

## 6. The persistent world map (Universe)

`Universe` is a `pcl::PointCloud<PointXYZI>` (geometry + LiDAR intensity) that **grows to
cover the whole explored world and is never pruned**. "Operate locally" is a *read-time*
crop, never a mutation.

- **Incremental voxel fusion:** each incoming point folds into its voxel's running centroid
  + mean intensity via a `voxel → index` hash (`vox_`). Revisiting a place refines its points
  instead of duplicating them. Cost is **O(incoming) per frame**, independent of world size
  (PCL's batch `VoxelGrid` re-run every frame would be quadratic). The data stays a PCL cloud;
  the hash is just an index. Voxel edge = config `voxel` (default 0.05 m).
- **Local crops:** `local()` / `projectLocal()` return the subset within a radius of a centre
  (the robot) **plus the matching GLOBAL indices**, so anything computed locally lifts straight
  back onto the persistent map. This is the load-bearing invariant of the whole pipeline —
  every tier stores global indices, never local ones.
- **Views:** each image registers as a `View` (Camera pose + intrinsics + image handle +
  robot world position, for re-centering local ops).

**Grow-only / index-stable invariant (critical):** the map is index-keyed and append-only.
Points are **never erased or reindexed**, because that would shift every downstream global
index (votes, colours, features, superpoint/seed/object membership). "Deletion" is
**tombstoning** (`killPoint` / `alive_`): a dead point keeps its index but is skipped by
every read-time enumeration (crops, projection candidates, features, clustering, export,
viz), and its vote histogram is cleared. A later fuse into the same voxel **revives** the
slot (real geometry reclaimed a spot a person vacated).

---

## 7. Per-stage detail (the online loop)

Driven one `SyncedFrame` at a time by `OnlineSemantic::onSynced`. Per frame: integrate +
vote. On a cadence (`refine_every` frames): seed → features → superpoints → grow →
consolidate. Then fire consumer hooks.

### 7.1 Integrate + semantic voting (every frame)

`stepFrameSynced`:
1. **Infer 2D labels first** (`inf.frame`), because labels gate which geometry is admitted.
2. **Dynamic rejection at ingest** (`rejectDynamic`): incoming scan points that project onto
   a "dynamic" 2D segment (people, …) are dropped *before* fusion, so a moving object never
   accumulates a smear. A single LiDAR sweep has one return per direction, so a plain pinhole
   projection + mask lookup is unambiguous — no z-buffer needed (if a point lands on a person
   pixel it *is* the person). Zero overhead when `dynamic:` is empty.
3. **Fuse** the surviving cloud into the map (`uni.integrate`).
4. **Project** the world (or the `proj_radius` crop) into this view and **vote** every visible
   point with the class at its pixel. Global indices lift local results onto the map.
   - Per-point **RGB is fused** from the camera image on every view that sees the point
     (running mean; feeds the Mask3D backbone, which needs colour — the map itself is only
     `PointXYZI`).
   - **Dynamic cleanup path:** if a *visible* point lands on a person mask, it is tombstoned
     (`killPoint`). This catches people the wide-FOV LiDAR fused off to the camera's side
     (no 2D mask there to reject them at ingest) that later surface as ghosts — a view that
     sees them on a person segment kills them. The z-buffer already gated visibility, so only
     points actually ON the person die; occluded geometry behind stays.

**Backprojected class — two modes (`voting.enabled`):**

- *Naive (default):* a point's class is whatever the **current** frame's 2D segmentation
  projects onto it — `setFrameLabel` overwrites a transient per-point label each frame, no
  accumulation. Downstream stages read only the current view's points, so they see a fresh
  current-frame label.
- *Voting (`voting.enabled: true`):* every observation instead **votes** into a persistent
  per-point sparse histogram of `(class_id, count)`, and the point's class is the argmax —
  a cross-frame consensus rather than the last frame's guess. The histogram survives across
  frames (reset only when its voxel is tombstoned/revived).

Class ids come from a stable `SemanticVocabulary` keyed on class **name**, so a runtime
`set_vocab` change merely interns new names — ids already assigned never shift, earlier votes
stay valid. (inf_server's per-call integer labels are only meaningful relative to that call's
sorted vocab; keying on name decouples the stored semantics from any single call.)

**Stuff bias (voting mode):** the argmax weights each class's votes by `stuff_bias` (≥ 1) when
it is a *stuff* class, so a thing overtakes the leading stuff class only once its votes exceed
stuff's by that factor — "a point seen as stuff is harder to turn into a thing." This is the
principled curb for the occlusion/label-leak that paints objects onto walls/floor (sparse
z-buffer holes let a wall point grab a stray `chair` pixel; one biased vote can't flip it).
Because the `active` working set that feeds superpoints/HDBSCAN is exactly `pointIsThing`, the
bias directly suppresses spurious thing seeds on stuff surfaces.

**Label gate (read-time, voting mode):** a point resolves to a class only if it has ≥
`min_votes` total votes AND the winning class holds ≥ `min_conf` of the raw votes. Below the
gate → unlabeled. Defaults `(1, 0)` = plain (stuff-biased) argmax.

### 7.2 Mask3D per-point features (cadence)

`PointFeatures::refreshLocal` runs the backbone on the robot-window crop and **scatters** the
96-d per-voxel features into a persistent, whole-cloud, grow-only store keyed on global index.
Overlapping windows overwrite (refresh) boundary points, so every point the robot passes ends
up with a feature without ever feeding the unbounded whole cloud to the GPU. Features use
local sparse-conv context — fine for local object discovery.
- Backbone voxel = `features.voxel` (0.02 m); coords = xyz / voxel.
- Colour input normalized with ScanNet stats: `feat = (rgb/255 − mean)/std` (from Mask3D
  `datasets/semseg.py`). Points no camera has seen get a fallback gray (128).
- `extra_gidx`: when growing is on, in-radius objects' member points that spill *past* the
  crop are unioned into the backbone request, so an object gets consistent features over its
  whole extent.

### 7.3 Tier 1 — VCCS superpoints (oversegmentation, EPHEMERAL)

`Superpoints::refreshLocal` runs PCL `SupervoxelClustering` (VCCS) over the robot crop and
**REPLACES** the list with just that window's supervoxels — nothing is merged or carried
across refreshes. The window slides with the robot; `list()` always reflects only the most
recent window (the working set the seed/grow stages consume). SGS-3D uses ScanNet mesh
oversegmentation (static scene); VCCS re-run periodically is the online substitute.

VCCS params (`superpoints:` block): `voxel_res` (= map voxel), `seed_res` (**the dominant
knob for superpoint size**), `spatial_w`, `normal_w` (geometry-driven), `color_w` (0 — run on
geometry-only `PointXYZ`), `refine` iterations.

Each superpoint carries a **thing class** by majority: among members whose resolved class is
a Thing, if one class holds ≥ `thing_frac`, the superpoint takes it; else Unknown. Stuff and
unlabeled members don't count toward the denominator; stuff is never assigned. Each superpoint
also gets a **feature dispersion** = trace(feature covariance)/dim (mean per-dim variance)
over members that have a feature — a homogeneity score (low = feature-consistent, likely one
object part); 0 when < 2 members have features.

**To oversegment more aggressively (more, smaller superpoints):** lower `seed_res` (e.g.
0.65 → 0.3–0.4); optionally raise `normal_w` to split flat-but-differently-oriented surfaces;
keep `refine` at 0 (refinement merges/smooths, i.e. the opposite direction).

### 7.4 Tier 2 — HDBSCAN* object seeds (per-class, LOCAL, PERSISTENT)

`ObjectSeeds::seedLocal` clusters the still-**unclaimed** THING points in the robot window,
**per class**, in 3-D xyz with the parallel HDBSCAN* library (wangyiqiu/hdbscan). Each dense
cluster becomes a new persistent `ObjectSeed`. Strict on purpose (large `min_pts` /
`min_cluster_size`) so it **under**- rather than over-detects; growing/consolidation expand.

- **Maturity gate:** histogram the window into coarse (~2 m³, edge `maturity_res`) cells; a
  cell is clusterable only once it holds ≥ `maturity_min` occupied fine voxels. Keeps HDBSCAN
  off sparse, half-observed geometry.
- **Claimed-list (`owner_`):** per-map-point owning-object id (−1 = unclaimed). Seeding only
  clusters unclaimed points, so grown objects are never re-clustered and the input shrinks as
  the map fills in. Objects are append-only — existing ones are never touched or wiped.
- Pre-gates per class: ≥ `min_class_points` unclaimed points, and n > `min_pts`.
- Pipeline: `hdbscan<3>` (min_pts) → `dendrogram` → `extractFlatClusters(min_cluster_size,
  allow_single_cluster)`; label −1 = noise, dropped. `allow_single_cluster` lets a
  single-object class resolve to one seed.
- Seeding cadence is **coupled to `refine_every`** (it births seeds that grow/consolidate
  consume the same frame); the *maturity gate*, not a frame counter, decides which cells seed.
- (The vendored lib spams stdout per call; it's silenced by redirecting `std::cout` for the
  clustering scope.)

### 7.5 Feature-guided growing (superpoints → seeds)

`ObjectSeeds::growLocal` absorbs feature-consistent, spatially-overlapping superpoints into
nearby seeds. For each in-radius object it caches a mean feature + a member-voxel set + AABB.
For each superpoint (pruned if `feat_count < 2` or `feat_dispersion > max_dispersion`):

```
containment = |superpoint_voxels ∩ object_voxels| / |superpoint_voxels|
affinity    = cosine(mean_feat_superpoint, mean_feat_object) · containment
```

The superpoint merges into the **best-affinity** object with `affinity ≥ affinity_thresh`
(and matching class if `require_class`). Only the superpoint's **unclaimed** points are
annexed (first-claim wins — points owned by another object are never stolen). Grown objects
re-derive centroid + mean feature. A cheap gate (centroid distance ≤ `cand_radius` OR inside
the AABB+1 voxel) skips non-overlapping candidates before the voxel-intersection test.

### 7.6 Tier 3 — Objects consolidation (seeds → objects, PRIMARY OUTPUT)

`Objects::consolidate` merges adjacent, feature-similar **seeds** into fewer, cleaner
persistent objects — fixing HDBSCAN oversegmentation where one physical object fragmented
into several seeds. Each `Object` owns a set of member **seed ids** and DERIVES its points /
mean feature / centroid from them, so as a member seed keeps accumulating superpoints, its
object re-derives and grows with it (a merged seed is never frozen). Monotonic: a seed joins
at most one object, and there is no un-merge.

The scoring rule is the subtle part (there's a documented bug this design avoids):

```
adjacency = fraction of the seed's voxels within Chebyshev distance K of any object voxel
            (K = round(adj_dilate / voxel))
merge  ⟺  adjacency ≥ adj_min   (GATE)   AND   cosine(mean_feats) ≥ affinity_thresh (SCORE)
```

- **Adjacency is a GATE, not a score.** It's a *boundary fraction* — for two comparably-sized
  fragments of one object it stays small even when they abut. Folding it into the score
  (cosine × adjacency) made the threshold mathematically unreachable → one object per seed
  (the original bug). So adjacency only *admits* a candidate; the score is pure cosine.
- **Plain containment can't be used** either: seeds are disjoint point sets and an object is
  built from *other* seeds, so a fresh seed shares zero voxels with it — hence the *dilated*
  adjacency (touch within K voxels) instead of exact overlap.

A seed with no feature yet is **deferred** (left unassigned this pass). A seed that matches no
object spawns a new one. Every in-radius object is re-derived each pass (its member seeds may
have grown via superpoints).

### 7.7 Cadence coupling (why one interval)

Seed → features → superpoints → grow → consolidate all share `refine_every` because they're a
dependency chain: growing consumes both features and superpoints; consolidation consumes the
grown seeds. Running them on separate intervals would let one lag the others and operate on
stale inputs. These window-ops are expensive, so the cadence is kept coarse. `finish()` does
one final whole-window pass at the last robot pose so the store is complete.

---

## 8. Configuration surface (`run.yaml`)

Loaded by `run_config.cpp` into `semantic::Params`; any omitted key falls back to a built-in
default. CLI flags `--log`, `--out`, `--spawn` override the file.

| Group | Keys | Meaning |
|-------|------|---------|
| recording | `log`, `calib` (empty ⇒ `<log>/../CameraCalRaiboAsQUGV113.xml`), `cam`, `out` | inputs + output path |
| window | `start`, `count`, `stride` | frame range; `stride` sub-samples the image window |
| map/projection | `voxel` (fusion size), `radius` (label crop; 0 = whole map), `proj_radius` (tighter z-buffer candidate crop), `tau` (z-buffer tolerance m), `splat` (depth dilation px) | §4, §6 |
| voting | `enabled`, `stuff_bias`, `min_votes`, `min_conf` | §7.1 (cross-frame class; off => naive per-frame) |
| label filter | `things_only` | §7.1 |
| inference | `endpoint`, `thing[]`, `stuff[]`, `dynamic[]` | vocab + dynamic-reject classes |
| cadence | `refine_every` | §7.7 |
| superpoints | `enabled`, `seed`, `radius`, `thing_frac`, `spatial`, `normal`, `refine` | §7.3 |
| features | `enabled`, `endpoint`, `radius`, `show` | §7.2 |
| hdbscan | `enabled`, `min_pts`, `min_cluster_size`, `min_class_points`, `allow_single_cluster`, `maturity_res`, `maturity_min` | §7.4 |
| grow | `enabled`, `affinity`, `max_dispersion`, `cand_radius`, `require_class` | §7.5 |
| objects | `enabled`, `affinity`, `adj_min`, `adj_dilate`, `cand_radius`, `require_class` | §7.6 |

**Stage-enable gating (from `OnlineSemantic`):** features need a `PointFeatures*` + a
`FeatClient*` + `features: enabled`. Grow needs features **and** superpoints **and** hdbscan
all on, plus `grow: enabled`. Objects need grow on plus `objects: enabled`. So enabling
`objects` implicitly requires the whole chain below it.

`dynamic:` classes must ALSO appear in `thing`/`stuff` so the 2D model still segments them —
that mask is exactly what the reject/cleanup filter keys on.

---

## 9. Outputs

Two executables, same config/pipeline:
- `run_semantic_universe` — headless; can write a labeled PCD (`PointXYZL`, label =
  classId+1, 0 = unlabeled) via `--out`.
- `viz_semantic_universe` — same plus a **rerun** recording (`.rrd`), `--spawn` for a live
  viewer. Objects reported via the objects hook.

**rerun entity tree** (colours/labels resolve via a static `AnnotationContext` on `world`):
- `world/map/{things,stuff,unlabeled}` — the full segmented map, class-coloured, split by
  kind so "stuff" can be toggled off. Logged **once as static** (not per-frame) — per-frame
  re-logging the whole map made the recording O(frames × map_size), many GB, which the
  viewer's memory-limit GC then purged mid-load (points flickered in then vanished).
- `world/superpoints/{all,things,centroids,window}` — the current ephemeral VCCS window,
  logged per-frame on the timeline only at strided recompute frames. Points/centroids coloured
  by a per-superpoint hash colour (`superpointColor(id)`); `window` is the crop sphere.
- `world/seeds/points` — tier-2 seed members (debug layer), per-seed hash colour.
- `world/objects/{points,centroids}` — **the headline output.** Member points and the centroid
  marker are both coloured by the object's stable id hash (`superpointColor(o.id)`), so a
  growing object keeps its colour and centroid/points visibly match. (This colour-matching was
  a recent fix; the centroid previously used the class colour.)
- `world/features` — optional; per-point colour = a 96→3 random projection of the feature
  vector, min-max normalized to RGB (a similar-colours ≈ similar-features sanity check).

---

## 10. Build & run

```bash
export PATH=/usr/local/cuda/bin:$PATH        # nvcc is not on PATH by default
cmake -S . -B build-cuda                     # ~6 min: PCL/VTK detection dominates configure
cmake --build build-cuda -j
# 1) start both GPU servers (see §2), then:
./build-cuda/run_semantic_universe --config run.yaml
./build-cuda/viz_semantic_universe --config run.yaml --spawn
```

**Two build dirs exist** (`build/` and `build-cuda/`). `run.yaml`'s header documents
`build-cuda/`. Rebuild the one you actually run — editing source and rebuilding `build/`
leaves a stale `build-cuda/` binary (this has bitten before).

---

## 11. Gotchas (read before debugging)

- **GPU passthrough is intermittent (WSL2).** The GPU (RTX 2080 Ti) *does* run, but
  passthrough is host-side and flaky; `nvidia-smi` failing / CUDA "driver version
  insufficient" means restart the Windows-host driver + the Python servers after recovery.
  `nvcc` always *compiles* fine regardless. Never install a Linux NVIDIA driver inside WSL.
  (The CLAUDE.md "GPU does not run here" line is stale — see the WSL-GPU memory.)
- **`nvcc` not on PATH by default.** Toolkit at `/usr/local/cuda`; `~/.bashrc` exports it for
  new shells, otherwise prepend `/usr/local/cuda/bin`.
- **PCL pulls in VTK, which needs `C` enabled.** The CMake `project()` must declare
  `LANGUAGES C CXX CUDA` or VTK 9.5 fails on a missing `MPI::MPI_C` target.
- **C++20 required** for the z-buffer's `cuda::atomic_ref` / `bit_cast`.
- **Configure is slow (~6 min)** due to PCL dependency probing — reuse the build dir; only
  `rm -rf` when changing project-level settings.
- **The two inference servers each have their own venv + build landmines** (detectron2 /
  MSDeformAttn / MinkowskiEngine built from source against cu118, gcc shims, open_clip patches,
  panopticapi shim). See the inf-server and minkowski-engine memories. If a run prints
  "inf_server not responding" / "mask3d_feat not responding", the server isn't up or the GPU
  is down.
- **Index stability is load-bearing.** Never erase/reindex map points — tombstone instead
  (§6). Every tier keys on global index.

---

## 12. Pointers for planning next steps

- **The 3-tier discovery is the active research surface.** Superpoints (tier 1) are ephemeral
  and geometry-only; seeds (tier 2) are per-class HDBSCAN; objects (tier 3) merge on
  cosine-feature similarity gated by dilated adjacency. Tuning lives almost entirely in the
  `superpoints`/`hdbscan`/`grow`/`objects` blocks of `run.yaml`.
- **No test framework is wired up** — `run_semantic_universe` on a log is the smoke test.
  The grow/consolidate stages read only `Universe`/`Superpoints`/`PointFeatures` + std (no
  parlay/PCL), so they're the easiest to unit-test in isolation.
- **Everything is online + local + grow-only.** Any new stage should follow the same contract:
  read a const `Universe`, operate on the robot-window crop, store results by global index,
  never mutate/reindex the map.
- **Known simplifications to be aware of:** pinhole cameras only (no fisheye model); features
  are local-context (no global refinement); objects never un-merge (monotonic); superpoints
  are discarded every window (no temporal superpoint tracking).
