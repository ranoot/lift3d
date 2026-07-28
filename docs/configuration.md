# Configuration reference

Everything the pipeline does is driven by one YAML file (`run.yaml` in the repo root), loaded
by `universe/run_config.cpp` into `RunConfig` / `semantic::Params`. A handful of CLI flags
override it.

**Every key is optional.** Anything omitted falls back to the built-in C++ default, so a
minimal config is just a `log:` line. The "Default" column below is the *C++* default, which is
often different from the value checked into `run.yaml` — that file is a tuned working
configuration, not a statement of defaults.

For *why* each stage works the way it does, see [`doc.md`](../doc.md).

---

## CLI flags

### `run_semantic_universe`

```
run_semantic_universe --config run.yaml [--log <folder>] [--out labeled.pcd] [--viz <endpoint>]
```

| Flag | Effect |
|------|--------|
| `--config` | **Required.** Path to the YAML config. |
| `--log` | Override `log:`. |
| `--out` | Write a labelled PCD at the end — `PointXYZL`, `label = classId + 1`, `0 = unlabeled`. Not the same thing as the YAML `out:` key. |
| `--viz` | Override `viz_endpoint:`. Empty ⇒ headless. |

### `run_on_input_demo`

```
run_on_input_demo --config run.yaml [--log <folder>] [--start N] [--count N] [--stride N]
```

Viz is controlled entirely by the YAML `viz_endpoint`; start `viz_server` with `--out run.rrd`
to persist a recording.

### `exp_per_view`

```
exp_per_view --config run.yaml [--log <folder>]
    env: EXP_SNAPSHOTS=400,600  EXP_WARMUP=<stop frame>
```

### `gmd_stream_check`, `object_publisher_check`

No arguments. No servers, no GPU, no log.

---

## Recording and inputs

| Key | Default | Meaning |
|-----|---------|---------|
| `log` | `""` | Dog-log recording folder. A leading `~` is expanded. |
| `calib` | `""` | Camera calibration XML. **Empty ⇒ `<log>/../CameraCalRaiboAsQUGV113.xml`.** |
| `cam` | `211` | Camera sensor id. **Use 111 or 211** (pinhole); 11/21 are fisheye and unsupported. |
| `out` | `sem_universe.rrd` | Legacy viz `.rrd` sink. With visualization now in Python, the recording is actually written by `viz_server --out`; this key is vestigial for the current drivers. |

### Accumulation window

| Key | Default | Meaning |
|-----|---------|---------|
| `start` | `0` | First image index. |
| `count` | `50` | Number of images to process. |
| `stride` | `1` | Sub-sample the image window (clamped to ≥ 1). |

---

## Map and projection

| Key | Default | Meaning |
|-----|---------|---------|
| `voxel` | `0.05` | World voxel fusion size (m). Also becomes the VCCS voxel resolution. |
| `radius` | `0.0` | Label crop radius (m) around the robot. `0` ⇒ whole map. Fallback for `proj_radius`. |
| `proj_radius` | `0.0` | Tighter z-buffer **candidate cull** radius (m) — a projection optimization, *not* a processing window. `0` ⇒ use `radius`. |
| `tau` | `0.1` | Z-buffer visibility tolerance (m). A point is visible iff in-frame and within `tau` of the buffered nearest depth at its pixel. |

### The depth splat

A sparse voxel map projected into a dense image leaves **holes**, and a point behind a wall
that lands in a hole is nobody's occludee — it leaks through as "visible". The splat rasterizes
each point's depth into a window around its pixel so a nearby surface voxel can win the hole
and occlude it. Only the **depth test** dilates; a point's own `(u,v)` is always its exact
centre pixel.

Hole size scales with `1/depth`, so a single constant radius either under-fills near surfaces
(occluded points leak the foreground label) or over-halos far ones (visible surfaces
self-occlude). The depth-scaled mode sizes the radius to one voxel's image footprint:

```
r(c) = clamp( round(fx · voxel · splat_mult / c), splat_min, splat )
```

| Key | Default | Meaning |
|-----|---------|---------|
| `splat` | `1` | Near-field **cap** `r_max` (px) — or the constant radius when `splat_mult == 0`. |
| `splat_mult` | `0.0` | Overlap factor on the voxel footprint. `0` ⇒ legacy constant radius `splat`. |
| `splat_min` | `2` | Far-field floor on the depth-scaled radius (px). |
| `surf_band` | `0.0` | Front-surface band (m) for the working set: at each pixel keep every visible point within this depth of the nearest. `0` ⇒ one point per pixel (sparsest); `~0.1` keeps the full front surface, giving clustering more geometry. Bounded by image resolution either way, so it does not grow with map density. |

---

## Semantic labelling

| Key | Default | Meaning |
|-----|---------|---------|
| `things_only` | `false` | Only stamp *thing* classes; skip stuff. |

### `voting:` — cross-frame per-point class

Off ⇒ **naive**: a point's class is whatever the *current* frame's 2D segmentation projects
onto it (overwritten every frame, no accumulation). On ⇒ every observation **votes** into a
persistent per-point sparse histogram and the class is its stuff-biased argmax — a multi-view
consensus rather than the last frame's guess. The histogram survives across frames and is
cleared only when its voxel is tombstoned.

| Key | Default | Meaning |
|-----|---------|---------|
| `voting.enabled` | `false` | Master switch. |
| `voting.stuff_bias` | `2.0` | Weight (≥ 1) on *stuff* votes in the argmax. A point already seen as wall/floor/ceiling resists flipping to a thing until things outvote stuff by this factor. This is the principled curb on label-leak painting objects onto walls: a sparse z-buffer hole lets a wall point grab a stray `chair` pixel, and one biased vote can't flip it. `1` ⇒ symmetric. |
| `voting.min_votes` | `1` | Label gate: minimum total votes before a class resolves. `1` ⇒ off. |
| `voting.min_conf` | `0.0` | Label gate: minimum winning fraction of raw votes. `0` ⇒ off. |
| `voting.whole_map` | `false` | **Vote scope A/B.** `false` ⇒ only this view's front-surface working set (the thin visible skin). `true` ⇒ the `proj_radius` crop is dropped for the frame, the **whole map** is projected, and every z-buffer-visible point at any range casts a vote. Costs one whole-map projection per frame. |

**Why `whole_map` matters:** the SAI3D affinity features (superpoint and proposal class
histograms) are built from the accumulated per-point votes. Voting on only the front shell
accumulates a per-frame crust instead of a map-wide histogram, so growing has nothing
meaningful to match on.

Class ids come from a vocabulary keyed on class **name**, so a runtime vocabulary change merely
interns new names — ids already assigned never shift and earlier votes stay valid.

---

## Inference server and vocabulary

| Key | Default | Meaning |
|-----|---------|---------|
| `endpoint` | `ipc:///tmp/inf_server.ipc` | ZMQ endpoint of `inf_server`. |
| `thing` | *(built-in list)* | Open-vocabulary **thing** class names. Object discovery runs on these. |
| `stuff` | *(built-in list)* | Open-vocabulary **stuff** class names (wall/floor/ceiling). |
| `dynamic` | `[]` | Classes whose **geometry is rejected at ingest**: incoming scan points projecting onto one of these 2D segments are dropped *before* fusion, so a moving object never accumulates a smear. |

> **`dynamic` classes must ALSO appear in `thing`/`stuff`** — that mask is exactly what the
> filter keys on, and if the model isn't asked to segment the class there is no mask.

A single LiDAR sweep has one return per direction, so the ingest rejection needs no z-buffer:
if a point lands on a person pixel it *is* the person. A second cleanup path tombstones already
fused points that a later view sees on a person mask — this catches people the wide-FOV LiDAR
fused off to the camera's side, where no 2D mask existed to reject them.

---

## What gets fused: `frustum:`

The LiDAR sweeps ~360° but the camera sees a narrow cone, so by default every frame fuses a lot
of geometry no image can ever label. With this on, each synced frame admits **only** the scan
points inside that frame's camera view. The map stays persistent and grow-only — this clips
each incoming sweep, it does not delete anything the camera saw earlier.

| Key | Default | Meaning |
|-----|---------|---------|
| `frustum.enabled` | `false` | Also accepted as the flat key `frustum_only`. |
| `frustum.near` | `0.0` | Min accepted depth (m). `0` ⇒ bound off. |
| `frustum.far` | `0.0` | Max accepted depth (m). `0` ⇒ bound off. |

---

## Working set

`working_set:` selects what is **read back out** of the map each frame. This is a different
thing from `frustum:` above, which clips what is **fused at ingest**.

| Key | Default | Meaning |
|-----|---------|---------|
| `working_set.mode` | `shell` | `shell` \| `frustum`. |
| `working_set.near` | `0.0` | Min depth (m); `0` ⇒ any point in front of the camera. |
| `working_set.far` | `0.0` | Max depth (m); `0` ⇒ **uncapped** (expensive). |

- **`shell`** — the front-surface skin: nearest visible point per image pixel plus `surf_band`.
  Cheap and bounded by image resolution, but it is one frame's visible crust, so a superpoint
  covers a sliver of surface and growing has almost nothing to absorb.
- **`frustum`** — *additionally* admits the **occluded** live map points inside the camera cone
  out to `far`. A superpoint then spans a whole object rather than its front face, so growing
  has volume to move into the shell-seeded proposals.

Two things to know:

1. **This does not affect HDBSCAN.** Object proposals are always seeded from the z-buffer-visible
   shell in both modes — a proposal is a hypothesis about geometry the camera *resolved*, and
   seeding on hidden points would invent objects behind walls. Occluded geometry reaches objects
   the intended way: superpoints spanning it are *grown* into shell-seeded proposals.
2. **`frustum` only makes sense with `voting.whole_map: true`.** A point the front shell hides
   carries no label from *this* frame, so its class — and hence the superpoint histogram it
   contributes to — has to come from the accumulated votes.

`far` is what keeps frustum mode bounded; uncapped down a corridor means most of the map, every
frame. The projection candidate crop is widened automatically to cover `far` (padded by the
camera↔robot offset, since the crop is a sphere about the robot while `far` is a depth from the
camera), so `proj_radius` never clips the requested frustum.

The per-frame `[timing]` line reports `cand` → `shell` → `seg` → `thing` → `voted`, which is how
these toggles are compared.

---

## Tier 1 — `superpoints:` (VCCS oversegmentation)

Runs on the **full** view (including stuff and unlabelled points) so superpoints follow real
surfaces and each is one object's local geometry; the class histogram supplies the semantics
that growing matches on. **Ephemeral** — the list is replaced every refresh, nothing is carried
across frames.

| Key | Default | Meaning |
|-----|---------|---------|
| `superpoints.enabled` | `false` | |
| `superpoints.seed` | `0.60` | VCCS seed resolution (m). **The dominant knob for superpoint size** — larger ⇒ fewer, larger superpoints. |
| `superpoints.thing_frac` | `0.50` | Min fraction of all member points carrying the winning thing class for the superpoint to adopt it. |
| `superpoints.spatial` | `0.40` | VCCS spatial importance. |
| `superpoints.normal` | `1.00` | VCCS normal importance (geometry-driven). |
| `superpoints.refine` | `0` | `refineSupervoxels` iterations. |

Colour importance is fixed at 0 — VCCS runs on geometry-only `PointXYZ`. The VCCS voxel
resolution is taken from the map `voxel`.

**To oversegment more aggressively:** lower `seed` (e.g. 0.65 → 0.3–0.4); optionally raise
`normal` to split flat-but-differently-oriented surfaces; keep `refine` at 0 (refinement merges
and smooths — the opposite direction).

---

## Tier 2 — `hdbscan:` (object proposals)

Per-class HDBSCAN\* over the frame's visible **thing** points. Proposals are ephemeral —
re-clustered every frame and non-claiming — so the same physical object is re-proposed with
overlapping voxels, which is what the tier-3 consolidation accumulates evidence from.

| Key | Default | Meaning |
|-----|---------|---------|
| `hdbscan.enabled` | `false` | |
| `hdbscan.min_pts` | `15` | HDBSCAN\* `min_samples` (core-distance k). **The outlier-aggressiveness knob**: higher ⇒ larger core distances ⇒ scattered points are genuinely isolated and flagged noise instead of forming phantom objects. It also inflates mutual reachability on thin bridges so they break. 5 is far too permissive; ~10–20 flags real outliers. |
| `hdbscan.min_cluster_size` | `50` | Flat-extraction strictness; smaller clusters dissolve into noise. |
| `hdbscan.min_class_points` | `50` | Skip a class with fewer points this frame. |
| `hdbscan.allow_single_cluster` | `true` | **Keep this `true`.** A compact single object never produces a 2-way split, so its only cluster is the root; with `false` the root is deselected and the entire object is flagged noise — the object is lost. With `leaf_selection: true` this does not cause bridging, because leaf mode selects child leaves when two objects genuinely split. Set `false` only to forbid a whole class resolving to one blob when you never expect single-object classes. |
| `hdbscan.leaf_selection` | `false` | Cluster selection: `false` = Excess-of-Mass (fewer, larger, prefers stable parents); `true` = **leaf** (more, smaller, more homogeneous — splits merged blobs). Pair `true` with a modest `min_cluster_size`. |
| `hdbscan.single_scan` | `true` | One `hdbscan<4>` over all classes (class as a far-separated 4th coordinate) instead of a scan per class. `false` ⇒ the per-class `hdbscan<3>` fallback. |
| `hdbscan.sor_enabled` | `false` | Statistical Outlier Removal before clustering: drop sparse points so thin noise bridges can't fuse distinct objects. |
| `hdbscan.sor_mean_k` | `16` | Neighbours per point for the density estimate. |
| `hdbscan.sor_std_mul` | `1.0` | Keep points with mean neighbour distance ≤ `μ + std_mul·σ`. **Smaller ⇒ more aggressive.** |
| `hdbscan.split_radius` | `0.25` | Post-HDBSCAN Euclidean split (m): break each flat cluster into spatially **connected** components. HDBSCAN\* clusters are MST-connected but can bridge empty space (a small tail glued to a dense blob), giving one proposal spanning two visibly separate objects. This forces contiguity. Set a touch above the voxel size (~5× voxel) so real surfaces stay whole; `0` ⇒ off. |

> The `maturity_res` / `maturity_min` gate documented in `doc.md` §7.4 applies to the
> radius-crop `seedLocal` path and is **not exposed in `run.yaml`** — the current per-view
> path (`seedFromIndices`) uses per-view visibility instead.

---

## `grow:` — class-histogram-guided growing

Absorbs class-consistent, volume-overlapping superpoints into the frame's proposals. SAI3D
region affinity:

```
affinity = cosine(class histograms) × containment(shared voxels / superpoint voxels)
```

Because cosine is ≈ 1 for a matched class, the threshold is effectively a **containment floor**.

| Key | Default | Meaning |
|-----|---------|---------|
| `grow.enabled` | `false` | Requires `superpoints` **and** `hdbscan` to also be on. |
| `grow.affinity` | `0.5` | Merge threshold on `cosine(hist) × containment`. Start ~0.4–0.5 and tune. |
| `grow.require_class` | `true` | Only merge a superpoint into a proposal of the same thing class. |
| `grow.neighbor_pool` | `false` | Pool each superpoint's histogram over its KNN neighbours before the cosine, so a boundary superpoint is decided by its neighbourhood rather than itself alone (a cheap `judge_connect`). |
| `grow.sweeps` | `1` | Assignment sweeps. `>1` revisits superpoints that only become reachable after an object grew earlier in the pass (SAI3D iterative region growing). |
| `grow.dis_decay` | `0.5` | Decay weight on neighbour histograms when pooling. |
| `grow.pool_k` | `6` | KNN count over superpoint centroids for pooling. |

---

## Tier 3 — `objects:` (the primary output)

A single-tier **Union-Find over the persistent seed layer**. Every per-frame proposal is
retained as a seed; a new seed is unioned into every same-class component it is *contained* in
past that component's level bar. A component's **support** is its member-seed count, and it
becomes a reported **object** once support reaches `min_merges`. Objects are virtual — there is
no committed object entity. This is progressive merging (SGS-3D / SAI3D), not a one-shot
promotion.

| Key | Default | Meaning |
|-----|---------|---------|
| `objects.enabled` | `false` | Requires `grow` (and therefore superpoints + hdbscan). |
| `objects.merge_thresh` | `[0.5, 0.35, 0.25, 0.2]` | Union bar **by component level** (level 1 strictest … last entry used for everything above). Clamped at both ends. |
| `objects.min_merges` | `3` | Merged observations a component needs before it is reported as an object. |
| `objects.require_class` | `true` | Only union seeds of the same thing class. |
| `objects.consol_stride` | `1` | Fold proposals only every Nth frame. Adjacent frames observe near-identical geometry, so folding every one inflates support with tiny-baseline duplicates; striding samples genuinely distinct viewpoints, making `min_merges` reflect real multi-view evidence. `≤1` ⇒ every frame. |

### Global-context merging (opt-in)

With `global_context: false` the legacy single-frame path is used. With it on, inter-object
merges are **not** decided from one frame's voxel overlap: each frame deposits
confidence-weighted affinity onto a persistent edge graph (`adj = Σ sim·conf / Σ conf`,
mirroring SAI3D's multi-view averaging), and two components merge only once that accumulated
affinity clears the level bar with enough corroborating evidence **and** a neighbourhood-aggregated
score agrees (`judge_connect`). This kills single-frame spurious bridges. A proposal still joins
its best-containment same-class component every frame exactly as before; only the **bridging of
two established objects** is gated.

| Key | Default | Meaning |
|-----|---------|---------|
| `objects.global_context` | `false` | Master switch for the accumulated-affinity path. |
| `objects.min_evidence` | `2.0` | Min accumulated confidence before any inter-object merge. |
| `objects.dis_decay` | `0.5` | Neighbourhood-aggregation decay. |
| `objects.use_semantic_affinity` | `true` | Include `cosine(class histogram)` in the per-frame deposit. |
| `objects.use_containment` | `true` | Multiply each deposit by voxel containment, so a grazing overlap accrues only weak evidence — curbs over-merging. |
| `objects.small_seg_min` | `0` | `finish()` cleanup: absorb components with support below this. `0` ⇒ off. |

### Overlapping point sets (opt-in)

| Key | Default | Meaning |
|-----|---------|---------|
| `objects.overlap_sets` | `false` | Applied to **all three** of `objects`, `hdbscan` and `grow` from this one key. |

With `false`, points are exclusively owned (first claim wins). With `true`, higher layers stop
*owning* points: a superpoint is grown into **every** matching proposal rather than only its
best, points are never exclusively claimed, and proposals become overlapping bags of points.
This cures the "points gobbled up → sparse proposals → meaningless containment" failure. Tier 3
then becomes a virtual Union-Find whose sets are collections of proposals; a voxel may belong to
several components (located via a voxel→{components} index), and the final objects are made
disjoint only at synthesis by assigning each contested voxel to its highest-support component.

### Stage-enable chain

Growing requires superpoints **and** hdbscan; the objects tier requires growing. **Enabling
`objects` therefore implicitly requires the whole chain below it.**

---

## Visualization

| Key | Default | Meaning |
|-----|---------|---------|
| `viz_endpoint` | `""` | ZMQ endpoint the runner PUSHes to (`viz_server` binds PULL). **Empty ⇒ headless, zero overhead.** Overridden by `--viz`. Also accepted as `viz.endpoint`. |
| `viz.seg_overlay` | `both` | `off` \| `class` \| `instance` \| `both` — which coloured 2D segmentation overlays the visualizer paints onto the camera image. |
| `viz.seg_alpha` | `0.5` | Blend strength of the colour fill over the RGB (0..1). |
| `viz.seg_min_area` | `80` | Min region size in px before a segment gets a labelled box. |

These are forwarded to the Python side in the `begin` message, so the visualizer paints exactly
what `run.yaml` configures. The raw frame at `world/camera/image` is left untouched.

---

## `sign:` — async sign-text egress

When enabled, the largest sign-class instance mask of a frame is sent to the sign-understanding
module (bbox + timestamp + a match id) **if nothing is already being read** — reading takes
~1–2 s and frames arriving meanwhile are dropped on purpose. The returned text is attached to
the object whose 3D footprint contains the sign (biggest proposal wins on conflict), filling
`EAIRoomObject::directionContent` and drawing on the object in rerun. Progress is logged to
stderr as `[sign] DETECTED / SENT / RECEIVED / ATTACHED`.

| Key | Default | Meaning |
|-----|---------|---------|
| `sign.enabled` | `false` | Master switch — nothing is constructed when false. |
| `sign.classes` | `[sign]` | Vocab names treated as signs. **Must also appear in `thing:`**, or the 2D model never segments a sign. |
| `sign.min_px` | `60` | Ignore sign masks smaller than this (px). |
| `sign.vlm_url` | `http://localhost:8080/v1` | Real backend only. |
| `sign.workers` | `1` | Detector worker threads. |
| `sign.queue` | `4` | Detector queue size. |
| `sign.mock_delay_ms` | `1500` | **Mock backend only** — stands in for the VLM round trip, reproducing the latency the single-in-flight gate exists for. |
| `sign.mock_text` | `"MOCK: toilet -> left; exit -> right"` | Mock backend answer. |

The real VLM backend needs a `-DLIFT3D_USE_SIGN=ON` build. Otherwise the mock backend answers,
which is how the whole path is exercised end to end without OpenCV/CURL/VLM present.

---

## Outputs

| Output | How |
|--------|-----|
| **Object instances** | The primary output. `ObjectsHook` in the C++ API; `EAIRoomObject` list returned by `SemanticRunner::runOnInput`; `world/objects/*` in rerun. |
| **Labelled point cloud** | `run_semantic_universe --out labeled.pcd` — `PointXYZL`, `label = classId + 1`, `0 = unlabeled`. |
| **rerun recording** | `viz_server --out <file>.rrd`. |
| **Live viewer** | `viz_server --spawn`. |

### rerun entity tree

| Entity | Contents |
|--------|----------|
| `world/map/{things,stuff,unlabeled}` | The full segmented map, class-coloured, split by kind so stuff can be toggled off. Logged **once as static** — per-frame re-logging made recordings O(frames × map_size), many GB, which the viewer's memory-limit GC then purged mid-load. |
| `world/superpoints/{all,things,centroids,window}` | The current ephemeral VCCS window, per-superpoint hash colour; `window` is the crop sphere. Logged only on refresh frames. |
| `world/seeds/points` | Tier-2 proposal members (debug layer), per-seed hash colour. |
| `world/objects/{points,centroids}` | **The headline output.** Member points and centroid marker both coloured by the object's stable id hash, so a growing object keeps its colour and its centroid visibly matches its points. |
| `world/camera/image` | The raw camera frame (untouched). |
| `world/camera/seg_by_{class,instance}` | The 2D segmentation overlays, per the `viz:` block. |
