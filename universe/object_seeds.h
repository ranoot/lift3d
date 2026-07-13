#pragma once
// Object store: per-class instance discovery + feature-guided growing over the
// semantic Universe.
//
// SGS-3D's object stage has two halves that this class owns together:
//   1. SEEDING (LOCAL window, HDBSCAN*). Within each semantic THING class, cluster the
//      still-UNCLAIMED points in xyz (3-d) with the parallel HDBSCAN* library
//      (wangyiqiu/hdbscan). Clustering runs ONLY on the crop around the robot (never the
//      whole map), and only over "matured" coarse cells (enough occupied fine voxels).
//      Each dense cluster becomes a new object seed. Strict on purpose (large min_pts /
//      min_cluster_size) so we under- rather than over-detect.
//   2. GROWING (local, class-histogram-guided). Within the robot radius, absorb VCCS
//      superpoints whose class histogram matches an object and whose volume overlaps it
//      (see growLocal). This is what makes objects grow past their seed.
//
// Objects are PERSISTENT: they keep identity and accumulate points across frames. A
// per-map-point `owner_` claimed-list records which object owns each point, so seeding
// only ever clusters unclaimed points and growing never double-claims. Unlike the
// ephemeral superpoints, the object list is only ever appended to and grown, never
// wiped.
//
// Like Superpoints, this layer only READS a const Universe (+ the superpoint store
// during growing); the heavy hdbscan/parlay dependency lives entirely in
// object_seeds.cpp, and the (parlay-free) growing lives in object_grow.cpp, so this
// header stays light for its includers.

#include "universe.h"        // Universe, ClassKind

#include <cstdint>
#include <vector>

class Superpoints;           // superpoints.h (pulled in by object_grow.cpp)
class InstanceIdGraph;       // instance_ids.h (pulled in by object_seeds.cpp)

// One discovered object instance: a dense HDBSCAN* cluster of same-class points,
// possibly grown by absorbing class-consistent superpoints. `points` are universe
// global indices (so it lifts straight back onto the world); `centroid` is their mean
// (the viz marker). `id` is the object's stable identity (its index in the store's
// list, which is append-only).
struct ObjectSeed {
    int                id = -1;                     // stable identity (index in list_)
    int                class_id = -1;               // resolved THING class id
    ClassKind          kind = ClassKind::Thing;     // always Thing (stuff isn't seeded)
    std::vector<int>   points;                      // universe global indices (members)
    float              centroid[3] = {0, 0, 0};     // mean of member xyz
    // SAI3D affinity histogram: per-thing-class member counts, indexed by registry class id
    // (size == semantics().size()). This is the object's side of the grow-affinity cosine:
    // cosine(superpoint.hist, seed.hist) * containment. A single-class HDBSCAN proposal is a
    // spike at its class; growing re-aggregates it.
    std::vector<float> hist;
    // Raw OV-DVIS++ instance ids observed on the members (deduped): from the seed's
    // points at birth, unioned with absorbed superpoints' ids during growLocal.
    // Consolidation bridges seeds whose ids share a co-touch group (InstanceIdGraph).
    std::vector<int>   inst_ids;
};

class ObjectSeeds {
public:
    // HDBSCAN* seeding strictness (used by seedLocal).
    struct Params {
        int  min_pts = 15;             // HDBSCAN* core-distance k (density estimate)
        int  min_cluster_size = 50;    // flat-extraction strictness (smaller -> noise)
        int  min_class_points = 50;    // skip a class with fewer unclaimed points
        bool allow_single_cluster = true;  // let a single-object class resolve to 1 seed
        // Maturity gate: a coarse cell of edge `maturity_res` (m) is clusterable only
        // once it holds >= `maturity_min` occupied fine voxels (map points). Keeps
        // HDBSCAN off half-observed, sparse geometry. (~2 m^3 default cell.) Used by
        // seedLocal only; the per-view seedFromIndices path skips maturity (per-view
        // visibility replaces it).
        float maturity_res = 1.26f;    // coarse-cell edge (m); 1.26^3 ~= 2 m^3
        int   maturity_min = 200;      // min occupied fine voxels in a cell to cluster it
        // seedFromIndices only: run ONE hdbscan<4> over all classes at once (class as a
        // far-separated 4th coordinate) instead of a scan per class. false => per-class
        // hdbscan<3> loop (the proven fallback).
        bool  single_scan = true;
        // seedFromIndices only: let the model's instance segmentation drive seeding. When
        // true (and an InstanceIdGraph is supplied), thing points are grouped by (class,
        // RESOLVED instance) -- the graph bridges DVIS over-segmentation so each group is
        // one physical object -- and each such group is seeded WHOLE (no geometric split),
        // bypassing single_scan/per-class HDBSCAN. Thing points with no instance id fall
        // back to per-class HDBSCAN. false => the pure-geometry paths above.
        bool  use_instance_ids = true;
        // Overlapping point-set model (mirrors Objects::Params::overlap_sets): when set, a
        // point is NOT removed from later clusters once claimed -- proposals may share points.
        // Off => the exclusive within-frame claim (byte-for-byte the current behaviour).
        bool  overlap_sets = false;
    };

    // Class-histogram-guided growing knobs (used by growLocal). Affinity of a superpoint to
    // an object = cosine(class histograms) * containment(shared voxels / superpoint voxels)
    // -- SAI3D region affinity. Only DOMINANT-THING superpoints (class_id >= 0) are growable.
    struct GrowParams {
        float affinity_thresh = 0.5f;  // merge iff cosine(hist) * containment >= this
        bool  require_class   = true;  // require sp.class_id == obj.class_id when sp has one
        // ---- SAI3D-style neighborhood-aware growing (opt-in; off => the legacy single pass) --
        // `neighbor_pool` smooths each superpoint's class histogram with its spatial
        // neighbors' (decayed) before the affinity cosine, so a boundary superpoint is
        // decided by its neighborhood rather than itself alone (a cheap judge_connect). `sweeps`
        // re-runs the assignment loop so a superpoint that becomes reachable only after an
        // object grew earlier in the pass gets reconsidered (SAI3D iterative region growing).
        bool  neighbor_pool = false;   // pool sp histogram over KNN neighbors before cosine
        int   sweeps        = 1;       // assignment sweeps (>1 revisits sps as objects grow)
        float dis_decay     = 0.5f;    // neighbor-pooling decay weight
        int   pool_k        = 6;       // KNN count over superpoint centroids for pooling
        // Overlapping point-set model (mirrors Objects::Params::overlap_sets): when set, a
        // superpoint is annexed by EVERY matching proposal (not just its best) and its points
        // are never exclusively claimed, so proposals may share points. Off => single-best,
        // first-claim-wins (byte-for-byte the current behaviour).
        bool  overlap_sets  = false;
    };

    // SEED new objects from the still-unclaimed THING points, once per thing class,
    // over the LOCAL window around `center` (radius <= 0 => whole map). Only points in
    // "matured" coarse cells (>= p.maturity_min occupied fine voxels) are clustered.
    // Appends surviving HDBSCAN* clusters as new objects (assigning ids and claiming
    // their points); existing objects are left untouched. Bumps version().
    void seedLocal(const Universe& uni, const Params& p,
                   const float center[3], float radius);

    // PER-VIEW seeding: cluster an EXPLICIT set of universe global indices (a frame's
    // visible ∩ non-frozen ∩ thing points) instead of a robot-radius crop. No maturity
    // gate (per-view visibility replaces it); still skips already-claimed points and
    // buckets by resolved thing class. Populates each new seed's inst_ids from its
    // members. Appends surviving clusters as new objects; bumps version().
    //
    // When `idg` is non-null and p.use_instance_ids is set, seeding is INSTANCE-GUIDED:
    // thing points are grouped by (class, idg->root(instance id)) and each group is seeded
    // whole (the model already segmented it; the graph has bridged its over-segmented ids),
    // with only the no-instance remainder falling back to per-class HDBSCAN. `idg == nullptr`
    // (or the flag off) reproduces the pure-geometry single_scan / per-class behaviour.
    void seedFromIndices(const Universe& uni, const std::vector<int>& gidx,
                         const Params& p, InstanceIdGraph* idg = nullptr);

    // GROW this frame's proposals by absorbing class-consistent, spatially-overlapping
    // superpoints from `sp` (affinity = cosine(class histograms) * containment). ALL of a
    // matched superpoint's unclaimed points are annexed -- including its unlabeled/stuff
    // boundary points -- so the object completes geometry the 2D labels missed (one
    // superpoint == one object). Points already claimed this frame are never stolen.
    // Operates over the whole ephemeral proposal list (already bounded to the current view).
    // Bumps version() if anything changed. `voxel` is the map resolution used for the
    // volume/containment test.
    void growLocal(const Universe& uni, const Superpoints& sp,
                   const GrowParams& g, float voxel);

    const std::vector<ObjectSeed>& list() const { return list_; }
    int  size() const { return static_cast<int>(list_.size()); }

    // EXPERIMENT helper: time a single HDBSCAN* run (hdbscan + dendrogram + flat-cluster
    // extraction) over an explicit set of universe global indices, clustering their xyz.
    // Returns the wall time in ms (0 if fewer than min_pts points), and optionally the
    // number of flat clusters found in `out_clusters`. Touches no store state -- it lets
    // the per-view experiment measure HDBSCAN cost on the visible set without seeding.
    // Kept here (not in the experiment TU) so parlay stays confined to object_seeds.cpp.
    static double hdbscanTimeOnly(const Universe& uni, const std::vector<int>& gidx,
                                  const Params& p, int* out_clusters = nullptr);

    // Claimed-list queries (owner object id per universe global index, -1 = unclaimed).
    int  owner(int g) const {
        return (g >= 0 && g < (int)owner_.size()) ? owner_[g] : -1;
    }
    bool claimed(int g) const { return owner(g) >= 0; }

    // Bumped on every seed/grow that changes the store (change detection for a viz).
    std::uint64_t version() const { return version_; }

private:
    std::vector<ObjectSeed> list_;             // persistent, append-only objects
    std::vector<int>        owner_;            // per map point: owning object id, or -1
    std::uint64_t           version_ = 0;

    // Claim map point `g` for object `oid`, growing owner_ as the map grows.
    void claim(int g, int oid) {
        if (g < 0) return;
        if (g >= (int)owner_.size()) owner_.resize((std::size_t)g + 1, -1);
        owner_[g] = oid;
    }
};
