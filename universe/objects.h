#pragma once
// Objects: the TOP tier of the discovery pipeline (superpoints -> proposals -> objects),
// as a SINGLE-TIER Union-Find over a persistent SEED layer. There is no committed "object"
// entity: an object is a purely VIRTUAL record of which part of the seed layer has been
// reinforced across enough views.
//
//   - SEEDS are the observations. Every per-frame object PROPOSAL (the ephemeral ObjectSeed,
//     tier 2) is retained here as one persistent, immutable Seed that holds its own member
//     points. Append-only; a seed is never mutated after birth.
//   - A Union-Find over seed ids groups seeds that are the same physical object. A new seed
//     is unioned into an existing component when it overlaps it (containment) past a
//     LEVEL-INDEXED bar: strict for a low-support component, progressively looser as the
//     component accumulates seeds (SGS-3D / SAI3D progressive multi-view merging). A
//     component's SIZE (its member-seed count) IS its multi-view support / level.
//   - OBJECTS are virtual. `list()` synthesises one Object per component whose support
//     reaches `min_merges`, gathering points by indexing into its member seeds. Promotion is
//     a graded, tunable property (`merge_thresh` / `min_merges`), never a one-shot binary
//     commit. This replaces the old brittle "clear one IoU threshold or spawn a permanent
//     new object" logic.
//
// Overlap is CONTAINMENT (shared / |new seed|), so a small re-observation inside a large
// mature component still scores high and keeps merging. Sub-`min_merges` components are
// retained but hidden (never pruned). Object `id` is the raw Union-Find root, so a merge may
// repoint (and recolour) a component once -- accepted. Parlay-free (pure std over Universe /
// ObjectSeeds); the fold-in + synthesis live in object_consolidate.cpp so this header stays
// light for its includers.

#include "universe.h"        // Universe, ClassKind
#include "object_seeds.h"    // ObjectSeeds, ObjectSeed
#include "voxel_util.h"      // objutil::VKey / VHash / VSet (per-component voxel maps)

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// One VIRTUAL object instance, synthesised from a Union-Find component of seeds (never
// stored between frames). `points` is the union of the component's member seeds' live points
// (one representative per voxel), `centroid` their mean, `level` the component's member-seed
// count (its multi-view support), and `id` the component's Union-Find root.
struct Object {
    int              id = -1;
    int              class_id = -1;
    ClassKind        kind = ClassKind::Thing;
    std::vector<int> points;                        // one representative gidx per voxel
    float            centroid[3] = {0, 0, 0};       // mean of member xyz
    int              level = 0;                      // member-seed count = multi-view support
};

class Objects {
public:
    // Progressive-merge knobs. A new seed is unioned into an existing component when its
    // containment in that component (shared voxels / new-seed voxels) clears the bar for the
    // component's current level: merge_thresh[level-1], clamped to the last entry (so the
    // top level uses the loosest listed bar). A component is REPORTED as an object once its
    // member-seed count (support) reaches min_merges.
    struct Params {
        std::vector<float> merge_thresh = {0.5f, 0.35f, 0.25f, 0.2f};  // by (level-1), clamped
        int  min_merges    = 3;      // report a component once its support >= this
        bool require_class = true;   // only union/overlap seeds of the same thing class

        // ---- SAI3D global-context merging (opt-in; off => the legacy one-shot path) -------
        // When `global_context` is set, inter-object merges are NOT decided from a single
        // frame's voxel overlap. Instead each frame DEPOSITS confidence-weighted affinity
        // onto a persistent edge graph (adj = Sum sim*conf / Sum conf, mirroring SAI3D's
        // multi-view averaging), and two components merge only once (a) that accumulated
        // affinity clears the progressive bar with enough corroborating evidence, and (b) a
        // neighborhood-aggregated score agrees (SAI3D's judge_connect). A proposal still
        // JOINS its best-containment same-class component every frame (support + fresh
        // voxels) exactly as before; only the bridging of two established objects is gated.
        bool  global_context      = false;  // master switch for the accumulated-graph path
        float min_evidence        = 2.0f;   // min accumulated conf before an inter-obj merge
        float dis_decay           = 0.5f;   // neighborhood-aggregation decay (judge_connect)
        bool  use_semantic_affinity = true; // cosine(class hist) term in the per-frame deposit
        bool  use_containment     = true;   // fold per-frame voxel containment into the deposit
        int   small_seg_min       = 0;      // mergeSmallComps: absorb comps with support < this

        // ---- Overlapping point-set model (opt-in; off => the exclusive-ownership path) ------
        // When set, proposals may share points (Phase 1) and components no longer EXCLUSIVELY
        // own voxels: a proposal contributes ALL its voxels to its home component (not just
        // unowned ones), a voxel may live in several components at once, overlap is located via
        // a voxel->{components} index instead of the single-owner pt_owner_ map, and the final
        // virtual objects are made disjoint at synthesis by assigning each contested voxel to
        // its HIGHEST-support component. Off => the fresh-only pt_owner_ path, byte-for-byte.
        bool  overlap_sets        = false;

        // The union bar for a component currently at `level` member seeds.
        float thresholdFor(int level) const {
            if (merge_thresh.empty()) return 0.3f;
            int i = level - 1;
            if (i < 0) i = 0;
            if (i >= static_cast<int>(merge_thresh.size())) i = static_cast<int>(merge_thresh.size()) - 1;
            return merge_thresh[static_cast<std::size_t>(i)];
        }
    };

    // Fold this frame's proposals (`seeds.list()`) into the persistent COMPONENT layer: each
    // proposal's voxels are matched against the components that already own them; the proposal
    // joins (and progressively unions) every same-class component whose containment clears
    // that component's level bar, contributing only its NEW (unowned) voxels and one unit of
    // support. Rebuilds the virtual object list. Whole-map, every frame; a proposal only
    // touches components that already own some of its voxels. `voxel` is the map resolution.
    // Bumps version().
    //
    // `view_vox` (optional) is the current frame's occupied-voxel set (the front-surface
    // shell): when supplied AND g.global_context is set, per-frame affinity confidence is
    // weighted by how fully each primitive is seen this frame (SAI3D `seg_seen`). Defaults
    // off => the legacy single-frame containment path, byte-for-byte.
    void consolidate(const Universe& uni, const ObjectSeeds& seeds, const Params& g,
                     float voxel, const objutil::VSet* view_vox = nullptr);

    // Postprocess (SAI3D merge_small_segs): absorb every component whose support is below
    // g.small_seg_min into its highest-accumulated-affinity neighbor from the edge graph;
    // components with no affinity neighbor are left as-is. No-op unless global_context and
    // small_seg_min > 0. Rebuilds the object cache + bumps version() if anything changed.
    void mergeSmallComps(const Universe& uni, const Params& g);

    // The promoted (support >= min_merges) virtual objects, rebuilt each consolidate().
    const std::vector<Object>& list() const { return cache_; }
    int  size() const { return static_cast<int>(cache_.size()); }

    // Total observations (proposals) folded in so far -- context for the promoted count.
    int  seedCount() const { return static_cast<int>(total_obs_); }

    // Bumped on every consolidate that changes the store (change detection for a viz).
    std::uint64_t version() const { return version_; }

private:
    // One persistent component: the aggregated footprint of every proposal that has merged
    // into it. `vox` maps each occupied voxel to one representative universe global index (so
    // the geometry is stored ONCE, deduped, not per observation); `cls` is a class -> vote
    // tally (majority resolves the object class; usually single-class under require_class);
    // `support` is the member-observation count == the object's level. Only Union-Find ROOTs
    // carry a populated Comp; a component absorbed by unite() has its maps cleared.
    struct Comp {
        std::unordered_map<objutil::VKey, int, objutil::VHash> vox;   // voxel -> representative gidx
        std::unordered_map<int, int>                           cls;   // class id -> vote weight
        std::unordered_set<int>                                ids;   // DVIS instance ids (global_context)
        int                                                    support = 0;   // observations == level
    };

    // One accumulated affinity edge between two components (global_context path). `sim_conf`
    // is the running Sum of per-frame similarity*confidence, `conf` the running Sum of
    // confidence, so the multi-view affinity is adj = sim_conf / conf (SAI3D eq.). Edges are
    // keyed by the packed component ids present at DEPOSIT time and resolved to roots lazily
    // (a resolved self-edge is an internal edge and dropped on the next compaction).
    struct EdgeAcc { float sim_conf = 0.0f; float conf = 0.0f; };

    std::vector<int>    parent_;   // Union-Find over component ids -> root
    std::vector<Comp>   comp_;     // per id; only roots are populated (parallel to parent_)
    std::vector<int>    pt_owner_; // gidx -> a component id (leaf); find() resolves to its root
    // overlap_sets: the non-exclusive replacement for pt_owner_ -- a voxel may belong to many
    // components at once, so this maps each occupied voxel to the (deduped) component leaf ids
    // that contain it. Used only to LOCATE which components a proposal overlaps (each leaf
    // resolved to its root via find()); it carries no ownership. Empty in the exclusive path.
    std::unordered_map<objutil::VKey, std::vector<int>, objutil::VHash> pt_vox_;
    std::vector<Object> cache_;    // synthesised promoted objects (rebuilt each consolidate)
    std::unordered_map<std::uint64_t, EdgeAcc> edges_;      // accumulated affinity (global_context)
    std::uint64_t       version_  = 0;
    std::uint64_t       total_obs_ = 0;   // proposals folded (informational seedCount)

    int  find(int c);                                       // path-compressed
    int  unite(int a, int b);                               // union by voxel count; returns surviving root
    int  newComp();                                         // allocate a fresh singleton component id
    void rebuildCache(const Universe& uni, const Params& g);

    // global_context helpers (defined in object_consolidate.cpp).
    static std::uint64_t ekey(int a, int b);               // order-independent packed edge key
    // Fraction of a component's voxels that lie in this frame's view set (SAI3D seg_seen).
    float visRatio(const Comp& c, const objutil::VSet* view) const;
    // After a frame's deposits, resolve every edge to roots, gate the touched edges on
    // accumulated affinity + neighborhood aggregation, apply the surviving merges, and
    // recompact the edge store onto the new roots. Returns true if any merge happened.
    bool resolveMerges(const std::vector<std::uint64_t>& touched, const Params& g);
};
