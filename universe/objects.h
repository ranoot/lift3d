#pragma once
// Objects: the TOP tier of the 3-tier discovery pipeline (superpoints -> seeds ->
// objects). It consolidates the persistent ObjectSeeds (tier 2) into fewer, cleaner
// instances via an affinity merge, fixing HDBSCAN oversegmentation: adjacent,
// feature-similar seed fragments of one physical object are pulled into a single Object.
//
// Each Object owns a set of member SEED ids and DERIVES its points / mean feature /
// centroid from them. So as a member seed keeps accumulating superpoints (tier 1 -> 2),
// its object re-derives and grows with it -- a merged seed is never frozen. Objects are
// PERSISTENT + append-only: a seed is assigned to at most one object and never moves
// (monotonic; there is no un-merge). The object list is the pipeline's primary output.
//
// Seed<->object merge = feature similarity SCORED, adjacency GATED. A seed joins the
// most feature-similar in-radius object (cosine of mean features >= threshold) that it
// physically touches (a fraction >= adj_min of the seed's voxels lie within a K-voxel
// dilation of the object's voxels). Adjacency must NOT be folded into the score: it is a
// boundary fraction, so for two comparably-sized fragments of one object it is small even
// when they abut, and cosine * adjacency can never clear the threshold (the old bug -- one
// object per seed). Plain containment can't be used either: seeds are disjoint point sets
// and an object is built from OTHER seeds, so a fresh seed shares zero voxels with it.
//
// Parlay-free (pure std over Universe / ObjectSeeds / PointFeatures); lives entirely in
// object_consolidate.cpp so this header stays light for its includers.

#include "universe.h"        // Universe, ClassKind
#include "object_seeds.h"    // ObjectSeeds, ObjectSeed

#include <cstddef>
#include <cstdint>
#include <vector>

class PointFeatures;         // point_features.h (pulled in by object_consolidate.cpp)

// One consolidated object instance: the union of one or more feature-consistent,
// spatially-adjacent seeds. `member_seeds` are seed ids into ObjectSeeds::list();
// `points` (universe global indices), `mean_feat`, and `centroid` are DERIVED from those
// seeds' current members. `id` is the object's stable identity (index in the store list).
struct Object {
    int                id = -1;
    int                class_id = -1;
    ClassKind          kind = ClassKind::Thing;
    std::vector<int>   member_seeds;                // seed ids owned by this object
    std::vector<int>   points;                      // derived: union of member seeds' points
    std::vector<float> mean_feat;                   // derived: mean feature over members
    float              centroid[3] = {0, 0, 0};     // derived: mean of member xyz
};

class Objects {
public:
    // Consolidation knobs. Adjacency is a GATE (do the seed and object physically
    // touch?), NOT a score: because it is a boundary fraction it is structurally small
    // for two comparably-sized fragments, so folding it into the affinity made the merge
    // unreachable. A seed merges into the most feature-similar in-radius object that it
    // touches (adjacency >= adj_min) and whose cosine >= affinity_thresh.
    struct Params {
        float affinity_thresh = 0.6f;  // merge a seed iff cosine(mean feats) >= this
        float adj_min         = 0.05f; // adjacency GATE: min fraction of seed voxels touching
        float adj_dilate      = 0.15f; // adjacency dilation (m); K = round(adj_dilate/voxel)
        float cand_radius     = 2.0f;  // seed<->object gating distance (m)
        bool  require_class   = true;  // require seed.class_id == obj.class_id
    };

    // Consolidate the seeds whose centroid is within `radius` of `center`: each still-
    // unassigned seed merges into the best-affinity object above threshold, or spawns a
    // new object. Every object with an in-radius member seed is re-derived (its seeds may
    // have grown via superpoints since the last call). `voxel` is the map resolution.
    // Bumps version() if anything changed.
    void consolidate(const Universe& uni, const ObjectSeeds& seeds, const PointFeatures& pf,
                     const Params& g, const float center[3], float radius, float voxel);

    const std::vector<Object>& list() const { return list_; }
    int  size() const { return static_cast<int>(list_.size()); }

    // Bumped on every consolidate that changes the store (change detection for a viz).
    std::uint64_t version() const { return version_; }

private:
    std::vector<Object> list_;          // persistent, append-only objects
    std::vector<int>    seed_owner_;    // seed id -> owning object id, or -1 (grow-only)
    std::uint64_t       version_ = 0;

    int  seedOwner(int sid) const {
        return (sid >= 0 && sid < (int)seed_owner_.size()) ? seed_owner_[sid] : -1;
    }
    void assignSeed(int sid, int oid) {
        if (sid < 0) return;
        if (sid >= (int)seed_owner_.size()) seed_owner_.resize((std::size_t)sid + 1, -1);
        seed_owner_[sid] = oid;
    }
};
