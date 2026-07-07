#pragma once
// Object store: per-class instance discovery + feature-guided growing over the
// semantic Universe.
//
// SGS-3D's object stage has two halves that this class owns together:
//   1. SEEDING (whole-map, HDBSCAN*). Within each semantic THING class, cluster the
//      still-UNCLAIMED points in xyz (3-d) with the parallel HDBSCAN* library
//      (wangyiqiu/hdbscan). Each dense cluster becomes a new object seed. Strict on
//      purpose (large min_pts / min_cluster_size) so we under- rather than over-detect.
//   2. GROWING (local, feature-guided). Within the robot radius, absorb VCCS
//      superpoints whose Mask3D feature matches an object and whose volume overlaps it
//      (see growLocal). This is what makes objects grow past their seed.
//
// Objects are PERSISTENT: they keep identity and accumulate points across frames. A
// per-map-point `owner_` claimed-list records which object owns each point, so seeding
// only ever clusters unclaimed points and growing never double-claims. Unlike the
// ephemeral superpoints, the object list is only ever appended to and grown, never
// wiped.
//
// Like Superpoints / PointFeatures, this layer only READS a const Universe (+ the
// superpoint / feature stores during growing); the heavy hdbscan/parlay dependency
// lives entirely in object_seeds.cpp, and the (parlay-free) growing lives in
// object_grow.cpp, so this header stays light for its includers.

#include "universe.h"        // Universe, ClassKind

#include <cstdint>
#include <vector>

class Superpoints;           // superpoints.h (pulled in by object_grow.cpp)
class PointFeatures;         // point_features.h (pulled in by object_grow.cpp)

// One discovered object instance: a dense HDBSCAN* cluster of same-class points,
// possibly grown by absorbing feature-consistent superpoints. `points` are universe
// global indices (so it lifts straight back onto the world); `centroid` is their mean
// (the viz marker); `mean_feat` is the averaged Mask3D feature over members that have
// one (empty until any member has a feature). `id` is the object's stable identity
// (its index in the store's list, which is append-only).
struct ObjectSeed {
    int                id = -1;                     // stable identity (index in list_)
    int                class_id = -1;               // resolved THING class id
    ClassKind          kind = ClassKind::Thing;     // always Thing (stuff isn't seeded)
    std::vector<int>   points;                      // universe global indices (members)
    float              centroid[3] = {0, 0, 0};     // mean of member xyz
    std::vector<float> mean_feat;                   // mean Mask3D feature over feat members
};

class ObjectSeeds {
public:
    // HDBSCAN* seeding strictness (used by seedUnclaimed).
    struct Params {
        int  min_pts = 15;             // HDBSCAN* core-distance k (density estimate)
        int  min_cluster_size = 50;    // flat-extraction strictness (smaller -> noise)
        int  min_class_points = 50;    // skip a class with fewer unclaimed points
        bool allow_single_cluster = true;  // let a single-object class resolve to 1 seed
    };

    // Feature-guided growing knobs (used by growLocal). Affinity of a superpoint to an
    // object = cosine(mean features) * containment(shared voxels / superpoint voxels).
    struct GrowParams {
        float affinity_thresh = 0.4f;  // merge iff cos * containment >= this
        float max_dispersion  = 0.5f;  // skip superpoints with feat_dispersion above this
        float cand_radius     = 1.5f;  // superpoint<->object gating distance (m)
        bool  require_class   = true;  // require sp.class_id == obj.class_id when sp has one
    };

    // SEED new objects from the still-unclaimed THING points, once per thing class,
    // over the WHOLE map. Appends surviving HDBSCAN* clusters as new objects (assigning
    // ids and claiming their points); existing objects are left untouched. Bumps
    // version(). Whole-map (not a local crop) by design.
    void seedUnclaimed(const Universe& uni, const Params& p);

    // GROW existing objects whose centroid is within `radius` of `center` by absorbing
    // feature-consistent, spatially-overlapping superpoints from `sp` (features from
    // `pf`). Merged superpoint points are claimed for the object; points already owned
    // by a different object are never stolen. Bumps version() if anything changed.
    // `voxel` is the map resolution used for the volume/containment test.
    void growLocal(const Universe& uni, const Superpoints& sp, const PointFeatures& pf,
                   const GrowParams& g, const float center[3], float radius, float voxel);

    // Union of member points of objects whose centroid is within `radius` of `center`,
    // appended to `out`. Feeds the whole-object feature call so an object extending past
    // the feature crop still gets features over its full extent.
    void collectInRadiusPoints(const float center[3], float radius,
                               std::vector<int>& out) const;

    const std::vector<ObjectSeed>& list() const { return list_; }
    int  size() const { return static_cast<int>(list_.size()); }

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
