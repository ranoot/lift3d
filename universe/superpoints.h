#pragma once
// VCCS superpoints (oversegmentation) over the semantic Universe.
//
// SGS-3D's stage after per-point semantics is oversegmentation into "superpoints".
// The paper uses ScanNet's mesh oversegmentation, which assumes a static, complete
// scene; ours grows online as the robot drives, so we use VCCS (Voxel Cloud
// Connectivity Segmentation) via PCL's SupervoxelClustering and RE-RUN it
// periodically over the local region around the robot to keep pace with the map.
//
// VCCS is a batch algorithm, and here the superpoints are EPHEMERAL: each refresh
// re-segments ONLY the local crop around the robot and REPLACES `list_` with just
// that window's supervoxels -- nothing is merged or carried across refreshes. The
// window slides with the robot and is discarded/regenerated every stride, so
// `list()` always reflects only the most recently processed local window (the
// working set a downstream per-window stage consumes, e.g. seed discovery).
//
// Each superpoint carries a semantic THING class: among its members whose resolved
// class is a Thing, if one class holds >= thing_frac (default 50%), the superpoint
// takes that class; otherwise Unknown. Stuff and unlabeled members are ignored
// (they don't count toward the denominator). Stuff classes are never assigned.
//
// This layer only READS a Universe (mirrors how projection/semantics compose), so
// the Universe class stays focused. PCL's heavy supervoxel header is pulled in only
// by superpoints.cpp, so this header stays light for its includers.

#include "universe.h"        // Universe, ClassKind

#include <cstdint>
#include <vector>

class PointFeatures;         // point_features.h (pulled in by the .cpp)

// One over-segment: a spatially-connected cluster of universe points plus its
// resolved semantic thing-class. `points` holds global indices into the Universe
// map, so results lift straight back onto the world.
struct Superpoint {
    std::uint32_t    id = 0;                       // per-window VCCS label (>=1), not global
    float            centroid[3] = {0, 0, 0};      // mean of member xyz
    std::vector<int> points;                       // universe global indices (members)
    int              class_id = -1;                // resolved THING class id (-1 = unknown)
    ClassKind        kind = ClassKind::Unknown;    // Thing or Unknown (never Stuff)
    float            confidence = 0.0f;            // winner fraction among thing-labeled members

    // Mask3D-feature spread of the members: the trace of their feature covariance
    // divided by the feature dimension (mean per-dimension variance). A per-superpoint
    // feature-homogeneity score -- low => members are feature-consistent (likely one
    // object part). Computed only over members that already have a feature; 0 when
    // fewer than two do (variance undefined), with feat_count reporting how many did.
    float            feat_dispersion = 0.0f;
    int              feat_count = 0;               // # members that had a Mask3D feature
};

class Superpoints {
public:
    struct Params {
        float voxel_res  = 0.05f;   // VCCS voxel resolution (m); ~ map voxel
        float seed_res   = 0.60f;   // VCCS seed resolution (m); drives superpoint size (> voxel_res)
        float spatial_w  = 0.40f;   // spatial importance
        float normal_w   = 1.00f;   // normal importance (geometry-driven)
        float color_w    = 0.00f;   // color importance (0: we run on geometry-only PointXYZ)
        int   refine     = 0;       // optional refineSupervoxels iterations (0 = none)
        float thing_frac = 0.50f;   // majority threshold among thing-labeled members
    };

    // Re-segment the crop of `uni` within `radius` of `center` and REPLACE the list
    // with just that window's supervoxels (ephemeral -- nothing is carried over from
    // the previous refresh). radius <= 0 falls back to the whole map. When `pf` is
    // non-null, each superpoint also gets its Mask3D-feature dispersion (see Superpoint).
    void refreshLocal(const Universe& uni, const float center[3], float radius,
                      const Params& p, const PointFeatures* pf = nullptr);

    // Whole-map convenience: segment the entire world in one pass.
    void refreshWhole(const Universe& uni, const Params& p,
                      const PointFeatures* pf = nullptr);

    const std::vector<Superpoint>& list() const { return list_; }
    int  size() const { return static_cast<int>(list_.size()); }

    // Bumped on every refresh (change detection for a visualizer).
    std::uint64_t version() const { return version_; }

private:
    std::vector<Superpoint> list_;        // ONLY the latest window's supervoxels
    std::uint64_t           version_ = 0; // bumped each refresh (change detection)

    // Build list_ (membership + centroid + semantics) straight from one crop's VCCS
    // labels: labels[j] (1..k) is the supervoxel of crop point j, gidx[j] its
    // universe global index. Clears any previous contents (no cross-window merge).
    void buildFromCrop(const std::vector<int>& gidx, const std::vector<int>& labels,
                       std::uint32_t k, const Universe& uni, const Params& p,
                       const PointFeatures* pf);
};
