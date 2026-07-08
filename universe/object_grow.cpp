#include "object_seeds.h"
#include "superpoints.h"        // Superpoint, Superpoints
#include "point_features.h"     // PointFeatures
#include "voxel_util.h"         // objutil::VKey/VHash/VSet, voxelOf, dist2, cosine, meanFeature

// The (parlay-free) growing half of the object store: absorb feature-consistent,
// spatially-overlapping superpoints into nearby objects. Kept OUT of object_seeds.cpp
// so it stays free of the heavy hdbscan/parlay includes and is unit-testable on its
// own (it reads only Universe / Superpoints / PointFeatures + std).

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

using objutil::VKey;
using objutil::VHash;
using objutil::VSet;
using objutil::voxelOf;
using objutil::dist2;
using objutil::cosine;
using objutil::meanFeature;

// Transient per-object cache built for one growLocal call over the in-radius objects.
struct Cand {
    int                oid = -1;
    VSet               vox;                        // member voxels (grows as we merge)
    std::vector<float> feat;                       // mean Mask3D feature
    float              centroid[3] = {0, 0, 0};
    float              lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};  // AABB
};

}  // namespace

void ObjectSeeds::growLocal(const Universe& uni, const Superpoints& sp,
                            const PointFeatures& pf, const GrowParams& g,
                            const float center[3], float radius, float voxel) {
    if (list_.empty() || sp.size() == 0) return;
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;
    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);
    const float r2  = radius > 0.0f ? radius * radius : 0.0f;

    // ---- build a cache for every object whose centroid is within `radius` ---------
    // Only objects with at least one feature-carrying member are growable (we need a
    // mean feature to compare); others are skipped.
    std::vector<Cand> cand;
    std::vector<int>  obj_to_cand((std::size_t)list_.size(), -1);
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        const ObjectSeed& obj = list_[oid];
        if (radius > 0.0f && dist2(obj.centroid, center) > r2) continue;
        std::vector<float> mf = meanFeature(obj.points, pf);
        if (mf.empty()) continue;                          // no feature -> cannot match
        Cand c;
        c.oid = oid;
        c.feat = std::move(mf);
        c.centroid[0] = obj.centroid[0];
        c.centroid[1] = obj.centroid[1];
        c.centroid[2] = obj.centroid[2];
        bool first = true;
        c.vox.reserve(obj.points.size());
        for (int gi : obj.points) {
            if (gi < 0 || gi >= (int)cloud->size()) continue;
            const Universe::PointT& p = (*cloud)[gi];
            c.vox.insert(voxelOf(p, inv));
            if (first) {
                c.lo[0] = c.hi[0] = p.x; c.lo[1] = c.hi[1] = p.y; c.lo[2] = c.hi[2] = p.z;
                first = false;
            } else {
                c.lo[0] = std::min(c.lo[0], p.x); c.hi[0] = std::max(c.hi[0], p.x);
                c.lo[1] = std::min(c.lo[1], p.y); c.hi[1] = std::max(c.hi[1], p.y);
                c.lo[2] = std::min(c.lo[2], p.z); c.hi[2] = std::max(c.hi[2], p.z);
            }
        }
        if (first) continue;                               // no valid points
        obj_to_cand[oid] = (int)cand.size();
        cand.push_back(std::move(c));
    }
    if (cand.empty()) return;

    // ---- evaluate every superpoint against the candidate objects ------------------
    const float cr2 = g.cand_radius * g.cand_radius;
    std::vector<std::uint8_t> dirty((std::size_t)list_.size(), 0);
    bool changed = false;

    for (const Superpoint& s : sp.list()) {
        // Reliability prune: need a defined feature spread, and it must be low enough.
        if (s.feat_count < 2 || s.feat_dispersion > g.max_dispersion) continue;

        std::vector<float> sf = meanFeature(s.points, pf);
        if (sf.empty()) continue;

        // Superpoint voxel set (its volume unit).
        VSet svox;
        svox.reserve(s.points.size());
        for (int gi : s.points) {
            if (gi < 0 || gi >= (int)cloud->size()) continue;
            svox.insert(voxelOf((*cloud)[gi], inv));
        }
        if (svox.empty()) continue;
        const float inv_svol = 1.0f / (float)svox.size();

        // Pick the best-affinity candidate above threshold.
        int   best_oid = -1;
        float best_aff = g.affinity_thresh;               // must strictly reach threshold
        for (const Cand& c : cand) {
            if (g.require_class && s.class_id >= 0 && s.class_id != list_[c.oid].class_id)
                continue;
            // Cheap gate: skip if the superpoint is both far from the object centroid
            // AND outside its AABB (expanded by one voxel) -- it can't overlap.
            const bool near = dist2(s.centroid, c.centroid) <= cr2;
            const bool in_box =
                s.centroid[0] >= c.lo[0] - voxel && s.centroid[0] <= c.hi[0] + voxel &&
                s.centroid[1] >= c.lo[1] - voxel && s.centroid[1] <= c.hi[1] + voxel &&
                s.centroid[2] >= c.lo[2] - voxel && s.centroid[2] <= c.hi[2] + voxel;
            if (!near && !in_box) continue;

            int shared = 0;
            for (const VKey& v : svox) if (c.vox.count(v)) ++shared;
            if (shared == 0) continue;
            const float containment = (float)shared * inv_svol;
            const float aff = cosine(sf, c.feat) * containment;
            if (aff >= best_aff) { best_aff = aff; best_oid = c.oid; }
        }
        if (best_oid < 0) continue;

        // Merge: annex the superpoint's UNCLAIMED points into the winning object.
        // Points owned by another object are never stolen (first-claim wins).
        Cand&       c   = cand[obj_to_cand[best_oid]];
        ObjectSeed& obj = list_[best_oid];
        for (int gi : s.points) {
            const int ow = owner(gi);
            if (ow != -1) continue;                        // claimed (by this or another)
            obj.points.push_back(gi);
            claim(gi, best_oid);
            if (gi >= 0 && gi < (int)cloud->size())
                c.vox.insert(voxelOf((*cloud)[gi], inv));  // grow the volume for later sps
            dirty[best_oid] = 1;
            changed = true;
        }
    }

    // ---- recompute centroid + mean feature for every grown object -----------------
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        if (!dirty[oid]) continue;
        ObjectSeed& obj = list_[oid];
        double csum[3] = {0, 0, 0};
        int nvalid = 0;
        for (int gi : obj.points) {
            if (gi < 0 || gi >= (int)cloud->size()) continue;
            const Universe::PointT& p = (*cloud)[gi];
            csum[0] += p.x; csum[1] += p.y; csum[2] += p.z;
            ++nvalid;
        }
        if (nvalid > 0) {
            const double vinv = 1.0 / (double)nvalid;
            obj.centroid[0] = (float)(csum[0] * vinv);
            obj.centroid[1] = (float)(csum[1] * vinv);
            obj.centroid[2] = (float)(csum[2] * vinv);
        }
        obj.mean_feat = meanFeature(obj.points, pf);
    }

    if (changed) ++version_;
}

// Union of member points of objects whose centroid is within `radius` of `center`,
// appended to `out`. radius <= 0 => all objects. Feeds the whole-object feature call.
void ObjectSeeds::collectInRadiusPoints(const float center[3], float radius,
                                        std::vector<int>& out) const {
    const float r2 = radius > 0.0f ? radius * radius : 0.0f;
    for (const ObjectSeed& obj : list_) {
        if (radius > 0.0f && dist2(obj.centroid, center) > r2) continue;
        out.insert(out.end(), obj.points.begin(), obj.points.end());
    }
}
