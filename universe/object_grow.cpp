#include "object_seeds.h"
#include "superpoints.h"        // Superpoint, Superpoints
#include "voxel_util.h"         // objutil::VKey/VHash/VSet, voxelOf, dist2, cosine

// The (parlay-free) growing half of the object store: absorb CLASS-consistent,
// spatially-overlapping superpoints into nearby objects. Affinity is SAI3D region affinity
// -- cosine of the per-thing-class histograms. Kept OUT of object_seeds.cpp so it stays free
// of the heavy hdbscan/parlay includes and is unit-testable on its own (it reads only
// Universe / Superpoints + std).

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <vector>

namespace {

using objutil::VKey;
using objutil::VHash;
using objutil::VSet;
using objutil::voxelOf;
using objutil::dist2;
using objutil::cosine;

// Thing-class histogram over the members of `pts`: scatter each member's resolved thing
// class into a dense vector keyed on the registry class id (size == semantics().size();
// stuff/unlabeled contribute nothing). Mirrors how superpoints/seeds build their hist, so
// the two sides of the grow cosine are indexed identically.
std::vector<float> histOf(const std::vector<int>& pts, const Universe& uni) {
    std::vector<float> h((std::size_t)uni.semantics().size(), 0.0f);
    for (int g : pts) {
        const int cid = uni.pointClassId(g);
        if (cid >= 0 && cid < (int)h.size() && uni.pointClassKind(g) == ClassKind::Thing)
            h[(std::size_t)cid] += 1.0f;
    }
    return h;
}

// Transient per-object cache built for one growLocal call over the in-radius objects.
struct Cand {
    int                oid = -1;
    VSet               vox;                        // member voxels (grows as we merge)
    std::vector<float> hist;                       // class histogram (SAI3D affinity)
    float              centroid[3] = {0, 0, 0};
    float              lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};  // AABB
};

}  // namespace

void ObjectSeeds::growLocal(const Universe& uni, const Superpoints& sp,
                            const GrowParams& g, float voxel) {
    if (list_.empty() || sp.size() == 0) return;
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;
    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);

    // ---- build a cache for every proposal (list_ is this frame's ephemeral set) -----
    // Every proposal is a candidate; its class histogram is the object side of the affinity
    // cosine (a per-class HDBSCAN seed is a single spike, re-aggregated as it grows).
    std::vector<Cand> cand;
    std::vector<int>  obj_to_cand((std::size_t)list_.size(), -1);
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        const ObjectSeed& obj = list_[oid];
        Cand c;
        c.oid = oid;
        c.hist = obj.hist;
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

    // ---- optional: neighborhood-pooled superpoint histograms (SAI3D judge_connect) ------
    // Smooth each superpoint's class histogram with its (decayed) KNN neighbors' histograms,
    // so a boundary superpoint's affinity is decided by its neighborhood, not itself alone.
    // O(S^2) over the ephemeral, small superpoint set -- cheap, no KD-tree dependency.
    const std::vector<Superpoint>& spl = sp.list();
    const int S = (int)spl.size();
    std::vector<std::vector<float>> pooled;
    if (g.neighbor_pool) {
        pooled.assign((std::size_t)S, {});
        std::vector<std::pair<float, int>> d;
        for (int i = 0; i < S; ++i) {
            if (spl[i].hist.empty()) continue;
            std::vector<float> h = spl[i].hist;            // own histogram (weight 1)
            d.clear();
            for (int j = 0; j < S; ++j)
                if (j != i && spl[j].hist.size() == h.size())
                    d.emplace_back(dist2(spl[i].centroid, spl[j].centroid), j);
            const int k = std::min((int)d.size(), std::max(0, g.pool_k));
            std::partial_sort(d.begin(), d.begin() + k, d.end());
            for (int t = 0; t < k; ++t) {
                const std::vector<float>& nh = spl[d[t].second].hist;
                for (std::size_t c = 0; c < h.size(); ++c) h[c] += g.dis_decay * nh[c];
            }
            pooled[i] = std::move(h);
        }
    }

    // ---- evaluate every superpoint against the candidate objects ------------------
    // `sweeps` > 1 re-runs the assignment: objects grow their voxel volume as superpoints are
    // annexed, so a superpoint decided (or skipped) before its eventual object grew gets a
    // second chance -- SAI3D's iterative region growing (bounded, order-insensitive).
    std::vector<std::uint8_t> dirty((std::size_t)list_.size(), 0);
    bool changed = false;
    const int sweeps = std::max(1, g.sweeps);
    // overlap_sets: a superpoint is annexed into a given proposal at most once (its whole
    // point set is added), so repeated sweeps never re-add it. Indexed by candidate index.
    std::vector<std::unordered_set<int>> annexed_sp;
    if (g.overlap_sets) annexed_sp.assign(cand.size(), {});
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        bool sweep_changed = false;
        for (int si = 0; si < S; ++si) {
            const Superpoint& s = spl[si];
            // Only DOMINANT-THING superpoints grow (class_id >= 0 == the thing cleared
            // thing_frac over the whole blob). This is what keeps wall/floor-dominated
            // superpoints -- with a near-empty thing histogram -- out of any object.
            if (s.class_id < 0 || s.hist.empty()) continue;
            const std::vector<float>& shist =
                (g.neighbor_pool && !pooled[si].empty()) ? pooled[si] : s.hist;

            // Superpoint voxel set (its volume unit) + any-unclaimed check. In the exclusive
            // path a fully-claimed superpoint is skipped cheaply on later sweeps; under
            // overlap_sets there is no claiming, so that skip is disabled.
            VSet svox;
            svox.reserve(s.points.size());
            bool any_free = false;
            for (int gi : s.points) {
                if (gi < 0 || gi >= (int)cloud->size()) continue;
                svox.insert(voxelOf((*cloud)[gi], inv));
                if (owner(gi) == -1) any_free = true;
            }
            if (svox.empty()) continue;
            if (!g.overlap_sets && !any_free) continue;
            const float inv_svol = 1.0f / (float)svox.size();

            // SAI3D region affinity of this superpoint to candidate `ci` (cand index):
            // cosine(pooled class hists) * volume containment, or -1 if class/AABB/overlap fail.
            auto affOf = [&](int ci) -> float {
                const Cand& c = cand[ci];
                if (g.require_class && s.class_id >= 0 && s.class_id != list_[c.oid].class_id)
                    return -1.0f;
                // Cheap gate: sp centroid must lie within the object's AABB (+one voxel).
                const bool in_box =
                    s.centroid[0] >= c.lo[0] - voxel && s.centroid[0] <= c.hi[0] + voxel &&
                    s.centroid[1] >= c.lo[1] - voxel && s.centroid[1] <= c.hi[1] + voxel &&
                    s.centroid[2] >= c.lo[2] - voxel && s.centroid[2] <= c.hi[2] + voxel;
                if (!in_box) return -1.0f;
                int shared = 0;
                for (const VKey& v : svox) if (c.vox.count(v)) ++shared;
                if (shared == 0) return -1.0f;
                return cosine(shist, c.hist) * ((float)shared * inv_svol);
            };

            // Annex superpoint `si` into candidate `ci`. Exclusive: only its UNCLAIMED points,
            // claimed first-come. overlap_sets: ALL its points, added to this proposal once
            // (annexed_sp blocks a re-add across sweeps); points shared with the proposal's own
            // seed / other superpoints are de-duplicated in the recompute pass below.
            auto annexInto = [&](int ci) {
                if (g.overlap_sets && !annexed_sp[ci].insert(si).second) return;
                Cand&       c   = cand[ci];
                ObjectSeed& obj = list_[c.oid];
                bool        annexed = false;
                for (int gi : s.points) {
                    if (!g.overlap_sets) {
                        if (owner(gi) != -1) continue;         // claimed (first-claim wins)
                        claim(gi, c.oid);
                    }
                    obj.points.push_back(gi);
                    if (gi >= 0 && gi < (int)cloud->size()) {
                        const Universe::PointT& p = (*cloud)[gi];
                        c.vox.insert(voxelOf(p, inv));         // grow the volume for later sps
                        c.lo[0] = std::min(c.lo[0], p.x); c.hi[0] = std::max(c.hi[0], p.x);
                        c.lo[1] = std::min(c.lo[1], p.y); c.hi[1] = std::max(c.hi[1], p.y);
                        c.lo[2] = std::min(c.lo[2], p.z); c.hi[2] = std::max(c.hi[2], p.z);
                    }
                    annexed = true;
                }
                if (annexed) {
                    obj.inst_ids.insert(obj.inst_ids.end(), s.inst_ids.begin(), s.inst_ids.end());
                    dirty[c.oid] = 1;
                    changed = true;
                    sweep_changed = true;
                }
            };

            if (g.overlap_sets) {
                // Annex into EVERY candidate whose affinity clears the bar (overlap allowed).
                for (int ci = 0; ci < (int)cand.size(); ++ci)
                    if (affOf(ci) >= g.affinity_thresh) annexInto(ci);
            } else {
                // Exclusive: only the single best-affinity candidate above the bar.
                int   best_ci  = -1;
                float best_aff = g.affinity_thresh;        // must strictly reach threshold
                for (int ci = 0; ci < (int)cand.size(); ++ci) {
                    const float aff = affOf(ci);
                    if (aff >= best_aff) { best_aff = aff; best_ci = ci; }
                }
                if (best_ci >= 0) annexInto(best_ci);
            }
        }
        if (!sweep_changed) break;                        // converged: no more annexations
    }

    // ---- recompute centroid + mean feature for every grown object -----------------
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        if (!dirty[oid]) continue;
        ObjectSeed& obj = list_[oid];
        // overlap_sets: an annexed superpoint's points may coincide with the proposal's own
        // seed (or another annexed superpoint) -- dedup so centroid/hist don't double-count.
        if (g.overlap_sets) {
            std::sort(obj.points.begin(), obj.points.end());
            obj.points.erase(std::unique(obj.points.begin(), obj.points.end()), obj.points.end());
        }
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
        // Re-aggregate the class histogram over the grown membership (absorbed superpoints
        // may have added other thing classes at the object's boundary).
        obj.hist = histOf(obj.points, uni);
        std::sort(obj.inst_ids.begin(), obj.inst_ids.end());
        obj.inst_ids.erase(std::unique(obj.inst_ids.begin(), obj.inst_ids.end()),
                           obj.inst_ids.end());
    }

    if (changed) ++version_;
}
