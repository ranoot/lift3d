#include "objects.h"
#include "point_features.h"     // PointFeatures
#include "voxel_util.h"         // objutil::VKey/VHash/VSet, voxelOf, dist2, cosine, meanFeature

// The seeds -> objects consolidation stage (tier 2 -> tier 3). Parlay-free: reads only
// Universe / ObjectSeeds / PointFeatures + std, so it stays clear of the heavy
// hdbscan/parlay runtime and is unit-testable on its own. Mirrors object_grow.cpp's
// affinity-merge shape, but scores by cosine * ADJACENCY (dilated overlap) instead of
// containment, because disjoint seeds never share voxels with an existing object.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using objutil::VKey;
using objutil::VHash;
using objutil::VSet;
using objutil::voxelOf;
using objutil::dist2;
using objutil::cosine;
using objutil::meanFeature;

// Transient per-object cache for one consolidate call over the in-radius objects.
struct OCand {
    int                oid = -1;
    VSet               vox;                        // union of member-seed voxels (grows)
    std::vector<float> feat;                       // mean Mask3D feature
    float              centroid[3] = {0, 0, 0};
    float              lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};  // AABB
};

// Fraction of `svox` whose voxels lie within Chebyshev distance K of any voxel in
// `ovox` (K==0 => plain containment). Early-exits per seed voxel on first touch.
float adjacencyFrac(const VSet& svox, const VSet& ovox, int K) {
    if (svox.empty() || ovox.empty()) return 0.0f;
    int hit = 0;
    for (const VKey& v : svox) {
        bool touch = false;
        for (int dz = -K; dz <= K && !touch; ++dz)
            for (int dy = -K; dy <= K && !touch; ++dy)
                for (int dx = -K; dx <= K && !touch; ++dx)
                    if (ovox.count(VKey{v.x + dx, v.y + dy, v.z + dz})) touch = true;
        if (touch) ++hit;
    }
    return (float)hit / (float)svox.size();
}

}  // namespace

void Objects::consolidate(const Universe& uni, const ObjectSeeds& seeds,
                          const PointFeatures& pf, const Params& g,
                          const float center[3], float radius, float voxel) {
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;
    const std::vector<ObjectSeed>& slist = seeds.list();
    if (slist.empty()) return;

    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);
    const float r2  = radius > 0.0f ? radius * radius : 0.0f;
    const float cr2 = g.cand_radius * g.cand_radius;
    const int   K   = std::max(0, (int)std::lround(g.adj_dilate * inv));  // dilation in voxels

    // Voxel set of a seed's member points.
    auto seedVox = [&](const ObjectSeed& s, VSet& out) {
        out.reserve(s.points.size());
        for (int gi : s.points)
            if (gi >= 0 && gi < (int)cloud->size())
                out.insert(voxelOf((*cloud)[gi], inv));
    };
    // AABB of a seed's member points into c.lo/c.hi (returns false if it had none).
    auto seedAabb = [&](const ObjectSeed& s, OCand& c) {
        bool first = true;
        for (int gi : s.points) {
            if (gi < 0 || gi >= (int)cloud->size()) continue;
            const Universe::PointT& p = (*cloud)[gi];
            if (first) { c.lo[0]=c.hi[0]=p.x; c.lo[1]=c.hi[1]=p.y; c.lo[2]=c.hi[2]=p.z; first=false; }
            else {
                c.lo[0]=std::min(c.lo[0],p.x); c.hi[0]=std::max(c.hi[0],p.x);
                c.lo[1]=std::min(c.lo[1],p.y); c.hi[1]=std::max(c.hi[1],p.y);
                c.lo[2]=std::min(c.lo[2],p.z); c.hi[2]=std::max(c.hi[2],p.z);
            }
        }
        return !first;
    };

    // ---- cache every object with an in-radius centroid (mark it dirty so it re-derives
    // this pass -- its member seeds may have grown via superpoints since last call) -----
    std::vector<OCand>        cand;
    std::vector<int>          obj_to_cand((std::size_t)list_.size(), -1);
    std::vector<std::uint8_t> dirty((std::size_t)list_.size(), 0);
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        const Object& obj = list_[oid];
        if (radius > 0.0f && dist2(obj.centroid, center) > r2) continue;
        OCand c;
        c.oid = oid;
        c.centroid[0]=obj.centroid[0]; c.centroid[1]=obj.centroid[1]; c.centroid[2]=obj.centroid[2];
        bool have = false;
        for (int sid : obj.member_seeds) {
            if (sid < 0 || sid >= (int)slist.size()) continue;
            const ObjectSeed& s = slist[sid];
            for (int gi : s.points)
                if (gi >= 0 && gi < (int)cloud->size()) c.vox.insert(voxelOf((*cloud)[gi], inv));
            OCand box; if (seedAabb(s, box)) {
                if (!have) { std::copy(box.lo, box.lo+3, c.lo); std::copy(box.hi, box.hi+3, c.hi); }
                else {
                    for (int k=0;k<3;++k){ c.lo[k]=std::min(c.lo[k],box.lo[k]); c.hi[k]=std::max(c.hi[k],box.hi[k]); }
                }
                have = true;
            }
        }
        if (!have) continue;
        // Object mean feature from the (disjoint) union of member-seed points.
        std::vector<int> opts;
        for (int sid : obj.member_seeds)
            if (sid >= 0 && sid < (int)slist.size())
                opts.insert(opts.end(), slist[sid].points.begin(), slist[sid].points.end());
        c.feat = meanFeature(opts, pf);
        obj_to_cand[oid] = (int)cand.size();
        dirty[oid] = 1;                                  // re-derive in-radius objects
        cand.push_back(std::move(c));
    }

    // ---- assign each still-unassigned, in-radius seed to the best object or a new one --
    bool changed = false;
    for (int sid = 0; sid < (int)slist.size(); ++sid) {
        if (seedOwner(sid) != -1) continue;              // already owned (monotonic)
        const ObjectSeed& s = slist[sid];
        if (radius > 0.0f && dist2(s.centroid, center) > r2) continue;

        VSet svox; seedVox(s, svox);
        if (svox.empty()) continue;
        std::vector<float> sf = meanFeature(s.points, pf);
        if (sf.empty()) continue;   // no feature yet -> defer (leave unassigned this pass)

        int   best_oid = -1;
        float best_cos = g.affinity_thresh;              // feature-similarity threshold
        for (const OCand& c : cand) {
            if (g.require_class && s.class_id >= 0 && s.class_id != list_[c.oid].class_id)
                continue;
            const bool near = dist2(s.centroid, c.centroid) <= cr2;
            const bool in_box =
                s.centroid[0] >= c.lo[0]-g.adj_dilate && s.centroid[0] <= c.hi[0]+g.adj_dilate &&
                s.centroid[1] >= c.lo[1]-g.adj_dilate && s.centroid[1] <= c.hi[1]+g.adj_dilate &&
                s.centroid[2] >= c.lo[2]-g.adj_dilate && s.centroid[2] <= c.hi[2]+g.adj_dilate;
            if (!near && !in_box) continue;
            // Geometric GATE: the seed must physically touch the object -- a fraction of
            // its voxels within K of the object's. Adjacency is a boundary quantity (small
            // for two comparable fragments), so it only ADMITS a candidate; it must not be
            // multiplied into the score or the merge becomes unreachable.
            if (adjacencyFrac(svox, c.vox, K) < g.adj_min) continue;
            // Feature SCORE: among touching, class-matched objects pick the most similar.
            const float cs = cosine(sf, c.feat);
            if (cs >= best_cos) { best_cos = cs; best_oid = c.oid; }
        }

        if (best_oid >= 0) {
            // Merge the seed into the winning object; grow that cand's voxels so later
            // seeds this pass see the enlarged object.
            assignSeed(sid, best_oid);
            list_[best_oid].member_seeds.push_back(sid);
            OCand& c = cand[obj_to_cand[best_oid]];
            for (const VKey& v : svox) c.vox.insert(v);
            dirty[best_oid] = 1;
            changed = true;
        } else {
            // Spawn a new object seeded by this one seed; register its cache so later
            // seeds in this same pass can merge into it.
            const int oid = (int)list_.size();
            Object obj;
            obj.id = oid; obj.class_id = s.class_id; obj.kind = s.kind;
            obj.member_seeds.push_back(sid);
            list_.push_back(std::move(obj));
            assignSeed(sid, oid);

            OCand c; c.oid = oid; c.vox = svox; c.feat = std::move(sf);
            c.centroid[0]=s.centroid[0]; c.centroid[1]=s.centroid[1]; c.centroid[2]=s.centroid[2];
            seedAabb(s, c);
            obj_to_cand.push_back((int)cand.size());     // keep parallel to list_
            dirty.push_back(1);
            cand.push_back(std::move(c));
            changed = true;
        }
    }

    // ---- re-derive points / mean feature / centroid for every dirtied object ----------
    for (int oid = 0; oid < (int)list_.size(); ++oid) {
        if (!dirty[oid]) continue;
        Object& obj = list_[oid];
        std::vector<int> pts;
        for (int sid : obj.member_seeds)
            if (sid >= 0 && sid < (int)slist.size())
                pts.insert(pts.end(), slist[sid].points.begin(), slist[sid].points.end());
        if (pts.size() != obj.points.size()) changed = true;   // seeds grew since last pass
        obj.points = std::move(pts);

        double csum[3] = {0, 0, 0};
        int nvalid = 0;
        for (int gi : obj.points) {
            if (gi < 0 || gi >= (int)cloud->size()) continue;
            const Universe::PointT& p = (*cloud)[gi];
            csum[0]+=p.x; csum[1]+=p.y; csum[2]+=p.z; ++nvalid;
        }
        if (nvalid > 0) {
            const double vinv = 1.0 / (double)nvalid;
            obj.centroid[0]=(float)(csum[0]*vinv); obj.centroid[1]=(float)(csum[1]*vinv);
            obj.centroid[2]=(float)(csum[2]*vinv);
        }
        obj.mean_feat = meanFeature(obj.points, pf);
    }

    if (changed) ++version_;
}
