#include "object_seeds.h"
#include "hdbscan_extract.h"
#include "voxel_util.h"          // objutil::VKey/VHash, voxelOf (coarse-cell maturity gate)

// The ONLY translation unit that pulls in the hdbscan / parlay parallel runtime, so
// the heavy template machinery stays out of everything that includes object_seeds.h.
#include "parlay/sequence.h"
#include "hdbscan/point.h"
#include "hdbscan/hdbscan.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <streambuf>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {
// The vendored hdbscan lib prints timing/diagnostics ("wspd-time", "kruskal-time",
// "dendrogram-time", beta/rho/edges, ...) straight to std::cout on every hdbscan()
// / dendrogram() call -- one burst per thing class per refresh. Redirect std::cout
// to a black hole for the clustering scope so it doesn't drown the run's own output.
struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }        // discard everything
};
struct CoutSilencer {
    NullBuf nb;
    std::streambuf* old;
    CoutSilencer() : old(std::cout.rdbuf(&nb)) {}
    ~CoutSilencer() { std::cout.rdbuf(old); }
};
} // namespace

// Run LOCAL-window HDBSCAN* per thing class over the still-UNCLAIMED, MATURED points.
// We crop the world to the robot window, gate points on coarse-cell maturity, gather
// the unclaimed points' universe global indices, cluster their xyz with hdbscan<3> +
// dendrogram, extract strict flat clusters, and APPEND one new ObjectSeed per surviving
// cluster (claiming its points). Objects are persistent: existing ones are never touched
// or wiped, so seeding only ever adds instances.
void ObjectSeeds::seedLocal(const Universe& uni, const Params& p,
                            const float center[3], float radius) {
    ++version_;
    if (uni.size() == 0) return;
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return;

    // Crop the world to the local window around the robot. HDBSCAN runs on this window
    // only -- never the whole map (radius <= 0 falls back to the whole world).
    std::vector<int> win;
    uni.local(center, radius, &win);
    if (win.empty()) return;

    // Maturity gate: histogram the window's points into coarse (~2 m^3) cells; a cell is
    // clusterable only once it holds >= maturity_min occupied fine voxels (map points).
    // Pure geometric density -- keeps HDBSCAN off sparse, half-observed geometry.
    const float cinv = 1.0f / (p.maturity_res > 0.0f ? p.maturity_res : 1.0f);
    std::unordered_map<objutil::VKey, int, objutil::VHash> cell_count;
    cell_count.reserve(win.size());
    for (int g : win) cell_count[objutil::voxelOf((*cloud)[g], cinv)]++;
    const int mat_min = std::max(1, p.maturity_min);

    // Bucket every UNCLAIMED thing-labeled point in a MATURED cell by its resolved class
    // id. Points already owned by an object are excluded, so grown objects are not
    // re-clustered and the input shrinks as the map fills in.
    std::map<int, std::vector<int>> by_class;
    for (int i : win) {
        if (!uni.pointIsThing(i) || claimed(i)) continue;
        const int cid = uni.pointClassId(i);
        if (cid < 0) continue;
        auto it = cell_count.find(objutil::voxelOf((*cloud)[i], cinv));
        if (it == cell_count.end() || it->second < mat_min) continue;   // immature cell
        by_class[cid].push_back(i);
    }

    const int min_pts = std::max(1, p.min_pts);
    CoutSilencer silence;                              // mute the lib's per-call cout spam
    for (const auto& kv : by_class) {
        const int                cid  = kv.first;
        const std::vector<int>&  gidx = kv.second;
        const int                n    = (int)gidx.size();
        // Strict pre-gates: enough points overall, and more than the density k.
        if (n < p.min_class_points || n <= min_pts) continue;

        // Build the xyz point set (hdbscan<3> takes a mutable sequence of doubles).
        parlay::sequence<pargeo::point<3>> P(n);
        for (int j = 0; j < n; ++j) {
            const Universe::PointT& pt = (*cloud)[gidx[j]];
            double xyz[3] = {pt.x, pt.y, pt.z};
            P[j] = pargeo::point<3>(xyz);
        }

        parlay::sequence<pargeo::wghEdge> E = pargeo::hdbscan<3>(P, (size_t)min_pts);
        parlay::sequence<pargeo::dendroNode> dendro = pargeo::dendrogram(E, (size_t)n);

        std::vector<LinkNode> linkage(dendro.size());
        for (std::size_t r = 0; r < dendro.size(); ++r)
            linkage[r] = LinkNode{(long)std::get<0>(dendro[r]), (long)std::get<1>(dendro[r]),
                                  std::get<2>(dendro[r]),        (long)std::get<3>(dendro[r])};

        std::vector<int> labels = extractFlatClusters(linkage, n, p.min_cluster_size,
                                                       p.allow_single_cluster);

        // Collect points per flat cluster label (>=0); -1 == noise, dropped.
        int k = 0;
        for (int l : labels) k = std::max(k, l + 1);
        if (k == 0) continue;
        std::vector<ObjectSeed> seeds((std::size_t)k);
        for (auto& s : seeds) { s.class_id = cid; s.kind = ClassKind::Thing; }
        for (int j = 0; j < n; ++j) {
            const int l = labels[j];
            if (l >= 0) seeds[(std::size_t)l].points.push_back(gidx[j]);
        }
        for (ObjectSeed& s : seeds) {
            if (s.points.empty()) continue;        // (shouldn't happen, but be safe)
            double c[3] = {0, 0, 0};
            for (int g : s.points) {
                const Universe::PointT& pt = (*cloud)[g];
                c[0] += pt.x; c[1] += pt.y; c[2] += pt.z;
            }
            const double inv = 1.0 / (double)s.points.size();
            s.centroid[0] = (float)(c[0] * inv);
            s.centroid[1] = (float)(c[1] * inv);
            s.centroid[2] = (float)(c[2] * inv);
            // Give the new object a stable id (= its slot) and claim its points, so a
            // later seed pass skips them and growing knows who owns them.
            s.id = (int)list_.size();
            for (int g : s.points) claim(g, s.id);
            list_.push_back(std::move(s));
        }
    }
}
