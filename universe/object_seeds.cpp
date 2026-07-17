#include "object_seeds.h"
#include "hdbscan_extract.h"
#include "voxel_util.h"          // objutil::VKey/VHash, voxelOf (coarse-cell maturity gate)

// The ONLY translation unit that pulls in the hdbscan / parlay parallel runtime, so
// the heavy template machinery stays out of everything that includes object_seeds.h.
#include "parlay/sequence.h"
#include "hdbscan/point.h"
#include "hdbscan/hdbscan.h"

#include <pcl/filters/statistical_outlier_removal.h>   // SOR pre-filter before HDBSCAN
#include <pcl/search/kdtree.h>                          // radius search for Euclidean split
#include <functional>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <streambuf>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {
// Opt-in HDBSCAN input/output dump: set LIFT3D_HDB_DEBUG=1 to print, per per-class
// clustering call, the exact point set fed to hdbscan<3> (count, xyz bounding box) and
// the flat-cluster result (k clusters + noise count). Zero cost when the env var is unset.
// This is the "what is HDBSCAN actually running on" probe -- the points are raw fused
// voxel-map xyz, never splatted; splat/surf_band only decide which real points are in `mg`.
inline bool hdbDebugOn() {
    static const bool on = std::getenv("LIFT3D_HDB_DEBUG") != nullptr;
    return on;
}

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

// The vendored pargeo::hdbscan MST/WSPD recursion has a base-case bug that reads out of
// bounds (SIGSEGV) on small inputs: empirically it crashes for n <= 11 (all geometries:
// cube / planar / collinear / duplicated, independent of minPts) and is always safe for
// n >= 12. We floor every hdbscan call at this size (margin above 12). Costs nothing: a
// surviving cluster needs >= min_cluster_size points anyway, so a sub-floor input can
// never yield one. See the standalone sweep in the segfault investigation.
constexpr int kHdbscanMinInput = 16;

// Statistical Outlier Removal over `gidx` (universe global indices into `cloud`). For each
// point computes the mean distance to its `mean_k` nearest neighbors and drops any point
// whose mean exceeds mu + std_mul*sigma (global mean/stddev) -- pruning the sparse, low-
// density points that HDBSCAN's MST would otherwise use to bridge distinct objects. Returns
// the SURVIVING global indices (input order among survivors preserved); accumulates its wall
// time into `acc_ms`. A no-op returning `gidx` unchanged when there are <= mean_k points
// (SOR needs a full neighborhood to estimate density). Callers gate on sor_enabled.
std::vector<int> sorFilterIndices(const Universe::Cloud& cloud, const std::vector<int>& gidx,
                                  int mean_k, float std_mul, double& acc_ms) {
    if (mean_k < 1) mean_k = 1;
    if ((int)gidx.size() <= mean_k) return gidx;

    const auto t0 = std::chrono::steady_clock::now();

    // Build a compact subcloud of just these points; SOR indices map 1:1 back to gidx.
    pcl::PointCloud<Universe::PointT>::Ptr sub(new pcl::PointCloud<Universe::PointT>);
    sub->reserve(gidx.size());
    for (int g : gidx) {
        if (g < 0 || g >= (int)cloud.size()) { sub->push_back(Universe::PointT()); continue; }
        sub->push_back(cloud[g]);
    }

    std::vector<int> kept_idx;                          // indices into `sub` that survive
    pcl::StatisticalOutlierRemoval<Universe::PointT> sor;
    sor.setInputCloud(sub);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh((double)std_mul);
    sor.filter(kept_idx);                               // default keeps inliers (dense points)

    std::vector<int> out;
    out.reserve(kept_idx.size());
    for (int i : kept_idx)
        if (i >= 0 && i < (int)gidx.size()) out.push_back(gidx[(std::size_t)i]);

    const auto t1 = std::chrono::steady_clock::now();
    acc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

// Split a set of global point indices into spatially-CONNECTED components at `radius` (m).
// HDBSCAN* returns MST-connected clusters that can still span a physical gap (a small
// sub-min_cluster_size tail glued to a dense blob across empty space never splits off in the
// condensed tree), so one flat label can cover two visibly separate blobs. A Euclidean
// connected-components pass at `radius` guarantees each emitted seed is spatially contiguous:
// a solid surface (neighbors within a few voxels) stays one component; two blobs separated by
// more than `radius` become two. radius <= 0 or <2 points => the input unchanged (one comp).
std::vector<std::vector<int>> euclideanSplit(const Universe::Cloud& cloud,
                                             const std::vector<int>& idx, float radius) {
    const int m = (int)idx.size();
    if (radius <= 0.0f || m <= 1) return {idx};

    pcl::PointCloud<Universe::PointT>::Ptr sub(new pcl::PointCloud<Universe::PointT>);
    sub->reserve(idx.size());
    for (int g : idx) sub->push_back(cloud[g]);

    pcl::search::KdTree<Universe::PointT> tree;
    tree.setInputCloud(sub);

    std::vector<int> parent(m);
    for (int i = 0; i < m; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };

    std::vector<int>   nbr;
    std::vector<float> d2;
    for (int i = 0; i < m; ++i) {
        nbr.clear(); d2.clear();
        tree.radiusSearch((*sub)[i], radius, nbr, d2);
        for (int nb : nbr) { int ra = find(i), rb = find(nb); if (ra != rb) parent[ra] = rb; }
    }

    std::unordered_map<int, std::vector<int>> comps;
    for (int i = 0; i < m; ++i) comps[find(i)].push_back(idx[(std::size_t)i]);
    std::vector<std::vector<int>> out;
    out.reserve(comps.size());
    for (auto& kv : comps) out.push_back(std::move(kv.second));
    return out;
}
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
        // Strict pre-gates: enough points overall, more than the density k, and above the
        // hdbscan small-input crash floor.
        if (n < p.min_class_points || n <= min_pts || n < kHdbscanMinInput) continue;

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
                                                       p.allow_single_cluster, p.leaf_selection);

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

// PER-VIEW seeding over an explicit thing-point set (no crop, no maturity gate).
void ObjectSeeds::seedFromIndices(const Universe& uni, const std::vector<int>& gidx,
                                  const Params& p) {
    ++version_;
    last_sor_ms_ = 0.0;                                 // reset per-frame SOR timing accumulator
    // Ephemeral per-frame proposals: wipe last frame's clusters + claims so this frame
    // re-clusters the current visible set from scratch. Proposals do NOT persist or claim
    // across frames -- the same physical object is re-proposed every frame with overlapping
    // voxels, and that overlap is exactly what tier-3 IoU consolidation feeds on.
    list_.clear();
    std::fill(owner_.begin(), owner_.end(), -1);
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty() || gidx.empty()) return;

    // Bucket the visible thing points by resolved class (input is already visible ∩ thing;
    // `claimed` here is a WITHIN-FRAME guard only, since owner_ was just reset above).
    // Under overlap_sets the guard is dropped so a point may seed more than one cluster.
    std::map<int, std::vector<int>> by_class;
    for (int g : gidx) {
        if (g < 0 || g >= (int)cloud->size()) continue;
        if (!p.overlap_sets && claimed(g)) continue;
        const int cid = uni.pointClassId(g);
        if (cid < 0 || uni.pointClassKind(g) != ClassKind::Thing) continue;
        by_class[cid].push_back(g);
    }
    if (by_class.empty()) return;

    const int min_pts = std::max(1, p.min_pts);

    // Append one new seed from a uniform-class member list: centroid + deduped instance
    // ids from the members, claim the points, assign a stable id.
    auto appendSeed = [&](int cid, const std::vector<int>& members) {
        if (members.empty()) return;
        ObjectSeed s;
        s.class_id = cid;
        s.kind = ClassKind::Thing;
        double c[3] = {0, 0, 0};
        std::set<int> iids;
        for (int g : members) {
            const Universe::PointT& pt = (*cloud)[g];
            c[0] += pt.x; c[1] += pt.y; c[2] += pt.z;
            const int iid = uni.pointInstanceId(g);
            if (iid >= 0) iids.insert(iid);
        }
        const double invn = 1.0 / (double)members.size();
        s.centroid[0] = (float)(c[0] * invn);
        s.centroid[1] = (float)(c[1] * invn);
        s.centroid[2] = (float)(c[2] * invn);
        s.inst_ids.assign(iids.begin(), iids.end());
        // SAI3D affinity histogram: a per-class proposal is a single spike at its class id
        // (weight = member count). growLocal re-aggregates it as superpoints are absorbed.
        s.hist.assign((std::size_t)uni.semantics().size(), 0.0f);
        if (cid >= 0 && cid < uni.semantics().size())
            s.hist[(std::size_t)cid] = (float)members.size();
        s.points = members;
        s.id = (int)list_.size();
        for (int g : members) claim(g, s.id);
        list_.push_back(std::move(s));
    };

    // Turn a hdbscan flat-cluster labeling into linkage + extract, grouping members.
    auto linkageOf = [](const parlay::sequence<pargeo::dendroNode>& dendro) {
        std::vector<LinkNode> linkage(dendro.size());
        for (std::size_t r = 0; r < dendro.size(); ++r)
            linkage[r] = LinkNode{(long)std::get<0>(dendro[r]), (long)std::get<1>(dendro[r]),
                                  std::get<2>(dendro[r]),        (long)std::get<3>(dendro[r])};
        return linkage;
    };

    // Geometric HDBSCAN* over one class's member points: cluster xyz, extract flat clusters,
    // and append one seed per surviving cluster. The proven per-class path, shared by the
    // pure-geometry fallback loop and the instance-guided no-instance remainder.
    auto clusterClass = [&](int cid, const std::vector<int>& mg_in) {
        // SOR pre-filter: drop sparse bridge points so HDBSCAN's MST can't fuse distinct
        // objects. No-op (returns mg_in) when disabled or too few points.
        const std::vector<int> mg = p.sor_enabled
            ? sorFilterIndices(*cloud, mg_in, p.sor_mean_k, p.sor_std_mul, last_sor_ms_)
            : mg_in;
        const int n = (int)mg.size();
        if (n < p.min_class_points || n <= min_pts || n < kHdbscanMinInput) return;
        parlay::sequence<pargeo::point<3>> P(n);
        for (int j = 0; j < n; ++j) {
            const Universe::PointT& pt = (*cloud)[mg[j]];
            double xyz[3] = {pt.x, pt.y, pt.z};
            P[j] = pargeo::point<3>(xyz);
        }
        parlay::sequence<pargeo::wghEdge> E = pargeo::hdbscan<3>(P, (size_t)min_pts);
        parlay::sequence<pargeo::dendroNode> dendro = pargeo::dendrogram(E, (size_t)n);
        std::vector<LinkNode> linkage = linkageOf(dendro);
        std::vector<int> labels = extractFlatClusters(linkage, n, p.min_cluster_size,
                                                      p.allow_single_cluster, p.leaf_selection);
        int k = 0;
        for (int l : labels) k = std::max(k, l + 1);
        if (k == 0) {
            if (hdbDebugOn())
                std::fprintf(stderr, "[hdb] class=%d in=%d sor=%d clustered=%d -> clusters=0 "
                             "noise=%d (all noise)\n",
                             cid, (int)mg_in.size(), (int)mg.size(), n, n);
            return;
        }
        std::vector<std::vector<int>> members((std::size_t)k);
        for (int j = 0; j < n; ++j) {
            const int l = labels[j];
            if (l >= 0) members[(std::size_t)l].push_back(mg[j]);
        }
        // Enforce spatial contiguity: split each HDBSCAN flat cluster into Euclidean-connected
        // components so no seed spans a physical gap. One entry per FINAL seed.
        std::vector<std::vector<int>> seeds_out;
        for (int l = 0; l < k; ++l) {
            if (members[(std::size_t)l].empty()) continue;
            for (auto& comp : euclideanSplit(*cloud, members[(std::size_t)l], p.split_radius))
                if (!comp.empty()) seeds_out.push_back(std::move(comp));
        }
        if (hdbDebugOn()) {
            // Per-FINAL-seed extent (post connected-components split). If seeds are still
            // OVERSIZE here, the blobs are genuinely linked within split_radius (lower it or
            // look upstream at label smear); otherwise the split already broke the bridge.
            int noise = 0; for (int l : labels) if (l < 0) ++noise;
            std::fprintf(stderr,
                "[hdb] class=%d in=%d sor=%d clustered=%d -> hdb_clusters=%d seeds=%d noise=%d\n",
                cid, (int)mg_in.size(), (int)mg.size(), n, k, (int)seeds_out.size(), noise);
            for (std::size_t s = 0; s < seeds_out.size(); ++s) {
                const std::vector<int>& M = seeds_out[s];
                float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
                for (int gi : M) {
                    const Universe::PointT& pt = (*cloud)[gi];
                    lo[0] = std::min(lo[0], pt.x); hi[0] = std::max(hi[0], pt.x);
                    lo[1] = std::min(lo[1], pt.y); hi[1] = std::max(hi[1], pt.y);
                    lo[2] = std::min(lo[2], pt.z); hi[2] = std::max(hi[2], pt.z);
                }
                const float dx = hi[0]-lo[0], dy = hi[1]-lo[1], dz = hi[2]-lo[2];
                const float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
                std::fprintf(stderr,
                    "        seed %zu: n=%d span=(%.2f %.2f %.2f)m diag=%.2fm%s\n",
                    s, (int)M.size(), dx, dy, dz, diag,
                    diag > 1.5f ? "  <-- OVERSIZE (blobs linked within split_radius)" : "");
            }
        }
        for (auto& comp : seeds_out) appendSeed(cid, comp);
    };

    CoutSilencer silence;                              // mute the lib's per-call cout spam

    if (p.single_scan) {
        // ONE hdbscan<4>: xyz + (class_slot * CLASS_GAP) as the 4th coordinate, so points
        // of different classes are astronomically far apart and never join a cluster.
        // Classes below min_class_points are dropped (too sparse to trust).
        std::vector<int> all_g, all_cid;
        std::map<int, int> cid_slot;
        int next_slot = 0;
        for (const auto& kv : by_class) {
            // Per-class SOR on real xyz BEFORE the class-gap 4th coord is added, so density
            // is estimated within each class's geometry (no-op when disabled).
            const std::vector<int> mg = p.sor_enabled
                ? sorFilterIndices(*cloud, kv.second, p.sor_mean_k, p.sor_std_mul, last_sor_ms_)
                : kv.second;
            if ((int)mg.size() < p.min_class_points) continue;
            cid_slot[kv.first] = next_slot++;
            for (int g : mg) { all_g.push_back(g); all_cid.push_back(kv.first); }
        }
        const int n = (int)all_g.size();
        if (n <= min_pts || n < kHdbscanMinInput) return;

        constexpr double CLASS_GAP = 1.0e6;            // >> any spatial extent
        parlay::sequence<pargeo::point<4>> P(n);
        for (int j = 0; j < n; ++j) {
            const Universe::PointT& pt = (*cloud)[all_g[j]];
            double xyzc[4] = {pt.x, pt.y, pt.z, (double)cid_slot[all_cid[j]] * CLASS_GAP};
            P[j] = pargeo::point<4>(xyzc);
        }
        parlay::sequence<pargeo::wghEdge> E = pargeo::hdbscan<4>(P, (size_t)min_pts);
        parlay::sequence<pargeo::dendroNode> dendro = pargeo::dendrogram(E, (size_t)n);
        std::vector<LinkNode> linkage = linkageOf(dendro);
        std::vector<int> labels = extractFlatClusters(linkage, n, p.min_cluster_size,
                                                      p.allow_single_cluster, p.leaf_selection);
        int k = 0;
        for (int l : labels) k = std::max(k, l + 1);
        if (k == 0) return;
        std::vector<std::vector<int>> members((std::size_t)k);
        std::vector<int> mem_cid((std::size_t)k, -1);
        for (int j = 0; j < n; ++j) {
            const int l = labels[j];
            if (l >= 0) { members[(std::size_t)l].push_back(all_g[j]); mem_cid[(std::size_t)l] = all_cid[j]; }
        }
        for (int l = 0; l < k; ++l) {
            if (members[(std::size_t)l].empty()) continue;
            for (auto& comp : euclideanSplit(*cloud, members[(std::size_t)l], p.split_radius))
                if (!comp.empty()) appendSeed(mem_cid[(std::size_t)l], comp);
        }
        return;
    }

    // Fallback: one hdbscan<3> per class (the proven path, minus the maturity gate).
    for (const auto& kv : by_class)
        clusterClass(kv.first, kv.second);
}

double ObjectSeeds::hdbscanTimeOnly(const Universe& uni, const std::vector<int>& gidx,
                                    const Params& p, int* out_clusters) {
    if (out_clusters) *out_clusters = 0;
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud || cloud->empty()) return 0.0;
    const int min_pts = std::max(1, p.min_pts);

    // Match the seeding path: SOR pre-filter (parity so the experiment measures the same
    // input HDBSCAN actually clusters). `dummy_sor` discards the SOR sub-timing here.
    double dummy_sor = 0.0;
    std::vector<int> gidx_sor;
    if (p.sor_enabled) gidx_sor = sorFilterIndices(*cloud, gidx, p.sor_mean_k, p.sor_std_mul, dummy_sor);
    const std::vector<int>& gidx_f = p.sor_enabled ? gidx_sor : gidx;
    const int n = (int)gidx_f.size();
    if (n <= min_pts || n < kHdbscanMinInput) return 0.0;   // too few points / hdbscan crash floor

    // Build the xyz point set (hdbscan<3> takes a mutable sequence of doubles).
    parlay::sequence<pargeo::point<3>> P(n);
    for (int j = 0; j < n; ++j) {
        const int g = gidx_f[j];
        if (g < 0 || g >= (int)cloud->size()) { double z[3] = {0,0,0}; P[j] = pargeo::point<3>(z); continue; }
        const Universe::PointT& pt = (*cloud)[g];
        double xyz[3] = {pt.x, pt.y, pt.z};
        P[j] = pargeo::point<3>(xyz);
    }

    CoutSilencer silence;                          // mute the lib's per-call cout spam
    const auto t0 = std::chrono::steady_clock::now();
    parlay::sequence<pargeo::wghEdge> E = pargeo::hdbscan<3>(P, (size_t)min_pts);
    parlay::sequence<pargeo::dendroNode> dendro = pargeo::dendrogram(E, (size_t)n);
    std::vector<LinkNode> linkage(dendro.size());
    for (std::size_t r = 0; r < dendro.size(); ++r)
        linkage[r] = LinkNode{(long)std::get<0>(dendro[r]), (long)std::get<1>(dendro[r]),
                              std::get<2>(dendro[r]),        (long)std::get<3>(dendro[r])};
    std::vector<int> labels = extractFlatClusters(linkage, n, p.min_cluster_size,
                                                  p.allow_single_cluster, p.leaf_selection);
    const auto t1 = std::chrono::steady_clock::now();

    if (out_clusters) {
        int k = 0;
        for (int l : labels) k = std::max(k, l + 1);
        *out_clusters = k;
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
