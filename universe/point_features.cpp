#include "point_features.h"
#include "feat_client.h"     // FeatClient, FeatResult

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace {

// Integer voxel key, matching the coords we send to the backbone (xyz / voxel).
struct VoxKey {
    int x, y, z;
    bool operator==(const VoxKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxHash {
    std::size_t operator()(const VoxKey& k) const {
        std::size_t h = std::hash<int>()(k.x);
        h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

inline int vround(float v) { return (int)std::lround(v); }

} // namespace

void PointFeatures::refreshLocal(const Universe& uni, FeatClient& feat,
                                 const float center[3], float radius, const Params& p,
                                 const std::vector<int>* extra_gidx) {
    ++version_;
    const int Nuni = uni.size();
    if (Nuni == 0) return;
    if ((int)valid_.size() < Nuni) valid_.resize(Nuni, 0);   // grow parallel to map_

    // Crop around the robot (radius <= 0 => whole world) + universe global indices.
    std::vector<int> gidx;
    Universe::Cloud::Ptr crop = uni.local(center, radius, &gidx);
    Universe::Cloud::ConstPtr full = uni.cloud();

    // Union in any extra points (e.g. member points of in-radius objects that spill
    // past the crop) so an object gets features over its whole extent. Dedup against
    // the crop and each other; coords are read by global index from `full`.
    if (extra_gidx && !extra_gidx->empty() && full) {
        std::unordered_set<int> seen(gidx.begin(), gidx.end());
        for (int g : *extra_gidx)
            if (g >= 0 && g < (int)full->size() && seen.insert(g).second)
                gidx.push_back(g);
    }
    const int N = (int)gidx.size();
    if (N == 0 || !full) return;

    // Voxelize xyz and build normalized RGB (the two backbone inputs), preserving
    // order so coords[j]/feats3[j] correspond to gidx[j].
    std::vector<int32_t> coords((std::size_t)N * 3);
    std::vector<float>   feats3((std::size_t)N * 3);
    const float inv = 1.0f / (p.voxel > 0.0f ? p.voxel : 0.02f);
    for (int j = 0; j < N; ++j) {
        const Universe::PointT& q = (*full)[gidx[j]];
        coords[3 * j + 0] = vround(q.x * inv);
        coords[3 * j + 1] = vround(q.y * inv);
        coords[3 * j + 2] = vround(q.z * inv);
        float rgb[3];
        if (!uni.pointColor(gidx[j], rgb))
            rgb[0] = rgb[1] = rgb[2] = p.fallback_gray;
        for (int c = 0; c < 3; ++c)
            feats3[3 * j + c] = (rgb[c] / 255.0f - p.color_mean[c]) / p.color_std[c];
    }

    FeatResult r = feat.features(coords, feats3, N);
    if (!r.ok()) return;
    dim_ = r.C;

    // The store rows are index*dim_; grow it to cover every valid_ slot at this dim.
    if (feat_.size() < (std::size_t)valid_.size() * dim_)
        feat_.resize((std::size_t)valid_.size() * dim_, 0.0f);

    // Map each returned voxel (M rows, coords = [batch,x,y,z]) -> its feature row,
    // then scatter onto the crop points that share that voxel. M <= N (ME coalesces).
    std::unordered_map<VoxKey, int, VoxHash> row;
    row.reserve((std::size_t)r.M * 2 + 1);
    for (int m = 0; m < r.M; ++m)
        row.emplace(VoxKey{r.coords[4 * m + 1], r.coords[4 * m + 2], r.coords[4 * m + 3]}, m);

    for (int j = 0; j < N; ++j) {
        auto it = row.find(VoxKey{coords[3 * j + 0], coords[3 * j + 1], coords[3 * j + 2]});
        if (it == row.end()) continue;                 // voxel not in output (shouldn't happen)
        const int g = gidx[j];
        if (g < 0 || g >= (int)valid_.size()) continue;
        const float* src = &r.feats[(std::size_t)it->second * dim_];
        std::copy(src, src + dim_, &feat_[(std::size_t)g * dim_]);
        if (!valid_[g]) ++covered_;                    // first feature for this point
        valid_[g] = 1;
    }
}
