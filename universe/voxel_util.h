#pragma once
// Shared, parlay-free voxel + feature helpers for the object store's growing
// (object_grow.cpp) and consolidation (object_consolidate.cpp) stages. Integer voxel
// keys at an arbitrary resolution (the volume unit for containment / adjacency), plus
// cosine similarity and a mean-feature reduction over universe global indices. Kept in
// one header so the two stages agree on the volume unit instead of duplicating the math.
// Deliberately free of the hdbscan/parlay runtime (which lives only in object_seeds.cpp).

#include "universe.h"           // Universe::PointT
#include "point_features.h"     // PointFeatures

#include <cmath>
#include <cstddef>
#include <functional>
#include <unordered_set>
#include <vector>

namespace objutil {

// Integer voxel key at a chosen resolution -- the volume unit for containment/adjacency.
struct VKey {
    int x, y, z;
    bool operator==(const VKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VHash {
    std::size_t operator()(const VKey& k) const {
        std::size_t h = std::hash<int>()(k.x);
        h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
using VSet = std::unordered_set<VKey, VHash>;

// Voxel key of a point at resolution 1/inv (pass inv = 1/voxel).
inline VKey voxelOf(const Universe::PointT& p, float inv) {
    return VKey{(int)std::floor(p.x * inv), (int)std::floor(p.y * inv),
                (int)std::floor(p.z * inv)};
}

inline float dist2(const float a[3], const float b[3]) {
    const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

// Cosine similarity of two equal-length feature vectors; 0 if either is empty/degenerate.
inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return 0.0f;
    return (float)(dot / std::sqrt(na * nb));
}

// Mean feature over the members of `pts` that already have a Mask3D feature; empty
// (size 0) if none do. dim comes from the feature store.
inline std::vector<float> meanFeature(const std::vector<int>& pts, const PointFeatures& pf) {
    const int dim = pf.dim();
    std::vector<double> acc((std::size_t)dim, 0.0);
    int n = 0;
    for (int g : pts) {
        const float* f = pf.feature(g);
        if (!f) continue;
        for (int c = 0; c < dim; ++c) acc[c] += f[c];
        ++n;
    }
    if (n == 0) return {};
    std::vector<float> out((std::size_t)dim);
    const double inv = 1.0 / (double)n;
    for (int c = 0; c < dim; ++c) out[c] = (float)(acc[c] * inv);
    return out;
}

}  // namespace objutil
