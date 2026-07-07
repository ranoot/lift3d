#pragma once
// Per-point deep features over the semantic Universe, from the Mask3D backbone
// (mask3d_feat server, via FeatClient). SGS-3D's later object-discovery stage needs
// a feature vector per point; this holds a PERSISTENT, whole-cloud per-point store
// (parallel to the Universe map, grow-only) that is filled INCREMENTALLY from local
// windows.
//
// Online scheme (locked with the user): the backbone is run per LOCAL WINDOW (the
// crop around the robot), and the resulting 96-d per-voxel features are SCATTERED
// into the whole-cloud store keyed on the universe global index. Overlapping windows
// simply overwrite (refresh) boundary points, so every point the robot passes ends
// up with a feature without ever feeding the whole (unbounded) cloud to the GPU at
// once. Features use local sparse-conv context -- fine for local object discovery.
//
// Like Superpoints, this layer only READS a Universe (plus a FeatClient), keeping the
// heavy IPC/backbone dependency out of the Universe class.

#include "universe.h"        // Universe

#include <cstdint>
#include <vector>

class FeatClient;            // inf_client/feat_client.h (pulled in by the .cpp)

class PointFeatures {
public:
    struct Params {
        float voxel = 0.02f;         // Mask3D backbone voxel size (m); coords = xyz/voxel
        // ScanNet colour normalization (albumentations A.Normalize, max=255):
        // feat = (rgb/255 - mean)/std. Values from Mask3D datasets/semseg.py.
        float color_mean[3] = {0.47793126f, 0.43032575f, 0.37495989f};
        float color_std[3]  = {0.28344755f, 0.27566158f, 0.27018971f};
        float fallback_gray = 128.0f; // colour for points no camera has seen yet (0..255)
    };

    int dim() const { return dim_; }            // feature dimension (96)
    int covered() const { return covered_; }    // # of universe points with a feature
    std::uint64_t version() const { return version_; }

    // Re-run the backbone on the crop of `uni` within `radius` of `center` and
    // scatter the features into the whole-cloud store. radius <= 0 => the whole map.
    // `extra_gidx`, when non-null, is a set of global indices (e.g. member points of
    // in-radius objects that spill past the crop) that are UNIONED into the backbone
    // request, so an object gets consistent features over its whole extent even when
    // parts of it sit outside `radius`. Duplicates (with the crop or each other) are
    // dropped.
    void refreshLocal(const Universe& uni, FeatClient& feat, const float center[3],
                      float radius, const Params& p,
                      const std::vector<int>* extra_gidx = nullptr);

    // Feature vector (dim() floats) for universe point `gidx`, or nullptr if that
    // point has no feature yet (never fell inside a processed window).
    const float* feature(int gidx) const {
        return has(gidx) ? &feat_[(std::size_t)gidx * dim_] : nullptr;
    }
    bool has(int gidx) const {
        return gidx >= 0 && gidx < (int)valid_.size() && valid_[gidx] != 0;
    }

private:
    int                  dim_ = 96;      // set from the first backbone response
    int                  covered_ = 0;   // count of valid_ != 0
    std::uint64_t        version_ = 0;   // bumped each refresh
    std::vector<float>   feat_;          // (grow-only) valid_.size() * dim_, row-major
    std::vector<uint8_t> valid_;         // per universe point: has a feature?
};
