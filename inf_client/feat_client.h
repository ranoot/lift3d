#pragma once
// C++ client for the Python mask3d_feat backbone server (Mask3D Res16UNet34C),
// spoken over ZeroMQ (REQ) + msgpack -- same wire pattern as inf_client, a
// different endpoint (ipc:///tmp/mask3d_feat.ipc). The server turns a sparse voxel
// cloud into PER-VOXEL 96-d feature vectors (it stops before the mask head).
//
// Contract (see mask3d_feat/main.py):
//   - Input  coords: int32 (N,3) INTEGER VOXEL INDICES (world xyz / 0.02, rounded).
//            feats : float32 (N,3) NORMALIZED RGB colour (the ScanNet transform,
//                    (rgb/255 - mean)/std). The server does NOT voxelize or
//                    normalize -- the caller must.
//   - Output coords: int32 (M,4) = [batch, x, y, z]; feats: float32 (M,96).
//     MinkowskiEngine coalesces duplicate voxels, so M <= N; align features back to
//     input points by matching voxel (x,y,z).
//
// Usage:
//   FeatClient feat("ipc:///tmp/mask3d_feat.ipc");
//   FeatResult r = feat.features(coords, feats3, N);   // r.feats: M*r.C floats

#include <cstdint>
#include <string>
#include <vector>

// Per-voxel backbone features for one request. coords is M*4 (batch,x,y,z), feats
// is M*C row-major. C is 96 for the scannet_val backbone.
struct FeatResult {
    int                  M = 0;   // number of output voxels
    int                  C = 0;   // feature dimension (96)
    std::vector<int32_t> coords;  // M*4: batch, voxel x, y, z
    std::vector<float>   feats;   // M*C row-major

    bool ok() const { return M > 0 && C > 0 &&
                             (int)coords.size() == M * 4 && (int)feats.size() == M * C; }
};

class FeatClient {
public:
    // Connects a REQ socket to the server endpoint (does not block on the model).
    explicit FeatClient(const std::string& endpoint = "ipc:///tmp/mask3d_feat.ipc",
                        int recv_timeout_ms = 600000);
    ~FeatClient();
    FeatClient(const FeatClient&) = delete;
    FeatClient& operator=(const FeatClient&) = delete;

    bool ping();

    // Run one voxel cloud through the backbone. coords: N*3 int32 voxel indices;
    // feats3: N*3 float32 normalized colour. Returns per-voxel features (M <= N).
    FeatResult features(const std::vector<int32_t>& coords,
                        const std::vector<float>& feats3, int N);

private:
    struct Impl;              // hides the cppzmq context/socket from consumers
    Impl* impl_;
};
