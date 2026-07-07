#pragma once
#include <cstdint>
#include <vector>

// Point-to-pixel mapping from SGS-3D (arXiv:2509.05144v2).
//
// Projects a world-space point cloud into a camera image, resolves visibility
// with a z-buffer reconstructed from the cloud itself (no depth sensor needed),
// and emits a per-point correspondence map [u, v, 1] for the visible points.
//
//   zi * [ui, vi, 1]^T = K_t * T_t^-1 * p_i          (projection)
//   D_t(u,v) = min { zi : point i lands on (u,v) }   (z-buffer)
//   V_t(i)   = 1[in-bounds] * 1[|zi - D_t(ui,vi)| <= tau_vis]
//
// All host-side buffers are caller-owned. The implementation is PCL-free so it
// can be reused by any front end.

struct Camera {
    float K[9];   // row-major 3x3 intrinsics
    float T[16];  // row-major 4x4 camera-to-world pose (extrinsic); assumed rigid
    int   width;  // image width  W
    int   height; // image height H
};

// Convenience constructor: fill a Camera from pinhole intrinsics + a 4x4
// camera-to-world pose (row-major). Avoids hand-laying-out the K matrix.
// T_cam_to_world may be null to use identity (camera at world origin, +Z forward).
inline Camera makeCamera(float fx, float fy, float cx, float cy,
                         int width, int height,
                         const float* T_cam_to_world = nullptr) {
    Camera c{};
    c.K[0] = fx; c.K[1] = 0;  c.K[2] = cx;
    c.K[3] = 0;  c.K[4] = fy; c.K[5] = cy;
    c.K[6] = 0;  c.K[7] = 0;  c.K[8] = 1;
    if (T_cam_to_world) {
        for (int i = 0; i < 16; ++i) c.T[i] = T_cam_to_world[i];
    } else {
        for (int i = 0; i < 16; ++i) c.T[i] = 0.0f;
        c.T[0] = c.T[5] = c.T[10] = c.T[15] = 1.0f; // identity
    }
    c.width = width;
    c.height = height;
    return c;
}

// Builds the 3x4 projection matrix M = K * (T^-1)[0:3,:] (row-major), so that
// [a,b,c]^T = M * [x,y,z,1]^T gives u=a/c, v=b/c and camera-space depth z=c.
// T is assumed a rigid camera-to-world pose, so T^-1 = [R^T | -R^T t]. Pure host
// arithmetic, shared by the CPU and CUDA back ends.
inline void buildProjection(const Camera& cam, float M[12]) {
    const float* K = cam.K;
    const float* T = cam.T; // row-major 4x4

    // R^T (camera-from-world rotation): Rt[r][c] = R[c][r] = T[c*4 + r]
    float Rt[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Rt[r * 3 + c] = T[c * 4 + r];

    float t[3] = { T[3], T[7], T[11] };       // translation of T
    float tinv[3];                            // -R^T t
    for (int r = 0; r < 3; ++r)
        tinv[r] = -(Rt[r * 3 + 0] * t[0] + Rt[r * 3 + 1] * t[1] + Rt[r * 3 + 2] * t[2]);

    float Tt[12];                             // T^-1 top 3 rows as 3x4: [ R^T | tinv ]
    for (int r = 0; r < 3; ++r) {
        Tt[r * 4 + 0] = Rt[r * 3 + 0];
        Tt[r * 4 + 1] = Rt[r * 3 + 1];
        Tt[r * 4 + 2] = Rt[r * 3 + 2];
        Tt[r * 4 + 3] = tinv[r];
    }

    for (int r = 0; r < 3; ++r)               // M (3x4) = K (3x3) * Tt (3x4)
        for (int col = 0; col < 4; ++col) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
                s += K[r * 3 + k] * Tt[k * 4 + col];
            M[r * 4 + col] = s;
        }
}

// points_xyz : N*3 host array, world coords (x0,y0,z0, x1,y1,z1, ...)
// zbuf_dilate: z-buffer SPLAT radius in pixels. Each point rasterizes its depth
//   into the (2r+1)x(2r+1) window around its pixel (nearest-wins), turning the
//   sparse projected cloud into a continuous occluding surface. This is the SGS-3D
//   fix for the resolution mismatch between a sparse voxel map and a dense image:
//   with r=0 (a single pixel) the depth map is full of holes, so a point behind a
//   wall that lands in a hole pixel is nobody's occludee and leaks through as
//   "visible". r>=1 lets a nearby surface voxel win the hole and occlude it.
//   Per-point (u,v) is always the exact centre pixel; only the depth test dilates.
// Outputs (all caller-allocated, host):
//   out_uv      : N*2 ints, integer pixel (u,v) per point; (-1,-1) if behind cam
//                 or out of frame. Same indices the z-buffer used.
//   out_depth   : N floats, camera-space depth zi per point (<=0 means behind).
//   out_visible : N bytes, 1 if the point is visible, else 0.
//   out_corr    : N*3 ints, [u,v,1] for visible points, else [-1,-1,0].
// Returns the number of visible points, or -1 on a CUDA error.
int projectPointsToPixels(const float* points_xyz,
                          int          N,
                          const Camera& cam,
                          float        tau_vis,
                          int*         out_uv,
                          float*       out_depth,
                          unsigned char* out_visible,
                          int*         out_corr,
                          int          zbuf_dilate = 0);

// ---------------------------------------------------------------------------
// Ergonomic wrapper. Owns its output buffers (no caller pre-allocation) and
// offers per-point accessors. Same backend (CPU or CUDA) under the hood.
// ---------------------------------------------------------------------------
struct PointPixelMap {
    int N           = 0;   // number of input points
    int num_visible = -1;  // count of visible points, or -1 on backend error

    std::vector<int>           uv;      // 2N: (u,v) per point; (-1,-1) if off-image
    std::vector<float>         depth;   // N : camera-space depth z per point
    std::vector<unsigned char> visible; // N : 1 if visible, else 0
    std::vector<int>           corr;    // 3N: [u,v,1] if visible, else [-1,-1,0]

    bool ok()             const { return num_visible >= 0; }   // backend succeeded
    bool isVisible(int i) const { return visible[i] != 0; }
    int  u(int i)         const { return uv[2 * i + 0]; }      // -1 if off-image
    int  v(int i)         const { return uv[2 * i + 1]; }
    bool inImage(int i)   const { return uv[2 * i + 0] >= 0; } // projected in-frame
};

// One-call mapping: allocates the buffers, runs the backend, returns the result.
// Check .ok() before trusting the contents (false only on a CUDA error).
// zbuf_dilate is the depth splat radius (see projectPointsToPixels).
inline PointPixelMap mapPointsToPixels(const float* points_xyz, int N,
                                       const Camera& cam, float tau_vis,
                                       int zbuf_dilate = 0) {
    PointPixelMap m;
    m.N = N;
    m.uv.assign(N > 0 ? (size_t)2 * N : 0, -1);
    m.depth.assign(N > 0 ? (size_t)N : 0, 0.0f);
    m.visible.assign(N > 0 ? (size_t)N : 0, 0);
    m.corr.assign(N > 0 ? (size_t)3 * N : 0, 0);
    m.num_visible = projectPointsToPixels(points_xyz, N, cam, tau_vis,
                                          m.uv.data(), m.depth.data(),
                                          m.visible.data(), m.corr.data(),
                                          zbuf_dilate);
    return m;
}

// Same, taking a flat xyz vector (size must be a multiple of 3).
inline PointPixelMap mapPointsToPixels(const std::vector<float>& points_xyz,
                                       const Camera& cam, float tau_vis,
                                       int zbuf_dilate = 0) {
    return mapPointsToPixels(points_xyz.data(), (int)(points_xyz.size() / 3),
                             cam, tau_vis, zbuf_dilate);
}
