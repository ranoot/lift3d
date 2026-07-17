// CPU back end for the SGS-3D point-to-pixel mapping. Implements the exact same
// API and semantics as point_pixel_mapping.cu so it is a drop-in replacement on
// machines without a usable GPU. Single-threaded and deterministic; the three
// passes mirror the two CUDA kernels (project, z-buffer, visibility).

#include "point_pixel_mapping.cuh"

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>

int projectPointsToPixels(const float* points_xyz,
                          int          N,
                          const Camera& cam,
                          float        tau_vis,
                          int*         out_uv,
                          float*       out_depth,
                          unsigned char* out_visible,
                          int*         out_corr,
                          int          zbuf_dilate,
                          int          splat_min,
                          float        splat_world)
{
    if (N <= 0) return 0;
    const int W = cam.width, H = cam.height;
    const int r_max = zbuf_dilate > 0 ? zbuf_dilate : 0;   // splat radius / cap (px)
    // Perspective-correct splat: a point at depth c gets a radius sized to one
    // voxel's image footprint. k_px = fx * splat_world is that footprint at unit
    // depth; the per-point radius is k_px / c, clamped to [splat_min, r_max].
    // splat_world <= 0 => the legacy constant radius r_max.
    const float k_px = splat_world > 0.0f ? cam.K[0] * splat_world : 0.0f;

    float M[12];
    buildProjection(cam, M);

    // Z-buffer of nearest depth per pixel; +inf means "empty".
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> zbuf((size_t)W * H, INF);

    // Pass 1: project every point, store per-point pixel + depth, rasterize depth.
    // With r>0 the depth is SPLATTED into a (2r+1)^2 window (nearest-wins) so the
    // sparse voxel cloud forms a continuous occluding surface instead of a scatter
    // of dots with holes a point behind a wall could slip through. Per-point (u,v)
    // stays the exact centre pixel -- only the depth test dilates.
    for (int i = 0; i < N; ++i) {
        float x = points_xyz[3 * i + 0];
        float y = points_xyz[3 * i + 1];
        float z = points_xyz[3 * i + 2];

        float a = M[0] * x + M[1] * y + M[2]  * z + M[3];
        float b = M[4] * x + M[5] * y + M[6]  * z + M[7];
        float c = M[8] * x + M[9] * y + M[10] * z + M[11]; // camera-space depth

        out_depth[i]      = c;
        out_uv[2 * i + 0] = -1;
        out_uv[2 * i + 1] = -1;

        if (c <= 0.0f) continue;                       // behind the camera

        int u = (int)std::floor(a / c + 0.5f);         // round to nearest pixel
        int v = (int)std::floor(b / c + 0.5f);
        if (u < 0 || u >= W || v < 0 || v >= H) continue; // outside frustum

        out_uv[2 * i + 0] = u;
        out_uv[2 * i + 1] = v;

        // Depth-scaled radius (constant r_max in the legacy path, k_px == 0).
        const int r = k_px > 0.0f
            ? std::clamp((int)std::lround(k_px / c), splat_min, r_max)
            : r_max;

        const int u0 = u - r > 0 ? u - r : 0;
        const int u1 = u + r < W - 1 ? u + r : W - 1;
        const int v0 = v - r > 0 ? v - r : 0;
        const int v1 = v + r < H - 1 ? v + r : H - 1;
        for (int vv = v0; vv <= v1; ++vv)
            for (int uu = u0; uu <= u1; ++uu) {
                float& cell = zbuf[(size_t)vv * W + uu];
                if (c < cell) cell = c;                // nearest-depth min
            }
    }

    // Pass 2: visibility against the resolved z-buffer.
    int visible = 0;
    for (int i = 0; i < N; ++i) {
        out_visible[i]      = 0;
        out_corr[3 * i + 0] = -1;
        out_corr[3 * i + 1] = -1;
        out_corr[3 * i + 2] = 0;

        int u = out_uv[2 * i + 0];
        int v = out_uv[2 * i + 1];
        if (u < 0) continue; // behind camera or out of frame

        float zi = out_depth[i];
        float d  = zbuf[(size_t)v * W + u];
        if (std::fabs(zi - d) <= tau_vis) {
            out_visible[i]      = 1;
            out_corr[3 * i + 0] = u;
            out_corr[3 * i + 1] = v;
            out_corr[3 * i + 2] = 1;
            ++visible;
        }
    }

    return visible;
}
