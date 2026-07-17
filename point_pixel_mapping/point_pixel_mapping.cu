#include "point_pixel_mapping.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/bit>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err__), __FILE__, __LINE__);     \
            return -1;                                                       \
        }                                                                    \
    } while (0)

// Sentinel for an empty z-buffer cell. For finite positive floats the IEEE-754
// bit pattern is monotonic, so a min over the reinterpreted ints gives the true
// minimum depth. We keep the int view so we can use an integral fetch_min (which
// libcu++ supports across all archs, unlike floating-point atomic min).
// 0x7f7fffff == FLT_MAX.
static const int Z_EMPTY = 0x7f7fffff;

// Fills an int buffer with a constant (used to init the z-buffer; cudaMemset is
// byte-wise so it can't write 0x7f7fffff directly).
__global__ void fillIntKernel(int* buf, int n, int val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

// --- Kernel 1: project every point, write per-point pixel + depth, and
//     race-free rasterize the nearest depth into the z-buffer. -----------------
//
// M is the precomputed 3x4 projection matrix M = K * (T^-1)[0:3, 0:4], row-major.
// [a,b,c]^T = M * [x,y,z,1]^T;  u = a/c, v = b/c, depth zi = c.
__global__ void projectKernel(const float* __restrict__ points_xyz,
                              int N,
                              const float* __restrict__ M, // 12 floats
                              int W, int H,
                              int r_max,                    // splat radius / cap (px)
                              int splat_min,                // lower clamp (depth-scaled)
                              float k_px,                   // fx*splat_world; 0 => constant r_max
                              int*   __restrict__ out_uv,
                              float* __restrict__ out_depth,
                              int*   __restrict__ zbuffer) // int view, size W*H
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = points_xyz[3 * i + 0];
    float y = points_xyz[3 * i + 1];
    float z = points_xyz[3 * i + 2];

    float a = M[0] * x + M[1] * y + M[2]  * z + M[3];
    float b = M[4] * x + M[5] * y + M[6]  * z + M[7];
    float c = M[8] * x + M[9] * y + M[10] * z + M[11]; // camera-space depth

    out_depth[i] = c;
    out_uv[2 * i + 0] = -1;
    out_uv[2 * i + 1] = -1;

    if (c <= 0.0f) return;             // behind the camera

    int u = (int)floorf(a / c + 0.5f); // round to nearest pixel
    int v = (int)floorf(b / c + 0.5f);
    if (u < 0 || u >= W || v < 0 || v >= H) return; // outside frustum

    out_uv[2 * i + 0] = u;
    out_uv[2 * i + 1] = v;

    // Splat depth into the (2r+1)^2 window (nearest-wins) so the sparse cloud forms
    // a continuous occluder; per-point (u,v) above stays the exact centre pixel.
    // Depth-scaled radius (constant r_max in the legacy path, k_px == 0).
    const int r = k_px > 0.0f
        ? min(r_max, max(splat_min, (int)lroundf(k_px / c)))
        : r_max;
    const int cbits = cuda::std::bit_cast<int>(c);
    const int u0 = u - r > 0 ? u - r : 0;
    const int u1 = u + r < W - 1 ? u + r : W - 1;
    const int v0 = v - r > 0 ? v - r : 0;
    const int v1 = v + r < H - 1 ? v + r : H - 1;
    for (int vv = v0; vv <= v1; ++vv)
        for (int uu = u0; uu <= u1; ++uu) {
            cuda::atomic_ref<int, cuda::thread_scope_device> z_cell(zbuffer[vv * W + uu]);
            z_cell.fetch_min(cbits, cuda::memory_order_relaxed);
        }
}

// --- Kernel 2: per-point visibility against the resolved z-buffer. ------------
__global__ void visibilityKernel(int N,
                                 const int*   __restrict__ out_uv,
                                 const float* __restrict__ out_depth,
                                 const int*   __restrict__ zbuffer,
                                 int W,
                                 float tau_vis,
                                 unsigned char* __restrict__ out_visible,
                                 int* __restrict__ out_corr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int u = out_uv[2 * i + 0];
    int v = out_uv[2 * i + 1];

    out_visible[i] = 0;
    out_corr[3 * i + 0] = -1;
    out_corr[3 * i + 1] = -1;
    out_corr[3 * i + 2] = 0;

    if (u < 0) return; // behind camera or out of frame (set in kernel 1)

    // Kernel 1 has finished, so this cell is no longer being written; a plain
    // load is sufficient. bit_cast reverses the float->int reinterpretation.
    float zi = out_depth[i];
    float d  = cuda::std::bit_cast<float>(zbuffer[v * W + u]);

    if (fabsf(zi - d) <= tau_vis) {
        out_visible[i] = 1;
        out_corr[3 * i + 0] = u;
        out_corr[3 * i + 1] = v;
        out_corr[3 * i + 2] = 1;
    }
}

// buildProjection (M = K * T^-1[0:3,:]) lives in the header, shared with the CPU
// back end.

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
    const int r_max = zbuf_dilate > 0 ? zbuf_dilate : 0;
    // Perspective-correct splat: k_px = fx * splat_world is one voxel's image
    // footprint at unit depth; the kernel divides by each point's depth and clamps
    // to [splat_min, r_max]. splat_world <= 0 => the legacy constant radius r_max.
    const float k_px = splat_world > 0.0f ? cam.K[0] * splat_world : 0.0f;
    const size_t npix = (size_t)W * H;

    float M[12];
    buildProjection(cam, M);

    float *d_points = nullptr, *d_M = nullptr, *d_depth = nullptr;
    int   *d_uv = nullptr, *d_zbuf = nullptr, *d_corr = nullptr;
    unsigned char* d_vis = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(float) * 3 * N));
    CUDA_CHECK(cudaMalloc(&d_M,      sizeof(float) * 12));
    CUDA_CHECK(cudaMalloc(&d_depth,  sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_uv,     sizeof(int) * 2 * N));
    CUDA_CHECK(cudaMalloc(&d_corr,   sizeof(int) * 3 * N));
    CUDA_CHECK(cudaMalloc(&d_vis,    sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_zbuf,   sizeof(int) * npix));

    CUDA_CHECK(cudaMemcpy(d_points, points_xyz, sizeof(float) * 3 * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M, sizeof(float) * 12, cudaMemcpyHostToDevice));

    // Initialize z-buffer to Z_EMPTY (every byte 0x7f -> 0x7f7f7f7f? no: use memset
    // per-int via a fill kernel-free trick). memset works byte-wise, so set each
    // int explicitly with a tiny kernel-free approach: use cudaMemset on bytes
    // only works for 0/-1. Use a fill via thrust-free loop kernel instead.
    {
        // Fill kernel inline via a lambda-free helper kernel.
        extern __global__ void fillIntKernel(int*, int, int);
        int threads = 256, blocks = (int)((npix + threads - 1) / threads);
        fillIntKernel<<<blocks, threads>>>(d_zbuf, (int)npix, Z_EMPTY);
        CUDA_CHECK(cudaGetLastError());
    }

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    projectKernel<<<blocks, threads>>>(d_points, N, d_M, W, H, r_max, splat_min, k_px,
                                       d_uv, d_depth, d_zbuf);
    CUDA_CHECK(cudaGetLastError());

    visibilityKernel<<<blocks, threads>>>(N, d_uv, d_depth, d_zbuf, W,
                                          tau_vis, d_vis, d_corr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_uv,    d_uv,    sizeof(int) * 2 * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_depth, d_depth, sizeof(float) * N,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_visible, d_vis, sizeof(unsigned char) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_corr,  d_corr,  sizeof(int) * 3 * N, cudaMemcpyDeviceToHost));

    int visible = 0;
    for (int i = 0; i < N; ++i) visible += out_visible[i];

    cudaFree(d_points); cudaFree(d_M); cudaFree(d_depth);
    cudaFree(d_uv); cudaFree(d_corr); cudaFree(d_vis); cudaFree(d_zbuf);
    return visible;
}
