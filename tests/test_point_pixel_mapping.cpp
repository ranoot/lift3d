// Tests for the SGS-3D point-to-pixel mapping, focused on the z-buffer
// (occlusion) behaviour. Backend-agnostic: links whichever projectPointsToPixels
// is built (CPU by default). Run: builds an executable that prints PASS/FAIL.
//
// Camera convention reminder: T is camera-to-world. With T = identity the camera
// sits at the world origin looking down +Z, so a world point (X,Y,Z>0) projects
// to u = fx*X/Z + cx, v = fy*Y/Z + cy, with camera-space depth = Z.

#include "point_pixel_mapping.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__);    \
        }                                                                       \
    } while (0)

// A simple, predictable camera: fx=fy=100, principal point at image centre,
// 100x100 image, identity pose.
static Camera testCamera() {
    return makeCamera(/*fx*/100, /*fy*/100, /*cx*/50, /*cy*/50,
                      /*W*/100, /*H*/100, /*T*/nullptr);
}

// --- Test 1: basic projection of a single in-frame point. --------------------
static void test_basic_projection() {
    std::printf("test_basic_projection\n");
    Camera cam = testCamera();
    std::vector<float> pts = { 0.0f, 0.0f, 5.0f }; // on the optical axis at z=5

    PointPixelMap m = mapPointsToPixels(pts, cam, /*tau*/0.05f);
    CHECK(m.ok(), "backend returned an error");
    CHECK(m.N == 1, "N");
    CHECK(m.u(0) == 50 && m.v(0) == 50, "axis point lands at image centre");
    CHECK(std::fabs(m.depth[0] - 5.0f) < 1e-5f, "depth == z");
    CHECK(m.isVisible(0), "the only point must be visible");
    CHECK(m.num_visible == 1, "num_visible");
    // corr mirrors uv + a 1 for visible points.
    CHECK(m.corr[0] == 50 && m.corr[1] == 50 && m.corr[2] == 1, "corr triplet");
}

// --- Test 2: a point behind the camera is rejected. --------------------------
static void test_behind_camera() {
    std::printf("test_behind_camera\n");
    Camera cam = testCamera();
    std::vector<float> pts = { 0.0f, 0.0f, -1.0f }; // negative depth

    PointPixelMap m = mapPointsToPixels(pts, cam, 0.05f);
    CHECK(m.ok(), "backend error");
    CHECK(!m.isVisible(0), "behind-camera point not visible");
    CHECK(!m.inImage(0), "behind-camera point has no pixel");
    CHECK(m.u(0) == -1 && m.v(0) == -1, "uv sentinel");
    CHECK(m.depth[0] <= 0.0f, "depth sign preserved (<=0)");
    CHECK(m.num_visible == 0, "nothing visible");
}

// --- Test 3: a point projecting outside the image is rejected. ---------------
static void test_out_of_frame() {
    std::printf("test_out_of_frame\n");
    Camera cam = testCamera();
    // u = 100*10/1 + 50 = 1050 -> way past width 100.
    std::vector<float> pts = { 10.0f, 0.0f, 1.0f };

    PointPixelMap m = mapPointsToPixels(pts, cam, 0.05f);
    CHECK(m.ok(), "backend error");
    CHECK(!m.inImage(0), "out-of-frame point has no pixel");
    CHECK(!m.isVisible(0), "out-of-frame point not visible");
    CHECK(m.num_visible == 0, "nothing visible");
}

// --- Test 4: THE z-buffer test. Two points on the same pixel at different
//     depths -> only the nearer survives; the farther is occluded. -----------
static void test_zbuffer_occlusion() {
    std::printf("test_zbuffer_occlusion\n");
    Camera cam = testCamera();
    // Both on the optical axis -> both project to (50,50). Near z=5, far z=10.
    std::vector<float> pts = {
        0.0f, 0.0f, 10.0f,   // index 0: FAR  (occluded)
        0.0f, 0.0f,  5.0f,   // index 1: NEAR (visible)
    };

    PointPixelMap m = mapPointsToPixels(pts, cam, /*tau*/0.05f);
    CHECK(m.ok(), "backend error");
    // Both land on the same pixel...
    CHECK(m.u(0) == 50 && m.v(0) == 50, "far point pixel");
    CHECK(m.u(1) == 50 && m.v(1) == 50, "near point pixel");
    // ...but only the nearer one is visible.
    CHECK(!m.isVisible(0), "far point is occluded");
    CHECK(m.isVisible(1),  "near point is visible");
    CHECK(m.num_visible == 1, "exactly one survivor");
    // Occluded point still keeps its true depth, just no corr.
    CHECK(std::fabs(m.depth[0] - 10.0f) < 1e-5f, "occluded depth retained");
    CHECK(m.corr[3*0+2] == 0, "occluded corr flag is 0");
    CHECK(m.corr[3*1+2] == 1, "visible corr flag is 1");
}

// --- Test 5: tau_vis acts as a surface-thickness tolerance. Two points on the
//     same pixel within tau are BOTH visible; widen the gap and the far one
//     drops out. -----------------------------------------------------------
static void test_tau_tolerance() {
    std::printf("test_tau_tolerance\n");
    Camera cam = testCamera();

    // Gap of 0.02 < tau 0.05 -> both kept (same surface).
    {
        std::vector<float> pts = { 0,0,5.00f,  0,0,5.02f };
        PointPixelMap m = mapPointsToPixels(pts, cam, 0.05f);
        CHECK(m.isVisible(0) && m.isVisible(1), "both within tau are visible");
        CHECK(m.num_visible == 2, "two visible within tau");
    }
    // Same geometry, tighter tau 0.01 < gap 0.02 -> far one occluded.
    {
        std::vector<float> pts = { 0,0,5.00f,  0,0,5.02f };
        PointPixelMap m = mapPointsToPixels(pts, cam, 0.01f);
        CHECK(m.isVisible(0), "near point still visible");
        CHECK(!m.isVisible(1), "far point beyond tau is occluded");
        CHECK(m.num_visible == 1, "one visible beyond tau");
    }
}

// --- Test 6: empty input and self-consistency invariants. --------------------
static void test_invariants() {
    std::printf("test_invariants\n");
    Camera cam = testCamera();

    // Empty cloud.
    {
        std::vector<float> pts;
        PointPixelMap m = mapPointsToPixels(pts, cam, 0.05f);
        CHECK(m.ok(), "empty cloud is not an error");
        CHECK(m.num_visible == 0, "empty -> 0 visible");
    }

    // A spread of points; check num_visible == sum(visible) and corr/uv agree.
    {
        std::vector<float> pts;
        for (int i = 0; i < 50; ++i) {
            float x = (float)(i % 7 - 3) * 0.1f;
            float y = (float)(i % 5 - 2) * 0.1f;
            pts.insert(pts.end(), { x, y, 2.0f + 0.01f * i });
        }
        PointPixelMap m = mapPointsToPixels(pts, cam, 0.05f);
        CHECK(m.ok(), "backend error");
        int sum = 0;
        for (int i = 0; i < m.N; ++i) {
            sum += m.isVisible(i) ? 1 : 0;
            if (m.isVisible(i)) {
                CHECK(m.corr[3*i+0] == m.u(i) && m.corr[3*i+1] == m.v(i),
                      "visible corr matches uv");
                CHECK(m.inImage(i), "visible implies in-image");
            }
        }
        CHECK(sum == m.num_visible, "num_visible equals visible-flag sum");
    }
}

int main() {
    test_basic_projection();
    test_behind_camera();
    test_out_of_frame();
    test_zbuffer_occlusion();
    test_tau_tolerance();
    test_invariants();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
    std::printf("SOME TESTS FAILED\n");
    return 1;
}
