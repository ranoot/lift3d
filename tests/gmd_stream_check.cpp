// Integration check for the GMD -> pipeline translation layer, WITHOUT the heavy
// semantic stack (no PCL, no inf_server, no GPU). It synthesizes a short trajectory of
// GMD deployment messages (Common::Entity::PrimaryPose + PointCloud at 10 Hz, Image at
// 5 Hz), feeds them through GmdStreamAdaptor into a real DogStream, and verifies the
// translation end-to-end:
//
//   1. Every synthesized image is emitted exactly once, in increasing image_t order.
//   2. The camera pose DogStream attaches equals the body pose INTERPOLATED to the image
//      timestamp -- i.e. the GMD pose/cloud/image timestamps flowed through correctly and
//      the sync still works (compared against a reference PoseInterpolator, as
//      tests/dog_stream_check.cpp does).
//   3. A known body-frame lidar point lifts to the expected world coordinate (pose6 and
//      the body->world lift round-trip).
//   4. Pixel conversion is correct for RGB8 (copy), BGR8 (channel-swap) and MONO8
//      (replicate), honouring a non-tight row step.
//
// Build (PCL/OpenCV-free TUs + Eigen; no CMake needed):
//   g++ -std=c++20 -I gmd_adaptor -I dog_log_adaptor -I point_pixel_mapping \
//       -I gmdDataTypesForInterns2 -I /usr/include/eigen3 \
//       tests/gmd_stream_check.cpp gmd_adaptor/gmd_stream_adaptor.cpp \
//       dog_log_adaptor/dog_log_adaptor.cpp dog_log_adaptor/pose_interp.cpp \
//       dog_log_adaptor/dog_stream.cpp \
//       gmdDataTypesForInterns2/GMDBase/Entity/TimeStamp.cpp -o /tmp/gsc && /tmp/gsc

#include "gmd_stream_adaptor.h"

#include "dog_log_adaptor.h"
#include "dog_stream.h"
#include "pose_interp.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using Common::Entity::Image;
using Common::Entity::PointCloud;
using Common::Entity::PrimaryPose;
using Common::Entity::ProcessedPoint;

// Build a GMD TimeStamp holding an exact ns-since-epoch value (round-trips through
// gmd::toNanoseconds on a nanosecond system_clock, which Linux provides).
static Common::Entity::TimeStamp tsFromNs(int64_t ns) {
    return Common::Entity::TimeStamp(
        std::chrono::duration_cast<Common::Entity::TimeDuration>(
            std::chrono::nanoseconds(ns)));
}

static PrimaryPose makePose(int64_t t_ns, double x, double y, double z,
                            double roll, double pitch, double yaw) {
    PrimaryPose pp;                 // default-constructs timestamps to now()
    pp.timestamp = tsFromNs(t_ns);
    pp.x = x; pp.y = y; pp.z = (float)z;
    pp.roll = (float)roll; pp.pitch = (float)pitch; pp.yaw = (float)yaw;
    return pp;
}

// One body-frame point at (px,py,pz) so we can check the world lift downstream.
static PointCloud makeCloud(int64_t t_ns, float px, float py, float pz) {
    PointCloud pc;
    pc.timestamp = tsFromNs(t_ns);
    ProcessedPoint p; p.x = px; p.y = py; p.z = pz; p.intensity = 200;
    pc.pointCloudData.push_back(p);
    // A no-return beam that must be dropped by the translation (matches scanRaw).
    ProcessedPoint zero; zero.x = zero.y = zero.z = 0.0f;
    pc.pointCloudData.push_back(zero);
    return pc;
}

int main() {
    const int W = 4, H = 2;
    const int cam_id = 211;

    // Camera == body (identity mount) so the emitted camera centre equals the body
    // translation and the pose check is easy to reason about.
    CameraCalib calib;
    calib.sensor_id = cam_id;
    calib.fu = 100; calib.fv = 100; calib.cu = W / 2.0; calib.cv = H / 2.0;
    calib.width = W; calib.height = H;
    for (int i = 0; i < 16; ++i) calib.T_body_cam[i] = 0.0;
    calib.T_body_cam[0] = calib.T_body_cam[5] = calib.T_body_cam[10] = calib.T_body_cam[15] = 1.0;

    // --- reference interpolator (independent recompute of the pose at any time) --------
    PoseInterpolator ref(64);

    // --- capture emitted synced frames ------------------------------------------------
    struct Emit { int64_t image_t; float camx, camy, camz; std::vector<float> cloud; };
    std::vector<Emit> emits;
    DogStream stream(calib, [&](const SyncedFrame& sf) {
        emits.push_back({sf.image_t, sf.cam.T[3], sf.cam.T[7], sf.cam.T[11], sf.cloud_world});
    });
    gmd::GmdStreamAdaptor adaptor(stream);

    // --- synthesize a moving trajectory: poses+clouds @ 10 Hz, images @ 20 Hz ---------
    // A straight-line translation with a slow yaw sweep so interpolation is non-trivial.
    const int64_t dt = 100'000'000LL;     // 100 ms scan period (10 Hz)
    const int nscan = 12;
    auto poseAt = [](int64_t t) {         // ground-truth continuous pose, sampled below
        double s = (double)t / 1e9;
        return makePose(t, /*x*/ 1.0 * s, /*y*/ 0.5 * s, /*z*/ 0.2,
                        /*roll*/ 0.0, /*pitch*/ 0.0, /*yaw*/ 0.3 * s);
    };

    // Images fall exactly between scans (t = k*dt + dt/2), so they must be bracketed and
    // interpolated -- never landing on a pose sample.
    std::vector<Image> imgs;
    auto makeImage = [&](int64_t t, ImageEncoding enc) {
        Image im;
        im.timestamp = tsFromNs(t);
        im.sensorId = cam_id;
        im.width = W; im.height = H; im.encoding = enc;
        const int bpp = (enc == ImageEncoding::MONO8) ? 1 : 3;
        im.step = W * bpp + 5;            // deliberately padded stride to test step handling
        im.data.assign((size_t)im.step * H, 0);
        return im;
    };

    // Interleave scans and images in increasing-timestamp order (scans win ties).
    int img_seq = 0;
    for (int k = 0; k < nscan; ++k) {
        const int64_t ts = k * dt;
        PrimaryPose pp = poseAt(ts);
        double pose6[6]; gmd::toPose6(pp, pose6);
        ref.push(ts, pose6);
        adaptor.pushScan(pp, makeCloud(ts, 1.0f, 0.0f, 0.0f)); // body point at +x

        // Emit an image mid-interval (except after the last scan -> unbracketable).
        if (k < nscan - 1) {
            const int64_t it = ts + dt / 2;
            // Cycle encodings so all three colour paths are exercised.
            ImageEncoding enc = (img_seq % 3 == 0) ? ImageEncoding::RGB8
                              : (img_seq % 3 == 1) ? ImageEncoding::BGR8
                                                   : ImageEncoding::MONO8;
            Image im = makeImage(it, enc);
            // Paint pixel (row 0, col 0) a known colour; store expected RGB for check 4.
            if (enc == ImageEncoding::RGB8) {          // R=10,G=20,B=30
                im.data[0] = 10; im.data[1] = 20; im.data[2] = 30;
            } else if (enc == ImageEncoding::BGR8) {   // B=30,G=20,R=10 -> RGB 10,20,30
                im.data[0] = 30; im.data[1] = 20; im.data[2] = 10;
            } else {                                    // MONO8 g=42 -> RGB 42,42,42
                im.data[0] = 42;
            }
            imgs.push_back(im);
            adaptor.pushImage(im);
            ++img_seq;
        }
    }
    adaptor.flush();

    // --- checks -----------------------------------------------------------------------
    int fails = 0;

    // (1) every image emitted once, strictly increasing image_t.
    if ((int)emits.size() != img_seq) {
        std::printf("[FAIL] emitted %zu frames, expected %d\n", emits.size(), img_seq);
        ++fails;
    }
    bool monotonic = true;
    for (size_t i = 1; i < emits.size(); ++i)
        if (emits[i].image_t <= emits[i - 1].image_t) monotonic = false;
    if (!monotonic) { std::printf("[FAIL] image_t not strictly increasing\n"); ++fails; }

    // (2) attached camera pose == body pose interpolated to image_t (identity mount).
    double max_pose_err = 0.0;
    for (const Emit& e : emits) {
        double Ti[16];
        if (ref.sample(e.image_t, Ti) != PoseInterpolator::Status::OK) {
            std::printf("[FAIL] reference could not bracket image_t %lld\n",
                        (long long)e.image_t);
            ++fails; continue;
        }
        Camera ci = cameraFromBodyPose(calib, Ti);
        max_pose_err = std::max(max_pose_err, (double)std::fabs(ci.T[3]  - e.camx));
        max_pose_err = std::max(max_pose_err, (double)std::fabs(ci.T[7]  - e.camy));
        max_pose_err = std::max(max_pose_err, (double)std::fabs(ci.T[11] - e.camz));
    }
    if (max_pose_err > 1e-4) {
        std::printf("[FAIL] camera centre != interpolated pose (max err %.2e)\n", max_pose_err);
        ++fails;
    }

    // (3) world lift: body point (1,0,0) -> R(yaw@image_t)*(1,0,0) + t. Recompute from the
    // interpolated pose and compare to the emitted cloud (also confirms the zero beam was
    // dropped: exactly one world point per frame).
    double max_lift_err = 0.0;
    bool one_point = true;
    for (const Emit& e : emits) {
        if (e.cloud.size() != 3) { one_point = false; continue; }
        double Ti[16];
        ref.sample(e.image_t, Ti);
        // NOTE: the cloud is lifted with the SCAN's own exact pose (nearest scan), not the
        // interpolated image-time pose. Recompute using the nearest scan pose to match.
        // Nearest scan to a mid-interval image is the scan at floor or ceil; both are dt/2
        // away, and DogStream picks the first minimum (the earlier scan). Use that.
        int64_t scan_t = (e.image_t / dt) * dt;   // earlier bracketing scan
        double Ts[16];
        ref.sample(scan_t, Ts);
        // world = Ts * [1,0,0,1]
        double wx = Ts[0] * 1 + Ts[3];
        double wy = Ts[4] * 1 + Ts[7];
        double wz = Ts[8] * 1 + Ts[11];
        max_lift_err = std::max(max_lift_err, (double)std::fabs(e.cloud[0] - wx));
        max_lift_err = std::max(max_lift_err, (double)std::fabs(e.cloud[1] - wy));
        max_lift_err = std::max(max_lift_err, (double)std::fabs(e.cloud[2] - wz));
    }
    if (!one_point) { std::printf("[FAIL] expected exactly 1 world point (zero beam not dropped)\n"); ++fails; }
    if (max_lift_err > 1e-3) {
        std::printf("[FAIL] world lift mismatch (max err %.2e)\n", max_lift_err);
        ++fails;
    }

    // (4) pixel conversion for each encoding: pixel (0,0) must be RGB (10,20,30) for the
    // RGB8/BGR8 frames and (42,42,42) for MONO8.
    int pix_fail = 0;
    for (const Image& im : imgs) {
        DogImage di;
        if (!gmd::toDogImage(im, di)) { ++pix_fail; continue; }
        unsigned char r = di.rgb[0], g = di.rgb[1], b = di.rgb[2];
        if (im.encoding == ImageEncoding::MONO8) {
            if (!(r == 42 && g == 42 && b == 42)) ++pix_fail;
        } else {
            if (!(r == 10 && g == 20 && b == 30)) ++pix_fail;
        }
    }
    if (pix_fail) { std::printf("[FAIL] %d image(s) converted to wrong pixels\n", pix_fail); ++fails; }

    // (5) unsupported encoding is rejected (and counted as a dropped image).
    Image bad; bad.width = W; bad.height = H; bad.encoding = ImageEncoding::NOT_APPLICABLE;
    DogImage tmp;
    if (gmd::toDogImage(bad, tmp)) { std::printf("[FAIL] NOT_APPLICABLE image not rejected\n"); ++fails; }

    std::printf("\nsynced frames: %zu (of %d images pushed; %ld dropped too_old, %ld overflow)\n",
                emits.size(), img_seq, stream.droppedTooOld(), stream.droppedOverflow());
    std::printf("image_t strictly increasing:      %s\n", monotonic ? "yes" : "NO");
    std::printf("camera == interpolated(image_t):  max err %.2e m\n", max_pose_err);
    std::printf("body->world lift:                 max err %.2e m\n", max_lift_err);
    std::printf("pixel conversion failures:        %d / %zu\n", pix_fail, imgs.size());

    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
