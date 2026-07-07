// Integration check for the sync layer, WITHOUT the heavy semantic stack (no PCL,
// no inf_server, no GPU): drive the real offline ingestor over a ~/dog_logs
// recording into a DogStream and verify the timeline behaviour + quantify the fix.
//
// It answers three things:
//   1. Every image in the window is emitted exactly once, in increasing image_t
//      order (minus the leading frames that precede the first pose and can't be
//      bracketed -- those are correctly dropped as TOO_OLD).
//   2. The camera pose attached to each frame is the body pose INTERPOLATED to the
//      image timestamp (the fix), not the nearest scan's pose (the old behaviour).
//   3. How much that matters: the position gap between the interpolated camera
//      centre and the old nearest-scan camera centre (mm), i.e. the geometric error
//      the desync used to inject.
//
// Build (PCL-free TUs + Eigen; no CMake needed):
//   g++ -std=c++20 -I dog_log_adaptor -I /usr/include/eigen3 \
//       tests/dog_stream_check.cpp dog_log_adaptor/dog_log_adaptor.cpp \
//       dog_log_adaptor/pose_interp.cpp dog_log_adaptor/dog_stream.cpp \
//       dog_log_adaptor/dog_log_ingestor.cpp -o /tmp/dsc && /tmp/dsc <log_folder>

#include "dog_log_adaptor.h"
#include "dog_stream.h"
#include "dog_log_ingestor.h"
#include "pose_interp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string log = argc > 1 ? argv[1]
        : std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") +
          "/dog_logs/20260608_test_log_signDetection";
    const std::string calib_xml =
        std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") +
        "/dog_logs/CameraCalRaiboAsQUGV113.xml";
    const int cam_id = 211;
    const int start_image = argc > 2 ? std::atoi(argv[2]) : 0;
    const int max_images  = argc > 3 ? std::atoi(argv[3]) : 60;

    CameraCalib calib = loadCameraCalib(calib_xml, cam_id);
    DogLogReader reader(log + "/PoseAndPointClouds.bin");
    reader.openCamera(log + "/camera" + std::to_string(cam_id) + ".gclf");
    std::printf("log=%s\n scans=%d images=%d\n", log.c_str(), reader.size(), reader.numImages());

    // Reference interpolator over ALL scans, to independently recompute the pose at any
    // time and to find the nearest scan (the old pipeline's camera anchor).
    struct ScanPose { int64_t t; double x, y, z; };
    std::vector<ScanPose> scan_poses;
    scan_poses.reserve(reader.size());
    PoseInterpolator ref(reader.size() + 4);
    for (int i = 0; i < reader.size(); ++i) {
        double pose6[6];
        std::vector<float> xyz; std::vector<unsigned char> in;
        reader.scanRaw(i, pose6, xyz, in);
        ref.push(reader.timestamp(i), pose6);
        scan_poses.push_back({reader.timestamp(i), pose6[0], pose6[1], pose6[2]});
    }
    auto nearest_scan = [&](int64_t t) -> const ScanPose& {
        size_t best = 0; int64_t bd = std::llabs(scan_poses[0].t - t);
        for (size_t k = 1; k < scan_poses.size(); ++k) {
            int64_t d = std::llabs(scan_poses[k].t - t);
            if (d < bd) { bd = d; best = k; }
        }
        return scan_poses[best];
    };

    int    fails = 0;
    int    n = 0;
    int64_t prev_t = -1;
    double sum_gap = 0, max_gap = 0;       // interpolated-vs-nearest camera centre (m)
    double sum_dt  = 0, max_dt  = 0;       // |image_t - scan_t| (ms)
    bool   monotonic = true, cam_matches_interp = true;

    DogStream stream(calib, [&](const SyncedFrame& sf) {
        ++n;
        if (prev_t >= 0 && sf.image_t <= prev_t) monotonic = false;
        prev_t = sf.image_t;

        // (2) The camera centre must equal cameraFromBodyPose(interp @ image_t).
        double Ti[16];
        PoseInterpolator::Status stx = ref.sample(sf.image_t, Ti);
        if (stx == PoseInterpolator::Status::OK) {
            Camera ci = cameraFromBodyPose(calib, Ti);
            for (int k = 0; k < 16; ++k)
                if (std::fabs(ci.T[k] - sf.cam.T[k]) > 1e-4f) cam_matches_interp = false;

            // (3) Old anchor: the camera the OLD pipeline would have used = the camera
            // built from the NEAREST scan's EXACT pose (sampled at that scan's own
            // timestamp -> an exact hit in the reference interpolator).
            const ScanPose& ns = nearest_scan(sf.image_t);
            double Tn[16];
            ref.sample(ns.t, Tn);
            Camera cn = cameraFromBodyPose(calib, Tn);
            double gx = sf.cam.T[3]  - cn.T[3];
            double gy = sf.cam.T[7]  - cn.T[7];
            double gz = sf.cam.T[11] - cn.T[11];
            double gap = std::sqrt(gx*gx + gy*gy + gz*gz);
            sum_gap += gap; if (gap > max_gap) max_gap = gap;

            double dt = std::fabs((double)(sf.image_t - sf.scan_t)) / 1e6;
            sum_dt += dt; if (dt > max_dt) max_dt = dt;
        }
    });

    DogLogIngestor ingestor(reader);
    ingestor.run(stream, start_image, max_images);

    std::printf("\nsynced frames: %d (of %ld images pushed; %ld dropped too_old, %ld overflow)\n",
                n, ingestor.imagesPushed(), stream.droppedTooOld(), stream.droppedOverflow());
    std::printf("image_t strictly increasing:       %s\n", monotonic ? "yes" : "NO");
    std::printf("camera == interpolated(image_t):    %s\n", cam_matches_interp ? "yes" : "NO");
    if (n > 0) {
        std::printf("|image_t - attached scan_t|:        mean %.1f ms   max %.1f ms\n",
                    sum_dt / n, max_dt);
        std::printf("interp vs nearest-scan cam centre:  mean %.1f mm   max %.1f mm\n",
                    1000.0 * sum_gap / n, 1000.0 * max_gap);
        std::printf("  (^ the position error the old scan-time camera injected; now 0 by construction)\n");
    }

    if (!(n > 0))               { std::printf("[FAIL] no frames emitted\n"); ++fails; }
    if (!monotonic)             { std::printf("[FAIL] image_t not monotonic\n"); ++fails; }
    if (!cam_matches_interp)    { std::printf("[FAIL] camera pose != interpolated pose\n"); ++fails; }
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
