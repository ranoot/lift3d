#pragma once
// DogStream: the push-based entry point to the semantic pipeline, and the single
// place where the image/pose timeline is synchronized. It is transport-agnostic --
// a live driver (ZMQ/ROS/sensor SDK) and the offline simulated ingestor both feed it
// through the SAME two calls:
//
//     stream.pushScan (t_ns, pose6, xyz_body, intensity, n);   // ~10 Hz
//     stream.pushImage(DogImage{...});                          // ~5 Hz
//
// It buffers poses in a PoseInterpolator and, for each image, INTERPOLATES the body
// pose to the image's exact timestamp (SLERP+LERP), composes the camera-to-world at
// that instant, attaches the nearest lidar scan (already world-lifted with its own
// exact scan pose), and fires onSyncedFrame(SyncedFrame). This removes the up-to-
// ~100 ms lag that came from projecting an image's labels through the *scan-time*
// camera pose.
//
// Policy (chosen): WAIT for the bracketing pose. An image is held until a pose with
// t >= image_t has arrived, so interpolation is always two-sided. Images older than
// the buffered pose history (e.g. camera frames logged before the first pose) can
// never be bracketed and are dropped.
//
// PCL-free by design (mirrors dog_log_adaptor.h): the world cloud is a flat
// std::vector<float>. The universe-side consumer converts it to a PCL cloud.

#include "dog_log_adaptor.h"     // Camera, CameraCalib, DogImage
#include "pose_interp.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

// One time-aligned observation emitted by DogStream: an image plus the camera pose
// at its exact capture instant and the lidar cloud nearest that instant.
struct SyncedFrame {
    int64_t image_t = 0;               // image capture timestamp (ns) -- the sync anchor
    Camera  cam{};                     // K + T_world_cam interpolated to image_t
    double  T_world_body[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // interp @ image_t
    float   robot[3] = {0, 0, 0};      // robot world position @ image_t
    DogImage image;                    // decoded RGB @ image_t

    int64_t            scan_t = 0;      // timestamp of the attached scan
    std::vector<float> cloud_world;     // N*3 world-lifted lidar (nearest scan)
    std::vector<float> intensity;       // N, aligned to cloud_world
};

class DogStream {
public:
    using SyncedHook = std::function<void(const SyncedFrame&)>;

    // calib supplies K + the camera-optical->body mount (T_body_cam). on_synced is
    // called once per successfully bracketed image, in increasing image_t order.
    DogStream(CameraCalib calib, SyncedHook on_synced,
              std::size_t pose_buffer = 256, std::size_t scan_ring = 8,
              std::size_t max_pending_images = 64);

    // Ingest one lidar scan + its body->world pose. xyz_body is n*3 body-frame points
    // (zero-return beams already dropped by the caller); intensity is n bytes (may be
    // null). The cloud is lifted to world here using this scan's own exact pose.
    void pushScan(int64_t t_ns, const double pose6[6],
                  const float* xyz_body, const unsigned char* intensity, int n);

    // Ingest one decoded image. Emits any now-bracketable pending frames.
    void pushImage(const DogImage& img);

    // End-of-stream: any pending images that could still be bracketed by the current
    // buffer are emitted; the rest (newer than the last pose) are dropped.
    void flush();

    // Diagnostics.
    long emitted()          const { return emitted_; }
    long droppedTooOld()    const { return dropped_old_; }
    long droppedOverflow()  const { return dropped_overflow_; }

private:
    void drain(bool at_flush);
    // Index into scans_ of the buffered scan nearest t_ns (-1 if none).
    int  nearestScan(int64_t t_ns) const;

    struct ScanBuf { int64_t t_ns; std::vector<float> world; std::vector<float> intensity; };

    CameraCalib          calib_;
    SyncedHook           on_synced_;
    PoseInterpolator     interp_;
    std::deque<ScanBuf>  scans_;
    std::deque<DogImage> pending_;
    std::size_t          scan_ring_;
    std::size_t          max_pending_;
    long emitted_ = 0, dropped_old_ = 0, dropped_overflow_ = 0;
};
