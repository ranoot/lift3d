#include "dog_stream.h"
#include "euler_pose.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

DogStream::DogStream(CameraCalib calib, SyncedHook on_synced,
                     std::size_t pose_buffer, std::size_t scan_ring,
                     std::size_t max_pending_images)
    : calib_(std::move(calib)),
      on_synced_(std::move(on_synced)),
      interp_(pose_buffer),
      scan_ring_(scan_ring < 1 ? 1 : scan_ring),
      max_pending_(max_pending_images < 1 ? 1 : max_pending_images) {}

void DogStream::pushScan(int64_t t_ns, const double pose6[6],
                         const float* xyz_body, const unsigned char* intensity, int n) {
    interp_.push(t_ns, pose6);

    // Lift the body-frame cloud into world using THIS scan's own exact pose (points
    // were captured at scan time; that pose is exact, no interpolation needed).
    doglog::Iso3 T = doglog::iso_from_pose6(pose6);
    ScanBuf sb;
    sb.t_ns = t_ns;
    sb.world.reserve((std::size_t)(n > 0 ? n : 0) * 3);
    sb.intensity.reserve((std::size_t)(n > 0 ? n : 0));
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d pw = T * Eigen::Vector3d(xyz_body[3 * i + 0],
                                                 xyz_body[3 * i + 1],
                                                 xyz_body[3 * i + 2]);
        sb.world.push_back((float)pw.x());
        sb.world.push_back((float)pw.y());
        sb.world.push_back((float)pw.z());
        sb.intensity.push_back(intensity ? (float)intensity[i] : 0.0f);
    }
    scans_.push_back(std::move(sb));
    while (scans_.size() > scan_ring_) scans_.pop_front();

    drain(false);
}

void DogStream::pushImage(const DogImage& img) {
    if (pending_.size() >= max_pending_) {
        // Backpressure: the pose stream has stalled relative to images. Drop the
        // oldest still-unbracketed image so memory stays bounded.
        pending_.pop_front();
        ++dropped_overflow_;
        std::fprintf(stderr, "[dog_stream] WARN: pending-image overflow, dropped oldest\n");
    }
    pending_.push_back(img);
    drain(false);
}

void DogStream::flush() { drain(true); }

int DogStream::nearestScan(int64_t t_ns) const {
    int best = -1;
    int64_t best_d = 0;
    for (int k = 0; k < (int)scans_.size(); ++k) {
        int64_t d = scans_[k].t_ns >= t_ns ? scans_[k].t_ns - t_ns : t_ns - scans_[k].t_ns;
        if (best < 0 || d < best_d) { best_d = d; best = k; }
    }
    return best;
}

void DogStream::drain(bool at_flush) {
    while (!pending_.empty()) {
        const DogImage& img = pending_.front();

        double T[16];
        PoseInterpolator::Status st = interp_.sample(img.t_ns, T);
        switch (st) {
            case PoseInterpolator::Status::EMPTY:
            case PoseInterpolator::Status::NEED_NEWER:
                // Not yet bracketed. Online: wait for the next scan. At flush there is
                // no more input, so these trailing images are undeliverable -> drop.
                if (at_flush) { pending_.pop_front(); ++dropped_old_; continue; }
                return;
            case PoseInterpolator::Status::TOO_OLD:
                // Older than any buffered pose (e.g. camera frames before the first
                // pose): can never be placed in the world. Drop.
                pending_.pop_front();
                ++dropped_old_;
                continue;
            case PoseInterpolator::Status::OK:
                break;
        }

        SyncedFrame sf;
        sf.image_t = img.t_ns;
        for (int k = 0; k < 16; ++k) sf.T_world_body[k] = T[k];
        sf.cam = cameraFromBodyPose(calib_, T);
        sf.robot[0] = (float)T[3];
        sf.robot[1] = (float)T[7];
        sf.robot[2] = (float)T[11];
        sf.image = img;

        const int s = nearestScan(img.t_ns);
        if (s >= 0) {
            sf.scan_t     = scans_[s].t_ns;
            sf.cloud_world = scans_[s].world;
            sf.intensity   = scans_[s].intensity;
        }

        pending_.pop_front();     // pop before the hook (hook may be slow / re-entrant-safe)
        ++emitted_;
        if (on_synced_) on_synced_(sf);
    }
}
