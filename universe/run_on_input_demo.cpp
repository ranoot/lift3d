// Offline demo/harness for the simplified SemanticRunner entry point. It stands in for the
// robot's getSynchronizedInputs(): it replays a ~/dog_logs recording, and for each selected
// camera image it interpolates the body pose to the image timestamp (SLERP+LERP, exactly as
// the live sync would) and pairs it with the nearest lidar scan, packages them as the native
// GMD entity types (PrimaryPose / PointCloud / Image), and hands each OpenVocabSegInput to
// SemanticRunner::runOnInput -- proving the new interface drives the whole pipeline and
// matches run_semantic_universe on the same log.
//
// Usage:
//   run_on_input_demo --config run.yaml [--log <folder>] [--start N] [--count N] [--stride N]
// Viz is controlled by run.yaml's viz_endpoint (set it to e.g. ipc:///tmp/lift3d_viz.ipc to
// stream). The .rrd is written by the Python viz_server (start it with `--out run.rrd`); this
// binary only streams frames to it over ZMQ.
//
// Needs the inf_server running + a healthy GPU for the full run (see CLAUDE.md gotchas); it
// always COMPILES without them.

#include "semantic_runner.h"     // semantic::SemanticRunner, OpenVocabSegInput
#include "run_config.h"          // RunConfig, loadRunConfig
#include "dog_log_adaptor.h"     // DogLogReader, DogImage
#include "pose_interp.h"         // PoseInterpolator (SLERP+LERP to the image timestamp)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string arg_str(int argc, char** argv, const char* key, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}
int arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    return def;
}

Common::Entity::TimeStamp tsFromNs(std::int64_t ns) {
    return Common::Entity::TimeStamp(
        std::chrono::duration_cast<Common::Entity::TimeDuration>(std::chrono::nanoseconds(ns)));
}

// Extract {x,y,z,roll,pitch,yaw} from a row-major body->world T[16], inverting the
// R = Rz(yaw)Ry(pitch)Rx(roll) extrinsic-XYZ convention iso_from_pose6 builds. (Gimbal lock
// is not special-cased -- fine for a demo trajectory.)
void pose6FromMatrix(const double T[16], double pose6[6]) {
    pose6[0] = T[3];   pose6[1] = T[7];   pose6[2] = T[11];          // translation
    const double r00 = T[0], r10 = T[4], r20 = T[8], r21 = T[9], r22 = T[10];
    pose6[4] = std::asin(std::max(-1.0, std::min(1.0, -r20)));       // pitch
    pose6[3] = std::atan2(r21, r22);                                 // roll
    pose6[5] = std::atan2(r10, r00);                                 // yaw
}

Common::Entity::Image toEntityImage(const DogImage& di, std::int64_t t_ns, int sensorId) {
    Common::Entity::Image img;                                       // ctor sets timestamp=now
    img.timestamp = tsFromNs(t_ns);
    img.sensorId  = sensorId;
    img.height    = di.h;
    img.width     = di.w;
    img.encoding  = ImageEncoding::RGB8;                             // DogImage is always RGB8
    img.step      = di.w * 3;
    img.data      = di.rgb;                                          // copy row-major RGB bytes
    return img;
}

Common::Entity::PointCloud toEntityCloud(std::int64_t t_ns,
                                         const std::vector<float>& xyz_body,
                                         const std::vector<unsigned char>& intensity) {
    Common::Entity::PointCloud pc;
    pc.timestamp        = tsFromNs(t_ns);
    pc.hardwareTimestamp = pc.timestamp;
    pc.sourceSegmentId  = 0;
    const int n = static_cast<int>(intensity.size());
    pc.pointCloudData.reserve(n);
    for (int i = 0; i < n; ++i) {
        Common::Entity::ProcessedPoint p;                           // ctor: label=UNLABELED
        p.x = xyz_body[3 * i + 0];
        p.y = xyz_body[3 * i + 1];
        p.z = xyz_body[3 * i + 2];
        p.intensity = intensity[i];
        pc.pointCloudData.push_back(p);
    }
    return pc;
}

}  // namespace

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--config")) {
        std::fprintf(stderr,
            "usage: %s --config run.yaml [--log <folder>] [--start N] [--count N] [--stride N]\n"
            "  (viz is controlled by run.yaml's viz_endpoint; start viz_server with --out\n"
            "   run.rrd to persist the .rrd)\n", argv[0]);
        return 2;
    }
    const std::string config_path = arg_str(argc, argv, "--config", "");

    try {
        RunConfig cfg = loadRunConfig(config_path);
        cfg.log      = arg_str(argc, argv, "--log", cfg.log);
        cfg.p.start  = arg_int(argc, argv, "--start",  cfg.p.start);
        cfg.p.count  = arg_int(argc, argv, "--count",  cfg.p.count);
        cfg.p.stride = arg_int(argc, argv, "--stride", cfg.p.stride);

        DogLogReader reader(cfg.log + "/PoseAndPointClouds.bin");
        reader.openCamera(cfg.log + "/camera" + std::to_string(cfg.cam) + ".gclf");
        std::printf("[demo] scans=%d images=%d | cam %d | replaying through runOnInput\n",
                    reader.size(), reader.numImages(), cfg.cam);
        if (reader.size() == 0 || reader.numImages() == 0) {
            std::fprintf(stderr, "[demo] empty log\n");
            return 1;
        }

        // Pre-pass: buffer every scan pose so any image timestamp can be bracketed offline.
        PoseInterpolator interp(static_cast<std::size_t>(reader.size()) + 4);
        std::vector<std::int64_t> scan_ts(reader.size());
        {
            double pose6[6];
            std::vector<float>         xyz;
            std::vector<unsigned char> inten;
            for (int i = 0; i < reader.size(); ++i) {
                reader.scanRaw(i, pose6, xyz, inten);               // decode (cloud discarded)
                scan_ts[i] = reader.timestamp(i);
                interp.push(scan_ts[i], pose6);
            }
        }

        // The SemanticRunner owns the entire pipeline; runOnInput is the only call we make.
        semantic::SemanticRunner runner(config_path);

        const semantic::Params& p = cfg.p;
        const int start  = std::max(0, std::min(p.start, reader.numImages() - 1));
        const int stride = p.stride > 0 ? p.stride : 1;
        int processed = 0, skipped = 0;
        std::size_t last_objects = 0;

        double pose6[6], scan_pose6[6];
        std::vector<float>         xyz_body;
        std::vector<unsigned char> inten;
        for (int k = start; k < reader.numImages(); k += stride) {
            if (p.count > 0 && processed >= p.count) break;
            const std::int64_t t = reader.imageTimestamp(k);

            double T[16];
            if (interp.sample(t, T) != PoseInterpolator::Status::OK) { ++skipped; continue; }
            pose6FromMatrix(T, pose6);                               // interpolated body pose

            // Nearest scan (body frame) to the image time -> the paired point cloud. scanRaw's
            // own pose is discarded (scan_pose6): the pose we hand over is the interpolated one.
            auto lo = std::lower_bound(scan_ts.begin(), scan_ts.end(), t);
            int s = static_cast<int>(lo - scan_ts.begin());
            if (s >= reader.size()) s = reader.size() - 1;
            if (s > 0 && (t - scan_ts[s - 1]) < (scan_ts[s] - t)) --s;
            reader.scanRaw(s, scan_pose6, xyz_body, inten);

            semantic::OpenVocabSegInput in;
            in.images.push_back(toEntityImage(reader.imageByIndex(k), t, cfg.cam));
            in.pointCloud = toEntityCloud(scan_ts[s], xyz_body, inten);
            in.pose.timestamp = tsFromNs(t);
            in.pose.x = pose6[0]; in.pose.y = pose6[1]; in.pose.z = static_cast<float>(pose6[2]);
            in.pose.roll  = static_cast<float>(pose6[3]);
            in.pose.pitch = static_cast<float>(pose6[4]);
            in.pose.yaw   = static_cast<float>(pose6[5]);

            std::vector<Common::Entity::EAIRoomObject> objs = runner.runOnInput(std::move(in));
            last_objects = objs.size();
            ++processed;
        }

        // Null-input contract: a null optional is a no-op that returns {}.
        if (!runner.runOnInput(std::nullopt).empty()) {
            std::fprintf(stderr, "[demo] FAIL: runOnInput(nullopt) returned non-empty\n");
            return 1;
        }

        std::printf("[demo] done: %d frames processed, %d skipped (unbracketed) | "
                    "last frame reported %zu objects | null-input check OK\n",
                    processed, skipped, last_objects);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[demo] error: %s\n", e.what());
        return 1;
    }
    return 0;
}
