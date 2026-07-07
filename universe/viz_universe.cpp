// Rerun visualizer for the Universe map.
//
// Replays a ~/dog_logs recording, integrating each frame into a Universe, and logs
// to a Rerun recording on the capture timeline:
//   - world/map/<i>    : the world map, built up NON-DESTRUCTIVELY. Each frame logs
//                        only the points that voxel became NEW that frame (the tail
//                        of uni.cloud(), since integrate() appends new voxels and
//                        refines existing ones in place) to its own per-frame entity,
//                        coloured by lidar intensity. Because each entity is logged
//                        once and never overwritten, rerun's latest-at query unions
//                        them: the cloud GROWS as the time cursor advances and shrinks
//                        as it moves back -- but a point, once added, is never dropped.
//                        This mirrors a live robot streaming each scan's new returns
//                        once, rather than re-sending the whole map.
//                        (A DESTRUCTIVE full-snapshot view -- needed once the
//                        superpoint/segmenter layer can relabel or remove points --
//                        is deliberately left for that layer; see run_universe.)
//   - world/robot      : robot body pose (axes) per frame.
//   - world/trajectory : growing polyline of the robot path.
//   - world/camera     : camera pose + a geometry-only Pinhole frustum (no image
//                        pixels exist in this codebase).
//
// Usage:
//   viz_universe --log <folder> [--calib xml] [--cam 211] [--start 0] [--count 50]
//                [--voxel 0.05] [--stride 1] [--out universe.rrd] [--spawn]
//
//   --out saves a .rrd (default sink); --spawn opens the live viewer.

#include "universe.h"
#include "dog_log_adaptor.h"

#include <rerun.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string arg_str(int argc, char** argv, const char* key, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

static Universe::Cloud::Ptr frameToCloud(const DogFrame& f) {
    Universe::Cloud::Ptr c(new Universe::Cloud);
    const int N = (int)(f.points_world.size() / 3);
    c->reserve(N);
    for (int i = 0; i < N; ++i) {
        pcl::PointXYZI p;
        p.x = f.points_world[3 * i + 0];
        p.y = f.points_world[3 * i + 1];
        p.z = f.points_world[3 * i + 2];
        p.intensity = (i < (int)f.intensity.size()) ? f.intensity[i] : 0.0f;
        c->push_back(p);
    }
    return c;
}

// Lidar intensity (0..255) -> Turbo colormap RGB. Turbo is perceptually ordered and
// far more legible than greyscale for scalar fields. Polynomial fit (Mikhailov),
// inputs/outputs normalised to [0,1].
static rerun::Color intensityToColor(float intensity) {
    float t = intensity / 255.0f;
    t = std::min(1.0f, std::max(0.0f, t));
    const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    float r = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 + 132.13108234f * t3
              - 152.94239396f * t4 + 59.28637943f * t5;
    float g = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 - 14.18503333f * t3
              + 4.27729857f * t4 + 2.82956604f * t5;
    float b = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 + 110.36276771f * t3
              - 89.90310912f * t4 + 27.34824973f * t5;
    auto u8 = [](float v) -> uint8_t {
        v = std::min(1.0f, std::max(0.0f, v));
        return (uint8_t)(v * 255.0f + 0.5f);
    };
    return rerun::Color(u8(r), u8(g), u8(b));
}

// Row-major rigid 4x4 (double or float) -> rerun Transform3D (translation + R).
// rerun takes the rotation as 3 column vectors, sidestepping row/col-major ambiguity.
template <typename T>
static rerun::Transform3D rigidToTransform(const T* M) {
    const rerun::datatypes::Vec3D cols[3] = {
        {(float)M[0], (float)M[4], (float)M[8]},   // column 0
        {(float)M[1], (float)M[5], (float)M[9]},   // column 1
        {(float)M[2], (float)M[6], (float)M[10]},  // column 2
    };
    return rerun::Transform3D::from_translation_mat3x3(
        {(float)M[3], (float)M[7], (float)M[11]}, cols);
}

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--log")) {
        std::fprintf(stderr,
            "usage: %s --log <folder> [--calib xml] [--cam 211] [--start 0] "
            "[--count 50] [--voxel 0.05] [--stride 1] "
            "[--out universe.rrd] [--spawn]\n",
            argv[0]);
        return 2;
    }
    std::string log    = arg_str(argc, argv, "--log", "");
    std::string calib  = arg_str(argc, argv, "--calib", log + "/../CameraCalRaiboAsQUGV113.xml");
    int   cam_id  = std::atoi(arg_str(argc, argv, "--cam", "211").c_str());
    int   start   = std::atoi(arg_str(argc, argv, "--start", "0").c_str());
    int   count   = std::atoi(arg_str(argc, argv, "--count", "50").c_str());
    float voxel   = (float)std::atof(arg_str(argc, argv, "--voxel", "0.05").c_str());
    int   stride  = std::max(1, std::atoi(arg_str(argc, argv, "--stride", "1").c_str()));
    bool  spawn   = has_flag(argc, argv, "--spawn");
    std::string out = arg_str(argc, argv, "--out", "universe.rrd");

    try {
        CameraCalib cam = loadCameraCalib(calib, cam_id);
        DogLogReader reader(log + "/PoseAndPointClouds.bin");
        if (reader.size() == 0) { std::fprintf(stderr, "empty recording\n"); return 1; }

        Universe uni(voxel);

        // Open the recording sink: live viewer (--spawn) or a .rrd file (default).
        rerun::RecordingStream rec("lift3d/universe");
        if (spawn) rec.spawn().exit_on_failure();
        else       rec.save(out).exit_on_failure();

        // World axes orientation (lidar world is Z-up, right-handed).
        rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

        const int end = std::min(reader.size(), start + count);
        const std::int64_t t0 = reader.timestamp(start);
        std::vector<rerun::datatypes::Vec3D> traj;
        int logged = 0;
        size_t prev = 0;  // # voxels already logged; the new tail is the per-frame delta

        for (int i = start; i < end; i += stride) {
            DogFrame f = reader.frame(i, cam);
            Camera   c = reader.camera(f, cam);

            const float robot[3] = { (float)f.T_world_body[3],
                                     (float)f.T_world_body[7],
                                     (float)f.T_world_body[11] };
            uni.integrate(*frameToCloud(f), c, f.t_ns, "frame" + std::to_string(i), robot);

            // Timeline: a frame index and the real capture time (s from first frame).
            rec.set_time_sequence("frame", i);
            rec.set_time_duration_secs("capture", (double)(f.t_ns - t0) * 1e-9);

            // Non-destructive accumulation: log ONLY the voxels that became new this
            // frame (the tail of the accumulator) to a per-frame entity. Latest-at
            // unions them, so the map grows with the cursor and a point, once added, is
            // never overwritten. Coloured by intensity (Turbo).
            Universe::Cloud::ConstPtr snap = uni.cloud();
            const size_t cur = snap->size();
            if (cur > prev) {
                std::vector<rerun::Position3D> pos;
                std::vector<rerun::Color>      col;
                pos.reserve(cur - prev);
                col.reserve(cur - prev);
                for (size_t k = prev; k < cur; ++k) {
                    const Universe::PointT& p = snap->points[k];
                    pos.emplace_back(p.x, p.y, p.z);
                    col.push_back(intensityToColor(p.intensity));
                }
                char path[32];
                std::snprintf(path, sizeof(path), "world/map/%06d", i);
                rec.log(path,
                        rerun::Points3D(pos).with_colors(col).with_radii(0.02f));
            }
            prev = cur;

            // Robot pose (axes) + growing trajectory polyline.
            rec.log("world/robot", rigidToTransform(f.T_world_body));
            traj.push_back({robot[0], robot[1], robot[2]});
            rec.log("world/trajectory", rerun::LineStrips3D(rerun::LineStrip3D(traj)));

            // Camera pose + geometry-only frustum.
            rec.log("world/camera", rigidToTransform(c.T));
            rec.log("world/camera",
                    rerun::Pinhole::from_focal_length_and_resolution(
                        {c.K[0], c.K[4]}, {(float)c.width, (float)c.height})
                        .with_image_plane_distance(0.4f));
            ++logged;
        }

        std::printf("logged %d frames | world=%d points | sink=%s\n",
                    logged, uni.size(), spawn ? "spawn" : out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
