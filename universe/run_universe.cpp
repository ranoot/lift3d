// Accumulate a ~/dog_logs recording into a Universe and exercise the
// point-to-pixel bridge.
//
// Usage:
//   run_universe --log <folder> [--calib <xml>] [--cam 211]
//                [--start 0] [--count 50] [--voxel 0.05] [--radius 0]
//                [--tau 0.1] [--out map.pcd]
//
// Builds a PointXYZI cloud per frame from the lifted lidar return, fuses it into
// the persistent whole-world map (voxel-deduplicated, nothing deleted), then
// projects into the last frame's camera. --radius is a LOCAL operating crop: it
// bounds what the projection reads around the robot, it does NOT prune the world.

#include "universe.h"
#include "dog_log_adaptor.h"

#include <pcl/io/pcd_io.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--log")) {
        std::fprintf(stderr,
            "usage: %s --log <folder> [--calib xml] [--cam 211] [--start 0] "
            "[--count 50] [--voxel 0.05] [--radius 0] [--tau 0.1] [--out map.pcd]\n",
            argv[0]);
        return 2;
    }
    std::string log   = arg_str(argc, argv, "--log", "");
    std::string calib = arg_str(argc, argv, "--calib", log + "/../CameraCalRaiboAsQUGV113.xml");
    int   cam_id  = std::atoi(arg_str(argc, argv, "--cam", "211").c_str());
    int   start   = std::atoi(arg_str(argc, argv, "--start", "0").c_str());
    int   count   = std::atoi(arg_str(argc, argv, "--count", "50").c_str());
    float voxel   = (float)std::atof(arg_str(argc, argv, "--voxel", "0.05").c_str());
    float radius  = (float)std::atof(arg_str(argc, argv, "--radius", "0").c_str());
    float tau     = (float)std::atof(arg_str(argc, argv, "--tau", "0.1").c_str());
    std::string out = arg_str(argc, argv, "--out", "");

    try {
        CameraCalib cam = loadCameraCalib(calib, cam_id);
        DogLogReader reader(log + "/PoseAndPointClouds.bin");
        std::printf("scan file: %d frames | cam %d %dx%d | voxel %.3f m | radius %.2f m\n",
                    reader.size(), cam.sensor_id, cam.width, cam.height, voxel,
                    radius > 0 ? radius : 0.0f);
        if (reader.size() == 0) return 1;

        Universe uni(voxel);
        uni.setLocalRadius(radius);   // 0 => local ops see the whole world

        const int end = std::min(reader.size(), start + count);
        long raw_total = 0;
        for (int i = start; i < end; ++i) {
            DogFrame f = reader.frame(i, cam);
            Camera   c = reader.camera(f, cam);
            Universe::Cloud::Ptr cloud = frameToCloud(f);
            raw_total += (long)cloud->size();
            // robot position = body-frame origin in world (translation of T_world_body)
            float robot[3] = { (float)f.T_world_body[3],
                               (float)f.T_world_body[7],
                               (float)f.T_world_body[11] };
            uni.integrate(*cloud, c, f.t_ns, "frame" + std::to_string(i), robot);
        }

        std::printf("integrated frames [%d,%d): raw points=%ld  world=%d  (%.1f%% after fusion)\n",
                    start, end, raw_total, uni.size(),
                    raw_total ? 100.0 * uni.size() / raw_total : 0.0);

        // Project into the last view. With --radius, project only the LOCAL region
        // around the robot (the world map is untouched); else project the whole world.
        if (uni.numViews() > 0) {
            int last = uni.numViews() - 1;
            std::vector<int> gidx;
            PointPixelMap m = (radius > 0.0f) ? uni.projectLocal(last, tau, radius, &gidx)
                                              : uni.project(last, tau);
            if (!m.ok()) { std::fprintf(stderr, "projection failed\n"); return 1; }
            const int op_n = (radius > 0.0f) ? (int)gidx.size() : uni.size();
            std::printf("project %s -> view %d: %d/%d points visible (%.1f%%)  [world=%d]\n",
                        radius > 0.0f ? "local" : "world", last,
                        m.num_visible, op_n, op_n ? 100.0 * m.num_visible / op_n : 0.0,
                        uni.size());
        }

        if (!out.empty()) {
            pcl::io::savePCDFileBinary(out, *uni.cloud());
            std::printf("wrote accumulated map (%d points) -> %s\n", uni.size(), out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
