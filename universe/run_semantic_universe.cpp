// Accumulate a ~/dog_logs recording into a Universe AND assign every world point a
// semantic class, by lifting the inf_server's 2D panoptic labels through the
// point-to-pixel mapping and majority-voting across the frames that see each point.
// Optionally also computes VCCS superpoints, per-point Mask3D features, and per-class
// object seeds (whole-map HDBSCAN*). All parameters come from a YAML config.
//
// Usage:
//   run_semantic_universe --config run.yaml [--log <folder>] [--out labeled.pcd]
//
// Requires the inf_server running (and, if features/enabled, the mask3d_feat server).
// See run.yaml for every tunable; the stored semantics key on class *names*, so
// changing the vocabulary never corrupts the map.

#include "universe.h"
#include "superpoints.h"
#include "point_features.h"
#include "object_seeds.h"
#include "run_config.h"
#include "dog_log_adaptor.h"
#include "dog_stream.h"
#include "dog_log_ingestor.h"
#include "inf_client.h"
#include "feat_client.h"
#include "semantic_pipeline.h"
#include "online_semantic.h"

#include <pcl/io/pcd_io.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>

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

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--config")) {
        std::fprintf(stderr,
            "usage: %s --config run.yaml [--log <folder>] [--out labeled.pcd]\n",
            argv[0]);
        return 2;
    }
    RunConfig cfg;
    try {
        cfg = loadRunConfig(arg_str(argc, argv, "--config", ""));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config error: %s\n", e.what());
        return 2;
    }
    cfg.log = arg_str(argc, argv, "--log", cfg.log);   // CLI overrides
    std::string out = arg_str(argc, argv, "--out", ""); // labeled PCD (optional)
    semantic::Params& p = cfg.p;

    try {
        CameraCalib cam = loadCameraCalib(cfg.calib, cfg.cam);
        DogLogReader reader(cfg.log + "/PoseAndPointClouds.bin");
        reader.openCamera(cfg.log + "/camera" + std::to_string(cfg.cam) + ".gclf");
        std::printf("scans=%d images=%d | cam %d %dx%d | voxel %.3f | radius %.2f | "
                    "tau %.2f | splat %d | gate votes>=%d conf>=%.2f\n",
                    reader.size(), reader.numImages(), cam.sensor_id, cam.width, cam.height,
                    p.voxel, p.radius, p.tau, p.splat, p.min_votes, p.min_conf);
        if (p.superpoints)
            std::printf("superpoints: VCCS seed=%.2f refine_every=%d frames | radius=%.2f | "
                        "thing_frac=%.2f (ephemeral: list = latest window only)\n",
                        p.sp.seed_res, p.refine_every,
                        p.sp_radius > 0.0f ? p.sp_radius : p.radius, p.sp.thing_frac);
        if (p.features)
            std::printf("features: mask3d_feat refine_every=%d frames | radius=%.2f | "
                        "voxel=%.3f | endpoint=%s\n", p.refine_every,
                        p.feat_radius > 0.0f ? p.feat_radius : p.radius, p.feat.voxel,
                        cfg.feat_endpoint.c_str());
        if (p.grow)
            std::printf("growing: affinity>=%.2f | max_dispersion=%.2f | cand_radius=%.2f | "
                        "require_class=%d\n", p.grow_p.affinity_thresh,
                        p.grow_p.max_dispersion, p.grow_p.cand_radius,
                        (int)p.grow_p.require_class);
        if (p.hdbscan)
            std::printf("object seeds: whole-map HDBSCAN* every=%d frames | min_pts=%d | "
                        "min_cluster_size=%d | min_class_points=%d\n", p.hdbscan_every,
                        p.hdb.min_pts, p.hdb.min_cluster_size, p.hdb.min_class_points);
        if (reader.size() == 0) return 1;

        InfClient inf(p.endpoint);
        if (!inf.ping()) { std::fprintf(stderr, "inf_server not responding at %s\n",
                                        p.endpoint.c_str()); return 1; }

        std::unique_ptr<FeatClient> feat;
        if (p.features) {
            feat.reset(new FeatClient(cfg.feat_endpoint));
            if (!feat->ping()) { std::fprintf(stderr, "mask3d_feat not responding at %s\n",
                                              cfg.feat_endpoint.c_str()); return 1; }
        }

        Universe uni(p.voxel);
        uni.setLocalRadius(p.radius);
        Superpoints sp;
        PointFeatures pf;
        ObjectSeeds os;

        // Build the pipeline: OnlineSemantic consumes time-aligned SyncedFrames from
        // DogStream (pose interpolated to each image timestamp). The offline
        // DogLogIngestor feeds it through the SAME pushScan/pushImage interface a live
        // driver would use. Objects (the primary output) are reported as discovered.
        semantic::OnlineSemantic online(uni, inf, p, &sp,
                                        p.features ? &pf : nullptr, feat.get(),
                                        p.hdbscan ? &os : nullptr);
        online.setObjectsHook([&](const std::vector<ObjectSeed>& objs, const Universe& u) {
            std::size_t pts = 0;
            for (const ObjectSeed& s : objs) pts += s.points.size();
            std::fprintf(stderr, "[objects] %zu objects over %zu map points\n",
                         objs.size(), pts);
        });

        DogStream stream(cam, online.hook());
        online.begin();
        DogLogIngestor ingestor(reader);
        // p.start/p.count index IMAGES (5 Hz), not scans; each image's body pose is
        // interpolated to its capture time. p.stride sub-samples the image window
        // (every stride-th image is processed; scans stay dense for interpolation).
        ingestor.run(stream, p.start, p.count > 0 ? p.count : -1, p.stride);
        online.finish();
        const int frames = online.frames();

        std::printf("labeled %d frames | world=%d points | vocab=%d classes seen\n",
                    frames, uni.size(), uni.semantics().size());
        std::printf("sync: images pushed=%ld synced=%ld dropped(too_old=%ld overflow=%ld)\n",
                    ingestor.imagesPushed(), stream.emitted(),
                    stream.droppedTooOld(), stream.droppedOverflow());

        // Per-class point tally over the whole world (argmax of each point's votes),
        // split by thing/stuff so the object-vs-region breakdown is visible.
        std::map<std::string, int> tally;
        int labeled = 0, n_thing = 0, n_stuff = 0;
        for (int i = 0; i < uni.size(); ++i) {
            int cid = uni.pointClassId(i);
            if (cid < 0) { tally["<unlabeled>"]++; continue; }
            tally[uni.semantics().name(cid)]++;
            ++labeled;
            if (uni.pointIsThing(i)) ++n_thing; else if (uni.pointIsStuff(i)) ++n_stuff;
        }
        std::printf("per-class point counts (%d/%d labeled, %.1f%% | %d thing, %d stuff):\n",
                    labeled, uni.size(), uni.size() ? 100.0 * labeled / uni.size() : 0.0,
                    n_thing, n_stuff);
        for (const auto& kv : tally) {
            const int cid = uni.semantics().id(kv.first);
            const ClassKind k = uni.semantics().kind(cid);
            const char* tag = k == ClassKind::Thing ? "thing"
                            : k == ClassKind::Stuff ? "stuff" : "-";
            std::printf("  %-14s %8d  [%s]\n", kv.first.c_str(), kv.second, tag);
        }

        // Superpoint (VCCS oversegmentation) tally: reflects only the LAST window.
        if (p.superpoints) {
            std::map<std::string, int> sp_tally;
            int n_thing_sp = 0, n_unknown_sp = 0;
            for (const Superpoint& s : sp.list()) {
                if (s.kind == ClassKind::Thing && s.class_id >= 0) {
                    ++n_thing_sp;
                    sp_tally[uni.semantics().name(s.class_id)]++;
                } else {
                    ++n_unknown_sp;
                }
            }
            std::printf("superpoints: %d total | %d thing, %d unknown\n",
                        sp.size(), n_thing_sp, n_unknown_sp);
            for (const auto& kv : sp_tally)
                std::printf("  %-14s %8d  [thing]\n", kv.first.c_str(), kv.second);
        }

        // Mask3D feature coverage: how many world points got a 96-d feature vector.
        if (p.features)
            std::printf("features: %d/%d pts (%.1f%%) got a %d-d vector\n",
                        pf.covered(), uni.size(),
                        uni.size() ? 100.0 * pf.covered() / uni.size() : 0.0, pf.dim());

        // Object seed tally: #instances discovered per thing class (whole-map, strict).
        if (p.hdbscan) {
            std::map<int, std::pair<int, int>> per;    // class id -> (#objects, #points)
            for (const ObjectSeed& s : os.list()) {
                auto& e = per[s.class_id];
                e.first += 1;
                e.second += (int)s.points.size();
            }
            std::printf("object seeds: %d instances over %d classes\n",
                        os.size(), (int)per.size());
            for (const auto& kv : per)
                std::printf("  %-14s %3d objects, %8d pts\n",
                            uni.semantics().name(kv.first).c_str(),
                            kv.second.first, kv.second.second);
        }

        if (!out.empty()) {
            pcl::io::savePCDFileBinary(out, *uni.labeledCloud());
            std::printf("wrote labeled cloud (PointXYZL, label=classid+1, 0=unlabeled) -> %s\n",
                        out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
