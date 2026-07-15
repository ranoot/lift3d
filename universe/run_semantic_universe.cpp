// Accumulate a ~/dog_logs recording into a Universe AND assign every world point a
// semantic class, by lifting the inf_server's 2D panoptic labels through the
// point-to-pixel mapping and majority-voting across the frames that see each point.
// Optionally also computes VCCS superpoints and per-class object seeds
// (whole-map HDBSCAN*). All parameters come from a YAML config.
//
// Usage:
//   run_semantic_universe --config run.yaml [--log <folder>] [--out labeled.pcd]
//
// Requires the inf_server running. See run.yaml for every tunable; the stored semantics
// key on class *names*, so changing the vocabulary never corrupts the map.

#include "universe.h"
#include "superpoints.h"
#include "object_seeds.h"
#include "objects.h"
#include "run_config.h"
#include "dog_log_adaptor.h"
#include "log_frame_source.h"   // LogFrameSource (offline replay; owns the DogStream engine)
#include "inf_client.h"
#include "semantic_pipeline.h"
#include "online_semantic.h"
#include "object_publisher.h"   // gmd:: EAIRoomObject egress (2D overhead hull + keyed topic)
#ifdef LIFT3D_HAVE_SIGN
#include "sign_bridge.h"        // sign:: async sign-text egress (LIFT3D_USE_SIGN builds only)
#endif

#include <pcl/io/pcd_io.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
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
                    "tau %.2f | splat %d\n",
                    reader.size(), reader.numImages(), cam.sensor_id, cam.width, cam.height,
                    p.voxel, p.radius, p.tau, p.splat);
        if (p.superpoints)
            std::printf("superpoints: VCCS seed=%.2f per-frame (FULL view incl. stuff) | "
                        "thing_frac=%.2f\n", p.sp.seed_res, p.sp.thing_frac);
        if (p.grow)
            std::printf("growing (superpoints->proposals): cosine(class hist)*containment "
                        ">=%.2f | require_class=%d\n", p.grow_p.affinity_thresh,
                        (int)p.grow_p.require_class);
        if (p.objects) {
            std::printf("objects (proposals->objects): seed Union-Find | min_merges=%d | "
                        "consol_stride=%d | require_class=%d | merge_thresh=[",
                        p.consol_p.min_merges, p.consol_stride,
                        (int)p.consol_p.require_class);
            for (std::size_t i = 0; i < p.consol_p.merge_thresh.size(); ++i)
                std::printf("%s%.2f", i ? "," : "", p.consol_p.merge_thresh[i]);
            std::printf("]\n");
        }
        if (p.hdbscan)
            std::printf("object proposals: per-frame seeding (visible thing set) | "
                        "min_pts=%d | min_cluster_size=%d | min_class_points=%d | "
                        "select=%s | sor=%s(k=%d,s=%.1f) | instance_ids=%s\n",
                        p.hdb.min_pts, p.hdb.min_cluster_size, p.hdb.min_class_points,
                        p.hdb.leaf_selection ? "leaf" : "eom",
                        p.hdb.sor_enabled ? "on" : "off", p.hdb.sor_mean_k, p.hdb.sor_std_mul,
                        p.hdb.use_instance_ids ? "on (bridged DVIS instances)" : "off (geometry)");
        if (reader.size() == 0) return 1;

        // OnlineSemantic OWNS the pipeline (Universe/InfClient/Superpoints/ObjectSeeds/
        // Objects), built from Params. The offline LogFrameSource feeds it time-aligned
        // SyncedFrames through the SAME interface a live GmdFrameSource would use. Objects
        // (the primary output) are reported as discovered.
        semantic::OnlineSemantic online(p);
        if (!online.inf().ping()) { std::fprintf(stderr, "inf_server not responding at %s\n",
                                                 p.endpoint.c_str()); return 1; }

        // Bind live references to the owned stages for the post-run reads / hook below.
        Universe&    uni = online.universe();
        Superpoints& sp  = online.superpoints();
        ObjectSeeds& os  = online.seeds();
        Objects&     obj = online.objects();
        // The objects hook returns the EAIRoomObjects to upsert into the global store: each
        // universe Object becomes an EAIRoomObject whose polygon is a 2D overhead (X-Y, world is
        // Z-up) convex hull of its member points, cleaned with a kNN statistical-outlier pass.
        // Writing an objectId that already exists overwrites it (updated); a new id is created.
        // NOTE: Object::id is a Union-Find root that can change when components merge, so id
        // stability is bounded by that root; a persistent id would come from the instance-id layer.
        const auto vehicle_id = static_cast<uint16_t>(cfg.cam);
        // `dir` (optional): object id -> directionContent from resolved sign text. When present,
        // an object that encompasses a sign carries its text; others stay "". Null in a non-sign
        // build, so directionContent is unchanged there.
        auto collectEAIObjects = [](const std::vector<Object>& objs, const Universe& u,
                                    uint16_t vehicleId,
                                    const std::unordered_map<int, std::string>* dir) {
            std::vector<gmd::EAIRoomObject> updates;
            Universe::Cloud::ConstPtr cloud = u.cloud();
            std::vector<std::array<float, 3>> xyz;
            for (const Object& o : objs) {
                xyz.clear();
                xyz.reserve(o.points.size());
                for (int gid : o.points) {
                    if (gid < 0 || gid >= static_cast<int>(cloud->size())) continue;
                    if (!u.pointAlive(gid)) continue;
                    const Universe::PointT& pt = (*cloud)[gid];
                    xyz.push_back({pt.x, pt.y, pt.z});
                }
                if (xyz.empty()) continue;
                updates.push_back(gmd::buildRoomObject(vehicleId, o.id,
                                                       u.semantics().name(o.class_id), xyz));
                if (dir) {
                    auto it = dir->find(o.id);
                    if (it != dir->end()) updates.back().directionContent = it->second;
                }
            }
            return updates;
        };

        // Sign-text egress (LIFT3D_USE_SIGN builds only): dispatch detected signs to the async
        // detector and, at publish time, fold resolved text into directionContent. Off => `dir`
        // stays null below and nothing sign-related is compiled.
#ifdef LIFT3D_HAVE_SIGN
        sign::SignBridge sign_bridge{sign::SignBridge::Config{}};
        online.setSignHook([&sign_bridge, &online, &p](
                const SyncedFrame& sf, const FrameResult& fr,
                const std::vector<int>& pix_gidx, int W, int H, const Universe& u) {
            sign_bridge.onFrame(sf, fr, pix_gidx, W, H, u, online.inf(), p.voxel);
        });
#endif

        gmd::ObjectDelta delta;  // remembers what was already emitted so we only send the changes
        online.setObjectsHook([&, vehicle_id](const std::vector<Object>& objs, const Universe& u) {
            const std::unordered_map<int, std::string>* dir = nullptr;
#ifdef LIFT3D_HAVE_SIGN
            std::unordered_map<int, std::string> sign_dir = sign_bridge.annotate(objs, u, p.voxel);
            dir = &sign_dir;
#endif
            std::vector<gmd::EAIRoomObject> all = collectEAIObjects(objs, u, vehicle_id, dir);
            std::vector<gmd::EAIRoomObject> updates = delta.changedSince(all);  // new or changed only
            // Hand `updates` to the global object store (same objectId overwrites).
            for (const gmd::EAIRoomObject& o : updates)
                std::fprintf(stderr,
                             "[objects] upsert id=(veh %u) label=%s hull=%zu verts "
                             "centroid=(%.2f,%.2f) dir=\"%s\"\n",
                             o.objectId.vehicleId, o.label.c_str(), o.polygon.size(),
                             o.centroid.x, o.centroid.y, o.directionContent.c_str());
        });

        LogFrameSource src(reader, cam);
        online.attach(src);
        online.begin();
        // p.start/p.count index IMAGES (5 Hz), not scans; each image's body pose is
        // interpolated to its capture time. p.stride sub-samples the image window
        // (every stride-th image is processed; scans stay dense for interpolation).
        src.run(p.start, p.count > 0 ? p.count : -1, p.stride);
        online.finish();
        const int frames = online.frames();

        std::printf("labeled %d frames | world=%d points (%d live, %d tombstoned dynamic) "
                    "| vocab=%d classes seen\n",
                    frames, uni.size(), uni.aliveCount(), uni.size() - uni.aliveCount(),
                    uni.semantics().size());
        std::printf("sync: images pushed=%ld synced=%ld dropped(too_old=%ld overflow=%ld)\n",
                    src.imagesPushed(), src.emitted(),
                    src.droppedTooOld(), src.droppedOverflow());

        // Per-class point tally over the whole world (argmax of each point's votes),
        // split by thing/stuff so the object-vs-region breakdown is visible.
        std::map<std::string, int> tally;
        int labeled = 0, n_thing = 0, n_stuff = 0;
        for (int i = 0; i < uni.size(); ++i) {
            if (!uni.pointAlive(i)) continue;             // tombstoned (dynamic) -> skip
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

        // Object seed tally: #seed clusters discovered per thing class (tier 2, strict).
        if (p.hdbscan) {
            std::map<int, std::pair<int, int>> per;    // class id -> (#seeds, #points)
            for (const ObjectSeed& s : os.list()) {
                auto& e = per[s.class_id];
                e.first += 1;
                e.second += (int)s.points.size();
            }
            std::printf("object seeds (tier 2): %d instances over %d classes\n",
                        os.size(), (int)per.size());
            for (const auto& kv : per)
                std::printf("  %-14s %3d seeds, %8d pts\n",
                            uni.semantics().name(kv.first).c_str(),
                            kv.second.first, kv.second.second);
        }

        // Objects tally (tier 3, the primary output): virtual Union-Find components promoted
        // past min_merges, per class. Far fewer than seedCount() -- most seeds are the
        // repeated per-view observations that merged into the same component.
        if (p.objects) {
            std::map<int, std::pair<int, int>> per;    // class id -> (#objects, #points)
            for (const Object& o : obj.list()) {
                auto& e = per[o.class_id];
                e.first += 1;
                e.second += (int)o.points.size();
            }
            std::printf("objects (tier 3): %d promoted over %d classes (from %d seeds)\n",
                        obj.size(), (int)per.size(), obj.seedCount());
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
