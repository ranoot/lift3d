// Rerun visualizer for the SEMANTIC Universe: replays a ~/dog_logs recording,
// fuses it into a Universe, lifts inf_server 2D labels onto every world point, and
// logs the world coloured by semantic class (plus robot pose, trajectory, camera
// pose + frustum, and the actual RGB frame the labels came from).
//
// Usage:
//   viz_semantic_universe --config run.yaml [--log <folder>] [--out sem_universe.rrd]
//       [--spawn]
// All parameters live in the YAML config (see run.yaml); the flags above are just
// convenience overrides.
//
// Requires the inf_server running. The final segmented world map (and, with
// --show-features, the per-point Mask3D feature colouring) is logged ONCE as a
// static snapshot (always visible at any time cursor); only the lightweight robot
// pose / trajectory / camera / RGB frame vary per frame. This keeps the recording
// small -- re-logging the whole world every frame made it O(frames x map_size)
// (many GB), which the viewer's memory GC purges mid-load, so points flickered in
// and vanished, leaving an empty scene.
// Points are split by panoptic kind into world/map/{things,stuff,unlabeled} so you
// can hide "stuff" (walls/floor/...) with one toggle in the viewer's entity tree
// and keep just the objects. --things-only skips voting stuff classes entirely.
//
// With --superpoints, the VCCS oversegmentation is EPHEMERAL (one local window per
// stride) and is logged ON THE TIMELINE at the strided frames where it is recomputed
// -- NOT in the static snapshot -- so scrubbing shows only the current window, moving
// with the robot. Under world/superpoints/*: /all colours each point by a palette
// keyed on its per-window superpoint id, /things shows only object superpoints
// coloured by class via the shared AnnotationContext, /centroids marks each object
// superpoint's centre with its class-name label, and /window draws the wireframe crop
// sphere (radius) the window was computed over, at the robot's position that frame.

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

#include <rerun.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

// class_ids are uint16_t, so points with no label (-1) are mapped to this sentinel,
// which the AnnotationContext registers as a grey "unlabeled" class.
static constexpr uint16_t UNLABELED_ID = 0xFFFF;

// Deterministic, high-contrast palette keyed by stable class id (-1 => grey).
static rerun::Color classColor(int cid) {
    static const rerun::Color pal[] = {
        {230, 25, 75}, {60, 180, 75},  {255, 225, 25}, {0, 130, 200},
        {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
        {210, 245, 60}, {250, 190, 190}, {0, 128, 128}, {230, 190, 255},
        {170, 110, 40}, {255, 250, 200}, {128, 0, 0},   {170, 255, 195},
        {128, 128, 0}, {255, 215, 180}, {0, 0, 128},    {128, 128, 128},
    };
    if (cid < 0) return rerun::Color(110, 110, 110);
    return pal[cid % (int)(sizeof(pal) / sizeof(pal[0]))];
}

// Pseudo-random but stable colour keyed on a superpoint id, so adjacent over-
// segments get contrasting colours in the /all oversegmentation view.
static rerun::Color superpointColor(std::uint32_t gid) {
    std::uint32_t h = gid * 2654435761u;               // Knuth multiplicative hash
    h ^= h >> 15;
    return rerun::Color((uint8_t)(40 + (h & 0xFF) % 200),
                        (uint8_t)(40 + ((h >> 8) & 0xFF) % 200),
                        (uint8_t)(40 + ((h >> 16) & 0xFF) % 200));
}

template <typename T>
static rerun::Transform3D rigidToTransform(const T* M) {
    const rerun::datatypes::Vec3D cols[3] = {
        {(float)M[0], (float)M[4], (float)M[8]},
        {(float)M[1], (float)M[5], (float)M[9]},
        {(float)M[2], (float)M[6], (float)M[10]},
    };
    return rerun::Transform3D::from_translation_mat3x3(
        {(float)M[3], (float)M[7], (float)M[11]}, cols);
}

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--config")) {
        std::fprintf(stderr,
            "usage: %s --config run.yaml [--log <folder>] [--out sem_universe.rrd] "
            "[--spawn]\n", argv[0]);
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
    cfg.out = arg_str(argc, argv, "--out", cfg.out);
    if (has_flag(argc, argv, "--spawn")) cfg.spawn = true;

    semantic::Params& p        = cfg.p;
    const std::string& log     = cfg.log;
    const std::string& calib   = cfg.calib;
    const int          cam_id  = cfg.cam;
    const std::string& feat_endpoint = cfg.feat_endpoint;
    const bool         show_features = cfg.show_features;
    const bool         spawn   = cfg.spawn;
    const std::string  out     = cfg.out;

    try {
        CameraCalib cam = loadCameraCalib(calib, cam_id);
        DogLogReader reader(log + "/PoseAndPointClouds.bin");
        reader.openCamera(log + "/camera" + std::to_string(cam_id) + ".gclf");
        if (reader.size() == 0) { std::fprintf(stderr, "empty recording\n"); return 1; }

        InfClient inf(p.endpoint);
        if (!inf.ping()) { std::fprintf(stderr, "inf_server not responding at %s\n",
                                        p.endpoint.c_str()); return 1; }

        std::unique_ptr<FeatClient> feat;
        if (p.features) {
            feat.reset(new FeatClient(feat_endpoint));
            if (!feat->ping()) { std::fprintf(stderr, "mask3d_feat not responding at %s\n",
                                              feat_endpoint.c_str()); return 1; }
        }

        Universe uni(p.voxel);
        uni.setLocalRadius(p.radius);
        Superpoints sp;
        PointFeatures pf;
        ObjectSeeds os;

        rerun::RecordingStream rec("lift3d/semantic_universe");
        if (spawn) rec.spawn().exit_on_failure();
        else       rec.save(out).exit_on_failure();
        rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

        // Timeline origin = the first replayed IMAGE timestamp (frames are now image-
        // driven, one per camera frame, each at its interpolated pose).
        const int start_img = std::min(p.start, reader.numImages() - 1);
        const std::int64_t t0 = reader.imageTimestamp(std::max(0, start_img));
        std::vector<rerun::datatypes::Vec3D> traj;

        // Build the full segmented world + superpoints and log them, either as a
        // STATIC snapshot (as_static=true; logged once, always visible at any time
        // cursor) or at the current time. We use the static form once after the run
        // instead of re-logging the whole world every frame: per-frame re-logging made
        // the recording O(frames x map_size) -- many GB -- which the rerun viewer's
        // memory-limit GC then purges mid-load, so points flicker in and get deleted,
        // leaving an empty scene. Logging the final map once keeps the recording tiny
        // and the result permanently visible.
        auto logWorld = [&](bool as_static) {
            Universe::Cloud::ConstPtr snap = uni.cloud();
            auto emit = [&](const char* path, const auto& arch) {
                if (as_static) rec.log_static(path, arch); else rec.log(path, arch);
            };
            // Route each point to a per-kind entity so "stuff" (walls/floor/...) can be
            // hidden with one toggle in the viewer's entity tree. Colours/labels resolve
            // via the AnnotationContext on "world" (an ancestor of all these paths).
            std::vector<rerun::Position3D>          pos_t, pos_s, pos_u;
            std::vector<rerun::components::ClassId> cid_t, cid_s;
            for (int k = 0; k < (int)snap->size(); ++k) {
                const Universe::PointT& pt = (*snap)[k];
                const rerun::Position3D xyz(pt.x, pt.y, pt.z);
                const int cid = uni.pointClassId(k);
                switch (uni.pointClassKind(k)) {
                    case ClassKind::Thing:
                        pos_t.push_back(xyz); cid_t.emplace_back((uint16_t)cid); break;
                    case ClassKind::Stuff:
                        pos_s.push_back(xyz); cid_s.emplace_back((uint16_t)cid); break;
                    default:                       // unlabeled/gated -> grey
                        pos_u.push_back(xyz); break;
                }
            }
            emit("world/map/things",
                 rerun::Points3D(pos_t).with_class_ids(cid_t).with_radii(0.02f)
                     .with_show_labels(false));
            emit("world/map/stuff",
                 rerun::Points3D(pos_s).with_class_ids(cid_s).with_radii(0.02f)
                     .with_show_labels(false));
            emit("world/map/unlabeled",
                 rerun::Points3D(pos_u)
                     .with_class_ids(std::vector<rerun::components::ClassId>(
                         pos_u.size(), UNLABELED_ID))
                     .with_radii(0.02f).with_show_labels(false));

            // NOTE: superpoints are NOT logged here. They are ephemeral (one local
            // window per stride), so they are logged per-frame on the timeline at the
            // frames where VCCS actually recomputes (see logSuperpointsAt + the hook),
            // not baked into this always-visible static snapshot.

            // Object seeds (per-class whole-map HDBSCAN*): the headline marker is a big
            // class-coloured CENTROID per discovered instance (centroid only -- no
            // bounding box, per the user), plus the member points at a larger radius
            // than superpoints. Whole-map + static => always visible, no GC risk.
            if (p.hdbscan && os.size() > 0) {
                std::vector<rerun::Position3D> ctr, mpos;
                std::vector<rerun::components::ClassId> ctr_cid;
                std::vector<rerun::components::Text>    ctr_lbl;
                std::vector<rerun::Color> mcol;
                for (const ObjectSeed& s : os.list()) {
                    ctr.emplace_back(s.centroid[0], s.centroid[1], s.centroid[2]);
                    ctr_cid.emplace_back((uint16_t)s.class_id);
                    ctr_lbl.emplace_back(uni.semantics().name(s.class_id));
                    // Colour member points by the object's STABLE id, so a grown object
                    // keeps its colour across refreshes (ids never shift; list is append-only).
                    const rerun::Color oc = superpointColor((std::uint32_t)s.id);
                    for (int idx : s.points) {
                        if (idx < 0 || idx >= (int)snap->size()) continue;
                        const Universe::PointT& pt = (*snap)[idx];
                        mpos.emplace_back(pt.x, pt.y, pt.z);
                        mcol.push_back(oc);
                    }
                }
                emit("world/seeds/points",
                     rerun::Points3D(mpos).with_colors(mcol).with_radii(0.04f)
                         .with_show_labels(false));
                emit("world/seeds/centroids",
                     rerun::Points3D(ctr).with_class_ids(ctr_cid).with_labels(ctr_lbl)
                         .with_radii(0.15f));
            }

            // Per-point Mask3D features under world/features: colour each point by a
            // deterministic 96->3 random projection of its feature vector, min-max
            // normalized to RGB. Not semantic -- similar colours ~ similar features, a
            // quick visual check that the per-point embeddings carry object structure.
            if (!show_features || pf.covered() == 0) return;
            std::vector<float> W((std::size_t)pf.dim() * 3);
            std::uint32_t s = 2463534242u;                 // fixed seed -> stable colours
            auto nextw = [&]() {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;   // xorshift32
                return (float)((int)(s & 0x00ffffff) / 8388607.5 - 1.0);   // ~[-1,1]
            };
            for (float& w : W) w = nextw();
            std::vector<rerun::Position3D> fpos;
            std::vector<std::array<float, 3>> proj;
            float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
            for (int k = 0; k < (int)snap->size(); ++k) {
                const float* fv = pf.feature(k);
                if (!fv) continue;
                float pr[3] = {0, 0, 0};
                for (int d = 0; d < pf.dim(); ++d) {
                    pr[0] += fv[d] * W[(std::size_t)d * 3 + 0];
                    pr[1] += fv[d] * W[(std::size_t)d * 3 + 1];
                    pr[2] += fv[d] * W[(std::size_t)d * 3 + 2];
                }
                const Universe::PointT& pt = (*snap)[k];
                fpos.emplace_back(pt.x, pt.y, pt.z);
                proj.push_back({pr[0], pr[1], pr[2]});
                for (int c = 0; c < 3; ++c) { mn[c] = std::min(mn[c], pr[c]); mx[c] = std::max(mx[c], pr[c]); }
            }
            std::vector<rerun::Color> fcol;
            fcol.reserve(proj.size());
            for (const auto& pr : proj) {
                uint8_t rgb[3];
                for (int c = 0; c < 3; ++c) {
                    const float t = (mx[c] > mn[c]) ? (pr[c] - mn[c]) / (mx[c] - mn[c]) : 0.5f;
                    rgb[c] = (uint8_t)(t * 255.0f);
                }
                fcol.emplace_back(rgb[0], rgb[1], rgb[2]);
            }
            emit("world/features",
                 rerun::Points3D(fpos).with_colors(fcol).with_radii(0.02f)
                     .with_show_labels(false));
        };

        // Log the CURRENT ephemeral superpoint window on the timeline (at whatever time
        // the caller has set). Called from the hook ONLY on the strided frames where
        // VCCS just recomputed, so scrubbing shows the superpoints just at those points
        // -- co-located with the robot and the crop radius where they were computed --
        // rather than as one always-on global snapshot. Each call REPLACES the previous
        // window (same entity paths + ephemeral list), so it never accumulates; the
        // window then stays visible until the next recompute.
        const float sp_r = p.sp_radius > 0.0f ? p.sp_radius : p.radius;
        auto logSuperpointsAt = [&](const float robot[3]) {
            Universe::Cloud::ConstPtr snap = uni.cloud();
            std::vector<rerun::Position3D> sp_pos, th_pos, ctr_pos;
            std::vector<rerun::Color>      sp_col;
            std::vector<rerun::components::ClassId> th_cid, ctr_cid;
            std::vector<rerun::components::Text>    ctr_lbl;
            for (const Superpoint& s : sp.list()) {
                const bool is_thing = s.kind == ClassKind::Thing && s.class_id >= 0;
                for (int idx : s.points) {
                    if (idx < 0 || idx >= (int)snap->size()) continue;
                    const Universe::PointT& pt = (*snap)[idx];
                    const rerun::Position3D xyz(pt.x, pt.y, pt.z);
                    sp_pos.push_back(xyz);
                    sp_col.push_back(superpointColor(s.id));
                    if (is_thing) { th_pos.push_back(xyz); th_cid.emplace_back((uint16_t)s.class_id); }
                }
                if (is_thing) {
                    ctr_pos.emplace_back(s.centroid[0], s.centroid[1], s.centroid[2]);
                    ctr_cid.emplace_back((uint16_t)s.class_id);
                    // Label shown on hover/selection (click the sphere): class name plus
                    // the Mask3D-feature dispersion attribute and how many members had a
                    // feature (n=0 => features off or window not yet covered).
                    char lbl[96];
                    std::snprintf(lbl, sizeof(lbl), "%s | feat disp %.4f (n=%d)",
                                  uni.semantics().name(s.class_id).c_str(),
                                  s.feat_dispersion, s.feat_count);
                    ctr_lbl.emplace_back(std::string(lbl));
                }
            }
            rec.log("world/superpoints/all",
                    rerun::Points3D(sp_pos).with_colors(sp_col).with_radii(0.02f)
                        .with_show_labels(false));
            rec.log("world/superpoints/things",
                    rerun::Points3D(th_pos).with_class_ids(th_cid).with_radii(0.025f)
                        .with_show_labels(false));
            // show_labels(false): the (now longer) label with the dispersion value is
            // surfaced on hover / in the selection panel when you click the sphere,
            // instead of floating permanently over every centroid.
            rec.log("world/superpoints/centroids",
                    rerun::Points3D(ctr_pos).with_class_ids(ctr_cid)
                        .with_labels(ctr_lbl).with_radii(0.08f)
                        .with_show_labels(false));
            // The crop sphere the window was computed over, as a wireframe ellipsoid at
            // the robot (skipped when radius<=0 => whole-map crop, i.e. no local window).
            if (sp_r > 0.0f)
                rec.log("world/superpoints/window",
                        rerun::Ellipsoids3D::from_centers_and_radii(
                            {{robot[0], robot[1], robot[2]}}, {sp_r})
                            .with_fill_mode(rerun::components::FillMode::MajorWireframe)
                            .with_colors({rerun::Color(255, 255, 0)}));
        };
        std::uint64_t last_sp_ver = sp.version();

        // Per-frame hook: only the LIGHTWEIGHT, genuinely time-varying entities
        // (robot pose, trajectory, camera frustum, RGB frame) plus -- on the strided
        // frames where VCCS recomputes -- the current superpoint window. The heavy
        // world map + features are logged once as a static snapshot after the loop.
        semantic::FrameHook hook =
            [&](int idx, const SyncedFrame& sf, const PointPixelMap&) {
                rec.set_time_sequence("frame", idx);
                rec.set_time_duration_secs("capture", (double)(sf.image_t - t0) * 1e-9);

                // Robot + camera poses are BOTH interpolated to the image timestamp, so
                // the frustum/RGB frame and the body pose agree in time (the sync fix).
                rec.log("world/robot", rigidToTransform(sf.T_world_body));
                traj.push_back({sf.robot[0], sf.robot[1], sf.robot[2]});
                rec.log("world/trajectory", rerun::LineStrips3D(rerun::LineStrip3D(traj)));

                rec.log("world/camera", rigidToTransform(sf.cam.T));
                rec.log("world/camera",
                        rerun::Pinhole::from_focal_length_and_resolution(
                            {sf.cam.K[0], sf.cam.K[4]},
                            {(float)sf.cam.width, (float)sf.cam.height})
                            .with_image_plane_distance(0.4f));
                if (sf.image.ok())
                    rec.log("world/camera/image",
                            rerun::Image::from_rgb24(
                                sf.image.rgb,
                                {(uint32_t)sf.image.w, (uint32_t)sf.image.h}));

                // Superpoints are ephemeral: log the current window ONLY when VCCS just
                // recomputed it (version bumped) this frame, anchored at this robot pos.
                if (p.superpoints && sp.version() != last_sp_ver) {
                    last_sp_ver = sp.version();
                    logSuperpointsAt(sf.robot);
                }
            };

        // Drive the SAME pipeline the headless runner uses; the visualizer is a pure
        // consumer (frame hook above). The offline ingestor feeds pushScan/pushImage.
        semantic::OnlineSemantic online(uni, inf, p, &sp,
                                        p.features ? &pf : nullptr, feat.get(),
                                        p.hdbscan ? &os : nullptr);
        online.setFrameHook(hook);
        DogStream stream(cam, online.hook());
        online.begin();
        DogLogIngestor ingestor(reader);
        ingestor.run(stream, p.start, p.count > 0 ? p.count : -1, p.stride);
        online.finish();
        int frames = online.frames();

        // Static class legend: an AnnotationContext maps each class id -> name +
        // palette colour. Logged on "world" (an ancestor of world/map) so the points'
        // class ids resolve to colours and labels, and the viewer renders it as a
        // proper legend with colour swatches (replacing the old legend/* text dump).
        const SemanticVocabulary& sv = uni.semantics();
        std::vector<rerun::datatypes::ClassDescriptionMapElem> classes;
        classes.reserve(sv.size() + 1);
        auto add_class = [&](uint16_t id, const std::string& name, rerun::Color cc) {
            classes.emplace_back(rerun::datatypes::ClassDescription(
                id, name, rerun::datatypes::Rgba32(cc.r(), cc.g(), cc.b())));
        };
        for (int id = 0; id < sv.size(); ++id)
            add_class((uint16_t)id, sv.name(id), classColor(id));
        add_class(UNLABELED_ID, "unlabeled", classColor(-1));
        rec.log_static("world", rerun::AnnotationContext(
            rerun::Collection<rerun::datatypes::ClassDescriptionMapElem>::take_ownership(
                std::move(classes))));

        // Final segmented world + superpoints as a STATIC snapshot: always visible at
        // any time cursor, logged once (no per-frame re-log => tiny recording).
        logWorld(true);

        std::printf("logged %d frames | world=%d points | %d classes | sink=%s\n",
                    frames, uni.size(), sv.size(), spawn ? "spawn" : out.c_str());
        if (p.features)
            std::printf("features: %d/%d pts (%.1f%%) got a %d-d vector%s\n",
                        pf.covered(), uni.size(),
                        uni.size() ? 100.0 * pf.covered() / uni.size() : 0.0, pf.dim(),
                        show_features ? " | shown at world/features" : "");
        if (p.hdbscan)
            std::printf("object seeds: %d instances | shown at world/seeds/{centroids,points}\n",
                        os.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
