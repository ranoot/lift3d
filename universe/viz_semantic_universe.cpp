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
// Requires the inf_server running. The final segmented world map is logged ONCE as a
// static snapshot (always visible at any time cursor); only the lightweight robot
// pose / trajectory / camera / RGB frame vary per frame. This keeps the recording
// small -- re-logging the whole world every frame made it O(frames x map_size)
// (many GB), which the viewer's memory GC purges mid-load, so points flickered in
// and vanished, leaving an empty scene.
// Points are split by panoptic kind into world/map/{things,stuff,unlabeled} so you
// can hide "stuff" (walls/floor/...) with one toggle in the viewer's entity tree
// and keep just the objects. --things-only skips voting stuff classes entirely.
//
// With --superpoints, the VCCS oversegmentation is EPHEMERAL (recomputed per frame on the
// per-view visible thing set) and is logged ON THE TIMELINE at each frame it is recomputed
// -- NOT in the static snapshot -- so scrubbing shows only the current view's superpoints,
// moving with the robot. Under world/superpoints/*: /all colours each point by a palette
// keyed on its superpoint id, /things shows only object superpoints coloured by class via
// the shared AnnotationContext, and /centroids marks each object superpoint's centre with
// its class-name label.
// The per-frame PROPOSALS (grown HDBSCAN seeds) are logged the same way under
// world/proposals/*: /points colours every member point (grown, incl. absorbed boundary
// geometry) by a palette keyed on the proposal's id and /centroids marks each proposal's
// centre with its class name. Proposals are re-seeded + re-grown every frame (seeds +
// superpoints => proposals, all within one view), so this layer is re-logged every frame
// and REPLACES the prior frame's -- scrubbing shows only the current view's proposals.

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
    const bool         spawn   = cfg.spawn;
    const std::string  out     = cfg.out;

    try {
        CameraCalib cam = loadCameraCalib(calib, cam_id);
        DogLogReader reader(log + "/PoseAndPointClouds.bin");
        reader.openCamera(log + "/camera" + std::to_string(cam_id) + ".gclf");
        if (reader.size() == 0) { std::fprintf(stderr, "empty recording\n"); return 1; }

        // OnlineSemantic OWNS the pipeline; the visualizer is a pure consumer. Bind live
        // references to the owned stages so the per-frame hook + final static snapshot below
        // (which capture uni/sp/os/obj by reference) read the pipeline's own instances.
        semantic::OnlineSemantic online(p);
        if (!online.inf().ping()) { std::fprintf(stderr, "inf_server not responding at %s\n",
                                                 p.endpoint.c_str()); return 1; }
        Universe&    uni = online.universe();
        Superpoints& sp  = online.superpoints();
        ObjectSeeds& os  = online.seeds();
        Objects&     obj = online.objects();

        rerun::RecordingStream rec("lift3d/semantic_universe");
        if (spawn) rec.spawn().exit_on_failure();
        else       rec.save(out).exit_on_failure();
        rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

        // Timeline origin = the first replayed IMAGE timestamp (frames are now image-
        // driven, one per camera frame, each at its interpolated pose).
        const int start_img = std::min(p.start, reader.numImages() - 1);
        const std::int64_t t0 = reader.imageTimestamp(std::max(0, start_img));
        std::vector<rerun::datatypes::Vec3D> traj;

        // EVERY per-frame proposal (tier-2 seed) accumulated across the whole run. The seed
        // store is wiped each frame (seedFromIndices), so the only way to see all proposals at
        // once is to snapshot their centroids in the frame hook (fired just after seeding, before
        // the next frame clears the list). Logged static at the end as world/seeds/all_centroids.
        // Distinct colour per proposal (global running index) so overlapping-but-separate
        // proposals in one spot -- the signature of a merging failure -- are visibly distinct;
        // sparse centroids where an object is missing signal a seeding/proposal gap instead.
        std::vector<rerun::Position3D>       seed_ctr;
        std::vector<rerun::Color>            seed_col;
        std::vector<rerun::components::Text> seed_lbl;
        std::uint32_t                        seed_running = 0;

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
                if (!uni.pointAlive(k)) continue;       // tombstoned (dynamic) -> hidden
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

            // NOTE: per-frame PROPOSALS (grown HDBSCAN seeds) are NOT logged here. Like the
            // superpoints, they are ephemeral (re-seeded + re-grown per frame), so they are
            // logged on the timeline at every frame under world/proposals/* (see
            // logProposalsAt + the hook), replacing the prior frame -- not baked into this
            // always-visible static snapshot. Only the run-wide proposal-centroid scatter
            // (world/seeds/all_centroids) stays static; it is logged after the loop.

            // OBJECTS (tier 3, the primary output): the headline marker is a big
            // class-coloured CENTROID per promoted Union-Find component (centroid only -- no
            // bounding box, per the user), plus its member points coloured by the object's
            // Union-Find root id. The colour is stable while a component stays intact; when
            // two components merge, the union picks one root, so the merged object recolours
            // once to the surviving root's colour (accepted -- objects are virtual).
            if (p.objects && obj.size() > 0) {
                std::vector<rerun::Position3D> ctr, mpos;
                std::vector<rerun::Color>               ctr_col;
                std::vector<rerun::components::Text>    ctr_lbl;
                std::vector<rerun::Color> mcol;
                for (const Object& o : obj.list()) {
                    ctr.emplace_back(o.centroid[0], o.centroid[1], o.centroid[2]);
                    ctr_lbl.emplace_back(uni.semantics().name(o.class_id));
                    const rerun::Color oc = superpointColor((std::uint32_t)o.id);
                    ctr_col.push_back(oc);
                    for (int idx : o.points) {
                        if (idx < 0 || idx >= (int)snap->size() || !uni.pointAlive(idx)) continue;
                        const Universe::PointT& pt = (*snap)[idx];
                        mpos.emplace_back(pt.x, pt.y, pt.z);
                        mcol.push_back(oc);
                    }
                }
                emit("world/objects/points",
                     rerun::Points3D(mpos).with_colors(mcol).with_radii(0.045f)
                         .with_show_labels(false));
                emit("world/objects/centroids",
                     rerun::Points3D(ctr).with_colors(ctr_col).with_labels(ctr_lbl)
                         .with_radii(0.15f));
            }
        };

        // Log the CURRENT ephemeral superpoints on the timeline (at whatever time the caller
        // has set). Called from the hook every frame VCCS recomputes, so scrubbing shows the
        // per-view superpoints at that instant rather than one always-on global snapshot.
        // Each call REPLACES the previous set (same entity paths + ephemeral list), so it
        // never accumulates; the set stays visible until the next recompute.
        auto logSuperpointsAt = [&]() {
            Universe::Cloud::ConstPtr snap = uni.cloud();
            std::vector<rerun::Position3D> sp_pos, th_pos, ctr_pos;
            std::vector<rerun::Color>      sp_col, th_col, ctr_col;
            std::vector<rerun::components::Text>    ctr_lbl;
            for (const Superpoint& s : sp.list()) {
                const bool is_thing = s.kind == ClassKind::Thing && s.class_id >= 0;
                const rerun::Color sc = superpointColor(s.id);
                for (int idx : s.points) {
                    if (idx < 0 || idx >= (int)snap->size() || !uni.pointAlive(idx)) continue;
                    const Universe::PointT& pt = (*snap)[idx];
                    const rerun::Position3D xyz(pt.x, pt.y, pt.z);
                    sp_pos.push_back(xyz);
                    sp_col.push_back(sc);
                    if (is_thing) { th_pos.push_back(xyz); th_col.push_back(sc); }
                }
                if (is_thing) {
                    ctr_pos.emplace_back(s.centroid[0], s.centroid[1], s.centroid[2]);
                    ctr_col.push_back(sc);
                    // Class name shown on hover/selection (click the sphere).
                    ctr_lbl.emplace_back(uni.semantics().name(s.class_id));
                }
            }
            rec.log("world/superpoints/all",
                    rerun::Points3D(sp_pos).with_colors(sp_col).with_radii(0.02f)
                        .with_show_labels(false));
            rec.log("world/superpoints/things",
                    rerun::Points3D(th_pos).with_colors(th_col).with_radii(0.025f)
                        .with_show_labels(false));
            // show_labels(false): the class-name label is surfaced on hover / in the
            // selection panel when you click the sphere, instead of floating permanently
            // over every centroid.
            rec.log("world/superpoints/centroids",
                    rerun::Points3D(ctr_pos).with_colors(ctr_col)
                        .with_labels(ctr_lbl).with_radii(0.08f)
                        .with_show_labels(false));
        };
        std::uint64_t last_sp_ver = sp.version();

        // Log THIS frame's PROPOSALS (grown HDBSCAN seeds) on the timeline, mirroring
        // logSuperpointsAt. Each proposal's member points (already grown by growLocal, incl.
        // absorbed unlabeled/stuff boundary geometry) are coloured by a stable hash of the
        // proposal's id -- one colour per proposal -- and its centroid gets a class-name
        // label. The seed store is re-seeded + re-grown every frame, so this is called every
        // frame and REPLACES the previous frame's entities (same paths); an empty log on a
        // proposal-less frame clears the prior frame's points rather than leaving them stale.
        auto logProposalsAt = [&]() {
            Universe::Cloud::ConstPtr snap = uni.cloud();
            std::vector<rerun::Position3D> pr_pos, ctr_pos;
            std::vector<rerun::Color>      pr_col, ctr_col;
            std::vector<rerun::components::Text> ctr_lbl;
            for (const ObjectSeed& s : os.list()) {
                const rerun::Color sc = superpointColor((std::uint32_t)s.id);
                for (int idx : s.points) {
                    if (idx < 0 || idx >= (int)snap->size() || !uni.pointAlive(idx)) continue;
                    const Universe::PointT& pt = (*snap)[idx];
                    pr_pos.emplace_back(pt.x, pt.y, pt.z);
                    pr_col.push_back(sc);
                }
                ctr_pos.emplace_back(s.centroid[0], s.centroid[1], s.centroid[2]);
                ctr_col.push_back(sc);
                ctr_lbl.emplace_back(uni.semantics().name(s.class_id));
            }
            rec.log("world/proposals/points",
                    rerun::Points3D(pr_pos).with_colors(pr_col).with_radii(0.035f)
                        .with_show_labels(false));
            // Class name surfaced on hover / in the selection panel (click the sphere),
            // not floating permanently over every centroid.
            rec.log("world/proposals/centroids",
                    rerun::Points3D(ctr_pos).with_colors(ctr_col)
                        .with_labels(ctr_lbl).with_radii(0.06f)
                        .with_show_labels(false));
        };

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
                    logSuperpointsAt();
                }

                // Proposals are re-seeded + re-grown every frame, so log them UNCONDITIONALLY
                // each frame (not version-gated like superpoints): this replaces the prior
                // frame's world/proposals/* and an empty log clears a proposal-less frame.
                if (p.hdbscan) logProposalsAt();

                // Snapshot THIS frame's proposals (the seed store is cleared next frame) into
                // the run-wide aggregate, so world/seeds/all_centroids ends up holding every
                // proposal ever made -- the coverage-vs-merging diagnostic.
                if (p.hdbscan) {
                    for (const ObjectSeed& s : os.list()) {
                        seed_ctr.emplace_back(s.centroid[0], s.centroid[1], s.centroid[2]);
                        seed_col.push_back(superpointColor(seed_running++));
                        seed_lbl.emplace_back(uni.semantics().name(s.class_id));
                    }
                }
            };

        // Drive the SAME pipeline the headless runner uses; the visualizer is a pure
        // consumer (frame hook above). The offline LogFrameSource feeds pushScan/pushImage.
        online.setFrameHook(hook);
        LogFrameSource src(reader, cam);
        online.attach(src);
        online.begin();
        src.run(p.start, p.count > 0 ? p.count : -1, p.stride);
        online.finish();
        int frames = online.frames();

        // Viz-side rendering to rerun (legend + final static world snapshot) runs AFTER
        // the pipeline's own end-of-run profile, so it is invisible to that report even
        // though for viz it is often the single biggest cost. Time it here and print a
        // matching "[timing]" line so every part of the viz run is accounted for.
        semantic::Stopwatch viz_sw;

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
        const double t_legend = viz_sw.lap();

        // Final segmented world + superpoints as a STATIC snapshot: always visible at
        // any time cursor, logged once (no per-frame re-log => tiny recording).
        logWorld(true);

        // All proposals from the whole run, as an always-visible static layer. Each is a
        // small distinct-coloured dot at the proposal centroid (class name on hover); a dense
        // clump that never became one object => merging problem, a bare region => proposal gap.
        // Radius kept SMALL (below the object centroids' 0.15 and points' 0.045) so this
        // run-wide scatter -- thousands of dots over a long run -- does not occlude the
        // headline object markers; it is a separate entity you can also hide in the tree.
        if (p.hdbscan && !seed_ctr.empty())
            rec.log_static("world/seeds/all_centroids",
                           rerun::Points3D(seed_ctr).with_colors(seed_col)
                               .with_labels(seed_lbl).with_radii(0.035f)
                               .with_show_labels(false));
        const double t_world = viz_sw.lap();
        std::fprintf(stderr,
                     "[timing] viz render | legend %7.1f  static_world %9.1f ms | "
                     "%.2f s total\n",
                     t_legend, t_world, (t_legend + t_world) / 1000.0);

        std::printf("logged %d frames | world=%d points (%d live, %d tombstoned dynamic) "
                    "| %d classes | sink=%s\n",
                    frames, uni.size(), uni.aliveCount(), uni.size() - uni.aliveCount(),
                    sv.size(), spawn ? "spawn" : out.c_str());
        if (p.hdbscan)
            std::printf("object seeds (tier 2): %zu proposals total over %d frames | "
                        "per-frame grown members at world/proposals/points (scrub the "
                        "timeline), ALL proposal centroids at world/seeds/all_centroids\n",
                        seed_ctr.size(), frames);
        if (p.objects)
            std::printf("objects (tier 3): %d promoted (from %d seeds) | shown at "
                        "world/objects/{centroids,points}\n",
                        obj.size(), obj.seedCount());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
