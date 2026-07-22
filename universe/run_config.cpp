#include "run_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

// Expand a leading '~' (or '~/') to $HOME. Leaves the path untouched otherwise.
std::string expandTilde(const std::string& s) {
    if (s.empty() || s[0] != '~') return s;
    const char* home = std::getenv("HOME");
    if (!home) return s;
    return std::string(home) + s.substr(1);
}

}  // namespace

RunConfig loadRunConfig(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("cannot load config '" + path + "': " + e.what());
    }

    RunConfig cfg;
    semantic::Params d;                               // built-in defaults

    cfg.log   = expandTilde(root["log"].as<std::string>(std::string()));
    cfg.calib = expandTilde(root["calib"].as<std::string>(std::string()));
    if (cfg.calib.empty() && !cfg.log.empty())
        cfg.calib = cfg.log + "/../CameraCalRaiboAsQUGV113.xml";
    cfg.cam = root["cam"].as<int>(cfg.cam);
    cfg.out = root["out"].as<std::string>(cfg.out);

    // viz: { seg_overlay, seg_alpha, seg_min_area } -- viz-only 2D segmentation overlay
    // painted onto the camera image. Missing keys keep the built-in defaults.
    YAML::Node viz = root["viz"];
    cfg.seg_overlay  = viz["seg_overlay"].as<std::string>(cfg.seg_overlay);
    cfg.seg_alpha    = viz["seg_alpha"].as<float>(cfg.seg_alpha);
    cfg.seg_min_area = viz["seg_min_area"].as<int>(cfg.seg_min_area);
    // ZMQ endpoint the runner pushes viz messages to (empty => no visualizer). Accepts a
    // top-level `viz_endpoint:` or a `viz: { endpoint: ... }` nested key.
    cfg.viz_endpoint = root["viz_endpoint"].as<std::string>(
                           viz["endpoint"].as<std::string>(cfg.viz_endpoint));

    // sign: { enabled, vlm_url, classes, min_px, workers, queue, mock_delay_ms, mock_text }
    // Sign-text egress. OFF unless explicitly enabled; the mock_* keys only bite in a build
    // without LIFT3D_USE_SIGN (where the stand-in backend replies instead of the VLM).
    YAML::Node sg = root["sign"];
    sign::SignConfig sd;                              // built-in defaults
    cfg.sign_cfg.enabled       = sg["enabled"].as<bool>(sd.enabled);
    cfg.sign_cfg.vlm_url       = sg["vlm_url"].as<std::string>(sd.vlm_url);
    cfg.sign_cfg.sign_classes  = sg["classes"].as<std::vector<std::string>>(sd.sign_classes);
    cfg.sign_cfg.num_workers   = sg["workers"].as<int>(sd.num_workers);
    cfg.sign_cfg.queue_size    = sg["queue"].as<int>(sd.queue_size);
    cfg.sign_cfg.min_sign_px   = sg["min_px"].as<int>(sd.min_sign_px);
    cfg.sign_cfg.mock_delay_ms = sg["mock_delay_ms"].as<int>(sd.mock_delay_ms);
    cfg.sign_cfg.mock_text     = sg["mock_text"].as<std::string>(sd.mock_text);

    semantic::Params& p = cfg.p;
    p.start      = root["start"].as<int>(d.start);
    p.count      = root["count"].as<int>(d.count);
    p.stride     = std::max(1, root["stride"].as<int>(d.stride));
    p.voxel      = root["voxel"].as<float>(d.voxel);
    p.radius     = root["radius"].as<float>(d.radius);
    p.proj_radius = root["proj_radius"].as<float>(d.proj_radius);
    p.tau        = root["tau"].as<float>(d.tau);
    p.splat      = root["splat"].as<int>(d.splat);
    p.splat_mult = root["splat_mult"].as<float>(d.splat_mult);
    p.splat_min  = root["splat_min"].as<int>(d.splat_min);
    p.surf_band  = root["surf_band"].as<float>(d.surf_band);
    p.things_only = root["things_only"].as<bool>(d.things_only);

    // voting: { enabled, stuff_bias, min_votes, min_conf }. Persistent cross-frame
    // vote histogram for the per-point class; off => naive per-frame labels. The stuff
    // bias makes a point seen as stuff resist flipping to a thing.
    YAML::Node vt = root["voting"];
    p.voting          = vt["enabled"].as<bool>(d.voting);
    p.vote_stuff_bias = vt["stuff_bias"].as<float>(d.vote_stuff_bias);
    p.vote_min_votes  = vt["min_votes"].as<int>(d.vote_min_votes);
    p.vote_min_conf   = vt["min_conf"].as<float>(d.vote_min_conf);

    p.endpoint   = root["endpoint"].as<std::string>(d.endpoint);
    p.thing = root["thing"].as<std::vector<std::string>>(std::vector<std::string>{
        "door", "chair", "table", "person", "trash can", "box", "monitor", "shelf", "window"});
    p.stuff = root["stuff"].as<std::vector<std::string>>(std::vector<std::string>{
        "wall", "floor", "ceiling"});
    // Dynamic-object classes whose geometry is rejected at ingest (default: none). Must
    // also appear in thing/stuff so the 2D model still segments them (the filter mask).
    p.dynamic = root["dynamic"].as<std::vector<std::string>>(std::vector<std::string>{});

    // superpoints: { enabled, seed, thing_frac, spatial, normal, refine }. VCCS runs on the
    // per-frame visible thing set (no crop radius).
    YAML::Node sp = root["superpoints"];
    p.superpoints   = sp["enabled"].as<bool>(d.superpoints);
    p.sp.seed_res   = sp["seed"].as<float>(d.sp.seed_res);
    p.sp.spatial_w  = sp["spatial"].as<float>(d.sp.spatial_w);
    p.sp.normal_w   = sp["normal"].as<float>(d.sp.normal_w);
    p.sp.refine     = sp["refine"].as<int>(d.sp.refine);
    p.sp.thing_frac = sp["thing_frac"].as<float>(d.sp.thing_frac);
    p.sp.voxel_res  = p.voxel;                        // VCCS voxel ~ map voxel

    // hdbscan: { enabled, min_pts, min_cluster_size, min_class_points,
    //            allow_single_cluster, leaf_selection, single_scan }
    // Seeding runs EVERY frame on the per-frame visible thing set (no crop, no maturity
    // gate). Proposals are ephemeral: re-clustered each frame and fed to consolidation.
    YAML::Node hd = root["hdbscan"];
    p.hdbscan       = hd["enabled"].as<bool>(d.hdbscan);
    p.hdb.min_pts              = hd["min_pts"].as<int>(d.hdb.min_pts);
    p.hdb.min_cluster_size     = hd["min_cluster_size"].as<int>(d.hdb.min_cluster_size);
    p.hdb.min_class_points     = hd["min_class_points"].as<int>(d.hdb.min_class_points);
    p.hdb.allow_single_cluster = hd["allow_single_cluster"].as<bool>(d.hdb.allow_single_cluster);
    p.hdb.leaf_selection       = hd["leaf_selection"].as<bool>(d.hdb.leaf_selection);
    p.hdb.single_scan          = hd["single_scan"].as<bool>(d.hdb.single_scan);
    // SOR pre-filter before the geometric HDBSCAN paths (drops sparse bridge points).
    p.hdb.sor_enabled          = hd["sor_enabled"].as<bool>(d.hdb.sor_enabled);
    p.hdb.sor_mean_k           = hd["sor_mean_k"].as<int>(d.hdb.sor_mean_k);
    p.hdb.sor_std_mul          = hd["sor_std_mul"].as<float>(d.hdb.sor_std_mul);
    p.hdb.split_radius         = hd["split_radius"].as<float>(d.hdb.split_radius);

    // grow: { enabled, affinity, require_class }. Absorbs this frame's superpoints into this
    // frame's proposals; affinity = cosine(class histograms) * containment (SAI3D).
    YAML::Node gr = root["grow"];
    p.grow                    = gr["enabled"].as<bool>(d.grow);
    p.grow_p.affinity_thresh  = gr["affinity"].as<float>(d.grow_p.affinity_thresh);
    p.grow_p.require_class     = gr["require_class"].as<bool>(d.grow_p.require_class);
    // SAI3D neighborhood-aware growing (opt-in): pool sp histograms over KNN neighbors
    // before the affinity cosine, and iterate the assignment as objects grow.
    p.grow_p.neighbor_pool     = gr["neighbor_pool"].as<bool>(d.grow_p.neighbor_pool);
    p.grow_p.sweeps            = gr["sweeps"].as<int>(d.grow_p.sweeps);
    p.grow_p.dis_decay         = gr["dis_decay"].as<float>(d.grow_p.dis_decay);
    p.grow_p.pool_k            = gr["pool_k"].as<int>(d.grow_p.pool_k);

    // objects: { enabled, merge_thresh, min_merges, require_class }
    // Proposals -> objects (tier 3): single-tier Union-Find over the persistent seed layer.
    // Each proposal is retained as a seed and progressively unioned into every same-class
    // component it is contained in past that component's level bar (merge_thresh); a
    // component is reported as an object once its support reaches min_merges.
    YAML::Node ob = root["objects"];
    p.objects                = ob["enabled"].as<bool>(d.objects);
    p.consol_p.merge_thresh  = ob["merge_thresh"].as<std::vector<float>>(d.consol_p.merge_thresh);
    p.consol_p.min_merges    = ob["min_merges"].as<int>(d.consol_p.min_merges);
    p.consol_p.require_class = ob["require_class"].as<bool>(d.consol_p.require_class);
    p.consol_stride          = ob["consol_stride"].as<int>(d.consol_stride);
    // SAI3D global-context merging (opt-in): accumulate confidence-weighted affinity across
    // frames and gate inter-object merges on multi-view evidence + neighborhood aggregation.
    p.consol_p.global_context        = ob["global_context"].as<bool>(d.consol_p.global_context);
    p.consol_p.min_evidence          = ob["min_evidence"].as<float>(d.consol_p.min_evidence);
    p.consol_p.dis_decay             = ob["dis_decay"].as<float>(d.consol_p.dis_decay);
    p.consol_p.use_semantic_affinity = ob["use_semantic_affinity"].as<bool>(d.consol_p.use_semantic_affinity);
    p.consol_p.use_containment       = ob["use_containment"].as<bool>(d.consol_p.use_containment);
    p.consol_p.small_seg_min         = ob["small_seg_min"].as<int>(d.consol_p.small_seg_min);
    // Overlapping point-set model: one master flag under objects:, fanned out to every stage
    // that enforces exclusivity today (seeding, growing, consolidation).
    const bool overlap_sets          = ob["overlap_sets"].as<bool>(d.consol_p.overlap_sets);
    p.consol_p.overlap_sets = overlap_sets;
    p.hdb.overlap_sets      = overlap_sets;
    p.grow_p.overlap_sets   = overlap_sets;

    return cfg;
}
