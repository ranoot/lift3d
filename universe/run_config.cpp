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

    semantic::Params& p = cfg.p;
    p.start      = root["start"].as<int>(d.start);
    p.count      = root["count"].as<int>(d.count);
    p.stride     = std::max(1, root["stride"].as<int>(d.stride));
    p.voxel      = root["voxel"].as<float>(d.voxel);
    p.radius     = root["radius"].as<float>(d.radius);
    p.proj_radius = root["proj_radius"].as<float>(d.proj_radius);
    p.tau        = root["tau"].as<float>(d.tau);
    p.splat      = root["splat"].as<int>(d.splat);
    p.min_votes  = root["min_votes"].as<int>(d.min_votes);
    p.min_conf   = root["min_conf"].as<float>(d.min_conf);
    p.things_only = root["things_only"].as<bool>(d.things_only);
    p.endpoint   = root["endpoint"].as<std::string>(d.endpoint);
    p.thing = root["thing"].as<std::vector<std::string>>(std::vector<std::string>{
        "door", "chair", "table", "person", "trash can", "box", "monitor", "shelf", "window"});
    p.stuff = root["stuff"].as<std::vector<std::string>>(std::vector<std::string>{
        "wall", "floor", "ceiling"});
    // Dynamic-object classes whose geometry is rejected at ingest (default: none). Must
    // also appear in thing/stuff so the 2D model still segments them (the filter mask).
    p.dynamic = root["dynamic"].as<std::vector<std::string>>(std::vector<std::string>{});

    // Shared cadence for features -> superpoints -> growing (they run together).
    p.refine_every = root["refine_every"].as<int>(d.refine_every);

    // superpoints: { enabled, seed, radius, thing_frac, spatial, normal, refine }
    YAML::Node sp = root["superpoints"];
    p.superpoints   = sp["enabled"].as<bool>(d.superpoints);
    p.sp_radius     = sp["radius"].as<float>(d.sp_radius);
    p.sp.seed_res   = sp["seed"].as<float>(d.sp.seed_res);
    p.sp.spatial_w  = sp["spatial"].as<float>(d.sp.spatial_w);
    p.sp.normal_w   = sp["normal"].as<float>(d.sp.normal_w);
    p.sp.refine     = sp["refine"].as<int>(d.sp.refine);
    p.sp.thing_frac = sp["thing_frac"].as<float>(d.sp.thing_frac);
    p.sp.voxel_res  = p.voxel;                        // VCCS voxel ~ map voxel

    // features: { enabled, endpoint, radius, show }
    YAML::Node ft = root["features"];
    p.features    = ft["enabled"].as<bool>(d.features);
    p.feat_radius = ft["radius"].as<float>(d.feat_radius);
    cfg.feat_endpoint = ft["endpoint"].as<std::string>(cfg.feat_endpoint);
    cfg.show_features = ft["show"].as<bool>(cfg.show_features);

    // hdbscan: { enabled, min_pts, min_cluster_size, min_class_points,
    //            allow_single_cluster, maturity_res, maturity_min }
    // Seeding runs on the LOCAL window (centred on the robot, radius `radius`), not the
    // whole map; the maturity gate keeps HDBSCAN off sparse, half-observed cells. Its
    // cadence is COUPLED to `refine_every` (no separate `every`): it births seeds that
    // grow/consolidate consume the same frame, so they share one interval.
    YAML::Node hd = root["hdbscan"];
    p.hdbscan       = hd["enabled"].as<bool>(d.hdbscan);
    p.hdb.min_pts              = hd["min_pts"].as<int>(d.hdb.min_pts);
    p.hdb.min_cluster_size     = hd["min_cluster_size"].as<int>(d.hdb.min_cluster_size);
    p.hdb.min_class_points     = hd["min_class_points"].as<int>(d.hdb.min_class_points);
    p.hdb.allow_single_cluster = hd["allow_single_cluster"].as<bool>(d.hdb.allow_single_cluster);
    p.hdb.maturity_res         = hd["maturity_res"].as<float>(d.hdb.maturity_res);
    p.hdb.maturity_min         = hd["maturity_min"].as<int>(d.hdb.maturity_min);

    // grow: { enabled, affinity, max_dispersion, cand_radius, require_class }
    YAML::Node gr = root["grow"];
    p.grow                    = gr["enabled"].as<bool>(d.grow);
    p.grow_p.affinity_thresh  = gr["affinity"].as<float>(d.grow_p.affinity_thresh);
    p.grow_p.max_dispersion   = gr["max_dispersion"].as<float>(d.grow_p.max_dispersion);
    p.grow_p.cand_radius      = gr["cand_radius"].as<float>(d.grow_p.cand_radius);
    p.grow_p.require_class     = gr["require_class"].as<bool>(d.grow_p.require_class);

    // objects: { enabled, affinity, adj_min, adj_dilate, cand_radius, require_class }
    // Seeds -> objects consolidation (tier 3). `affinity` is a pure cosine threshold now
    // (adjacency is a separate GATE, adj_min); enabled iff objects AND grow (which requires
    // features/superpoints/hdbscan).
    YAML::Node ob = root["objects"];
    p.objects                  = ob["enabled"].as<bool>(d.objects);
    p.consol_p.affinity_thresh = ob["affinity"].as<float>(d.consol_p.affinity_thresh);
    p.consol_p.adj_min         = ob["adj_min"].as<float>(d.consol_p.adj_min);
    p.consol_p.adj_dilate      = ob["adj_dilate"].as<float>(d.consol_p.adj_dilate);
    p.consol_p.cand_radius     = ob["cand_radius"].as<float>(d.consol_p.cand_radius);
    p.consol_p.require_class    = ob["require_class"].as<bool>(d.consol_p.require_class);

    return cfg;
}
