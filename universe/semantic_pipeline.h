#pragma once
// Per-synced-frame label step for the semantic Universe: fuse one time-aligned
// observation into the world map AND vote a per-point semantic class by lifting the
// inf_server 2D label map through the point-to-pixel mapping. The input is a
// SyncedFrame from DogStream (image + camera pose interpolated to the image
// timestamp + nearest lidar cloud), so this step is source-agnostic -- it has no
// notion of files, a DogLogReader, or any transport. The loop that drives it lives
// in OnlineSemantic (online_semantic.h); both drivers go through that.

#include "universe.h"
#include "superpoints.h"
#include "point_features.h"
#include "object_seeds.h"
#include "timing.h"
#include "dog_stream.h"          // DogStream, SyncedFrame (pulls in dog_log_adaptor.h)
#include "inf_client.h"
#include "feat_client.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace semantic {

// "a, b ,c" -> {"a","b","c"} (trims surrounding whitespace, drops empties).
inline std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos) out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// DogFrame (world-lifted lidar) -> PointXYZI cloud, mirroring run_universe.
inline Universe::Cloud::Ptr frameToCloud(const DogFrame& f) {
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

// SyncedFrame (world-lifted lidar in a flat vector) -> PointXYZI cloud.
inline Universe::Cloud::Ptr syncedToCloud(const SyncedFrame& sf) {
    Universe::Cloud::Ptr c(new Universe::Cloud);
    const int N = (int)(sf.cloud_world.size() / 3);
    c->reserve(N);
    for (int i = 0; i < N; ++i) {
        pcl::PointXYZI p;
        p.x = sf.cloud_world[3 * i + 0];
        p.y = sf.cloud_world[3 * i + 1];
        p.z = sf.cloud_world[3 * i + 2];
        p.intensity = (i < (int)sf.intensity.size()) ? sf.intensity[i] : 0.0f;
        c->push_back(p);
    }
    return c;
}

struct Params {
    int   start = 0, count = 50, stride = 1;
    float voxel = 0.05f;
    float radius = 0.0f;   // >0 => label only the local crop around the robot
    float tau   = 0.1f;    // z-buffer visibility tolerance (m)
    int   splat = 1;       // z-buffer depth dilation radius (px); 0 = single pixel
    int   min_votes = 1;   // label gate: min total votes to assign a class
    float min_conf  = 0.0f;// label gate: min winning fraction (0..1) to assign
    bool  things_only = false; // if set, only "thing" classes are voted (stuff skipped)
    std::vector<std::string> thing;
    std::vector<std::string> stuff;
    std::string endpoint = "ipc:///tmp/inf_server.ipc";

    // Refine cadence: frames between local refreshes of features -> superpoints ->
    // growing. These three run TOGETHER (the affinity merge consumes both features and
    // superpoints), so they share one cadence instead of separate feat/sp intervals.
    int   refine_every = 10;

    // VCCS superpoints (oversegmentation). Off unless a Superpoints* is passed to
    // accumulate() AND `superpoints` is set. Refreshed every `refine_every` frames;
    // `sp_radius` is the local crop radius (<=0 => fall back to `radius`, else whole
    // map). `sp` holds the VCCS/majority-rule parameters (seed_res, thing_frac, ...).
    bool  superpoints = false;
    float sp_radius = 0.0f;
    Superpoints::Params sp;

    // Per-point Mask3D features (mask3d_feat backbone). Off unless a PointFeatures*
    // AND a FeatClient* are passed to accumulate() AND `features` is set. The backbone
    // is run every `refine_every` frames over the crop around the robot (radius
    // `feat_radius`, <=0 => fall back to `radius`, else whole map), scattering the
    // 96-d per-voxel features into a persistent whole-cloud store. Requires per-point
    // RGB, which is fused from the camera image during voting whenever `features` is
    // set. `feat` holds the voxel/colour-normalization params.
    bool  features = false;
    float feat_radius = 0.0f;
    PointFeatures::Params feat;

    // Object seeds (per-class whole-map HDBSCAN* over UNCLAIMED points). Off unless an
    // ObjectSeeds* is passed AND `hdbscan` is set. Seeds new objects every
    // `hdbscan_every` frames + once after the loop. `hdb` holds the seeding strictness.
    bool  hdbscan = false;
    int   hdbscan_every = 100;
    ObjectSeeds::Params hdb;

    // Feature-guided growing: within the robot radius, absorb feature-consistent,
    // volume-overlapping superpoints into nearby objects (runs on the `refine_every`
    // cadence, after features + superpoints). Enabled iff `grow` AND all three of
    // features/superpoints/hdbscan are on. `grow_p` holds the affinity thresholds.
    bool  grow = false;
    ObjectSeeds::GrowParams grow_p;
};

// Per-frame hook (for viz / consumers): the synced-frame index, the SyncedFrame
// (image + interpolated camera pose + cloud), and the projection result already
// voted for this view. Fired by OnlineSemantic after the vote (and after any
// superpoint refresh) for this frame.
using FrameHook = std::function<void(int, const SyncedFrame&, const PointPixelMap&)>;

// Objects hook (the pipeline's primary output): the current list of discovered
// object instances (each = a subset of the global map + a centroid) and the Universe
// they index into. Fired when object seeds are (re)computed.
using ObjectsHook = std::function<void(const std::vector<ObjectSeed>&, const Universe&)>;

// One synced frame's core work: integrate geometry, infer 2D labels, project, vote.
// The camera pose in sf.cam is already interpolated to sf.image_t, so the 2D labels
// are lifted through the camera pose AT THE IMAGE INSTANT (this is the timeline fix).
// Returns the new view id. Source-agnostic: no reader, no files.
inline int stepFrameSynced(Universe& uni, InfClient& inf, const Params& p,
                           const SyncedFrame& sf, PointPixelMap& out_map,
                           StepTiming* st = nullptr) {
    Stopwatch sw;
    Universe::Cloud::Ptr cloud = syncedToCloud(sf);
    const int view = uni.integrate(*cloud, sf.cam, sf.image_t,
                                   "img" + std::to_string(sf.image_t), sf.robot);
    if (st) st->read_integrate += sw.lap();

    FrameResult fr = inf.frame(sf.image.rgb, sf.image.h, sf.image.w);
    if (st) st->infer += sw.lap();

    // Project the world (or local crop) into this view; vote every visible point
    // with the class at its pixel. global indices lift local results onto the map.
    std::vector<int> gidx;
    out_map = (p.radius > 0.0f) ? uni.projectLocal(view, p.tau, p.radius, &gidx, p.splat)
                                : uni.project(view, p.tau, p.splat);
    if (st) st->project += sw.lap();
    for (int k = 0; k < out_map.N; ++k) {
        if (!out_map.isVisible(k)) continue;
        const int u = out_map.u(k), v = out_map.v(k);
        if (u < 0 || v < 0 || u >= fr.w || v >= fr.h) continue;
        const int idx = p.radius > 0.0f ? gidx[k] : k;
        // Fuse this point's observed RGB (for the Mask3D backbone), independent of
        // the semantic label -- a point's colour accumulates whenever a camera sees it.
        if (p.features && sf.image.ok() && u < sf.image.w && v < sf.image.h) {
            const std::size_t off = ((std::size_t)v * sf.image.w + u) * 3;
            uni.voteColor(idx, sf.image.rgb[off], sf.image.rgb[off + 1], sf.image.rgb[off + 2]);
        }
        const int lbl = fr.labelAt(u, v);
        if (lbl < 0) continue;                         // -1 == background, no vote
        const std::string& name = inf.className(lbl);
        if (name.empty()) continue;
        // "things only": drop region/stuff labels so the map keeps only object votes.
        if (p.things_only && uni.semantics().kindOf(name) == ClassKind::Stuff) continue;
        uni.voteLabel(idx, name);
    }
    if (st) st->vote += sw.lap();
    return view;
}

} // namespace semantic
