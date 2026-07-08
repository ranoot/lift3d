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
#include "objects.h"
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

// True iff `name` is one of the configured dynamic-object classes (people, ...).
inline bool isDynamic(const std::vector<std::string>& dyn, const std::string& name) {
    for (const std::string& d : dyn) if (d == name) return true;
    return false;
}

// Drop incoming scan points that project onto a "dynamic" 2D segment (people, ...),
// so moving objects never enter the persistent map in the first place. This is done
// BEFORE fusion on purpose: the map is index-keyed and grow-only, so removing a point
// after it has been fused would shift every downstream global index (vox_, votes,
// colours, features, superpoints/seeds/objects) -- far cheaper and safer to never
// admit it. The incoming scan is one lidar sweep (a single return per direction), so a
// plain pinhole projection + mask lookup is unambiguous: no z-buffer needed, because if
// a point lands on a person pixel it *is* the person (the nearest surface on that ray).
// `dyn` is a set of class NAMES; a point is dropped iff its pixel's 2D label is one of
// them. Returns the input unchanged when `dyn` is empty (zero overhead when off).
inline Universe::Cloud::Ptr rejectDynamic(const Universe::Cloud::Ptr& in,
                                          const Camera& cam, const FrameResult& fr,
                                          const InfClient& inf,
                                          const std::vector<std::string>& dyn) {
    if (dyn.empty() || !in || in->empty()) return in;
    float M[12];
    buildProjection(cam, M);                       // M = K * (T^-1)[0:3]
    Universe::Cloud::Ptr out(new Universe::Cloud);
    out->reserve(in->size());
    for (const Universe::PointT& p : *in) {
        const float a = M[0] * p.x + M[1] * p.y + M[2]  * p.z + M[3];
        const float b = M[4] * p.x + M[5] * p.y + M[6]  * p.z + M[7];
        const float c = M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11];   // = depth
        bool drop = false;
        if (c > 0.0f) {                            // in front of the camera
            const int u = (int)(a / c + 0.5f), v = (int)(b / c + 0.5f);
            if (u >= 0 && v >= 0 && u < fr.w && v < fr.h) {
                const int lbl = fr.labelAt(u, v);
                if (lbl >= 0 && isDynamic(dyn, inf.className(lbl))) drop = true;
            }
        }
        if (!drop) out->push_back(p);
    }
    return out;
}

struct Params {
    int   start = 0, count = 50, stride = 1;
    float voxel = 0.05f;
    float radius = 0.0f;   // >0 => label only the local crop around the robot
    float proj_radius = 0.0f; // >0 => cap the z-buffer candidate crop tighter than
                              // `radius` (fewer points projected); <=0 => use `radius`
    float tau   = 0.1f;    // z-buffer visibility tolerance (m)
    int   splat = 1;       // z-buffer depth dilation radius (px); 0 = single pixel
    int   min_votes = 1;   // label gate: min total votes to assign a class
    float min_conf  = 0.0f;// label gate: min winning fraction (0..1) to assign
    bool  things_only = false; // if set, only "thing" classes are voted (stuff skipped)
    std::vector<std::string> thing;
    std::vector<std::string> stuff;
    // Class names whose GEOMETRY is rejected from the map at ingest (dynamic objects,
    // e.g. people): incoming scan points that project onto one of these 2D segments are
    // dropped BEFORE fusion, so a moving object never accumulates a smear/trail in the
    // persistent world. They stay in the inference vocab (thing/stuff) so the 2D model
    // still segments them -- that mask is exactly what we filter by.
    std::vector<std::string> dynamic;
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

    // Object seeds (per-class LOCAL-window HDBSCAN* over UNCLAIMED, mature points). Off
    // unless an ObjectSeeds* is passed AND `hdbscan` is set. Seeding is COUPLED to the
    // `refine_every` cadence (it births the seeds that grow/consolidate consume that same
    // frame), so it has no separate interval; the maturity gate — not a frame counter —
    // decides when a region is ready. `hdb` holds the seeding strictness + maturity gate.
    bool  hdbscan = false;
    ObjectSeeds::Params hdb;

    // Feature-guided growing: within the robot radius, absorb feature-consistent,
    // volume-overlapping superpoints into nearby SEEDS (runs on the `refine_every`
    // cadence, after features + superpoints). Enabled iff `grow` AND all three of
    // features/superpoints/hdbscan are on. `grow_p` holds the affinity thresholds.
    bool  grow = false;
    ObjectSeeds::GrowParams grow_p;

    // Objects tier (seeds -> objects consolidation): merge adjacent, feature-similar
    // seeds into persistent objects (the pipeline's primary output). Runs on the same
    // seed/grow cadences, after them. Enabled iff `objects` AND grow (which itself
    // requires features/superpoints/hdbscan). `consol_p` holds the merge thresholds.
    bool  objects = false;
    Objects::Params consol_p;
};

// Per-frame hook (for viz / consumers): the synced-frame index, the SyncedFrame
// (image + interpolated camera pose + cloud), and the projection result already
// voted for this view. Fired by OnlineSemantic after the vote (and after any
// superpoint refresh) for this frame.
using FrameHook = std::function<void(int, const SyncedFrame&, const PointPixelMap&)>;

// Objects hook (the pipeline's primary output): the current list of discovered
// object instances (each = a subset of the global map + a centroid) and the Universe
// they index into. Fired when the objects tier is (re)computed.
using ObjectsHook = std::function<void(const std::vector<Object>&, const Universe&)>;

// One synced frame's core work: integrate geometry, infer 2D labels, project, vote.
// The camera pose in sf.cam is already interpolated to sf.image_t, so the 2D labels
// are lifted through the camera pose AT THE IMAGE INSTANT (this is the timeline fix).
// Returns the new view id. Source-agnostic: no reader, no files.
inline int stepFrameSynced(Universe& uni, InfClient& inf, const Params& p,
                           const SyncedFrame& sf, PointPixelMap& out_map,
                           StepTiming* st = nullptr) {
    Stopwatch sw;
    Universe::Cloud::Ptr cloud = syncedToCloud(sf);
    const double t_build = sw.lap();               // folded into read_integrate below

    // Infer BEFORE integrating: the 2D labels gate which geometry is allowed into the
    // map (dynamic-object rejection), so the label map has to exist first.
    FrameResult fr = inf.frame(sf.image.rgb, sf.image.h, sf.image.w);
    if (st) st->infer += sw.lap();

    // Reject dynamic-object points (people, ...) before they are ever fused, so a moving
    // object leaves no smear in the persistent world (see rejectDynamic). No-op if unset.
    if (!p.dynamic.empty())
        cloud = rejectDynamic(cloud, sf.cam, fr, inf, p.dynamic);
    const int view = uni.integrate(*cloud, sf.cam, sf.image_t,
                                   "img" + std::to_string(sf.image_t), sf.robot);
    if (st) st->read_integrate += t_build + sw.lap();

    // Project the world (or local crop) into this view; vote every visible point
    // with the class at its pixel. global indices lift local results onto the map.
    // The z-buffer candidate crop uses `proj_radius` when set (tighter than the
    // feature/grow `radius`), else falls back to `radius`.
    const float pr = p.proj_radius > 0.0f ? p.proj_radius : p.radius;
    std::vector<int> gidx;
    out_map = (pr > 0.0f) ? uni.projectLocal(view, p.tau, pr, &gidx, p.splat)
                          : uni.project(view, p.tau, p.splat);
    if (st) st->project += sw.lap();
    for (int k = 0; k < out_map.N; ++k) {
        if (!out_map.isVisible(k)) continue;
        const int u = out_map.u(k), v = out_map.v(k);
        if (u < 0 || v < 0 || u >= fr.w || v >= fr.h) continue;
        const int idx = pr > 0.0f ? gidx[k] : k;
        if (!uni.pointAlive(idx)) continue;            // tombstoned -> don't revote/recolour
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
        // Tombstone dynamic-object geometry the moment any view catches it on a "person"
        // segment. This is the cleanup path the ingest filter can't cover: the wide-FOV
        // lidar fuses people seen OFF to the camera's side (no 2D mask there to reject
        // them), and they surface later as unlabeled ghosts from other views -- until a
        // view sees them on a person mask and kills them here. The z-buffer already gated
        // visibility (isVisible), so only the points actually ON the person die; anything
        // occluded behind stays. Killed => never voted, never projected/clustered again.
        if (isDynamic(p.dynamic, name)) { uni.killPoint(idx); continue; }
        // "things only": drop region/stuff labels so the map keeps only object votes.
        if (p.things_only && uni.semantics().kindOf(name) == ClassKind::Stuff) continue;
        uni.voteLabel(idx, name);
    }
    if (st) st->vote += sw.lap();
    return view;
}

} // namespace semantic
