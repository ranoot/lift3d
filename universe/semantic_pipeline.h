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
#include "object_seeds.h"
#include "objects.h"
#include "instance_ids.h"        // InstanceIdGraph (per-frame DVIS id bridging)
#include "timing.h"
#include "dog_stream.h"          // DogStream, SyncedFrame (pulls in dog_log_adaptor.h)
#include "inf_client.h"

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

// DogFrame (world-lifted lidar) -> PointXYZI cloud, mirroring run_semantic_universe.
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
// colours, superpoints/seeds/objects) -- far cheaper and safer to never
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
    // Per-pixel front-surface band (m) for the working set: keep every visible point within
    // this depth of the nearest at its pixel. 0 => one point per pixel (tightest); larger =>
    // the full front surface at each pixel (more geometry for superpoints/HDBSCAN). Does not
    // grow with map density -- it is bounded by the visible surface either way.
    float surf_band = 0.0f;
    bool  things_only = false; // if set, only "thing" classes are stamped (stuff skipped)

    // Persistent cross-frame voting for the per-point class (Universe::setVoting). Off =>
    // naive per-frame labels (a point's class is the last frame's 2D guess). On => each
    // observation votes into a persistent histogram and the class is its stuff-biased
    // argmax, so the backprojection is a multi-view consensus. `vote_stuff_bias` (>= 1)
    // up-weights stuff votes so a point seen as stuff resists flipping to a thing (curbs
    // objects leaking onto walls); `vote_min_votes`/`vote_min_conf` are the optional
    // label gate (1, 0 => plain argmax).
    bool  voting = false;
    float vote_stuff_bias = 2.0f;
    int   vote_min_votes = 1;
    float vote_min_conf = 0.0f;

    std::vector<std::string> thing;
    std::vector<std::string> stuff;
    // Class names whose GEOMETRY is rejected from the map at ingest (dynamic objects,
    // e.g. people): incoming scan points that project onto one of these 2D segments are
    // dropped BEFORE fusion, so a moving object never accumulates a smear/trail in the
    // persistent world. They stay in the inference vocab (thing/stuff) so the 2D model
    // still segments them -- that mask is exactly what we filter by.
    std::vector<std::string> dynamic;
    std::string endpoint = "ipc:///tmp/inf_server.ipc";

    // VCCS superpoints (oversegmentation). Off unless a Superpoints* is passed AND
    // `superpoints` is set. Runs EVERY frame on the per-view visible thing set (no crop).
    // `sp` holds the VCCS/majority-rule parameters (seed_res, thing_frac, ...).
    bool  superpoints = false;
    Superpoints::Params sp;

    // Object proposals (per-class HDBSCAN* over the per-view visible thing set). Off unless
    // an ObjectSeeds* is passed AND `hdbscan` is set. Runs EVERY frame; proposals are
    // EPHEMERAL (re-clustered each frame, non-claiming across frames) so the same object is
    // re-proposed with overlapping voxels for the IoU consolidation. `hdb` holds strictness.
    bool  hdbscan = false;
    ObjectSeeds::Params hdb;

    // Class-histogram-guided growing: absorb this frame's class-consistent, volume-
    // overlapping superpoints into this frame's proposals (runs after superpoints). Enabled
    // iff `grow` AND superpoints/hdbscan are on. `grow_p` holds the affinity thresholds.
    bool  grow = false;
    ObjectSeeds::GrowParams grow_p;

    // Objects tier (proposals -> objects): single-tier Union-Find over the persistent seed
    // layer -- each proposal is a seed, progressively unioned into the same-class components
    // it is contained in; objects are the VIRTUAL components promoted past min_merges. The
    // pipeline's primary output. Enabled iff `objects` AND grow.
    // `consol_p` holds the merge_thresh schedule + min_merges.
    bool  objects = false;
    Objects::Params consol_p;

    // Consolidate only every `consol_stride` frames (a proposal is folded in on frames
    // 0, stride, 2*stride, ...). Adjacent frames observe near-identical geometry, so folding
    // every one inflates a component's support with tiny-baseline duplicates; striding samples
    // genuinely distinct viewpoints, so `support`/`min_merges` reflects real multi-view
    // evidence. Proposals on skipped frames are simply not folded (they are ephemeral anyway).
    // <=1 => consolidate every frame (default, unchanged behavior).
    int   consol_stride = 1;
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

// Sign hook (optional egress side-channel for sign-text detection): fired per synced
// frame with this frame's 2D masks (`fr`), a pixel -> nearest-surface global universe
// index lookup (`pix_gidx`, size W*H, -1 where nothing is in front), the image dims, and
// the Universe. A consumer scans `fr` for sign instances, turns their pixels into 3D
// footprints via pix_gidx, and dispatches them to the async sign detector. Only fired
// when set (setSignHook), so the core pipeline pays nothing unless a sign build attaches
// it. Deliberately sign-agnostic: it carries only core types.
using SignFrameHook = std::function<void(const SyncedFrame&, const FrameResult&,
                                         const std::vector<int>& pix_gidx,
                                         int W, int H, const Universe&)>;

// One synced frame's core work: integrate geometry, infer 2D labels, project, and STAMP
// each visible point with the current frame's class + DVIS instance id (no cross-view
// voting -- labels are per-frame transient). `out_gidx` receives the NEAREST-PER-PIXEL
// visible, live global indices of this view (one point per image pixel: the back-projected
// surface, bounded by image resolution), which is the pipeline's per-view working set
// before the thing filter. The camera pose in sf.cam is already interpolated to sf.image_t,
// so labels are lifted at the image instant. Returns the new view id. Source-agnostic.
// `out_fr` (optional): receives a copy of this frame's 2D FrameResult (masks + class
// ids) for a sign/egress consumer. `out_pix_gidx` (optional): receives a W*H pixel ->
// nearest-visible-surface global index map (-1 where nothing is in front), built in the
// front-shell pass at no extra cost. Both nullptr by default -> zero overhead.
inline int stepFrameSynced(Universe& uni, InfClient& inf,
                           const Params& p, const SyncedFrame& sf, PointPixelMap& out_map,
                           std::vector<int>& out_gidx, StepTiming* st = nullptr,
                           InstanceIdGraph* idg = nullptr,
                           FrameResult* out_fr = nullptr,
                           std::vector<int>* out_pix_gidx = nullptr) {
    Stopwatch sw;
    Universe::Cloud::Ptr cloud = syncedToCloud(sf);
    const double t_build = sw.lap();               // folded into read_integrate below

    // Infer BEFORE integrating: the 2D labels gate which geometry is allowed into the
    // map (dynamic-object rejection), so the label map has to exist first.
    FrameResult fr = inf.frame(sf.image.rgb, sf.image.h, sf.image.w);
    if (st) st->infer += sw.lap();

    // Bridge this frame's DVIS over-segmentation into the persistent instance graph: union
    // same-class, physically-touching instance ids so seeding can group by resolved instance
    // (see ObjectSeeds::seedFromIndices). Thing membership is decided on the RAW model label.
    if (idg)
        idg->ingestFrame(fr, [&](int lbl) {
            return lbl >= 0 &&
                   uni.semantics().kindOf(inf.className(lbl)) == ClassKind::Thing;
        });

    // Reject dynamic-object points (people, ...) before they are ever fused, so a moving
    // object leaves no smear in the persistent world (see rejectDynamic). No-op if unset.
    if (!p.dynamic.empty())
        cloud = rejectDynamic(cloud, sf.cam, fr, inf, p.dynamic);
    const int view = uni.integrate(*cloud, sf.cam, sf.image_t,
                                   "img" + std::to_string(sf.image_t), sf.robot);
    if (st) st->read_integrate += t_build + sw.lap();

    // Project the world (or local crop) into this view. `proj_radius` caps the z-buffer
    // candidate crop tighter than the feature/grow `radius` when set, else falls back.
    const float pr = p.proj_radius > 0.0f ? p.proj_radius : p.radius;
    std::vector<int> gidx;
    out_map = (pr > 0.0f) ? uni.projectLocal(view, p.tau, pr, &gidx, p.splat)
                          : uni.project(view, p.tau, p.splat);
    if (st) st->project += sw.lap();

    // Reduce the visible set to the FRONT-SURFACE SHELL per pixel. Pass 1: find the nearest
    // visible depth at each image pixel. Pass 2: keep every visible point within `surf_band`
    // metres of that per-pixel nearest. `surf_band == 0` => strict nearest-per-pixel (one
    // point/pixel, tightest bound); larger => keep the whole front surface at each pixel
    // (many more points for the geometry stages) while still dropping background that leaks
    // in behind the shell. Either way the set is bounded by the visible surface, not by how
    // densely the map has fused -- it does not grow with the map. (Raw z-buffer visibility
    // admits every point within `tau` of nearest per pixel, which used to make it balloon.)
    const int   W = fr.w, H = fr.h;
    const float band = p.surf_band > 0.0f ? p.surf_band : 0.0f;
    std::vector<float> best_d((std::size_t)W * H, 0.0f);
    std::vector<char>  seen((std::size_t)W * H, 0);
    // Optional per-pixel nearest-surface global index for an egress consumer (sign
    // footprints). Filled alongside best_d below; -1 = no visible surface at that pixel.
    if (out_pix_gidx) out_pix_gidx->assign((std::size_t)W * H, -1);
    for (int k = 0; k < out_map.N; ++k) {
        if (!out_map.isVisible(k)) continue;
        const int u = out_map.u(k), v = out_map.v(k);
        if (u < 0 || v < 0 || u >= W || v >= H) continue;
        const int idx = pr > 0.0f ? gidx[k] : k;
        if (!uni.pointAlive(idx)) continue;            // tombstoned -> skip
        const std::size_t pix = (std::size_t)v * W + u;
        const float d = out_map.depth[k];
        if (!seen[pix] || d < best_d[pix]) {
            best_d[pix] = d; seen[pix] = 1;
            if (out_pix_gidx) (*out_pix_gidx)[pix] = idx;   // nearest surface at this pixel
        }
    }
    if (out_fr) *out_fr = fr;                           // hand the 2D masks to the consumer

    // Stamp per-frame labels (replaces the old vote loop): reset every point to unlabeled,
    // then paint the front-shell points from THIS frame's maps -- transient by design
    // (clearFrameLabels each frame). Collect the working-set indices for the stages.
    uni.clearFrameLabels();
    out_gidx.clear();
    out_gidx.reserve(out_map.N);
    for (int k = 0; k < out_map.N; ++k) {
        if (!out_map.isVisible(k)) continue;
        const int u = out_map.u(k), v = out_map.v(k);
        if (u < 0 || v < 0 || u >= W || v >= H) continue;
        const int idx = pr > 0.0f ? gidx[k] : k;
        if (!uni.pointAlive(idx)) continue;            // tombstoned -> skip
        const std::size_t pix = (std::size_t)v * W + u;
        if (out_map.depth[k] > best_d[pix] + band) continue;   // behind the front shell
        out_gidx.push_back(idx);                       // front-surface shell (working set)
        const int lbl = fr.labelAt(u, v);
        if (lbl < 0) continue;                         // -1 == background, unlabeled
        const std::string& name = inf.className(lbl);
        if (name.empty()) continue;
        // Kill dynamic-object geometry the moment a view catches it on a "person" segment
        // (the cleanup path the ingest filter can't cover -- see the old note).
        if (isDynamic(p.dynamic, name)) { uni.killPoint(idx); continue; }
        // "things only": drop stuff labels so only object geometry carries a class.
        if (p.things_only && uni.semantics().kindOf(name) == ClassKind::Stuff) continue;
        // id_map is OPTIONAL in the server reply (inf_client only fills it when present),
        // and FrameResult::ok() validates label_map only -- so idAt() would read past an
        // empty/short id_map. Guard it exactly as the experiment path does.
        const int inst = fr.id_map.size() == (std::size_t)fr.h * fr.w ? fr.idAt(u, v) : -1;
        uni.setFrameLabel(idx, name, inst);
    }

    if (st) st->vote += sw.lap();
    return view;
}

} // namespace semantic
