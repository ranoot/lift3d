#pragma once
// OnlineSemantic: the pipeline loop, driven ONE SYNCED FRAME AT A TIME. It is the
// bridge between DogStream (which emits time-aligned SyncedFrames) and the semantic
// Universe: for each synced frame it integrates + votes (stepFrameSynced), and on a
// cadence recomputes VCCS superpoints / HDBSCAN* object seeds. Its primary output is
// the list of discovered objects (each a subset of the global map + a centroid),
// delivered through onObjects.
//
// This replaces the old reader-driven semantic::accumulate() loop: instead of pulling
// frames from a file, the pipeline is PUSHED frames. The same OnlineSemantic serves
// the live path and the offline simulated ingestor -- the only difference is who calls
// DogStream::pushScan/pushImage. Visualizers and publishers attach as consumers via
// setFrameHook / setObjectsHook and never touch the pipeline internals.
//
// OnlineSemantic OWNS its stages (Universe / InfClient / Superpoints / ObjectSeeds /
// Objects), built from Params -- callers no longer wire them by hand. The input transport
// is a pluggable FrameSource (offline LogFrameSource or live GmdFrameSource); attach() binds
// its synced-frame output straight into onSynced.
//
// Usage:
//   OnlineSemantic online(p);              // owns every stage; accessors expose them live
//   online.setObjectsHook(publish);        // primary output
//   LogFrameSource src(reader, calib);     // or GmdFrameSource(calib) for the live path
//   online.attach(src);                    // route the source's synced frames into onSynced
//   online.begin();
//   src.run(start, count, stride);         // or src.pushScan(...) / src.pushImage(...)
//   online.finish();

#include "semantic_pipeline.h"   // Params, stepFrameSynced, FrameHook, ObjectsHook, SyncedFrame
#include "frame_source.h"        // FrameSource (attach binds its synced-frame hook)
#include "voxel_util.h"          // objutil::VSet / voxelOf (consolidate view-voxel confidence)

#include <cstdio>
#include <functional>
#include <string>
#include <utility>

namespace semantic {

class OnlineSemantic {
public:
    // Owns every stage, built from Params: Universe(p.voxel) + setLocalRadius(p.radius),
    // InfClient(p.endpoint), and default-constructed Superpoints/ObjectSeeds/Objects. Which
    // discovery stages actually run is gated by the Params flags (p.superpoints/hdbscan/
    // objects), not by their presence. Non-copyable (owns a non-copyable InfClient).
    explicit OnlineSemantic(const Params& p)
        : p_(p), uni_(p_.voxel), inf_(p_.endpoint) {
        uni_.setLocalRadius(p_.radius);
    }

    void setFrameHook(FrameHook h)     { on_frame_   = std::move(h); }
    void setObjectsHook(ObjectsHook h) { on_objects_ = std::move(h); }
    // Optional sign-egress hook (see semantic_pipeline.h): when set, each frame's 2D masks
    // + a pixel->3D lookup are handed out for async sign-text detection. Dormant otherwise.
    void setSignHook(SignFrameHook h)  { on_sign_    = std::move(h); }

    // Route a FrameSource's time-synced frames into this pipeline. Call once before the
    // source's first push (it builds the source's sync engine). The source is borrowed --
    // it must outlive the run.
    void attach(FrameSource& src) {
        src_ = &src;
        src.setSyncedHook([this](const SyncedFrame& sf) { onSynced(sf); });
    }

    // Live-reference accessors for the owned stages (valid for the pipeline's lifetime) --
    // drivers/visualizers read these instead of holding their own instances.
    Universe&    universe()    { return uni_; }
    InfClient&   inf()         { return inf_; }
    Superpoints& superpoints() { return sp_; }
    ObjectSeeds& seeds()       { return os_; }
    Objects&     objects()     { return obj_; }

    // Prepare the tracker + vocabulary. Call once before the first pushed frame.
    void begin() {
        inf_.setVocab(p_.thing, p_.stuff);
        inf_.reset();                                  // next frame = video frame 0
        uni_.declareVocab(p_.thing, p_.stuff);         // register thing/stuff kinds
        idg_ = InstanceIdGraph{};                      // fresh instance-bridge graph per run
        done_ = 0;
        wall_.reset();
    }

    // Process one time-aligned frame: integrate + vote, then the cadence stages, then
    // the consumer hook (fired AFTER any superpoint refresh so a visualizer sees the
    // current window for this frame).
    void onSynced(const SyncedFrame& sf) {
        const bool  do_sp   = p_.superpoints;
        const bool  do_hdb  = p_.hdbscan;
        // Growing needs superpoints + seeds; objects tier needs growing.
        const bool  do_grow = do_sp && do_hdb && p_.grow;
        const bool  do_objects = do_grow && p_.objects;
        // Fold proposals into the object layer only on the consolidation stride (frames
        // 0, stride, 2*stride, ...) -- adjacent frames are near-duplicate views, so striding
        // makes accumulated support reflect distinct viewpoints. <=1 => every frame.
        const bool  do_consol = do_objects &&
            (p_.consol_stride <= 1 || (done_ % p_.consol_stride) == 0);
        // Instance-guided seeding: only bother maintaining the DVIS id-bridge graph when
        // seeding will actually consult it (HDBSCAN on + the knob set).
        const bool  use_ids = do_hdb && p_.hdb.use_instance_ids;

        // Progress heartbeat printed BEFORE the (potentially slow / server-bound)
        // inference, so you can see the run advance live and, if a server hangs, exactly
        // which frame it stalled on -- the detailed per-phase timing only prints once the
        // frame COMPLETES, which is no use when frame 0's inference never returns.
        std::fprintf(stderr, "[timing] frame %s  ... (img_t %lld)\n",
                     frameNo(done_ + 1).c_str(), (long long)sf.image_t);

        PointPixelMap    m;
        StepTiming       st;
        std::vector<int> gidx;
        // Sign egress: only ask stepFrameSynced for the 2D masks + pixel->3D map when a
        // consumer is attached (the copies are skipped otherwise).
        FrameResult      sign_fr;
        std::vector<int> sign_pix_gidx;
        const bool       do_sign = static_cast<bool>(on_sign_);
        // Integrate + infer + project + stamp per-frame labels.
        stepFrameSynced(uni_, inf_, p_, sf, m, gidx, &st, use_ids ? &idg_ : nullptr,
                        do_sign ? &sign_fr : nullptr, do_sign ? &sign_pix_gidx : nullptr);
        if (do_sign && sign_fr.ok())
            on_sign_(sf, sign_fr, sign_pix_gidx, sign_fr.w, sign_fr.h, uni_);
        prof_.add("read+integrate", st.read_integrate);
        prof_.add("infer",          st.infer);
        prof_.add("project",        st.project);
        prof_.add("vote",           st.vote);

        // The per-view working set: visible ∩ live ∩ resolves to a thing class this frame.
        // `gidx` is already the nearest-per-pixel surface (bounded by image resolution), so
        // this set does not grow with map density. All three geometry stages run on this
        // only -- never on stuff/unlabeled geometry. There is no freeze: object regions stay
        // re-observable so proposals keep overlapping them (tier-3 IoU depends on it).
        std::vector<int> active;
        active.reserve(gidx.size());
        for (int g : gidx)
            if (uni_.pointIsThing(g)) active.push_back(g);

        Stopwatch sw;
        double t_sp = 0, t_hdb = 0, t_sor = 0, t_grow = 0, t_consol = 0, t_hook = 0;
        bool   objects_changed = false;

        // Per-frame discovery stack on `active`: seed -> superpoints -> grow -> consolidate.
        // Seeding runs FIRST so the seeds it births are growable + consolidatable this same
        // frame.
        if (do_hdb) {
            os_.seedFromIndices(uni_, active, p_.hdb, use_ids ? &idg_ : nullptr);
            const double t_seed = sw.lap();          // whole seeding lap (SOR + clustering)
            t_sor = os_.lastSorMs();                // SOR sub-cost, measured inside seeding
            t_hdb = t_seed - t_sor;                  // clustering only -> the two sum to t_seed
            prof_.add("sor",     t_sor);
            prof_.add("hdbscan", t_hdb);
        }
        if (do_sp) {
            // VCCS over the FULL view (gidx: front-surface shell incl. unlabeled + stuff), NOT
            // just the thing subset -- so superpoints follow real surfaces and each one is a
            // single object's local geometry. The SAI3D class histogram (built in-superpoint)
            // then supplies the semantics.
            sp_.refreshFromIndices(uni_, gidx, p_.sp);
            prof_.add("superpoints", t_sp = sw.lap());
        }
        if (do_grow) {
            os_.growLocal(uni_, sp_, p_.grow_p, p_.voxel);
            prof_.add("grow", t_grow = sw.lap());
            if (do_consol) {
                // SAI3D global-context path: weight this frame's affinity deposits by how
                // fully each primitive is seen -> build the frame's occupied-voxel shell from
                // `gidx` (the front-surface set) and hand the DVIS id-bridge for the co-touch
                // bonus. Both are skipped (nullptr) when global_context is off, so the legacy
                // single-frame containment path is byte-for-byte unchanged.
                const bool gc = p_.consol_p.global_context;
                objutil::VSet view_vox;
                if (gc) {
                    const float invv = 1.0f / (p_.voxel > 0.0f ? p_.voxel : 0.05f);
                    if (Universe::Cloud::ConstPtr cl = uni_.cloud()) {
                        view_vox.reserve(gidx.size());
                        for (int gg : gidx)
                            if (gg >= 0 && gg < (int)cl->size())
                                view_vox.insert(objutil::voxelOf((*cl)[gg], invv));
                    }
                }
                InstanceIdGraph* cidg = (gc && p_.consol_p.use_instance_ids) ? &idg_ : nullptr;
                obj_.consolidate(uni_, os_, p_.consol_p, p_.voxel,
                                  gc ? &view_vox : nullptr, cidg);
                const double lc = sw.lap();
                prof_.add("consolidate", lc);
                t_consol += lc;
                objects_changed = true;
            }
        }
        // Publish the OBJECTS tier whenever it changed this frame (seed and/or grow).
        if (objects_changed && on_objects_) on_objects_(obj_.list(), uni_);
        if (on_frame_) { on_frame_(done_, sf, m); prof_.add("hook", t_hook = sw.lap()); }
        ++done_;

        // Per-frame progress + the residual image/scan offset the sync now absorbs:
        // scan_dt is (attached scan time - image time); the camera pose used is
        // interpolated to image time regardless, so this is just visibility.
        // Show "done/total" when the target frame count is known (count > 0).
        const double frame_ms = st.total() + t_sp + t_sor + t_hdb + t_grow + t_consol + t_hook;
        std::fprintf(stderr,
            "[timing] frame %s | img_t %lld scan_dt %+6.1fms | integ %6.1f  infer %6.1f  "
            "proj %6.1f  vote %6.1f  sp %6.1f  sor %6.1f  hdb %6.1f  grow %6.1f  cons %6.1f  "
            "hook %6.1f | %7.1f ms\n",
            frameNo(done_).c_str(), (long long)sf.image_t, (double)(sf.scan_t - sf.image_t) / 1e6,
            st.read_integrate, st.infer, st.project, st.vote, t_sp, t_sor, t_hdb, t_grow,
            t_consol, t_hook, frame_ms);
    }

    // End-of-run: publish the final object list + print the timing report. The discovery
    // stack now runs EVERY frame on the per-view set, so there is no separate whole-map
    // final pass -- the last onSynced already produced the final objects.
    void finish() {
        const bool do_objects = p_.objects && p_.hdbscan && p_.superpoints && p_.grow;
        // SAI3D merge_small_segs: one end-of-run cleanup pass that absorbs tiny components
        // into their best-affinity neighbor (no-op unless global_context + small_seg_min>0).
        if (do_objects) obj_.mergeSmallComps(uni_, p_.consol_p);
        if (do_objects && on_objects_) on_objects_(obj_.list(), uni_);
        char title[64];
        std::snprintf(title, sizeof(title), "run summary (%d frames)", done_);
        prof_.report(title, wall_.elapsed());
    }

    int frames() const { return done_; }

private:
    // "n/total" for progress lines (or just "n" when the target count is unknown). total =
    // ceil(count / stride): the ingestor sub-samples the image window by stride, so that is
    // how many frames actually get processed.
    std::string frameNo(int n) const {
        const int step = p_.stride > 0 ? p_.stride : 1;
        char b[24];
        if (p_.count > 0) std::snprintf(b, sizeof(b), "%d/%d", n, (p_.count + step - 1) / step);
        else              std::snprintf(b, sizeof(b), "%d", n);
        return b;
    }

    // Owned stages (declaration order = construction order; p_ first as uni_/inf_ read it).
    Params         p_;
    Universe       uni_;
    InfClient      inf_;
    Superpoints    sp_;
    ObjectSeeds    os_;
    Objects        obj_;

    // The attached input transport (borrowed; set in attach(), must outlive the run).
    FrameSource*   src_ = nullptr;

    // Persistent DVIS instance-id bridge: unions same-class, physically-touching instance
    // ids across the run so instance-guided seeding groups by resolved (bridged) instance.
    // Fed in stepFrameSynced, consulted in seedFromIndices; reset each begin().
    InstanceIdGraph idg_;

    FrameHook     on_frame_;
    ObjectsHook   on_objects_;
    SignFrameHook on_sign_;

    int       done_ = 0;
    Profile   prof_;
    Stopwatch wall_;
};

} // namespace semantic
