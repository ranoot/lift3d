#pragma once
// OnlineSemantic: the pipeline loop, driven ONE SYNCED FRAME AT A TIME. It is the
// bridge between DogStream (which emits time-aligned SyncedFrames) and the semantic
// Universe: for each synced frame it integrates + votes (stepFrameSynced), and on a
// cadence recomputes VCCS superpoints / Mask3D features / HDBSCAN* object seeds. Its
// primary output is the list of discovered objects (each a subset of the global map
// + a centroid), delivered through onObjects.
//
// This replaces the old reader-driven semantic::accumulate() loop: instead of pulling
// frames from a file, the pipeline is PUSHED frames. The same OnlineSemantic serves
// the live path and the offline simulated ingestor -- the only difference is who calls
// DogStream::pushScan/pushImage. Visualizers and publishers attach as consumers via
// setFrameHook / setObjectsHook and never touch the pipeline internals.
//
// Usage:
//   OnlineSemantic online(uni, inf, p, &sp, &pf, &feat, &os);
//   online.setObjectsHook(publish);            // primary output
//   DogStream stream(calib, online.hook());    // wire the synced-frame callback
//   online.begin();
//   ... drive stream.pushScan / stream.pushImage (live or ingestor) ...
//   stream.flush();
//   online.finish();

#include "semantic_pipeline.h"   // Params, stepFrameSynced, FrameHook, ObjectsHook, SyncedFrame

#include <cstdio>
#include <functional>
#include <utility>

namespace semantic {

class OnlineSemantic {
public:
    // sp/pf/feat/os are optional feature stages (nullptr => that stage is off, same
    // gating as the Params flags). uni/inf are borrowed for the pipeline's lifetime.
    OnlineSemantic(Universe& uni, InfClient& inf, const Params& p,
                   Superpoints* sp = nullptr, PointFeatures* pf = nullptr,
                   FeatClient* feat = nullptr, ObjectSeeds* os = nullptr)
        : uni_(uni), inf_(inf), p_(p), sp_(sp), pf_(pf), feat_(feat), os_(os) {}

    void setFrameHook(FrameHook h)     { on_frame_   = std::move(h); }
    void setObjectsHook(ObjectsHook h) { on_objects_ = std::move(h); }

    // The DogStream synced-frame callback (bind once, pass to DogStream's ctor).
    DogStream::SyncedHook hook() {
        return [this](const SyncedFrame& sf) { onSynced(sf); };
    }

    // Prepare the tracker + vocabulary. Call once before the first pushed frame.
    void begin() {
        inf_.setVocab(p_.thing, p_.stuff);
        inf_.reset();                                  // next frame = video frame 0
        uni_.declareVocab(p_.thing, p_.stuff);         // register thing/stuff kinds
        uni_.setLabelGate(p_.min_votes, p_.min_conf);  // gate applied at read-time
        done_ = 0;
        wall_.reset();
    }

    // Process one time-aligned frame: integrate + vote, then the cadence stages, then
    // the consumer hook (fired AFTER any superpoint refresh so a visualizer sees the
    // current window for this frame).
    void onSynced(const SyncedFrame& sf) {
        const bool  do_sp   = sp_ && p_.superpoints;
        const bool  do_feat = pf_ && feat_ && p_.features;
        const bool  do_hdb  = os_ && p_.hdbscan;
        // Feature-guided growing needs all three stages present + its own flag.
        const bool  do_grow = do_feat && do_sp && do_hdb && p_.grow;
        const float sp_r    = p_.sp_radius   > 0.0f ? p_.sp_radius   : p_.radius;
        const float feat_r  = p_.feat_radius > 0.0f ? p_.feat_radius : p_.radius;

        PointPixelMap m;
        StepTiming    st;
        stepFrameSynced(uni_, inf_, p_, sf, m, &st);
        prof_.add("read+integrate", st.read_integrate);
        prof_.add("infer",          st.infer);
        prof_.add("project",        st.project);
        prof_.add("vote",           st.vote);

        Stopwatch sw;
        double t_sp = 0, t_feat = 0, t_hdb = 0, t_grow = 0, t_hook = 0;
        bool   objects_changed = false;

        // SEED new objects from unclaimed points over the WHOLE map (heavy) -> coarser
        // cadence. Runs BEFORE growing so fresh seeds are growable this same frame.
        if (do_hdb && p_.hdbscan_every > 0 && (done_ % p_.hdbscan_every) == 0) {
            os_->seedUnclaimed(uni_, p_.hdb);
            prof_.add("hdbscan", t_hdb = sw.lap());
            objects_changed = true;
        }
        // REFINE cadence: features -> superpoints -> grow, which must co-occur (the
        // affinity merge consumes both). Features still refresh here even if
        // superpoints/grow are off, preserving the standalone feature path.
        if (p_.refine_every > 0 && (done_ % p_.refine_every) == 0) {
            if (do_feat) {
                // Whole-object features: pull in-radius objects' points (which may spill
                // past the crop) into the backbone call so their mean feature is complete.
                std::vector<int> extra;
                if (do_grow) os_->collectInRadiusPoints(uni_.robotWorld(), p_.radius, extra);
                pf_->refreshLocal(uni_, *feat_, uni_.robotWorld(), feat_r, p_.feat,
                                  do_grow ? &extra : nullptr);
                prof_.add("features", t_feat = sw.lap());
            }
            if (do_sp) {
                sp_->refreshLocal(uni_, uni_.robotWorld(), sp_r, p_.sp, pf_);
                prof_.add("superpoints", t_sp = sw.lap());
            }
            if (do_grow) {
                os_->growLocal(uni_, *sp_, *pf_, p_.grow_p, uni_.robotWorld(), p_.radius,
                               p_.voxel);
                prof_.add("grow", t_grow = sw.lap());
                objects_changed = true;
            }
        }
        // Publish the object list whenever the store changed this frame (seed and/or grow).
        if (objects_changed && on_objects_) on_objects_(os_->list(), uni_);
        if (on_frame_) { on_frame_(done_, sf, m); prof_.add("hook", t_hook = sw.lap()); }
        ++done_;

        // Per-frame progress + the residual image/scan offset the sync now absorbs:
        // scan_dt is (attached scan time - image time); the camera pose used is
        // interpolated to image time regardless, so this is just visibility.
        // Show "done/total" when the target frame count is known (count > 0).
        const double frame_ms = st.total() + t_sp + t_feat + t_hdb + t_grow + t_hook;
        char fno[24];
        if (p_.count > 0) {
            // Frames actually processed = ceil(count / stride) (stride sub-samples the
            // image window in the ingestor); show that as the denominator.
            const int step  = p_.stride > 0 ? p_.stride : 1;
            const int total = (p_.count + step - 1) / step;
            std::snprintf(fno, sizeof(fno), "%5d/%d", done_, total);
        } else {
            std::snprintf(fno, sizeof(fno), "%5d", done_);
        }
        std::fprintf(stderr,
            "[timing] frame %s | img_t %lld scan_dt %+6.1fms | integ %6.1f  infer %6.1f  "
            "proj %6.1f  vote %6.1f  sp %6.1f  feat %6.1f  hdb %6.1f  grow %6.1f  hook %6.1f | %7.1f ms\n",
            fno, (long long)sf.image_t, (double)(sf.scan_t - sf.image_t) / 1e6,
            st.read_integrate, st.infer, st.project, st.vote, t_sp, t_feat, t_hdb, t_grow,
            t_hook, frame_ms);
    }

    // Final whole-map refreshes + the end-of-run timing report. Call once after the
    // last pushed frame (and after DogStream::flush()).
    void finish() {
        const bool  do_sp   = sp_ && p_.superpoints;
        const bool  do_feat = pf_ && feat_ && p_.features;
        const bool  do_hdb  = os_ && p_.hdbscan;
        const bool  do_grow = do_feat && do_sp && do_hdb && p_.grow;
        const float sp_r    = p_.sp_radius   > 0.0f ? p_.sp_radius   : p_.radius;
        const float feat_r  = p_.feat_radius > 0.0f ? p_.feat_radius : p_.radius;

        Stopwatch sw;
        // Final pass at the last robot pose: seed any remaining unclaimed points, then
        // refresh features/superpoints and grow once more so the store is complete.
        if (do_hdb) { os_->seedUnclaimed(uni_, p_.hdb); prof_.add("hdbscan", sw.lap()); }
        if (do_feat) {
            std::vector<int> extra;
            if (do_grow) os_->collectInRadiusPoints(uni_.robotWorld(), p_.radius, extra);
            pf_->refreshLocal(uni_, *feat_, uni_.robotWorld(), feat_r, p_.feat,
                              do_grow ? &extra : nullptr);
            prof_.add("features", sw.lap());
        }
        if (do_sp)   { sp_->refreshLocal(uni_, uni_.robotWorld(), sp_r, p_.sp, pf_); prof_.add("superpoints", sw.lap()); }
        if (do_grow) {
            os_->growLocal(uni_, *sp_, *pf_, p_.grow_p, uni_.robotWorld(), p_.radius, p_.voxel);
            prof_.add("grow", sw.lap());
        }
        if (do_hdb && on_objects_) on_objects_(os_->list(), uni_);
        char title[64];
        std::snprintf(title, sizeof(title), "run summary (%d frames)", done_);
        prof_.report(title, wall_.elapsed());
    }

    int frames() const { return done_; }

private:
    Universe&      uni_;
    InfClient&     inf_;
    Params         p_;
    Superpoints*   sp_;
    PointFeatures* pf_;
    FeatClient*    feat_;
    ObjectSeeds*   os_;

    FrameHook   on_frame_;
    ObjectsHook on_objects_;

    int       done_ = 0;
    Profile   prof_;
    Stopwatch wall_;
};

} // namespace semantic
