#pragma once
// -------------------------------------------------------------------------------------------------
// SignBridge: egress glue between the lift3d pipeline and the sign-understanding module.
//
// Per synced frame (via OnlineSemantic::setSignHook) it:
//   1. scans the frame's 2D masks for a sign-class instance and, IF no request is currently in
//      flight (the agreed single-in-flight gate -- reading a sign takes ~1-2 s, so frames arriving
//      during processing are dropped),
//   2. computes that sign's pixel bounding box + its 3D voxel footprint (via the pixel->3D map),
//      stashes the footprint under a freshly generated match id, and
//   3. dispatches (image, bbox, timestamp, id) to the detector's worker thread (non-blocking).
//
// The detector answers asynchronously, keyed by the same id (guaranteed by the interface). We move
// the footprint + text into `resolved_`. Later, annotate() attaches each resolved sign's text to
// the object whose 3D footprint best encompasses it (biggest proposal wins on conflict), yielding
// the directionContent for the published EAIRoomObject and the sign label drawn by the rerun viz.
//
// The detector itself is behind the IDetector seam (sign_detector.h): the real SignUnderstanding
// VLM wrapper under LIFT3D_USE_SIGN, a sleep-then-canned-answer mock otherwise. This TU therefore
// needs no OpenCV/CURL/GMD headers and is built in every configuration.
//
// Every stage logs to stderr with a `[sign]` prefix -- dispatch, gate drops, results and the
// object each result lands on -- so a run shows the whole path without a debugger.
//
// Thread-safety: onFrame() runs on the pipeline thread; onResult() runs on a detector worker
// thread; annotate() on the pipeline thread. All shared state is guarded by mu_.
// -------------------------------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "sign_config.h"     // sign::SignConfig
#include "sign_detector.h"   // sign::IDetector
#include "universe.h"        // Universe
#include "objects.h"         // Object
#include "voxel_util.h"      // objutil::VSet
#include "inf_client.h"      // FrameResult, InfClient
#include "dog_stream.h"      // SyncedFrame

namespace sign {

class SignBridge {
public:
    using Config = SignConfig;   // kept as a member alias: callers wrote sign::SignBridge::Config

    explicit SignBridge(const SignConfig& cfg);
    ~SignBridge();
    SignBridge(const SignBridge&)            = delete;
    SignBridge& operator=(const SignBridge&) = delete;

    // Bound to OnlineSemantic::setSignHook (the runner supplies `inf` + `voxel`). Picks at most
    // one sign instance from this frame and, if the gate is open, dispatches it. Never blocks.
    void onFrame(const SyncedFrame& sf, const FrameResult& fr,
                 const std::vector<int>& pix_gidx, int W, int H,
                 const Universe& uni, const InfClient& inf, float voxel);

    // object id -> sign text for every object that encompasses a resolved sign. Objects absent
    // from the map carry no sign text. Cheap to call every frame: returns immediately while no
    // sign has resolved, and otherwise serves a cached mapping that is recomputed when a new sign
    // arrives (or every `kAnnotateStride` calls, so growing/merging objects are picked up).
    std::unordered_map<int, std::string> annotate(const std::vector<Object>& objs,
                                                   const Universe& uni, float voxel);

    // Diagnostics for the runner's end-of-run summary.
    std::size_t dispatched();
    std::size_t droppedBusy();
    std::size_t resolvedCount();
    std::size_t failedCount();
    std::size_t inFlight();

private:
    void onResult(int id, bool ok, const std::string& text);   // detector worker thread

    struct Resolved {
        objutil::VSet footprint;   // the sign's 3D voxels, recorded at dispatch time
        std::string   text;        // "content: direction; ..." joined instructions
    };

    static constexpr int kAnnotateStride = 10;   // re-match every N calls as objects grow

    SignConfig                 cfg_;
    std::unique_ptr<IDetector> det_;

    std::mutex                             mu_;
    int                                    inflight_ = 0;   // 0 or 1 (single-in-flight gate)
    std::int64_t                           inflight_ms_ = 0;  // dispatch instant of the in-flight id
    int                                    inflight_id_ = 0;
    std::uint64_t                          next_id_  = 1;
    std::unordered_map<int, objutil::VSet> pending_;        // dispatched, awaiting a result
    std::vector<Resolved>                  resolved_;       // results ready to annotate

    std::size_t n_dispatched_ = 0, n_dropped_ = 0, n_failed_ = 0;
    int         frames_       = 0;

    // annotate() cache + the "already logged this attachment" set (so the attach line prints on
    // change, not once per frame).
    std::uint64_t                        gen_ = 0, cached_gen_ = ~0ULL;
    int                                  calls_since_ = 0;
    std::unordered_map<int, std::string> cached_;
    std::unordered_map<int, std::string> logged_;
};

}  // namespace sign
