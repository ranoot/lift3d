#pragma once
// -------------------------------------------------------------------------------------------------
// SignBridge: egress glue between the lift3d pipeline and a teammate's async sign-understanding
// module (sign-unds/, SignUnderstanding::ConcurrentDetectorWrapper). It is compiled only under the
// LIFT3D_USE_SIGN CMake option.
//
// Per synced frame (via OnlineSemantic::setSignHook) it:
//   1. scans the frame's 2D masks for a sign-class instance and, IF no request is currently in
//      flight (the agreed single-in-flight gate -- the VLM takes ~1-2 s, so frames arriving during
//      processing are dropped),
//   2. computes that sign's pixel bounding box + its 3D voxel footprint (via the pixel->3D map),
//      stashes the footprint under a freshly generated match id, and
//   3. dispatches (image, bbox, timestamp, id) to the detector's worker thread (non-blocking).
//
// The detector returns SignContentOutput asynchronously, keyed by the same id (guaranteed by the
// interface). We move the footprint + formatted text into `resolved_`. Later, annotate() attaches
// each resolved sign's text to the object whose 3D footprint best encompasses it (biggest proposal
// wins on conflict), yielding the directionContent for the published EAIRoomObject.
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

#include "universe.h"        // Universe
#include "objects.h"         // Object
#include "voxel_util.h"      // objutil::VSet
#include "inf_client.h"      // FrameResult, InfClient
#include "dog_stream.h"      // SyncedFrame

namespace SignUnderstanding {
class ConcurrentDetectorWrapper;
struct ProcessResult;
}  // namespace SignUnderstanding

namespace sign {

class SignBridge {
public:
    struct Config {
        std::string              vlm_url      = "http://localhost:8080/v1";
        std::vector<std::string> sign_classes = {"sign"};  // vocab names treated as signs
        int                      num_workers  = 1;
        int                      queue_size   = 4;
        int                      min_sign_px  = 60;         // ignore sign masks smaller than this
    };

    explicit SignBridge(const Config& cfg);
    ~SignBridge();
    SignBridge(const SignBridge&)            = delete;
    SignBridge& operator=(const SignBridge&) = delete;

    // Bound to OnlineSemantic::setSignHook (the runner supplies `inf` + `voxel`). Picks at most
    // one sign instance from this frame and, if the gate is open, dispatches it. Never blocks.
    void onFrame(const SyncedFrame& sf, const FrameResult& fr,
                 const std::vector<int>& pix_gidx, int W, int H,
                 const Universe& uni, const InfClient& inf, float voxel);

    // object id -> directionContent for every object that encompasses a resolved sign. Objects
    // absent from the map carry no sign text. Recomputed from the persistent resolved-sign set on
    // each call, so a sign that resolves later starts annotating its object on the next objects tick.
    std::unordered_map<int, std::string> annotate(const std::vector<Object>& objs,
                                                   const Universe& uni, float voxel);

    // Diagnostics for the runner's logging.
    std::size_t resolvedCount();
    std::size_t inFlight();

private:
    void onResult(const SignUnderstanding::ProcessResult& r);   // detector worker thread

    struct Resolved {
        objutil::VSet footprint;   // the sign's 3D voxels, recorded at dispatch time
        std::string   text;        // "content: direction; ..." joined instructions
    };

    Config                                                        cfg_;
    std::unique_ptr<SignUnderstanding::ConcurrentDetectorWrapper> det_;

    std::mutex                                    mu_;
    int                                           inflight_ = 0;   // 0 or 1 (single-in-flight gate)
    std::uint64_t                                 next_id_  = 1;
    std::unordered_map<int, objutil::VSet>        pending_;        // dispatched, awaiting a result
    std::vector<Resolved>                         resolved_;       // results ready to annotate
};

}  // namespace sign
