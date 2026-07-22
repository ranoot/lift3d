#pragma once
// VizPublisher: serialize the semantic pipeline's per-frame + final state onto a ZeroMQ
// PUSH socket (msgpack, msgpack_lite.h) for the Python visualizer (viz_server/) to render
// with rerun. This is the C++ side of the visualizer/logic split: run_semantic_universe
// links this instead of the rerun C++ SDK, so the core build carries no rerun/arrow dep.
//
// Transport is PUSH -> PULL, non-blocking: per-frame messages are sent with ZMQ_DONTWAIT
// (dropped on backpressure -- a live viewer only needs the latest), and the final world
// map is sent best-effort. An empty endpoint makes every call a no-op (active() == false),
// so a headless run pays nothing. The wire schema mirrors inf_server/protocol.py's ndarray
// envelope {"__ndarray__",shape,dtype,data} so the Python side rebuilds numpy arrays
// directly; see viz_server/main.py for the message catalogue.
//
// Usage (in run_semantic_universe, mirroring the old viz hook bodies):
//   VizPublisher viz(cfg.viz_endpoint);              // "" => inactive
//   viz.begin(uni.semantics(), seg_cfg);
//   online.setFrameHook([&](int idx, const SyncedFrame& sf, const PointPixelMap&) {
//       viz.frame(idx, capture_secs, sf, seg_ptr, sp, sp_changed, os, obj, uni, flags);
//   });
//   viz.finish(uni, obj);

#include "universe.h"            // Universe, SemanticVocabulary, ClassKind
#include "superpoints.h"         // Superpoints
#include "object_seeds.h"        // ObjectSeeds
#include "objects.h"             // Objects
#include "inf_client.h"          // FrameResult (2D masks for the seg overlay)
#include "dog_stream.h"          // SyncedFrame

#include <string>
#include <unordered_map>
#include <vector>

// Viz-only 2D segmentation overlay config, forwarded to Python in the begin message so it
// paints exactly what run.yaml's `viz:` block configures. Mirrors RunConfig's seg fields.
struct VizSegConfig {
    std::string overlay = "off";   // off | class | instance | both
    float       alpha   = 0.5f;    // fill blend strength (0..1)
    int         min_area = 80;     // min region px before a labelled box is drawn
};

// What to include in a frame message -- the pipeline's discovery-stage gates (p.superpoints
// / p.hdbscan / p.objects), so viz sends only the tiers that actually ran.
struct VizFrameFlags {
    bool superpoints = false;      // include the superpoint window (only when sp_changed)
    bool proposals   = false;      // include this frame's grown proposals
    bool objects     = false;      // include the objects tier + member points
    bool sp_changed  = false;      // VCCS recomputed this frame (sp.version() bumped)
};

class VizPublisher {
public:
    // Connects a PUSH socket to `endpoint` (e.g. ipc:///tmp/lift3d_viz.ipc). An empty
    // endpoint leaves the publisher inactive -- every method is then a no-op.
    explicit VizPublisher(const std::string& endpoint);
    ~VizPublisher();
    VizPublisher(const VizPublisher&) = delete;
    VizPublisher& operator=(const VizPublisher&) = delete;

    bool active() const { return active_; }

    // Run header: the full class vocabulary (id -> name + thing/stuff kind) for the viewer's
    // AnnotationContext legend, plus the seg-overlay config. Send once after begin().
    // `inf_vocab` is the InfClient's ORDERED vocab (sorted(thing) ++ sorted(stuff)) that the
    // 2D seg label_map/id_map class ids index into -- a DIFFERENT id space than `sem` (the
    // Universe registry, declared unsorted), so the Python viewer must resolve seg-overlay
    // labels through this list, not the legend. (3D points are stamped by name upstream, so
    // they already use `sem`.)
    void begin(const SemanticVocabulary& sem, const std::vector<std::string>& inf_vocab,
               const VizSegConfig& seg);

    // One frame's lightweight, time-varying state. `seg` (nullable) carries this frame's 2D
    // masks for the painted overlay; `sp`/`os`/`obj`/`uni` are read live (member points are
    // gathered from the Universe cloud, alive-filtered exactly as the old viz did). `flags`
    // selects which discovery tiers to include. `obj_text` (nullable) maps object id -> free
    // text -- today the sign-understanding result (sign_adaptor/) -- which the viewer draws as a
    // label on that object; objects absent from the map carry no text.
    void frame(int idx, double capture_secs, const SyncedFrame& sf,
               const FrameResult* seg,
               const Superpoints& sp, const ObjectSeeds& os, const Objects& obj,
               const Universe& uni, const VizFrameFlags& flags,
               const std::unordered_map<int, std::string>* obj_text = nullptr);

    // Final world snapshot: every ALIVE point (position + class id + thing/stuff kind) plus
    // the final object list. Sent best-effort at end of run so the viewer can log the static
    // world map + legend-coloured points.
    void finish(const Universe& uni, const Objects& obj);

private:
    struct Impl;               // hides the cppzmq context/socket
    Impl* impl_ = nullptr;
    bool  active_ = false;
};
