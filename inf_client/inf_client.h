#pragma once
// C++ client for the Python inf_server (OV-DVIS++ 2D panoptic segmentation),
// spoken over ZeroMQ (REQ) + msgpack. Inference cannot run in C++, so the model
// lives behind a Python IPC server; this is the consumer side that the lift3d
// point-to-pixel lifting drives (see inf_server/protocol.py).
//
// Usage:
//   InfClient inf("ipc:///tmp/inf_server.ipc");
//   inf.setVocab({"door","chair"}, {"wall","floor"});   // returns num_classes
//   FrameResult r = inf.frame(rgb, H, W);                // rgb = HxWx3 uint8
//   int16_t cls = r.labelAt(u, v);                       // class id, -1 = bg
//   std::string name = inf.className(cls);               // stable class name

#include <cstdint>
#include <string>
#include <vector>

// One instance/query returned for a frame (id matches values in id_map).
struct InfInstance {
    int                id    = -1;
    int                label = -1;   // class id, indexes the ordered vocab
    float              score = 0.0f;
    std::vector<float> embedding;     // (C,) query embedding
};

// Per-pixel + per-instance result for one frame. Maps are at the input (H,W).
struct FrameResult {
    int h = 0, w = 0;
    std::vector<int16_t>     label_map;   // h*w row-major, class id, -1 = background
    std::vector<int16_t>     id_map;      // h*w row-major, instance id, -1 = background
    std::vector<InfInstance> instances;

    bool ok() const { return h > 0 && w > 0 && label_map.size() == (size_t)h * w; }
    // Class id at pixel (u=col, v=row). Caller guards on visibility/in-frame.
    int16_t labelAt(int u, int v) const { return label_map[(size_t)v * w + u]; }
    int16_t idAt(int u, int v)    const { return id_map[(size_t)v * w + u]; }
};

class InfClient {
public:
    // Connects a REQ socket to the server endpoint (does not block on the model).
    explicit InfClient(const std::string& endpoint = "ipc:///tmp/inf_server.ipc",
                       int recv_timeout_ms = 600000);
    ~InfClient();
    InfClient(const InfClient&) = delete;
    InfClient& operator=(const InfClient&) = delete;

    bool ping();

    // Set the open-vocabulary classes. The server orders labels as
    // sorted(thing) ++ sorted(stuff); we store that same order so className()
    // can turn a returned label id back into its stable name. Returns num_classes.
    int  setVocab(const std::vector<std::string>& thing,
                  const std::vector<std::string>& stuff);

    void reset();  // start a fresh tracking sequence (next frame = video frame 0)

    // Run one RGB frame (HxWx3 uint8, row-major) through the model.
    FrameResult frame(const uint8_t* rgb, int h, int w);
    FrameResult frame(const std::vector<uint8_t>& rgb, int h, int w) {
        return frame(rgb.data(), h, w);
    }

    // Stable class name for a label id (as returned in label_map / instances).
    // Empty string for -1 or out-of-range.
    const std::string& className(int label) const;
    const std::vector<std::string>& vocab() const { return vocab_; }

private:
    struct Impl;              // hides the cppzmq context/socket from consumers
    Impl* impl_;
    std::vector<std::string> vocab_;         // sorted(thing) ++ sorted(stuff)
    std::string              empty_;         // returned for out-of-range labels
};
