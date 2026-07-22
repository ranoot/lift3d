#pragma once
// IDetector: the narrow, dependency-free seam between SignBridge (which decides WHAT to send)
// and the thing that actually reads a sign (which decides HOW).
//
// Two backends implement it, selected at build time by the LIFT3D_USE_SIGN CMake option:
//   * sign_detector_real.cpp  (ON)  -- wraps the teammate's SignUnderstanding::
//                                      ConcurrentDetectorWrapper (VLM over HTTP, ~1-2 s).
//                                      Needs OpenCV/CURL/libjpeg/libpng/nlohmann_json.
//   * sign_detector_mock.cpp  (OFF) -- a sleep-then-canned-answer worker thread. No third-party
//                                      deps, so the whole sign path (detection, gating, logging,
//                                      object annotation, rerun viz) is runnable in the default
//                                      build. Never enabled unless run.yaml sets sign.enabled.
//
// Because this header names no SignUnderstanding / GMD / OpenCV type, sign_bridge.cpp compiles
// identically in both builds.

#include "sign_config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace sign {

// One sign to read. `rgb` is BORROWED for the duration of the enqueue() call only -- a backend
// that defers work must copy it.
struct SignRequest {
    int                  id   = 0;        // our match id; echoed back verbatim in the result
    std::int64_t         t_ns = 0;        // image capture instant (the module syncs on this)
    const unsigned char* rgb  = nullptr;  // packed RGB8, w*h*3 bytes
    int                  w    = 0;
    int                  h    = 0;
    double               bbox[4] = {0, 0, 0, 0};   // umin, vmin, umax, vmax (pixels)
};

class IDetector {
public:
    // (id, ok, text) -- called on a WORKER thread. `text` is the flattened sign content
    // ("content: direction; ..."), empty when nothing was read.
    using ResultFn = std::function<void(int id, bool ok, const std::string& text)>;

    virtual ~IDetector() = default;
    virtual void setResultCallback(ResultFn fn) = 0;
    virtual void start()                        = 0;
    virtual void stop()                         = 0;
    virtual bool enqueue(const SignRequest& req) = 0;   // false => rejected (queue full)
    virtual const char* backend() const          = 0;   // "sign-unds" | "mock", for the logs
};

// Defined by exactly one of the two backend TUs.
std::unique_ptr<IDetector> makeDetector(const SignConfig& cfg);

}  // namespace sign
