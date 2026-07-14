#pragma once
// FrameSource: the abstract input front end for the semantic pipeline. It owns the
// DogStream sync engine (calib + PoseInterpolator + image bracketing -> SyncedFrame) and
// hides it behind a single virtual base, so the pipeline never sees the concrete transport.
//
// A concrete subclass adds only its SOURCE-SPECIFIC typed ingest methods (native message
// -> raw pushScan/pushImage on engine()); everything downstream -- interpolation, time
// alignment, world lifting -- is shared. See:
//   * GmdFrameSource (gmd_adaptor/gmd_frame_source.h) -- live RA/EAICommon message types.
//   * LogFrameSource (dog_log_adaptor/log_frame_source.h) -- offline ~/dog_logs replay.
//
// Wiring (OnlineSemantic::attach does the first two for you):
//   src.setSyncedHook(hook);   // MUST be called before any push -- builds the engine
//   src.pushScan(...); src.pushImage(...);   // (typed, on the concrete subclass)
//   src.flush();
//
// The DogStream engine's ctor needs the synced-frame hook, which is not known until
// attach() time, so the engine is built lazily in setSyncedHook(). Calling engine()
// (hence any push) before setSyncedHook() dereferences an empty optional -- always attach
// (or setSyncedHook) first.

#include "dog_stream.h"          // DogStream, SyncedFrame, CameraCalib (via dog_log_adaptor.h)

#include <optional>
#include <utility>

class FrameSource {
public:
    explicit FrameSource(CameraCalib calib) : calib_(std::move(calib)) {}
    virtual ~FrameSource() = default;

    FrameSource(const FrameSource&)            = delete;   // owns a DogStream; not copyable
    FrameSource& operator=(const FrameSource&) = delete;

    // Route synced frames into a consumer. Called once by OnlineSemantic::attach (or by a
    // test) BEFORE the first push; (re)builds the engine bound to `h`.
    void setSyncedHook(DogStream::SyncedHook h) { engine_.emplace(calib_, std::move(h)); }

    // End-of-stream: flush any still-bracketable pending image. No-op before setSyncedHook.
    void flush() { if (engine_) engine_->flush(); }

    // Diagnostics passthrough (valid after setSyncedHook).
    long emitted()         const { return engine_ ? engine_->emitted() : 0; }
    long droppedTooOld()   const { return engine_ ? engine_->droppedTooOld() : 0; }
    long droppedOverflow() const { return engine_ ? engine_->droppedOverflow() : 0; }

protected:
    // The shared sync engine. Valid only after setSyncedHook(); a subclass's typed ingest
    // converts its native message and forwards to engine().pushScan / engine().pushImage.
    DogStream& engine() { return *engine_; }

private:
    CameraCalib              calib_;
    std::optional<DogStream> engine_;   // built in setSyncedHook (DogStream needs the hook)
};
