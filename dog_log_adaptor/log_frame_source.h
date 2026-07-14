#pragma once
// LogFrameSource: the OFFLINE FrameSource. Replays a ~/dog_logs recording into the shared
// DogStream sync engine through the exact same pushScan/pushImage calls a live sensor
// driver would use -- so replay and live share one downstream code path. It is a thin
// FrameSource skin over the existing DogLogIngestor (which self-drives the file), exposed
// as run() rather than typed per-message pushes because the log is a pull source.
//
// Usage (OnlineSemantic::attach wires the synced hook):
//   LogFrameSource src(reader, calib);
//   online.attach(src);
//   online.begin();
//   src.run(start, count, stride);   // replays the window; calls flush() at the end
//   online.finish();

#include "frame_source.h"        // FrameSource
#include "dog_log_adaptor.h"     // DogLogReader
#include "dog_log_ingestor.h"    // DogLogIngestor

#include <utility>

class LogFrameSource : public FrameSource {
public:
    // reader must already have openCamera()'d the gclf stream; it is borrowed (by the
    // ingestor) for the source's lifetime, so it must outlive this object.
    LogFrameSource(const DogLogReader& reader, CameraCalib calib)
        : FrameSource(std::move(calib)), ing_(reader) {}

    // Replay images [start, start+count) (count < 0 => all), sub-sampling by stride, into
    // the sync engine. Requires setSyncedHook (attach) to have run first. Flushes at the
    // end. Returns the number of images actually pushed.
    long run(int start = 0, int count = -1, int stride = 1) {
        return ing_.run(engine(), start, count, stride);
    }

    long scansPushed()  const { return ing_.scansPushed(); }
    long imagesPushed() const { return ing_.imagesPushed(); }

private:
    DogLogIngestor ing_;
};
