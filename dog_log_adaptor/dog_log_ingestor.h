#pragma once
// DogLogIngestor: the OFFLINE "simulated ingestor". It replays a ~/dog_logs recording
// into a DogStream through exactly the same pushScan/pushImage calls a live sensor
// driver would use -- merging the 10 Hz scan stream and the 5 Hz image stream into one
// increasing-timestamp order. That single ordering is what lets DogStream bracket each
// image between two poses, so replay and live share one code path.
//
// It is deliberately SEPARATE from the pipeline (DogStream / OnlineSemantic): swapping
// this out for a ZMQ/ROS source changes nothing downstream.

#include "dog_log_adaptor.h"
#include "dog_stream.h"

#include <cstdint>

class DogLogIngestor {
public:
    // reader must already have openCamera()'d the gclf stream. Borrowed for the call.
    explicit DogLogIngestor(const DogLogReader& reader) : reader_(reader) {}

    // Replay images [start_image, start_image+max_images) into `stream`, interleaving
    // the scans needed to bracket them (a guard band of poses on each side is pushed
    // so the first/last images can be interpolated). max_images < 0 => all images.
    // `stride` sub-samples the images: only every `stride`-th image in the window is
    // pushed (scans are ALWAYS pushed in full so the kept images still interpolate
    // against dense poses). stride <= 1 => every image. Calls stream.flush() at the
    // end. Returns the number of images actually pushed.
    long run(DogStream& stream, int start_image = 0, int max_images = -1,
             int stride = 1) const;

    long scansPushed()  const { return scans_pushed_; }
    long imagesPushed() const { return images_pushed_; }

private:
    const DogLogReader& reader_;
    mutable long scans_pushed_  = 0;
    mutable long images_pushed_ = 0;
};
