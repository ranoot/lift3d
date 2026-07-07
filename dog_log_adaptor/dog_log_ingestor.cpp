#include "dog_log_ingestor.h"

#include <algorithm>
#include <vector>

namespace {
// Guard band (ns) of poses pushed on each side of the selected image window, so the
// boundary images are bracketed. 0.5 s >> one scan period (100 ms), so it always
// covers the two poses straddling an image.
constexpr int64_t GUARD_NS = 500'000'000LL;
} // namespace

long DogLogIngestor::run(DogStream& stream, int start_image, int max_images,
                         int stride) const {
    scans_pushed_ = images_pushed_ = 0;
    const int nimg  = reader_.numImages();
    const int nscan = reader_.size();
    if (nimg == 0 || nscan == 0) { stream.flush(); return 0; }

    const int start = std::max(0, start_image);
    const int end   = (max_images < 0) ? nimg : std::min(nimg, start + max_images);
    const int step  = std::max(1, stride);           // sub-sample every `step`-th image
    if (start >= end) { stream.flush(); return 0; }

    const int64_t lo_guard = reader_.imageTimestamp(start)   - GUARD_NS;
    const int64_t hi_guard = reader_.imageTimestamp(end - 1) + GUARD_NS;

    auto push_scan = [&](int i) {
        double pose6[6];
        std::vector<float>         xyz;
        std::vector<unsigned char> inten;
        reader_.scanRaw(i, pose6, xyz, inten);
        stream.pushScan(reader_.timestamp(i), pose6, xyz.data(), inten.data(),
                        (int)(xyz.size() / 3));
        ++scans_pushed_;
    };
    auto push_image = [&](int k) {
        stream.pushImage(reader_.imageByIndex(k));
        ++images_pushed_;
    };

    // Advance the scan cursor to the first scan inside the guard band.
    int si = 0;
    while (si < nscan && reader_.timestamp(si) < lo_guard) ++si;
    int ki = start;

    // Merge-walk the two streams by timestamp; scans win ties so an image is never
    // pushed before a pose at the same instant (keeps it bracketable).
    while (true) {
        const bool have_scan = si < nscan && reader_.timestamp(si) <= hi_guard;
        const bool have_img  = ki < end;
        if (!have_scan && !have_img) break;
        if (have_scan && (!have_img || reader_.timestamp(si) <= reader_.imageTimestamp(ki))) {
            push_scan(si++);
        } else {
            // Keep dense scan interleaving, but only push every `step`-th image so the
            // pipeline runs on the sub-sampled window (poses stay fully available).
            if (((ki - start) % step) == 0) push_image(ki);
            ++ki;
        }
    }

    stream.flush();
    return images_pushed_;
}
