// Real sign detector -- the backend compiled when LIFT3D_USE_SIGN is ON. Thin adapter from our
// dependency-free IDetector seam onto the teammate's SignUnderstanding::ConcurrentDetectorWrapper
// (bounded queue + worker threads + VLM HTTP call, ~1-2 s per sign).
//
// Everything that needs the teammate's headers or the GMD entity types lives HERE, so
// sign_bridge.cpp stays buildable without OpenCV/CURL/nlohmann_json.

#include "sign_detector.h"

#include <chrono>
#include <cstdio>

#include "SignUnderstanding/ConcurrentDetectorWrapper.hpp"   // ConcurrentDetectorWrapper, ProcessResult
#include "SignUnderstanding/Types.hpp"                       // SignBBoxInput, SignContentOutput
#include "Common/Entity/Image.h"                             // Common::Entity::Image
#include "Common/ConstantsEnum.h"                            // ImageEncoding

namespace sign {
namespace {

// Join a sign's content/direction instructions into one directionContent string:
// "content: direction; content2: direction2" (content alone when direction is empty).
std::string formatInstructions(const SignUnderstanding::SignContentOutput& out) {
    std::string s;
    for (const SignUnderstanding::SignInstruction& in : out.instructions) {
        if (in.content.empty() && in.direction.empty()) continue;
        if (!s.empty()) s += "; ";
        s += in.content;
        if (!in.direction.empty()) { s += ": "; s += in.direction; }
    }
    return s;
}

class RealDetector final : public IDetector {
public:
    explicit RealDetector(const SignConfig& cfg)
        : det_(cfg.vlm_url, cfg.num_workers, cfg.queue_size) {}
    ~RealDetector() override { det_.Stop(); }

    void setResultCallback(ResultFn fn) override {
        cb_ = std::move(fn);
        det_.SetResultCallback([this](const SignUnderstanding::ProcessResult& r) {
            if (!cb_) return;
            const bool ok = r.success && r.output.has_value();
            cb_(r.instanceId, ok, ok ? formatInstructions(*r.output) : std::string{});
        });
    }

    void start() override { det_.Start(); }
    void stop()  override { det_.Stop(); }

    bool enqueue(const SignRequest& req) override {
        // image.timestamp must match bboxInput.timestamp exactly (DetectorWrapper::ProcessSign
        // checks it), so derive ONE TimeStamp from the frame instant and use it for both.
        const Common::Entity::TimeStamp stamp{
            std::chrono::duration_cast<Common::Entity::TimeClock::duration>(
                std::chrono::nanoseconds(req.t_ns))};

        Common::Entity::Image img;
        img.timestamp = stamp;
        img.height    = req.h;
        img.width     = req.w;
        img.encoding  = ::ImageEncoding::RGB8;
        img.step      = req.w * 3;
        img.data.assign(req.rgb, req.rgb + static_cast<std::size_t>(req.w) * req.h * 3);

        SignUnderstanding::SignBBoxInput in;
        in.instanceId = req.id;
        in.timestamp  = stamp;
        in.bbox       = {req.bbox[0], req.bbox[1], req.bbox[2], req.bbox[3]};
        return det_.Enqueue(img, in);
    }

    const char* backend() const override { return "sign-unds"; }

private:
    SignUnderstanding::ConcurrentDetectorWrapper det_;
    ResultFn                                     cb_;
};

}  // namespace

std::unique_ptr<IDetector> makeDetector(const SignConfig& cfg) {
    std::fprintf(stderr, "[sign] backend=sign-unds vlm=%s workers=%d queue=%d\n",
                 cfg.vlm_url.c_str(), cfg.num_workers, cfg.queue_size);
    return std::make_unique<RealDetector>(cfg);
}

}  // namespace sign
