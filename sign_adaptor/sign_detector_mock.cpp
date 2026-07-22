// Mock sign detector -- the backend compiled when LIFT3D_USE_SIGN is OFF (i.e. the default
// build, which carries no OpenCV/CURL/VLM dependency). It reproduces the real module's
// observable behaviour: an enqueue returns immediately, a worker thread sleeps for a
// configurable stand-in latency, then the result callback fires on that worker thread with a
// canned text. That is enough to exercise the single-in-flight gate, both log lines, the
// footprint->object annotation and the rerun text overlay end to end.
//
// It is only ever constructed when run.yaml sets sign.enabled, so a normal run is unaffected.

#include "sign_detector.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace sign {
namespace {

class MockDetector final : public IDetector {
public:
    explicit MockDetector(const SignConfig& cfg) : cfg_(cfg) {}
    ~MockDetector() override { stop(); }

    void setResultCallback(ResultFn fn) override { cb_ = std::move(fn); }

    void start() override {
        if (running_.exchange(true)) return;
        worker_ = std::thread([this] { loop(); });
    }

    void stop() override {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool enqueue(const SignRequest& req) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (static_cast<int>(q_.size()) >= cfg_.queue_size) return false;
        q_.push_back(req.id);
        cv_.notify_one();
        return true;
    }

    const char* backend() const override { return "mock"; }

private:
    void loop() {
        while (true) {
            int id;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return !q_.empty() || !running_.load(); });
                if (!running_.load() && q_.empty()) return;
                id = q_.front();
                q_.pop_front();
            }
            // Stand in for the VLM round trip so the gate is actually observable.
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.mock_delay_ms));
            if (cb_) cb_(id, true, cfg_.mock_text);
        }
    }

    SignConfig              cfg_;
    ResultFn                cb_;
    std::atomic<bool>       running_{false};
    std::thread             worker_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::deque<int>         q_;
};

}  // namespace

std::unique_ptr<IDetector> makeDetector(const SignConfig& cfg) {
    std::fprintf(stderr,
                 "[sign] backend=mock (LIFT3D_USE_SIGN is OFF -- no VLM contacted; replies "
                 "\"%s\" after %d ms)\n",
                 cfg.mock_text.c_str(), cfg.mock_delay_ms);
    return std::make_unique<MockDetector>(cfg);
}

}  // namespace sign
