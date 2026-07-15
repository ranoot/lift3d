#include "sign_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <unordered_map>

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

// Voxelize an object's representative points into a set (alive points only).
objutil::VSet objectVoxels(const Object& o, const Universe& uni, float inv) {
    objutil::VSet vs;
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud) return vs;
    vs.reserve(o.points.size());
    for (int g : o.points) {
        if (g < 0 || g >= static_cast<int>(cloud->size()) || !uni.pointAlive(g)) continue;
        vs.insert(objutil::voxelOf((*cloud)[g], inv));
    }
    return vs;
}

// Count shared voxels between two sets (iterate the smaller).
std::size_t overlapCount(const objutil::VSet& a, const objutil::VSet& b) {
    const objutil::VSet& small = a.size() <= b.size() ? a : b;
    const objutil::VSet& big   = a.size() <= b.size() ? b : a;
    std::size_t n = 0;
    for (const objutil::VKey& k : small)
        if (big.count(k)) ++n;
    return n;
}

}  // namespace

SignBridge::SignBridge(const Config& cfg) : cfg_(cfg) {
    det_ = std::make_unique<SignUnderstanding::ConcurrentDetectorWrapper>(
        cfg_.vlm_url, cfg_.num_workers, cfg_.queue_size);
    det_->SetResultCallback([this](const SignUnderstanding::ProcessResult& r) { onResult(r); });
    det_->Start();
    std::fprintf(stderr, "[sign] bridge up: vlm=%s workers=%d queue=%d classes=%zu\n",
                 cfg_.vlm_url.c_str(), cfg_.num_workers, cfg_.queue_size, cfg_.sign_classes.size());
}

SignBridge::~SignBridge() {
    if (det_) det_->Stop();   // drains workers, joins threads
}

void SignBridge::onFrame(const SyncedFrame& sf, const FrameResult& fr,
                         const std::vector<int>& pix_gidx, int W, int H,
                         const Universe& uni, const InfClient& inf, float voxel) {
    // Gate: skip entirely while a request is in flight (single-in-flight, per the agreement).
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_ != 0) return;
    }
    if (W <= 0 || H <= 0 || !fr.ok() || fr.id_map.size() != static_cast<std::size_t>(W) * H)
        return;

    // Which class ids are "signs"? Cache the className lookup so we don't re-compare strings for
    // every pixel of a repeated label.
    std::unordered_map<int, bool> is_sign;
    auto signLabel = [&](int lbl) -> bool {
        if (lbl < 0) return false;
        auto it = is_sign.find(lbl);
        if (it != is_sign.end()) return it->second;
        const std::string& name = inf.className(lbl);
        bool s = false;
        for (const std::string& c : cfg_.sign_classes)
            if (c == name) { s = true; break; }
        is_sign.emplace(lbl, s);
        return s;
    };

    // Per sign INSTANCE id: pixel count + pixel bbox. Pick the largest instance ("send one").
    struct Acc { int count = 0, umin = 1 << 30, vmin = 1 << 30, umax = -1, vmax = -1; };
    std::unordered_map<int, Acc> insts;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const int lbl = fr.labelAt(u, v);
            if (!signLabel(lbl)) continue;
            const int id = fr.idAt(u, v);
            if (id < 0) continue;
            Acc& a = insts[id];
            ++a.count;
            a.umin = std::min(a.umin, u); a.umax = std::max(a.umax, u);
            a.vmin = std::min(a.vmin, v); a.vmax = std::max(a.vmax, v);
        }
    }
    int best_inst = -1; const Acc* best = nullptr;
    for (const auto& [id, a] : insts)
        if (a.count >= cfg_.min_sign_px && (!best || a.count > best->count)) { best_inst = id; best = &a; }
    if (!best) return;   // no sign this frame

    // The sign's 3D footprint: the nearest surface voxel behind each of its pixels.
    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    objutil::VSet footprint;
    if (cloud) {
        for (int v = best->vmin; v <= best->vmax; ++v) {
            for (int u = best->umin; u <= best->umax; ++u) {
                if (fr.idAt(u, v) != best_inst || !signLabel(fr.labelAt(u, v))) continue;
                const int g = pix_gidx[static_cast<std::size_t>(v) * W + u];
                if (g < 0 || g >= static_cast<int>(cloud->size()) || !uni.pointAlive(g)) continue;
                footprint.insert(objutil::voxelOf((*cloud)[g], inv));
            }
        }
    }
    if (footprint.empty()) return;   // no 3D geometry -> can't match it to an object later

    // Reserve a match id + stash the footprint, then dispatch. Bail the reservation if the queue
    // rejects it (should not happen with the gate, but keep pending_/inflight_ consistent).
    int id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_ != 0) return;                 // re-check after the (unlocked) scan
        id = static_cast<int>(next_id_++);
        pending_.emplace(id, std::move(footprint));
        inflight_ = 1;
    }

    // image.timestamp must match bboxInput.timestamp exactly (DetectorWrapper::ProcessSign checks
    // it), so derive one TimeStamp from the frame's image instant and use it for both.
    Common::Entity::TimeStamp stamp{std::chrono::duration_cast<Common::Entity::TimeClock::duration>(
        std::chrono::nanoseconds(sf.image_t))};

    Common::Entity::Image img;
    img.timestamp = stamp;
    img.height    = sf.image.h;
    img.width     = sf.image.w;
    img.encoding  = ::ImageEncoding::RGB8;
    img.step      = sf.image.w * 3;
    img.data.assign(sf.image.rgb.begin(), sf.image.rgb.end());

    SignUnderstanding::SignBBoxInput in;
    in.instanceId = id;
    in.timestamp  = stamp;
    in.bbox = {static_cast<double>(best->umin), static_cast<double>(best->vmin),
               static_cast<double>(best->umax), static_cast<double>(best->vmax)};

    if (!det_->Enqueue(img, in)) {                  // queue full -> undo the reservation
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(id);
        inflight_ = 0;
        return;
    }
    std::fprintf(stderr, "[sign] dispatch id=%d bbox=[%d,%d,%d,%d] px=%d\n",
                 id, best->umin, best->vmin, best->umax, best->vmax, best->count);
}

void SignBridge::onResult(const SignUnderstanding::ProcessResult& r) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(r.instanceId);
    if (it != pending_.end()) {
        if (r.success && r.output) {
            std::string text = formatInstructions(*r.output);
            if (!text.empty()) {
                resolved_.push_back(Resolved{std::move(it->second), std::move(text)});
                std::fprintf(stderr, "[sign] resolved id=%d -> \"%s\"\n",
                             r.instanceId, resolved_.back().text.c_str());
            }
        } else {
            std::fprintf(stderr, "[sign] id=%d failed/empty\n", r.instanceId);
        }
        pending_.erase(it);
    }
    inflight_ = 0;   // gate reopens for the next frame
}

std::unordered_map<int, std::string> SignBridge::annotate(const std::vector<Object>& objs,
                                                          const Universe& uni, float voxel) {
    std::unordered_map<int, std::string> out;
    std::vector<Resolved> signs;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (resolved_.empty()) return out;
        signs = resolved_;                          // snapshot (footprints are small)
    }
    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);

    // Precompute each object's voxel set + point count once.
    std::vector<objutil::VSet> obj_vox(objs.size());
    for (std::size_t i = 0; i < objs.size(); ++i) obj_vox[i] = objectVoxels(objs[i], uni, inv);

    // obj id -> (best overlap so far, object point count). "Biggest proposal wins on conflict":
    // a sign picks its max-overlap object (ties broken by more points); if two signs land on the
    // same object, the stronger overlap keeps it.
    std::unordered_map<int, std::size_t> claimed_overlap;
    for (const Resolved& s : signs) {
        int best_i = -1; std::size_t best_ov = 0; std::size_t best_pts = 0;
        for (std::size_t i = 0; i < objs.size(); ++i) {
            const std::size_t ov = overlapCount(s.footprint, obj_vox[i]);
            if (ov == 0) continue;
            const std::size_t pts = objs[i].points.size();
            if (ov > best_ov || (ov == best_ov && pts > best_pts)) {
                best_ov = ov; best_pts = pts; best_i = static_cast<int>(i);
            }
        }
        if (best_i < 0) continue;                   // sign not (yet) inside any object
        const int oid = objs[best_i].id;
        auto pc = claimed_overlap.find(oid);
        if (pc == claimed_overlap.end() || best_ov > pc->second) {
            claimed_overlap[oid] = best_ov;
            out[oid] = s.text;
        }
    }
    return out;
}

std::size_t SignBridge::resolvedCount() {
    std::lock_guard<std::mutex> lk(mu_);
    return resolved_.size();
}

std::size_t SignBridge::inFlight() {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<std::size_t>(inflight_);
}

}  // namespace sign
