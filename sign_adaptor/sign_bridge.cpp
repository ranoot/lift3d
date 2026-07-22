#include "sign_bridge.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <unordered_map>

namespace sign {

namespace {

// Monotonic milliseconds, for the dispatch->result latency in the logs.
std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
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

SignBridge::SignBridge(const SignConfig& cfg) : cfg_(cfg) {
    det_ = makeDetector(cfg_);
    det_->setResultCallback(
        [this](int id, bool ok, const std::string& text) { onResult(id, ok, text); });
    det_->start();
    std::string classes;
    for (const std::string& c : cfg_.sign_classes) {
        if (!classes.empty()) classes += ",";
        classes += c;
    }
    std::fprintf(stderr, "[sign] bridge up: classes=[%s] min_px=%d gate=single-in-flight\n",
                 classes.c_str(), cfg_.min_sign_px);
}

SignBridge::~SignBridge() {
    if (det_) det_->stop();   // drains workers, joins threads
}

void SignBridge::onFrame(const SyncedFrame& sf, const FrameResult& fr,
                         const std::vector<int>& pix_gidx, int W, int H,
                         const Universe& uni, const InfClient& inf, float voxel) {
    const int frame = ++frames_;
    if (W <= 0 || H <= 0 || !fr.ok() || fr.id_map.size() != static_cast<std::size_t>(W) * H)
        return;
    if (pix_gidx.size() != static_cast<std::size_t>(W) * H) return;

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
    struct Acc { int count = 0, lbl = -1, umin = 1 << 30, vmin = 1 << 30, umax = -1, vmax = -1; };
    std::unordered_map<int, Acc> insts;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const int lbl = fr.labelAt(u, v);
            if (!signLabel(lbl)) continue;
            const int id = fr.idAt(u, v);
            if (id < 0) continue;
            Acc& a = insts[id];
            ++a.count;
            a.lbl  = lbl;
            a.umin = std::min(a.umin, u); a.umax = std::max(a.umax, u);
            a.vmin = std::min(a.vmin, v); a.vmax = std::max(a.vmax, v);
        }
    }
    int best_inst = -1; const Acc* best = nullptr;
    for (const auto& [id, a] : insts)
        if (a.count >= cfg_.min_sign_px && (!best || a.count > best->count)) { best_inst = id; best = &a; }
    if (!best) return;   // no sign this frame (nothing to say)

    const std::string& cls = inf.className(best->lbl);
    std::fprintf(stderr, "[sign] frame %d: DETECTED \"%s\" inst=%d px=%d bbox=[%d,%d %d,%d]",
                 frame, cls.c_str(), best_inst, best->count,
                 best->umin, best->vmin, best->umax, best->vmax);

    // Gate: one request at a time. Frames that arrive while the module is reading are dropped.
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_ != 0) {
            ++n_dropped_;
            std::fprintf(stderr, " -> dropped (busy with id=%d for %lldms; %zu dropped so far)\n",
                         inflight_id_, static_cast<long long>(nowMs() - inflight_ms_), n_dropped_);
            return;
        }
    }

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
    if (footprint.empty()) {          // no 3D geometry -> could never be matched to an object
        std::fprintf(stderr, " -> skipped (no 3D points behind the mask)\n");
        return;
    }
    const std::size_t nvox = footprint.size();

    // Reserve a match id + stash the footprint, then dispatch. Undo the reservation if the queue
    // rejects it (should not happen with the gate, but keep pending_/inflight_ consistent).
    int id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (inflight_ != 0) { std::fprintf(stderr, " -> dropped (raced the gate)\n"); return; }
        id = static_cast<int>(next_id_++);
        pending_.emplace(id, std::move(footprint));
        inflight_    = 1;
        inflight_id_ = id;
        inflight_ms_ = nowMs();
    }

    SignRequest req;
    req.id      = id;
    req.t_ns    = sf.image_t;
    req.rgb     = sf.image.rgb.data();
    req.w       = sf.image.w;
    req.h       = sf.image.h;
    req.bbox[0] = best->umin; req.bbox[1] = best->vmin;
    req.bbox[2] = best->umax; req.bbox[3] = best->vmax;

    if (!det_->enqueue(req)) {                     // queue full -> undo the reservation
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(id);
        inflight_ = 0;
        std::fprintf(stderr, " -> dropped (detector queue full)\n");
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++n_dispatched_;
    }
    std::fprintf(stderr, " -> SENT to %s id=%d (%zu voxels, image %dx%d, t=%lldns)\n",
                 det_->backend(), id, nvox, req.w, req.h,
                 static_cast<long long>(req.t_ns));
}

void SignBridge::onResult(int id, bool ok, const std::string& text) {
    std::lock_guard<std::mutex> lk(mu_);
    const std::int64_t ms = (id == inflight_id_) ? nowMs() - inflight_ms_ : -1;
    auto it = pending_.find(id);
    if (it != pending_.end()) {
        if (ok && !text.empty()) {
            resolved_.push_back(Resolved{std::move(it->second), text});
            ++gen_;
            std::fprintf(stderr, "[sign] RECEIVED id=%d after %lldms -> \"%s\" (%zu signs read)\n",
                         id, static_cast<long long>(ms), text.c_str(), resolved_.size());
        } else {
            ++n_failed_;
            std::fprintf(stderr, "[sign] RECEIVED id=%d after %lldms -> no content (%s)\n",
                         id, static_cast<long long>(ms), ok ? "empty text" : "failed");
        }
        pending_.erase(it);
    } else {
        std::fprintf(stderr, "[sign] RECEIVED id=%d -- unknown id, ignored\n", id);
    }
    inflight_ = 0;   // gate reopens for the next frame
}

std::unordered_map<int, std::string> SignBridge::annotate(const std::vector<Object>& objs,
                                                          const Universe& uni, float voxel) {
    std::vector<Resolved> signs;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (resolved_.empty()) return {};
        const bool fresh = (gen_ == cached_gen_) && (++calls_since_ < kAnnotateStride);
        if (fresh) return cached_;                  // serve the cache: no re-voxelization
        cached_gen_  = gen_;
        calls_since_ = 0;
        signs        = resolved_;                   // snapshot (footprints are small)
    }
    const float inv = 1.0f / (voxel > 0.0f ? voxel : 0.05f);

    // Precompute each object's voxel set once.
    std::vector<objutil::VSet> obj_vox(objs.size());
    for (std::size_t i = 0; i < objs.size(); ++i) obj_vox[i] = objectVoxels(objs[i], uni, inv);

    // obj id -> (best overlap so far). "Biggest proposal wins on conflict": a sign picks its
    // max-overlap object (ties broken by more points); if two signs land on the same object, the
    // stronger overlap keeps it.
    std::unordered_map<int, std::string> out;
    std::unordered_map<int, std::size_t> claimed_overlap;
    for (const Resolved& s : signs) {
        int best_i = -1; std::size_t best_ov = 0, best_pts = 0;
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
            // Log the propagation once per (object, text) change, not once per frame.
            auto lg = logged_.find(oid);
            if (lg == logged_.end() || lg->second != s.text) {
                logged_[oid] = s.text;
                std::fprintf(stderr,
                             "[sign] ATTACHED to object %d (\"%s\", %zu pts, %zu shared voxels)"
                             " -> \"%s\"\n",
                             oid, uni.semantics().name(objs[best_i].class_id).c_str(),
                             best_pts, best_ov, s.text.c_str());
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        cached_ = out;
    }
    return out;
}

std::size_t SignBridge::dispatched() {
    std::lock_guard<std::mutex> lk(mu_); return n_dispatched_;
}
std::size_t SignBridge::droppedBusy() {
    std::lock_guard<std::mutex> lk(mu_); return n_dropped_;
}
std::size_t SignBridge::resolvedCount() {
    std::lock_guard<std::mutex> lk(mu_); return resolved_.size();
}
std::size_t SignBridge::failedCount() {
    std::lock_guard<std::mutex> lk(mu_); return n_failed_;
}
std::size_t SignBridge::inFlight() {
    std::lock_guard<std::mutex> lk(mu_); return static_cast<std::size_t>(inflight_);
}

}  // namespace sign
