#include "viz_publisher.h"
#include "msgpack_lite.h"        // mpk::Packer (shared with inf_client)

#include <zmq.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

// ---- cppzmq state (kept out of the public header) ---------------------------
struct VizPublisher::Impl {
    zmq::context_t ctx{1};
    zmq::socket_t  sock{ctx, zmq::socket_type::push};
};

namespace {

// Pack a numpy-style ndarray envelope {"__ndarray__":true,"shape":[..],"dtype":str,
// "data":bin}, matching inf_server/protocol.py so the Python side's object_hook rebuilds
// it. `dtype` is a numpy dtype.str (e.g. "<f4", "<i2", "|i1", "|u1"); `data` is the raw
// little-endian buffer (host is LE), packed as a msgpack bin.
void packNdarray(mpk::Packer& p, const char* dtype,
                 std::initializer_list<uint32_t> shape,
                 const void* data, size_t nbytes) {
    p.mapHeader(4);
    p.str("__ndarray__"); p.boolean(true);
    p.str("shape"); p.arrHeader((uint32_t)shape.size());
    for (uint32_t s : shape) p.uint(s);
    p.str("dtype"); p.str(dtype);
    p.str("data");  p.bin((const uint8_t*)data, nbytes);
}

// Gather the world xyz of a member-index list into a flat [M*3] float buffer, skipping
// out-of-range / tombstoned points exactly as the old viz did (so M is the ALIVE count).
std::vector<float> gatherXYZ(const Universe& uni, Universe::Cloud::ConstPtr cloud,
                             const std::vector<int>& idxs) {
    std::vector<float> out;
    out.reserve(idxs.size() * 3);
    const int n = cloud ? (int)cloud->size() : 0;
    for (int idx : idxs) {
        if (idx < 0 || idx >= n || !uni.pointAlive(idx)) continue;
        const Universe::PointT& pt = (*cloud)[idx];
        out.push_back(pt.x); out.push_back(pt.y); out.push_back(pt.z);
    }
    return out;
}

// f4 ndarray from a flat float buffer already laid out as (rows, cols).
void packF32(mpk::Packer& p, const std::vector<float>& v, uint32_t rows, uint32_t cols) {
    packNdarray(p, "<f4", {rows, cols}, v.data(), v.size() * sizeof(float));
}

} // namespace

VizPublisher::VizPublisher(const std::string& endpoint) {
    if (endpoint.empty()) return;                 // inactive sink -> every call is a no-op
    impl_ = new Impl;
    // Small send queue so a slow viewer drops stale frames instead of buffering a backlog;
    // bounded send timeout so the best-effort finish() never hangs the run when no viewer
    // is attached; linger 0 so teardown is immediate.
    impl_->sock.set(zmq::sockopt::sndhwm, 8);
    impl_->sock.set(zmq::sockopt::sndtimeo, 2000);
    impl_->sock.set(zmq::sockopt::linger, 0);
    impl_->sock.connect(endpoint);               // PULL side binds; connect may precede it
    active_ = true;
}

VizPublisher::~VizPublisher() { delete impl_; }

// Send helper: per-frame messages are non-blocking (drop on backpressure / no peer); the
// final map is best-effort (blocks up to sndtimeo). Any zmq error (EAGAIN, no route) is
// swallowed -- the visualizer is optional and must never take down the run.
static void sendBuf(zmq::socket_t& sock, const std::vector<uint8_t>& buf, bool best_effort) {
    try {
        sock.send(zmq::buffer(buf),
                  best_effort ? zmq::send_flags::none : zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) { /* viz gone or would block -> drop */ }
}

void VizPublisher::begin(const SemanticVocabulary& sem,
                         const std::vector<std::string>& inf_vocab, const VizSegConfig& seg) {
    if (!active_) return;
    mpk::Packer p;
    p.mapHeader(6);
    p.str("type"); p.str("begin");
    p.str("classes");
    p.arrHeader((uint32_t)sem.size());
    for (int id = 0; id < sem.size(); ++id) {
        p.mapHeader(3);
        p.str("id");   p.uint((uint64_t)id);
        p.str("name"); p.str(sem.name(id));
        p.str("kind"); p.uint((uint64_t)static_cast<uint8_t>(sem.kind(id)));
    }
    // Ordered InfClient vocab (index == the class id in seg label_map/id_map). Distinct id
    // space from `classes` above, so the viewer resolves seg-overlay labels through this.
    p.str("seg_vocab");
    p.arrHeader((uint32_t)inf_vocab.size());
    for (const std::string& name : inf_vocab) p.str(name);
    p.str("seg_overlay");  p.str(seg.overlay);
    p.str("seg_alpha");    p.f64((double)seg.alpha);
    p.str("seg_min_area"); p.integer(seg.min_area);
    sendBuf(impl_->sock, p.buf, /*best_effort=*/true);   // header must land; sent before frames
}

void VizPublisher::frame(int idx, double capture_secs, const SyncedFrame& sf,
                         const FrameResult* seg,
                         const Superpoints& sp, const ObjectSeeds& os, const Objects& obj,
                         const Universe& uni, const VizFrameFlags& flags,
                         const std::unordered_map<int, std::string>* obj_text) {
    if (!active_) return;
    Universe::Cloud::ConstPtr cloud = uni.cloud();

    const bool has_image = sf.image.ok();
    const bool has_seg   = seg && seg->ok();
    const bool has_obj   = flags.objects;
    const bool has_prop  = flags.proposals;
    const bool has_sp    = flags.superpoints && flags.sp_changed;

    // One map per discovered entity {id,class_id,name,centroid,points(+kind)(+text)}.
    auto packEntity = [&](mpk::Packer& p, int id, int class_id,
                          const float centroid[3], const std::vector<int>& pts,
                          const int* kind, const std::string* text = nullptr) {
        p.mapHeader(5 + (kind ? 1 : 0) + (text ? 1 : 0));
        p.str("id");       p.integer(id);
        p.str("class_id"); p.integer(class_id);
        p.str("name");     p.str(uni.semantics().name(class_id));
        if (kind) { p.str("kind"); p.integer(*kind); }
        if (text) { p.str("text"); p.str(*text); }
        p.str("centroid"); packNdarray(p, "<f4", {3}, centroid, 3 * sizeof(float));
        std::vector<float> xyz = gatherXYZ(uni, cloud, pts);
        p.str("points");   packF32(p, xyz, (uint32_t)(xyz.size() / 3), 3);
    };

    mpk::Packer p;
    int nkeys = 6 + (int)has_image + (int)has_seg + (int)has_obj + (int)has_prop + (int)has_sp;
    p.mapHeader((uint32_t)nkeys);
    p.str("type");         p.str("frame");
    p.str("idx");          p.uint((uint64_t)idx);
    p.str("capture_secs"); p.f64(capture_secs);

    // Robot body pose (double[16] row-major) + world position, as f4 for rerun.
    float pose[16];
    for (int i = 0; i < 16; ++i) pose[i] = (float)sf.T_world_body[i];
    p.str("robot_pose"); packNdarray(p, "<f4", {16}, pose, sizeof(pose));
    p.str("robot");      packNdarray(p, "<f4", {3}, sf.robot, 3 * sizeof(float));

    // Camera intrinsics + camera-to-world pose + image size (rerun Pinhole + Transform3D).
    p.str("cam");
    p.mapHeader(4);
    p.str("K"); packNdarray(p, "<f4", {9},  sf.cam.K, 9 * sizeof(float));
    p.str("T"); packNdarray(p, "<f4", {16}, sf.cam.T, 16 * sizeof(float));
    p.str("w"); p.uint((uint64_t)sf.cam.width);
    p.str("h"); p.uint((uint64_t)sf.cam.height);

    if (has_image) {
        p.str("image");
        packNdarray(p, "|u1", {(uint32_t)sf.image.h, (uint32_t)sf.image.w, 3},
                    sf.image.rgb.data(), sf.image.rgb.size());
    }

    // Raw DVIS 2D masks for the Python-side painted overlay (label + optional instance id).
    if (has_seg) {
        const bool has_id = seg->id_map.size() == (size_t)seg->h * seg->w;
        p.str("seg");
        p.mapHeader(has_id ? 2 : 1);
        p.str("label_map");
        packNdarray(p, "<i2", {(uint32_t)seg->h, (uint32_t)seg->w},
                    seg->label_map.data(), seg->label_map.size() * sizeof(int16_t));
        if (has_id) {
            p.str("id_map");
            packNdarray(p, "<i2", {(uint32_t)seg->h, (uint32_t)seg->w},
                        seg->id_map.data(), seg->id_map.size() * sizeof(int16_t));
        }
    }

    if (has_obj) {
        p.str("objects");
        p.arrHeader((uint32_t)obj.list().size());
        for (const Object& o : obj.list()) {
            const std::string* text = nullptr;      // e.g. this object's resolved sign text
            if (obj_text) {
                auto it = obj_text->find(o.id);
                if (it != obj_text->end() && !it->second.empty()) text = &it->second;
            }
            packEntity(p, o.id, o.class_id, o.centroid, o.points, nullptr, text);
        }
    }
    if (has_prop) {
        p.str("proposals");
        p.arrHeader((uint32_t)os.list().size());
        for (const ObjectSeed& s : os.list())
            packEntity(p, s.id, s.class_id, s.centroid, s.points, nullptr);
    }
    if (has_sp) {
        p.str("superpoints");
        p.arrHeader((uint32_t)sp.list().size());
        for (const Superpoint& s : sp.list()) {
            const int kind = static_cast<int>(s.kind);
            packEntity(p, (int)s.id, s.class_id, s.centroid, s.points, &kind);
        }
    }

    sendBuf(impl_->sock, p.buf, /*best_effort=*/false);   // drop on backpressure (live viewer)
}

void VizPublisher::finish(const Universe& uni, const Objects& obj) {
    if (!active_) return;
    Universe::Cloud::ConstPtr cloud = uni.cloud();

    // Flatten every ALIVE point: position (f4 N,3) + class id (i2 N) + kind (i1 N). Python
    // routes into world/map/{things,stuff,unlabeled} by kind and colours via the legend.
    std::vector<float>   pos;
    std::vector<int16_t> cls;
    std::vector<int8_t>  kind;
    const int n = uni.size();
    pos.reserve((size_t)uni.aliveCount() * 3);
    cls.reserve(uni.aliveCount());
    kind.reserve(uni.aliveCount());
    for (int i = 0; i < n; ++i) {
        if (!uni.pointAlive(i)) continue;
        const Universe::PointT& pt = (*cloud)[i];
        pos.push_back(pt.x); pos.push_back(pt.y); pos.push_back(pt.z);
        cls.push_back((int16_t)uni.pointClassId(i));
        kind.push_back((int8_t)static_cast<uint8_t>(uni.pointClassKind(i)));
    }

    mpk::Packer p;
    p.mapHeader(4);
    p.str("type"); p.str("finish");
    p.str("positions"); packF32(p, pos, (uint32_t)(pos.size() / 3), 3);
    p.str("class_ids"); packNdarray(p, "<i2", {(uint32_t)cls.size()},
                                    cls.data(), cls.size() * sizeof(int16_t));
    p.str("kinds");     packNdarray(p, "|i1", {(uint32_t)kind.size()},
                                    kind.data(), kind.size() * sizeof(int8_t));

    sendBuf(impl_->sock, p.buf, /*best_effort=*/true);
    (void)obj;   // final object list is already streamed per-frame; static objects render
                 // from the last frame message the viewer received.
}
