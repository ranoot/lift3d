#include "semantic_runner.h"

#include "run_config.h"          // RunConfig, loadRunConfig
#include "online_semantic.h"     // semantic::OnlineSemantic (owns the pipeline; onSynced)
#include "universe.h"            // Universe, SemanticVocabulary, Cloud/PointT
#include "objects.h"             // Object
#include "viz_publisher.h"       // VizPublisher, VizFrameFlags, VizSegConfig (+ SyncedFrame etc.)
#include "dog_log_adaptor.h"     // CameraCalib, cameraFromBodyPose, loadCameraCalib
#include "euler_pose.h"          // doglog::iso_from_pose6 (Eigen; .cpp-only per its contract)

#include "gmd_frame_source.h"    // gmd::toNanoseconds / toPose6 / toPackedCloud / toDogImage
#include "object_publisher.h"    // gmd::buildRoomObject, EAIRoomObject

#ifdef LIFT3D_HAVE_SIGN
#include "sign_bridge.h"         // sign::SignBridge (LIFT3D_USE_SIGN builds only)
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace semantic {

// ---------------------------------------------------------------------------------------
// Owned pipeline state (hidden from the caller via PImpl).
// ---------------------------------------------------------------------------------------
struct SemanticRunner::Impl {
    // A cached egress result + the cheap membership signature it was built from, so the
    // O(n^2) SOR + convex hull in buildRoomObject only re-runs for objects that changed.
    struct CacheEntry {
        std::uint64_t                  sig;
        Common::Entity::EAIRoomObject  obj;
    };

    RunConfig                  cfg;      // first: online/calib read it
    CameraCalib                calib;
    OnlineSemantic             online;
    VizPublisher               viz;
    std::uint16_t              vehicleId;
    float                      voxel;

    // viz frame-hook scratch (mirrors run_semantic_universe's hook bodies).
    const FrameResult*         g_fr        = nullptr;
    std::uint64_t              last_sp_ver = 0;
    std::int64_t               t0          = 0;
    bool                       have_t0     = false;
    bool                       seg_on      = false;

    std::unordered_map<int, CacheEntry> cache;   // object id -> cached hull
#ifdef LIFT3D_HAVE_SIGN
    std::unique_ptr<sign::SignBridge>   sign;
#endif

    explicit Impl(const std::string& config_path)
        : cfg(loadRunConfig(config_path)),
          calib(loadCameraCalib(cfg.calib, cfg.cam)),
          online(cfg.p),
          viz(cfg.viz_endpoint),
          vehicleId(static_cast<std::uint16_t>(cfg.cam)),  // non-zero => node ids never NULL(0,0)
          voxel(cfg.p.voxel) {}

    // universe Objects -> EAIRoomObjects, reusing cached hulls for unchanged objects.
    std::vector<Common::Entity::EAIRoomObject>
        collect(const std::vector<Object>& objs, const Universe& uni);
};

// ---------------------------------------------------------------------------------------
// Cheap per-object membership signature. Detects growth / relabel / re-root without hashing
// every member point: mixes the counts + a strided sample of the point indices. (A member
// point being tombstoned without the membership vector changing is not caught -- a documented
// tradeoff for skipping the hull recompute; thing objects are stable enough for it.)
// ---------------------------------------------------------------------------------------
static std::uint64_t objectSig(const Object& o) {
    std::uint64_t h = 1469598103934665603ULL;                // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    const std::size_t n = o.points.size();
    mix(static_cast<std::uint64_t>(n));
    mix(static_cast<std::uint32_t>(o.class_id));
    mix(static_cast<std::uint32_t>(o.level));
    if (n) {
        const std::size_t step = n / 8 + 1;
        for (std::size_t i = 0; i < n; i += step)
            mix(static_cast<std::uint32_t>(o.points[i]));
        mix(static_cast<std::uint32_t>(o.points.back()));
    }
    return h;
}

// ---------------------------------------------------------------------------------------
// universe Objects -> EAIRoomObjects, reusing cached hulls for unchanged objects. Returns the
// FULL list every call (per the plan); the hull cache is the "don't re-run the convex polygon
// for all objects every time" mitigation. Ported from gmd::collectEAIObjects.
// ---------------------------------------------------------------------------------------
std::vector<Common::Entity::EAIRoomObject>
SemanticRunner::Impl::collect(const std::vector<Object>& objs, const Universe& uni) {
    std::vector<Common::Entity::EAIRoomObject> out;
    out.reserve(objs.size());
    Universe::Cloud::ConstPtr cloud = uni.cloud();
    if (!cloud) return out;

#ifdef LIFT3D_HAVE_SIGN
    std::unordered_map<int, std::string> sign_dir =
        sign ? sign->annotate(objs, uni, voxel) : std::unordered_map<int, std::string>{};
#endif

    std::unordered_set<int>           present;
    std::vector<std::array<float, 3>> xyz;
    for (const Object& o : objs) {
        const std::uint64_t sig = objectSig(o);
        auto it = cache.find(o.id);

        Common::Entity::EAIRoomObject obj;
        if (it != cache.end() && it->second.sig == sig) {
            obj = it->second.obj;                            // reuse: no SOR / convex hull
        } else {
            xyz.clear();
            xyz.reserve(o.points.size());
            for (int gid : o.points) {
                if (gid < 0 || gid >= static_cast<int>(cloud->size())) continue;
                if (!uni.pointAlive(gid)) continue;
                const Universe::PointT& pt = (*cloud)[gid];
                xyz.push_back({pt.x, pt.y, pt.z});
            }
            if (xyz.empty()) { cache.erase(o.id); continue; }
            obj = gmd::buildRoomObject(vehicleId, o.id,
                                       uni.semantics().name(o.class_id), xyz);
            cache[o.id] = CacheEntry{sig, obj};
        }
#ifdef LIFT3D_HAVE_SIGN
        auto sit = sign_dir.find(o.id);                      // signs can change without geometry
        obj.directionContent = (sit != sign_dir.end()) ? sit->second : std::string{};
#endif
        present.insert(o.id);
        out.push_back(std::move(obj));
    }

    for (auto it = cache.begin(); it != cache.end();)        // evict vanished objects
        it = present.count(it->first) ? std::next(it) : cache.erase(it);
    return out;
}

// ---------------------------------------------------------------------------------------
SemanticRunner::SemanticRunner(const std::string& config_path)
    : impl_(std::make_unique<Impl>(config_path)) {
    Impl* d = impl_.get();

    if (!d->online.inf().ping())
        std::fprintf(stderr,
                     "[semantic_runner] WARN: inf_server not responding at %s "
                     "(runOnInput will still run the geometry stages)\n",
                     d->cfg.p.endpoint.c_str());

    d->seg_on      = d->viz.active() && d->cfg.seg_overlay != "off";
    d->last_sp_ver = d->online.superpoints().version();

    // Seg-overlay hook: stash this frame's 2D masks for the frame hook to forward (viz only).
    if (d->seg_on)
        d->online.setSegHook(
            [d](const SyncedFrame&, const FrameResult& fr) { d->g_fr = &fr; });

    // Per-frame viz hook: forward pose/camera/image/masks/discovery tiers to the Python viz.
    if (d->viz.active())
        d->online.setFrameHook([d](int idx, const SyncedFrame& sf, const PointPixelMap&) {
            VizFrameFlags flags;
            flags.superpoints = d->cfg.p.superpoints;
            flags.proposals   = d->cfg.p.hdbscan;
            flags.objects     = d->cfg.p.objects;
            flags.sp_changed  =
                d->cfg.p.superpoints && d->online.superpoints().version() != d->last_sp_ver;
            if (flags.sp_changed) d->last_sp_ver = d->online.superpoints().version();
            const double capture_secs =
                d->have_t0 ? static_cast<double>(sf.image_t - d->t0) * 1e-9 : 0.0;
            d->viz.frame(idx, capture_secs, sf, d->g_fr, d->online.superpoints(),
                         d->online.seeds(), d->online.objects(), d->online.universe(), flags);
            d->g_fr = nullptr;   // consume: never repaint stale masks on a no-inference frame
        });

#ifdef LIFT3D_HAVE_SIGN
    d->sign = std::make_unique<sign::SignBridge>(sign::SignBridge::Config{});
    d->online.setSignHook([d](const SyncedFrame& sf, const FrameResult& fr,
                              const std::vector<int>& pix_gidx, int W, int H, const Universe& u) {
        d->sign->onFrame(sf, fr, pix_gidx, W, H, u, d->online.inf(), d->voxel);
    });
#endif

    d->online.begin();
    if (d->viz.active())
        d->viz.begin(d->online.universe().semantics(), d->online.inf().vocab(),
                     VizSegConfig{d->cfg.seg_overlay, d->cfg.seg_alpha, d->cfg.seg_min_area});
}

SemanticRunner::~SemanticRunner() {
    if (!impl_) return;
    impl_->online.finish();                                  // final merge + timing report
    if (impl_->viz.active())
        impl_->viz.finish(impl_->online.universe(), impl_->online.objects());  // final world map
}

// ---------------------------------------------------------------------------------------
std::vector<Common::Entity::EAIRoomObject>
SemanticRunner::runOnInput(std::optional<OpenVocabSegInput> input) {
    if (!input) return {};                                   // null optional => no-op
    Impl* d = impl_.get();
    const OpenVocabSegInput& in = *input;

    if (in.images.size() != 1)
        throw std::invalid_argument(
            "SemanticRunner::runOnInput: expected exactly 1 image, got " +
            std::to_string(in.images.size()));
    const Common::Entity::Image& image = in.images.front();

    // --- Build the SyncedFrame directly (inputs are already synced/interpolated) ---------
    SyncedFrame sf;
    sf.image_t = gmd::toNanoseconds(image.timestamp);
    sf.scan_t  = sf.image_t;                                 // pre-synced: scan aligned to image

    double pose6[6];
    gmd::toPose6(in.pose, pose6);                            // {x,y,z,roll,pitch,yaw} body->world
    const doglog::Iso3 T = doglog::iso_from_pose6(pose6);
    const doglog::Mat4 M = T.matrix();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            sf.T_world_body[r * 4 + c] = M(r, c);            // row-major fill
    sf.cam      = cameraFromBodyPose(d->calib, sf.T_world_body);
    sf.robot[0] = static_cast<float>(sf.T_world_body[3]);
    sf.robot[1] = static_cast<float>(sf.T_world_body[7]);
    sf.robot[2] = static_cast<float>(sf.T_world_body[11]);

    if (!gmd::toDogImage(image, sf.image))
        throw std::invalid_argument(
            "SemanticRunner::runOnInput: unsupported/invalid image encoding");
    sf.image.t_ns = sf.image_t;

    // Body-frame cloud -> world (the same lift DogStream::pushScan applies).
    std::vector<float>         xyz_body;
    std::vector<unsigned char> inten;
    gmd::toPackedCloud(in.pointCloud, xyz_body, inten);
    const int n = static_cast<int>(inten.size());
    sf.cloud_world.reserve(static_cast<std::size_t>(n) * 3);
    sf.intensity.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3d pw =
            T * Eigen::Vector3d(xyz_body[3 * i + 0], xyz_body[3 * i + 1], xyz_body[3 * i + 2]);
        sf.cloud_world.push_back(static_cast<float>(pw.x()));
        sf.cloud_world.push_back(static_cast<float>(pw.y()));
        sf.cloud_world.push_back(static_cast<float>(pw.z()));
        sf.intensity.push_back(static_cast<float>(inten[i]));
    }

    if (!d->have_t0) { d->t0 = sf.image_t; d->have_t0 = true; }

    // --- Drive the pipeline: integrate + infer + project + vote + discovery + viz hooks ---
    d->online.onSynced(sf);

    // --- Egress: full object list, hull-cached ------------------------------------------
    std::vector<Common::Entity::EAIRoomObject> objects =
        d->collect(d->online.objects().list(), d->online.universe());

    std::printf("[semantic_runner] frame %d -> %zu objects\n",
                d->online.frames(), objects.size());
    std::fflush(stdout);
    return objects;
}

}  // namespace semantic
