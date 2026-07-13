// exp_per_view: a THROWAWAY measurement driver that answers two questions before we
// commit to a per-view redesign of the semantic pipeline (see the plan / CLAUDE.md):
//
//   1. How expensive is it to run the geometry stages -- VCCS superpoints, HDBSCAN --
//      plus inference + back-projection on ONLY one view's associated points (the
//      projected/visible subset of the matured global map), as opposed to the 6 m
//      robot-radius window the production pipeline uses?
//   2. Do the OV-DVIS++ instance ids over-segment -- i.e. does one physical object
//      frequently carry several stable instance ids?
//
// It reuses the offline DogLog ingestor + DogStream sync exactly like
// run_semantic_universe, but inlines the per-frame step (so it can capture the
// FrameResult for the id analysis -- DVIS is a stateful video model, so inference must
// run exactly once per frame). During warmup it just integrates + votes (colour +
// label) to mature the map cheaply; at each snapshot frame it times the stages on that
// view's visible set and prints an "[exp] ..." line. At the end it prints an
// "[exp-ids] ..." summary of instance-id over-segmentation.
//
// Usage:
//   ./exp_per_view --config run.yaml [--log <folder>]
//   EXP_SNAPSHOTS=400,600 EXP_WARMUP=600 ./exp_per_view --config run.yaml
//
// Requires inf_server running.

#include "universe.h"
#include "superpoints.h"
#include "object_seeds.h"
#include "run_config.h"
#include "dog_log_adaptor.h"
#include "dog_stream.h"
#include "dog_log_ingestor.h"
#include "inf_client.h"
#include "semantic_pipeline.h"     // Params, syncedToCloud, rejectDynamic, isDynamic, Stopwatch

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static std::string arg_str(int argc, char** argv, const char* key, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

// "400,600" -> {400,600}. Empty / unparseable tokens dropped.
static std::vector<int> parseInts(const std::string& s) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        try { if (!tok.empty()) out.push_back(std::stoi(tok)); } catch (...) {}
        i = j + 1;
    }
    return out;
}

class Experiment {
public:
    Experiment(Universe& uni, InfClient& inf, const semantic::Params& p,
               std::vector<int> snaps)
        : uni_(uni), inf_(inf), p_(p), snaps_(std::move(snaps)) {
        for (int s : snaps_) stop_ = std::max(stop_, s);
        thing_names_.insert(p_.thing.begin(), p_.thing.end());
    }

    void begin() {
        inf_.setVocab(p_.thing, p_.stuff);
        inf_.reset();                                  // next frame = video frame 0
        uni_.declareVocab(p_.thing, p_.stuff);
    }

    // DogStream synced-frame callback.
    void onSynced(const SyncedFrame& sf) {
        if (done_ > stop_) return;                     // past the last snapshot: stop working
        using namespace semantic;
        Stopwatch sw;

        Universe::Cloud::Ptr cloud = syncedToCloud(sf);
        FrameResult fr = inf_.frame(sf.image.rgb, sf.image.h, sf.image.w);
        const double t_infer = sw.lap();

        analyzeIds(fr);                                // Part 2: id over-segmentation

        if (!p_.dynamic.empty())
            cloud = rejectDynamic(cloud, sf.cam, fr, inf_, p_.dynamic);
        const int view = uni_.integrate(*cloud, sf.cam, sf.image_t,
                                        "img" + std::to_string(sf.image_t), sf.robot);

        // Project this view; gidx = "all the points associated with this frame".
        const float pr = p_.proj_radius > 0.0f ? p_.proj_radius : p_.radius;
        std::vector<int> gidx;
        PointPixelMap out_map = (pr > 0.0f)
            ? uni_.projectLocal(view, p_.tau, pr, &gidx, p_.splat)
            : uni_.project(view, p_.tau, p_.splat);
        const double t_proj = sw.lap();

        // Per-frame label/id for the thing-only HDBSCAN variant. Labels are per-frame
        // transient now, so reset then stamp this view's points. Mirrors stepFrameSynced.
        uni_.clearFrameLabels();
        for (int k = 0; k < out_map.N; ++k) {
            if (!out_map.isVisible(k)) continue;
            const int u = out_map.u(k), v = out_map.v(k);
            if (u < 0 || v < 0 || u >= fr.w || v >= fr.h) continue;
            const int idx = pr > 0.0f ? gidx[k] : k;
            if (!uni_.pointAlive(idx)) continue;
            const int lbl = fr.labelAt(u, v);
            if (lbl < 0) continue;
            const std::string& name = inf_.className(lbl);
            if (name.empty()) continue;
            if (isDynamic(p_.dynamic, name)) { uni_.killPoint(idx); continue; }
            uni_.setFrameLabel(idx, name, fr.idAt(u, v));
        }

        if (std::find(snaps_.begin(), snaps_.end(), done_) != snaps_.end())
            runSnapshot(done_, gidx, pr, t_proj, t_infer);

        ++done_;
        if ((done_ % 50) == 0)
            std::fprintf(stderr, "[warmup] %d/%d synced frames | world=%d pts\n",
                         done_, stop_, uni_.size());
    }

    DogStream::SyncedHook hook() {
        return [this](const SyncedFrame& sf) { onSynced(sf); };
    }

    // Time each stage on ONE view's visible set.
    void runSnapshot(int frame, const std::vector<int>& gidx, float pr,
                     double t_proj, double t_infer) {
        using namespace semantic;
        const int Nvis = (int)gidx.size();

        // VCCS superpoints on the visible set (ephemeral store, fresh each call).
        Stopwatch sw;
        Superpoints sp;
        sp.refreshFromIndices(uni_, gidx, p_.sp);
        const double t_vccs = sw.lap();
        const int nsp = sp.size();

        // HDBSCAN over the whole visible set, and over the thing points only (closer to
        // real per-class usage). hdbscanTimeOnly touches no store; times cluster only.
        int kc_all = 0;
        const double t_hdb_all = ObjectSeeds::hdbscanTimeOnly(uni_, gidx, p_.hdb, &kc_all);
        std::vector<int> thing_idx;
        thing_idx.reserve(gidx.size());
        for (int g : gidx) if (uni_.pointIsThing(g)) thing_idx.push_back(g);
        int kc_thing = 0;
        const double t_hdb_thing = ObjectSeeds::hdbscanTimeOnly(uni_, thing_idx, p_.hdb, &kc_thing);

        const double perview = t_proj + t_vccs + t_hdb_all;
        std::fprintf(stderr,
            "[exp] frame=%d Nvis=%d nsp=%d nthing=%d clu_all=%d clu_thing=%d | "
            "proj=%.1f infer=%.1f vccs=%.1f hdb_all=%.1f hdb_thing=%.1f | "
            "perview_total(proj+vccs+hdb_all)=%.1f ms\n",
            frame, Nvis, nsp, (int)thing_idx.size(), kc_all, kc_thing,
            t_proj, t_infer, t_vccs, t_hdb_all, t_hdb_thing, perview);
    }

    void report() {
        std::set<int> roots;
        int raw_thing_ids = 0;
        for (int id : all_ids_) {
            auto it = id_label_global_.find(id);
            if (it == id_label_global_.end() || !isThingLabel(it->second)) continue;
            ++raw_thing_ids;
            roots.insert(dsuFind(id));
        }
        const double mean_ids = id_perblob_cnt_ ? id_perblob_sum_ / id_perblob_cnt_ : 0.0;
        const double frac_adj = id_frames_ ? (double)id_frames_with_adj_ / id_frames_ : 0.0;
        const double overseg  = roots.empty() ? 0.0 : (double)raw_thing_ids / (double)roots.size();
        std::fprintf(stderr,
            "[exp-ids] frames=%d mean_ids_per_thing_class_per_frame=%.2f "
            "frac_frames_with_adjacent_sameclass_ids=%.2f raw_thing_ids=%d "
            "merged_groups(by_co-touch)=%zu overseg_factor=%.2f\n",
            id_frames_, mean_ids, frac_adj, raw_thing_ids, roots.size(), overseg);
        std::fprintf(stderr,
            "[exp-ids] interpretation: overseg_factor>1 => same-class ids that physically "
            "touch (one object split into several ids); ~1 => ids already 1-per-object.\n");
    }

    int frames() const { return done_; }

private:
    bool isThingLabel(int lbl) const {
        if (lbl < 0) return false;
        const std::string& n = inf_.className(lbl);
        return !n.empty() && thing_names_.count(n) > 0;
    }

    // Instance-id over-segmentation accounting for one frame.
    void analyzeIds(const FrameResult& fr) {
        if (fr.h <= 0 || fr.w <= 0 || fr.id_map.size() != (std::size_t)fr.h * fr.w) return;
        ++id_frames_;
        std::unordered_map<int, int> id_label;         // id -> class label (this frame)
        for (const InfInstance& in : fr.instances) {
            id_label[in.id] = in.label;
            id_label_global_[in.id] = in.label;
            all_ids_.insert(in.id);
        }
        // Mean distinct ids per present thing class (>1 => over-segmentation).
        std::map<int, std::set<int>> class_ids;
        for (const auto& kv : id_label)
            if (isThingLabel(kv.second)) class_ids[kv.second].insert(kv.first);
        for (const auto& kv : class_ids) {
            id_perblob_sum_ += (double)kv.second.size();
            id_perblob_cnt_ += 1;
        }
        // 4-adjacency of same-class ids (right + down neighbours) -> co-touch union-find.
        bool frame_has_adj = false;
        const int H = fr.h, W = fr.w;
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                const int a = fr.id_map[(std::size_t)v * W + u];
                if (a < 0) continue;
                if (u + 1 < W) adjPair(a, fr.id_map[(std::size_t)v * W + u + 1], id_label, frame_has_adj);
                if (v + 1 < H) adjPair(a, fr.id_map[(std::size_t)(v + 1) * W + u], id_label, frame_has_adj);
            }
        }
        if (frame_has_adj) ++id_frames_with_adj_;
    }
    void adjPair(int a, int b, const std::unordered_map<int, int>& id_label, bool& has_adj) {
        if (b < 0 || a == b) return;
        auto ia = id_label.find(a), ib = id_label.find(b);
        if (ia == id_label.end() || ib == id_label.end()) return;
        if (ia->second != ib->second) return;          // different class -> not one object
        if (!isThingLabel(ia->second)) return;
        has_adj = true;
        dsuUnion(a, b);
    }
    int dsuFind(int x) {
        auto it = parent_.find(x);
        if (it == parent_.end()) { parent_[x] = x; return x; }
        int root = x;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[x] != root) { int nx = parent_[x]; parent_[x] = root; x = nx; }
        return root;
    }
    void dsuUnion(int a, int b) {
        int ra = dsuFind(a), rb = dsuFind(b);
        if (ra != rb) parent_[ra] = rb;
    }

    Universe&        uni_;
    InfClient&       inf_;
    semantic::Params p_;
    std::vector<int> snaps_;
    int              stop_ = 0;
    int              done_ = 0;

    std::unordered_set<std::string> thing_names_;

    // id-overlap accumulators
    int    id_frames_ = 0;
    int    id_frames_with_adj_ = 0;
    double id_perblob_sum_ = 0.0;
    long   id_perblob_cnt_ = 0;
    std::unordered_map<int, int> id_label_global_;     // id -> class label (last seen)
    std::unordered_set<int>      all_ids_;
    std::unordered_map<int, int> parent_;              // co-touch DSU over ids
};

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "--config")) {
        std::fprintf(stderr, "usage: %s --config run.yaml [--log <folder>]\n"
                             "  env: EXP_SNAPSHOTS=400,600  EXP_WARMUP=<stop frame>\n", argv[0]);
        return 2;
    }
    RunConfig cfg;
    try {
        cfg = loadRunConfig(arg_str(argc, argv, "--config", ""));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config error: %s\n", e.what());
        return 2;
    }
    cfg.log = arg_str(argc, argv, "--log", cfg.log);
    semantic::Params& p = cfg.p;

    // Snapshot frames + stop point from env (defaults {400,600}).
    std::vector<int> snaps;
    if (const char* e = std::getenv("EXP_SNAPSHOTS")) snaps = parseInts(e);
    if (snaps.empty()) snaps = {400, 600};
    if (const char* e = std::getenv("EXP_WARMUP")) {
        try { snaps.push_back(std::stoi(e)); } catch (...) {}
    }
    std::sort(snaps.begin(), snaps.end());
    snaps.erase(std::unique(snaps.begin(), snaps.end()), snaps.end());
    int stop = 0; for (int s : snaps) stop = std::max(stop, s);

    try {
        CameraCalib cam = loadCameraCalib(cfg.calib, cfg.cam);
        DogLogReader reader(cfg.log + "/PoseAndPointClouds.bin");
        reader.openCamera(cfg.log + "/camera" + std::to_string(cfg.cam) + ".gclf");
        std::printf("exp_per_view | scans=%d images=%d | cam %d %dx%d | voxel %.3f | "
                    "radius %.2f proj_radius %.2f | tau %.2f splat %d\n",
                    reader.size(), reader.numImages(), cam.sensor_id, cam.width, cam.height,
                    p.voxel, p.radius, p.proj_radius, p.tau, p.splat);
        std::printf("snapshots at synced frames: ");
        for (int s : snaps) std::printf("%d ", s);
        std::printf("| stop after %d\n", stop);
        if (reader.size() == 0) return 1;

        InfClient inf(p.endpoint);
        if (!inf.ping()) { std::fprintf(stderr, "inf_server not responding at %s\n",
                                        p.endpoint.c_str()); return 1; }

        Universe uni(p.voxel);
        uni.setLocalRadius(p.radius);

        Experiment exp(uni, inf, p, snaps);
        DogStream stream(cam, exp.hook());
        exp.begin();
        DogLogIngestor ingestor(reader);
        // Feed enough images to reach the last snapshot: images are 5 Hz and stride-
        // subsampled, so ~stop*stride images (+ margin for sync drops) yields ~stop
        // synced frames. The hook stops working once done_ > stop.
        const int step = p.stride > 0 ? p.stride : 1;
        const int max_images = (stop + 8) * step;
        ingestor.run(stream, p.start, max_images, p.stride);

        std::printf("processed %d synced frames | world=%d points (%d live)\n",
                    exp.frames(), uni.size(), uni.aliveCount());
        exp.report();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
