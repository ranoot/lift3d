#include "universe.h"

#include <pcl/kdtree/kdtree_flann.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

// ---- SemanticVocabulary ------------------------------------------------------
int SemanticVocabulary::intern(const std::string& name) {
    if (name.empty()) return -1;
    auto it = ids_.find(name);
    if (it != ids_.end()) return it->second;
    const int id = (int)names_.size();
    names_.push_back(name);
    kinds_.push_back(ClassKind::Unknown);            // parallel; kept in lockstep
    ids_.emplace(name, id);
    return id;
}
int SemanticVocabulary::declare(const std::string& name, ClassKind k) {
    const int id = intern(name);
    if (id >= 0 && kinds_[id] == ClassKind::Unknown) kinds_[id] = k;
    return id;
}
int SemanticVocabulary::id(const std::string& name) const {
    auto it = ids_.find(name);
    return it == ids_.end() ? -1 : it->second;
}
const std::string& SemanticVocabulary::name(int id) const {
    if (id < 0 || id >= (int)names_.size()) return empty_;
    return names_[id];
}
ClassKind SemanticVocabulary::kind(int id) const {
    if (id < 0 || id >= (int)kinds_.size()) return ClassKind::Unknown;
    return kinds_[id];
}

Universe::Universe(float voxel_size)
    : voxel_(voxel_size > 0.0f ? voxel_size : 0.05f),
      map_(new Cloud) {}

void Universe::setVoxelSize(float v) {
    if (v > 0.0f) voxel_ = v;
}

void Universe::setLocalRadius(float r) {
    local_radius_ = (r > 0.0f) ? r : 0.0f;
}

void Universe::clear() {
    map_->clear();
    count_.clear();
    alive_.clear();
    vox_.clear();
    views_.clear();
    votes_.clear();
    color_.clear();
    color_n_.clear();
    sem_ = SemanticVocabulary{};
    last_robot_[0] = last_robot_[1] = last_robot_[2] = 0.0f;
}

std::size_t Universe::VKeyHash::operator()(const VKey& k) const {
    std::size_t h = std::hash<int>()(k.x);
    h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

Universe::VKey Universe::voxelOf(float x, float y, float z) const {
    return VKey{ static_cast<int>(std::floor(x / voxel_)),
                 static_cast<int>(std::floor(y / voxel_)),
                 static_cast<int>(std::floor(z / voxel_)) };
}

int Universe::addView(const Camera& cam, std::int64_t t_ns, const std::string& ref,
                      const float* robot_world_pos) {
    View v;
    v.cam = cam;
    v.t_ns = t_ns;
    v.image_ref = ref;
    if (robot_world_pos) {
        v.robot[0] = robot_world_pos[0];
        v.robot[1] = robot_world_pos[1];
        v.robot[2] = robot_world_pos[2];
    } else {
        v.robot[0] = cam.T[3]; v.robot[1] = cam.T[7]; v.robot[2] = cam.T[11];
    }
    views_.push_back(v);
    return static_cast<int>(views_.size()) - 1;
}

int Universe::integrate(const Cloud& cloud_world, const Camera& view,
                        std::int64_t t_ns, const std::string& image_ref,
                        const float* robot_world_pos) {
    // Record where the robot was for this observation (centre for later local ops).
    if (robot_world_pos) {
        last_robot_[0] = robot_world_pos[0];
        last_robot_[1] = robot_world_pos[1];
        last_robot_[2] = robot_world_pos[2];
    } else {
        last_robot_[0] = view.T[3]; last_robot_[1] = view.T[7]; last_robot_[2] = view.T[11];
    }

    // Fold every incoming point into its voxel's running centroid + mean intensity.
    // New voxel -> append a point; seen voxel -> refine the existing one in place.
    // The world only ever grows; nothing is dropped. O(incoming) per frame.
    map_->reserve(map_->size() + cloud_world.size());
    for (const PointT& p : cloud_world.points) {
        const VKey k = voxelOf(p.x, p.y, p.z);
        auto it = vox_.find(k);
        if (it == vox_.end()) {
            const int idx = static_cast<int>(map_->size());
            map_->push_back(p);
            count_.push_back(1);
            alive_.push_back(1);            // born live; parallel to map_
            votes_.emplace_back();          // keep the vote store parallel to map_
            color_.push_back({0.0f, 0.0f, 0.0f});   // colour fused later from images
            color_n_.push_back(0);
            vox_.emplace(k, idx);
        } else {
            const int idx = it->second;
            PointT& m = (*map_)[idx];
            if (!alive_[idx]) {
                // This voxel was tombstoned (a dynamic object sat here) and its
                // previous occupant is gone. Reclaim the slot as a FRESH point so
                // real static geometry can take over the spot -- otherwise a place a
                // person walked through would blackhole forever. Old votes/colour are
                // discarded; the new observation defines the point from scratch.
                m = p;
                count_[idx] = 1;
                votes_[idx].clear();
                color_[idx] = {0.0f, 0.0f, 0.0f};
                color_n_[idx] = 0;
                alive_[idx] = 1;
            } else {
                const float n = static_cast<float>(count_[idx]);
                const float inv = 1.0f / (n + 1.0f);
                m.x = (m.x * n + p.x) * inv;
                m.y = (m.y * n + p.y) * inv;
                m.z = (m.z * n + p.z) * inv;
                m.intensity = (m.intensity * n + p.intensity) * inv;
                count_[idx] = count_[idx] + 1;
            }
        }
    }

    return addView(view, t_ns, image_ref, robot_world_pos);
}

Universe::Cloud::Ptr Universe::local(const float center[3], float radius,
                                     std::vector<int>* out_indices) const {
    Cloud::Ptr out(new Cloud);
    if (out_indices) out_indices->clear();

    // radius <= 0 => the whole world. Still filters tombstoned points (they are not
    // part of the readable world), so this is a compacting copy, not an identity one.
    const int M = static_cast<int>(map_->size());
    if (radius <= 0.0f) {
        out->reserve(M);
        for (int i = 0; i < M; ++i) {
            if (!alive_[i]) continue;
            out->push_back((*map_)[i]);
            if (out_indices) out_indices->push_back(i);
        }
        return out;
    }
    if (map_->empty()) return out;

    // Linear distance (sphere) scan over the world map -> local subset, skipping dead
    // points. O(M) with no allocation beyond the output. This deliberately replaces a
    // per-call KdTreeFLANN build over the whole (growing) map: that tree build ran every
    // frame and dominated the ~300 ms projection cost. Same sphere semantics.
    const float r2 = radius * radius;
    for (int i = 0; i < M; ++i) {
        if (!alive_[i]) continue;
        const PointT& p = (*map_)[i];
        const float dx = p.x - center[0];
        const float dy = p.y - center[1];
        const float dz = p.z - center[2];
        if (dx * dx + dy * dy + dz * dz <= r2) {
            out->push_back(p);
            if (out_indices) out_indices->push_back(i);
        }
    }
    return out;
}

Universe::Cloud::Ptr Universe::localAroundRobot(std::vector<int>* out_indices) const {
    return local(last_robot_, local_radius_, out_indices);
}

namespace {
// Copy a PointXYZI cloud's xyz into contiguous 3N floats (PointXYZI is padded).
std::vector<float> xyzOf(const Universe::Cloud& c) {
    const int N = static_cast<int>(c.size());
    std::vector<float> xyz(static_cast<std::size_t>(3) * N);
    for (int i = 0; i < N; ++i) {
        const Universe::PointT& p = c[i];
        xyz[3 * i + 0] = p.x;
        xyz[3 * i + 1] = p.y;
        xyz[3 * i + 2] = p.z;
    }
    return xyz;
}
} // namespace

PointPixelMap Universe::project(int view_id, float tau_vis, int zbuf_dilate) const {
    const View& v = views_.at(view_id);
    std::vector<float> xyz = xyzOf(*map_);
    return mapPointsToPixels(xyz.data(), size(), v.cam, tau_vis, zbuf_dilate);
}

PointPixelMap Universe::projectLocal(int view_id, float tau_vis, float radius,
                                     std::vector<int>* global_indices,
                                     int zbuf_dilate) const {
    const View& v = views_.at(view_id);
    const float r = (radius < 0.0f) ? local_radius_ : radius;

    Cloud::Ptr sub = local(v.robot, r, global_indices);
    std::vector<float> xyz = xyzOf(*sub);
    return mapPointsToPixels(xyz.data(), static_cast<int>(sub->size()), v.cam,
                             tau_vis, zbuf_dilate);
}

void Universe::setLabelGate(int min_votes, float min_conf) {
    min_votes_ = min_votes > 1 ? min_votes : 1;
    min_conf_  = min_conf < 0.0f ? 0.0f : (min_conf > 1.0f ? 1.0f : min_conf);
}

// ---- per-point semantics -----------------------------------------------------
void Universe::declareVocab(const std::vector<std::string>& thing,
                            const std::vector<std::string>& stuff) {
    for (const std::string& n : thing) sem_.declare(n, ClassKind::Thing);
    for (const std::string& n : stuff) sem_.declare(n, ClassKind::Stuff);
}

void Universe::voteLabel(int global_idx, const std::string& class_name) {
    if (global_idx < 0 || global_idx >= (int)votes_.size()) return;
    const int cid = sem_.intern(class_name);
    if (cid < 0) return;                                   // empty/background -> no vote
    auto& hist = votes_[global_idx];
    for (auto& pc : hist) {
        if (pc.first == (uint16_t)cid) {
            if (pc.second < 0xffff) ++pc.second;           // saturate, don't wrap
            return;
        }
    }
    hist.emplace_back((uint16_t)cid, (uint16_t)1);
}

void Universe::voteColor(int global_idx, unsigned char r, unsigned char g,
                         unsigned char b) {
    if (global_idx < 0 || global_idx >= (int)color_.size()) return;
    std::array<float, 3>& c = color_[global_idx];
    const float n = (float)color_n_[global_idx];
    const float inv = 1.0f / (n + 1.0f);
    c[0] = (c[0] * n + (float)r) * inv;
    c[1] = (c[1] * n + (float)g) * inv;
    c[2] = (c[2] * n + (float)b) * inv;
    ++color_n_[global_idx];
}

bool Universe::pointColor(int i, float rgb[3]) const {
    if (i < 0 || i >= (int)color_.size() || color_n_[i] == 0) return false;
    rgb[0] = color_[i][0];
    rgb[1] = color_[i][1];
    rgb[2] = color_[i][2];
    return true;
}

int Universe::pointColorCount(int i) const {
    return (i >= 0 && i < (int)color_n_.size()) ? (int)color_n_[i] : 0;
}

namespace {
// argmax of a sparse histogram; ties broken by smaller class id (deterministic).
// Returns {class_id, winning_count, total}; class_id = -1 when empty.
struct ArgMax { int id; int win; int total; };
ArgMax argmaxHist(const std::vector<std::pair<uint16_t, uint16_t>>& h) {
    ArgMax r{-1, 0, 0};
    for (const auto& pc : h) {
        r.total += pc.second;
        if (pc.second > r.win || (pc.second == r.win && (r.id < 0 || pc.first < r.id))) {
            r.win = pc.second; r.id = pc.first;
        }
    }
    return r;
}
} // namespace

int Universe::pointClassId(int i) const {
    if (i < 0 || i >= (int)votes_.size()) return -1;
    ArgMax a = argmaxHist(votes_[i]);
    if (a.id < 0) return -1;
    if (a.total < min_votes_) return -1;                       // too few votes
    if (min_conf_ > 0.0f && (float)a.win < min_conf_ * (float)a.total)
        return -1;                                             // too ambiguous
    return a.id;
}

std::string Universe::pointClassName(int i) const {
    return sem_.name(pointClassId(i));
}

float Universe::pointClassConfidence(int i) const {
    if (i < 0 || i >= (int)votes_.size()) return 0.0f;
    ArgMax a = argmaxHist(votes_[i]);
    return a.total > 0 ? (float)a.win / (float)a.total : 0.0f;
}

int Universe::pointVoteTotal(int i) const {
    if (i < 0 || i >= (int)votes_.size()) return 0;
    return argmaxHist(votes_[i]).total;
}

ClassKind Universe::pointClassKind(int i) const {
    return sem_.kind(pointClassId(i));             // -1 (unlabeled/gated) -> Unknown
}

// ---- tombstoning -------------------------------------------------------------
void Universe::killPoint(int i) {
    if (i < 0 || i >= (int)alive_.size()) return;
    alive_[i] = 0;
    votes_[i].clear();                             // also reads as unlabeled if not skipped
}

bool Universe::pointAlive(int i) const {
    return i >= 0 && i < (int)alive_.size() && alive_[i] != 0;
}

int Universe::aliveCount() const {
    int n = 0;
    for (std::uint8_t a : alive_) n += (a != 0);
    return n;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr Universe::labeledCloud() const {
    pcl::PointCloud<pcl::PointXYZL>::Ptr out(new pcl::PointCloud<pcl::PointXYZL>);
    out->reserve(map_->size());
    for (int i = 0; i < (int)map_->size(); ++i) {
        if (!alive_[i]) continue;                         // tombstoned -> not exported
        const PointT& p = (*map_)[i];
        pcl::PointXYZL q;
        q.x = p.x; q.y = p.y; q.z = p.z;
        const int cid = pointClassId(i);
        q.label = (cid < 0) ? 0u : (uint32_t)(cid + 1);   // 0 == unlabeled
        out->push_back(q);
    }
    out->width = out->size();
    out->height = 1;
    out->is_dense = false;
    return out;
}
