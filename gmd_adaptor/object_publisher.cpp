#include "object_publisher.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gmd {

std::vector<Point2Df> projectToXY(const std::vector<std::array<float, 3>>& pts3d) {
    std::vector<Point2Df> out;
    out.reserve(pts3d.size());
    for (const auto& p : pts3d) out.push_back(Point2Df{p[0], p[1]});
    return out;
}

std::vector<Point2Df> statisticalOutlierRemoval(const std::vector<Point2Df>& pts,
                                                 int k, float alpha) {
    const int n = static_cast<int>(pts.size());
    if (k < 1 || n <= k + 1) return pts;  // too few points to define k neighbours -> no-op

    // Mean distance from each point to its k nearest neighbours (brute-force; per-object n is small).
    std::vector<float> mean_knn(n, 0.0f);
    std::vector<float> d(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const float dx = pts[i].x - pts[j].x;
            const float dy = pts[i].y - pts[j].y;
            d[j] = std::sqrt(dx * dx + dy * dy);
        }
        // Partial-sort so d[1..k] are the k smallest non-self distances (d[0] is the self 0).
        std::partial_sort(d.begin(), d.begin() + (k + 1), d.end());
        float s = 0.0f;
        for (int m = 1; m <= k; ++m) s += d[m];
        mean_knn[i] = s / static_cast<float>(k);
    }

    // Global mean / stddev of the per-point mean-knn distances.
    double sum = 0.0;
    for (float v : mean_knn) sum += v;
    const double mean = sum / n;
    double var = 0.0;
    for (float v : mean_knn) var += (v - mean) * (v - mean);
    const double stddev = std::sqrt(var / n);
    const double thresh = mean + alpha * stddev;

    std::vector<Point2Df> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i)
        if (mean_knn[i] <= thresh) kept.push_back(pts[i]);
    return kept;
}

namespace {
// Cross product of OA x OB. > 0 => counter-clockwise turn at O.
double cross(const Point2Df& O, const Point2Df& A, const Point2Df& B) {
    return (static_cast<double>(A.x) - O.x) * (static_cast<double>(B.y) - O.y) -
           (static_cast<double>(A.y) - O.y) * (static_cast<double>(B.x) - O.x);
}
}  // namespace

std::vector<Point2Df> convexHull2D(std::vector<Point2Df> pts) {
    // Deduplicate + sort lexicographically by (x, y).
    std::sort(pts.begin(), pts.end(), [](const Point2Df& a, const Point2Df& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const Point2Df& a, const Point2Df& b) {
                              return a.x == b.x && a.y == b.y;
                          }),
              pts.end());

    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;  // 0, 1, or 2 unique points -> nothing to hull

    std::vector<Point2Df> hull(2 * n);
    int m = 0;
    // Lower hull.
    for (int i = 0; i < n; ++i) {
        while (m >= 2 && cross(hull[m - 2], hull[m - 1], pts[i]) <= 0) --m;
        hull[m++] = pts[i];
    }
    // Upper hull.
    const int lower = m + 1;
    for (int i = n - 2; i >= 0; --i) {
        while (m >= lower && cross(hull[m - 2], hull[m - 1], pts[i]) <= 0) --m;
        hull[m++] = pts[i];
    }
    hull.resize(m - 1);  // last point == first point; drop the duplicate
    // For all-collinear input the chain already collapses to the 2 extreme points.
    return hull;
}

Point2Df centroid2D(const std::vector<Point2Df>& pts) {
    if (pts.empty()) return Point2Df{0.0f, 0.0f};
    double sx = 0.0, sy = 0.0;
    for (const auto& p : pts) { sx += p.x; sy += p.y; }
    return Point2Df{static_cast<float>(sx / pts.size()), static_cast<float>(sy / pts.size())};
}

EAINodeId makeNodeId(uint16_t vehicleId, int object_id) {
    EAINodeId id;
    id.vehicleId = vehicleId;
    id.ident = Common::Entity::TimeStamp{
        Common::Entity::TimeDuration{static_cast<Common::Entity::TimeDuration::rep>(object_id)}};
    return id;
}

EAINodeId nullNodeId() {
    EAINodeId id;
    id.vehicleId = 0;
    id.ident = Common::Entity::TimeStamp{};  // epoch
    return id;
}

EAIRoomObject buildRoomObject(uint16_t vehicleId, int object_id, std::string label,
                              const std::vector<std::array<float, 3>>& member_xyz,
                              int sor_k, float sor_alpha) {
    std::vector<Point2Df> flat = projectToXY(member_xyz);
    std::vector<Point2Df> clean = statisticalOutlierRemoval(flat, sor_k, sor_alpha);

    EAIRoomObject obj;
    obj.objectId = makeNodeId(vehicleId, object_id);
    obj.roomId = nullNodeId();
    obj.label = std::move(label);
    obj.directionContent = "";  // sign detection integrated later
    obj.polygon = convexHull2D(clean);
    obj.centroid = centroid2D(clean);
    return obj;
}

namespace {
// Pack an EAINodeId into a single 64-bit key (vehicleId in the high bits, ident tick below).
std::uint64_t nodeIdKey(const EAINodeId& n) {
    return (static_cast<std::uint64_t>(n.vehicleId) << 48) ^
           static_cast<std::uint64_t>(n.ident.time_since_epoch().count());
}
void hashCombine(std::size_t& h, std::size_t v) {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
}
std::size_t hashFloat(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));  // hash the exact bits: same input -> same fingerprint
    return std::hash<std::uint32_t>{}(u);
}
// Fingerprint of the fields a downstream store cares about (everything but objectId itself).
std::size_t fingerprint(const EAIRoomObject& o) {
    std::size_t h = std::hash<std::string>{}(o.label);
    hashCombine(h, std::hash<std::string>{}(o.directionContent));
    hashCombine(h, std::hash<std::uint64_t>{}(nodeIdKey(o.roomId)));
    hashCombine(h, hashFloat(o.centroid.x));
    hashCombine(h, hashFloat(o.centroid.y));
    for (const Point2Df& p : o.polygon) {
        hashCombine(h, hashFloat(p.x));
        hashCombine(h, hashFloat(p.y));
    }
    return h;
}
}  // namespace

std::vector<EAIRoomObject> ObjectDelta::changedSince(const std::vector<EAIRoomObject>& objects) {
    std::vector<EAIRoomObject> out;
    for (const EAIRoomObject& o : objects) {
        const std::uint64_t key = nodeIdKey(o.objectId);
        const std::size_t fp = fingerprint(o);
        auto it = seen_.find(key);
        if (it == seen_.end() || it->second != fp) {  // new id, or fingerprint changed
            seen_[key] = fp;
            out.push_back(o);
        }
    }
    return out;
}

}  // namespace gmd
