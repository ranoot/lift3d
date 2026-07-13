#include "object_publisher.h"

#include <algorithm>
#include <cmath>

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

bool sameRoomObject(const EAIRoomObject& a, const EAIRoomObject& b, float eps) {
    if (a.label != b.label) return false;
    if (a.directionContent != b.directionContent) return false;
    if (!nodeIdEqual(a.roomId, b.roomId)) return false;
    if (std::fabs(a.centroid.x - b.centroid.x) > eps ||
        std::fabs(a.centroid.y - b.centroid.y) > eps)
        return false;
    if (a.polygon.size() != b.polygon.size()) return false;
    for (std::size_t i = 0; i < a.polygon.size(); ++i) {
        if (std::fabs(a.polygon[i].x - b.polygon[i].x) > eps ||
            std::fabs(a.polygon[i].y - b.polygon[i].y) > eps)
            return false;
    }
    return true;
}

bool ObjectTopic::publish(const EAIRoomObject& obj) {
    auto it = store_.find(obj.objectId);
    if (it == store_.end()) {
        store_.emplace(obj.objectId, obj);
        if (on_update_) on_update_(obj, /*is_new=*/true);
        return true;
    }
    if (sameRoomObject(it->second, obj)) return false;  // identical re-publish -> no-op
    it->second = obj;
    if (on_update_) on_update_(obj, /*is_new=*/false);
    return true;
}

}  // namespace gmd
