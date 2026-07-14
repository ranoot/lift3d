// Unit check for the object egress utilities (universe::Object -> EAIRoomObject publisher),
// WITHOUT the heavy semantic stack (no PCL, no inf_server, no GPU). Everything under test is pure
// C++ over the GMD entity types, so this runs anywhere. It verifies:
//
//   1. convexHull2D on a filled unit square returns exactly the 4 corners (interior points dropped).
//   2. convexHull2D degenerate cases: single point, two points, all-collinear.
//   3. statisticalOutlierRemoval drops a lone far point from a tight cluster (so the hull ignores
//      it), and is a no-op on tiny inputs.
//   4. buildRoomObject invariants: directionContent == "", roomId is NULL, objectId is stable
//      across calls with the same id and differs for a different id, centroid near cluster center.
//   5. ObjectDelta returns only new/changed objects: first call returns all, an identical repeat
//      returns none, a changed object returns just that one, a new id returns just the new one.
//
// Build (Eigen-free; no CMake needed):
//   g++ -std=c++20 -I gmd_adaptor -I gmdDataTypesForInterns2 \
//       tests/object_publisher_check.cpp gmd_adaptor/object_publisher.cpp \
//       gmdDataTypesForInterns2/GMDBase/Entity/TimeStamp.cpp -o /tmp/opc && /tmp/opc

#include "object_publisher.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using gmd::Point2Df;

namespace {

int g_fails = 0;

void check(bool cond, const char* msg) {
    if (!cond) { std::printf("[FAIL] %s\n", msg); ++g_fails; }
}

// Is point (x,y) among the hull vertices (within eps)?
bool hullHas(const std::vector<Point2Df>& hull, float x, float y, float eps = 1e-3f) {
    for (const auto& p : hull)
        if (std::fabs(p.x - x) < eps && std::fabs(p.y - y) < eps) return true;
    return false;
}

}  // namespace

int main() {
    // ---- 1. convex hull of a filled unit square -------------------------------------------------
    {
        std::vector<Point2Df> pts = {
            {0, 0}, {1, 0}, {1, 1}, {0, 1},          // corners
            {0.5f, 0.5f}, {0.25f, 0.75f}, {0.9f, 0.1f}  // interior
        };
        auto hull = gmd::convexHull2D(pts);
        check(hull.size() == 4, "square hull has 4 vertices");
        check(hullHas(hull, 0, 0) && hullHas(hull, 1, 0) && hullHas(hull, 1, 1) &&
              hullHas(hull, 0, 1), "square hull is the 4 corners");
        check(!hullHas(hull, 0.5f, 0.5f), "interior point excluded from hull");
    }

    // ---- 2. degenerate hulls --------------------------------------------------------------------
    {
        auto h1 = gmd::convexHull2D({{3, 4}});
        check(h1.size() == 1, "single-point hull has 1 vertex");

        auto h2 = gmd::convexHull2D({{0, 0}, {2, 2}});
        check(h2.size() == 2, "two-point hull has 2 vertices");

        // Collinear points -> extremes only (<= 2 vertices, never a degenerate polygon).
        auto h3 = gmd::convexHull2D({{0, 0}, {1, 1}, {2, 2}, {3, 3}});
        check(h3.size() <= 2, "collinear hull collapses to extremes");
    }

    // ---- 3. statistical outlier removal ---------------------------------------------------------
    {
        std::vector<Point2Df> pts;
        // Tight 4x4 grid cluster around the origin (spacing 0.1).
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                pts.push_back({i * 0.1f, j * 0.1f});
        pts.push_back({100.0f, 100.0f});  // lone far outlier

        auto clean = gmd::statisticalOutlierRemoval(pts, /*k=*/8, /*alpha=*/1.0f);
        bool outlier_kept = false;
        for (const auto& p : clean)
            if (p.x > 50.0f) outlier_kept = true;
        check(!outlier_kept, "SOR drops the lone far outlier");
        check(clean.size() == 16, "SOR keeps the tight cluster");

        auto hull = gmd::convexHull2D(clean);
        check(!hullHas(hull, 100.0f, 100.0f), "outlier absent from cleaned hull");

        // No-op on tiny input (fewer than k+1 points).
        std::vector<Point2Df> few = {{0, 0}, {1, 0}, {0, 1}};
        auto same = gmd::statisticalOutlierRemoval(few, 8, 1.0f);
        check(same.size() == few.size(), "SOR is a no-op on tiny input");
    }

    // ---- 4. buildRoomObject invariants ----------------------------------------------------------
    {
        std::vector<std::array<float, 3>> xyz = {
            {0, 0, 5}, {1, 0, 6}, {1, 1, 7}, {0, 1, 4}, {0.5f, 0.5f, 3}
        };  // a unit square footprint at varying heights (Z must be dropped)
        auto obj = gmd::buildRoomObject(/*vehicleId=*/211, /*object_id=*/42, "chair", xyz);

        check(obj.directionContent.empty(), "directionContent is empty");
        check(gmd::nodeIdEqual(obj.roomId, gmd::nullNodeId()), "roomId is NULL");
        check(obj.objectId.vehicleId == 211, "objectId carries vehicleId");
        check(obj.polygon.size() == 4, "footprint hull is the 4 corners (Z dropped)");
        check(std::fabs(obj.centroid.x - 0.5f) < 0.2f &&
              std::fabs(obj.centroid.y - 0.5f) < 0.2f, "centroid near footprint center");

        // Stable id: same object_id -> equal node id; different id -> different.
        auto obj_same = gmd::buildRoomObject(211, 42, "chair", xyz);
        auto obj_diff = gmd::buildRoomObject(211, 43, "chair", xyz);
        check(gmd::nodeIdEqual(obj.objectId, obj_same.objectId), "objectId stable across builds");
        check(!gmd::nodeIdEqual(obj.objectId, obj_diff.objectId), "different object_id -> different id");
    }

    // ---- 5. ObjectDelta returns only new/changed objects ----------------------------------------
    {
        std::vector<std::array<float, 3>> sq = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        gmd::ObjectDelta delta;

        std::vector<gmd::EAIRoomObject> frame = {
            gmd::buildRoomObject(211, 1, "door", sq),
            gmd::buildRoomObject(211, 2, "table", sq),
        };
        auto d0 = delta.changedSince(frame);
        check(d0.size() == 2, "first call returns all objects");
        check(delta.trackedCount() == 2, "delta now tracks 2 ids");

        // Identical frame -> nothing changed.
        auto d1 = delta.changedSince(frame);
        check(d1.empty(), "identical repeat returns no updates");

        // Change object 2's geometry (a bigger footprint) -> only it comes back.
        std::vector<std::array<float, 3>> big = {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}};
        frame[1] = gmd::buildRoomObject(211, 2, "table", big);
        auto d2 = delta.changedSince(frame);
        check(d2.size() == 1, "one changed object returns just that one");
        check(!d2.empty() && d2[0].objectId.ident == frame[1].objectId.ident,
              "the returned object is the changed one (id 2)");

        // A relabel of object 1 is also a change.
        frame[0].label = "window";
        auto d3 = delta.changedSince(frame);
        check(d3.size() == 1 && d3[0].label == "window", "a relabel counts as changed");

        // A brand-new id returns only the new one.
        frame.push_back(gmd::buildRoomObject(211, 3, "chair", sq));
        auto d4 = delta.changedSince(frame);
        check(d4.size() == 1, "a new id returns just the new object");
        check(delta.trackedCount() == 3, "delta now tracks 3 ids");
    }

    std::printf("%s\n", g_fails ? "FAILED" : "ALL PASS");
    return g_fails ? 1 : 0;
}
