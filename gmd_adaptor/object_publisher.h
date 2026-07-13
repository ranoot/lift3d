#pragma once
// -------------------------------------------------------------------------------------------------
// Object egress: universe::Object -> Common::Entity::EAIRoomObject, published to a ROS2-style topic.
//
// This is the sample egress side of the GMD adaptor (the mirror of gmd_stream_adaptor's ingress).
// It is deliberately PCL/CUDA/server-free: the pipeline caller pulls each object's alive member xyz
// out of the Universe cloud and hands us plain (x,y,z) triples; everything here is pure C++ + the
// GMD entity types, so it unit-tests without a GPU.
//
// The overhead polygon is a 2D bird's-eye convex hull: the lidar world is Z-up right-handed, so we
// project member points onto the X-Y plane (drop Z), reject strays with a kNN statistical-outlier
// pass so the hull isn't ballooned by grown boundary points, then take the convex hull.
//
// The "topic" models a ROS2/DDS latched keyed topic: publishing an object with the same objectId
// overwrites the prior entry in a global store, and an update is reported through a sink callback
// (the seam a real transport publisher plugs into later).
// -------------------------------------------------------------------------------------------------

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>                              // MUST precede Object.h: that vendor header uses
                                               // std::vector without including <vector> itself.
#include "Common/Entity/Object.h"

namespace gmd {

using Point2Df    = Common::Entity::Point2D<float>;
using EAINodeId   = Common::Entity::EAINodeId;
using EAIRoomObject = Common::Entity::EAIRoomObject;

// ------------------------------- pure 2D geometry (no PCL) ---------------------------------------

// Overhead projection: drop the Z (up) axis, keeping (x, y).
std::vector<Point2Df> projectToXY(const std::vector<std::array<float, 3>>& pts3d);

// kNN statistical outlier removal. For each point, take the mean distance to its k nearest
// neighbours; drop points whose mean distance exceeds global_mean + alpha * global_stddev.
// No-op (returns a copy) when there are too few points to define k neighbours (size <= k + 1).
std::vector<Point2Df> statisticalOutlierRemoval(const std::vector<Point2Df>& pts,
                                                 int k = 8, float alpha = 1.0f);

// Andrew's monotone-chain convex hull. Returns the hull vertices in CCW order without repeating the
// first vertex at the end. Degenerate input (< 3 unique points, or all-collinear) returns the
// unique extreme points (0, 1, or 2 of them).
std::vector<Point2Df> convexHull2D(std::vector<Point2Df> pts);

// Arithmetic mean of the points. Returns {0,0} for an empty set.
Point2Df centroid2D(const std::vector<Point2Df>& pts);

// ------------------------------- EAIRoomObject construction --------------------------------------

// Deterministic, frame-stable node id so the same universe object id always maps to the same
// EAINodeId (required for upsert-by-id). ident encodes object_id as a raw tick count; vehicleId
// should be non-zero so the id is never the NULL (0,0) node.
EAINodeId makeNodeId(uint16_t vehicleId, int object_id);

// The NULL node id: {vehicleId = 0, ident = epoch}. Used for roomId (assigned later).
EAINodeId nullNodeId();

// Build a published object from its 3D member points: project -> SOR -> convex hull, centroid from
// the cleaned points. roomId is NULL and directionContent is "" (sign detection integrated later).
EAIRoomObject buildRoomObject(uint16_t vehicleId, int object_id, std::string label,
                              const std::vector<std::array<float, 3>>& member_xyz,
                              int sor_k = 8, float sor_alpha = 1.0f);

// ------------------------------- ROS2-style keyed topic ------------------------------------------

// Equality / hashing for EAINodeId so it can key the global store.
inline bool nodeIdEqual(const EAINodeId& a, const EAINodeId& b) {
    return a.vehicleId == b.vehicleId &&
           a.ident.time_since_epoch().count() == b.ident.time_since_epoch().count();
}
struct EAINodeIdHash {
    std::size_t operator()(const EAINodeId& n) const {
        std::size_t h1 = std::hash<uint16_t>{}(n.vehicleId);
        std::size_t h2 = std::hash<long long>{}(
            static_cast<long long>(n.ident.time_since_epoch().count()));
        return h1 ^ (h2 * 0x9E3779B97F4A7C15ULL);
    }
};
struct EAINodeIdEq {
    bool operator()(const EAINodeId& a, const EAINodeId& b) const { return nodeIdEqual(a, b); }
};

// A latched, keyed topic backed by an in-process global store. publish() upserts by objectId and
// reports (fires the sink) only when the object is new or actually changed; an identical re-publish
// is a silent no-op. A real ROS2/DDS publisher attaches via setOnUpdate().
class ObjectTopic {
public:
    // (object, is_new): is_new == true on first insert, false when an existing entry changed.
    using UpdateSink = std::function<void(const EAIRoomObject&, bool /*is_new*/)>;
    using Store = std::unordered_map<EAINodeId, EAIRoomObject, EAINodeIdHash, EAINodeIdEq>;

    void setOnUpdate(UpdateSink sink) { on_update_ = std::move(sink); }

    // Returns true if the store changed (new or updated), false if identical to the stored entry.
    bool publish(const EAIRoomObject& obj);

    const Store& store() const { return store_; }
    std::size_t  size() const { return store_.size(); }

private:
    Store      store_;
    UpdateSink on_update_;
};

// True when two published objects are semantically identical (used by ObjectTopic to suppress
// no-op re-publishes). Compares label, directionContent, roomId, centroid, and polygon.
bool sameRoomObject(const EAIRoomObject& a, const EAIRoomObject& b, float eps = 1e-4f);

}  // namespace gmd
