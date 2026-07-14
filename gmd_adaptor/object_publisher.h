#pragma once
// -------------------------------------------------------------------------------------------------
// Object egress: universe::Object -> Common::Entity::EAIRoomObject.
//
// This is the sample egress side of the GMD adaptor (the mirror of gmd_frame_source's ingress).
// It is deliberately PCL/CUDA/server-free: the pipeline caller pulls each object's alive member xyz
// out of the Universe cloud and hands us plain (x,y,z) triples; everything here is pure C++ + the
// GMD entity types, so it unit-tests without a GPU.
//
// The overhead polygon is a 2D bird's-eye convex hull: the lidar world is Z-up right-handed, so we
// project member points onto the X-Y plane (drop Z), reject strays with a kNN statistical-outlier
// pass so the hull isn't ballooned by grown boundary points, then take the convex hull.
//
// Each object carries a frame-stable objectId derived from the universe object id: the downstream
// global store overwrites by objectId, so returning an object whose id already exists updates it,
// and a new id creates it.
// -------------------------------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>                              // MUST precede Object.h: that vendor header uses
                                               // std::vector without including <vector> itself.
#include "Common/Entity/Object.h"

namespace gmd {

using Point2Df      = Common::Entity::Point2D<float>;
using EAINodeId     = Common::Entity::EAINodeId;
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
// EAINodeId (so the downstream store upserts it). ident encodes object_id as a raw tick count;
// vehicleId should be non-zero so the id is never the NULL (0,0) node.
EAINodeId makeNodeId(uint16_t vehicleId, int object_id);

// The NULL node id: {vehicleId = 0, ident = epoch}. Used for roomId (assigned later).
EAINodeId nullNodeId();

// Equality for EAINodeId (compares vehicleId and the ident tick count).
inline bool nodeIdEqual(const EAINodeId& a, const EAINodeId& b) {
    return a.vehicleId == b.vehicleId &&
           a.ident.time_since_epoch().count() == b.ident.time_since_epoch().count();
}

// Build a published object from its 3D member points: project -> SOR -> convex hull, centroid from
// the cleaned points. roomId is NULL and directionContent is "" (sign detection integrated later).
EAIRoomObject buildRoomObject(uint16_t vehicleId, int object_id, std::string label,
                              const std::vector<std::array<float, 3>>& member_xyz,
                              int sor_k = 8, float sor_alpha = 1.0f);

// ------------------------------- change tracking (return only the delta) -------------------------

// Remembers a fingerprint of each object it has emitted, keyed by objectId, so it can return only
// the objects that are NEW (id never seen) or CHANGED (label / polygon / centroid differs) since
// the last call. It stores one hash per id, NOT the objects themselves. Deletions are not tracked
// (it never reports an object that vanished from the input).
class ObjectDelta {
public:
    // Returns the subset of `objects` that are new or changed, and records their fingerprints so a
    // later identical object is suppressed. Unchanged objects are dropped.
    std::vector<EAIRoomObject> changedSince(const std::vector<EAIRoomObject>& objects);

    void clear() { seen_.clear(); }
    std::size_t trackedCount() const { return seen_.size(); }

private:
    std::unordered_map<std::uint64_t, std::size_t> seen_;  // objectId key -> fingerprint
};

}  // namespace gmd
