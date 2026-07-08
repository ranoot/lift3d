#pragma once
// Universe: a persistent WHOLE-WORLD point map accumulated across robot-log
// frames. The global map grows to cover the entire explored world and is never
// pruned -- geometry the robot drives away from stays in the map. "Operate
// locally" is a *read-time* crop: operations pull out the subset around the robot
// on demand; that never mutates or deletes from the world map.
//
// Design (PCL-backed):
//   - The map is a pcl::PointCloud<pcl::PointXYZI>, so geometry + lidar intensity
//     ride together and PCL spatial tooling applies.
//   - Incremental voxel fusion on add: each incoming point folds into its voxel's
//     running centroid + mean intensity through a voxel->index hash, so revisiting
//     a place refines its points instead of duplicating them. Cost is O(incoming)
//     per frame, independent of how big the world has grown. (PCL's VoxelGrid is
//     batch-only -- re-running it on the whole map every frame would be quadratic;
//     the hash is just an index, the data itself stays a PCL cloud.)
//   - Local ops: local() / localAroundRobot() / projectLocal() return the subset
//     within a radius of a centre (the robot) PLUS the matching GLOBAL indices, so
//     anything computed locally lifts straight back onto the persistent world map.
//   - Images attach as Views (Camera pose + intrinsics + an opaque image handle);
//     each View also records the robot centre so local ops can re-find it.

#include "point_pixel_mapping.cuh"   // Camera, PointPixelMap, mapPointsToPixels

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>         // pcl::PointXYZI

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Panoptic distinction for a class: a countable object ("thing", e.g. chair, door)
// vs an amorphous region ("stuff", e.g. wall, floor). Mirrors the inf_server's
// thing/stuff split. Unknown = never declared (e.g. a name only ever seen via a
// vote, or before declareVocab()).
enum class ClassKind : std::uint8_t { Unknown = 0, Thing = 1, Stuff = 2 };

// Stable, monotonically-growing registry mapping class-name strings to compact
// integer ids. This decouples the per-point semantics stored in the map from any
// single inf_server `set_vocab` call -- those calls return integer labels that are
// only meaningful relative to that call's (sorted) vocabulary. By keying stored
// votes on the class *name* via this registry, a runtime vocabulary change merely
// interns new names; ids already assigned never shift, so earlier votes stay valid.
// Each id also carries a thing/stuff ClassKind (see declare()).
class SemanticVocabulary {
public:
    int intern(const std::string& name);            // name -> id (creates on first sight); "" -> -1
    // Like intern(), but also records the class KIND. If the name already exists its
    // kind is set only when currently Unknown, so an explicit thing/stuff declaration
    // is never clobbered by a later default. "" -> -1.
    int declare(const std::string& name, ClassKind kind);
    int id(const std::string& name) const;          // -1 if unknown
    const std::string& name(int id) const;          // "" if out of range
    ClassKind kind(int id) const;                    // Unknown if out of range/undeclared
    ClassKind kindOf(const std::string& name) const { return kind(id(name)); }
    int  size() const { return (int)names_.size(); }
    const std::vector<std::string>& names() const { return names_; }

private:
    std::vector<std::string>              names_;
    std::vector<ClassKind>                kinds_;    // parallel to names_
    std::unordered_map<std::string, int>  ids_;
    std::string                           empty_;
};

// One camera observation registered against the map. Pixels/features live outside
// the universe; image_ref is a handle (path/id) to fetch them on demand. robot[]
// is the robot world position at this observation, used to re-centre local ops.
struct View {
    Camera       cam;             // T_world_cam + K + W/H (from point_pixel_mapping)
    std::int64_t t_ns = 0;        // observation timestamp (ns)
    std::string  image_ref;       // opaque handle to the image/feature map
    float        robot[3] = {0, 0, 0};  // robot world position at this observation
};

class Universe {
public:
    using PointT = pcl::PointXYZI;
    using Cloud  = pcl::PointCloud<PointT>;

    explicit Universe(float voxel_size = 0.05f);

    // ---- configuration ----
    void  setVoxelSize(float v);            // dedup/fusion resolution (m)
    float voxelSize() const { return voxel_; }
    // Default radius (m) for local operations; <= 0 means "use the whole world".
    // This NEVER deletes anything -- it only bounds what local ops read.
    void  setLocalRadius(float r);
    float localRadius() const { return local_radius_; }

    // ---- accumulation (grows the persistent world; nothing is ever removed) ----
    // Fuse a world-space cloud into the global map and register a View.
    // robot_world_pos (3 floats) records the robot centre for this observation;
    // defaults to the camera centre (view.T translation). Returns the new view id.
    int integrate(const Cloud& cloud_world, const Camera& view, std::int64_t t_ns,
                  const std::string& image_ref = "",
                  const float* robot_world_pos = nullptr);

    // ---- whole world ----
    int size() const { return map_ ? static_cast<int>(map_->size()) : 0; }
    Cloud::ConstPtr cloud() const { return map_; }   // the full accumulated world
    // Most recent robot world position (centre for local ops, e.g. VCCS crops).
    const float* robotWorld() const { return last_robot_; }

    // ---- local region (read-only crop; never mutates/deletes the map) ----
    // Subset of the world within `radius` of `center`. If out_indices is given it
    // is filled with the matching indices into the global map (so results map
    // back). radius <= 0 returns a copy of the whole world.
    Cloud::Ptr local(const float center[3], float radius,
                     std::vector<int>* out_indices = nullptr) const;
    // Same, centred on the most recent robot position, using localRadius().
    Cloud::Ptr localAroundRobot(std::vector<int>* out_indices = nullptr) const;

    // ---- views / 2D<->3D bridge ----
    int         numViews() const { return static_cast<int>(views_.size()); }
    const View& view(int i) const { return views_.at(i); }
    int         addView(const Camera& cam, std::int64_t t_ns, const std::string& ref,
                        const float* robot_world_pos = nullptr);
    // Project the WHOLE world into a registered view. The PointPixelMap indexes the
    // world points (0..size()-1). zbuf_dilate splats the z-buffer depth into a
    // (2r+1)^2 window so the sparse voxel map occludes as a continuous surface (see
    // projectPointsToPixels); leave 0 for the legacy single-pixel behaviour.
    PointPixelMap project(int view_id, float tau_vis, int zbuf_dilate = 0) const;
    // Project only the local region around that view's robot centre (radius < 0
    // uses localRadius()). The returned map indexes the LOCAL subset; if
    // global_indices is given, global_indices[i] is the world index of local
    // point i -- so per-pixel results lift back onto the global map.
    PointPixelMap projectLocal(int view_id, float tau_vis, float radius = -1.0f,
                               std::vector<int>* global_indices = nullptr,
                               int zbuf_dilate = 0) const;

    // ---- per-point semantics (grow-only, fused across views) -----------------
    // Cast one vote for `class_name` on world point `global_idx` (an index into the
    // map, e.g. from project()/projectLocal() global_indices). The name is interned
    // into the shared registry; votes accumulate so a point observed in many frames
    // resolves to its majority class. Out-of-range indices and empty names are no-ops.
    void voteLabel(int global_idx, const std::string& class_name);

    // Fuse an observed RGB colour into world point `global_idx` (running mean, so a
    // point seen in many frames converges to its average colour). Colour is sampled
    // from the camera image during projection/voting and is NOT lidar-derived (the
    // map is PointXYZI); it feeds the Mask3D backbone, which expects RGB. Out-of-range
    // indices are no-ops. Channels are 0..255.
    void voteColor(int global_idx, unsigned char r, unsigned char g, unsigned char b);
    // Mean accumulated colour of point i into rgb[3] (0..255). Returns false if the
    // point has never been coloured (no camera ever saw it).
    bool pointColor(int i, float rgb[3]) const;
    int  pointColorCount(int i) const;             // # of colour observations fused

    // Declare the working vocabulary's thing/stuff kinds up front, so every point's
    // resolved class carries the distinction even before (or without) any votes.
    // Names are interned into the registry; safe to call repeatedly (e.g. on each
    // runtime vocab change) -- it is additive and never reshuffles existing ids.
    void declareVocab(const std::vector<std::string>& thing,
                      const std::vector<std::string>& stuff);

    // Label gate: a point resolves to a class only if it has at least `min_votes`
    // total votes AND the winning class holds at least `min_conf` of them (0..1).
    // Points below the gate report as unlabeled (pointClassId == -1). This filters
    // sparse/ambiguous points -- e.g. specular-reflection points that only leaked a
    // handful of frustum votes -- without touching the stored histograms. Defaults
    // (1, 0) reproduce the old ungated argmax. Confidence is unaffected by the gate.
    void setLabelGate(int min_votes, float min_conf);
    int   labelMinVotes() const { return min_votes_; }
    float labelMinConf()  const { return min_conf_; }

    // Resolved class for world point i = argmax of its vote histogram, subject to
    // the label gate above.
    int         pointClassId(int i)   const;   // stable registry id, -1 if unvoted/gated
    std::string pointClassName(int i) const;   // "" if unvoted
    float       pointClassConfidence(int i) const;  // winning_votes/total, 0 if unvoted
    int         pointVoteTotal(int i) const;   // total votes cast on point i
    ClassKind   pointClassKind(int i) const;   // thing/stuff of the resolved class
    bool        pointIsThing(int i) const { return pointClassKind(i) == ClassKind::Thing; }
    bool        pointIsStuff(int i) const { return pointClassKind(i) == ClassKind::Stuff; }

    const SemanticVocabulary& semantics() const { return sem_; }

    // ---- tombstoning (soft delete) -------------------------------------------
    // Mark a world point DEAD so every read-time enumeration skips it: local()
    // crops (hence projection candidates, features, superpoints, seeds), export,
    // and viz. The point KEEPS its index -- nothing is erased or reindexed -- so
    // all downstream global indices (vox_, votes_, features, seeds/objects) stay
    // valid; dead points are simply filtered wherever the world is read. Its vote
    // histogram is cleared so it also reads as unlabeled if a caller forgets to
    // skip it. Used to drop dynamic objects (people) that the wide-FOV lidar fused
    // but a camera later catches on a "dynamic" 2D segment. Out-of-range => no-op.
    // A later fuse into the same voxel REVIVES the slot (the geometry there changed),
    // so a spot a person vacated can be reclaimed by real static geometry.
    void killPoint(int global_idx);
    bool pointAlive(int i) const;   // false if dead OR out of range
    int  aliveCount() const;        // # live points (size() still counts dead)

    // Build a PointXYZL cloud (xyz + resolved class id) for PCD export / PCL label
    // tools. label = pointClassId+1 (0 == unlabeled). Geometry stays PointXYZI live.
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeledCloud() const;

    void clear();

private:
    struct VKey {
        int x, y, z;
        bool operator==(const VKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct VKeyHash { std::size_t operator()(const VKey& k) const; };

    VKey voxelOf(float x, float y, float z) const;

    float      voxel_;
    float      local_radius_ = 0.0f;          // 0 => local ops see the whole world
    int        min_votes_ = 1;                // label gate: min total votes
    float      min_conf_  = 0.0f;             // label gate: min winning fraction
    Cloud::Ptr map_;                          // persistent PointXYZI world (world frame)
    std::vector<int> count_;                  // per-point fusion count, parallel to map_
    std::vector<std::uint8_t> alive_;         // per-point tombstone (1=live), parallel to map_
    std::unordered_map<VKey, int, VKeyHash> vox_;  // voxel -> index into map_
    std::vector<View> views_;
    float      last_robot_[3] = {0, 0, 0};    // most recent robot world position

    // Per-point semantic vote histogram, parallel to map_ (grows with it). Sparse:
    // each point holds only the (class_id,count) pairs it has actually seen, so
    // memory is proportional to observed class diversity, and adding classes at
    // runtime is free. Class ids index sem_ (stable across vocab changes).
    SemanticVocabulary                                sem_;
    std::vector<std::vector<std::pair<uint16_t, uint16_t>>> votes_;

    // Per-point running-mean RGB (0..255) + observation count, parallel to map_.
    // Populated from camera images during voting (see voteColor); the Mask3D backbone
    // consumes it as its 3-channel colour input. color_[i] is meaningful only when
    // color_n_[i] > 0.
    std::vector<std::array<float, 3>>                 color_;
    std::vector<std::uint32_t>                        color_n_;
};
