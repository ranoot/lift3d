#include "superpoints.h"
#include "point_features.h"                           // PointFeatures (per-point Mask3D feats)

#include <pcl/point_types.h>                          // pcl::PointXYZ, PointXYZL
#include <pcl/segmentation/supervoxel_clustering.h>   // VCCS (heavy; kept out of the header)

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace {

// Run VCCS on a geometry-only PointXYZ cloud. Fills `out_labels` (parallel to the
// input: label 0 = unlabeled, >0 = supervoxel id local to THIS call). Returns the
// number of distinct supervoxels (max label). getLabeledCloud() copyPointCloud's
// the input, so the labels stay input-aligned (same size/order) -- which is what
// lets us map crop point j straight back to its universe global index.
std::uint32_t segmentCrop(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          const Superpoints::Params& p,
                          std::vector<int>& out_labels) {
    out_labels.assign(cloud ? cloud->size() : 0, 0);
    if (!cloud || cloud->size() < 8) return 0;        // too few points to segment

    pcl::SupervoxelClustering<pcl::PointXYZ> vccs(p.voxel_res, p.seed_res);
    // Our world map is an UNORGANIZED cloud, so disable the single-camera (Kinect
    // depth) transform, which is meant for organized range images.
    vccs.setUseSingleCameraTransform(false);
    vccs.setInputCloud(cloud);
    vccs.setColorImportance(p.color_w);               // 0: geometry only (no RGB on PointXYZ)
    vccs.setSpatialImportance(p.spatial_w);
    vccs.setNormalImportance(p.normal_w);

    std::map<std::uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> clusters;
    vccs.extract(clusters);
    if (p.refine > 0) vccs.refineSupervoxels(p.refine, clusters);
    if (clusters.empty()) return 0;

    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled = vccs.getLabeledCloud();
    if (!labeled || labeled->size() != cloud->size()) return 0;

    std::uint32_t maxlabel = 0;
    for (std::size_t i = 0; i < labeled->size(); ++i) {
        const std::uint32_t l = (*labeled)[i].label;
        out_labels[i] = (int)l;
        if (l > maxlabel) maxlabel = l;
    }
    return maxlabel;
}

}  // namespace

void Superpoints::refreshLocal(const Universe& uni, const float center[3],
                               float radius, const Params& p,
                               const PointFeatures* pf) {
    ++version_;                            // every refresh is a new version
    list_.clear();                         // ephemeral: drop the previous window
    if (uni.size() == 0) return;

    // Pull the crop around the robot (radius <= 0 => whole world) + its global indices.
    std::vector<int> gidx;
    Universe::Cloud::Ptr crop = uni.local(center, radius, &gidx);
    if (!crop || crop->empty()) return;

    // Geometry-only PointXYZ copy, order preserved so labels align with gidx.
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>);
    xyz->reserve(crop->size());
    for (const Universe::PointT& q : crop->points)
        xyz->emplace_back(q.x, q.y, q.z);

    std::vector<int> labels;
    const std::uint32_t k = segmentCrop(xyz, p, labels);
    if (k == 0) return;

    // Build the list from THIS crop's labels only; nothing from earlier windows.
    buildFromCrop(gidx, labels, k, uni, p, pf);
}

void Superpoints::refreshWhole(const Universe& uni, const Params& p,
                               const PointFeatures* pf) {
    // A zero-centred whole-world crop (radius <= 0 => the entire map).
    const float c[3] = {0.0f, 0.0f, 0.0f};
    refreshLocal(uni, c, -1.0f, p, pf);
}

void Superpoints::buildFromCrop(const std::vector<int>& gidx,
                                const std::vector<int>& labels, std::uint32_t k,
                                const Universe& uni, const Params& p,
                                const PointFeatures* pf) {
    list_.clear();

    // Group crop points by their per-window VCCS label (1..k) into compact list
    // slots; each member stores its universe global index gidx[j] so results lift
    // straight back onto the world map.
    std::vector<int> label_to_slot(k + 1, -1);     // label 0 = unlabeled -> skipped
    for (std::size_t j = 0; j < labels.size(); ++j) {
        const int l = labels[j];
        if (l <= 0 || (std::uint32_t)l > k) continue;
        int slot = label_to_slot[l];
        if (slot < 0) {
            slot = (int)list_.size();
            label_to_slot[l] = slot;
            list_.emplace_back();
            list_.back().id = (std::uint32_t)l;    // per-window label, not a global id
        }
        list_[slot].points.push_back(gidx[j]);
    }
    Universe::Cloud::ConstPtr map = uni.cloud();

    // Centroid + semantics per superpoint. Semantics: pick the majority THING class,
    // but measure its share over ALL member points (thing + stuff + unlabeled), so a
    // superpoint is only called a thing when that thing dominates the whole blob --
    // mostly-unlabeled/stuff superpoints stay Unknown even if their few labeled points agree.
    const SemanticVocabulary& sv = uni.semantics();
    for (Superpoint& sp : list_) {
        double cx = 0, cy = 0, cz = 0;
        std::map<int, int> thing_votes;   // thing class id -> member count
        int thing_total = 0;
        for (int i : sp.points) {
            const Universe::PointT& q = (*map)[i];
            cx += q.x; cy += q.y; cz += q.z;
            const int cid = uni.pointClassId(i);
            if (cid >= 0 && sv.kind(cid) == ClassKind::Thing) {
                ++thing_votes[cid];
                ++thing_total;
            }
        }
        const double inv = sp.points.empty() ? 0.0 : 1.0 / (double)sp.points.size();
        sp.centroid[0] = (float)(cx * inv);
        sp.centroid[1] = (float)(cy * inv);
        sp.centroid[2] = (float)(cz * inv);

        const int n_total = (int)sp.points.size();     // ALL members, incl. stuff/unlabeled
        sp.class_id = -1;
        sp.kind = ClassKind::Unknown;
        sp.confidence = 0.0f;
        if (thing_total > 0 && n_total > 0) {
            int best_id = -1, best_n = 0;
            for (const auto& kv : thing_votes)
                if (kv.second > best_n) { best_n = kv.second; best_id = kv.first; }
            // Share is over the whole superpoint (n_total), not just its thing members,
            // so thing_frac now means "this fraction of the entire blob is this thing".
            if (best_id >= 0 && (float)best_n >= p.thing_frac * (float)n_total) {
                sp.class_id = best_id;
                sp.kind = ClassKind::Thing;
                sp.confidence = (float)best_n / (float)n_total;
            }
        }

        // Mask3D-feature dispersion: normalized trace of the members' feature
        // covariance (trace(cov) / dim == mean per-dimension variance), over the
        // members that already have a feature. Two-pass (mean, then squared
        // deviations) for numerical stability; population variance (divide by n).
        sp.feat_dispersion = 0.0f;
        sp.feat_count = 0;
        if (pf) {
            const int D = pf->dim();
            std::vector<double> mean(D, 0.0);
            int n = 0;
            for (int i : sp.points) {
                const float* fv = pf->feature(i);
                if (!fv) continue;
                for (int d = 0; d < D; ++d) mean[d] += fv[d];
                ++n;
            }
            sp.feat_count = n;
            if (n >= 2) {
                const double invn = 1.0 / (double)n;
                for (int d = 0; d < D; ++d) mean[d] *= invn;
                double sumsq = 0.0;                 // sum over members & dims of dev^2
                for (int i : sp.points) {
                    const float* fv = pf->feature(i);
                    if (!fv) continue;
                    for (int d = 0; d < D; ++d) {
                        const double dv = (double)fv[d] - mean[d];
                        sumsq += dv * dv;
                    }
                }
                const double trace = sumsq * invn;  // sum_d var_d = trace(cov)
                sp.feat_dispersion = (float)(trace / (double)D);
            }
        }
    }
}
