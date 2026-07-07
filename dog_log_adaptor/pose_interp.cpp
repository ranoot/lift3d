#include "pose_interp.h"
#include "euler_pose.h"

#include <algorithm>

// SLERP rotation + LERP translation between the two samples bracketing the query
// time. Euler poses are converted to quaternions so the rotational interpolation is
// geodesic (constant angular velocity) rather than the wobble a naive per-euler-angle
// lerp would give on fast turns.

PoseInterpolator::PoseInterpolator(std::size_t max_samples)
    : max_samples_(max_samples < 2 ? 2 : max_samples) {}

void PoseInterpolator::push(int64_t t_ns, const double pose6[6]) {
    if (!buf_.empty() && t_ns <= buf_.back().t_ns) return;   // keep monotonic
    Sample s;
    s.t_ns = t_ns;
    std::copy(pose6, pose6 + 6, s.pose6);
    buf_.push_back(s);
    while (buf_.size() > max_samples_) buf_.pop_front();
}

PoseInterpolator::Status
PoseInterpolator::sample(int64_t t_ns, double T_world_body[16]) const {
    if (buf_.empty())            return Status::EMPTY;
    if (t_ns < buf_.front().t_ns) return Status::TOO_OLD;
    if (t_ns > buf_.back().t_ns)  return Status::NEED_NEWER;

    // Find the first sample with t_ns_i >= query, giving the bracketing pair
    // [prev, hi]. buf_ is sorted ascending, so a binary search locates it.
    auto hi = std::lower_bound(buf_.begin(), buf_.end(), t_ns,
                               [](const Sample& s, int64_t t) { return s.t_ns < t; });

    // Exact hit on a sample (including the front): no interpolation needed.
    if (hi->t_ns == t_ns) {
        doglog::as_mat4(T_world_body) = doglog::iso_from_pose6(hi->pose6).matrix();
        return Status::OK;
    }

    const Sample& b1 = *hi;          // t1 > t_ns
    const Sample& b0 = *(hi - 1);    // t0 < t_ns  (hi != begin(), since t_ns > front)

    const double span = double(b1.t_ns - b0.t_ns);
    const double alpha = span > 0.0 ? double(t_ns - b0.t_ns) / span : 0.0;

    // Rotation: SLERP the two body->world orientations.
    Eigen::Quaterniond q0(doglog::euler_xyz_extrinsic(b0.pose6[3], b0.pose6[4], b0.pose6[5]));
    Eigen::Quaterniond q1(doglog::euler_xyz_extrinsic(b1.pose6[3], b1.pose6[4], b1.pose6[5]));
    Eigen::Quaterniond q = q0.slerp(alpha, q1);

    // Translation: straight LERP.
    Eigen::Vector3d p0(b0.pose6[0], b0.pose6[1], b0.pose6[2]);
    Eigen::Vector3d p1(b1.pose6[0], b1.pose6[1], b1.pose6[2]);
    Eigen::Vector3d p = p0 + alpha * (p1 - p0);

    doglog::Iso3 T = doglog::Iso3::Identity();
    T.linear()      = q.toRotationMatrix();
    T.translation() = p;
    doglog::as_mat4(T_world_body) = T.matrix();
    return Status::OK;
}
