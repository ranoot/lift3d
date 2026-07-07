#pragma once
// PoseInterpolator: turn a stream of timestamped body->world poses into the pose at
// ANY query timestamp, by interpolating between the two bracketing samples --
// SLERP on rotation, LERP on translation. This is the fix for the image/pose
// timeline desync: the dog logs stream pose+lidar at 10 Hz but camera images at
// 5 Hz, so an image's true capture time almost never lands on a pose sample. Given
// the image timestamp, this yields the body pose AT THAT INSTANT (not the nearest
// scan's pose, which can be up to ~100 ms off while the robot is moving/turning).
//
// Source-agnostic: samples arrive via push() from either a live driver or the
// offline simulated ingestor, in increasing-time order. Bounded ring buffer, so it
// runs unbounded online without growing.
//
// Plain-array API (no Eigen in this header) to match the adaptor's PCL/Eigen-free
// public contract; the SLERP/LERP math lives in pose_interp.cpp.

#include <cstddef>
#include <cstdint>
#include <deque>

class PoseInterpolator {
public:
    // pose6 layout = {x, y, z, roll, pitch, yaw} (extrinsic-XYZ euler), matching the
    // MLOAMC scan header pose. max_samples bounds the ring (oldest dropped on push).
    explicit PoseInterpolator(std::size_t max_samples = 256);

    // Ingest one pose sample. Timestamps are expected non-decreasing; an out-of-order
    // sample (t_ns <= newest) is ignored so the buffer stays monotonic.
    void push(int64_t t_ns, const double pose6[6]);

    enum class Status {
        OK,          // t_ns bracketed by two samples (or exactly on one): T filled
        NEED_NEWER,  // t_ns is newer than the newest sample -> wait for more input
        TOO_OLD,     // t_ns is older than the oldest buffered sample -> unrecoverable
        EMPTY        // no samples yet
    };

    // Interpolate the body->world pose at t_ns into T (row-major 4x4). On anything
    // other than OK, T is left untouched.
    Status sample(int64_t t_ns, double T_world_body[16]) const;

    bool    empty()  const { return buf_.empty(); }
    int64_t oldest() const { return buf_.empty() ? 0 : buf_.front().t_ns; }
    int64_t newest() const { return buf_.empty() ? 0 : buf_.back().t_ns; }
    std::size_t size() const { return buf_.size(); }

private:
    struct Sample { int64_t t_ns; double pose6[6]; };
    std::deque<Sample> buf_;
    std::size_t        max_samples_;
};
