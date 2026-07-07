// Standalone sanity check for PoseInterpolator: SLERP+LERP correctness at a
// bracketed query, plus the NEED_NEWER / TOO_OLD / EMPTY status boundaries.
//
// Build (no CMake needed):
//   g++ -std=c++20 -I dog_log_adaptor -I /usr/include/eigen3 \
//       tests/pose_interp_check.cpp dog_log_adaptor/pose_interp.cpp -o /tmp/pic && /tmp/pic

#include "pose_interp.h"

#include <cmath>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// Yaw (rotation about world Z) recovered from a row-major 4x4: atan2(R10, R00).
static double yaw_of(const double T[16]) { return std::atan2(T[4], T[0]); }

int main() {
    const double PI = 3.14159265358979323846;

    // Empty buffer.
    {
        PoseInterpolator pi;
        double T[16];
        check(pi.sample(0, T) == PoseInterpolator::Status::EMPTY, "empty -> EMPTY");
    }

    PoseInterpolator pi;
    double p0[6] = {0, 0, 0, 0, 0, 0};                 // t=0ms  : origin, yaw 0
    double p1[6] = {2, 0, 0, 0, 0, PI / 2};            // t=100ms: x=2,  yaw 90deg
    const int64_t t0 = 0, t1 = 100'000'000LL;          // 100 ms in ns
    pi.push(t0, p0);
    pi.push(t1, p1);

    double T[16];
    // Bracketed midpoint: pos ~ (1,0,0), yaw ~ 45 deg.
    check(pi.sample(50'000'000LL, T) == PoseInterpolator::Status::OK, "midpoint -> OK");
    check(std::fabs(T[3] - 1.0) < 1e-9, "midpoint x == 1.0 (LERP)");
    check(std::fabs(T[7] - 0.0) < 1e-9 && std::fabs(T[11] - 0.0) < 1e-9, "midpoint y,z == 0");
    check(std::fabs(yaw_of(T) - PI / 4) < 1e-6, "midpoint yaw == 45 deg (SLERP)");

    // Quarter point: yaw ~ 22.5 deg, x ~ 0.5.
    check(pi.sample(25'000'000LL, T) == PoseInterpolator::Status::OK, "quarter -> OK");
    check(std::fabs(T[3] - 0.5) < 1e-9, "quarter x == 0.5");
    check(std::fabs(yaw_of(T) - PI / 8) < 1e-6, "quarter yaw == 22.5 deg");

    // Exact hits on the samples.
    check(pi.sample(t0, T) == PoseInterpolator::Status::OK && std::fabs(yaw_of(T)) < 1e-9,
          "exact t0 -> yaw 0");
    check(pi.sample(t1, T) == PoseInterpolator::Status::OK &&
          std::fabs(yaw_of(T) - PI / 2) < 1e-6, "exact t1 -> yaw 90");

    // Boundaries: newer than newest waits; older than oldest is unrecoverable.
    check(pi.sample(t1 + 1, T) == PoseInterpolator::Status::NEED_NEWER, "beyond newest -> NEED_NEWER");
    check(pi.sample(t0 - 1, T) == PoseInterpolator::Status::TOO_OLD, "before oldest -> TOO_OLD");

    // Ring bound: pushing past capacity drops the oldest, so old queries go TOO_OLD.
    PoseInterpolator small(3);
    double q[6] = {0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 6; ++k) { q[0] = k; small.push(k * 10'000'000LL, q); }
    check(small.size() == 3, "ring capped at 3 samples");
    check(small.sample(0, T) == PoseInterpolator::Status::TOO_OLD, "evicted sample -> TOO_OLD");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
