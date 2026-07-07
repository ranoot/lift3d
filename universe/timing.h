#pragma once
// Lightweight wall-clock instrumentation for the semantic accumulate loop: a
// per-frame progress line plus an end-of-run per-phase bottleneck report. All
// output goes to stderr (prefixed "[timing]") so it never mixes with the data /
// tally that the drivers write to stdout.

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace semantic {

using Clock = std::chrono::steady_clock;

inline double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// Reusable stopwatch: lap() returns ms since the last lap()/reset() and rearms.
struct Stopwatch {
    Clock::time_point t0 = Clock::now();
    void   reset()  { t0 = Clock::now(); }
    double lap()    { auto n = Clock::now(); double e = ms(n - t0); t0 = n; return e; }
    double elapsed() const { return ms(Clock::now() - t0); }
};

// Sub-phase breakdown of a single stepFrame() call (all in ms).
struct StepTiming {
    double read_integrate = 0;  // decode frame + build cloud + integrate geometry
    double infer          = 0;  // fetch image + inf_server 2D panoptic inference
    double project        = 0;  // point-to-pixel projection / z-buffer (GPU when CUDA)
    double vote           = 0;  // per-visible-point colour + label voting
    double total() const { return read_integrate + infer + project + vote; }
};

// Insertion-ordered accumulator of named phase totals + call counts for the
// end-of-run report.
struct Profile {
    struct Acc { double total_ms = 0; long calls = 0; };
    std::vector<std::pair<std::string, Acc>> order;

    Acc& slot(const std::string& name) {
        for (auto& kv : order) if (kv.first == name) return kv.second;
        order.push_back({name, {}});
        return order.back().second;
    }
    void add(const std::string& name, double ms_) {
        Acc& a = slot(name);
        a.total_ms += ms_;
        ++a.calls;
    }

    // Per-phase total / calls / mean / share of wall time, then the profiled sum.
    void report(const char* title, double wall_ms) const {
        std::fprintf(stderr, "[timing] ===== %s (%.2f s wall) =====\n",
                     title, wall_ms / 1000.0);
        double sum = 0;
        for (const auto& kv : order) {
            const Acc& a = kv.second;
            sum += a.total_ms;
            const double mean = a.calls ? a.total_ms / a.calls : 0.0;
            const double pct  = wall_ms > 0 ? 100.0 * a.total_ms / wall_ms : 0.0;
            std::fprintf(stderr,
                         "[timing]   %-16s %10.1f ms  %6ld calls  %8.2f ms/call  %5.1f%%\n",
                         kv.first.c_str(), a.total_ms, a.calls, mean, pct);
        }
        const double other = wall_ms - sum;
        std::fprintf(stderr, "[timing]   %-16s %10.1f ms  %6s  %8s  %5.1f%%\n",
                     "profiled", sum, "", "", wall_ms > 0 ? 100.0 * sum / wall_ms : 0.0);
        std::fprintf(stderr, "[timing]   %-16s %10.1f ms  %6s  %8s  %5.1f%%\n",
                     "unprofiled", other, "", "", wall_ms > 0 ? 100.0 * other / wall_ms : 0.0);
    }
};

} // namespace semantic
