#pragma once
// One YAML config for the whole semantic-Universe pipeline, shared by both drivers
// (run_semantic_universe + viz_semantic_universe). Replaces the sprawling per-flag
// CLI: everything lives in a `run.yaml`, and the drivers keep only a handful of
// convenience overrides (--config, --log, --out, --spawn).

#include "semantic_pipeline.h"   // semantic::Params

#include <string>

struct RunConfig {
    std::string log;                                  // dog_logs recording folder
    std::string calib;                                // camera calib xml ("" => derive from log)
    std::string out = "sem_universe.rrd";             // viz .rrd sink (viz only)
    int         cam = 211;                            // camera sensor id
    bool        spawn = false;                        // viz: stream to a live viewer
    semantic::Params p;                               // all pipeline parameters
};

// Parse a YAML config file into a RunConfig. Missing keys fall back to the built-in
// defaults (a default-constructed semantic::Params). A leading '~' in `log`/`calib`
// is expanded to $HOME; if `calib` is empty it is derived from `log`. Throws
// std::runtime_error on a missing/unparseable file.
RunConfig loadRunConfig(const std::string& path);
