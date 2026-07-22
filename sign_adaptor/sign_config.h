#pragma once
// Sign-text egress configuration (run.yaml `sign:` block). Kept in its own dependency-free
// header so run_config can carry it without pulling in the bridge, the detector interface,
// or (under LIFT3D_USE_SIGN) the teammate's SignUnderstanding headers.

#include <string>
#include <vector>

namespace sign {

struct SignConfig {
    bool                     enabled      = false;   // master switch (run.yaml sign.enabled)
    std::string              vlm_url      = "http://localhost:8080/v1";
    std::vector<std::string> sign_classes = {"sign"};  // vocab names treated as signs
    int                      num_workers  = 1;
    int                      queue_size   = 4;
    int                      min_sign_px  = 60;       // ignore sign masks smaller than this

    // Mock backend only (builds WITHOUT LIFT3D_USE_SIGN): stands in for the VLM so the
    // detect -> dispatch -> result -> object-annotation -> viz path can be exercised with no
    // OpenCV/CURL/VLM present. Sleeps `mock_delay_ms` on a worker thread, then answers with
    // `mock_text`, reproducing the real ~1-2 s latency the single-in-flight gate exists for.
    int                      mock_delay_ms = 1500;
    std::string              mock_text     = "MOCK: toilet -> left; exit -> right";
};

}  // namespace sign
