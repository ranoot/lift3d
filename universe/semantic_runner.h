#pragma once
// SemanticRunner: the simplified, self-contained entry point to the whole online 2D->3D
// semantic mapping pipeline. It replaces the hook-wired, long-running run_semantic_universe
// driver with a single blocking call the robot's own code invokes once per already-synced
// frame:
//
//     semantic::SemanticRunner runner("run.yaml");
//     for (;;) {
//         std::optional<OpenVocabSegInput> in = robotGetSynchronizedInputs();  // robot side
//         std::vector<Common::Entity::EAIRoomObject> objects = runner.runOnInput(in);
//         publish(objects);
//     }
//
// One SemanticRunner owns EVERY stage (Universe / InfClient / Superpoints / ObjectSeeds /
// Objects, plus the ZMQ viz publisher and the EAIRoomObject egress) behind a PImpl, so the
// caller never sees any pipeline internals, PCL, or CUDA -- only the GMD entity types.
//
// runOnInput is blocking (the plan explicitly does not care): it integrates the frame,
// runs 2D inference, projects + votes, runs the discovery stack, publishes a viz frame over
// ZMQ (the Python viz_server persists the .rrd via `--out`), logs to stderr/stdout, and
// returns the FULL current object list as EAIRoomObjects. Inputs are assumed already
// time-synced + pose-interpolated (that is the robot system's getSynchronizedInputs job),
// so no DogStream sync engine is involved here.

#include <memory>
#include <optional>
#include <string>
#include <vector>                                   // MUST precede Object.h: that vendor header
                                                    // uses std::vector without including <vector>.
#include "Common/Entity/Image.h"
#include "Common/Entity/PointCloud.h"
#include "Common/Entity/PrimaryPose.h"
#include "Common/Entity/Object.h"                   // Common::Entity::EAIRoomObject (output type)

namespace semantic {

// One already-time-synced frame. `images` must contain EXACTLY one image (the pose is
// interpolated to that image's timestamp, and the point cloud is the body-frame scan aligned
// to it). All fields are the robot's native GMD deployment types.
struct OpenVocabSegInput {
    std::vector<Common::Entity::Image> images;      // exactly 1 (checked in runOnInput)
    Common::Entity::PointCloud         pointCloud;  // BODY frame (lifted to world internally)
    Common::Entity::PrimaryPose        pose;        // body->world at the image timestamp
};

class SemanticRunner {
public:
    // Load `config_path` (run.yaml), build every stage, and prime the pipeline (begin()).
    // Pings the inf_server (warns, does not throw, if it is down). Throws std::runtime_error
    // on a missing/unparseable config or camera calibration.
    explicit SemanticRunner(const std::string& config_path = "run.yaml");
    ~SemanticRunner();

    SemanticRunner(const SemanticRunner&)            = delete;  // owns non-copyable stages
    SemanticRunner& operator=(const SemanticRunner&) = delete;

    // Process one synced frame and return the FULL current object list. A null optional is a
    // no-op that returns {} (the caller's convention for "no new input this tick"). Throws
    // std::invalid_argument if images.size() != 1 or the image encoding is unsupported.
    std::vector<Common::Entity::EAIRoomObject>
        runOnInput(std::optional<OpenVocabSegInput> input);

    // Functor sugar: runner(input) == runner.runOnInput(input).
    std::vector<Common::Entity::EAIRoomObject>
        operator()(std::optional<OpenVocabSegInput> input) {
        return runOnInput(std::move(input));
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace semantic
