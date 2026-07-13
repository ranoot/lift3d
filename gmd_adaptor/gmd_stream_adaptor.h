#pragma once
// Translation layer: the robot's GMD deployment message types (Common::Entity::*,
// supplied in gmdDataTypesForInterns2) -> the lift3d online semantic pipeline entry
// points (DogStream::pushScan / pushImage). Nothing on either side is modified; this
// only converts between them.
//
// The robot's real-time stack publishes:
//   Common::Entity::PrimaryPose  -- 6-DOF body pose in the robot-origin (world) frame
//   Common::Entity::PointCloud   -- lidar points (ProcessedPoint) in the robot BODY frame
//   Common::Entity::Image        -- camera frame (RGB8 / BGR8 / MONO8 / MONO16)
//
// The pipeline consumes:
//   DogStream::pushScan(t_ns, pose6{x,y,z,r,p,y}, xyz_body*, intensity*, n)   ~10 Hz
//   DogStream::pushImage(DogImage{HxWx3 row-major RGB, t_ns})                 ~5 Hz
//
// This is the live-source analogue of DogLogIngestor (the offline replayer): instead of
// replaying a ~/dog_logs recording, the caller hands each incoming GMD message straight
// to the matching push* method here, and DogStream performs the pose interpolation +
// image/scan time alignment exactly as it does for the offline path. Wire a DogStream to
// OnlineSemantic's hook as usual, then drive it through a GmdStreamAdaptor.
//
// Conventions that must hold for the sync to be correct (they match the dog-log path):
//  * PrimaryPose is body->world (the robot's pose in the origin frame); its
//    {x,y,z,roll,pitch,yaw} map directly onto pose6, whose extrinsic-XYZ euler
//    (R = Rz(yaw)Ry(pitch)Rx(roll)) is the same convention Transformation() uses.
//  * PointCloud points are in the BODY frame; DogStream lifts them to world with the
//    paired pose. Each scan carries exactly one pose, keyed on the scan (cloud)
//    timestamp -- the same contract the MLOAMC scan format encodes.
//  * All timestamps are Common::Entity::TimeStamp (system_clock), which the robot keeps
//    time-synced across computers, so pose/cloud and image share one clock -- which is
//    exactly what DogStream needs to bracket each image between two poses.
//
// OpenCV-free: pixel conversion is hand-rolled here (honouring the encoding + row step),
// so this unit needs only the GMD entity headers + Eigen, not the GMD algorithm libs.

#include "Common/Entity/Image.h"
#include "Common/Entity/PointCloud.h"
#include "Common/Entity/PrimaryPose.h"
#include "GMDBase/Entity/TimeStamp.h"

#include "dog_log_adaptor.h"   // DogImage, CameraCalib, Camera
#include "dog_stream.h"        // DogStream

#include <cstdint>
#include <vector>

namespace gmd {

// --- stateless converters (usable on their own; exercised directly by the test) ------

// Common::Entity::TimeStamp (a system_clock time_point) -> nanoseconds since epoch, the
// int64 clock DogStream / PoseInterpolator compare on. Robust to the clock's tick period.
int64_t toNanoseconds(const Common::Entity::TimeStamp& ts);

// PrimaryPose -> pose6 {x, y, z, roll, pitch, yaw} (body->world), the layout
// PoseInterpolator / DogStream expect.
void toPose6(const Common::Entity::PrimaryPose& pose, double pose6[6]);

// Flatten a body-frame PointCloud into a packed xyz buffer (3*N floats, x0 y0 z0 x1...)
// plus an aligned intensity buffer (N bytes) -- the raw form DogStream::pushScan takes.
// No frame change is applied (DogStream does the world lift). No-return beams (x=y=z=0)
// are dropped, matching DogLogReader::scanRaw.
void toPackedCloud(const Common::Entity::PointCloud& cloud,
                   std::vector<float>& xyz_body,
                   std::vector<unsigned char>& intensity);

// Convert a GMD Image of any supported encoding (RGB8 / BGR8 / MONO8 / MONO16) to a
// row-major RGB DogImage (HxWx3 uint8), the form the inf_server expects. Mono encodings
// are replicated across channels; MONO16 is downcast to the high byte. Returns false and
// leaves `out` empty for an unsupported / NOT_APPLICABLE encoding, an empty image, or a
// data buffer too small for the declared width/height/step.
bool toDogImage(const Common::Entity::Image& img, DogImage& out);

// --- the streaming adaptor -----------------------------------------------------------

// Thin, stateful bridge over a DogStream. Construct the DogStream (with its synced-frame
// hook already wired to OnlineSemantic) elsewhere, then feed each incoming GMD message to
// the matching method here. Scratch buffers are reused across scans to avoid per-frame
// allocation. Single-threaded, mirroring DogStream (call from one ingest thread).
class GmdStreamAdaptor {
public:
    explicit GmdStreamAdaptor(DogStream& stream) : stream_(stream) {}

    // One lidar observation: the body->world pose at the scan instant + the body-frame
    // cloud captured then. Keyed on the cloud's timestamp (the scan time), so the pose is
    // registered as the robot pose at that time -- the dog-log "one pose per scan" model.
    void pushScan(const Common::Entity::PrimaryPose& pose,
                  const Common::Entity::PointCloud& cloud);

    // One camera frame. Silently dropped (with a one-line warning) if the encoding is
    // unsupported or the buffer is inconsistent with the declared size.
    void pushImage(const Common::Entity::Image& img);

    // Forward end-of-stream to the DogStream (drains any still-bracketable pending image).
    void flush() { stream_.flush(); }

    long droppedImages() const { return dropped_images_; }

private:
    DogStream&                 stream_;
    std::vector<float>         xyz_;    // reused scan scratch
    std::vector<unsigned char> inten_;  // reused scan scratch
    long                       dropped_images_ = 0;
};

} // namespace gmd
