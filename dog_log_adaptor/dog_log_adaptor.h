#pragma once
// Adaptor for ~/dog_logs (Raibo quadruped EAI logs) -> lift3d point-to-pixel mapping.
//
// A log folder contains:
//   PoseAndPointClouds.bin  - MLOAMC lidar scans: per scan a 60-byte header
//                             (int64 timestamp[100ns], 6x float64 pose
//                             x,y,z,roll,pitch,yaw, int32 numPoints) followed by
//                             numPoints * 13-byte points (3x float32 xyz, uint8
//                             intensity). Little-endian, no padding. Clouds are
//                             in the robot BODY frame; the pose is body->world.
//   camera<id>.gclf         - image stream (NOT needed for the geometry; the
//                             mapping only needs K, the pose and the image size).
//
// Calibration XML (e.g. CameraCalRaiboAsQUGV113.xml) holds one <cameraCalibration>
// block per sensorId with projection (fu,fv,cu,cv), image size, and the
// imuToCamera offset+quaternion giving the camera-optical -> body transform.
//
// This adaptor indexes the scan file by frame and produces, per frame, a lift3d
// Camera (K + camera-to-world pose) plus the cloud already lifted to the world
// frame, ready to hand to projectPointsToPixels().

#include "../point_pixel_mapping/point_pixel_mapping.cuh"

#include <cstdint>
#include <string>
#include <vector>

// Pinhole + mounting calibration for one camera (one <cameraCalibration> block).
struct CameraCalib {
    int    sensor_id = 0;
    double fu = 0, fv = 0, cu = 0, cv = 0; // intrinsics
    double xi = 0;                          // omnidir param: 0 == plain pinhole
    int    width = 0, height = 0;
    double T_body_cam[16];                  // row-major 4x4, camera-optical -> body
};

// One decoded camera image, always normalised to row-major RGB uint8 (HxWx3),
// ready to hand to the inf_server (which expects HxWx3 RGB). Mono encodings are
// replicated across the three channels; MONO16 is downcast to 8-bit.
struct DogImage {
    int     w = 0, h = 0;
    int64_t t_ns = 0;                    // timestamp of the source gclf frame (ns)
    std::vector<unsigned char> rgb;      // size h*w*3, row-major RGB
    bool ok() const { return w > 0 && h > 0 && rgb.size() == (size_t)w * h * 3; }
};

// One decoded lidar scan.
struct DogFrame {
    int64_t            t_ns = 0;       // scan timestamp in nanoseconds
    std::vector<float> points_world;   // N*3, lidar cloud lifted into the world frame
    std::vector<float> intensity;      // N, lidar return intensity (0..255), aligned to points
    double             T_world_body[16]; // row-major 4x4, body -> world (from pose)
    double             T_world_cam[16];  // row-major 4x4, camera-optical -> world
};

// Parse the calibration XML and return the block matching sensor_id.
// Throws std::runtime_error if the file or sensor id is missing.
CameraCalib loadCameraCalib(const std::string& xml_path, int sensor_id);

// Build a lift3d Camera (K + camera-to-world pose) from a camera calibration and a
// body->world pose (row-major 4x4). Composes T_world_cam = T_world_body * T_body_cam
// and fills K/width/height from the calib. This is the single place the camera-optical
// -> world pose is formed, so both the file adaptor and the online DogStream (which
// feeds it an INTERPOLATED body pose at the image timestamp) agree.
Camera cameraFromBodyPose(const CameraCalib& cam, const double T_world_body[16]);

// Indexes PoseAndPointClouds.bin without loading point data up front: the
// constructor walks the scan headers once to record file offsets, then frames
// are decoded on demand.
class DogLogReader {
public:
    // bin_path: path to PoseAndPointClouds.bin. Throws on open/parse failure.
    explicit DogLogReader(const std::string& bin_path);

    int     size() const { return (int)scans_.size(); }
    int64_t timestamp(int i) const { return scans_.at(i).t_ns; }

    // Decode frame i: read the cloud, build body->world from the pose, lift the
    // cloud to world, and compose camera-to-world with the calibration.
    DogFrame frame(int i, const CameraCalib& cam) const;

    // Convenience: the lift3d Camera (K + camera-to-world pose) for frame i.
    Camera camera(const DogFrame& f, const CameraCalib& cam) const;

    // ---- push-ready raw decode (for the online DogStream / simulated ingestor) ----
    // Decode scan i WITHOUT lifting to world: the raw body-frame points and the
    // 6-DOF pose {x,y,z,roll,pitch,yaw}. Zero-return beams (x=y=z=0) are dropped;
    // intensity stays aligned to xyz_body (3 floats per point). This feeds
    // DogStream::pushScan, which does its own interpolation/lifting.
    void scanRaw(int i, double pose6[6],
                 std::vector<float>& xyz_body,
                 std::vector<unsigned char>& intensity) const;

    // ---- camera image stream (camera<id>.gclf) -------------------------------
    // Attach a gclf image stream so image()/imageAt() can decode RGB frames.
    // Walks the chunk index once (lazy per-frame decode after), mirroring the
    // scan-index idiom. Throws on open/parse failure. All dog-log access -- lidar
    // and camera -- goes through this one reader.
    void openCamera(const std::string& gclf_path);
    bool hasCamera()  const { return !imgs_.empty(); }
    int  numImages()  const { return (int)imgs_.size(); }
    // Timestamp (ns) of the k-th gclf image, in file (increasing-time) order. Lets a
    // driver merge the image and scan streams by timestamp without decoding pixels.
    int64_t imageTimestamp(int k) const { return imgs_.at(k).t_ns; }

    // Decode the gclf frame nearest in time to t_ns (or to scan i's timestamp).
    // Requires openCamera() first. Output is always HxWx3 RGB uint8.
    DogImage imageAt(int64_t t_ns) const;
    DogImage image(int i) const { return imageAt(scans_.at(i).t_ns); }
    // Decode the k-th gclf image by index (file order), no nearest-search.
    DogImage imageByIndex(int k) const;

private:
    struct ScanIndex { int64_t t_ns; uint64_t offset; int32_t num_points; };
    struct ImgIndex  { int64_t t_ns; uint64_t offset; }; // offset of the IHDR chunk
    // Decode the image whose IHDR chunk starts at file offset `ihdr_off`.
    DogImage decodeImageAt(uint64_t ihdr_off) const;
    std::string            path_;
    std::vector<ScanIndex> scans_;
    std::string            cam_path_;
    std::vector<ImgIndex>  imgs_;
};
