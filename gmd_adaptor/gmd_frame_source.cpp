#include "gmd_frame_source.h"

#include <chrono>
#include <cstdio>

namespace gmd {

int64_t toNanoseconds(const Common::Entity::TimeStamp& ts) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               ts.time_since_epoch())
        .count();
}

void toPose6(const Common::Entity::PrimaryPose& pose, double pose6[6]) {
    // PrimaryPose is body->world in the robot-origin frame; {x,y,z,roll,pitch,yaw} map
    // straight onto pose6 (extrinsic-XYZ euler, matching euler_pose.h / Transformation()).
    pose6[0] = pose.x;
    pose6[1] = pose.y;
    pose6[2] = static_cast<double>(pose.z);
    pose6[3] = static_cast<double>(pose.roll);
    pose6[4] = static_cast<double>(pose.pitch);
    pose6[5] = static_cast<double>(pose.yaw);
}

void toPackedCloud(const Common::Entity::PointCloud& cloud,
                   std::vector<float>& xyz_body,
                   std::vector<unsigned char>& intensity) {
    const std::size_t n = cloud.pointCloudData.size();
    xyz_body.clear();
    intensity.clear();
    xyz_body.reserve(n * 3);
    intensity.reserve(n);
    for (const Common::Entity::ProcessedPoint& pt : cloud.pointCloudData) {
        // Drop no-return beams (x=y=z=0), matching DogLogReader::scanRaw so the world
        // lift + downstream geometry see the same cloud the offline path would.
        if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) continue;
        xyz_body.push_back(pt.x);
        xyz_body.push_back(pt.y);
        xyz_body.push_back(pt.z);
        intensity.push_back(pt.intensity);
    }
}

namespace {
// Bytes per pixel for a GMD Image encoding (0 => unsupported / NOT_APPLICABLE).
int bytesPerPixel(ImageEncoding enc) {
    switch (enc) {
        case ImageEncoding::RGB8:
        case ImageEncoding::BGR8:   return 3;
        case ImageEncoding::MONO8:  return 1;
        case ImageEncoding::MONO16: return 2;
        default:                    return 0;
    }
}
} // namespace

bool toDogImage(const Common::Entity::Image& img, DogImage& out) {
    out = DogImage{};

    const int W = img.width;
    const int H = img.height;
    if (W <= 0 || H <= 0) return false;

    const int bpp = bytesPerPixel(img.encoding);
    if (bpp == 0) return false;                          // unsupported / NOT_APPLICABLE

    // Row stride in BYTES. Fall back to the tightly-packed stride when step is unset.
    const int step = img.step > 0 ? img.step : W * bpp;
    if (step < W * bpp) return false;                    // step too small for the row
    if (img.data.size() < static_cast<std::size_t>(step) * H) return false; // truncated

    out.w = W;
    out.h = H;
    out.t_ns = toNanoseconds(img.timestamp);
    out.rgb.resize(static_cast<std::size_t>(W) * H * 3);

    for (int r = 0; r < H; ++r) {
        const unsigned char* src = img.data.data() + static_cast<std::size_t>(r) * step;
        unsigned char* dst = out.rgb.data() + static_cast<std::size_t>(r) * W * 3;
        switch (img.encoding) {
            case ImageEncoding::RGB8:
                for (int c = 0; c < W; ++c) {
                    dst[c * 3 + 0] = src[c * 3 + 0];
                    dst[c * 3 + 1] = src[c * 3 + 1];
                    dst[c * 3 + 2] = src[c * 3 + 2];
                }
                break;
            case ImageEncoding::BGR8:
                for (int c = 0; c < W; ++c) {
                    dst[c * 3 + 0] = src[c * 3 + 2]; // R <- B position
                    dst[c * 3 + 1] = src[c * 3 + 1]; // G
                    dst[c * 3 + 2] = src[c * 3 + 0]; // B <- R position
                }
                break;
            case ImageEncoding::MONO8:
                for (int c = 0; c < W; ++c) {
                    const unsigned char g = src[c];
                    dst[c * 3 + 0] = g;
                    dst[c * 3 + 1] = g;
                    dst[c * 3 + 2] = g;
                }
                break;
            case ImageEncoding::MONO16:
                for (int c = 0; c < W; ++c) {
                    // Little-endian 16-bit sample -> high byte (downcast to 8-bit).
                    const unsigned v = static_cast<unsigned>(src[c * 2 + 0]) |
                                       (static_cast<unsigned>(src[c * 2 + 1]) << 8);
                    const unsigned char g = static_cast<unsigned char>(v >> 8);
                    dst[c * 3 + 0] = g;
                    dst[c * 3 + 1] = g;
                    dst[c * 3 + 2] = g;
                }
                break;
            default:
                return false; // unreachable (bpp check above), but keeps the switch total
        }
    }
    return out.ok();
}

void GmdFrameSource::pushScan(const Common::Entity::PrimaryPose& pose,
                              const Common::Entity::PointCloud& cloud) {
    double pose6[6];
    toPose6(pose, pose6);
    toPackedCloud(cloud, xyz_, inten_);
    const int n = static_cast<int>(xyz_.size() / 3);
    // Key on the cloud (scan) timestamp: DogStream uses this single t_ns both to register
    // the pose in the interpolator and to time-stamp the buffered scan (one pose per scan).
    engine().pushScan(toNanoseconds(cloud.timestamp), pose6, xyz_.data(), inten_.data(), n);
}

void GmdFrameSource::pushImage(const Common::Entity::Image& img) {
    DogImage di;
    if (!toDogImage(img, di)) {
        ++dropped_images_;
        std::fprintf(stderr,
                     "[gmd_adaptor] WARN: dropped image (encoding %d, %dx%d, step %d, "
                     "%zu bytes)\n",
                     static_cast<int>(img.encoding), img.width, img.height, img.step,
                     img.data.size());
        return;
    }
    engine().pushImage(di);
}

} // namespace gmd
