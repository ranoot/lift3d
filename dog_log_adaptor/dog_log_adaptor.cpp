#include "dog_log_adaptor.h"
#include "euler_pose.h"        // shared Eigen pose helpers (as_mat4, euler_xyz_extrinsic, ...)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <iostream>

// NOTE: assumes a little-endian host (x86/ARM as used here). The log format is
// little-endian, so fields are memcpy'd straight out of the byte stream.
//
// Pose math goes through Eigen's Geometry module (via euler_pose.h). The public
// structs stay plain row-major arrays (no Eigen in the headers), so the CUDA/PCL-free
// contract of point_pixel_mapping.cuh is untouched; Eigen is an internal detail here.
// We map those arrays with Eigen::Map<row-major> to read/write without manual loops.

using doglog::Mat4;
using doglog::Iso3;
using doglog::as_mat4;
using doglog::euler_xyz_extrinsic;

namespace {

// ---- tiny XML scalar extraction ---------------------------------------------
// Returns the text between the first <tag>...</tag> found at/after `from`.
// Good enough for this flat, regular calibration format (no attributes/CDATA).
std::string extract_tag(const std::string& s, const std::string& tag, size_t from = 0) {
    std::string open = "<" + tag + ">", close = "</" + tag + ">";
    size_t a = s.find(open, from);
    if (a == std::string::npos) return {};
    a += open.size();
    size_t b = s.find(close, a);
    if (b == std::string::npos) return {};
    std::string v = s.substr(a, b - a);
    size_t i = v.find_first_not_of(" \t\r\n");
    size_t j = v.find_last_not_of(" \t\r\n");
    return (i == std::string::npos) ? std::string{} : v.substr(i, j - i + 1);
}

double tag_d(const std::string& s, const std::string& tag) {
    std::string v = extract_tag(s, tag);
    return v.empty() ? 0.0 : std::stod(v);
}

} // namespace

// ---- calibration -------------------------------------------------------------
CameraCalib loadCameraCalib(const std::string& xml_path, int sensor_id) {
    std::ifstream f(xml_path);
    if (!f) throw std::runtime_error("cannot open calib xml: " + xml_path);
    std::string xml((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());

    const std::string open = "<cameraCalibration>", close = "</cameraCalibration>";
    size_t pos = 0;
    while (true) {
        size_t a = xml.find(open, pos);
        if (a == std::string::npos) break;
        size_t b = xml.find(close, a);
        if (b == std::string::npos) break;
        std::string block = xml.substr(a, b - a);
        pos = b + close.size();

        if ((int)tag_d(block, "sensorId") != sensor_id) continue;

        CameraCalib c;
        c.sensor_id = sensor_id;
        c.fu = tag_d(block, "fu");
        c.fv = tag_d(block, "fv");
        c.cu = tag_d(block, "cu");
        c.cv = tag_d(block, "cv");
        c.xi = tag_d(block, "xi");
        c.width  = (int)tag_d(block, "imageWidth");
        c.height = (int)tag_d(block, "imageHeight");

        // imuToCamera offset + quaternion -> camera-optical -> body rigid transform.
        // Eigen::Quaterniond is w-first Hamilton, matching the log's (qw,qx,qy,qz).
        Eigen::Quaterniond q(tag_d(block, "imuToCameraQw"), tag_d(block, "imuToCameraQx"),
                             tag_d(block, "imuToCameraQy"), tag_d(block, "imuToCameraQz"));
        q.normalize();

        Iso3 T_body_cam = Iso3::Identity();
        T_body_cam.linear()      = q.toRotationMatrix();
        T_body_cam.translation() = Eigen::Vector3d(tag_d(block, "imuToCameraOffsetX"),
                                                   tag_d(block, "imuToCameraOffsetY"),
                                                   tag_d(block, "imuToCameraOffsetZ"));
        as_mat4(c.T_body_cam) = T_body_cam.matrix();

        if (std::fabs(c.xi) > 1e-9)
            std::cerr << "[dog_log_adaptor] WARNING: sensor " << sensor_id
                      << " has xi=" << c.xi << " (omnidirectional/fisheye). The "
                      << "lift3d mapping is pinhole-only; use a pinhole cam "
                      << "(e.g. 111/211) or add an omnidir model.\n";
        return c;
    }
    throw std::runtime_error("sensorId " + std::to_string(sensor_id) +
                             " not found in " + xml_path);
}

// ---- scan file ---------------------------------------------------------------
namespace {
constexpr int    HEADER_SIZE = 60; // int64 + 6*float64 + int32
constexpr int    POINT_SIZE  = 13; // 3*float32 + uint8
} // namespace

// Compose T_world_cam = T_world_body * T_body_cam and fill K / size from the calib.
// Shared by DogLogReader::camera() and the online DogStream.
Camera cameraFromBodyPose(const CameraCalib& cam, const double T_world_body[16]) {
    Iso3 T_world_body_iso(as_mat4(T_world_body));
    Iso3 T_body_cam(as_mat4(cam.T_body_cam));
    Iso3 T_world_cam = T_world_body_iso * T_body_cam;

    Camera c{};
    c.K[0] = (float)cam.fu; c.K[1] = 0;             c.K[2] = (float)cam.cu;
    c.K[3] = 0;             c.K[4] = (float)cam.fv; c.K[5] = (float)cam.cv;
    c.K[6] = 0;             c.K[7] = 0;             c.K[8] = 1;
    Mat4 M = T_world_cam.matrix();
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 4; ++col) c.T[r * 4 + col] = (float)M(r, col);
    c.width  = cam.width;
    c.height = cam.height;
    return c;
}

DogLogReader::DogLogReader(const std::string& bin_path) : path_(bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open scan file: " + bin_path);

    char hdr[HEADER_SIZE];
    while (true) {
        uint64_t off = (uint64_t)f.tellg();
        f.read(hdr, HEADER_SIZE);
        if (f.gcount() < HEADER_SIZE) break;

        int64_t t100ns;  int32_t num;
        std::memcpy(&t100ns, hdr + 0, 8);
        std::memcpy(&num,    hdr + 56, 4);

        scans_.push_back({ t100ns * 100, off, num });
        f.seekg((std::streamoff)num * POINT_SIZE, std::ios::cur);
    }
}

DogFrame DogLogReader::frame(int i, const CameraCalib& cam) const {
    const ScanIndex& s = scans_.at(i);

    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot reopen scan file: " + path_);
    f.seekg((std::streamoff)s.offset, std::ios::beg);

    char hdr[HEADER_SIZE];
    f.read(hdr, HEADER_SIZE);
    if (f.gcount() < HEADER_SIZE) throw std::runtime_error("short header at frame");

    double pose[6]; // x,y,z,roll,pitch,yaw
    std::memcpy(pose, hdr + 8, 48);

    std::vector<char> buf((size_t)s.num_points * POINT_SIZE);
    f.read(buf.data(), buf.size());

    // body->world from the pose, then compose camera-optical->world.
    Iso3 T_world_body = Iso3::Identity();
    T_world_body.linear()      = euler_xyz_extrinsic(pose[3], pose[4], pose[5]);
    T_world_body.translation() = Eigen::Vector3d(pose[0], pose[1], pose[2]);

    Iso3 T_body_cam(as_mat4(cam.T_body_cam));
    Iso3 T_world_cam = T_world_body * T_body_cam;

    DogFrame out;
    out.t_ns = s.t_ns;
    as_mat4(out.T_world_body) = T_world_body.matrix();
    as_mat4(out.T_world_cam)  = T_world_cam.matrix();

    // lift each point into the world frame, dropping zero-return beams (x=y=z=0).
    // The 13th byte of each point is the uint8 intensity, kept aligned to points.
    out.points_world.reserve((size_t)s.num_points * 3);
    out.intensity.reserve((size_t)s.num_points);
    for (int p = 0; p < s.num_points; ++p) {
        float xyz[3];
        std::memcpy(xyz, buf.data() + (size_t)p * POINT_SIZE, 12);
        if (xyz[0] == 0.0f && xyz[1] == 0.0f && xyz[2] == 0.0f) continue;
        Eigen::Vector3d pw = T_world_body * Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
        out.points_world.push_back((float)pw.x());
        out.points_world.push_back((float)pw.y());
        out.points_world.push_back((float)pw.z());
        unsigned char intensity = (unsigned char)buf[(size_t)p * POINT_SIZE + 12];
        out.intensity.push_back((float)intensity);
    }
    return out;
}

Camera DogLogReader::camera(const DogFrame& f, const CameraCalib& cam) const {
    return cameraFromBodyPose(cam, f.T_world_body);
}

void DogLogReader::scanRaw(int i, double pose6[6],
                           std::vector<float>& xyz_body,
                           std::vector<unsigned char>& intensity) const {
    const ScanIndex& s = scans_.at(i);

    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot reopen scan file: " + path_);
    f.seekg((std::streamoff)s.offset, std::ios::beg);

    char hdr[HEADER_SIZE];
    f.read(hdr, HEADER_SIZE);
    if (f.gcount() < HEADER_SIZE) throw std::runtime_error("short header at scanRaw");
    std::memcpy(pose6, hdr + 8, 48);

    std::vector<char> buf((size_t)s.num_points * POINT_SIZE);
    f.read(buf.data(), buf.size());

    xyz_body.clear();  intensity.clear();
    xyz_body.reserve((size_t)s.num_points * 3);
    intensity.reserve((size_t)s.num_points);
    for (int p = 0; p < s.num_points; ++p) {
        float xyz[3];
        std::memcpy(xyz, buf.data() + (size_t)p * POINT_SIZE, 12);
        if (xyz[0] == 0.0f && xyz[1] == 0.0f && xyz[2] == 0.0f) continue;   // no-return beam
        xyz_body.push_back(xyz[0]);
        xyz_body.push_back(xyz[1]);
        xyz_body.push_back(xyz[2]);
        intensity.push_back((unsigned char)buf[(size_t)p * POINT_SIZE + 12]);
    }
}

// ---- camera image stream (camera<id>.gclf) ----------------------------------
// The .gclf container is a chunk stream of RAW (uncompressed) pixels -- no codec.
// Each chunk: 8-byte header (4-byte ASCII type + uint32 LE payload size), then the
// payload. Chunk types on disk are ASCII "GCLF" (file header), "IHDR" (image
// header) and "IDAT" (image data) -- these are the Python reader's hex magics with
// the type bytes reversed. IHDR fixed fields (LE) are, in order: int64 timestamp(ns),
// uint32 sensorId, height, width, encoding, uint64 seq, int32 exposureTime, gain,
// uint32 attributeCount (then variable attributes). encoding: 1=RGB8, 2=MONO8,
// 4=MONO16. Reference: ~/dog_logs/pygenerictools/pygenericdata/camera/reader.py.
namespace {
constexpr int GCLF_CHUNK_HDR = 8;

// Read a chunk header: 4-byte ASCII type + uint32 LE payload size. false at EOF.
bool read_chunk_header(std::istream& f, char type[4], uint32_t& size) {
    char h[GCLF_CHUNK_HDR];
    f.read(h, GCLF_CHUNK_HDR);
    if (f.gcount() < GCLF_CHUNK_HDR) return false;
    std::memcpy(type, h, 4);
    std::memcpy(&size, h + 4, 4);
    return true;
}
bool type_is(const char t[4], const char* s) { return std::memcmp(t, s, 4) == 0; }
} // namespace

void DogLogReader::openCamera(const std::string& gclf_path) {
    std::ifstream f(gclf_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open gclf: " + gclf_path);
    cam_path_ = gclf_path;
    imgs_.clear();

    char type[4]; uint32_t size;
    if (!read_chunk_header(f, type, size) || !type_is(type, "GCLF"))
        throw std::runtime_error("bad gclf file header: " + gclf_path);
    f.seekg(size, std::ios::cur);  // skip file-header payload (segment id + index pos)

    // Walk IHDR/IDAT pairs; record each image's timestamp + IHDR chunk offset.
    while (true) {
        uint64_t ihdr_off = (uint64_t)f.tellg();
        if (!read_chunk_header(f, type, size)) break;  // clean EOF
        if (!type_is(type, "IHDR"))
            throw std::runtime_error("expected IHDR chunk in " + gclf_path);
        if (size < 8) throw std::runtime_error("short IHDR in " + gclf_path);
        std::vector<char> payload(size);
        f.read(payload.data(), size);
        if (f.gcount() < (std::streamsize)size) break;
        int64_t ts; std::memcpy(&ts, payload.data(), 8);
        imgs_.push_back({ ts, ihdr_off });

        char t2[4]; uint32_t s2;                        // following IDAT
        if (!read_chunk_header(f, t2, s2)) break;
        if (!type_is(t2, "IDAT"))
            throw std::runtime_error("expected IDAT chunk in " + gclf_path);
        f.seekg((std::streamoff)s2, std::ios::cur);     // skip pixel payload
    }
    if (imgs_.empty()) throw std::runtime_error("no images in gclf: " + gclf_path);
}

DogImage DogLogReader::imageAt(int64_t t_ns) const {
    if (imgs_.empty()) throw std::runtime_error("openCamera() not called before imageAt");

    // Nearest timestamp (imgs_ are in file order == increasing time).
    size_t best = 0;
    int64_t best_d = std::llabs(imgs_[0].t_ns - t_ns);
    for (size_t k = 1; k < imgs_.size(); ++k) {
        int64_t d = std::llabs(imgs_[k].t_ns - t_ns);
        if (d < best_d) { best_d = d; best = k; }
    }
    return decodeImageAt(imgs_[best].offset);
}

DogImage DogLogReader::imageByIndex(int k) const {
    return decodeImageAt(imgs_.at(k).offset);
}

DogImage DogLogReader::decodeImageAt(uint64_t ihdr_off) const {
    if (imgs_.empty()) throw std::runtime_error("openCamera() not called before decode");

    std::ifstream f(cam_path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot reopen gclf: " + cam_path_);
    f.seekg((std::streamoff)ihdr_off, std::ios::beg);

    char type[4]; uint32_t hsize;
    if (!read_chunk_header(f, type, hsize) || !type_is(type, "IHDR"))
        throw std::runtime_error("IHDR reread failed in " + cam_path_);
    std::vector<char> hdr(hsize);
    f.read(hdr.data(), hsize);
    int64_t  ts;      std::memcpy(&ts,       hdr.data() + 0,  8);
    uint32_t height;  std::memcpy(&height,   hdr.data() + 12, 4);
    uint32_t width;   std::memcpy(&width,    hdr.data() + 16, 4);
    uint32_t encoding;std::memcpy(&encoding, hdr.data() + 20, 4);

    char t2[4]; uint32_t dsize;
    if (!read_chunk_header(f, t2, dsize) || !type_is(t2, "IDAT"))
        throw std::runtime_error("IDAT reread failed in " + cam_path_);
    std::vector<unsigned char> data(dsize);
    f.read(reinterpret_cast<char*>(data.data()), dsize);
    if (f.gcount() < (std::streamsize)dsize)
        throw std::runtime_error("short IDAT in " + cam_path_);

    DogImage img;
    img.w = (int)width; img.h = (int)height; img.t_ns = ts;
    const size_t npx = (size_t)width * height;
    img.rgb.resize(npx * 3);
    if (encoding == 1) {                              // RGB8
        if (dsize != npx * 3) throw std::runtime_error("RGB8 size mismatch in " + cam_path_);
        std::memcpy(img.rgb.data(), data.data(), npx * 3);
    } else if (encoding == 2) {                       // MONO8 -> replicate to 3ch
        if (dsize != npx) throw std::runtime_error("MONO8 size mismatch in " + cam_path_);
        for (size_t p = 0; p < npx; ++p) {
            unsigned char v = data[p];
            img.rgb[3 * p] = img.rgb[3 * p + 1] = img.rgb[3 * p + 2] = v;
        }
    } else if (encoding == 4) {                       // MONO16 LE -> high byte, 3ch
        if (dsize != npx * 2) throw std::runtime_error("MONO16 size mismatch in " + cam_path_);
        for (size_t p = 0; p < npx; ++p) {
            unsigned char v = data[2 * p + 1];        // high byte of the LE u16
            img.rgb[3 * p] = img.rgb[3 * p + 1] = img.rgb[3 * p + 2] = v;
        }
    } else {
        throw std::runtime_error("unsupported gclf encoding " + std::to_string(encoding));
    }
    return img;
}
