#pragma once
// Internal (Eigen-backed) pose helpers shared by the dog-log adaptor translation
// units -- dog_log_adaptor.cpp and pose_interp.cpp. This header pulls in Eigen, so
// it is included ONLY from .cpp files, never from a public adaptor header: the
// public structs stay plain row-major double[16] arrays (see dog_log_adaptor.h),
// keeping the CUDA/PCL-free contract of point_pixel_mapping.cuh intact.

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace doglog {

using Mat4 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Iso3 = Eigen::Transform<double, 3, Eigen::Isometry>;

// View a row-major double[16] as a 4x4 Eigen matrix (no copy).
inline Eigen::Map<Mat4>       as_mat4(double M[16])       { return Eigen::Map<Mat4>(M); }
inline Eigen::Map<const Mat4> as_mat4(const double M[16]) { return Eigen::Map<const Mat4>(M); }

// Extrinsic XYZ euler -> rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll). Matches
// pytransform3d matrix_from_euler([roll,pitch,yaw], 0,1,2, extrinsic=True), the
// convention used by readLidarBinaryFormat.ScanFile and the log's pose fields.
inline Eigen::Matrix3d euler_xyz_extrinsic(double roll, double pitch, double yaw) {
    return (Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ())
          * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
}

// pose6 = {x,y,z,roll,pitch,yaw} -> body->world rigid transform.
inline Iso3 iso_from_pose6(const double pose6[6]) {
    Iso3 T = Iso3::Identity();
    T.linear()      = euler_xyz_extrinsic(pose6[3], pose6[4], pose6[5]);
    T.translation() = Eigen::Vector3d(pose6[0], pose6[1], pose6[2]);
    return T;
}

} // namespace doglog
