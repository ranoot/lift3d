#include "PoseUtils.h"

namespace Common
{
namespace Algorithm
{
// Converts roll-pitch-yaw (in radians) to normalized quaternion
Eigen::Quaterniond rollPitchYawToQuaternion(double aRoll, double aPitch, double aYaw)
{
	const double s1 = sin(aYaw / 2.0);
	const double c1 = cos(aYaw / 2.0);
	const double s2 = sin(aPitch / 2.0);
	const double c2 = cos(aPitch / 2.0);
	const double s3 = sin(aRoll / 2.0);
	const double c3 = cos(aRoll / 2.0);
	return Eigen::Quaterniond(s1 * s2 * s3 + c1 * c2 * c3, c1 * c2 * s3 - s1 * s2 * c3, s1 * c2 * s3 + c1 * s2 * c3,
		s1 * c2 * c3 - c1 * s2 * s3)
		.normalized();
}

Eigen::Quaterniond rollPitchYawToQuaternion(const Eigen::Vector3d& aRollPitchYaw)
{
	return rollPitchYawToQuaternion(aRollPitchYaw.x(), aRollPitchYaw.y(), aRollPitchYaw.z());
}

Eigen::Matrix<double, 4, 3> quaternionJacobian(double aRoll, double aPitch, double aYaw)
{
	const double s1 = sin(aYaw / 2.0);
	const double c1 = cos(aYaw / 2.0);
	const double s2 = sin(aPitch / 2.0);
	const double c2 = cos(aPitch / 2.0);
	const double s3 = sin(aRoll / 2.0);
	const double c3 = cos(aRoll / 2.0);
	Eigen::Matrix<double, 4, 3> J;
	J << c1 * c2 * c3 + s1 * s2 * s3, -c1 * s2 * s3 - s1 * c2 * c3, -s1 * c2 * s3 - c1 * s2 * c3,
		s1 * c2 * c3 - c1 * s2 * s3, -s1 * s2 * s3 + c1 * c2 * c3, c1 * c2 * s3 - s1 * s2 * c3,
		-s1 * c2 * s3 - c1 * s2 * c3, -s1 * s2 * c3 - c1 * c2 * s3, c1 * c2 * c3 + s1 * s2 * s3,
		s1 * s2 * c3 - c1 * c2 * s3, s1 * c2 * s3 - c1 * s2 * c3, c1 * s2 * s3 - s1 * c2 * c3;
	return 0.5 * J;
}

Eigen::Matrix<double, 4, 3> quaternionJacobian(const Eigen::Vector3d& aRollPitchYaw)
{
	return quaternionJacobian(aRollPitchYaw.x(), aRollPitchYaw.y(), aRollPitchYaw.z());
}

// Converts roll-pitch-yaw covariance to quaternion covariance
Eigen::Matrix<double, 4, 4> rollPitchYawCovToQuaternionCov(
	const Eigen::Vector3d& aRollPitchYaw, const Eigen::Matrix3d& aRollPitchYawCov)
{
	const auto J = quaternionJacobian(aRollPitchYaw);
	return J * aRollPitchYawCov * J.transpose();
}

Common::Entity::PredictedEulerAngle predictYaw(const Eigen::Quaterniond& aAtt)
{
	const double x = 1.0 - 2.0 * (aAtt.y() * aAtt.y() + aAtt.z() * aAtt.z());
	const double y = 2.0 * (aAtt.w() * aAtt.z() + aAtt.x() * aAtt.y());
	Eigen::Matrix<double, 1, 4> H;
	H << 2.0 * aAtt.y() * x, 2.0 * aAtt.x() * x + 4.0 * aAtt.y() * y, 2.0 * aAtt.w() * x + 4.0 * aAtt.z() * y,
		2.0 * aAtt.z() * x;
	H /= (x * x + y * y);
	return Common::Entity::PredictedEulerAngle(std::atan2(y, x), H);
}

Common::Entity::PredictedEulerAngle predictPitch(const Eigen::Quaterniond& aAtt)
{
	const double twoWYMinusXZ = 2.0 * (aAtt.w() * aAtt.y() - aAtt.x() * aAtt.z());
	Eigen::Matrix<double, 1, 4> H;
	H << -2.0 * aAtt.z(), 2.0 * aAtt.w(), -2.0 * aAtt.x(), 2.0 * aAtt.y();
	H /= std::sqrt(1.0 - twoWYMinusXZ * twoWYMinusXZ);
	return Common::Entity::PredictedEulerAngle(std::asin(twoWYMinusXZ), H);
}

Common::Entity::PredictedEulerAngle predictRoll(const Eigen::Quaterniond& aAtt)
{
	const double x = 1.0 - 2.0 * (aAtt.x() * aAtt.x() + aAtt.y() * aAtt.y());
	const double y = 2.0 * (aAtt.w() * aAtt.x() + aAtt.y() * aAtt.z());
	Eigen::Matrix<double, 1, 4> H;
	H << 2.0 * aAtt.w() * x + 4.0 * aAtt.x() * y, 2.0 * aAtt.z() * x + 4.0 * aAtt.y() * y, 2.0 * aAtt.y() * x,
		2.0 * aAtt.x() * x;
	H /= (x * x + y * y);
	return Common::Entity::PredictedEulerAngle(std::atan2(y, x), H);
}

Eigen::Vector3d predictRollPitchYaw(const Eigen::Quaterniond& aAtt)
{
	auto pRoll = predictRoll(aAtt);
	auto pPitch = predictPitch(aAtt);
	auto pYaw = predictYaw(aAtt);

	return Eigen::Vector3d(pRoll.mAngle, pPitch.mAngle, pYaw.mAngle);
}

Eigen::Matrix<double, 3, 4> rollPitchYawJacobian(const Eigen::Quaterniond& aAtt)
{
	auto pRoll = predictRoll(aAtt);
	auto pPitch = predictPitch(aAtt);
	auto pYaw = predictYaw(aAtt);

	Eigen::Matrix<double, 3, 4> H;
	H << pRoll.mJacobian, pPitch.mJacobian, pYaw.mJacobian;
	return H;
}

// Converts both attitude and its covariance to roll-pitch-yaw
Common::Entity::EulerAnglesEstimate quaternionToRollPitchYawEstimate(
	const Eigen::Quaterniond& aAtt, const Eigen::Matrix4d& aAttCov)
{
	auto pRoll = predictRoll(aAtt);
	auto pPitch = predictPitch(aAtt);
	auto pYaw = predictYaw(aAtt);

	Eigen::Vector3d angles(pRoll.mAngle, pPitch.mAngle, pYaw.mAngle);

	Eigen::Matrix<double, 3, 4> H;
	H << pRoll.mJacobian, pPitch.mJacobian, pYaw.mJacobian;

	return Common::Entity::EulerAnglesEstimate(angles, H * aAttCov * H.transpose());
}

// Jacobian of rotating a vector by a quaternion (Jacobian of aRotation.toRotationMatrix() * aVec)
Eigen::Matrix<double, 3, 4> rotationJacobian(const Eigen::Quaterniond& aRotation, const Eigen::Vector3d& aVec)
{
	const double qr = aRotation.w();
	const double qi = aRotation.x();
	const double qj = aRotation.y();
	const double qk = aRotation.z();
	const double ax = aVec.x();
	const double ay = aVec.y();
	const double az = aVec.z();
	Eigen::Matrix<double, 3, 4> J;
	J << qj * ay + qk * az, -2.0 * qj * ax + qi * ay + qr * az, -2.0 * qk * ax - qr * ay + qi * az, -qk * ay + qj * az,
		qj * ax - 2.0 * qi * ay - qr * az, qi * ax + qk * az, qr * ax - 2.0 * qk * ay + qj * az, qk * ax - qi * az,
		qk * ax + qr * ay - 2.0 * qi * az, -qr * ax + qk * ay - 2.0 * qj * az, qi * ax + qj * ay, -qj * ax + qi * ay;
	return 2.0 * J;
}

// Jacobian of rotating a vector by inverse of a quaternion (Jacobian of aRotation.conjugate().toRotationMatrix() *
// aVec)
Eigen::Matrix<double, 3, 4> invRotationJacobian(const Eigen::Quaterniond& aRotation, const Eigen::Vector3d& aVec)
{
	const double qr = aRotation.w();
	const double qi = aRotation.x();
	const double qj = aRotation.y();
	const double qk = aRotation.z();
	const double ax = aVec.x();
	const double ay = aVec.y();
	const double az = aVec.z();
	Eigen::Matrix<double, 3, 4> J;
	J << qj * ay + qk * az, -2.0 * qj * ax + qi * ay - qr * az, -2.0 * qk * ax + qr * ay + qi * az, qk * ay - qj * az,
		qj * ax - 2.0 * qi * ay + qr * az, qi * ax + qk * az, -qr * ax - 2.0 * qk * ay + qj * az, -qk * ax + qi * az,
		qk * ax - qr * ay - 2.0 * qi * az, qr * ax + qk * ay - 2.0 * qj * az, qi * ax + qj * ay, qj * ax - qi * ay;
	return 2.0 * J;
}

Eigen::Matrix4d bigOmega(const Eigen::Vector3d& aAngVel)
{
	const double wx = aAngVel.x();
	const double wy = aAngVel.y();
	const double wz = aAngVel.z();
	Eigen::Matrix4d result;
	result << 0.0, wz, -wy, wx, -wz, 0.0, wx, wy, wy, -wx, 0.0, wz, -wx, -wy, -wz, 0.0;
	return result;
}

Eigen::Matrix<double, 4, 3> delQdot_delAngVel(const Eigen::Quaterniond& aAtt)
{
	const double qr = aAtt.w();
	const double qi = aAtt.x();
	const double qj = aAtt.y();
	const double qk = aAtt.z();
	Eigen::Matrix<double, 4, 3> J;
	J << -qr, qk, -qj, -qk, -qr, qi, qj, -qi, -qr, qi, qj, qk;
	J *= -0.5;
	return J;
}
} // namespace Algorithm
} // namespace Common
