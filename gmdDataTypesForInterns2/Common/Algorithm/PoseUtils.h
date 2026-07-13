#ifndef POSE_UTILS_H
#define POSE_UTILS_H

#include <Eigen/Dense>

namespace Common
{
namespace Entity
{
struct PredictedEulerAngle
{
	PredictedEulerAngle(double aAngle, const Eigen::Matrix<double, 1, 4>& aJacobian) :
		mAngle(aAngle),
		mJacobian(aJacobian)
	{
	}
	double mAngle;
	Eigen::Matrix<double, 1, 4> mJacobian;
};

struct EulerAnglesEstimate
{
	EulerAnglesEstimate(const Eigen::Vector3d& aAngles, const Eigen::Matrix3d& aCov) :
		mAngles(aAngles),
		mCov(aCov)
	{
	}
	Eigen::Vector3d mAngles;
	Eigen::Matrix3d mCov;
};
} // namespace Entity

namespace Algorithm
{
Eigen::Quaterniond rollPitchYawToQuaternion(double aRoll, double aPitch, double aYaw);
Eigen::Quaterniond rollPitchYawToQuaternion(const Eigen::Vector3d& aRollPitchYaw);
Eigen::Matrix<double, 4, 3> quaternionJacobian(double aRoll, double aPitch, double aYaw);
Eigen::Matrix<double, 4, 3> quaternionJacobian(const Eigen::Vector3d& aRollPitchYaw);
Eigen::Matrix<double, 4, 4> rollPitchYawCovToQuaternionCov(
	const Eigen::Vector3d& aRollPitchYaw, const Eigen::Matrix3d& aRollPitchYawCov);

Common::Entity::PredictedEulerAngle predictYaw(const Eigen::Quaterniond& aAtt);
Common::Entity::PredictedEulerAngle predictPitch(const Eigen::Quaterniond& aAtt);
Common::Entity::PredictedEulerAngle predictRoll(const Eigen::Quaterniond& aAtt);
Eigen::Vector3d predictRollPitchYaw(const Eigen::Quaterniond& aAtt);
Eigen::Matrix<double, 3, 4> rollPitchYawJacobian(const Eigen::Quaterniond& aAtt);
Common::Entity::EulerAnglesEstimate quaternionToRollPitchYawEstimate(
	const Eigen::Quaterniond& aAtt, const Eigen::Matrix4d& aAttCov);

Eigen::Matrix<double, 3, 4> rotationJacobian(const Eigen::Quaterniond& aRotation, const Eigen::Vector3d& aVec);
Eigen::Matrix<double, 3, 4> invRotationJacobian(const Eigen::Quaterniond& aRotation, const Eigen::Vector3d& aVec);

Eigen::Matrix4d bigOmega(const Eigen::Vector3d& aAngVel);
Eigen::Matrix<double, 4, 3> delQdot_delAngVel(const Eigen::Quaterniond& aAtt);
} // namespace Algorithm
} // namespace Common

#endif
