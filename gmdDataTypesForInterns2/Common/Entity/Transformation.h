#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

/**
	\file Transformation.h
	\see Common::Entity::Transformation
	\class Common::Entity::Transformation
	\brief The Transformation class encodes a 3D transformation which is
		   parameterized by a rotation quaternion and translation vector.
*/
#include <Eigen/Dense>

namespace Common
{
namespace Entity
{

class Transformation
{
  public:
	/// \name Constructor and Destructor functions

	/**
		Default constructor.
	*/
	explicit Transformation(void);

	explicit Transformation(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
	explicit Transformation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	explicit Transformation(double roll, double pitch, double yaw, const Eigen::Vector3d& t);
	explicit Transformation(double roll, double pitch, double yaw, double x, double y, double z);

	/**
		\brief Obtain the 3x3 rotation matrix corresponding to the rotation
			   quaternion.
		\return The 3x3 rotation matrix corresponding to the rotation quaternion.
	*/
	Eigen::Matrix3d getRotationMatrix(void) const;

	/**
		\brief Obtain the roll, pitch, and yaw angles corresponding to the rotation
			   quaternion.
		\return The roll, pitch, and yaw angles corresponding to the rotation quaternion.
	*/
	void getRollPitchYaw(double& roll, double& pitch, double& yaw) const;

	/**
		\brief Obtain the 4x4 transformation matrix corresponding to the
			   transformation.
		\return The 4x4 transformation matrix.
	*/
	Eigen::Matrix4d getTransformationMatrix(void) const;

	/**
		\brief Set the transformation to an identity transformation.
	*/
	void setIdentity(void);

	/**
		\brief Set the rotation.
		\param[in] rotationMatrix is a 3x3 rotation matrix.
	*/
	void setRotation(const Eigen::Matrix3d& rotationMatrix);

	/**
		\brief Set the roll, pitch, and yaw angles.
		\param[in] Roll is the roll angle in radians.
		\param[in] Pitch is the pitch angle in radians.
		\param[in] Yaw is the yaw angle in radians.
	*/
	void setRollPitchYaw(double roll, double pitch, double yaw);

	/**
		\brief Set the transformation.
		\param[in] Transformation is a 4x4 matrix.
	*/
	void setTransformation(const Eigen::Matrix4d& transformationMatrix);

	/**
		\brief Returns an identity transformation.
		\return Identity transformation.
	*/
	static Transformation identity(void);

	/**
		\brief Returns the inverse of the transformation.
		\return The inverse of the transformation.
	*/
	Transformation inverse(void) const;
	Transformation inverse(Eigen::Matrix<double, 7, 7>& J) const;

	/**
		\brief Returns an adjoint matrix.
		\return The adjoint representation of the transformation.
	*/
	Eigen::Matrix<double, 6, 6> adjoint() const;

	Transformation leftMultiply(const Transformation& aT, Eigen::Matrix<double, 7, 7>& J) const;
	Transformation rightMultiply(const Transformation& aT, Eigen::Matrix<double, 7, 7>& J) const;

	Transformation interpolateLinear(double s, Eigen::Matrix<double, 7, 7>& J) const;

	Eigen::Vector3d transformPoint(const Eigen::Vector3d& P, Eigen::Matrix<double, 3, 7>& J) const;

	Transformation& operator*=(const Transformation& aT);

	/// \name Accessor functions
	Eigen::Quaterniond getRotation(void) const;
	Eigen::Vector3d getTranslation(void) const;
	double* getRotationData(void);
	double* getTranslationData(void);

	/// \name Mutator functions
	void setRotation(const Eigen::Quaterniond& aRotation);
	void setTranslation(const Eigen::Vector3d& aTranslation);

	/// \name Static functions
	static void rollPitchYawToQuaternion(
		double roll, double pitch, double yaw, Eigen::Quaterniond& q, Eigen::Matrix<double, 4, 3>& J);

	static void angleAxisToQuaternion(
		const Eigen::Vector3d& angleAxis, Eigen::Quaterniond& q, Eigen::Matrix<double, 4, 3>& J);

	static void quaternionToRollPitchYaw(
		const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw, Eigen::Matrix<double, 3, 4>& J);

  protected:
	void resetRotation(void);
	void resetTranslation(void);

  protected:
	Eigen::Quaterniond
		mRotation; ///< Rotation quaternion. /* parasoft-suppress  OOP-19 "intentional protected member variables" */
	Eigen::Vector3d
		mTranslation; ///< Translation vector. /* parasoft-suppress  OOP-19 "intentional protected member variables" */
}; // Transformation

Transformation operator*(const Transformation& aT1, const Transformation& aT2);
Eigen::Vector3d operator*(const Transformation& aT, const Eigen::Vector3d& aP);

} // namespace Entity
} // namespace Common

#endif
