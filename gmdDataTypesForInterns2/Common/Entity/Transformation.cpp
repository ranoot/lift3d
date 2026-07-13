#include "Transformation.h"

namespace Common
{
namespace Entity
{

Transformation::Transformation(void) :
	mRotation(Eigen::Quaterniond::Identity()),
	mTranslation(Eigen::Vector3d::Zero())
{
}

Transformation::Transformation(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) :
	mRotation(q),
	mTranslation(t)
{
}

Transformation::Transformation(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) :
	mRotation(R),
	mTranslation(t)
{
}

Transformation::Transformation(double roll, double pitch, double yaw, const Eigen::Vector3d& t) :
	mRotation(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())),
	mTranslation(t)
{
}

Transformation::Transformation(double roll, double pitch, double yaw, double x, double y, double z) :
	mRotation(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())),
	mTranslation(x, y, z)
{
}

Eigen::Matrix3d Transformation::getRotationMatrix(void) const
{
	return mRotation.toRotationMatrix();
}

void Transformation::getRollPitchYaw(double& roll, double& pitch, double& yaw) const
{
	double w = mRotation.w();
	double x = mRotation.x();
	double y = mRotation.y();
	double z = mRotation.z();

	roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
	pitch = asin(2.0 * (w * y - x * z));
	yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

Eigen::Matrix4d Transformation::getTransformationMatrix(void) const
{
	Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
	H.topLeftCorner(3, 3) = getRotationMatrix();
	H.topRightCorner(3, 1) = mTranslation;

	return H;
}

void Transformation::setIdentity(void)
{
	resetRotation();
	resetTranslation();
}

void Transformation::setRotation(const Eigen::Matrix3d& rotationMatrix)
{
	mRotation = Eigen::Quaterniond(rotationMatrix);
}

void Transformation::setRollPitchYaw(double roll, double pitch, double yaw)
{
	double cr = cos(0.5 * roll);
	double sr = sin(0.5 * roll);
	double cp = cos(0.5 * pitch);
	double sp = sin(0.5 * pitch);
	double cy = cos(0.5 * yaw);
	double sy = sin(0.5 * yaw);

	mRotation.coeffs() << sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
		cr * cp * cy + sr * sp * sy;
}

void Transformation::setTransformation(const Eigen::Matrix4d& transformationMatrix)
{
	mRotation = Eigen::Quaterniond(transformationMatrix.block<3, 3>(0, 0));
	mTranslation = transformationMatrix.block<3, 1>(0, 3);
}

Transformation Transformation::identity(void)
{
	Transformation T;
	T.resetRotation();
	T.resetTranslation();
	return T;
}

Transformation Transformation::inverse(void) const
{
	Transformation T;

	T.mRotation = mRotation.conjugate();
	T.mTranslation = -(T.mRotation * mTranslation);

	return T;
}

Transformation Transformation::inverse(Eigen::Matrix<double, 7, 7>& J) const
{
	Transformation T;

	T.mRotation = mRotation.conjugate();
	T.mTranslation = -(T.mRotation * mTranslation);

	const double qx = mRotation.x();
	const double qy = mRotation.y();
	const double qz = mRotation.z();
	const double qw = mRotation.w();
	const double tx = mTranslation.x();
	const double ty = mTranslation.y();
	const double tz = mTranslation.z();

	J.setZero();

	J(0, 0) = -1.0;
	J(4, 0) = -2.0 * (qy * ty + qz * tz);
	J(5, 0) = 4.0 * qx * ty - 2.0 * (qw * tz + qy * tx);
	J(6, 0) = 4.0 * qx * tz + 2.0 * (qw * ty - qz * tx);

	J(1, 1) = -1.0;
	J(4, 1) = 4.0 * qy * tx + 2.0 * (qw * tz - qx * ty);
	J(5, 1) = -2.0 * (qx * tx + qz * tz);
	J(6, 1) = 4.0 * qy * tz - 2.0 * (qw * tx + qz * ty);

	J(2, 2) = -1.0;
	J(4, 2) = 4.0 * qz * tx - 2.0 * (qw * ty + qx * tz);
	J(5, 2) = 4.0 * qz * ty + 2.0 * (qw * tx - qy * tz);
	J(6, 2) = -2.0 * (qx * tx + qy * ty);

	J(3, 3) = 1.0;
	J(4, 3) = 2.0 * (qy * tz - qz * ty);
	J(5, 3) = 2.0 * (qz * tx - qx * tz);
	J(6, 3) = 2.0 * (qx * ty - qy * tx);

	J(4, 4) = 2.0 * (qy * qy + qz * qz) - 1.0;
	J(5, 4) = 2.0 * (qw * qz - qx * qy);
	J(6, 4) = -2.0 * (qw * qy + qx * qz);

	J(4, 5) = -2.0 * (qw * qz + qx * qy);
	J(5, 5) = 2.0 * (qx * qx + qz * qz) - 1.0;
	J(6, 5) = 2.0 * (qw * qx - qy * qz);

	J(4, 6) = 2.0 * (qw * qy - qx * qz);
	J(5, 6) = -2.0 * (qw * qx + qy * qz);
	J(6, 6) = 2.0 * (qx * qx + qy * qy) - 1.0;

	return T;
}

Eigen::Matrix<double, 6, 6> Transformation::adjoint() const
{
	Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();

	const Eigen::Matrix3d R = getRotationMatrix();
	const Eigen::Vector3d T = getTranslation();
	const Eigen::Matrix3d T_skew =
		(Eigen::Matrix3d() << 0.0, -T(2), T(1), T(2), 0.0, -T(0), -T(1), T(0), 0.0).finished();

	adj.topLeftCorner(3, 3) = R;
	adj.bottomRightCorner(3, 3) = R;
	adj.topRightCorner(3, 3) = T_skew * R;

	return adj;
}

Transformation Transformation::leftMultiply(const Transformation& aT, Eigen::Matrix<double, 7, 7>& J) const
{
	Transformation T;

	T.mRotation = aT.mRotation * mRotation;
	T.mTranslation = aT.mRotation * mTranslation + aT.mTranslation;

	const double x = aT.mRotation.x();
	const double y = aT.mRotation.y();
	const double z = aT.mRotation.z();
	const double w = aT.mRotation.w();

	const double xx = x * x;
	const double xy = x * y;
	const double xz = x * z;
	const double xw = x * w;
	const double yy = y * y;
	const double yz = y * z;
	const double yw = y * w;
	const double zz = z * z;
	const double zw = z * w;

	J.setZero();

	J(0, 0) = w;
	J(1, 0) = z;
	J(2, 0) = -y;
	J(3, 0) = -x;

	J(0, 1) = -z;
	J(1, 1) = w;
	J(2, 1) = x;
	J(3, 1) = -y;

	J(0, 2) = y;
	J(1, 2) = -x;
	J(2, 2) = w;
	J(3, 2) = -z;

	J(0, 3) = x;
	J(1, 3) = y;
	J(2, 3) = z;
	J(3, 3) = w;

	J(4, 4) = 1.0 - 2.0 * (yy + zz);
	J(5, 4) = 2.0 * (xy + zw);
	J(6, 4) = 2.0 * (xz - yw);

	J(4, 5) = 2.0 * (xy - zw);
	J(5, 5) = 1.0 - 2.0 * (xx + zz);
	J(6, 5) = 2.0 * (xw + yz);

	J(4, 6) = 2.0 * (xz + yw);
	J(5, 6) = 2.0 * (yz - xw);
	J(6, 6) = 1.0 - 2.0 * (xx + yy);

	return T;
}

Transformation Transformation::rightMultiply(const Transformation& aT, Eigen::Matrix<double, 7, 7>& J) const
{
	Transformation T;

	T.mRotation = mRotation * aT.mRotation;
	T.mTranslation = mRotation * aT.mTranslation + mTranslation;

	const double qx1 = mRotation.x();
	const double qy1 = mRotation.y();
	const double qz1 = mRotation.z();
	const double qw1 = mRotation.w();

	const double qx2 = aT.mRotation.x();
	const double qy2 = aT.mRotation.y();
	const double qz2 = aT.mRotation.z();
	const double qw2 = aT.mRotation.w();
	const double tx2 = aT.mTranslation.x();
	const double ty2 = aT.mTranslation.y();
	const double tz2 = aT.mTranslation.z();

	J.setZero();

	J(0, 0) = qw2;
	J(1, 0) = -qz2;
	J(2, 0) = qy2;
	J(3, 0) = -qx2;
	J(4, 0) = 2.0 * (qy1 * ty2 + qz1 * tz2);
	J(5, 0) = 2.0 * (qy1 * tx2 - qw1 * tz2) - 4.0 * qx1 * ty2;
	J(6, 0) = 2.0 * (qw1 * ty2 + qz1 * tx2) - 4.0 * qx1 * tz2;

	J(0, 1) = qz2;
	J(1, 1) = qw2;
	J(2, 1) = -qx2;
	J(3, 1) = -qy2;
	J(4, 1) = 2.0 * (qw1 * tz2 + qx1 * ty2) - 4.0 * qy1 * tx2;
	J(5, 1) = 2.0 * (qx1 * tx2 + qz1 * tz2);
	J(6, 1) = 2.0 * (qz1 * ty2 - qw1 * tx2) - 4.0 * qy1 * tz2;

	J(0, 2) = -qy2;
	J(1, 2) = qx2;
	J(2, 2) = qw2;
	J(3, 2) = -qz2;
	J(4, 2) = 2.0 * (qx1 * tz2 - qw1 * ty2) - 4.0 * qz1 * tx2;
	J(5, 2) = 2.0 * (qw1 * tx2 + qy1 * tz2) - 4.0 * qz1 * ty2;
	J(6, 2) = 2.0 * (qx1 * tx2 + qy1 * ty2);

	J(0, 3) = qx2;
	J(1, 3) = qy2;
	J(2, 3) = qz2;
	J(3, 3) = qw2;
	J(4, 3) = 2.0 * (qy1 * tz2 - qz1 * ty2);
	J(5, 3) = 2.0 * (qz1 * tx2 - qx1 * tz2);
	J(6, 3) = 2.0 * (qx1 * ty2 - qy1 * tx2);

	J(4, 4) = 1.0;
	J(5, 5) = 1.0;
	J(6, 6) = 1.0;

	return T;
}

Transformation Transformation::interpolateLinear(double s, Eigen::Matrix<double, 7, 7>& J) const
{
	J.setZero();

	const double one = 1.0 - std::numeric_limits<double>::epsilon();

	Eigen::Vector4d q_coeffs;
	if (mRotation.w() < 0.0)
	{
		q_coeffs = -mRotation.coeffs();
	}
	else
	{
		q_coeffs = mRotation.coeffs();
	}

	double d = q_coeffs(3);

	double scale0;
	double scale1;

	if (d >= one)
	{
		scale0 = 1.0 - s;
		scale1 = s;

		// q
		J(0, 0) = scale1;
		J(1, 1) = scale1;
		J(2, 2) = scale1;
		J(3, 3) = scale1;
	}
	else
	{
		double theta = acos(d);
		double sinTheta = sin(theta);

		scale0 = sin((1.0 - s) * theta) / sinTheta;
		scale1 = sin(s * theta) / sinTheta;

		// q
		J(0, 0) = scale1;
		J(1, 1) = scale1;
		J(2, 2) = scale1;

		double sinTheta2 = sinTheta * sinTheta;
		double sinTheta3 = sinTheta * sinTheta2;

		J.block<3, 1>(0, 3) = (d * scale1 - s * cos(s * theta)) / sinTheta2 * q_coeffs.topRows(3);
		J(3, 3) = d * d * scale1 / sinTheta2 - d * s * cos(s * theta) / sinTheta2 +
				  d * sin((1.0 - s) * theta) / sinTheta3 - (1.0 - s) * cos((1 - s) * theta) / sinTheta2 + scale1;
	}

	// t
	J(4, 4) = s;
	J(5, 5) = s;
	J(6, 6) = s;

	return Transformation(
		Eigen::Quaterniond(scale0 * (Eigen::Vector4d() << 0.0, 0.0, 0.0, 1.0).finished() + scale1 * q_coeffs),
		s * mTranslation);
}

Eigen::Vector3d Transformation::transformPoint(const Eigen::Vector3d& P, Eigen::Matrix<double, 3, 7>& J) const
{
	const double qx = mRotation.x();
	const double qy = mRotation.y();
	const double qz = mRotation.z();
	const double qw = mRotation.w();
	const double px = P.x();
	const double py = P.y();
	const double pz = P.z();

	J.setZero();

	// q_x
	J.col(0) << 2.0 * qy * py + 2.0 * qz * pz, -2.0 * qw * pz - 4.0 * qx * py + 2.0 * qy * px,
		2.0 * qw * py - 4.0 * qx * pz + 2.0 * qz * px;

	// q_y
	J.col(1) << 2.0 * qw * pz + 2.0 * qx * py - 4.0 * qy * px, 2.0 * qx * px + 2.0 * qz * pz,
		-2.0 * qw * px - 4.0 * qy * pz + 2.0 * qz * py;

	// q_z
	J.col(2) << -2.0 * qw * py + 2.0 * qx * pz - 4.0 * qz * px, 2.0 * qw * px + 2.0 * qy * pz - 4.0 * qz * py,
		2.0 * qx * px + 2.0 * qy * py;

	// q_w
	J.col(3) << 2.0 * qy * pz - 2.0 * qz * py, -2.0 * qx * pz + 2.0 * qz * px, 2.0 * qx * py - 2.0 * qy * px;

	// t_x, t_y, t_z
	J.block<3, 3>(0, 4).setIdentity();

	return mRotation * P + mTranslation;
}

Transformation& Transformation::operator*=(const Transformation& aT)
{
	const auto R1 = mRotation;
	setRotation(R1 * aT.getRotation());
	setTranslation(R1 * aT.getTranslation() + mTranslation);
	return *this;
}

Eigen::Quaterniond Transformation::getRotation(
	void) const // parasoft-suppress OPT-14 "Avoid returning reference to member variable"
{
	return mRotation;
}

Eigen::Vector3d Transformation::getTranslation(
	void) const // parasoft-suppress OPT-14 "Avoid returning reference to member variable"
{
	return mTranslation;
}

double* Transformation::getRotationData(void) // parasoft-suppress OPT-14 "Avoid returning reference to member variable"
{
	return mRotation.coeffs().data(); // parasoft-suppress  CODSTA-CPP-06 "False alarm, intentional reference" //
									  // parasoft-suppress  OOP-36 "False alarm, intentional reference"
}

double* Transformation::getTranslationData(
	void) // parasoft-suppress OPT-14 "Avoid returning reference to member variable"
{
	return mTranslation.data();
}

void Transformation::setRotation(const Eigen::Quaterniond& aRotation)
{
	mRotation = aRotation;
}

void Transformation::setTranslation(const Eigen::Vector3d& aTranslation)
{
	mTranslation = aTranslation;
}

void Transformation::rollPitchYawToQuaternion(
	double roll, double pitch, double yaw, Eigen::Quaterniond& q, Eigen::Matrix<double, 4, 3>& J)
{
	double cr = cos(0.5 * roll);
	double sr = sin(0.5 * roll);
	double cp = cos(0.5 * pitch);
	double sp = sin(0.5 * pitch);
	double cy = cos(0.5 * yaw);
	double sy = sin(0.5 * yaw);

	double ccc = cr * cp * cy;
	double ccs = cr * cp * sy;
	double csc = cr * sp * cy;
	double css = cr * sp * sy;
	double scc = sr * cp * cy;
	double scs = sr * cp * sy;
	double ssc = sr * sp * cy;
	double sss = sr * sp * sy;

	q.coeffs() << scc - css, csc + scs, ccs - ssc, ccc + sss;

	J << 0.5 * (ccc + sss), -0.5 * (ssc + ccs), -0.5 * (csc + scs), 0.5 * (ccs - ssc), 0.5 * (ccc - sss),
		0.5 * (scc - css), -0.5 * (csc + scs), -0.5 * (css + scc), 0.5 * (ccc + sss), 0.5 * (css - scc),
		0.5 * (scs - csc), 0.5 * (ssc - ccs);
}

void Transformation::angleAxisToQuaternion(
	const Eigen::Vector3d& angleAxis, Eigen::Quaterniond& q, Eigen::Matrix<double, 4, 3>& J)
{
	double t = angleAxis.norm();

	double x = angleAxis(0);
	double y = angleAxis(1);
	double z = angleAxis(2);

	double nx = x / t;
	double ny = y / t;
	double nz = z / t;

	double ct = cos(t / 2.0);
	double st = sin(t / 2.0);

	q.coeffs() << nx * st, ny * st, nz * st, ct;

	Eigen::Vector3d Jt(nx, ny, nz);

	double a = ct / (2.0 * t) - st / (t * t);
	Eigen::Vector4d Jq(x * a, y * a, z * a, -0.5 * st);

	J = Jq * Jt.transpose();
}

void Transformation::quaternionToRollPitchYaw(
	const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw, Eigen::Matrix<double, 3, 4>& J)
{
	double x = q.x();
	double y = q.y();
	double z = q.z();
	double w = q.w();

	double a = w * x + y * z;
	double b = 1.0 - 2.0 * (x * x + y * y);

	double c = w * z + x * y;
	double d = 1.0 - 2.0 * (y * y + z * z);

	roll = atan2(2.0 * a, b);
	pitch = asin(2.0 * (w * y - x * z));
	yaw = atan2(2.0 * c, d);

	J(0, 0) = 2.0 * (w / b + 4.0 * x * a / (b * b));
	J(0, 1) = 2.0 * (z / b + 4.0 * y * a / (b * b));
	J(0, 2) = 2.0 * y / b;
	J(0, 3) = 2.0 * x / b;
	J.row(0) *= cos(roll) * cos(roll);

	J(1, 0) = -2.0 * z;
	J(1, 1) = 2.0 * w;
	J(1, 2) = -2.0 * x;
	J(1, 3) = 2.0 * y;
	J.row(1) /= sin(pitch);

	J(2, 0) = 2.0 * y / d;
	J(2, 1) = 2.0 * (x / d + 4.0 * y * c / (d * d));
	J(2, 2) = 2.0 * (w / d + 4.0 * z * c / (d * d));
	J(2, 3) = 2.0 * z / d;
	J.row(2) *= cos(yaw) * cos(yaw);
}

void Transformation::resetRotation(void)
{
	mRotation.setIdentity();
}

void Transformation::resetTranslation(void)
{
	mTranslation.setZero();
}

Transformation operator*(const Transformation& aT1, const Transformation& aT2)
{
	Transformation newT = aT1;
	return (newT *= aT2);
}

Eigen::Vector3d operator*(const Transformation& aT, const Eigen::Vector3d& aP)
{
	return aT.getRotation() * aP + aT.getTranslation();
}

} // namespace Entity
} // namespace Common
