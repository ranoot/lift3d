#ifndef PRIMARY_POSE_H
#define PRIMARY_POSE_H

#include "../ConstantsEnum.h"
#include "GMDBase/Entity/TimeStamp.h"
#include "GMDBase/SystemTypeDef.h"
#include "PoseTrajectoryId.h"

namespace Common
{

namespace Entity
{

struct PrimaryPose
{
	PrimaryPose() :
		timestamp(Common::Time::now()),
		hardwareTimestamp(),
		sourceSegmentId(0U),
		destinationSegmentId(0U),
		frameId(0U),
		poseTrajectoryId(),
		x(0.0),
		y(0.0),
		z(0.0f),
		xVel(0.0f),
		yVel(0.0f),
		zVel(0.0f),
		roll(0.0f),
		pitch(0.0f),
		yaw(0.0f),
		rollRate(0.0f),
		pitchRate(0.0f),
		yawRate(0.0f)
	{
	}

	explicit PrimaryPose(const Common::Entity::TimeStamp& aTimestamp,
		const Common::Entity::TimeStamp& aHardwareTimestamp, const uint16 aSourceSegmentId,
		const uint16 aDestinationSegmentId, const uint8 aFrameId,
		const Common::Entity::PoseTrajectoryId& aPoseTrajectoryId, const double aX, const double aY, const float aZ,
		const float aXVelocity, const float aYVelocity, const float aZVelocity, const float aRoll, const float aPitch,
		const float aYaw, const float aRollRate, const float aPitchRate, const float aYawRate) :
		timestamp(aTimestamp),
		hardwareTimestamp(aHardwareTimestamp),
		sourceSegmentId(aSourceSegmentId),
		destinationSegmentId(aDestinationSegmentId),
		frameId(aFrameId),
		poseTrajectoryId(aPoseTrajectoryId),
		x(aX),
		y(aY),
		z(aZ),
		xVel(aXVelocity),
		yVel(aYVelocity),
		zVel(aZVelocity),
		roll(aRoll),
		pitch(aPitch),
		yaw(aYaw),
		rollRate(aRollRate),
		pitchRate(aPitchRate),
		yawRate(aYawRate)
	{
	}

	// 100 nanosecond, time relative to 1970-01-01 0000hr. This refers to "system time", a monotonically increasing
	// clock based on the computer timestamp.
	Common::Entity::TimeStamp timestamp;
	// 100 nanosecond, time relative to 1970-01-01 0000hr. This refers to time on the IMU (may not be the same as system
	// time)
	Common::Entity::TimeStamp hardwareTimestamp;
	// Can be safely ignored for you
	uint16 sourceSegmentId;
	// Can be safely ignored for you
	uint16 destinationSegmentId;
	uint8 frameId;
	Common::Entity::PoseTrajectoryId poseTrajectoryId;
	double x;		 // meters
	double y;		 // meters
	float z;		 // meters
	float xVel;		 // meters/second
	float yVel;		 // meters/second
	float zVel;		 // meters/second
	float roll;		 // radians
	float pitch;	 // radians
	float yaw;		 // radians
	float rollRate;	 // radians/second
	float pitchRate; // radians/second
	float yawRate;	 // radians/second
};

template <typename T> T& operator<<(T& aStream, const PrimaryPose& aPose)
{
	aStream << "Primary Pose:";
	aStream << "\n\ttimestamp: " << aPose.timestamp.time_since_epoch().count();
	aStream << "\n\thardwareTimestamp: " << aPose.hardwareTimestamp.time_since_epoch().count();
	aStream << "\n\tsourceSegmentId: " << aPose.sourceSegmentId;
	aStream << "\n\tdestinationSegmentId: " << aPose.destinationSegmentId;
	aStream << "\n\tframeId: " << static_cast<int>(aPose.frameId);
	aStream << "\n\tposeTrajectoryId: " << aPose.poseTrajectoryId.value;
	aStream << "\n\tx: " << aPose.x;
	aStream << "\n\ty: " << aPose.y;
	aStream << "\n\tz: " << aPose.z;
	aStream << "\n\txVel: " << aPose.xVel;
	aStream << "\n\tyVel: " << aPose.yVel;
	aStream << "\n\tzVel: " << aPose.zVel;
	aStream << "\n\troll: " << aPose.roll;
	aStream << "\n\tpitch: " << aPose.pitch;
	aStream << "\n\tyaw: " << aPose.yaw;
	aStream << "\n\trollRate: " << aPose.rollRate;
	aStream << "\n\tpitchRate: " << aPose.pitchRate;
	aStream << "\n\tyawRate: " << aPose.yawRate;
	aStream << "\n";
	return aStream;
}

} // namespace Entity

} // namespace Common

#endif // PRIMARY_POSE_H
