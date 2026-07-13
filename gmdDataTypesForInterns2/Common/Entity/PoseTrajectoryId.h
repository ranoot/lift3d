#ifndef POSE_TRAJECTORY_ID_H
#define POSE_TRAJECTORY_ID_H
#include <cstdint>
#include <ostream>

namespace Common
{

namespace Entity
{

struct PoseTrajectoryId
{
	using UnderlyingType = int64_t;

	PoseTrajectoryId() :
		value(0)
	{
	}
	explicit PoseTrajectoryId(UnderlyingType aValue) :
		value(aValue)
	{
	}

	template <class Archive> void serialize(Archive& ar)
	{
		ar(value);
	}

	UnderlyingType value;
};

inline bool operator==(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value == rhs.value;
}

inline bool operator!=(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value != rhs.value;
}

inline bool operator<(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value < rhs.value;
}

inline bool operator<=(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value <= rhs.value;
}

inline bool operator>(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value > rhs.value;
}

inline bool operator>=(const PoseTrajectoryId& lhs, const PoseTrajectoryId& rhs) noexcept
{
	return lhs.value >= rhs.value;
}

inline std::ostream& operator<<(std::ostream& aStream, const PoseTrajectoryId& aPoseTrajectoryId)
{
	return aStream << aPoseTrajectoryId.value;
}

} // namespace Entity

namespace Interface
{

inline int64_t toDDSPoseTrajectoryId(const Common::Entity::PoseTrajectoryId& aPoseTrajectoryId)
{
	return aPoseTrajectoryId.value;
}

inline Common::Entity::PoseTrajectoryId fromDDSPoseTrajectoryId(const int64_t aDDSPoseTrajectoryId)
{
	return Common::Entity::PoseTrajectoryId(aDDSPoseTrajectoryId);
}

} // namespace Interface
} // namespace Common

#endif
