#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include <Eigen/Dense>
#include <vector>

#include "Common/Entity/Transformation.h"
#include "GMDBase/Entity/TimeStamp.h"

#include "../ConstantsEnum.h"
#include "GMDBase/SystemTypeDef.h"

namespace Common
{

namespace Entity
{

// parasoft suppress item OPT-17 "EL-SUPP-15: Pure Entity class."
// parasoft suppress item MISRA2008-11_0_1 "False alarm, attribute is in a structure"
struct ProcessedPoint
{
	explicit ProcessedPoint() :
		x(0.0f),
		y(0.0f),
		z(0.0f),
		intensity(0U),
		sensorId(0U),
		semanticLabel(SemanticLabel::UNLABELED),
		semanticLikelihood(0.0f)
	{
	}

	float x;
	float y;
	float z;
	uint8 intensity;
	uint8 sensorId;
	SemanticLabel semanticLabel;
	float semanticLikelihood;
}; // class ProcessedPoint

struct PointCloud
{
	Common::Entity::TimeStamp timestamp;
	Common::Entity::TimeStamp hardwareTimestamp;
	uint16 sourceSegmentId;
	std::vector<ProcessedPoint> pointCloudData;

	void transform(const Common::Entity::Transformation& aT)
	{
		for (auto& pt : pointCloudData)
		{
			const Eigen::Vector3f ptVec(pt.x, pt.y, pt.z);
			const Eigen::Vector3f transformedPtVec = (aT * ptVec.cast<double>()).cast<float>();

			pt.x = transformedPtVec.x();
			pt.y = transformedPtVec.y();
			pt.z = transformedPtVec.z();
		}
	}
};

} // namespace Entity

} // namespace Common

#endif // POINT_CLOUD_H
