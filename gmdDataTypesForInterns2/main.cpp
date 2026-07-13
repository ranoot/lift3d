#include <iostream>
#include <opencv2/opencv.hpp>

#include "Common/Algorithm/ImageUtils.h"
#include "Common/Entity/Image.h"
#include "Common/Entity/PointCloud.h"
#include "Common/Entity/PoseTrajectoryId.h"
#include "Common/Entity/PrimaryPose.h"
#include "Common/Entity/Transformation.h"

int main()
{
	// We use Common::Entity::PrimaryPose to define 6DOF pose w.r.t robot origin
	Common::Entity::PrimaryPose pp(Common::Time::now(), Common::Time::now(), 1, 1, 0,
		Common::Entity::PoseTrajectoryId(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

	std::cout << "Created a primary pose:\n" << pp << std::endl;

	// Transformation matrix.
	Common::Entity::Transformation t; // You can use this, or Eigen::Affine3d

	// Point Cloud example.
	Common::Entity::PointCloud pc;
	for (int i = 0; i < 100; i++)
	{
		Common::Entity::ProcessedPoint newPt;
		newPt.x = newPt.y = newPt.z = static_cast<float>(i);
		newPt.intensity = 128;
		newPt.sensorId = 71;
		pc.pointCloudData.push_back(newPt);
	}

	// Image
	Common::Entity::Image img;
	Common::Algorithm::setupCommonImage(img, 0, {640, 480});
	std::cout << "Image has WH=" << img.width << ", " << img.height << " encoding=" << static_cast<int>(img.encoding)
			  << std::endl;
}
