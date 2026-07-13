#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "Common/Entity/Image.h"
#include <opencv2/opencv.hpp>

namespace Common
{
namespace Algorithm
{
bool convertRGB8ToMono8(const Common::Entity::Image& aImageIn, Common::Entity::Image& aImageOut);
bool convertRGB8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut);
bool convertMono8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut);
bool convertBGR8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut);
void setupCommonImage(Common::Entity::Image& msg, const int camId, const cv::Size imgSize);
void cvMatToImage(const cv::Mat& img, const Common::Entity::TimeStamp& timestamp,
	const Common::Entity::TimeStamp& hwTimestamp, Common::Entity::Image& msg);
void imageToCvMat(const Common::Entity::Image& msg, cv::Mat& img, bool singleChannel = false);
} // namespace Algorithm
} // namespace Common

#endif
