#include "ImageUtils.h"
#include <iostream>
#include <opencv2/core/base.hpp>

static const float R_WEIGHT = 0.299f;
static const float G_WEIGHT = 0.587f;
static const float B_WEIGHT = 0.114f;

namespace Common
{
namespace Algorithm
{
bool convertRGB8ToMono8(const Common::Entity::Image& aImageIn, Common::Entity::Image& aImageOut)
{
	bool success = false;

	if (aImageIn.encoding == ImageEncoding::RGB8)
	{
		aImageOut.timestamp = aImageIn.timestamp;
		aImageOut.sourceSegmentId = aImageIn.sourceSegmentId;
		aImageOut.sensorId = aImageIn.sensorId;
		aImageOut.height = aImageIn.height;
		aImageOut.width = aImageIn.width;
		aImageOut.encoding = ImageEncoding::MONO8;
		aImageOut.step = aImageIn.width;
		aImageOut.seq = aImageIn.seq;
		aImageOut.exposureTime = aImageIn.exposureTime;
		aImageOut.gain = aImageIn.gain;
		aImageOut.data.resize(aImageOut.height * aImageOut.width);

		for (auto row = 0; row < aImageOut.height; ++row)
		{
			for (auto col = 0; col < aImageOut.width; ++col)
			{
				const float intensity = R_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 0)) +
										G_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 1)) +
										B_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 2));
				aImageOut.mono8(row, col) = static_cast<uint8>(intensity);
			}
		}

		success = true;
	}

	return success;
}

bool convertRGB8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut)
{
	bool success = false;

	if (aImageIn.encoding == ImageEncoding::RGB8)
	{
		aImageOut.timestamp = aImageIn.timestamp;
		aImageOut.height = aImageIn.height;
		aImageOut.width = aImageIn.width;
		aImageOut.seq = aImageIn.seq;
		aImageOut.exposureTime = aImageIn.exposureTime;
		aImageOut.data.resize(aImageOut.height * aImageOut.width);

		static const float SCALE_UP = 255.0f;
		for (auto row = 0; row < aImageOut.height; ++row)
		{
			for (auto col = 0; col < aImageOut.width; ++col)
			{
				const float intensity = R_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 0)) +
										G_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 1)) +
										B_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 2));
				aImageOut(row, col) = static_cast<uint16>(SCALE_UP * intensity);
			}
		}

		success = true;
	}

	return success;
}

bool convertMono8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut)
{
	bool success = false;

	if (aImageIn.encoding == ImageEncoding::MONO8)
	{
		aImageOut.height = aImageIn.height;
		aImageOut.width = aImageIn.width;
		aImageOut.data.resize(aImageOut.height * aImageOut.width);

		static const uint16 SCALE_UP = 255U;
		for (auto row = 0; row < aImageOut.height; ++row)
		{
			for (auto col = 0; col < aImageOut.width; ++col)
			{
				aImageOut(row, col) = static_cast<uint16>(aImageIn.mono8(row, col)) * SCALE_UP;
			}
		}

		success = true;
	}

	return success;
}

bool convertBGR8ToMono16(const Common::Entity::Image& aImageIn, Common::Entity::ImageMono16& aImageOut)
{
	bool success = false;

	if (aImageIn.encoding == ImageEncoding::BGR8)
	{
		aImageOut.timestamp = aImageIn.timestamp;
		aImageOut.height = aImageIn.height;
		aImageOut.width = aImageIn.width;
		aImageOut.seq = aImageIn.seq;
		aImageOut.exposureTime = aImageIn.exposureTime;
		aImageOut.data.resize(aImageOut.height * aImageOut.width);

		static const float SCALE_UP = 255.0f;
		for (auto row = 0; row < aImageOut.height; ++row)
		{
			for (auto col = 0; col < aImageOut.width; ++col)
			{
				const float intensity = B_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 0)) +
										G_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 1)) +
										R_WEIGHT * static_cast<float>(aImageIn.rgb8(row, col, 2));
				aImageOut(row, col) = static_cast<uint16>(SCALE_UP * intensity);
			}
		}

		success = true;
	}

	return success;
}

void setupCommonImage(Common::Entity::Image& msg, const int camId, const cv::Size imgSize)
{
	msg.sensorId = camId;
	// TODO: Only supports MONO8. Will need some editing to support e.g. RGB8
	msg.width = imgSize.width;
	msg.height = imgSize.height;
	msg.step = msg.width;
	msg.encoding = ImageEncoding::MONO8;
	msg.data.resize((size_t)msg.height * msg.width);
}

void cvMatToImage(const cv::Mat& img, const Common::Entity::TimeStamp& timestamp,
	const Common::Entity::TimeStamp& hwTimestamp, Common::Entity::Image& msg)
{
	msg.timestamp = timestamp;
	msg.hardwareTimestamp = hwTimestamp;
	msg.seq++;

	const int imageHeight = img.rows;
	const int imageWidth = img.cols;
	const int nChannels = img.channels();
	CV_Assert(nChannels == 1);

	const int cachedHeight = msg.height;
	const int cachedChannels = msg.gain;
	const int cachedWidth = msg.width / cachedChannels;

	// Check if things are setup properly
	// We accept only uint8_t images for now. You will have to modify this for RGB8 and other types!!
	CV_Assert(CV_MAT_DEPTH(img.type()) == CV_8U);
	if (nChannels != cachedChannels || imageHeight != cachedHeight || imageWidth != cachedWidth)
	{
		std::cerr << "Input image: " << nChannels << "channels, WxH=" << imageWidth << ", " << imageHeight;
		std::cerr << "Cached image: " << cachedChannels << "channels, WxH=" << cachedWidth << ", " << cachedHeight;
		setupCommonImage(msg, msg.sensorId, cv::Size(imageWidth, imageHeight));
	}

	// Copy into the image
	std::memcpy(msg.data.data(), img.data, sizeof(uint8_t) * imageWidth * imageHeight * nChannels);
}

void imageToTimeSurface(const Common::Entity::Image& msg, cv::Mat& img, bool singleChannel)
{
	const int imageHeight = msg.height;
	const int nChannels = singleChannel ? 1 : msg.gain;
	const int imageWidth = msg.width / nChannels;

	// Same thing: ser/de to other data widths, not just MONO8
	img = cv::Mat::zeros({imageWidth, imageHeight}, CV_MAKETYPE(CV_8U, nChannels));

	// Copy into the image
	std::memcpy(img.data, msg.data.data(), sizeof(uint8_t) * imageHeight * imageWidth * nChannels);
}

} // namespace Algorithm
} // namespace Common
