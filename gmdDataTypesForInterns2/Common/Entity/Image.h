#ifndef IMAGE_H
#define IMAGE_H

#include "GMDBase/Entity/TimeStamp.h"
#include <cstdint>
#include <vector>

#include "../ConstantsEnum.h"
#include "GMDBase/SystemTypeDef.h"

namespace Common
{
namespace Entity
{

// parasoft suppress item OPT-17 "EL-SUPP-15: Pure Entity class."
// parasoft suppress item MISRA2008-11_0_1 "False alarm, attribute is in a structure"
struct Image
{
	Image() :
		timestamp(Common::Time::now()),
		hardwareTimestamp(),
		sourceSegmentId(0U),
		sensorId(0),
		height(0),
		width(0),
		encoding(ImageEncoding::NOT_APPLICABLE),
		step(0),
		seq(0),
		exposureTime(0),
		gain(0),
		data()
	{
	}

	uint8& rgb8(const int32 row, const int32 col, const int32 channel)
	{
		return data[row * step + col * 3 + channel];
	}

	const uint8& rgb8(const int32 row, const int32 col, const int32 channel) const
	{
		return data[row * step + col * 3 + channel];
	}

	uint8& mono8(const int32 row, const int32 col)
	{
		return data[row * step + col];
	}

	const uint8& mono8(const int32 row, const int32 col) const
	{
		return data[row * step + col];
	}

	template <class Archive> void save(Archive& ar) const
	{
		int64 ddsTimestamp = Common::Time::toDDSTimeStamp(timestamp);
		int64 ddsHardwareTimestamp = Common::Time::toDDSTimeStamp(hardwareTimestamp);
		ar(ddsTimestamp, ddsHardwareTimestamp, sourceSegmentId, sensorId, height, width, encoding, step, seq,
			exposureTime, gain, data);
	}

	template <class Archive> void load(Archive& ar)
	{
		int64 ddsTimestamp = 0;
		int64 ddsHardwareTimestamp = 0;
		ar(ddsTimestamp, ddsHardwareTimestamp, sourceSegmentId, sensorId, height, width, encoding, step, seq,
			exposureTime, gain, data);
		timestamp = Common::Time::fromDDSTimeStamp(ddsTimestamp);
		hardwareTimestamp = Common::Time::fromDDSTimeStamp(ddsHardwareTimestamp);
	}

	Common::Entity::TimeStamp timestamp;
	Common::Entity::TimeStamp hardwareTimestamp;
	uint16 sourceSegmentId;
	int32 sensorId;
	int32 height;
	int32 width;
	ImageEncoding encoding;
	int32 step;
	int64 seq;
	int32 exposureTime; // microseconds
	int32 gain;			// dB
	std::vector<uint8> data;
};

struct RGBPixel
{
	uint8 r;
	uint8 g;
	uint8 b;
};

class RGBImage : public Image
{
	RGBPixel& operator()(const int32 row, const int32 col)
	{
		return *(reinterpret_cast<RGBPixel*>(&(data[row * step + col * 3])));
	}
};

struct ImageMono16
{
	ImageMono16() :
		timestamp(Common::Time::now()),
		height(0),
		width(0),
		seq(0),
		exposureTime(0),
		data()
	{
	}

	uint16& operator()(const int32 row, const int32 col)
	{
		return data[row * width + col];
	}

	const uint16& operator()(const int32 row, const int32 col) const
	{
		return data[row * width + col];
	}

	template <class Archive> void save(Archive& ar) const
	{
		int64 ddsTimestamp = Common::Time::toDDSTimeStamp(timestamp);
		ar(ddsTimestamp, height, width, seq, exposureTime, data);
	}

	template <class Archive> void load(Archive& ar)
	{
		int64 ddsTimestamp = 0;
		ar(ddsTimestamp, height, width, seq, exposureTime, data);
		timestamp = Common::Time::fromDDSTimeStamp(ddsTimestamp);
	}

	Common::Entity::TimeStamp timestamp;
	int32 height;
	int32 width;
	int64 seq;
	int32 exposureTime; // microseconds
	std::vector<uint16> data;
};

} // namespace Entity
} // namespace Common

#endif
