#ifndef KEYED_BYTE_ID_SEQUENCE_H
#define KEYED_BYTE_ID_SEQUENCE_H

/**
	\file KeyedByteIdSequence.h
	\see Common::Entity::KeyedByteIdSequence
	\class Common::Entity::KeyedByteIdSequence
	\brief The KeyedByteIdSequence entity class containing video request & stream.
*/

#include <vector>

#include "../../Common/ConstantsEnum.h"
#include "GMDBase/Entity/TimeStamp.h"


namespace Common
{

namespace Entity
{

// parasoft suppress item OPT-17 reason "EL-SUPP-15: Pure Entity class."

struct KeyedByteIdSequence
{
	/// \name Constructor and Destructor functions
	/// \{
	/**
		Default constructor that initializes all data fields to 0
	*/
	KeyedByteIdSequence() :
		timestamp(Common::Time::now()),
		sourceSegmentId(0U),
		destinationSegmentId(0U),
		deviceId(0U),
		dataSensorFeedType(DataSensorFeedType::NOT_APPLICABLE),
		byteSequence()
	{
	}

	/**
	Constructor that allows a KeyedByteIdSequence to be initialized by values
	\param[in] aSourceSegmentId
	\param[in] aDestinationSegmentId
	\param[in] aDeviceId
	*/
	explicit KeyedByteIdSequence(const unsigned short aSourceSegmentId, const unsigned short aDestinationSegmentId,
		const unsigned char aDeviceId, DataSensorFeedType aDataSensorFeedType) :
		timestamp(Common::Time::now()),
		sourceSegmentId(aSourceSegmentId),
		destinationSegmentId(aDestinationSegmentId),
		deviceId(aDeviceId),
		dataSensorFeedType(aDataSensorFeedType),
		byteSequence()
	{
	}
	/// \}

	/// the timestamp of the data
	Common::Entity::TimeStamp timestamp;

	/// source/target UGV segment
	unsigned short sourceSegmentId;
	unsigned short destinationSegmentId;

	/// \name device id (e.g. Last byte of camera ip address)
	/// \{
	unsigned char deviceId;
	DataSensorFeedType dataSensorFeedType;
	/// \}

	/// \name video byte stream
	/// \{
	std::vector<unsigned char> byteSequence;
	/// \}

}; // class KeyedByteIdSequence

} // namespace Entity

} // namespace Common

#endif // KEYED_BYTE_ID_SEQUENCE_H