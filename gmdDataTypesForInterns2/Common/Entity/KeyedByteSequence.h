#ifndef KEYED_BYTE_SEQUENCE_H
#define KEYED_BYTE_SEQUENCE_H

/**
	\file KeyedByteSequence.h
	\see Common::Entity::KeyedByteSequence
	\class Common::Entity::KeyedByteSequence
*/

#include <vector>

#include "../../Common/ConstantsEnum.h"
#include "GMDBase/Entity/TimeStamp.h"


namespace Common
{

namespace Entity
{

// parasoft suppress item OPT-17 reason "EL-SUPP-15: Pure Entity class."

struct KeyedByteSequence
{
	/// \name Constructor and Destructor functions
	/// \{
	/**
		Default constructor that initializes all data fields to 0
	*/
	KeyedByteSequence() :
		timestamp(Common::Time::now()),
		sourceSegmentId(0U),
		destinationSegmentId(0U),
		byteSequence()
	{
	}

	explicit KeyedByteSequence(const unsigned short aSourceSegmentId, const unsigned short aDestinationSegmentId) :
		timestamp(Common::Time::now()),
		sourceSegmentId(aSourceSegmentId),
		destinationSegmentId(aDestinationSegmentId),
		byteSequence()
	{
	}
	/// \}

	/// the timestamp of the data
	Common::Entity::TimeStamp timestamp;

	/// source/target UGV segment
	unsigned short sourceSegmentId;
	unsigned short destinationSegmentId;

	std::vector<unsigned char> byteSequence;

}; // class KeyedByteSequence

} // namespace Entity

} // namespace Common

#endif // KEYED_BYTE_SEQUENCE_H