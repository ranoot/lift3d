/// @purpose
/// @author
/// @amendment history
/// @functions
/// @global data

#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <chrono>
#include <string>

#include "../SystemTypeDef.h"

namespace Common
{

namespace Entity
{
// time in the system shall be using system_clock as this can be time-sync across computers over the
// network. high_resolution_clock and steady_clock is not used as it cannot be time-sync and resets
// upon computer shutdown.
typedef std::chrono::system_clock TimeClock; /* parasoft-suppress  MISRA2008-0_1_5 "False alarm" */
typedef TimeClock::duration TimeDuration;
typedef TimeClock::time_point TimeStamp;
typedef TimeClock::rep TimeRep; /* parasoft-suppress  MISRA2008-0_1_5 "Typedef to provide completeness" */

} // namespace Entity

namespace Time
{

// gets the current timestamp
Entity::TimeStamp now();

// converts a TimeStamp object to a long long type
int64 toDDSTimeStamp(const Entity::TimeStamp& aTimeStamp) noexcept;

// converts a long long type to a TimeStamp object
Entity::TimeStamp fromDDSTimeStamp(const int64 aDDSTimeStamp);

// converts a TimeDuration object to a long long type
int64 toDDSTimeDuration(const Entity::TimeDuration& aTimeDuration) noexcept;

// converts a long long type to a TimeDuration object
Entity::TimeDuration fromDDSTimeDuration(const int64 aDDSTimeDuration);

// convenience function casts a duration to seconds
// *** IMPT ***: note that this function truncates the underlying duration type
// to return it as std::chrono::seconds
std::chrono::seconds toSeconds(const Entity::TimeDuration& aTimeDuration);

// convenience function casts a duration to milliseconds
// *** IMPT ***: note that this function truncates the underlying duration type
// to return it as std::chrono::milliseconds
std::chrono::milliseconds toMilliSec(const Entity::TimeDuration& aTimeDuration);

// convenience function casts a duration to microseconds
// *** IMPT ***: note that this function truncates the underlying duration type
// to return it as std::chrono::microseconds
std::chrono::microseconds toMicroSec(const Entity::TimeDuration& aTimeDuration);

// convenience function casts a duration to nanoseconds
// Note: might not be useful as the underlying duration type stores
// each tick as 100 nanoseconds
std::chrono::nanoseconds toNanoSec(const Entity::TimeDuration& aTimeDuration);

// convenience function casts a double in seconds to a TimeDuration
// Note: underlying duration type stores each tick in nanoseconds
Entity::TimeDuration durationInSeconds(const double& aDurationInSeconds);

// convenience function casts a double in milliseconds to a TimeDuration
// Note: underlying duration type stores each tick in nanoseconds
Entity::TimeDuration durationInMilliseconds(const double& aDurationInMilliseconds);

// convenience function casts a double in microseconds to a TimeDuration
// Note: underlying duration type stores each tick in nanoseconds
Entity::TimeDuration durationInMicroseconds(const double& aDurationInMicroseconds);

// convenience function casts an int in nanoseconds to a TimeDuration
// (no casting from floating point as fractional duration will be truncated anyways)
// Note: underlying duration type stores each tick in nanoseconds
Entity::TimeDuration durationInNanoseconds(const int64& aDurationInNanoseconds);

// convenience function for printing out the timestamp as a string
std::string getDateTime(const Entity::TimeStamp& aTimeStamp);

// convenience function for printing out the timestamp as a string
std::string getDateTime(const Entity::TimeStamp& aTimeStamp, const char* format);

} // namespace Time

} // namespace Common

#endif // TIMESTAMP_H
