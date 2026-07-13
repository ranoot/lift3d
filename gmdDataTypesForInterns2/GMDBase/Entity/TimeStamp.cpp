/// @purpose
/// @author
/// @amendment history
/// @functions
/// @global data

#include "TimeStamp.h"

#include <array>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace Common
{

namespace Time
{

Entity::TimeStamp now()
{
	return Entity::TimeClock::now();
}

int64 toDDSTimeStamp(const Entity::TimeStamp& aTimeStamp) noexcept
{
	// Linux OS provides time up to nanoseconds
	// Windows OS provides time up to 100s of nanoseconds
	// IDL uses 100s of nanoseconds
#ifdef LINUX_OS // parasoft-suppress  MISRA2008-16_2_1_c "Required for cross
				// compilation"
	const int64 TO_100S_NANOSECOND(100);
	auto timestamp = aTimeStamp.time_since_epoch().count() / TO_100S_NANOSECOND;
#else  // parasoft-suppress  MISRA2008-16_2_1_c "Required for cross compilation"
	auto timestamp = aTimeStamp.time_since_epoch().count();
#endif /* parasoft-suppress  MISRA2008-16_2_1_c "Intended Design" */
	return timestamp;
}

Entity::TimeStamp fromDDSTimeStamp(const int64 aDDSTimeStamp)
{
	// Linux OS provides time up to nanoseconds
	// Windows OS provides time up to 100s of nanoseconds
	// IDL uses 100s of nanoseconds
#ifdef LINUX_OS // parasoft-suppress  MISRA2008-16_2_1_c "Required for cross
				// compilation"
	const int64 TO_NANOSECOND(100);
	auto timestamp = aDDSTimeStamp * TO_NANOSECOND;
#else  // parasoft-suppress  MISRA2008-16_2_1_c "Required for cross compilation"
	auto timestamp = aDDSTimeStamp;
#endif /* parasoft-suppress  MISRA2008-16_2_1_c "Intended Design" */
	return Entity::TimeStamp(Entity::TimeDuration(timestamp));
}

int64 toDDSTimeDuration(const Entity::TimeDuration& aTimeDuration) noexcept
{
	return aTimeDuration.count();
}

Common::Entity::TimeDuration fromDDSTimeDuration(const int64 aDDSTimeDuration)
{
	return Common::Entity::TimeDuration(aDDSTimeDuration);
}

std::chrono::seconds toSeconds(const Entity::TimeDuration& aTimeDuration)
{
	return std::chrono::duration_cast<std::chrono::seconds>(aTimeDuration);
}

std::chrono::milliseconds toMilliSec(const Entity::TimeDuration& aTimeDuration)
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(aTimeDuration);
}

std::chrono::microseconds toMicroSec(const Entity::TimeDuration& aTimeDuration)
{
	return std::chrono::duration_cast<std::chrono::microseconds>(aTimeDuration);
}

std::chrono::nanoseconds toNanoSec(const Entity::TimeDuration& aTimeDuration)
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(aTimeDuration);
}

Entity::TimeDuration durationInSeconds(const double& aDurationInSeconds)
{
	return std::chrono::duration_cast<Entity::TimeDuration>(std::chrono::duration<double>(aDurationInSeconds));
}

Entity::TimeDuration durationInMilliseconds(const double& aDurationInMilliseconds)
{
	return std::chrono::duration_cast<Entity::TimeDuration>(
		std::chrono::duration<double, std::milli>(aDurationInMilliseconds));
}

Entity::TimeDuration durationInMicroseconds(const double& aDurationInMicroseconds)
{
	return std::chrono::duration_cast<Entity::TimeDuration>(
		std::chrono::duration<double, std::micro>(aDurationInMicroseconds));
}

Entity::TimeDuration durationInNanoseconds(const int64& aDurationInNanoseconds)
{
	return Entity::TimeDuration(aDurationInNanoseconds);
}

std::string getDateTime(const Entity::TimeStamp& aTimeStamp)
{
	const std::chrono::system_clock::time_point timePointAccurateToSec(
		std::chrono::duration_cast<std::chrono::seconds>(aTimeStamp.time_since_epoch()));
	const auto tTimeInTimeT = std::chrono::system_clock::to_time_t(timePointAccurateToSec);
	const std::tm tm = *std::localtime(&tTimeInTimeT);
	std::ostringstream tStringStream;
	const int32 FIELD_WIDTH(9);
	const auto tTimeAccurateToSec = std::chrono::system_clock::from_time_t(tTimeInTimeT);
	tStringStream << std::put_time(&tm, "%Y-%b-%d %H:%M:%S") << "." << std::setw(FIELD_WIDTH)
				  << std::chrono::nanoseconds(aTimeStamp - tTimeAccurateToSec).count();

	return tStringStream.str();
}

std::string getDateTime(const Entity::TimeStamp& aTimeStamp, const char* format)
{
	const std::chrono::system_clock::time_point timePointAccurateToSec(
		std::chrono::duration_cast<std::chrono::seconds>(aTimeStamp.time_since_epoch()));
	const auto tTimeInTimeT = std::chrono::system_clock::to_time_t(timePointAccurateToSec);
	const std::tm tm = *std::localtime(&tTimeInTimeT);
	std::ostringstream tStringStream;
	const int32 FIELD_WIDTH(9);
	const auto tTimeAccurateToSec = std::chrono::system_clock::from_time_t(tTimeInTimeT);
	tStringStream << std::put_time(&tm, format) << "." << std::setw(FIELD_WIDTH)
				  << std::chrono::nanoseconds(aTimeStamp - tTimeAccurateToSec).count();

	return tStringStream.str();
}

} // namespace Time

} // namespace Common
