#ifndef GMD_ENUM_H
#define GMD_ENUM_H
/**
	\file GmdEnum.h
	\brief GmdEnum.h defines all common Enums used in GMD framework

	$LastChangedBy$
	$LastChangedRevision$
	$LastChangedDate$
	*/

enum class SegmentPriState /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	INITIALISE = 3,
	READY = 12,
	DEGRADED = 48,
	FAILURE = 192,
	SHUTDOWN = 255
};

enum class ConvertResult /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	SUCCESS = 1,
	FAIL_OUT_OF_BOUNDS = 2
};

/// Segment System Failure SubStates Enum
enum class SegmentStateFailure /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	SegmentStateFailure_Invalid = 0,
	SegmentStateFailure_Failure = 1,
	SegmentStateFailure_Timeout = 2,
	SegmentStateFailure_OutOfDateTimeStamp = 3,
	// SegmentStateFailure_OutOfSyncTimeStamp = 5,
	SegmentStateFailure_Last = 4
};

//==================== /Enums used by Sys Admin Daemin - start =======================

enum class DaemonCommand /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	START = 3,
	RESTART = 12,
	SHUTDOWN = 48,
	SHUTDOWN_DAEMON = 51,
	SHUTDOWN_MACHINE = 63,
	OFFLINE = 192,
	TIMESYNC = 204
};

enum class DaemonStatus /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	STARTING = 3,
	RESTARTING = 12,
	SHUTTING_DOWN = 48,
	RUNNING = 255
};
//==================== /Enums used by Sys Admin Daemin - end =======================

//==================== /Enums used by GMD Base - start =======================

enum class InitStatus /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	INIT_WAITING = 1,
	INIT_COMPLETE = 2,
	INIT_FAILURE = 3
};

enum class InterfaceStatus /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	// 0-15 are critical
	GENERAL_ERROR = 0,
	LOST_CONNECTION = 1,
	TIME_OUT = 2,
	MIDDLEWARE_UNABLE_TO_SETUP_READER = 3,
	MIDDLEWARE_UNABLE_TO_SETUP_WRITER = 4,
	SECONDARY_THREADS_TIMED_OUT = 5,

	// 16-31 are warnings
	GENERAL_WARNING = 16,
	INPUT_VALIDATION_FAILED = 17,
	OUTPUT_VALIDATION_FAILED = 18
};

enum class BaseWarning /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	READY__INTERFACE_WARNING = 0,
	READY__MODULE_SLOW = 1,
	READY__REDUCED_REVERSE_MOBILITY = 6,
	READY__REDUCED_FORWARD_MOBILITY = 7
};

enum class BaseError /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	INIT__CONFIG_FILE_READ_FAIL = 0,
	INIT__PLATFORM_TYPE_CONFIG_MISMATCH = 1,
	INIT__INTERFACE_SETUP_FAIL = 2,
	READY__INTERFACE_ERROR = 3,
	READY__THREAD_NOT_ALIVE = 4,
	READY__LOGIC_EXCEPTION = 5
};

enum class ModuleState /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	STATE_INIT = 3,
	STATE_READY = 12,
	STATE_FAILURE = 192,
	STATE_SHUTDOWN = 255
};

enum class PlatformControlType /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	ACKERMANN = 1,
	DIFFERENTIAL = 2,

	FOUR_LEGGED = 11,

	QUADCOPTER = 21,
	HEXACOPTER = 22,
	FIXED_WING = 23,
};

enum class PlatformType /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	UAV = 1,
	UGV = 2,
	USV = 3,
	QUGV = 4,
	SSUGV = 5,
	HUMANOID = 6,
	WQUGV = 7,
	OCU = 255
};

//==================== /Enums used by GMD Base -end  =======================

#endif // GMD_ENUM_H
