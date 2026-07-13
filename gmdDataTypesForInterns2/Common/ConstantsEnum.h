#ifndef CONSTANTS_ENUM_H
#define CONSTANTS_ENUM_H
#include "GMDBase/GmdEnum.h"

enum class ImageEncoding
{
	NOT_APPLICABLE = 0,
	RGB8 = 1,
	MONO8 = 2,
	BGR8 = 3,
	MONO16 = 4
};

enum class FrameTransformType
{
	NOT_APPLICABLE = 0,
	LOCAL_TO_LOCAL = 1,
	LOCAL_TO_SYSTEM = 2,
	LOCAL_TO_GLOBAL = 3,
	SYSTEM_TO_GLOBAL = 4
};

enum class PoseFrame /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	GLOBAL_FRAME = 1,
	LOCAL_FRAME = 26,
	SYSTEM_FRAME = 47,
	SUBMAP_FRAME = 51,
	PGL_FRAME = 63,
	EGOCENTRIC_FRAME = 251
};

enum class PoseQuality /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	GOOD = 1,
	BAD = 2
};

enum class ZFrameType /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	HEIGHT_ABOVE_FLOOR_PLANE = 27,
	NOT_USED = 255
};

enum class SemanticLabel /* parasoft-suppress  MISRA2008-7_3_1 "Common project enum declaration" */
{
	NOT_APPLICABLE = 0,
	UNLABELED = 1,
	OTHER_GROUND = 2,
	TERRAIN_GROUND = 3,
	VEGETATION = 4,
	PERSON = 5,
	SPECIAL_PERSON = 6,
	VEHICLE = 7,
	QUGV = 8,
	SSUGV = 9,
	BUILDING = 10,
	MASKED = 11,
	OBSCURANT = 12,
	END = 13,
};

#endif // CONSTANTS_ENUM_H
