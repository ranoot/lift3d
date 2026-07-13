#include <string>
#include "GMDBase/Entity/TimeStamp.h"

namespace Common::Entity {

  // Dummy point initialization (idk what is supposed to be here)
  template <typename T>
    struct Point2D {
      T x;
      T y;
    };

  // NodeId of (0,0) is NULL
  struct EAINodeId{
    uint16_t vehicleId;
    Common::Entity::TimeStamp ident;
  };

  struct EAIRoomObject{
    Common::Entity::EAINodeId objectId; // unique id for each object
    Common::Entity::EAINodeId roomId; // this can be NULL, will be assigned later.
    std::string label;
    std::string directionContent;
    std::vector<Common::Entity::Point2D<float> > polygon;
    Common::Entity::Point2D<float> centroid;
  };
}
