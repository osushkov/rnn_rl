#pragma once

#include "../common/Common.hpp"
#include <btBulletDynamicsCommon.h>

namespace simulation {

class PhysicsWorld {
public:
  PhysicsWorld();
  ~PhysicsWorld();

  void Step(float seconds);
  btDiscreteDynamicsWorld *GetWorld(void);

private:
  struct PhysicsWorldImpl;
  uptr<PhysicsWorldImpl> impl;
};
}
