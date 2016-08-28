#pragma once

#include "../common/Common.hpp"
#include "../renderer/Renderer.hpp"
#include <btBulletDynamicsCommon.h>

namespace simulation {

class PhysicsWorld {
public:
  PhysicsWorld();
  ~PhysicsWorld();

  void Step(float seconds);
  btDiscreteDynamicsWorld *GetWorld(void);

  void Render(renderer::Renderer *renderer);

private:
  struct PhysicsWorldImpl;
  uptr<PhysicsWorldImpl> impl;
};
}
