
#include "PhysicsWorld.hpp"

using namespace simulation;

struct PhysicsWorld::PhysicsWorldImpl {
  PhysicsWorldImpl() {}

  void Step(void) {}

  btDiscreteDynamicsWorld *GetWorld(void) { return nullptr; }
};

PhysicsWorld::PhysicsWorld() : impl(new PhysicsWorldImpl()) {}

PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::Step(void) { impl->Step(); }

btDiscreteDynamicsWorld *PhysicsWorld::GetWorld(void) { return impl->GetWorld(); }
