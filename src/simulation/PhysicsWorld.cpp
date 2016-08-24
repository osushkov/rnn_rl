
#include "PhysicsWorld.hpp"

using namespace simulation;

static constexpr float GRAVITY = -9.8f;
static constexpr float MARGIN = 0.001f;

struct PhysicsWorld::PhysicsWorldImpl {
  uptr<btDefaultCollisionConfiguration> collisionConf;
  uptr<btCollisionDispatcher> dispatcher;
  uptr<btBroadphaseInterface> overlappingPairCache;
  uptr<btSequentialImpulseConstraintSolver> solver;
  uptr<btDiscreteDynamicsWorld> dynamicsWorld;

  uptr<btStaticPlaneShape> groundPlane;
  uptr<btDefaultMotionState> groundMotionState;
  uptr<btRigidBody> groundRigidBody;

  PhysicsWorldImpl()
      : collisionConf(new btDefaultCollisionConfiguration()),
        dispatcher(new btCollisionDispatcher(collisionConf.get())),
        overlappingPairCache(new btDbvtBroadphase()),
        solver(new btSequentialImpulseConstraintSolver()),
        dynamicsWorld(new btDiscreteDynamicsWorld(dispatcher.get(), overlappingPairCache.get(),
                                                  solver.get(), collisionConf.get())),
        groundPlane(new btStaticPlaneShape(btVector3(0.0f, 1.0f, 0.0f), btScalar(0.0f))),
        groundMotionState(new btDefaultMotionState(
            btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, 0.0f, 0.0f)))),
        groundRigidBody(new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(
            0, groundMotionState.get(), groundPlane.get(), btVector3(0.0f, 0.0f, 0.0f)))) {

    dynamicsWorld->setGravity(btVector3(0.0f, GRAVITY, 0.0f));

    // Add a ground plane
    groundPlane->setMargin(btScalar(0.001f));
    groundRigidBody->setFriction(0.7f);
    dynamicsWorld->addRigidBody(groundRigidBody.get());
  }

  void Step(void) {}

  btDiscreteDynamicsWorld *GetWorld(void) { return nullptr; }
};

PhysicsWorld::PhysicsWorld() : impl(new PhysicsWorldImpl()) {}

PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::Step(void) { impl->Step(); }

btDiscreteDynamicsWorld *PhysicsWorld::GetWorld(void) { return impl->GetWorld(); }
