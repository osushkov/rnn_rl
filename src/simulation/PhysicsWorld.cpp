
#include "PhysicsWorld.hpp"

using namespace simulation;

static constexpr float GRAVITY = -9.8f;

static constexpr float MARGIN = 0.001f;
static constexpr float RESTITUTION = 0.01f;
static constexpr float LINEAR_DAMPING = 0.0f;
static constexpr float ANGULAR_DAMPING = 0.0f;

static constexpr float WALL_HEIGHT = 10.0f;
static constexpr float WALL_MASS = 1000.0f;

struct Wall {
  uptr<btDefaultMotionState> motionState;
  uptr<btCollisionShape> shape;
  uptr<btRigidBody> body;

  Wall(float xPos, float groundHeight) {
    float dir = xPos < 0.0f ? -1.0f : 1.0f;
    motionState = make_unique<btDefaultMotionState>(btTransform(
        btQuaternion(0.0f, 0.0f, 0.0f, 1.0f),
        btVector3(xPos + dir * WALL_HEIGHT / 2.0, groundHeight + WALL_HEIGHT / 2.0, 0.0f)));

    shape =
        make_unique<btBoxShape>(btVector3(WALL_HEIGHT / 2.0, WALL_HEIGHT / 2.0, WALL_HEIGHT / 2.0));
    shape->setMargin(btScalar(MARGIN));

    btVector3 boxInertia(0.0f, 0.0f, 0.0f);
    shape->calculateLocalInertia(WALL_MASS, boxInertia);

    body = make_unique<btRigidBody>(btRigidBody::btRigidBodyConstructionInfo(
        WALL_MASS, motionState.get(), shape.get(), boxInertia));
    body->setFriction(1.0f);
    body->setRestitution(RESTITUTION);
    body->setDamping(LINEAR_DAMPING, ANGULAR_DAMPING);
  }
};

struct PhysicsWorld::PhysicsWorldImpl {
  uptr<btDefaultCollisionConfiguration> collisionConf;
  uptr<btCollisionDispatcher> dispatcher;
  uptr<btBroadphaseInterface> overlappingPairCache;
  uptr<btSequentialImpulseConstraintSolver> solver;
  uptr<btDiscreteDynamicsWorld> dynamicsWorld;

  uptr<btStaticPlaneShape> groundPlane;
  uptr<btDefaultMotionState> groundMotionState;
  uptr<btRigidBody> groundRigidBody;

  uptr<Wall> leftWall;
  uptr<Wall> rightWall;

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

    addGroundPlane();
    addWalls();
  }

  void addGroundPlane(void) {
    groundPlane->setMargin(btScalar(0.001f));
    groundRigidBody->setFriction(0.7f);
    dynamicsWorld->addRigidBody(groundRigidBody.get());
  }

  void addWalls(void) {
    leftWall = make_unique<Wall>(-100.0f, 0.0f);
    rightWall = make_unique<Wall>(100.0f, 0.0f);

    dynamicsWorld->addRigidBody(leftWall->body.get());
    dynamicsWorld->addRigidBody(leftWall->body.get());
  }

  void Step(float seconds) { dynamicsWorld->stepSimulation(seconds, 1, seconds); }

  btDiscreteDynamicsWorld *GetWorld(void) { return dynamicsWorld.get(); }

  void Render(renderer::Renderer *renderer) {
    renderer->DrawLine(Vector2(-500.0f, 0.0f), Vector2(500.0f, 0.0f));
    renderWall(renderer, leftWall.get());
    renderWall(renderer, rightWall.get());
  }

  void renderWall(renderer::Renderer *renderer, Wall *wall) {
    btTransform trans;
    wall->body->getMotionState()->getWorldTransform(trans);

    Vector2 wallSize(WALL_HEIGHT / 2.0f, WALL_HEIGHT / 2.0f);
    Vector2 wallPos(trans.getOrigin().getX(), trans.getOrigin().getY());
    renderer->DrawRectangle(wallSize, wallPos);
  }
};

PhysicsWorld::PhysicsWorld() : impl(new PhysicsWorldImpl()) {}

PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::Step(float seconds) { impl->Step(seconds); }

btDiscreteDynamicsWorld *PhysicsWorld::GetWorld(void) { return impl->GetWorld(); }

void PhysicsWorld::Render(renderer::Renderer *renderer) { impl->Render(renderer); }
