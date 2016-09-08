
#include "Cart.hpp"
#include "../common/Common.hpp"
#include <btBulletDynamicsCommon.h>
#include <cassert>

using namespace simulation;

static constexpr float MARGIN = 0.001f;
static constexpr float RESTITUTION = 0.01f;
static constexpr float LINEAR_DAMPING = 0.0f;
static constexpr float ANGULAR_DAMPING = 0.0f;

static constexpr float BOX_FRICTION = 0.6f;
static constexpr float BOX_LENGTH = 30.0f;
static constexpr float BOX_HEIGHT = 1.0f;
static constexpr float BOX_DEPTH = 20.0f;

static constexpr float WHEEL_FRICTION = 1.0f;
static constexpr float WHEEL_RADIUS = 3.0f;
static constexpr float WHEEL_WIDTH = BOX_DEPTH;
static constexpr float WHEEL_MASS = 1.0f;
static constexpr float WHEEL_DAMPING = 0.75f;

struct BoxSpec {
  btVector3 halfExtents;
  float mass;

  BoxSpec(const btVector3 &halfExtents, float mass) : halfExtents(halfExtents), mass(mass) {}
};

struct Box {
  BoxSpec spec;

  uptr<btDefaultMotionState> motionState;
  uptr<btCollisionShape> shape;
  uptr<btRigidBody> body;

  Box(const BoxSpec &spec) : spec(spec) {
    motionState = make_unique<btDefaultMotionState>(
        btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, 0.0f, 0.0f)));

    shape = make_unique<btBoxShape>(spec.halfExtents);
    shape->setMargin(btScalar(MARGIN));

    btVector3 boxInertia(0.0f, 0.0f, 0.0f);
    shape->calculateLocalInertia(spec.mass, boxInertia);

    body = make_unique<btRigidBody>(btRigidBody::btRigidBodyConstructionInfo(
        spec.mass, motionState.get(), shape.get(), boxInertia));
    body->setFriction(BOX_FRICTION);
    body->setRestitution(RESTITUTION);
    body->setDamping(LINEAR_DAMPING, ANGULAR_DAMPING);
  }
};

struct WheelSpec {
  btVector3 position;
  btVector3 axle;

  float radius;
  float width;

  float mass;

  WheelSpec(const btVector3 &position, const btVector3 &axle, float radius, float width, float mass)
      : position(position), axle(axle), radius(radius), width(width), mass(mass) {}
};

struct Wheel {
  WheelSpec spec;

  uptr<btDefaultMotionState> motionState;
  uptr<btCollisionShape> shape;
  uptr<btRigidBody> body;
  uptr<btHingeConstraint> joint;

  Wheel(const WheelSpec &spec, btRigidBody *box) : spec(spec) {
    motionState = make_unique<btDefaultMotionState>(
        btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, 0.0f, 0.0f)));

    shape = make_unique<btCylinderShapeZ>(btVector3(spec.radius, spec.radius, spec.width / 2.0f));
    shape->setMargin(btScalar(0.001f));

    btVector3 wheelInertia(0.0f, 0.0f, 0.0f);
    shape->calculateLocalInertia(spec.mass, wheelInertia);

    body = make_unique<btRigidBody>(btRigidBody::btRigidBodyConstructionInfo(
        spec.mass, motionState.get(), shape.get(), wheelInertia));
    body->setFriction(WHEEL_FRICTION);
    body->setRestitution(RESTITUTION);
    body->setDamping(LINEAR_DAMPING, ANGULAR_DAMPING);

    joint = make_unique<btHingeConstraint>(*box, *body.get(), spec.position,
                                           btVector3(0.0f, 0.0f, 0.0f), spec.axle,
                                           btVector3(0.0f, 0.0f, 1.0f), true);
  }
};

struct PendulumSpec {
  float length;
  float mass;

  PendulumSpec(float length, float mass) : length(length), mass(mass) {
    assert(length > 0.0f && mass > 0.0f);
  }
};

struct Pendulum {
  PendulumSpec spec;

  uptr<btDefaultMotionState> motionState;
  uptr<btCollisionShape> shape;
  uptr<btRigidBody> body;
  uptr<btHingeConstraint> joint;

  Pendulum(const PendulumSpec &spec, btRigidBody *box) : spec(spec) {
    motionState = make_unique<btDefaultMotionState>(
        btTransform(btQuaternion(0.0f, 0.0f, 0.0f, 1.0f), btVector3(0.0f, 0.0f, 0.0f)));

    shape = make_unique<btCylinderShape>(btVector3(0.1f, spec.length / 2.0f, 0.1f));
    shape->setMargin(btScalar(0.001f));

    btVector3 pendulumInertia(0.0f, 0.0f, 0.0f);
    shape->calculateLocalInertia(spec.mass, pendulumInertia);

    body = make_unique<btRigidBody>(btRigidBody::btRigidBodyConstructionInfo(
        spec.mass, motionState.get(), shape.get(), pendulumInertia));
    body->setFriction(0.6f);
    body->setRestitution(RESTITUTION);
    body->setDamping(LINEAR_DAMPING, ANGULAR_DAMPING);

    joint = make_unique<btHingeConstraint>(
        *box, *body.get(), btVector3(0.0f, BOX_HEIGHT / 2.0f + 0.1f, 0.0f),
        btVector3(0.0f, -spec.length / 2.0f, 0.0f), btVector3(0.0f, 0.0f, 1.0f),
        btVector3(0.0f, 0.0f, 1.0f), true);
  }
};

struct Cart::CartImpl {
  CartSpec spec;

  uptr<Box> box;
  uptr<Pendulum> pendulum;
  vector<uptr<Wheel>> wheels;

  CartImpl(const CartSpec &spec, btDiscreteDynamicsWorld *pWorld) : spec(spec) {
    assert(pWorld != nullptr);

    createBox();
    createPendulum();
    createWheels();

    pWorld->addRigidBody(box->body.get());
    box->body->activate();

    pWorld->addRigidBody(pendulum->body.get());
    pWorld->addConstraint(pendulum->joint.get());
    pendulum->body->activate();

    for (auto &wheel : wheels) {
      pWorld->addRigidBody(wheel->body.get());
      pWorld->addConstraint(wheel->joint.get());
      wheel->body->activate();
    }
  }

  void createBox(void) {
    auto boxExtents = btVector3(BOX_LENGTH / 2.0f, BOX_HEIGHT / 2.0f, BOX_DEPTH / 2.0f);
    box = make_unique<Box>(BoxSpec(boxExtents, spec.cartWeightKg));
  }

  void createPendulum(void) {
    pendulum = make_unique<Pendulum>(PendulumSpec(30.0f, 2.0f), box->body.get());
  }

  void createWheels(void) {
    WheelSpec baseWheel(btVector3(0.0f, 0.0f, 0.0f), btVector3(0.0f, 0.0f, 1.0f), WHEEL_RADIUS,
                        WHEEL_WIDTH, WHEEL_MASS);

    WheelSpec wheel0 = baseWheel;
    wheel0.position = btVector3(BOX_LENGTH / 2.0f, -WHEEL_RADIUS - BOX_HEIGHT / 2.0f - 0.5f, 0.0f);
    wheels.push_back(make_unique<Wheel>(wheel0, box->body.get()));

    WheelSpec wheel1 = baseWheel;
    wheel1.position = btVector3(-BOX_LENGTH / 2.0f, -WHEEL_RADIUS - BOX_HEIGHT / 2.0f - 0.5f, 0.0f);
    wheels.push_back(make_unique<Wheel>(wheel1, box->body.get()));
  }

  void Reset(float groundHeight) {
    float boxYOffset = WHEEL_RADIUS + groundHeight - wheels[0]->spec.position.y();

    btTransform boxTransform;
    boxTransform.setIdentity();
    boxTransform.setOrigin(btVector3(0.0f, boxYOffset, 0.0f));
    box->motionState->setWorldTransform(boxTransform);
    box->body->setMotionState(box->motionState.get());
    box->body->setLinearVelocity(btVector3(0.0f, 0.0f, 0.0f));
    box->body->setAngularVelocity(btVector3(0.0f, 0.0f, 0.0f));

    btTransform pendulumTransform;
    pendulumTransform.setIdentity();
    pendulumTransform.setOrigin(btVector3(
        0.0f, boxYOffset + BOX_HEIGHT / 2.0f + pendulum->spec.length / 2.0f + 0.1f, 0.0f));
    pendulum->motionState->setWorldTransform(pendulumTransform);
    pendulum->body->setMotionState(pendulum->motionState.get());
    pendulum->body->setLinearVelocity(btVector3(0.0f, 0.0f, 0.0f));
    pendulum->body->setAngularVelocity(btVector3(0.0f, 0.0f, 0.0f));

    for (auto &wheel : wheels) {
      btTransform wheelTransform;
      wheelTransform.setIdentity();
      wheelTransform.setOrigin(btVector3(wheel->spec.position.x(),
                                         boxYOffset + wheel->spec.position.y(),
                                         wheel->spec.position.z()));
      wheel->motionState->setWorldTransform(wheelTransform);
      wheel->body->setMotionState(wheel->motionState.get());
      wheel->body->setLinearVelocity(btVector3(0.0f, 0.0f, 0.0f));
      wheel->body->setAngularVelocity(btVector3(0.0f, 0.0f, 0.0f));
    }
  }

  void Remove(btDiscreteDynamicsWorld *pWorld) {
    pWorld->removeRigidBody(box->body.get());
    pWorld->removeRigidBody(pendulum->body.get());
    pWorld->removeConstraint(pendulum->joint.get());

    for (auto &wheel : wheels) {
      pWorld->removeRigidBody(wheel->body.get());
      pWorld->removeConstraint(wheel->joint.get());
    }
  }

  void Render(renderer::Renderer *renderer) {
    renderWheels(renderer);
    renderPendulum(renderer);
    renderBody(renderer);
  }

  void renderBody(renderer::Renderer *renderer) {
    btTransform trans;
    box->body->getMotionState()->getWorldTransform(trans);

    Vector2 boxSize(BOX_LENGTH / 2.0f, BOX_HEIGHT / 2.0f);
    Vector2 boxPos(trans.getOrigin().getX(), trans.getOrigin().getY());
    renderer->DrawRectangle(boxSize, boxPos);
  }

  void renderWheels(renderer::Renderer *renderer) {
    for (auto &wheel : wheels) {
      btTransform wtrans;
      wheel->body->getMotionState()->getWorldTransform(wtrans);

      Vector2 wheelPos(wtrans.getOrigin().getX(), wtrans.getOrigin().getY());
      renderer->DrawCircle(wheelPos, WHEEL_RADIUS);
    }
  }

  void renderPendulum(renderer::Renderer *renderer) {
    btTransform trans;
    pendulum->body->getMotionState()->getWorldTransform(trans);
    float angle = pendulum->joint->getHingeAngle();

    Vector2 start(trans.getOrigin().getX() + sin(-angle) * pendulum->spec.length / 2.0f,
                  trans.getOrigin().getY() + cos(-angle) * pendulum->spec.length / 2.0f);

    Vector2 end(trans.getOrigin().getX() - sin(-angle) * pendulum->spec.length / 2.0f,
                trans.getOrigin().getY() - cos(-angle) * pendulum->spec.length / 2.0f);

    renderer->DrawLine(start, end);
    renderer->DrawCircle(start, 1.0);
  }

  void ApplyCartImpulse(float newtons) {
    btVector3 impulse(newtons, 0.0f, 0.0f);
    btVector3 pos(box->spec.halfExtents.x() * (newtons < 0.0f ? 1.0f : -1.0f), 0.0f, 0.0f);

    box->body->applyImpulse(impulse, pos);
  }

  void ApplyPendulumImpulse(float newtons) {
    btVector3 impulse(newtons, 0.0f, 0.0f);
    btVector3 pos(0.0f, pendulum->spec.length / 2.0f, 0.0f);

    pendulum->body->applyImpulse(impulse, pos);
  }

  float GetHingeAngle(void) const { return pendulum->joint->getHingeAngle(); }

  float GetCartXPos(void) const {
    btTransform trans;
    box->body->getMotionState()->getWorldTransform(trans);
    return trans.getOrigin().getX();
  }

  float GetPendulumX(void) const {
    btTransform trans;
    pendulum->body->getMotionState()->getWorldTransform(trans);
    float angle = pendulum->joint->getHingeAngle();

    return trans.getOrigin().getX() + sin(-angle) * pendulum->spec.length / 2.0f;
  }

  float GetPendulumY(void) const {
    btTransform trans;
    pendulum->body->getMotionState()->getWorldTransform(trans);
    float angle = pendulum->joint->getHingeAngle();

    return trans.getOrigin().getY() + cos(-angle) * pendulum->spec.length / 2.0f;
  }
};

Cart::Cart(const CartSpec &spec, btDiscreteDynamicsWorld *pWorld)
    : impl(new CartImpl(spec, pWorld)) {}

Cart::~Cart() = default;

void Cart::Reset(float groundHeight) { impl->Reset(groundHeight); }

void Cart::Remove(btDiscreteDynamicsWorld *pWorld) { impl->Remove(pWorld); }

void Cart::Render(renderer::Renderer *renderer) { impl->Render(renderer); }

void Cart::ApplyCartImpulse(float newtons) { impl->ApplyCartImpulse(newtons); }

void Cart::ApplyPendulumImpulse(float newtons) { impl->ApplyPendulumImpulse(newtons); }

float Cart::GetHingeAngle(void) const { return impl->GetHingeAngle(); }
float Cart::GetCartXPos(void) const { return impl->GetCartXPos(); }
float Cart::GetPendulumX(void) const { return impl->GetPendulumX(); }
float Cart::GetPendulumY(void) const { return impl->GetPendulumY(); }
