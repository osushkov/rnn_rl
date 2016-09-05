
#include "ExperienceGenerator.hpp"
#include "../simulation/Action.hpp"
#include "../simulation/Cart.hpp"
#include "../simulation/PhysicsWorld.hpp"
#include <vector>

using namespace learning;
using namespace simulation;

static constexpr float CART_WEIGHT_KG = 10.0f;
static constexpr float PENDULUM_LENGTH = 50.0f;
static constexpr float PENDULUM_WEIGHT_KG = 2.0f;
static const CartSpec CART_SPEC(CART_WEIGHT_KG, PENDULUM_LENGTH, PENDULUM_WEIGHT_KG);

static constexpr float STEP_LENGTH_SECS = 1.0f / 30.0f;
static constexpr unsigned STEPS_PER_ACTION = 30;

static constexpr float HINGE_ANGLE_THRESHOLD = 60.0f * M_PI / 180.0f;
static constexpr float PENALTY = -1.0f;

static constexpr float PENDULUM_WIND_STDDEV = 1.0f;

static constexpr unsigned MAX_TRACE_LENGTH = 50;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  uptr<PhysicsWorld> world;
  uptr<Cart> cart;

  ExperienceGeneratorImpl()
      : world(new PhysicsWorld()), cart(new Cart(CART_SPEC, world->GetWorld())) {}

  vector<ExperienceMoment> GenerateTrace(Agent *agent) {
    assert(agent != nullptr);

    vector<ExperienceMoment> result;
    result.reserve(MAX_TRACE_LENGTH);

    cart->Reset(0.0f);
    for (unsigned i = 0; i < MAX_TRACE_LENGTH; i++) {
      State observedState(cart->GetCartXPos(), cart->GetPendulumX(), cart->GetPendulumY());
      Action performedAction = agent->SelectAction(&observedState);

      cart->ApplyCartImpulse(performedAction.GetImpulse());

      for (unsigned j = 0; j < STEPS_PER_ACTION; j++) {
        cart->ApplyPendulumImpulse(getRandomPendulumImpulse());
        world->Step(STEP_LENGTH_SECS);
      }

      bool thresholdExceeded = fabsf(cart->GetHingeAngle()) > HINGE_ANGLE_THRESHOLD;
      float reward = thresholdExceeded ? PENALTY : 0.0f;

      result.emplace_back(observedState.Encode(), performedAction, reward);
      if (thresholdExceeded) {
        break;
      }
    }

    return result;
  }

  float getRandomPendulumImpulse(void) { return math::GaussianSample(0.0f, PENDULUM_WIND_STDDEV); }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

vector<ExperienceMoment> ExperienceGenerator::GenerateTrace(Agent *agent) {
  return impl->GenerateTrace(agent);
}
