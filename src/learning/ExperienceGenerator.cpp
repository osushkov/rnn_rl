
#include "ExperienceGenerator.hpp"
#include "../simulation/Action.hpp"
#include "../simulation/Cart.hpp"
#include "../simulation/PhysicsWorld.hpp"
#include "Constants.hpp"
#include <vector>

using namespace learning;
using namespace simulation;

static const CartSpec CART_SPEC(CART_WEIGHT_KG, PENDULUM_LENGTH, PENDULUM_WEIGHT_KG);

static constexpr float STEP_LENGTH_SECS = 1.0f / 10.0f;
static constexpr unsigned STEPS_PER_ACTION = 10; // Agent gets to perform an action once a second.
static constexpr unsigned MAX_TRACE_LENGTH = 20;
static constexpr unsigned MIN_TRACE_LENGTH = 10;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  uptr<PhysicsWorld> world;
  uptr<Cart> cart;

  ExperienceGeneratorImpl()
      : world(new PhysicsWorld()), cart(new Cart(CART_SPEC, world->GetWorld())) {}

  Experience GenerateExperience(Agent *agent) {
    assert(agent != nullptr);

    Experience result;
    result.moments.reserve(MAX_TRACE_LENGTH);

    cart->Reset(0.0f);
    agent->ResetMemory();
    for (unsigned i = 0; i < MAX_TRACE_LENGTH; i++) {
      State observedState(cart->GetCartXPos(), cart->GetPendulumX(), cart->GetPendulumY());
      Action performedAction = agent->SelectAction(&observedState);

      cart->ApplyCartImpulse(performedAction.GetImpulse());

      for (unsigned j = 0; j < STEPS_PER_ACTION; j++) {
        cart->ApplyPendulumImpulse(getRandomPendulumImpulse());
        world->Step(STEP_LENGTH_SECS);
      }

      bool thresholdExceeded = fabsf(cart->GetHingeAngle()) > HINGE_ANGLE_THRESHOLD;
      // cout << observedState << endl << performedAction << endl;
      // cout << thresholdExceeded << endl;
      float reward = thresholdExceeded ? PENALTY : 0.0f;
      // cout << "reward: " << reward << endl;

      result.moments.emplace_back(observedState.Encode(), performedAction, reward);

      if (thresholdExceeded && i >= MIN_TRACE_LENGTH) {
        break;
      }
    }

    return result;
  }

  float getRandomPendulumImpulse(void) { return math::GaussianSample(0.0f, PENDULUM_WIND_STDDEV); }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

Experience ExperienceGenerator::GenerateExperience(Agent *agent) {
  return impl->GenerateExperience(agent);
}
