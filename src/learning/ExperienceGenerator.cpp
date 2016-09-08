
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
static constexpr unsigned STEPS_PER_ACTION = 5; // Agent gets to perform an action once a second.
static constexpr unsigned MAX_TRACE_LENGTH = 30;
static constexpr unsigned MIN_TRACE_LENGTH = 10;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  uptr<PhysicsWorld> world;

  ExperienceGeneratorImpl() : world(new PhysicsWorld()) {}

  Experience GenerateExperience(LearningAgent *agent) {
    assert(agent != nullptr);

    Experience result;
    result.moments.reserve(MAX_TRACE_LENGTH);

    uptr<Cart> cart = make_unique<Cart>(CART_SPEC, world->GetWorld());
    cart->Reset(0.1f);

    agent->ResetMemory();
    for (unsigned i = 0; i < MAX_TRACE_LENGTH; i++) {
      State observedState(cart->GetCartXPos(), cart->GetPendulumX(), cart->GetPendulumY(),
                          cart->GetHingeAngle());
      Action performedAction = agent->SelectLearningAction(&observedState);

      cart->ApplyCartImpulse(performedAction.GetImpulse());
      cart->ApplyPendulumImpulse(getRandomPendulumImpulse());

      for (unsigned j = 0; j < STEPS_PER_ACTION; j++) {
        world->Step(STEP_LENGTH_SECS);
      }

      bool thresholdExceeded = fabsf(cart->GetHingeAngle()) > HINGE_ANGLE_THRESHOLD;
      float reward = thresholdExceeded ? PENALTY : 0.0f;
      result.moments.emplace_back(observedState.Encode(), performedAction, reward);

      if (thresholdExceeded && i >= MIN_TRACE_LENGTH) {
        break;
      }
    }

    cart->Remove(world->GetWorld());
    summedRewards(result, REWARD_DELAY_DISCOUNT);
    return result;
  }

  void summedRewards(Experience &experience, float discount) {
    float totalReward = 0.0f;
    for (int i = experience.moments.size() - 1; i >= 0; i--) {
      totalReward += experience.moments[i].reward;
      experience.moments[i].reward = totalReward;
      totalReward *= discount;
    }
  }

  float getRandomPendulumImpulse(void) { return math::GaussianSample(0.0f, PENDULUM_WIND_STDDEV); }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

Experience ExperienceGenerator::GenerateExperience(LearningAgent *agent) {
  return impl->GenerateExperience(agent);
}
