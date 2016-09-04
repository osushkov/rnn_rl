
#include "ExperienceGenerator.hpp"
#include "../simulation/Action.hpp"
#include "../simulation/Cart.hpp"
#include "../simulation/PhysicsWorld.hpp"
#include <cmath>
#include <vector>

using namespace learning;
using namespace simulation;

static constexpr float CART_WEIGHT_KG = 10.0f;
static constexpr float PENDULUM_LENGTH = 50.0f;
static constexpr float PENDULUM_WEIGHT_KG = 2.0f;
static const CartSpec CART_SPEC(CART_WEIGHT_KG, PENDULUM_LENGTH, PENDULUM_WEIGHT_KG);

static constexpr unsigned MAX_TRACE_LENGTH = 50;
static constexpr float STEP_TIME_SECS = 1.0f / 30.0f;
static constexpr unsigned DECISION_RATE = 30;

static constexpr float PERTURBATION_STD_DEV = 1.0f;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  uptr<PhysicsWorld> world;
  uptr<Cart> cart;

  ExperienceGeneratorImpl()
      : world(new PhysicsWorld()), cart(new Cart(CART_SPEC, world->GetWorld())) {}

  vector<ExperienceMoment> GenerateTrace(Agent *agent) {
    assert(agent != nullptr);

    vector<EVector> observedStates;
    vector<Action> actionsTaken;
    vector<float> rewardsReceived;

    observedStates.reserve(MAX_TRACE_LENGTH);
    actionsTaken.reserve(MAX_TRACE_LENGTH);
    rewardsReceived.reserve(MAX_TRACE_LENGTH);

    cart->Reset(0.0f);
    for (unsigned i = 0; i < MAX_TRACE_LENGTH; i++) {
      State observedState = getObservedState();
      observedStates.push_back(observedState.Encode());

      Action action = agent->SelectAction(&observedState);
      actionsTaken.push_back(action);
      applyAction(action);

      for (unsigned j = 0; j < DECISION_RATE; j++) {
        applyPerturbation();
        world->Step(STEP_TIME_SECS);
      }

      rewardsReceived.push_back(getCurrentReward());
      if (shouldStop()) {
        break;
      }
    }

    assert(observedStates.size() == actionsTaken.size());
    assert(observedStates.size() == rewardsReceived.size());

    vector<ExperienceMoment> result;
    result.reserve(observedStates.size() - 1);

    for (unsigned i = 0; i < observedStates.size() - 1; i++) {
      result.emplace_back(observedStates[i], actionsTaken[i], observedStates[i + 1],
                          rewardsReceived[i], i == observedStates.size() - 2);
    }

    return result;
  }

  bool shouldStop(void) { return fabsf(cart->GetHingeAngle()) > (60.0f * M_PI / 180.0f); }

  State getObservedState(void) {
    return State(cart->GetCartXPos(), cart->GetPendulumX(), cart->GetPendulumY());
  }

  float getCurrentReward(void) {
    return fabsf(cart->GetHingeAngle()) > (45.0f * M_PI / 180.0f) ? -1.0f : 0.0f;
  }

  void applyAction(const Action &action) { cart->ApplyCartImpulse(action.GetImpulse()); }

  void applyPerturbation(void) {
    cart->ApplyPendulumImpulse(math::GaussianSample(0.0f, PERTURBATION_STD_DEV));
  }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

vector<ExperienceMoment> ExperienceGenerator::GenerateTrace(Agent *agent) {
  return impl->GenerateTrace(agent);
}
