
#include "Evaluator.hpp"
#include "common/Common.hpp"
#include "learning/Constants.hpp"
#include "simulation/Action.hpp"
#include "simulation/Cart.hpp"
#include "simulation/PhysicsWorld.hpp"

using namespace learning;
using namespace simulation;

static const CartSpec CART_SPEC(CART_WEIGHT_KG, PENDULUM_LENGTH, PENDULUM_WEIGHT_KG);
static constexpr unsigned NUM_EPISODES = 100;
static constexpr float STEP_LENGTH_SECS = 1.0f / 10.0f;
static constexpr unsigned STEPS_PER_ACTION = 5;
static constexpr unsigned EPISODE_LENGTH = 30;

float Evaluator::Evaluate(Agent *agent, bool debug) {
  uptr<PhysicsWorld> world = make_unique<PhysicsWorld>();
  uptr<Cart> cart = make_unique<Cart>(CART_SPEC, world->GetWorld());
  cart->Reset(0.1f);

  float reward = 0.0f;
  for (unsigned i = 0; i < NUM_EPISODES; i++) {
    agent->ResetMemory();

    for (unsigned j = 0; j < EPISODE_LENGTH; j++) {
      State observedState(cart->GetCartXPos(), cart->GetPendulumX(), cart->GetPendulumY(),
                          cart->GetHingeAngle());
      if (debug) {
        cout << j << ": " << observedState << endl;
      }
      Action performedAction = agent->SelectAction(&observedState);

      cart->ApplyCartImpulse(performedAction.GetImpulse());

      // 0.1f * (-1.0f + 2 * (j % 2)));
      cart->ApplyPendulumImpulse(math::GaussianSample(0.0f, PENDULUM_WIND_STDDEV));

      for (unsigned k = 0; k < STEPS_PER_ACTION; k++) {
        world->Step(STEP_LENGTH_SECS);
      }

      if (fabsf(cart->GetHingeAngle()) > HINGE_ANGLE_THRESHOLD) {
        reward += PENALTY;
      }
    }

    cart->Remove(world->GetWorld());
    cart = make_unique<Cart>(CART_SPEC, world->GetWorld());
    cart->Reset(0.1f);
  }

  return reward / NUM_EPISODES;
}
