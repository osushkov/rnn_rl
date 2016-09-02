
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

static constexpr unsigned MAX_TRACE_LENGTH = 1000;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  uptr<PhysicsWorld> world;
  uptr<Cart> cart;

  ExperienceGeneratorImpl() :
      world(new PhysicsWorld()),
      cart(new Cart(CART_SPEC, world->GetWorld())) {}

  vector<Experience> GenerateTrace(Agent *agent) {
    assert(agent != nullptr);

    vector<Experience> result;
    result.reserve(MAX_TRACE_LENGTH);

    cart->Reset(0.0f);
    for (unsigned i = 0; i < MAX_TRACE_LENGTH; i++) {

    }

    return result;
  }

  float getRandomPendulumImpulse(void) {
    return 0.0f;
  }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

vector<Experience> ExperienceGenerator::GenerateTrace(Agent *agent) {
  return impl->GenerateTrace(agent);
}
