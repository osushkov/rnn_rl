
#include "ExperienceGenerator.hpp"
#include "../simulation/Action.hpp"
#include "../simulation/Cart.hpp"
#include "../simulation/PhysicsWorld.hpp"
#include <vector>

using namespace learning;

struct ExperienceGenerator::ExperienceGeneratorImpl {
  ExperienceGeneratorImpl() {}

  vector<Experience> GenerateTrace(Agent *agent) {
    assert(agent != nullptr);
    return vector<Experience>();
  }
};

ExperienceGenerator::ExperienceGenerator() : impl(new ExperienceGeneratorImpl()) {}

ExperienceGenerator::~ExperienceGenerator() = default;

vector<Experience> ExperienceGenerator::GenerateTrace(Agent *agent) {
  return impl->GenerateTrace(agent);
}
