
#pragma once

#include "../simulation/Action.hpp"
#include "../simulation/State.hpp"

using namespace simulation;

namespace learning {

class Agent {
public:
  virtual ~Agent() = default;
  virtual Action SelectAction(const State *state) = 0;
  virtual void ResetMemory(void) = 0;
};
}
