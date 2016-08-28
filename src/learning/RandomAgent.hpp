
#pragma once

#include "Agent.hpp"
#include <vector>

using namespace std;

namespace learning {

class RandomAgent : public Agent {
public:
  Action SelectAction(const State *state) override {
    auto actions = state->AvailableActions();
    assert(actions.size() > 0);

    return Action::ACTION(actions[rand() % actions.size()]);
  }
};
}
