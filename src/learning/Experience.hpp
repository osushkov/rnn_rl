#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "../simulation/State.hpp"
#include "../simulation/Action.hpp"
#include <cstdlib>
#include <vector>

using namespace simulation;

namespace learning {

struct ExperienceMoment {
  EVector observedState;
  Action actionTaken;
  float reward;

  ExperienceMoment() = default;
  ExperienceMoment(const EVector &observedState, const Action &actionTaken, float reward)
      : observedState(observedState), actionTaken(actionTaken), reward(reward) {}
};

struct Experience {
  vector<ExperienceMoment> moments;

  Experience() = default;
  Experience(const vector<ExperienceMoment> &moments) : moments(moments) {}
};
}
