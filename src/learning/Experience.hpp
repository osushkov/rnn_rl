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
  EVector initialState;
  Action actionTaken;
  EVector successorState;
  float reward;
  bool isSuccessorTerminal;

  ExperienceMoment() = default;
  ExperienceMoment(EVector initialState, Action actionTaken, EVector successorState, float reward,
                   bool isSuccessorTerminal)
      : initialState(initialState), actionTaken(actionTaken), successorState(successorState),
        reward(reward), isSuccessorTerminal(isSuccessorTerminal) {}
};

struct Experience {
  vector<ExperienceMoment> experienceSequence;

  Experience() = default;
  Experience(const vector<ExperienceMoment> moments) : experienceSequence(moments) {}
};
}
