#pragma once

#include "../math/Math.hpp"
#include "../simulation/State.hpp"
#include <cstdlib>

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
}
