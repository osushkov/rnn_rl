
#pragma once

#include "../common/Common.hpp"
#include "Action.hpp"
#include <array>
#include <iosfwd>
#include <vector>

namespace simulation {

class State {
  float cartXPos;
  float hingeAngle;

public:
  State();
  State(float cartXPos, float hingeAngle);
  State(const State &other);

  State &operator=(const State &other);
  bool operator==(const State &other) const;
  size_t HashCode() const;

  float GetCartXPos(void) const;
  float GetHingeAngle(void) const;

  // Returns indices into the GameAction::ALL_ACTIONS vector.
  vector<unsigned> AvailableActions(void) const;
};
}

std::ostream &operator<<(std::ostream &stream, const simulation::State &gs);
