
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "Action.hpp"
#include <array>
#include <iosfwd>
#include <vector>

namespace simulation {

class State {
  float cartXPos;
  float pendulumXPos;
  float pendulumYPos;

public:
  State();
  State(float cartXPos, float pendulumXPos, float pendulumYPos);
  State(const State &other);

  State &operator=(const State &other);
  bool operator==(const State &other) const;
  size_t HashCode() const;

  float GetCartXPos(void) const;
  float GetPendulumXPos(void) const;
  float GetPendulumYPos(void) const;

  // Returns indices into the GameAction::ALL_ACTIONS vector.
  vector<unsigned> AvailableActions(void) const;

  EVector Encode(void) const;
};
}

std::ostream &operator<<(std::ostream &stream, const simulation::State &gs);
