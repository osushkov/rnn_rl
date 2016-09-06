
#pragma once

#include <cmath>
#include <ostream>
#include <vector>

namespace simulation {

class Action {
  float pushImpulse;

public:
  static unsigned NUM_ACTIONS(void);

  static Action ACTION(unsigned index);
  static unsigned ACTION_INDEX(const Action &ga);

  Action() = default; // useful to have a no args constructor
  explicit Action(float pushImpulse) : pushImpulse(pushImpulse) {}

  inline float GetImpulse(void) const { return pushImpulse; }

  inline bool operator==(const Action &other) const {
    return fabsf(pushImpulse - other.pushImpulse) < 0.0001f;
  }

  inline size_t HashCode(void) const { return static_cast<size_t>(pushImpulse * 378551); }
};
}

inline std::ostream &operator<<(std::ostream &stream, const simulation::Action &ga) {
  stream << "action_impulse( " << ga.GetImpulse() << " )";
  return stream;
}
