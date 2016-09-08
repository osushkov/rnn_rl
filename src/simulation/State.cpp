
#include "State.hpp"
#include "Action.hpp"
#include <cassert>
#include <cmath>
#include <ostream>

using namespace simulation;

State::State() : cartXPos(0.0f), pendulumXPos(0.0f), pendulumYPos(0.0f), hingeAngle(0.0f) {}

State::State(float cartXPos, float pendulumXPos, float pendulumYPos, float hingeAngle)
    : cartXPos(cartXPos), pendulumXPos(pendulumXPos), pendulumYPos(pendulumYPos),
      hingeAngle(hingeAngle) {}

bool State::operator==(const State &other) const {
  return fabsf(cartXPos - other.cartXPos) < 0.0001f &&
         fabsf(pendulumXPos - other.pendulumXPos) < 0.0001f &&
         fabsf(pendulumYPos - other.pendulumYPos) < 0.0001f &&
         fabsf(hingeAngle - other.hingeAngle) < 0.00001f;
}

float State::GetCartXPos(void) const { return cartXPos; }

float State::GetPendulumXPos(void) const { return pendulumXPos; }

float State::GetPendulumYPos(void) const { return pendulumYPos; }

float State::GetHingeAngle(void) const { return hingeAngle; }

vector<unsigned> State::AvailableActions(void) const {
  vector<unsigned> result(Action::NUM_ACTIONS());
  for (unsigned i = 0; i < result.size(); i++) {
    result[i] = i;
  }
  return result;
}

EVector State::Encode(void) const {
  EVector result(3);
  result(0) = GetCartXPos() / 100.0f;
  // result(1) = GetHingeAngle();
  result(1) = GetPendulumXPos() / 100.0f;
  result(2) = GetPendulumYPos() / 100.0f;
  return result;
}

std::ostream &operator<<(std::ostream &stream, const simulation::State &gs) {
  stream << "cart: " << gs.GetCartXPos() << endl;
  stream << "pendulumX: " << gs.GetPendulumXPos() << endl;
  stream << "pendulumY: " << gs.GetPendulumYPos() << endl;
  stream << "hinge: " << gs.GetHingeAngle() << endl;
  return stream;
}
