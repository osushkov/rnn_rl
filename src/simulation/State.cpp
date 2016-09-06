
#include "State.hpp"
#include "Action.hpp"
#include <cassert>
#include <cmath>
#include <ostream>

using namespace simulation;

State::State() : cartXPos(0.0f), pendulumXPos(0.0f), pendulumYPos(0.0f) {}

State::State(float cartXPos, float pendulumXPos, float pendulumYPos)
    : cartXPos(cartXPos), pendulumXPos(pendulumXPos), pendulumYPos(pendulumYPos) {}

State::State(const State &other)
    : cartXPos(other.cartXPos), pendulumXPos(other.pendulumXPos), pendulumYPos(other.pendulumYPos) {
}

State &State::operator=(const State &other) {
  this->cartXPos = other.cartXPos;
  this->pendulumXPos = other.pendulumXPos;
  this->pendulumYPos = other.pendulumYPos;
  return *this;
}

bool State::operator==(const State &other) const {
  return fabsf(cartXPos - other.cartXPos) < 0.0001f &&
         fabsf(pendulumXPos - other.pendulumXPos) < 0.0001f &&
         fabsf(pendulumYPos - other.pendulumYPos) < 0.0001f;
}

size_t State::HashCode(void) const {
  return static_cast<size_t>(cartXPos * 378551) + static_cast<size_t>(pendulumXPos * 1999) +
         static_cast<size_t>(pendulumYPos);
}

float State::GetCartXPos(void) const { return cartXPos; }

float State::GetPendulumXPos(void) const { return pendulumXPos; }

float State::GetPendulumYPos(void) const { return pendulumYPos; }

vector<unsigned> State::AvailableActions(void) const {
  vector<unsigned> result(Action::NUM_ACTIONS());
  for (unsigned i = 0; i < result.size(); i++) {
    result[i] = i;
  }
  return result;
}

EVector State::Encode(void) const {
  EVector result(3);
  result(0) = GetCartXPos();
  result(1) = GetPendulumXPos();
  result(2) = GetPendulumYPos();
  return result;
}

std::ostream &operator<<(std::ostream &stream, const simulation::State &gs) {
  stream << "cart: " << gs.GetCartXPos() << endl;
  stream << "pendulumX: " << gs.GetPendulumXPos() << endl;
  stream << "pendulumY: " << gs.GetPendulumYPos() << endl;
  return stream;
}
