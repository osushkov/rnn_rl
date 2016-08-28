
#include "State.hpp"
#include "Action.hpp"
#include <cassert>
#include <cmath>
#include <ostream>

using namespace simulation;

State::State() : cartXPos(0.0f), hingeAngle(0.0f) {}

State::State(float cartXPos, float hingeAngle) : cartXPos(cartXPos), hingeAngle(hingeAngle) {}

State::State(const State &other) : cartXPos(other.cartXPos), hingeAngle(other.hingeAngle) {}

State &State::operator=(const State &other) {
  this->cartXPos = other.cartXPos;
  this->hingeAngle = other.hingeAngle;
  return *this;
}

bool State::operator==(const State &other) const {
  return fabsf(cartXPos - other.cartXPos) < 0.0001f &&
         fabsf(hingeAngle - other.hingeAngle) < 0.0001f;
}

size_t State::HashCode(void) const {
  return static_cast<size_t>(cartXPos * 378551) + static_cast<size_t>(hingeAngle * 1999);
}

float State::GetCartXPos(void) const { return cartXPos; }

float State::GetHingeAngle(void) const { return hingeAngle; }

vector<unsigned> State::AvailableActions(void) const { return Action::ALL_ACTIONS(); }

std::ostream &operator<<(std::ostream &stream, const simulation::State &gs) {
  stream << "cart: " << gs.GetCartXPos() << endl;
  stream << "hinge: " << gs.GetHingeAngle() << endl;
  return stream;
}
