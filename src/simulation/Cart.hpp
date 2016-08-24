#pragma once

#include "../common/Common.hpp"

namespace simulation {

class Cart {
public:
  Cart(float cartWeightKg, float pendulumLength, float pendulumWeightKg);
  ~Cart();

  void Render(void);

  // Applies impulse in the x-axis (left or right push).
  void ApplyImpulse(float newtons);

private:
  struct CartImpl;
  uptr<CartImpl> impl;
};
}
