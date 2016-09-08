#pragma once

#include "../common/Common.hpp"
#include "../renderer/Renderer.hpp"
#include <btBulletDynamicsCommon.h>
#include <cassert>

namespace simulation {

struct CartSpec {
  float cartWeightKg;
  float pendulumLength;
  float pendulumWeightKg;

  CartSpec(float cartWeightKg, float pendulumLength, float pendulumWeightKg)
      : cartWeightKg(cartWeightKg), pendulumLength(pendulumLength),
        pendulumWeightKg(pendulumWeightKg) {
    assert(cartWeightKg > 0.0f);
    assert(pendulumLength > 0.0f);
    assert(pendulumWeightKg > 0.0f);
  }
};

class Cart {
public:
  Cart(const CartSpec &spec, btDiscreteDynamicsWorld *pWorld);
  ~Cart();

  void Reset(float groundHeight);
  void Remove(btDiscreteDynamicsWorld *pWorld);
  void Render(renderer::Renderer *renderer);

  // Applies impulse in the x-axis (left or right push).
  void ApplyCartImpulse(float newtons);
  void ApplyPendulumImpulse(float newtons);

  float GetHingeAngle(void) const;
  float GetCartXPos(void) const;
  float GetPendulumX(void) const;
  float GetPendulumY(void) const;

private:
  struct CartImpl;
  uptr<CartImpl> impl;
};
}
