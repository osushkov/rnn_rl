#pragma once

#include "../math/Vector2.hpp"

namespace renderer {

class Renderer {
public:
  virtual void SwapBuffers(void) = 0;

  virtual void DrawCircle(const Vector2 &pos, float radius) = 0;
  virtual void DrawRectangle(const Vector2 &halfExtents, const Vector2 &pos) = 0;
  virtual void DrawLine(const Vector2 &start, const Vector2 &end) = 0;
};
}
