#pragma once

#include "../common/Common.hpp"
#include "Renderer.hpp"
#include <string>

namespace renderer {

class SFMLRenderer : public Renderer {
public:
  SFMLRenderer(unsigned width, unsigned height, const string &windowName);
  ~SFMLRenderer();

  void SwapBuffers(void) override;

  void DrawCircle(const Vector2 &pos, float radius) override;
  void DrawRectangle(const Vector2 &halfExtents, const Vector2 &pos) override;
  void DrawLine(const Vector2 &start, const Vector2 &end) override;

private:
  struct SFMLRendererImpl;
  uptr<SFMLRendererImpl> impl;
};
}
