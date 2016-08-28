
#include "SFMLRenderer.hpp"
#include <SFML/Graphics.hpp>
#include <cassert>

using namespace renderer;

struct SFMLRenderer::SFMLRendererImpl {
  sf::RenderWindow window;
  sf::View view;

  SFMLRendererImpl(unsigned width, unsigned height, const string &windowName)
      : window(sf::VideoMode(width, height), windowName),
        view(sf::Vector2f(0, 0), sf::Vector2f(400, 200)) {
    assert(width > 0 && height > 0);

    view.rotate(180.0f);
    window.setView(view);
    window.clear();
  }

  void SwapBuffers(void) {
    window.display();
    window.clear();
  }

  void DrawCircle(const Vector2 &pos, float radius) {
    assert(radius > 0.0f);

    sf::CircleShape circle(radius);
    circle.setPosition(pos.x - radius, pos.y - radius);
    circle.setFillColor(sf::Color::Green);

    window.draw(circle);
  }

  void DrawRectangle(const Vector2 &halfExtents, const Vector2 &pos) {
    sf::RectangleShape rect(sf::Vector2f(halfExtents.x * 2.0f, halfExtents.y * 2.0f));
    rect.setPosition(pos.x - halfExtents.x, pos.y - halfExtents.y);
    rect.setFillColor(sf::Color::Red);

    window.draw(rect);
  }

  void DrawLine(const Vector2 &start, const Vector2 &end) {
    sf::Vertex line[] = {sf::Vertex(sf::Vector2f(start.x, start.y)),
                         sf::Vertex(sf::Vector2f(end.x, end.y))};

    window.draw(line, 2, sf::Lines);
  }
};

SFMLRenderer::SFMLRenderer(unsigned width, unsigned height, const string &windowName)
    : impl(new SFMLRendererImpl(width, height, windowName)) {}

SFMLRenderer::~SFMLRenderer() = default;

void SFMLRenderer::SwapBuffers(void) { impl->SwapBuffers(); }

void SFMLRenderer::DrawCircle(const Vector2 &pos, float radius) { impl->DrawCircle(pos, radius); }

void SFMLRenderer::DrawRectangle(const Vector2 &halfExtents, const Vector2 &pos) {
  impl->DrawRectangle(halfExtents, pos);
}

void SFMLRenderer::DrawLine(const Vector2 &start, const Vector2 &end) {
  impl->DrawLine(start, end);
}
