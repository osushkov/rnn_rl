
#include "renderer/SFMLRenderer.hpp"
#include "simulation/Cart.hpp"
#include "simulation/PhysicsWorld.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

using namespace simulation;
using namespace renderer;

int main(int argc, char **argv) {
  std::cout << "hello world" << std::endl;

  PhysicsWorld world;
  Cart cart(CartSpec(5.0f, 1.0f, 1.0f), world.GetWorld());
  cart.Reset(0.0f);
  cart.ApplyImpulse(-10.0f);

  uptr<Renderer> renderer = make_unique<SFMLRenderer>(400, 400, "Cart Sim");

  for (unsigned i = 0; i < 1000; i++) {
    world.Step(1.0 / 30.0f);
    cart.Render(renderer.get());
    renderer->SwapBuffers();
    cout << "step: " << i << endl;
    getchar();
  }

  // sf::RenderWindow window(sf::VideoMode(300, 300), "SFML works!");
  // sf::CircleShape shape(100.0f);
  // shape.setPosition(50.0f, 100.0f);
  // shape.setFillColor(sf::Color::Green);
  //
  // sf::RectangleShape rect(sf::Vector2f(10.0f, 30.0f));
  // rect.setPosition(100.0f, 100.0f);
  //
  // float rot = 0.0f;
  // while (window.isOpen()) {
  //   sf::Event event;
  //   while (window.pollEvent(event)) {
  //     if (event.type == sf::Event::Closed)
  //       window.close();
  //   }
  //
  //   rect.setRotation(rot);
  //   rot += 0.01f;
  //
  //   window.clear();
  //   window.draw(rect);
  //   window.display();
  // }
  return 0;
}
