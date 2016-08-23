
#include <SFML/Graphics.hpp>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "hello world" << std::endl;

  sf::RenderWindow window(sf::VideoMode(300, 300), "SFML works!");
  sf::CircleShape shape(100.0f);
  shape.setPosition(50.0f, 100.0f);
  shape.setFillColor(sf::Color::Green);

  sf::RectangleShape rect(sf::Vector2f(10.0f, 30.0f));
  rect.setPosition(100.0f, 100.0f);

  float rot = 0.0f;
  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    rect.setRotation(rot);
    rot += 0.01f;

    window.clear();
    window.draw(rect);
    window.display();
  }
  return 0;
}
