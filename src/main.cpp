
#include "Evaluator.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "learning/Constants.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"
#include "learning/Trainer.hpp"
#include "renderer/SFMLRenderer.hpp"
#include "simulation/Cart.hpp"
#include "simulation/PhysicsWorld.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

using namespace simulation;
using namespace renderer;
using namespace learning;

int main(int argc, char **argv) {
  std::cout << "hello world" << std::endl;

  uptr<learning::Agent> randomAgent = make_unique<learning::RandomAgent>();
  // cout << "random agent: " << Evaluator::Evaluate(randomAgent.get()) << endl;

  uptr<learning::LearningAgent> learningAgent = make_unique<learning::LearningAgent>();
  cout << "learning agent start: " << Evaluator::Evaluate(learningAgent.get()) << endl;

  learning::Trainer trainer;
  trainer.TrainAgent(learningAgent.get(), 200000);

  cout << "learning agent end: " << Evaluator::Evaluate(learningAgent.get()) << endl;

  uptr<Renderer> renderer = make_unique<SFMLRenderer>(1200, 600, "Cart Sim");

  for (unsigned iters = 0; iters < 20; iters++) {
    cout << "iters: " << iters << endl;
    renderer->SwapBuffers();
    getchar();

    PhysicsWorld world;

    CartSpec CART_SPEC(CART_WEIGHT_KG, PENDULUM_LENGTH, PENDULUM_WEIGHT_KG);
    Cart cart(CART_SPEC, world.GetWorld());
    cart.Reset(0.1f);
    learningAgent->ResetMemory();

    Timer timer;
    timer.Start();

    float secondsSinceLastWorldUpdate = 0.0f;
    float secondsSinceLastAction = 0.0f;
    float lastTime = timer.GetElapsedSeconds();
    while (true) {
      float time = timer.GetElapsedSeconds();
      if (time > 15.0f) {
        break;
      }

      float frameGap = time - lastTime;
      lastTime = time;

      if (secondsSinceLastWorldUpdate > (1.0f / 10.0f)) {
        world.Step(1.0f / 10.0f);
        secondsSinceLastWorldUpdate = 0.0f;
      }

      if (secondsSinceLastAction > (1.0f / 2.0f)) {
        State observedState(cart.GetCartXPos(), cart.GetPendulumX(), cart.GetPendulumY(),
                            cart.GetHingeAngle());
        Action performedAction = learningAgent->SelectAction(&observedState);

        cart.ApplyCartImpulse(performedAction.GetImpulse());
        cart.ApplyPendulumImpulse(math::GaussianSample(0.0f, PENDULUM_WIND_STDDEV));

        secondsSinceLastAction = 0.0f;
      }

      secondsSinceLastWorldUpdate += frameGap;
      secondsSinceLastAction += frameGap;

      world.Render(renderer.get());
      cart.Render(renderer.get());
      renderer->SwapBuffers();
    }
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
