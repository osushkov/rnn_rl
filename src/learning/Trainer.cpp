
#include "Trainer.hpp"
#include "../common/Common.hpp"
#include "../common/Timer.hpp"
#include "../simulation/Action.hpp"
#include "../simulation/State.hpp"
#include "Constants.hpp"
#include "ExperienceGenerator.hpp"
#include "ExperienceMemory.hpp"
#include "LearningAgent.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>
#include <vector>

using namespace learning;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 100000;

static constexpr float INITIAL_PRANDOM = 0.9f;
static constexpr float TARGET_PRANDOM = 0.1f;

static constexpr float INITIAL_TEMPERATURE = 0.5f;
static constexpr float TARGET_TEMPERATURE = 0.01f;

static constexpr float INITIAL_LEARN_RATE = 1.0f;
static constexpr float TARGET_LEARN_RATE = 0.1f;

struct Trainer::TrainerImpl {
  atomic<unsigned> numLearnIters;

  void TrainAgent(LearningAgent *agent, unsigned iters) {
    auto experienceMemory = make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);
    auto experienceGenerator = make_unique<ExperienceGenerator>();
    numLearnIters = 0;

    std::thread playoutThread =
        startExperienceThread(agent, experienceMemory.get(), experienceGenerator.get(), iters);
    std::thread learnThread = startLearnThread(agent, experienceMemory.get(), iters);

    playoutThread.join();
    learnThread.join();
  }

  std::thread startExperienceThread(LearningAgent *agent, ExperienceMemory *memory,
                                    ExperienceGenerator *generator, unsigned iters) {

    return std::thread([this, agent, memory, generator, iters]() {
      float pRandDecay = powf(TARGET_PRANDOM / INITIAL_PRANDOM, 1.0f / iters);
      assert(pRandDecay > 0.0f && pRandDecay <= 1.0f);

      float tempDecay = powf(TARGET_TEMPERATURE / INITIAL_TEMPERATURE, 1.0f / iters);
      assert(tempDecay > 0.0f && tempDecay <= 1.0f);

      while (true) {
        unsigned doneIters = numLearnIters.load();
        if (doneIters >= iters) {
          break;
        }

        float prand = INITIAL_PRANDOM * powf(pRandDecay, doneIters);
        float temp = INITIAL_TEMPERATURE * powf(tempDecay, doneIters);

        agent->SetPRandom(prand);
        agent->SetTemperature(temp);

        memory->AddExperience(generator->GenerateExperience(agent));
        // cout << "experiences generated: " << memory->NumMemories() << endl;
      }
    });
  }

  std::thread startLearnThread(LearningAgent *agent, ExperienceMemory *memory, unsigned iters) {
    return std::thread([this, agent, memory, iters]() {
      float lrDecay = powf(TARGET_LEARN_RATE / INITIAL_LEARN_RATE, 1.0f / iters);
      assert(lrDecay > 0.0f && lrDecay <= 1.0f);

      while (memory->NumMemories() < 5 * EXPERIENCE_BATCH_SIZE) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      // unsigned it = 0;
      // for (unsigned i = 2; i <= EXPERIENCE_MAX_TRACE_LENGTH; i++) {
      //   for (unsigned j = 0; j < iters / EXPERIENCE_MAX_TRACE_LENGTH; j++) {

      for (unsigned it = 0; it < iters; it++) {
        float lr = INITIAL_LEARN_RATE * powf(lrDecay, it);
        agent->Learn(memory->Sample(EXPERIENCE_BATCH_SIZE, EXPERIENCE_MAX_TRACE_LENGTH), lr);

        if (it % 1000 == 0) {
          cout << "learn: " << ((100 * it) / iters) << "%" << endl;
        }
        this->numLearnIters++;
        // it++;
        // }
      }

      // for (unsigned i = 0; i < iters; i++) {
      //   float lr = INITIAL_LEARN_RATE * powf(lrDecay, i);
      //   agent->Learn(memory->Sample(EXPERIENCE_BATCH_SIZE, EXPERIENCE_MAX_TRACE_LENGTH), lr);
      //
      //   if (i % 1000 == 0) {
      //     cout << "learn: " << ((100 * i) / iters) << "%" << endl;
      //   }
      //   this->numLearnIters++;
      // }

      this->numLearnIters = iters;
      cout << "done" << endl;
      agent->Finalise();
    });
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

void Trainer::TrainAgent(LearningAgent *agent, unsigned iters) {
  return impl->TrainAgent(agent, iters);
}
