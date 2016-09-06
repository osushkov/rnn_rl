
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

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 1000000;

static constexpr float INITIAL_PRANDOM = 0.5f;
static constexpr float TARGET_PRANDOM = 0.05f;

static constexpr float INITIAL_TEMPERATURE = 0.5f;
static constexpr float TARGET_TEMPERATURE = 0.001f;

struct Trainer::TrainerImpl {
  atomic<unsigned> numLearnIters;

  uptr<LearningAgent> TrainAgent(unsigned iters) {
    auto experienceMemory = make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);
    auto experienceGenerator = make_unique<ExperienceGenerator>();
    numLearnIters = 0;

    uptr<LearningAgent> agent = make_unique<LearningAgent>();

    std::thread playoutThread =
        startExperienceThread(agent.get(), experienceMemory.get(), experienceGenerator.get(), iters);
    std::thread learnThread = startLearnThread(agent.get(), experienceMemory.get(), iters);

    playoutThread.join();
    learnThread.join();

    return move(agent);
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
      while (memory->NumMemories() < 5 * EXPERIENCE_BATCH_SIZE) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      for (unsigned i = 0; i < iters; i++) {
        agent->Learn(memory->Sample(EXPERIENCE_BATCH_SIZE, EXPERIENCE_MAX_TRACE_LENGTH));
        cout << "learn: " << i << endl;
        this->numLearnIters++;
      }

      agent->Finalise();
    });
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

uptr<LearningAgent> Trainer::TrainAgent(unsigned iters) { return impl->TrainAgent(iters); }
