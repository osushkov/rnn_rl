#pragma once

#include "../common/Common.hpp"
#include "LearningAgent.hpp"

namespace learning {

class Trainer {
public:
  Trainer();
  ~Trainer();

  uptr<LearningAgent> TrainAgent(unsigned iters);

private:
  struct TrainerImpl;
  uptr<TrainerImpl> impl;
};
}
