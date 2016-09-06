#pragma once

#include "../common/Common.hpp"
#include "LearningAgent.hpp"

namespace learning {

class Trainer {
public:
  Trainer();
  ~Trainer();

  void TrainAgent(LearningAgent *agent, unsigned iters);

private:
  struct TrainerImpl;
  uptr<TrainerImpl> impl;
};
}
