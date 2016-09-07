#pragma once

#include "../common/Common.hpp"
#include "LearningAgent.hpp"
#include "Experience.hpp"
#include <vector>

namespace learning {

class ExperienceGenerator {
public:
  ExperienceGenerator();
  ~ExperienceGenerator();

  ExperienceGenerator(const ExperienceGenerator &other) = delete;
  ExperienceGenerator(ExperienceGenerator &&other) = delete;
  ExperienceGenerator &operator=(const ExperienceGenerator &other) = delete;

  Experience GenerateExperience(LearningAgent *agent);

private:
  struct ExperienceGeneratorImpl;
  uptr<ExperienceGeneratorImpl> impl;
};
}
