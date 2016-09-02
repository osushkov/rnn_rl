#pragma once

#include "../common/Common.hpp"
#include "Agent.hpp"
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

  vector<Experience> GenerateTrace(Agent *agent);

private:
  struct ExperienceGeneratorImpl;
  uptr<ExperienceGeneratorImpl> impl;
};
}
