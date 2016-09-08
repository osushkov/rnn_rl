#pragma once

#include "../math/Math.hpp"
#include "../simulation/State.hpp"
#include "Experience.hpp"
#include <boost/thread/shared_mutex.hpp>
#include <cstdlib>

using namespace simulation;

namespace learning {

class ExperienceMemory {
  mutable boost::shared_mutex smutex;

  vector<Experience> pastExperiences;
  unsigned head;
  unsigned tail;
  unsigned occupancy;

public:
  ExperienceMemory(unsigned maxSize);
  ~ExperienceMemory() = default;

  ExperienceMemory(const ExperienceMemory &other) = delete;
  ExperienceMemory(ExperienceMemory &&other) = delete;
  ExperienceMemory &operator=(const ExperienceMemory &other) = delete;

  void AddExperience(const Experience &moment);
  void AddExperiences(const vector<Experience> &moments);

  vector<Experience> Sample(unsigned numSamples, unsigned experienceLength) const;
  unsigned NumMemories(void) const;

private:
  unsigned wrappedIndex(unsigned i) const;
  void purgeOldMemories(void);

  Experience trimmed(const Experience &experience, unsigned targetLength) const;
};
}
