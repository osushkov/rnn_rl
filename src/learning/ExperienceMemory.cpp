
#include "ExperienceMemory.hpp"
#include "../common/Common.hpp"
#include <cassert>

using namespace learning;

// How much memory to purge (as a ratio of max size) when the occupancy is saturated.
static constexpr float PURGE_RATIO = 0.2f;

ExperienceMemory::ExperienceMemory(unsigned maxSize)
    : pastExperiences(maxSize), head(0), tail(0), occupancy(0) {}

void ExperienceMemory::AddExperience(const Experience &moment) {
  // obtain a write lock
  boost::unique_lock<boost::shared_mutex> lock(smutex);

  pastExperiences[tail] = moment;
  tail = (tail + 1) % pastExperiences.size();
  occupancy++;

  if (occupancy == pastExperiences.size()) {
    purgeOldMemories();
  }
}

void ExperienceMemory::AddExperiences(const vector<Experience> &moments) {
  // obtain a write lock
  boost::unique_lock<boost::shared_mutex> lock(smutex);

  for (unsigned i = 0; i < moments.size(); i++) {
    pastExperiences[tail] = moments[i];
    tail = (tail + 1) % pastExperiences.size();
    occupancy++;

    if (occupancy == pastExperiences.size()) {
      purgeOldMemories();
    }
  }
}

vector<Experience> ExperienceMemory::Sample(unsigned numSamples, unsigned experienceLength) const {
  // obtain a read lock
  boost::shared_lock<boost::shared_mutex> lock(smutex);

  vector<Experience> result;
  result.reserve(numSamples);

  for (unsigned i = 0; i < numSamples; i++) {
    result.push_back(trimmed(pastExperiences[wrappedIndex(rand())], experienceLength));
  }

  return result;
}

unsigned ExperienceMemory::NumMemories(void) const {
  boost::shared_lock<boost::shared_mutex> lock(smutex);
  return occupancy;
}

unsigned ExperienceMemory::wrappedIndex(unsigned i) const {
  assert(occupancy > 0 && pastExperiences.size() > 0);
  return ((i % occupancy) + head) % pastExperiences.size();
}

void ExperienceMemory::purgeOldMemories(void) {
  unsigned purgeAmount = PURGE_RATIO * occupancy;
  occupancy -= purgeAmount;
  head = (head + purgeAmount) % pastExperiences.size();
}

Experience ExperienceMemory::trimmed(const Experience &experience, unsigned targetLength) const {
  if (experience.moments.size() <= targetLength) {
    return experience;
  }

  Experience result;
  unsigned startIndex = rand() % (experience.moments.size() - targetLength);
  for (unsigned i = 0; i < targetLength; i++) {
    result.moments.push_back(experience.moments[startIndex + i]);
  }
  return result;
}
