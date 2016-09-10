
#pragma once

#include <sys/time.h>
#include <time.h>

class Timer {
public:
  void Start(void);
  void Stop(void);

  float GetIntervalElapsedSeconds(void) const;
  unsigned GetIntervalElapsedMicroseconds(void) const;

  float GetElapsedSeconds(void) const;
  unsigned GetElapsedMicroseconds(void) const;

private:
  struct timeval startTime;
  struct timeval endTime;
};
