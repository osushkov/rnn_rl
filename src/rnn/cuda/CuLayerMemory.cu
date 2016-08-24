
#include "CuLayerMemory.hpp"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuLayerMemory::CuLayerMemory(const RNNSpec &spec, unsigned maxTraceLength) {
  assert(maxTraceLength > 0);

  memory.reserve(maxTraceLength);
  for (int timestamp = 0; timestamp < maxTraceLength; timestamp++) {
    memory.emplace_back(spec, timestamp);
  }
}

void CuLayerMemory::Cleanup(void) {
  for (auto &ts : memory) {
    ts.Cleanup();
  }
}

CuTimeSlice *CuLayerMemory::GetTimeSlice(int timestamp) {
  for (auto &ts : memory) {
    if (ts.timestamp == timestamp) {
      return &ts;
    }
  }

  return nullptr;
}

void CuLayerMemory::Clear(void) {
  for (auto &ts : memory) {
    ts.Clear();
  }
}
