
#include "CuDeltaAccum.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuDeltaAccum::CuDeltaAccum(const RNNSpec &spec, unsigned maxTraceLength) {

  assert(maxTraceLength > 0);

  allDeltaAccum.reserve(maxTraceLength * spec.layers.size());
  for (int timestamp = 0; timestamp < maxTraceLength; timestamp++) {
    for (const auto &layer : spec.layers) {
      allDeltaAccum.emplace_back(layer.uid, timestamp, spec.maxBatchSize, layer.numNodes);
    }
  }
}

void CuDeltaAccum::Cleanup(void) {
  for (auto &da : allDeltaAccum) {
    da.Cleanup();
  }
}

CuLayerAccum *CuDeltaAccum::GetDelta(unsigned layerId, int timestamp) {
  for (auto &da : allDeltaAccum) {
    if (da.layerId == layerId && da.timestamp == timestamp) {
      return &da;
    }
  }

  assert(false);
  return nullptr;
}

void CuDeltaAccum::Clear(void) {
  for (auto &da : allDeltaAccum) {
    da.samples = 0;
    MatrixFillKernel::Apply(da.accumDelta, 0.0f, 0);
  }
}
