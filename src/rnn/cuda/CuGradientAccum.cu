
#include "CuGradientAccum.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuGradientAccum::CuGradientAccum(const RNNSpec &spec) {
  allWeightsAccum.reserve(spec.connections.size());
  for (const auto &connection : spec.connections) {
    unsigned inputSize = spec.LayerSize(connection.srcLayerId) + 1;
    unsigned layerSize = spec.LayerSize(connection.dstLayerId);
    allWeightsAccum.emplace_back(connection, layerSize, inputSize);
  }
}

void CuGradientAccum::Cleanup(void) {
  for (auto &wa : allWeightsAccum) {
    wa.Cleanup();
  }
}

CuConnectionAccum *CuGradientAccum::GetConnection(const LayerConnection &connection) {
  for (auto &wa : allWeightsAccum) {
    if (wa.connection == connection) {
      return &wa;
    }
  }

  assert(false);
  return nullptr;
}

void CuGradientAccum::Clear(void) {
  for (auto &wa : allWeightsAccum) {
    wa.samples = 0;
    MatrixFillKernel::Apply(wa.accumGradient, 0.0f, 0);
  }
}
