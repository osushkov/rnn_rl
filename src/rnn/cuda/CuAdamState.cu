
#include "CuAdamState.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuAdamState::CuAdamState(const RNNSpec &spec) {
  allConnections.reserve(spec.connections.size());
  for (const auto &connection : spec.connections) {
    unsigned inputSize = spec.LayerSize(connection.srcLayerId) + 1;
    unsigned layerSize = spec.LayerSize(connection.dstLayerId);
    allConnections.emplace_back(connection, layerSize, inputSize);
  }
}

void CuAdamState::Cleanup(void) {
  for (auto &c : allConnections) {
    c.Cleanup();
  }
}

CuAdamConnection *CuAdamState::GetConnection(const LayerConnection &connection) {
  for (auto &c : allConnections) {
    if (c.connection == connection) {
      return &c;
    }
  }

  assert(false);
  return nullptr;
}

void CuAdamState::Clear(void) {
  for (auto &c : allConnections) {
    MatrixFillKernel::Apply(c.momentum, 0.0f, 0);
    MatrixFillKernel::Apply(c.rms, 0.0f, 0);
  }
}
