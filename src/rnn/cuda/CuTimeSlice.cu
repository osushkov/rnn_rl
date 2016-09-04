
#include "CuTimeSlice.hpp"
#include "kernels/MatrixFillKernel.cuh"
#include "kernels/InitialiseOutputKernel.cuh"
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

CuTimeSlice::CuTimeSlice(const RNNSpec &spec, int timestamp)
    : timestamp(timestamp),
      networkOutput(LayerConnection(0, 0, 0), spec.maxBatchSize, spec.numOutputs + 1),
      actionsMask(util::AllocMatrix(spec.maxBatchSize, spec.numOutputs)),
      rewards(util::AllocMatrix(spec.maxBatchSize, 1)) {

  assert(timestamp >= 0);
  for (const auto &connection : spec.connections) {
    unsigned connectionCols = spec.LayerSize(connection.srcLayerId) + 1;
    connectionData.emplace_back(connection, spec.maxBatchSize, connectionCols);
  }
}

void CuTimeSlice::Cleanup(void) {
  util::FreeMatrix(actionsMask);
  util::FreeMatrix(rewards);
  
  networkOutput.Cleanup();
  for (auto &cd : connectionData) {
    cd.Cleanup();
  }
}

CuConnectionMemoryData *CuTimeSlice::GetConnectionData(const LayerConnection &connection) {
  for (auto &cmd : connectionData) {
    if (cmd.connection == connection) {
      return &cmd;
    }
  }

  assert(false);
  return nullptr;
}

void CuTimeSlice::Clear(void) {
  networkOutput.haveActivation = false;
  MatrixFillKernel::Apply(networkOutput.activation, 0.0f, 0);
  MatrixFillKernel::Apply(networkOutput.derivative, 0.0f, 0);

  for (auto &cd : connectionData) {
    cd.haveActivation = false;
    MatrixFillKernel::Apply(cd.activation, 0.0f, 0);
    MatrixFillKernel::Apply(cd.derivative, 0.0f, 0);
    InitialiseOutputKernel::Apply(cd.activation, 0);
  }
}
