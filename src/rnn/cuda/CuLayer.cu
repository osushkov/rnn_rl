
#include "CuLayer.hpp"
#include "Util.hpp"
#include <cassert>
#include <iostream>

using namespace rnn;
using namespace rnn::cuda;

CuLayer::CuLayer(const RNNSpec &nnSpec, const LayerSpec &layerSpec)
    : layerId(layerSpec.uid),
      activation(layerSpec.isOutput ? nnSpec.outputActivation : nnSpec.hiddenActivation),
      numNodes(layerSpec.numNodes), isOutput(layerSpec.isOutput) {

  assert(layerSpec.uid != 0); // 0 is reserved for the input.
  if (isOutput) {
    assert(numNodes == nnSpec.numOutputs);
  }

  for (const auto &lc : nnSpec.connections) {
    assert(lc.timeOffset == 0 || lc.timeOffset == 1);
    assert(lc.dstLayerId != 0);

    if (lc.dstLayerId == layerId) {
      // +1 accounts for the bias.
      unsigned inputSize = nnSpec.LayerSize(lc.srcLayerId) + 1;
      CuMatrix weights = util::AllocMatrix(numNodes, inputSize);
      CuMatrix weightsT = util::AllocMatrix(inputSize, numNodes);
      incoming.emplace_back(lc, CuWeights(weights, weightsT));
    }

    if (lc.srcLayerId == layerId) {
      outgoing.push_back(lc);
    }
  }
}

void CuLayer::Cleanup(void) {
  for (auto& ic : incoming) {
    util::FreeMatrix(ic.second.weights);
    util::FreeMatrix(ic.second.weightsT);
  }
}

CuWeights* CuLayer::GetWeights(const LayerConnection &connection) {
  for (auto& c : incoming) {
    if (c.first == connection) {
      return &c.second;
    }
  }

  return nullptr;
}
