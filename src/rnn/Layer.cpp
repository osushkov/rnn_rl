
#include "Layer.hpp"
#include "../math/Math.hpp"
#include <cassert>
#include <cmath>

using namespace rnn;

static unsigned numLayerOutputs(const RNNSpec &spec, unsigned layerId) {
  if (layerId == 0) {
    return spec.numInputs;
  }

  for (const auto &ls : spec.layers) {
    if (ls.uid == layerId) {
      return ls.numNodes;
    }
  }

  assert(false);
  return 0;
}

static EMatrix createWeightsMatrix(unsigned inputSize, unsigned outputSize) {
  float initRange = 0.1f / sqrtf(inputSize);

  EMatrix result(outputSize, inputSize);
  result.fill(0.0f);

  for (unsigned r = 0; r < result.rows(); r++) {
    for (unsigned c = 0; c < result.cols(); c++) {
      result(r, c) = math::GaussianSample(0.0, initRange);
      // result(r, c) = math::RandInterval(-initRange, initRange);
    }
  }

  return result;
}

Layer::Layer(const RNNSpec &nnSpec, const LayerSpec &layerSpec)
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
      unsigned inputSize = numLayerOutputs(nnSpec, lc.srcLayerId) + 1;
      EMatrix weightsMatrix = createWeightsMatrix(inputSize, numNodes);
      weights.emplace_back(lc, weightsMatrix);
    }

    if (lc.srcLayerId == layerId) {
      outgoing.push_back(lc);
    }
  }
}

void Layer::Read(std::istream &in) {
  for (auto &weight : weights) {
    weight.second = math::ReadMatrix(in);
  }
}

void Layer::Write(std::ostream &out) const {
  for (const auto &weight : weights) {
    math::WriteMatrix(weight.second, out);
  }
}
