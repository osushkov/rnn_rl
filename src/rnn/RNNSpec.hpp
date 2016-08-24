
#pragma once

#include "LayerDef.hpp"
#include <cassert>
#include <vector>

namespace rnn {

struct RNNSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<LayerSpec> layers;
  std::vector<LayerConnection> connections;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;

  float nodeActivationRate; // for dropout regularization.
  unsigned maxBatchSize;
  unsigned maxTraceLength;

  // Helper function.
  unsigned LayerSize(unsigned layerId) const {
    if (layerId == 0) {
      return numInputs;
    }

    for (const auto &ls : layers) {
      if (ls.uid == layerId) {
        return ls.numNodes;
      }
    }

    assert(false);
    return 0;
  }
};
}
