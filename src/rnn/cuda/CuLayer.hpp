#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include <cassert>
#include <utility>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuWeights {
  CuMatrix weights;
  CuMatrix weightsT; // transposed.

  CuWeights(const CuMatrix &weights, const CuMatrix &weightsT)
      : weights(weights), weightsT(weightsT) {
    assert(weights.rows == weightsT.cols);
    assert(weights.cols == weightsT.rows);
    assert(weights.data != nullptr && weightsT.data != nullptr);
  }
};

struct CuLayer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;
  bool isOutput;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, CuWeights>> incoming;
  vector<LayerConnection> outgoing;

  CuLayer(const RNNSpec &nnSpec, const LayerSpec &layerSpec);
  void Cleanup(void);

  CuWeights *GetWeights(const LayerConnection &connection);
};
}
}
