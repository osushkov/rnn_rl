#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "LayerDef.hpp"
#include "RNNSpec.hpp"
#include <cassert>
#include <vector>

namespace rnn {

struct Layer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;
  bool isOutput;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, EMatrix>> weights;
  vector<LayerConnection> outgoing;

  Layer(const RNNSpec &nnSpec, const LayerSpec &layerSpec);

  void Read(std::istream &in);
  void Write(std::ostream &out) const;
};
}
