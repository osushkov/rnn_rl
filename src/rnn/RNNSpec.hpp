
#pragma once

#include "LayerDef.hpp"
#include <cassert>
#include <vector>

using namespace std;

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

  inline void Write(std::ostream &out) const {
    out << numInputs << endl;
    out << numOutputs << endl;

    out << layers.size() << endl;
    for (const auto &layer : layers) {
      out << layer.uid << endl;
      out << layer.numNodes << endl;
      out << static_cast<int>(layer.isOutput) << endl;
    }

    out << connections.size() << endl;
    for (const auto &connection : connections) {
      out << connection.srcLayerId << endl;
      out << connection.dstLayerId << endl;
      out << connection.timeOffset << endl;
    }

    out << static_cast<int>(hiddenActivation) << std::endl;
    out << static_cast<int>(outputActivation) << std::endl;

    out << nodeActivationRate << std::endl;
    out << maxBatchSize << std::endl;
    out << maxTraceLength << std::endl;
  }

  static inline RNNSpec Read(std::istream &in) {
    RNNSpec spec;

    in >> spec.numInputs;
    in >> spec.numOutputs;

    unsigned numLayers;
    in >> numLayers;
    for (unsigned i = 0; i < numLayers; i++) {
      LayerSpec layer;
      in >> layer.uid;
      in >> layer.numNodes;
      in >> layer.isOutput;
      spec.layers.push_back(layer);
    }

    unsigned numConnections;
    in >> numConnections;
    for (unsigned i = 0; i < numConnections; i++) {
      LayerConnection connection;
      in >> connection.srcLayerId;
      in >> connection.dstLayerId;
      in >> connection.timeOffset;
      spec.connections.push_back(connection);
    }

    int ha, oa;
    in >> ha;
    in >> oa;

    spec.hiddenActivation = static_cast<LayerActivation>(ha);
    spec.outputActivation = static_cast<LayerActivation>(oa);

    in >> spec.nodeActivationRate;
    in >> spec.maxBatchSize;
    in >> spec.maxTraceLength;

    return spec;
  }
};
}
