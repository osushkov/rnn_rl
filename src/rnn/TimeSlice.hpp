#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "Layer.hpp"
#include "LayerDef.hpp"
#include <cassert>
#include <vector>

namespace rnn {

struct ConnectionMemoryData {
  LayerConnection connection;
  bool haveActivation;

  EVector activation; // batch output, column per batch element.
  EVector derivative;

  ConnectionMemoryData(const LayerConnection &connection, int rows)
      : connection(connection), haveActivation(false), activation(rows), derivative(rows) {
    activation.fill(0.0f);
    derivative.fill(0.0f);
  }
};

struct TimeSlice {
  int timestamp;
  EVector networkInput;
  EVector networkOutput;
  vector<ConnectionMemoryData> connectionData;

  TimeSlice(int timestamp, const EVector &networkInput, const vector<Layer> &layers)
      : timestamp(timestamp), networkInput(networkInput) {
    assert(networkInput.cols() > 0);

    for (const auto &layer : layers) {
      for (const auto &c : layer.outgoing) {
        connectionData.push_back(ConnectionMemoryData(c, layer.numNodes));
      }
    }
  }

  const ConnectionMemoryData *GetConnectionData(const LayerConnection &c) const {
    for (auto &cmd : connectionData) {
      if (cmd.connection == c) {
        return &cmd;
      }
    }

    return nullptr;
  }

  // TODO: use const_cast to not have duplicate code.
  ConnectionMemoryData *GetConnectionData(const LayerConnection &connection) {
    for (auto &cmd : connectionData) {
      if (cmd.connection == connection) {
        return &cmd;
      }
    }
    return nullptr;
  }
};
}
