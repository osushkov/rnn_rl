#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include "Util.hpp"
#include <cassert>
#include <utility>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuConnectionMemoryData {
  LayerConnection connection;
  bool haveActivation;

  CuMatrix activation; // batch output, row per batch element.
  CuMatrix derivative;

  CuConnectionMemoryData(const LayerConnection &connection, unsigned rows, unsigned cols)
      : connection(connection), haveActivation(false), activation(util::AllocMatrix(rows, cols)),
        derivative(util::AllocMatrix(rows, cols)) {}

  void Cleanup(void) {
    util::FreeMatrix(activation);
    util::FreeMatrix(derivative);
  }
};

struct CuTimeSlice {
  int timestamp;
  CuConnectionMemoryData networkOutput;
  CuMatrix actionsMask;
  CuMatrix rewards;

  vector<CuConnectionMemoryData> connectionData;

  CuTimeSlice(const RNNSpec &spec, int timestamp);
  void Cleanup(void);

  CuConnectionMemoryData *GetConnectionData(const LayerConnection &connection);
  void Clear(void);
};
}
}
