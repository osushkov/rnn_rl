#pragma once

#include "../LayerDef.hpp"
#include "../RNNSpec.hpp"
#include "Types.hpp"
#include "Util.hpp"
#include <cassert>
#include <vector>

using namespace std;

namespace rnn {
namespace cuda {

struct CuAdamConnection {
  LayerConnection connection;

  CuMatrix momentum;
  CuMatrix rms;

  CuAdamConnection(const LayerConnection &connection, unsigned rows, unsigned cols)
      : connection(connection), momentum(util::AllocMatrix(rows, cols)),
        rms(util::AllocMatrix(rows, cols)) {}

  void Cleanup(void) {
    util::FreeMatrix(momentum);
    util::FreeMatrix(rms);
  }
};

struct CuAdamState {
  vector<CuAdamConnection> allConnections;

  CuAdamState(const RNNSpec &spec);
  void Cleanup(void);

  CuAdamConnection *GetConnection(const LayerConnection &connection);
  void Clear(void);
};
}
}
