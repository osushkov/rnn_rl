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

struct CuConnectionAccum {
  LayerConnection connection;

  unsigned samples;
  CuMatrix accumGradient;

  CuConnectionAccum(const LayerConnection &connection, unsigned connRows, unsigned connCols)
      : connection(connection), samples(0), accumGradient(util::AllocMatrix(connRows, connCols)) {}

  void Cleanup(void) { util::FreeMatrix(accumGradient); }
};

struct CuGradientAccum {
  vector<CuConnectionAccum> allWeightsAccum;

  CuGradientAccum(const RNNSpec &spec);
  void Cleanup(void);

  CuConnectionAccum *GetConnection(const LayerConnection &connection);
  void Clear(void);
};
}
}
