#pragma once

#include "../math/Math.hpp"

namespace rnn {

struct SliceBatch {
  EMatrix batchInput;
  EMatrix batchOutput;

  SliceBatch(const EMatrix &batchInput, const EMatrix &batchOutput)
      : batchInput(batchInput), batchOutput(batchOutput) {}
};
}
