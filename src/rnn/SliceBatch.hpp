#pragma once

#include "../math/Math.hpp"

namespace rnn {

struct SliceBatch {
  EMatrix batchInput; // row-vectors, one per batch element.
  EMatrix batchActions; // one hot encoding
  EMatrix batchRewards;

  SliceBatch(const EMatrix &batchInput, const EMatrix &batchActions, const EMatrix &batchRewards)
      : batchInput(batchInput), batchActions(batchActions), batchRewards(batchRewards) {}
};
}
