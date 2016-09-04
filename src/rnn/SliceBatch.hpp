#pragma once

#include "../math/Math.hpp"

namespace rnn {

struct SliceBatch {
  EMatrix batchInput;
  EMatrix batchActions; // one hot encoding
  EMatrix batchRewards;

  SliceBatch(const EMatrix &batchInput, const EMatrix &batchActions, const EMatrix &batchRewards)
      : batchInput(batchInput), batchActions(batchActions), batchRewards(batchRewards) {}
};
}
