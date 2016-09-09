#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "RNNSpec.hpp"
#include "SliceBatch.hpp"
#include <utility>

namespace rnn {

class CudaTrainer {
public:
  CudaTrainer(const RNNSpec &spec);
  ~CudaTrainer();

  void SetWeights(const vector<pair<LayerConnection, math::MatrixView>> &weights);
  void GetWeights(vector<pair<LayerConnection, math::MatrixView>> &outWeights);
  void UpdateTarget(void);

  void Train(const vector<SliceBatch> &trace, float learnRate);

private:
  struct CudaTrainerImpl;
  uptr<CudaTrainerImpl> impl;
};
}
