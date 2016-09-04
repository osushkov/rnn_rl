
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "RNNSpec.hpp"
#include "SliceBatch.hpp"

namespace rnn {

class RNN {
public:
  RNN(const RNNSpec &spec);
  virtual ~RNN();

  RNN(const RNN &) = delete;
  RNN &operator=(const RNN &) = delete;

  void ClearMemory(void);
  EMatrix Process(const EMatrix &input);

  void Update(const vector<SliceBatch> &trace);
  void RefreshAndGetTarget(void);

private:
  struct RNNImpl;
  uptr<RNNImpl> impl;
};
}
