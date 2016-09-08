
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

  RNNSpec GetSpec(void) const;

  void ClearMemory(void);
  EVector Process(const EVector &input);

  void Update(const vector<SliceBatch> &trace);
  void Refresh(void);

private:
  struct RNNImpl;
  uptr<RNNImpl> impl;
};
}
