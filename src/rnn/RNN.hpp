
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "RNNSpec.hpp"
#include "SliceBatch.hpp"
#include <iostream>

namespace rnn {

class RNN {
public:
  RNN(const RNNSpec &spec);
  virtual ~RNN();

  RNN(const RNN &) = delete;
  RNN &operator=(const RNN &) = delete;

  static uptr<RNN> Read(std::istream &in);
  void Write(std::ostream &out) const;

  RNNSpec GetSpec(void) const;

  void ClearMemory(void);
  EVector Process(const EVector &input);

  void Update(const vector<SliceBatch> &trace, float learnRate);
  void RefreshAndGetTarget(void);

private:
  struct RNNImpl;
  uptr<RNNImpl> impl;
};
}
