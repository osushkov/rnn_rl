#pragma once

#include "Math.hpp"
#include <cassert>

namespace math {

struct OneHotVector {
  unsigned dim;
  unsigned index;

  OneHotVector(unsigned dim, unsigned index) : dim(dim), index(index) {
    assert(dim > 0 && index < dim);
  };

  ~OneHotVector() = default;

  EVector DenseVector() const {
    EVector result(dim);
    result.fill(0.0f);
    result(index) = 1.0f;
    return result;
  }
};
}
