#pragma once

#include "Types.hpp"

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {

inline __device__ float *Elem(CuMatrix m, unsigned r, unsigned c) {
  assert(r < m.rows && c < m.cols);
  return (float *)((char *)m.data + r * m.pitch) + c;
}

}
}
