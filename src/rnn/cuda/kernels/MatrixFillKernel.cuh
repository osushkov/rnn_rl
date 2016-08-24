#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace MatrixFillKernel {

void Apply(const CuMatrix &target, float value, cudaStream_t stream);
}
}
}
